"""
Position sizing engine using Kelly Criterion + Portfolio Optimisation.

Kelly Criterion theory:
  The Kelly fraction maximises the expected logarithm of portfolio wealth,
  which is equivalent to maximising long-run geometric growth.
  f* = (μ - r) / σ²  [continuous]
  f* = (p*b - q) / b  [discrete bets]

We use FRACTIONAL Kelly (25%) to:
  1. Reduce drawdown risk (full Kelly has huge variance)
  2. Account for estimation error in p and b
  3. Maintain geometric mean growth while limiting ruin probability

Kelly proof (discrete):
  Let W_t be wealth at time t. After n bets with fraction f:
  W_n = W_0 * (1 + f*b)^(n*p) * (1 - f)^(n*q)
  log(W_n/W_0) = n * [p*log(1+f*b) + q*log(1-f)]
  dG/df = p*b/(1+f*b) - q/(1-f) = 0
  → f* = (p*b - q) / b

Dynamic leverage schedule:
  As track record builds and capital grows, leverage increases.
  This reflects the Bayesian update of our confidence in the strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from utils.math_utils import (
    kelly_criterion_discrete,
    kelly_criterion_continuous,
    fractional_kelly,
    compute_position_size,
)
from utils.logger import logger


class PositionSizer:
    """
    Dynamic position sizer combining:
      1. Kelly Criterion (signal-based)
      2. Volatility targeting (risk parity)
      3. Dynamic leverage schedule
      4. Portfolio heat limits
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.30,
        max_portfolio_heat: float = 0.60,
        vol_target_daily: float = 0.02,  # 2% daily vol target
        leverage_schedule: Optional[Dict[float, float]] = None,
        initial_capital: float = 5000.0,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_portfolio_heat = max_portfolio_heat
        self.vol_target_daily = vol_target_daily
        self.leverage_schedule = leverage_schedule or {
            0.0: 3.0, 1.0: 5.0, 5.0: 7.0, 10.0: 10.0
        }
        self.initial_capital = initial_capital

        # Track historical signal accuracy for Kelly estimation
        self._trade_history: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Main sizing method
    # ──────────────────────────────────────────────────────────────────────────

    def compute_size(
        self,
        portfolio_value: float,
        signal_strength: float,        # [-1, +1] from master signal
        signal_confidence: float,      # [0, 1] from ML confidence
        current_price: float,
        daily_volatility: float,       # Estimated daily vol of the asset
        current_heat: float = 0.0,     # Current portfolio exposure fraction
        stop_loss_pct: float = 0.02,   # Distance to stop-loss as pct
        atr_pct: float = 0.015,        # ATR as pct of price
    ) -> Dict[str, float]:
        """
        Compute position size in USDC and units.

        Returns dict with:
          size_usdc: dollar amount to invest
          size_units: number of tokens to buy/sell
          leverage: leverage multiplier to use
          kelly_full: raw Kelly fraction
          kelly_applied: fractional Kelly used
          position_pct: fraction of portfolio
          rationale: human-readable sizing explanation
        """
        # Determine leverage from schedule
        leverage = self._get_leverage(portfolio_value)

        # ── Kelly Criterion (from signal statistics) ──────────────────────────
        win_prob, win_loss_ratio = self._estimate_win_stats(signal_strength, signal_confidence)
        kelly_full = kelly_criterion_discrete(win_prob, win_loss_ratio)

        # Also compute continuous Kelly from vol
        if daily_volatility > 0:
            # Expected excess return from signal
            expected_daily_return = abs(signal_strength) * daily_volatility * 1.5
            kelly_continuous = kelly_criterion_continuous(expected_daily_return, daily_volatility)
            # Take the more conservative estimate
            kelly_full = min(kelly_full, kelly_continuous)

        # Apply fractional Kelly
        kelly_applied = fractional_kelly(kelly_full, self.kelly_fraction)

        # ── Volatility Targeting ──────────────────────────────────────────────
        # Risk parity: size so that position contributes target_vol to portfolio
        if daily_volatility > 0:
            vol_target_pct = self.vol_target_daily / (daily_volatility * leverage)
            vol_target_pct = min(vol_target_pct, self.max_position_pct)
        else:
            vol_target_pct = self.max_position_pct

        # ── Risk-adjusted sizing (1% risk per trade) ──────────────────────────
        # Lose at most 1% of portfolio if stop-loss is hit
        risk_pct = 0.01
        if stop_loss_pct > 0:
            max_size_from_risk = risk_pct / stop_loss_pct
        else:
            max_size_from_risk = self.max_position_pct

        # ── Combine constraints ────────────────────────────────────────────────
        # Take minimum of all constraints
        position_pct = min(
            kelly_applied,
            vol_target_pct,
            max_size_from_risk,
            self.max_position_pct,
        )

        # ── Portfolio heat limit ──────────────────────────────────────────────
        available_heat = max(0.0, self.max_portfolio_heat - current_heat)
        position_pct = min(position_pct, available_heat)

        # ── Scale by signal strength ──────────────────────────────────────────
        # Stronger signal → larger fraction of computed size
        signal_scale = min(abs(signal_strength), 1.0)
        position_pct *= signal_scale

        # ── Final dollar and unit size ────────────────────────────────────────
        size_usdc = portfolio_value * position_pct * leverage
        size_units = size_usdc / current_price if current_price > 0 else 0.0

        logger.debug(
            f"Position sizing | Kelly={kelly_full:.3f} | Applied={kelly_applied:.3f} | "
            f"VolTarget={vol_target_pct:.3f} | RiskLimit={max_size_from_risk:.3f} | "
            f"Final={position_pct:.3f} | {size_usdc:.0f} USDC @ {leverage}x"
        )

        return {
            "size_usdc": float(size_usdc),
            "size_units": float(size_units),
            "leverage": float(leverage),
            "kelly_full": float(kelly_full),
            "kelly_applied": float(kelly_applied),
            "position_pct": float(position_pct),
            "win_prob": float(win_prob),
            "win_loss_ratio": float(win_loss_ratio),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_leverage(self, portfolio_value: float) -> float:
        """
        Dynamic leverage based on portfolio growth multiple.
        Higher growth → higher leverage (confidence in edge builds).
        """
        growth_multiple = portfolio_value / self.initial_capital
        max_leverage = 1.0
        for threshold, lev in sorted(self.leverage_schedule.items()):
            if growth_multiple >= threshold:
                max_leverage = lev
        return max_leverage

    def _estimate_win_stats(
        self,
        signal_strength: float,
        signal_confidence: float,
    ) -> Tuple[float, float]:
        """
        Estimate win probability and win/loss ratio from signal characteristics
        and historical trade performance.
        """
        # Base win probability from signal strength
        # Bayesian prior: 50% + signal contribution
        base_win_prob = 0.50 + abs(signal_strength) * 0.20 * signal_confidence

        # Win/loss ratio from historical trades
        if len(self._trade_history) >= 20:
            recent = self._trade_history[-50:]
            wins = [t["pnl"] for t in recent if t["pnl"] > 0]
            losses = [abs(t["pnl"]) for t in recent if t["pnl"] < 0]

            historical_win_prob = len(wins) / len(recent) if recent else 0.5
            avg_win = np.mean(wins) if wins else 1.5
            avg_loss = np.mean(losses) if losses else 1.0
            win_loss_ratio = avg_win / (avg_loss + 1e-9)

            # Blend historical with base estimate (weight increases with sample size)
            n = len(recent)
            historical_weight = min(n / 100, 0.7)
            win_prob = (1 - historical_weight) * base_win_prob + historical_weight * historical_win_prob
        else:
            # No history: conservative estimates
            win_prob = base_win_prob
            win_loss_ratio = 1.5  # Conservative: assume 1.5:1 reward/risk

        return float(np.clip(win_prob, 0.35, 0.75)), float(np.clip(win_loss_ratio, 0.5, 5.0))

    def record_trade(self, pnl_pct: float, signal_strength: float) -> None:
        """Record completed trade outcome for Kelly estimation."""
        self._trade_history.append({"pnl": pnl_pct, "signal": signal_strength})
        if len(self._trade_history) > 500:
            self._trade_history = self._trade_history[-500:]

    def get_performance_summary(self) -> Dict[str, float]:
        """Summary statistics of recent trading performance."""
        if len(self._trade_history) < 5:
            return {}
        recent = self._trade_history[-100:]
        pnls = [t["pnl"] for t in recent]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        return {
            "win_rate": len(wins) / len(pnls),
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "profit_factor": sum(wins) / (sum(abs(l) for l in losses) + 1e-9) if wins else 0.0,
            "expectancy": float(np.mean(pnls)),
            "trades_count": len(recent),
        }

    def compute_portfolio_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        max_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Mean-variance portfolio optimization (Markowitz).
        Maximises Sharpe ratio subject to long-only, sum-to-1 constraints.
        For multi-asset expansion of the strategy.
        """
        try:
            import cvxpy as cp
            n = len(expected_returns)
            w = cp.Variable(n)

            ret = expected_returns @ w
            risk = cp.quad_form(w, covariance_matrix)

            constraints = [
                cp.sum(w) == 1,
                w >= 0,
            ]
            if max_weights is not None:
                constraints.append(w <= max_weights)

            # Maximise Sharpe (proxy: maximise return / sqrt(variance))
            # Use L2 regularisation for numerical stability
            objective = cp.Maximize(ret - 0.5 * risk)
            problem = cp.Problem(objective, constraints)
            problem.solve(warm_start=True)

            if problem.status == "optimal":
                return np.array(w.value).flatten()
            else:
                return np.ones(n) / n
        except Exception:
            n = len(expected_returns)
            return np.ones(n) / n
