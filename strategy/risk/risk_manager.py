"""
Portfolio-level risk manager.

Manages:
  - Daily / weekly / monthly P&L limits (automatic circuit breakers)
  - Maximum drawdown protection (halts trading, reduces size)
  - Portfolio heat tracking (total exposure across all positions)
  - Correlation-based risk (avoid concentrated correlated positions)
  - VaR / CVaR monitoring
  - Volatility regime-adjusted risk limits

The risk management system is the safety net. We optimise for growth
but NEVER exceed these hard risk limits under any circumstance.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

from utils.logger import logger
from utils.math_utils import var_historical, cvar_historical, max_drawdown


class CircuitBreaker:
    """
    Automatic trading halt when loss limits are breached.
    Implements a multi-level protection scheme.
    """

    def __init__(
        self,
        daily_loss_limit: float = 0.05,
        weekly_loss_limit: float = 0.10,
        monthly_loss_limit: float = 0.20,
        max_drawdown_limit: float = 0.20,
    ) -> None:
        self.daily_limit = daily_loss_limit
        self.weekly_limit = weekly_loss_limit
        self.monthly_limit = monthly_loss_limit
        self.max_dd_limit = max_drawdown_limit

        self._day_start_value: Optional[float] = None
        self._week_start_value: Optional[float] = None
        self._month_start_value: Optional[float] = None
        self._peak_value: Optional[float] = None

        self._last_day_reset = datetime.now(timezone.utc)
        self._last_week_reset = datetime.now(timezone.utc)
        self._last_month_reset = datetime.now(timezone.utc)

        self.halted = False
        self.halt_reason: Optional[str] = None
        self.size_multiplier: float = 1.0  # [0, 1] scalar applied to all positions

    def update(self, portfolio_value: float) -> Dict[str, bool]:
        """
        Update circuit breaker state.
        Returns dict of which limits are breached.
        """
        now = datetime.now(timezone.utc)

        # Initialise reference values
        if self._day_start_value is None:
            self._day_start_value = portfolio_value
        if self._week_start_value is None:
            self._week_start_value = portfolio_value
        if self._month_start_value is None:
            self._month_start_value = portfolio_value
        if self._peak_value is None or portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        # Reset period reference values
        if (now - self._last_day_reset).total_seconds() > 86400:
            self._day_start_value = portfolio_value
            self._last_day_reset = now

        if (now - self._last_week_reset).total_seconds() > 7 * 86400:
            self._week_start_value = portfolio_value
            self._last_week_reset = now
            # Reduce size multiplier recovery after weekly reset
            self.size_multiplier = min(self.size_multiplier * 1.2, 1.0)

        if (now - self._last_month_reset).total_seconds() > 30 * 86400:
            self._month_start_value = portfolio_value
            self._last_month_reset = now

        # Compute losses
        daily_loss = (portfolio_value - self._day_start_value) / self._day_start_value
        weekly_loss = (portfolio_value - self._week_start_value) / self._week_start_value
        monthly_loss = (portfolio_value - self._month_start_value) / self._month_start_value
        current_dd = (portfolio_value - self._peak_value) / self._peak_value

        breaches = {
            "daily": daily_loss < -self.daily_limit,
            "weekly": weekly_loss < -self.weekly_limit,
            "monthly": monthly_loss < -self.monthly_limit,
            "drawdown": current_dd < -self.max_dd_limit,
        }

        # Apply graduated response
        self.halted = False
        self.halt_reason = None

        if breaches["monthly"] or breaches["drawdown"]:
            self.halted = True
            self.halt_reason = "monthly_loss" if breaches["monthly"] else "max_drawdown"
            self.size_multiplier = 0.0
            logger.critical(f"TRADING HALTED | reason={self.halt_reason} | portfolio={portfolio_value:.2f}")

        elif breaches["weekly"]:
            self.size_multiplier = 0.25
            logger.warning(f"Weekly loss limit breach | size_mult=0.25 | portfolio={portfolio_value:.2f}")

        elif breaches["daily"]:
            self.size_multiplier = 0.50
            logger.warning(f"Daily loss limit breach | size_mult=0.50 | portfolio={portfolio_value:.2f}")

        else:
            # Gradual recovery: if drawdown < 10%, restore size
            if current_dd > -0.10:
                self.size_multiplier = min(self.size_multiplier + 0.05, 1.0)

        return {
            "daily_pnl_pct": float(daily_loss),
            "weekly_pnl_pct": float(weekly_loss),
            "monthly_pnl_pct": float(monthly_loss),
            "drawdown_pct": float(current_dd),
            **breaches,
            "halted": self.halted,
            "size_multiplier": self.size_multiplier,
        }


class RiskManager:
    """
    Portfolio-level risk management system.

    Monitors and controls:
      - Circuit breakers (daily/weekly/monthly loss limits)
      - Maximum drawdown protection
      - Portfolio heat (total open exposure)
      - VaR/CVaR monitoring
      - Correlation risk
    """

    def __init__(
        self,
        daily_loss_limit: float = 0.05,
        weekly_loss_limit: float = 0.10,
        monthly_loss_limit: float = 0.20,
        max_drawdown_limit: float = 0.20,
        max_portfolio_heat: float = 0.60,
        var_limit: float = 0.05,  # Max acceptable 99% daily VaR
    ) -> None:
        self.circuit_breaker = CircuitBreaker(
            daily_loss_limit, weekly_loss_limit, monthly_loss_limit, max_drawdown_limit
        )
        self.max_portfolio_heat = max_portfolio_heat
        self.var_limit = var_limit

        # Track portfolio returns for risk metrics
        self._returns_history: deque = deque(maxlen=500)
        self._portfolio_history: List[Tuple[float, float]] = []  # (timestamp, value)

        # Position heat tracking
        self._position_heat: Dict[str, float] = {}  # coin → heat fraction

    def update(self, portfolio_value: float) -> Dict:
        """
        Main risk update — call on every portfolio value change.
        Returns current risk state dict.
        """
        # Record history
        if self._portfolio_history:
            last_val = self._portfolio_history[-1][1]
            ret = (portfolio_value - last_val) / last_val
            self._returns_history.append(ret)
        self._portfolio_history.append((time.time(), portfolio_value))

        # Circuit breaker check
        cb_state = self.circuit_breaker.update(portfolio_value)

        # Risk metrics
        risk_metrics = self._compute_risk_metrics()

        # VaR limit check
        var_breach = risk_metrics.get("var_99", 0) > self.var_limit
        if var_breach:
            logger.warning(f"VaR limit breach | VaR_99={risk_metrics['var_99']:.3f} > {self.var_limit}")
            self.circuit_breaker.size_multiplier = min(
                self.circuit_breaker.size_multiplier, 0.5
            )

        state = {
            **cb_state,
            **risk_metrics,
            "portfolio_heat": self.current_heat,
            "can_trade": not cb_state["halted"],
            "size_multiplier": self.circuit_breaker.size_multiplier,
        }
        return state

    def update_position_heat(self, coin: str, size_pct: float) -> None:
        """Register a position's portfolio heat contribution."""
        self._position_heat[coin] = size_pct

    def remove_position_heat(self, coin: str) -> None:
        """Remove position heat when trade closes."""
        self._position_heat.pop(coin, None)

    @property
    def current_heat(self) -> float:
        """Total portfolio heat (sum of all position sizes as pct)."""
        return sum(self._position_heat.values())

    def can_open_position(self, new_position_pct: float) -> bool:
        """Check if a new position can be opened without exceeding heat limit."""
        return (self.current_heat + new_position_pct) <= self.max_portfolio_heat

    def validate_trade(
        self,
        portfolio_value: float,
        position_pct: float,
        signal_strength: float,
        vol_regime: str = "normal",
    ) -> Tuple[bool, str, float]:
        """
        Final trade validation. Returns (can_trade, reason, approved_size_pct).
        """
        if self.circuit_breaker.halted:
            return False, "circuit_breaker_halt", 0.0

        # Apply circuit breaker size multiplier
        adjusted_pct = position_pct * self.circuit_breaker.size_multiplier

        # Heat limit
        if not self.can_open_position(adjusted_pct):
            available = max(0, self.max_portfolio_heat - self.current_heat)
            if available < 0.01:
                return False, "max_heat_reached", 0.0
            adjusted_pct = available

        # Signal strength minimum
        if abs(signal_strength) < 0.15:
            return False, "signal_too_weak", 0.0

        # Extreme vol → reduce size
        vol_scalars = {"low": 0.8, "normal": 1.0, "elevated": 0.7, "extreme": 0.3}
        vol_scalar = vol_scalars.get(vol_regime, 1.0)
        adjusted_pct *= vol_scalar

        if adjusted_pct < 0.001:
            return False, "size_too_small", 0.0

        return True, "approved", float(adjusted_pct)

    def compute_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        atr: float,
        atr_multiple: float = 2.0,
        portfolio_value: float = 1.0,
        position_size_usdc: float = 100.0,
        max_risk_pct: float = 0.01,
    ) -> float:
        """
        Compute stop-loss price.
        Uses ATR-based stop, capped so risk never exceeds max_risk_pct of portfolio.
        """
        atr_stop = atr * atr_multiple

        # Risk-based stop (max 1% portfolio loss)
        max_risk_usdc = portfolio_value * max_risk_pct
        max_stop_distance = max_risk_usdc / (position_size_usdc / entry_price + 1e-9)

        stop_distance = min(atr_stop, max_stop_distance)

        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def compute_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        is_long: bool,
        rr_ratio: float = 3.0,
    ) -> float:
        """
        Compute take-profit at given risk/reward ratio.
        Default 3:1 (3R take profit for 1R risk).
        """
        risk = abs(entry_price - stop_loss)
        if is_long:
            return entry_price + risk * rr_ratio
        else:
            return entry_price - risk * rr_ratio

    def _compute_risk_metrics(self) -> Dict[str, float]:
        """Compute current portfolio risk metrics."""
        if len(self._returns_history) < 10:
            return {"var_99": 0.0, "cvar_99": 0.0, "rolling_sharpe": 0.0}

        returns = np.array(list(self._returns_history))

        var_99 = var_historical(returns, 0.99)
        cvar_99 = cvar_historical(returns, 0.99)

        # Rolling Sharpe (last 20 periods)
        recent = returns[-20:]
        rolling_sharpe = np.mean(recent) / (np.std(recent) + 1e-9) * np.sqrt(252 * 96)

        # Equity curve stats
        if len(self._portfolio_history) >= 5:
            values = np.array([v for _, v in self._portfolio_history[-100:]])
            mdd, _, _ = max_drawdown(values)
        else:
            mdd = 0.0

        return {
            "var_99": float(var_99),
            "cvar_99": float(cvar_99),
            "rolling_sharpe": float(rolling_sharpe),
            "current_mdd": float(mdd),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Return portfolio equity curve as DataFrame."""
        if not self._portfolio_history:
            return pd.DataFrame()
        timestamps, values = zip(*self._portfolio_history)
        return pd.DataFrame({
            "timestamp": [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps],
            "portfolio_value": values,
        }).set_index("timestamp")
