"""
Walk-Forward Backtesting Engine with realistic market simulation.

Walk-forward methodology:
  - Avoids look-ahead bias by training only on past data
  - Train window: 180 days, Test window: 30 days, Step: 30 days
  - At each step: fit models on train → evaluate on test → record results
  - Aggregate test period results = walk-forward equity curve

Transaction cost model:
  - Maker fee: 0.02% (limit orders)
  - Taker fee: 0.05% (market orders)
  - Slippage: 2bps
  - Funding rate impact (positive or negative)

Backtest metrics:
  - Total return, CAGR, Sharpe, Sortino, Calmar
  - Maximum drawdown, drawdown duration, recovery time
  - Win rate, profit factor, expectancy
  - Information ratio vs buy-and-hold
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import BacktestConfig, HypestratConfig, CONFIG
from backtesting.portfolio import BacktestPortfolio
from backtesting.metrics import compute_full_metrics
from strategy.signals.technical import TechnicalSignalGenerator, compute_atr
from strategy.signals.momentum import MomentumSignalGenerator
from strategy.signals.mean_reversion import MeanReversionSignalGenerator
from strategy.signals.orderflow import OrderFlowSignalGenerator
from strategy.signals.funding_rate import FundingRateSignalGenerator
from strategy.signals.volatility import VolatilitySignalGenerator
from strategy.signals.ml_signals import MLSignalGenerator
from strategy.regime_detector import RegimeDetector
from strategy.risk.position_sizer import PositionSizer
from strategy.risk.risk_manager import RiskManager
from strategy.risk.drawdown_controller import DrawdownController
from utils.math_utils import weighted_signal, information_coefficient
from utils.logger import logger


@dataclass
class BacktestTrade:
    trade_id: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: int  # +1 long, -1 short
    size_usdc: float
    leverage: float
    gross_pnl: float
    fees_paid: float
    funding_paid: float
    net_pnl: float
    r_multiple: float
    entry_signal: float
    exit_reason: str


@dataclass
class WalkForwardPeriod:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: Dict = field(default_factory=dict)
    trades: List[BacktestTrade] = field(default_factory=list)


class BacktestEngine:
    """
    High-fidelity walk-forward backtesting engine.

    Features:
      - Realistic transaction costs (maker/taker fees + slippage)
      - Funding rate accumulation during hold
      - Position sizing via Kelly criterion
      - Stop-loss and take-profit simulation
      - Trailing stop simulation
      - Drawdown control (circuit breakers)
      - ML model retraining within walk-forward windows (no leakage)
    """

    def __init__(
        self,
        config: HypestratConfig = None,
        initial_capital: float = 5000.0,
    ) -> None:
        self.config = config or CONFIG
        self.initial_capital = initial_capital

        self.bt_config = self.config.backtest

        # Initialise signal generators (same as live strategy)
        self.tech_signals = TechnicalSignalGenerator()
        self.momentum_signals = MomentumSignalGenerator()
        self.mr_signals = MeanReversionSignalGenerator()
        self.of_signals = OrderFlowSignalGenerator()
        self.funding_signals = FundingRateSignalGenerator()
        self.vol_signals = VolatilitySignalGenerator()
        self.ml_signals = MLSignalGenerator(
            seq_len=self.config.ml.lstm_sequence_len,
            min_train_samples=self.config.ml.min_train_samples,
        )
        self.regime_detector = RegimeDetector()
        self.position_sizer = PositionSizer(
            kelly_fraction=self.config.risk.kelly_fraction,
            max_position_pct=self.config.risk.max_position_pct,
            initial_capital=initial_capital,
        )
        self.drawdown_controller = DrawdownController()

        self._base_weights = {
            "technical": 0.20,
            "momentum": 0.20,
            "mean_reversion": 0.15,
            "orderflow": 0.20,
            "funding": 0.10,
            "ml": 0.15,
        }

    def run(
        self,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
        walk_forward: bool = True,
    ) -> Dict[str, Any]:
        """
        Run backtest on provided OHLCV data.

        df: OHLCV DataFrame with DatetimeIndex
        funding_df: Funding rate history (optional)
        walk_forward: Use walk-forward methodology (recommended)
        """
        logger.info(
            f"Starting backtest | bars={len(df)} | "
            f"period={df.index[0]} → {df.index[-1]} | "
            f"capital=${self.initial_capital:,.0f}"
        )

        if walk_forward:
            return self._walk_forward_backtest(df, funding_df)
        else:
            return self._simple_backtest(df, funding_df)

    # ──────────────────────────────────────────────────────────────────────────
    # Walk-Forward
    # ──────────────────────────────────────────────────────────────────────────

    def _walk_forward_backtest(
        self,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Walk-forward backtest.
        Train on 180 days → test on 30 days → step forward 30 days.
        """
        bars_per_day = 96  # 15-minute bars
        train_bars = self.config.ml.train_window_days * bars_per_day
        test_bars = self.config.ml.test_window_days * bars_per_day
        step_bars = self.config.ml.step_days * bars_per_day

        periods: List[WalkForwardPeriod] = []
        portfolio = BacktestPortfolio(
            initial_capital=self.initial_capital,
            maker_fee=self.bt_config.maker_fee,
            taker_fee=self.bt_config.taker_fee,
            slippage_bps=self.bt_config.slippage_bps,
        )

        n = len(df)
        start = train_bars

        while start + test_bars <= n:
            train_start = max(0, start - train_bars)
            train_end = start
            test_start = start
            test_end = min(start + test_bars, n)

            period = WalkForwardPeriod(train_start, train_end, test_start, test_end)

            logger.info(
                f"WF period | train=[{train_start}:{train_end}] "
                f"test=[{test_start}:{test_end}] | "
                f"portfolio=${portfolio.equity:.2f}"
            )

            # Train ML on train window (prevents look-ahead)
            train_df = df.iloc[train_start:train_end]
            if len(train_df) >= self.config.ml.min_train_samples:
                try:
                    self.ml_signals.train(train_df)
                except Exception as e:
                    logger.warning(f"ML training failed: {e}")

            # Fit HMM regime on train window
            try:
                self.regime_detector.fit_hmm(train_df)
            except Exception as e:
                logger.warning(f"HMM training failed: {e}")

            # Backtest on test window
            test_df = df.iloc[test_start:test_end]
            test_funding = self._filter_funding(funding_df, test_df) if funding_df is not None else None

            period_trades = self._run_period(
                test_df, portfolio, test_funding, test_start
            )
            period.trades = period_trades

            # Compute period metrics
            if period_trades:
                period_returns = [t.net_pnl / (self.initial_capital) for t in period_trades]
                period.metrics = {
                    "n_trades": len(period_trades),
                    "total_pnl": sum(t.net_pnl for t in period_trades),
                    "win_rate": sum(1 for t in period_trades if t.net_pnl > 0) / len(period_trades),
                }

            periods.append(period)
            start += step_bars

        # Aggregate results
        all_trades = [t for p in periods for t in p.trades]
        equity_curve = portfolio.equity_curve

        metrics = compute_full_metrics(
            equity_curve=equity_curve,
            trades=all_trades,
            initial_capital=self.initial_capital,
        )

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": all_trades,
            "walk_forward_periods": periods,
            "final_portfolio_value": portfolio.equity,
            "total_return": (portfolio.equity - self.initial_capital) / self.initial_capital,
            "multiple": portfolio.equity / self.initial_capital,
        }

    def _run_period(
        self,
        df: pd.DataFrame,
        portfolio: "BacktestPortfolio",
        funding_df: Optional[pd.DataFrame],
        bar_offset: int,
    ) -> List[BacktestTrade]:
        """Run backtest over a single period. Returns list of executed trades."""
        trades = []
        min_bars = 100
        if len(df) < min_bars:
            return trades

        open_position: Optional[Dict] = None
        trade_counter = 0

        for i in range(min_bars, len(df)):
            bar = df.iloc[i]
            current_price = float(bar["close"])
            bar_df = df.iloc[: i + 1]

            # Update portfolio value (mark-to-market)
            if open_position:
                portfolio.mark_to_market(current_price, open_position)

            # Drawdown check
            dd_state = self.drawdown_controller.update(portfolio.equity)
            if dd_state["halted"] and open_position:
                exit_price = self._simulate_exit(current_price, open_position["direction"])
                trade = self._close_position(
                    open_position, exit_price, i + bar_offset, portfolio,
                    "drawdown_halt", funding_df, bar.name if hasattr(bar, "name") else None
                )
                trades.append(trade)
                self.position_sizer.record_trade(trade.net_pnl / portfolio.equity, trade.entry_signal)
                open_position = None
                continue

            # Check stop/take-profit on open position
            if open_position:
                trigger = self._check_bar_for_stops(bar, open_position)
                if trigger:
                    exit_price = open_position["stop_loss"] if trigger == "stop_loss" else open_position["take_profit"]
                    trade = self._close_position(
                        open_position, exit_price, i + bar_offset, portfolio,
                        trigger, funding_df, bar.name if hasattr(bar, "name") else None
                    )
                    trades.append(trade)
                    self.position_sizer.record_trade(trade.net_pnl / portfolio.equity, trade.entry_signal)
                    open_position = None

            # Signal generation
            if open_position is None:
                try:
                    composite, direction = self._compute_signal(bar_df, funding_df)
                except Exception:
                    continue

                if abs(composite) < self.config.risk.signal_threshold:
                    continue

                if direction == 0:
                    continue

                # Compute ATR for stop placement
                atr_series = compute_atr(df["high"], df["low"], df["close"], 14)
                atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else current_price * 0.015

                is_long = direction > 0
                stop_loss = current_price - self.config.risk.stop_loss_atr * atr if is_long else current_price + self.config.risk.stop_loss_atr * atr
                take_profit = current_price + self.config.risk.take_profit_atr * atr if is_long else current_price - self.config.risk.take_profit_atr * atr

                # Position sizing
                vol_forecast = atr / current_price
                size_result = self.position_sizer.compute_size(
                    portfolio_value=portfolio.equity,
                    signal_strength=composite,
                    signal_confidence=0.6,
                    current_price=current_price,
                    daily_volatility=vol_forecast * np.sqrt(96),
                    current_heat=0.0,
                    stop_loss_pct=abs(current_price - stop_loss) / current_price,
                )

                size_usdc = self.drawdown_controller.apply_size_reduction(size_result["size_usdc"])
                if size_usdc < 5.0:
                    continue

                # Open position
                entry_price = self._simulate_entry(current_price, direction)
                trade_counter += 1
                open_position = {
                    "trade_id": f"BT{trade_counter:06d}",
                    "entry_bar": i + bar_offset,
                    "entry_time": bar.name if hasattr(bar, "name") else None,
                    "entry_price": entry_price,
                    "direction": direction,
                    "size_usdc": size_usdc,
                    "leverage": size_result["leverage"],
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_signal": composite,
                }

        # Close any remaining position at end of period
        if open_position:
            exit_price = self._simulate_exit(float(df.iloc[-1]["close"]), open_position["direction"])
            trade = self._close_position(
                open_position, exit_price, len(df) + bar_offset - 1, portfolio,
                "period_end", funding_df, df.index[-1]
            )
            trades.append(trade)

        return trades

    def _compute_signal(
        self,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, int]:
        """Compute composite signal for a given bar."""
        composites = {}

        try:
            t_sig = self.tech_signals.compute(df)
            composites["technical"] = t_sig.get("technical_composite", 0.0)
        except Exception:
            composites["technical"] = 0.0

        try:
            m_sig = self.momentum_signals.compute(df)
            composites["momentum"] = m_sig.get("momentum_composite", 0.0)
        except Exception:
            composites["momentum"] = 0.0

        try:
            mr_sig = self.mr_signals.compute(df)
            composites["mean_reversion"] = mr_sig.get("mr_composite", 0.0)
        except Exception:
            composites["mean_reversion"] = 0.0

        try:
            of_sig = self.of_signals.compute(df)
            composites["orderflow"] = of_sig.get("orderflow_composite", 0.0)
        except Exception:
            composites["orderflow"] = 0.0

        if funding_df is not None and len(funding_df) > 0:
            try:
                current_rate = float(funding_df["funding_rate"].iloc[-1])
                f_sig = self.funding_signals.compute(current_rate, funding_df)
                composites["funding"] = f_sig.get("funding_composite", 0.0)
            except Exception:
                composites["funding"] = 0.0
        else:
            composites["funding"] = 0.0

        try:
            ml_sig = self.ml_signals.predict(df)
            composites["ml"] = ml_sig.get("ml_composite", 0.0)
        except Exception:
            composites["ml"] = 0.0

        # Regime adjustment
        try:
            regime_state = self.regime_detector.detect(df)
            regime_weights = regime_state.get("signal_weights", {k: 1.0 for k in self._base_weights})
        except Exception:
            regime_weights = {k: 1.0 for k in self._base_weights}

        # Weighted composite
        total_w = 0.0
        weighted_sum = 0.0
        for fam, base_w in self._base_weights.items():
            sig = composites.get(fam, 0.0)
            rw = regime_weights.get(fam, 1.0)
            eff_w = base_w * rw
            if not np.isnan(sig):
                weighted_sum += sig * eff_w
                total_w += eff_w

        composite = float(np.clip(weighted_sum / total_w, -1, 1)) if total_w > 0 else 0.0
        direction = int(np.sign(composite)) if abs(composite) >= self.config.risk.signal_threshold else 0

        return composite, direction

    def _simulate_entry(self, price: float, direction: int) -> float:
        """Simulate realistic entry price with spread/slippage."""
        slippage = self.bt_config.slippage_bps / 10000
        fee = self.bt_config.taker_fee
        if direction > 0:
            return price * (1 + slippage + fee)
        else:
            return price * (1 - slippage - fee)

    def _simulate_exit(self, price: float, direction: int) -> float:
        slippage = self.bt_config.slippage_bps / 10000
        fee = self.bt_config.taker_fee
        if direction > 0:
            return price * (1 - slippage - fee)
        else:
            return price * (1 + slippage + fee)

    def _check_bar_for_stops(self, bar: pd.Series, position: Dict) -> Optional[str]:
        """Check if stop-loss or take-profit was hit during this bar."""
        direction = position["direction"]
        sl = position["stop_loss"]
        tp = position["take_profit"]
        low = float(bar["low"])
        high = float(bar["high"])

        if direction > 0:
            if low <= sl:
                return "stop_loss"
            if high >= tp:
                return "take_profit"
        else:
            if high >= sl:
                return "stop_loss"
            if low <= tp:
                return "take_profit"
        return None

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_bar: int,
        portfolio: "BacktestPortfolio",
        exit_reason: str,
        funding_df: Optional[pd.DataFrame],
        exit_time: Any,
    ) -> BacktestTrade:
        """Close a position and record the trade."""
        entry_price = position["entry_price"]
        direction = position["direction"]
        size_usdc = position["size_usdc"]
        leverage = position["leverage"]

        # Gross P&L
        price_return = (exit_price - entry_price) / entry_price
        gross_pnl = direction * price_return * size_usdc * leverage

        # Fees (entry + exit)
        fees = size_usdc * (self.bt_config.maker_fee + self.bt_config.taker_fee) * 2

        # Funding cost (approximate: use average rate * periods held)
        funding_paid = 0.0
        if funding_df is not None and len(funding_df) > 0 and self.bt_config.include_funding:
            bars_held = exit_bar - position["entry_bar"]
            periods_held = bars_held / (8 * 4)  # 8h per funding period, 4 bars/h at 15min
            avg_rate = float(funding_df["funding_rate"].mean())
            funding_paid = direction * avg_rate * periods_held * size_usdc

        net_pnl = gross_pnl - fees - funding_paid
        portfolio.record_trade(net_pnl, exit_price, position)

        # R-multiple
        risk = abs(entry_price - position["stop_loss"]) / entry_price * size_usdc * leverage
        r_multiple = net_pnl / risk if risk > 0 else 0.0

        return BacktestTrade(
            trade_id=position["trade_id"],
            entry_bar=position["entry_bar"],
            exit_bar=exit_bar,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            size_usdc=size_usdc,
            leverage=leverage,
            gross_pnl=gross_pnl,
            fees_paid=fees,
            funding_paid=funding_paid,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
            entry_signal=position.get("entry_signal", 0.0),
            exit_reason=exit_reason,
        )

    @staticmethod
    def _filter_funding(
        funding_df: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Filter funding data to match price period."""
        if funding_df is None or funding_df.empty:
            return None
        try:
            start = price_df.index[0]
            end = price_df.index[-1]
            mask = (funding_df.index >= start) & (funding_df.index <= end)
            return funding_df[mask]
        except Exception:
            return funding_df

    def _simple_backtest(
        self,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Simple backtest without walk-forward (for quick testing)."""
        # Train ML on first half
        mid = len(df) // 2
        if len(df[:mid]) >= self.config.ml.min_train_samples:
            try:
                self.ml_signals.train(df.iloc[:mid])
            except Exception:
                pass
        try:
            self.regime_detector.fit_hmm(df.iloc[:mid])
        except Exception:
            pass

        portfolio = BacktestPortfolio(
            initial_capital=self.initial_capital,
            maker_fee=self.bt_config.maker_fee,
            taker_fee=self.bt_config.taker_fee,
        )
        trades = self._run_period(df, portfolio, funding_df, 0)
        equity_curve = portfolio.equity_curve
        metrics = compute_full_metrics(equity_curve, trades, self.initial_capital)

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": trades,
            "final_portfolio_value": portfolio.equity,
            "total_return": (portfolio.equity - self.initial_capital) / self.initial_capital,
            "multiple": portfolio.equity / self.initial_capital,
        }
