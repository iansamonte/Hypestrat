"""
Master Strategy Orchestrator — the brain of Hypestrat.

Architecture:
  1. Multi-timeframe data ingestion
  2. Regime detection (HMM + Kalman)
  3. Signal generation (7 independent signal sources)
  4. Regime-adjusted signal weighting
  5. Drawdown control gate
  6. Position sizing (Kelly + vol targeting)
  7. Risk validation
  8. Order execution

Signal weighting (base, before regime adjustment):
  technical:     20%  — EMA, RSI, MACD, BB, ADX
  momentum:      20%  — MTF momentum
  mean_reversion:15%  — OU process, Hurst
  orderflow:     20%  — CVD, VWAP, book imbalance
  funding:       10%  — Hyperliquid-specific edge
  ml:            15%  — LSTM + XGBoost ensemble

Dynamic IC-weighted update:
  Every N bars, update weights based on recent signal IC.
  Signals with higher IC get higher weight.
  This makes the system adaptive to changing market conditions.

Growth target math:
  Start: $5,000
  Target: $1,000,000 in 4 years (48 months)
  Required CAGR: 200^(1/4) - 1 ≈ 276%
  Required monthly: 200^(1/48) - 1 ≈ 11.34%
  Required daily (15min compounding): 200^(1/1460) - 1 ≈ 0.36%

At 5x leverage with 2% daily alpha: net 10% daily → overkill but allows for losers.
Realistic target: ~15-20% monthly with controlled drawdown < 20%.
"""
from __future__ import annotations

import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config import HypestratConfig, CONFIG
from exchange.hyperliquid_client import HyperliquidClient
from exchange.order_manager import OrderManager, OrderSide
from strategy.regime_detector import RegimeDetector
from strategy.signals.technical import TechnicalSignalGenerator
from strategy.signals.momentum import MomentumSignalGenerator
from strategy.signals.mean_reversion import MeanReversionSignalGenerator
from strategy.signals.orderflow import OrderFlowSignalGenerator
from strategy.signals.funding_rate import FundingRateSignalGenerator
from strategy.signals.volatility import VolatilitySignalGenerator
from strategy.signals.ml_signals import MLSignalGenerator
from strategy.risk.position_sizer import PositionSizer
from strategy.risk.risk_manager import RiskManager
from strategy.risk.drawdown_controller import DrawdownController
from utils.math_utils import weighted_signal, signal_consensus, information_coefficient
from utils.logger import logger


class MasterStrategy:
    """
    Orchestrates all signals, regime detection, risk management, and execution.

    The master loop:
      1. Fetch latest OHLCV + orderbook + funding rate data
      2. Detect market regime via HMM + Kalman
      3. Generate all 7 signal families
      4. Weight signals by regime + dynamic IC
      5. Compute composite signal
      6. Risk check (circuit breakers, heat, drawdown)
      7. Size position (Kelly + vol targeting)
      8. Execute trade (limit order with TWAP for large)
      9. Monitor and manage open position (trailing stop, TP)
      10. Record outcome → update Kelly estimates
    """

    # Base signal weights (before regime and IC adjustment)
    BASE_WEIGHTS = {
        "technical": 0.20,
        "momentum": 0.20,
        "mean_reversion": 0.15,
        "orderflow": 0.20,
        "funding": 0.10,
        "ml": 0.15,
    }

    def __init__(self, config: HypestratConfig = None) -> None:
        self.config = config or CONFIG
        self.coin = self.config.asset.symbol  # "HYPE"

        # Exchange
        self.client = HyperliquidClient(
            private_key=self.config.exchange.private_key,
            wallet_address=self.config.exchange.wallet_address,
            mainnet=self.config.exchange.mainnet,
        )
        self.order_manager = OrderManager(
            self.client,
            max_slippage_bps=self.config.execution.max_slippage_bps,
        )

        # Regime detector
        self.regime_detector = RegimeDetector(
            hmm_n_regimes=self.config.regime.n_regimes,
            hmm_n_iter=self.config.regime.hmm_n_iter,
            kalman_process_noise=self.config.regime.kalman_transition_cov,
            kalman_obs_noise=self.config.regime.kalman_observation_cov,
        )

        # Signal generators
        self.tech_signals = TechnicalSignalGenerator(
            rsi_period=self.config.signal.rsi_period,
            rsi_ob=self.config.signal.rsi_overbought,
            rsi_os=self.config.signal.rsi_oversold,
            macd_fast=self.config.signal.macd_fast,
            macd_slow=self.config.signal.macd_slow,
            macd_signal_period=self.config.signal.macd_signal,
            bb_period=self.config.signal.bb_period,
            bb_std=self.config.signal.bb_std,
            ema_periods=(self.config.signal.ema_short, self.config.signal.ema_medium, self.config.signal.ema_long),
            atr_period=self.config.signal.atr_period,
        )
        self.momentum_signals = MomentumSignalGenerator(windows=self.config.signal.momentum_windows)
        self.mr_signals = MeanReversionSignalGenerator(
            lookback=self.config.signal.mr_lookback,
            z_entry=self.config.signal.z_score_entry,
            z_exit=self.config.signal.z_score_exit,
            half_life_min=self.config.signal.ou_half_life_min,
            half_life_max=self.config.signal.ou_half_life_max,
        )
        self.of_signals = OrderFlowSignalGenerator()
        self.funding_signals = FundingRateSignalGenerator(
            long_threshold=self.config.signal.funding_long_threshold,
            short_threshold=self.config.signal.funding_short_threshold,
        )
        self.vol_signals = VolatilitySignalGenerator()
        self.ml_signals = MLSignalGenerator(
            seq_len=self.config.ml.lstm_sequence_len,
            lstm_weights=self.config.ml.ensemble_weights[0],
            xgb_weights=self.config.ml.ensemble_weights[1],
            min_train_samples=self.config.ml.min_train_samples,
        )

        # Risk management
        self.position_sizer = PositionSizer(
            kelly_fraction=self.config.risk.kelly_fraction,
            max_position_pct=self.config.risk.max_position_pct,
            max_portfolio_heat=self.config.risk.max_portfolio_heat,
            leverage_schedule=self.config.risk.leverage_schedule,
            initial_capital=self.config.capital.initial_capital,
        )
        self.risk_manager = RiskManager(
            daily_loss_limit=self.config.risk.daily_loss_limit_pct,
            weekly_loss_limit=self.config.risk.weekly_loss_limit_pct,
            monthly_loss_limit=self.config.risk.monthly_loss_limit_pct,
            max_drawdown_limit=self.config.risk.max_drawdown_pct,
            max_portfolio_heat=self.config.risk.max_portfolio_heat,
        )
        self.drawdown_controller = DrawdownController()

        # Dynamic IC-based signal weights
        self._dynamic_weights = dict(self.BASE_WEIGHTS)
        self._signal_history: List[Dict[str, float]] = []
        self._return_history: List[float] = []

        # State
        self._data_cache: Optional[pd.DataFrame] = None
        self._funding_cache: Optional[pd.DataFrame] = None
        self._ml_trained = False
        self._bars_since_ml_train = 0
        self._regime_state: Dict = {}
        self._portfolio_value: float = self.config.capital.initial_capital

        logger.info(
            f"MasterStrategy initialised | {self.coin}/USDC | "
            f"Initial capital: ${self.config.capital.initial_capital:,.0f} | "
            f"Target: ${self.config.capital.target_capital:,.0f} in {self.config.capital.target_years}y"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────────────────────────

    def run_once(self) -> Dict[str, Any]:
        """
        Single strategy iteration: fetch data → signals → risk check → trade.
        Returns current state dict for monitoring.
        """
        # 1. Fetch data
        df = self._fetch_data()
        if df is None or len(df) < 100:
            logger.warning("Insufficient data for strategy computation")
            return {"status": "insufficient_data"}

        # 2. Update portfolio value
        self._portfolio_value = self.client.get_portfolio_value()
        if self._portfolio_value <= 0:
            self._portfolio_value = self.config.capital.initial_capital

        # 3. Risk state update
        risk_state = self.risk_manager.update(self._portfolio_value)
        dd_state = self.drawdown_controller.update(self._portfolio_value)

        if risk_state["halted"] or dd_state["halted"]:
            self._manage_open_positions(df, force_close=True)
            return {"status": "halted", "reason": risk_state.get("halt_reason", "drawdown")}

        # 4. Detect regime
        self._regime_state = self.regime_detector.detect(df)
        regime = self._regime_state["regime"]

        # 5. Generate all signals
        all_signals = self._generate_signals(df)

        # 6. Compute composite signal with regime + IC weighting
        composite, direction, signal_strength, n_agreeing = self._compute_composite_signal(
            all_signals, self._regime_state
        )

        # 7. ML training check
        self._maybe_train_ml(df)

        # 8. Check if we should act
        should_trade, trade_reason = self.drawdown_controller.should_take_trade(
            signal_strength=signal_strength,
            n_agreeing_signals=n_agreeing,
        )

        # 9. Manage existing positions
        self._manage_open_positions(df)

        # 10. Open new trade if conditions met
        trade_result = None
        if should_trade and abs(composite) >= self.config.risk.signal_threshold:
            trade_result = self._execute_trade(df, composite, direction, all_signals)

        # 11. Update dynamic weights (every 50 bars)
        self._bars_since_ml_train += 1
        if len(self._signal_history) > 50 and self._bars_since_ml_train % 50 == 0:
            self._update_signal_weights()

        state = {
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": self._portfolio_value,
            "regime": regime,
            "composite_signal": composite,
            "signal_direction": direction,
            "signal_strength": signal_strength,
            "n_agreeing_signals": n_agreeing,
            "should_trade": should_trade,
            "trade_reason": trade_reason,
            "trade_result": trade_result,
            "risk_state": risk_state,
            "dd_state": dd_state,
            "open_trades": len(self.order_manager.get_open_trades()),
            "portfolio_heat": self.risk_manager.current_heat,
            "signal_weights": self._dynamic_weights,
        }

        self._log_state(state)
        return state

    def run_loop(self, interval_seconds: int = 900) -> None:
        """
        Continuous trading loop. interval_seconds = 900 for 15-min bars.
        """
        logger.info(f"Starting trading loop | interval={interval_seconds}s | coin={self.coin}")
        while True:
            try:
                state = self.run_once()
                logger.info(
                    f"Loop | portfolio=${state.get('portfolio_value', 0):,.2f} | "
                    f"regime={state.get('regime', '?')} | signal={state.get('composite_signal', 0):.3f} | "
                    f"trades={state.get('open_trades', 0)}"
                )
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Strategy stopped by user")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(60)  # Brief pause on error

    # ──────────────────────────────────────────────────────────────────────────
    # Data
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch primary timeframe OHLCV data."""
        try:
            interval = "15m"
            df = self.client.get_candles_n(self.coin, interval, n=600)
            if not df.empty:
                self._data_cache = df
                # Also refresh funding data
                end_ms = int(time.time() * 1000)
                start_ms = end_ms - 30 * 24 * 3600 * 1000
                self._funding_cache = self.client.get_funding_history(self.coin, start_ms, end_ms)
            return df
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return self._data_cache

    # ──────────────────────────────────────────────────────────────────────────
    # Signal Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Generate all signal families. Returns nested dict:
        {family_name: {signal_name: value}}
        """
        signals = {}

        # Technical signals
        try:
            signals["technical"] = self.tech_signals.compute(df)
        except Exception as e:
            logger.error(f"Technical signals error: {e}")
            signals["technical"] = {"technical_composite": 0.0}

        # Momentum signals
        try:
            signals["momentum"] = self.momentum_signals.compute(df)
        except Exception as e:
            logger.error(f"Momentum signals error: {e}")
            signals["momentum"] = {"momentum_composite": 0.0}

        # Mean reversion signals
        try:
            signals["mean_reversion"] = self.mr_signals.compute(df)
        except Exception as e:
            logger.error(f"MR signals error: {e}")
            signals["mean_reversion"] = {"mr_composite": 0.0}

        # Order flow signals
        try:
            book = self.client.get_orderbook(self.coin)
            signals["orderflow"] = self.of_signals.compute(
                df, bids=book.get("bids"), asks=book.get("asks")
            )
        except Exception as e:
            logger.error(f"Orderflow signals error: {e}")
            signals["orderflow"] = {"orderflow_composite": 0.0}

        # Funding rate signals
        try:
            current_funding = self.client.get_current_funding_rate(self.coin)
            signals["funding"] = self.funding_signals.compute(
                current_funding=current_funding,
                funding_history=self._funding_cache,
            )
        except Exception as e:
            logger.error(f"Funding signals error: {e}")
            signals["funding"] = {"funding_composite": 0.0}

        # Volatility signals
        try:
            signals["volatility"] = self.vol_signals.compute(df)
        except Exception as e:
            logger.error(f"Volatility signals error: {e}")
            signals["volatility"] = {"volatility_composite": 0.0, "position_size_scalar": 0.5, "vol_regime": 0.5}

        # ML signals
        try:
            signals["ml"] = self.ml_signals.predict(df)
        except Exception as e:
            logger.error(f"ML signals error: {e}")
            signals["ml"] = {"ml_composite": 0.0, "ml_confidence": 0.0}

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # Signal Combination
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_composite_signal(
        self,
        all_signals: Dict[str, Dict],
        regime_state: Dict,
    ) -> Tuple[float, int, float, int]:
        """
        Combine all signal families into a single composite.
        Returns (composite_signal, direction, strength, n_agreeing).
        direction: +1 = long, -1 = short, 0 = flat
        """
        regime_weights = regime_state.get("signal_weights", {k: 1.0 for k in self.BASE_WEIGHTS})
        vol_scalar = all_signals.get("volatility", {}).get("position_size_scalar", 1.0)

        # Extract composite from each family
        family_composites = {
            "technical": all_signals["technical"].get("technical_composite", 0.0),
            "momentum": all_signals["momentum"].get("momentum_composite", 0.0),
            "mean_reversion": all_signals["mean_reversion"].get("mr_composite", 0.0),
            "orderflow": all_signals["orderflow"].get("orderflow_composite", 0.0),
            "funding": all_signals["funding"].get("funding_composite", 0.0),
            "ml": all_signals["ml"].get("ml_composite", 0.0),
        }

        # Strategy mode: volatility signal tells us to emphasise momentum or MR
        vol_mode = all_signals.get("volatility", {}).get("strategy_mode", 0.0)
        if vol_mode > 0:
            # High vol → favour momentum over mean reversion
            regime_weights["momentum"] = regime_weights.get("momentum", 1.0) * (1.0 + vol_mode * 0.5)
            regime_weights["mean_reversion"] = regime_weights.get("mean_reversion", 1.0) * (1.0 - vol_mode * 0.3)
        elif vol_mode < 0:
            # Low vol → favour mean reversion
            regime_weights["mean_reversion"] = regime_weights.get("mean_reversion", 1.0) * (1.0 - vol_mode * 0.5)

        # Compute regime × IC weighted composite
        total_weight = 0.0
        weighted_sum = 0.0
        for family, base_w in self._dynamic_weights.items():
            if family not in family_composites:
                continue
            sig = family_composites[family]
            regime_w = regime_weights.get(family, 1.0)
            effective_w = base_w * regime_w
            if not np.isnan(sig):
                weighted_sum += sig * effective_w
                total_weight += effective_w

        composite = float(np.clip(weighted_sum / total_weight, -1, 1)) if total_weight > 0 else 0.0

        # Kalman trend signal as tiebreaker / amplifier
        trend_signal = regime_state.get("trend_signal", 0.0)
        if abs(composite) > 0.2 and np.sign(composite) == np.sign(trend_signal):
            composite = float(np.clip(composite * 1.1, -1, 1))  # Amplify if trend confirms

        # Signal consensus count
        threshold = self.config.risk.signal_threshold * 0.5
        n_agreeing = sum(
            1 for s in family_composites.values() if abs(s) > threshold and np.sign(s) == np.sign(composite)
        )

        direction = int(np.sign(composite)) if abs(composite) >= self.config.risk.signal_threshold else 0
        strength = float(abs(composite))

        # Store for IC weight update
        self._signal_history.append(family_composites)

        return composite, direction, strength, n_agreeing

    # ──────────────────────────────────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_trade(
        self,
        df: pd.DataFrame,
        composite: float,
        direction: int,
        all_signals: Dict,
    ) -> Optional[Dict]:
        """Open a new trade based on composite signal."""
        if direction == 0:
            return None

        current_price = float(df["close"].iloc[-1])
        atr = all_signals.get("technical", {}).get("_atr", current_price * 0.015)

        # Position sizing
        vol_scalar = all_signals.get("volatility", {}).get("position_size_scalar", 1.0)
        ml_confidence = all_signals.get("ml", {}).get("ml_confidence", 0.5)
        vol_forecast = all_signals.get("volatility", {}).get("_vol_forecast", 0.02)

        is_long = direction > 0
        stop_pct = self.config.risk.stop_loss_atr * (atr / current_price)
        stop_loss = self.risk_manager.compute_stop_loss(
            current_price, is_long, atr,
            atr_multiple=self.config.risk.stop_loss_atr,
            portfolio_value=self._portfolio_value,
            position_size_usdc=self._portfolio_value * 0.1,
        )
        take_profit = self.risk_manager.compute_take_profit(
            current_price, stop_loss, is_long,
            rr_ratio=self.config.risk.take_profit_atr / self.config.risk.stop_loss_atr,
        )

        # Drawdown-adjusted stop
        stop_loss = self.drawdown_controller.adjust_stop_loss(stop_loss, current_price, is_long)

        size_result = self.position_sizer.compute_size(
            portfolio_value=self._portfolio_value,
            signal_strength=composite,
            signal_confidence=ml_confidence,
            current_price=current_price,
            daily_volatility=vol_forecast * np.sqrt(96),  # 15min vol → daily vol
            current_heat=self.risk_manager.current_heat,
            stop_loss_pct=stop_pct,
            atr_pct=atr / current_price,
        )

        # Apply drawdown size reduction
        size_usdc = self.drawdown_controller.apply_size_reduction(size_result["size_usdc"])
        size_usdc *= vol_scalar

        if size_usdc < 10.0:
            return None

        size_units = size_usdc / current_price
        leverage = size_result["leverage"]

        # Risk validate
        position_pct = size_usdc / (self._portfolio_value * leverage)
        can_trade, reason, approved_pct = self.risk_manager.validate_trade(
            self._portfolio_value, position_pct, composite
        )

        if not can_trade:
            logger.debug(f"Trade rejected: {reason}")
            return None

        # Execute
        side = OrderSide.BUY if is_long else OrderSide.SELL
        trade = self.order_manager.open_trade(
            coin=self.coin,
            side=side,
            size=round(size_units, 4),
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            use_limit=self.config.execution.use_limit_orders,
            limit_offset_bps=self.config.execution.limit_order_offset_bps,
        )

        if trade:
            self.risk_manager.update_position_heat(
                self.coin, approved_pct
            )
            logger.info(
                f"Trade opened | {side.value} {size_units:.4f} {self.coin} "
                f"@ {current_price:.4f} | SL={stop_loss:.4f} TP={take_profit:.4f} | "
                f"Signal={composite:.3f} | {leverage}x | ${size_usdc:.0f}"
            )
            return {
                "trade_id": trade.trade_id,
                "side": side.value,
                "size": size_units,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "leverage": leverage,
                "size_usdc": size_usdc,
            }
        return None

    def _manage_open_positions(self, df: pd.DataFrame, force_close: bool = False) -> None:
        """Check stop-loss/take-profit on all open trades."""
        current_price = float(df["close"].iloc[-1])
        atr = float(df["high"].iloc[-1] - df["low"].iloc[-1])  # Quick ATR approx

        open_trades = self.order_manager.get_open_trades()
        for trade in open_trades:
            if force_close:
                pnl = self.order_manager.close_trade(trade.trade_id, current_price)
                self.risk_manager.remove_position_heat(trade.coin)
                self.position_sizer.record_trade(pnl / (self._portfolio_value + 1e-9), 0.5)
                continue

            # Update trailing stop
            self.order_manager.update_trailing_stop(
                trade, current_price, atr,
                atr_multiple=self.config.risk.trailing_stop_atr,
            )

            # Drawdown-adjusted stop
            trade.stop_loss = self.drawdown_controller.adjust_stop_loss(
                trade.stop_loss, current_price, trade.side.value == "BUY"
            )

            # Check stops
            trigger = self.order_manager.check_stops(trade, current_price)
            if trigger:
                pnl = self.order_manager.close_trade(trade.trade_id, current_price)
                self.risk_manager.remove_position_heat(trade.coin)
                pnl_pct = pnl / (self._portfolio_value + 1e-9)
                self.position_sizer.record_trade(pnl_pct, 0.5)
                self._return_history.append(pnl_pct)
                logger.info(f"Position closed | trigger={trigger} | PnL={pnl:+.2f} USDC ({pnl_pct:+.2%})")

    # ──────────────────────────────────────────────────────────────────────────
    # Adaptive Signal Weights (IC-based)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_signal_weights(self) -> None:
        """
        Update signal weights based on recent Information Coefficient.
        Signals with higher IC → higher weight.
        IC = rank correlation between signal and subsequent return.
        """
        if len(self._signal_history) < 30 or len(self._return_history) < 30:
            return

        n = min(len(self._signal_history), len(self._return_history), 100)
        sig_hist = self._signal_history[-n:]
        ret_hist = np.array(self._return_history[-n:])

        new_weights = {}
        for family in self.BASE_WEIGHTS:
            family_sigs = np.array([s.get(family, 0.0) for s in sig_hist])
            if len(family_sigs) >= 20:
                ic = information_coefficient(family_sigs, ret_hist)
                # IC-based weight: max(IC, 0) so negative-IC signals get minimal weight
                new_weights[family] = max(ic, 0.02) * self.BASE_WEIGHTS[family] * 10
            else:
                new_weights[family] = self.BASE_WEIGHTS[family]

        # Normalise weights to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            self._dynamic_weights = {k: v / total for k, v in new_weights.items()}
            logger.info(f"Signal weights updated: {self._dynamic_weights}")

    # ──────────────────────────────────────────────────────────────────────────
    # ML Training
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_train_ml(self, df: pd.DataFrame) -> None:
        """Retrain ML models if enough time has passed."""
        retrain_every = self.config.ml.xgb_retrain_hours * 4  # 4 bars per hour at 15min
        if self.ml_signals.needs_retraining(self._bars_since_ml_train, retrain_every):
            logger.info("Initiating ML model retraining...")
            try:
                self.ml_signals.train(df)
                logger.info("ML retraining complete")
            except Exception as e:
                logger.error(f"ML training error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Monitoring
    # ──────────────────────────────────────────────────────────────────────────

    def _log_state(self, state: Dict) -> None:
        """Log current state for monitoring."""
        pv = state.get("portfolio_value", 0)
        target = self.config.capital.target_capital
        initial = self.config.capital.initial_capital
        progress = (pv - initial) / (target - initial) * 100
        multiple = pv / initial

        logger.debug(
            f"Portfolio: ${pv:,.2f} ({multiple:.2f}x) | "
            f"Progress to goal: {progress:.1f}% | "
            f"Regime: {state.get('regime')} | "
            f"Signal: {state.get('composite_signal', 0):.3f} | "
            f"Heat: {state.get('portfolio_heat', 0):.1%}"
        )

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        pv = self._portfolio_value
        initial = self.config.capital.initial_capital
        target = self.config.capital.target_capital

        equity_curve = self.risk_manager.get_equity_curve()
        sizing_perf = self.position_sizer.get_performance_summary()

        monthly_r = self.config.capital.target_monthly_return
        n_months_elapsed = len(self._return_history) / (96 * 20)  # Approx
        expected_value = initial * (1 + monthly_r) ** n_months_elapsed

        report = {
            "portfolio_value": pv,
            "initial_capital": initial,
            "target_capital": target,
            "current_multiple": pv / initial,
            "progress_pct": (pv - initial) / (target - initial) * 100,
            "expected_value_on_target": expected_value,
            "on_track": pv >= expected_value * 0.8,  # Within 20% of target path
            "regime": self._regime_state.get("regime", "unknown"),
            "signal_weights": self._dynamic_weights,
            "sizing_performance": sizing_perf,
            "risk_metrics": self.risk_manager._compute_risk_metrics(),
            "open_trades": len(self.order_manager.get_open_trades()),
            "total_trades": len(self.order_manager.get_all_trades()),
            "drawdown_level": self.drawdown_controller._dd_level,
            "current_drawdown": self.drawdown_controller.current_drawdown,
        }
        return report
