"""
Volatility regime signal generator using GARCH and realised volatility.

Volatility regime determines:
  1. Position sizing (higher vol → smaller size via risk parity)
  2. Strategy selection (high vol → breakout/momentum; low vol → mean reversion)
  3. Stop placement (ATR-based stops scale with vol)
  4. Entry timing (avoid entering in high-vol chop)

Models:
  - GARCH(1,1): conditional volatility forecast
  - EGARCH: asymmetric volatility (leverage effect)
  - Realised volatility: Yang-Zhang estimator
  - Volatility regime detection (low/normal/high)
  - Volatility of volatility (vol-of-vol): detects vol regime transitions
  - ATR percentile rank for normalisation
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

warnings.filterwarnings("ignore")

from utils.math_utils import (
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
)


def fit_garch_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> Tuple[float, float, Dict]:
    """
    Fit GARCH(p,q) model and return one-step-ahead volatility forecast.
    Returns (forecast_vol, current_vol, model_params)
    """
    try:
        from arch import arch_model
        model = arch_model(
            returns.dropna() * 100,
            vol="Garch",
            p=p,
            q=q,
            mean="Constant",
            dist="t",  # Student-t for fat tails
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)

        forecast = result.forecast(horizon=1, reindex=False)
        forecast_vol = float(np.sqrt(forecast.variance.values[-1, 0]) / 100)
        current_vol = float(np.sqrt(result.conditional_volatility.iloc[-1]) / 100)

        params = {
            "omega": float(result.params.get("omega", 0)),
            "alpha": float(result.params.get("alpha[1]", 0)),
            "beta": float(result.params.get("beta[1]", 0)),
            "persistence": float(result.params.get("alpha[1]", 0) + result.params.get("beta[1]", 0)),
            "half_life": np.log(2) / (1 - float(result.params.get("alpha[1]", 0) + result.params.get("beta[1]", 0)) + 1e-9),
        }
        return forecast_vol, current_vol, params

    except Exception:
        vol = float(returns.std()) if len(returns) > 2 else 0.01
        return vol, vol, {}


def fit_egarch_volatility(returns: pd.Series) -> Tuple[float, Dict]:
    """
    EGARCH(1,1): captures asymmetric volatility (leverage effect).
    Negative returns increase volatility more than positive returns of same magnitude.
    """
    try:
        from arch import arch_model
        model = arch_model(
            returns.dropna() * 100,
            vol="EGarch",
            p=1,
            o=1,
            q=1,
            mean="Constant",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)

        forecast = result.forecast(horizon=1, reindex=False)
        forecast_vol = float(np.sqrt(forecast.variance.values[-1, 0]) / 100)
        gamma = float(result.params.get("gamma[1]", 0))  # Asymmetry parameter
        return forecast_vol, {"gamma": gamma, "leverage_effect": gamma < 0}

    except Exception:
        vol = float(returns.std()) if len(returns) > 2 else 0.01
        return vol, {}


def realised_volatility(
    df: pd.DataFrame,
    window: int = 20,
    estimator: str = "yang_zhang",
) -> pd.Series:
    """
    Rolling realised volatility using the specified estimator.
    estimator: 'yang_zhang' | 'garman_klass' | 'parkinson' | 'close_to_close'
    """
    def _compute_window(idx):
        chunk = df.iloc[max(0, idx - window) : idx + 1]
        if len(chunk) < 3:
            return np.nan
        o = chunk["open"].values
        h = chunk["high"].values
        l = chunk["low"].values
        c = chunk["close"].values
        if estimator == "yang_zhang":
            return yang_zhang_volatility(o, h, l, c)
        elif estimator == "garman_klass":
            return garman_klass_volatility(o, h, l, c)
        elif estimator == "parkinson":
            return parkinson_volatility(h, l)
        else:
            returns = pd.Series(c).pct_change().dropna()
            return float(returns.std())

    result = pd.Series(
        [_compute_window(i) for i in range(len(df))],
        index=df.index,
    )
    return result


def atr_percentile_rank(atr_series: pd.Series, lookback: int = 252) -> pd.Series:
    """
    ATR normalised as percentile rank over the lookback period.
    0 = historically low vol, 1 = historically high vol.
    """
    def rank_at(idx):
        history = atr_series.iloc[max(0, idx - lookback) : idx + 1]
        if len(history) < 5:
            return 0.5
        return float((history.iloc[-1] > history).mean())

    return pd.Series([rank_at(i) for i in range(len(atr_series))], index=atr_series.index)


def volatility_of_volatility(vol_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Vol-of-vol: rolling standard deviation of the volatility series.
    High vol-of-vol = regime change occurring → reduce size, widen stops.
    """
    return vol_series.rolling(window).std() / (vol_series.rolling(window).mean() + 1e-9)


def detect_volatility_regime(
    current_vol: float,
    vol_mean: float,
    vol_std: float,
) -> str:
    """
    Classify current volatility into regime.
    Returns 'low', 'normal', 'elevated', 'extreme'
    """
    z = (current_vol - vol_mean) / (vol_std + 1e-9)
    if z < -0.5:
        return "low"
    elif z < 0.5:
        return "normal"
    elif z < 1.5:
        return "elevated"
    else:
        return "extreme"


class VolatilitySignalGenerator:
    """
    Generates volatility-based trading signals and regime classification.

    Key outputs:
      - vol_regime: 'low' | 'normal' | 'elevated' | 'extreme'
      - position_size_scalar: multiply position size by this (0 to 1)
      - strategy_mode: 'mean_reversion' | 'momentum' | 'breakout' | 'reduce'
      - breakout_signal: +1/-1 on volatility breakouts
    """

    def __init__(
        self,
        garch_lookback: int = 500,
        rv_window: int = 20,
        atr_period: int = 14,
    ) -> None:
        self.garch_lookback = garch_lookback
        self.rv_window = rv_window
        self.atr_period = atr_period

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all volatility signals.
        Returns dict including metadata and signals.
        """
        if len(df) < 30:
            return {
                "vol_regime": 0.5,
                "position_size_scalar": 0.5,
                "volatility_composite": 0.0,
                "_vol_forecast": 0.02,
            }

        signals = {}
        close = df["close"]
        returns = close.pct_change().dropna()

        # ── GARCH Volatility Forecast ─────────────────────────────────────────
        garch_data = returns.iloc[-min(self.garch_lookback, len(returns)):]
        if len(garch_data) >= 100:
            forecast_vol, current_vol, garch_params = fit_garch_volatility(garch_data)
            signals["_garch_forecast"] = forecast_vol
            signals["_garch_current"] = current_vol
            signals["_garch_persistence"] = garch_params.get("persistence", 0)
        else:
            current_vol = float(returns.std())
            forecast_vol = current_vol
            signals["_garch_forecast"] = forecast_vol
            signals["_garch_current"] = current_vol

        # ── Realised Volatility (Yang-Zhang) ──────────────────────────────────
        if len(df) >= self.rv_window + 3:
            rv = realised_volatility(df, self.rv_window, "yang_zhang")
            rv_last = float(rv.iloc[-1]) if not pd.isna(rv.iloc[-1]) else current_vol
            signals["_realised_vol"] = rv_last

            # Volatility ratio: forecast / realised — tells if vol is expanding
            vol_ratio = forecast_vol / (rv_last + 1e-9)
            signals["_vol_expansion_ratio"] = vol_ratio

            # Rolling vol stats for percentile ranking
            rv_clean = rv.dropna()
            if len(rv_clean) >= 30:
                vol_mean = float(rv_clean.rolling(100).mean().iloc[-1])
                vol_std = float(rv_clean.rolling(100).std().iloc[-1])
                if pd.isna(vol_mean):
                    vol_mean = float(rv_clean.mean())
                    vol_std = float(rv_clean.std())
            else:
                vol_mean = float(rv_clean.mean())
                vol_std = float(rv_clean.std())
        else:
            rv_last = current_vol
            vol_mean = current_vol
            vol_std = current_vol * 0.3

        signals["_vol_mean"] = vol_mean
        signals["_vol_std"] = vol_std

        # ── ATR-based normalised volatility ──────────────────────────────────
        from strategy.signals.technical import compute_atr
        atr = compute_atr(df["high"], df["low"], df["close"], self.atr_period)
        atr_last = float(atr.iloc[-1])
        signals["_atr"] = atr_last
        atr_pct_of_price = atr_last / float(close.iloc[-1])
        signals["_atr_pct"] = atr_pct_of_price

        # ── Regime Classification ─────────────────────────────────────────────
        regime = detect_volatility_regime(rv_last, vol_mean, vol_std)
        regime_score = {"low": 0.0, "normal": 0.33, "elevated": 0.67, "extreme": 1.0}[regime]
        signals["vol_regime"] = regime_score
        signals["_vol_regime_str"] = regime

        # ── Position Size Scalar ──────────────────────────────────────────────
        # Risk parity: size inversely proportional to vol
        # At 1x normal vol → scalar = 1.0, at 2x → 0.5, at 0.5x → 1.5 (capped)
        if vol_mean > 0:
            vol_normalized = rv_last / vol_mean
        else:
            vol_normalized = 1.0
        size_scalar = float(np.clip(1.0 / vol_normalized, 0.2, 2.0))
        signals["position_size_scalar"] = size_scalar

        # ── Strategy Mode Recommendation ──────────────────────────────────────
        # Low vol → mean reversion works; High vol → momentum/breakout
        if regime == "low":
            signals["strategy_mode"] = -0.5  # Favour mean reversion
        elif regime == "normal":
            signals["strategy_mode"] = 0.0  # Balanced
        elif regime == "elevated":
            signals["strategy_mode"] = 0.5  # Lean momentum/breakout
        else:
            signals["strategy_mode"] = -1.0  # Reduce risk in extreme vol

        # ── Bollinger Band Width (squeeze detection) ──────────────────────────
        bb_period = 20
        bb_mean = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mean
        bb_width_last = float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0.04

        if len(bb_width.dropna()) >= 20:
            bb_width_pct = float((bb_width.iloc[-1] > bb_width.dropna()).mean())
        else:
            bb_width_pct = 0.5
        signals["_bb_width_percentile"] = bb_width_pct

        # Squeeze: width in bottom 20% → imminent breakout
        if bb_width_pct < 0.2:
            signals["squeeze_alert"] = 1.0
        else:
            signals["squeeze_alert"] = 0.0

        # ── Breakout Signal (post-squeeze direction) ──────────────────────────
        if bb_width_pct < 0.3 and len(close) > 5:
            price_pos = float((close.iloc[-1] - bb_mean.iloc[-1]) / (bb_width_last * bb_mean.iloc[-1] + 1e-9))
            signals["breakout_direction"] = float(np.clip(price_pos * 3, -1, 1))
        else:
            signals["breakout_direction"] = 0.0

        # ── Vol-of-Vol ────────────────────────────────────────────────────────
        if len(returns) >= 40:
            vol_series = returns.rolling(10).std()
            vov = float(vol_series.rolling(20).std().iloc[-1] / (vol_series.rolling(20).mean().iloc[-1] + 1e-9))
            signals["_vol_of_vol"] = vov
            # High VoV = regime transition → signal uncertainty → reduce position
            signals["vov_scalar"] = float(np.clip(1.0 - vov * 2, 0.3, 1.0))
        else:
            signals["vov_scalar"] = 1.0

        # ── GARCH-based VaR ───────────────────────────────────────────────────
        from scipy.stats import t as t_dist
        portfolio_val = 1.0  # Placeholder; caller scales
        df_param = 5  # Student-t degrees of freedom
        var_99 = float(t_dist.ppf(0.01, df_param) * forecast_vol * portfolio_val)
        signals["_var_99_pct"] = abs(var_99)

        # ── Composite Volatility Signal ────────────────────────────────────────
        # This signal tells the master strategy HOW to trade, not WHERE
        # Positive = favour momentum strategies; Negative = favour mean reversion
        signals["volatility_composite"] = float(np.clip(signals["strategy_mode"], -1, 1))
        signals["_vol_forecast"] = forecast_vol

        return signals
