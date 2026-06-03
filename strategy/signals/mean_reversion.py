"""
Statistical mean reversion signal generator.

Theory: Prices that deviate significantly from their statistical equilibrium
(as modelled by an Ornstein-Uhlenbeck process) tend to revert. This module
identifies reversion opportunities with edge and statistical significance.

Signals:
  - Ornstein-Uhlenbeck Z-score
  - Bollinger Band reversion
  - RSI-based reversion (extreme reading)
  - Rolling Z-score of log prices
  - Half-life filter (only trade if half-life is in a tradeable range)
  - Hurst Exponent (confirms mean-reverting regime H < 0.5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

from utils.math_utils import estimate_ou_params, ou_z_score, rolling_zscore


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Estimate Hurst Exponent via R/S analysis.
    H < 0.5 → mean-reverting
    H ≈ 0.5 → random walk
    H > 0.5 → trending
    """
    if len(series) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        chunks = [series.iloc[i : i + lag] for i in range(0, len(series) - lag, lag)]
        if not chunks:
            continue
        rs_vals = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            deviation = (chunk - mean).cumsum()
            r = deviation.max() - deviation.min()
            s = chunk.std()
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            tau.append(np.log(np.mean(rs_vals)))

    if len(tau) < 3:
        return 0.5
    log_lags = [np.log(l) for l in range(2, len(tau) + 2)]
    try:
        slope, _, _, _, _ = stats.linregress(log_lags, tau)
        return float(np.clip(slope, 0, 1))
    except Exception:
        return 0.5


def adf_test_pvalue(series: pd.Series) -> float:
    """
    Augmented Dickey-Fuller test for stationarity.
    p-value < 0.05 suggests the series is mean-reverting (stationary).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), maxlags=5, autolag="AIC")
        return float(result[1])
    except Exception:
        return 0.5


class MeanReversionSignalGenerator:
    """
    Detects mean reversion opportunities using statistical process theory.

    Only generates trades when:
      1. Hurst exponent < 0.5 (confirmed mean-reverting)
      2. ADF p-value < 0.1 (statistically significant stationarity)
      3. Price is |Z-score| > z_entry standard deviations from equilibrium
      4. Half-life is within tradeable range [2, 50] bars
    """

    def __init__(
        self,
        lookback: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        half_life_min: float = 2.0,
        half_life_max: float = 50.0,
    ) -> None:
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.half_life_min = half_life_min
        self.half_life_max = half_life_max

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute mean reversion signals.
        Returns dict with keys including 'mr_composite'.
        """
        if len(df) < self.lookback:
            return {"mr_composite": 0.0, "hurst": 0.5, "ou_half_life": np.inf}

        close = df["close"]
        log_price = np.log(close)
        signals = {}

        # ── Ornstein-Uhlenbeck estimation ─────────────────────────────────────
        ou_series = log_price.iloc[-self.lookback :]
        theta, ou_mu, ou_sigma, half_life = estimate_ou_params(ou_series)
        signals["ou_theta"] = theta
        signals["ou_mu"] = ou_mu
        signals["ou_sigma"] = ou_sigma
        signals["ou_half_life"] = half_life

        # Only generate signal if half-life is tradeable
        tradeable_halflife = self.half_life_min <= half_life <= self.half_life_max
        signals["_tradeable_ou"] = 1.0 if tradeable_halflife else 0.0

        if tradeable_halflife and ou_sigma > 0:
            z = ou_z_score(float(log_price.iloc[-1]), ou_mu, ou_sigma)
            signals["ou_zscore"] = z
            # Mean reversion: negative z → buy, positive z → sell
            # Scale: clip at 3σ, maps to [-1, +1]
            signals["ou_signal"] = float(np.clip(-z / self.z_entry, -1, 1))
        else:
            signals["ou_zscore"] = 0.0
            signals["ou_signal"] = 0.0

        # ── Rolling Z-score (simpler alternative) ─────────────────────────────
        zscore_series = rolling_zscore(log_price, self.lookback)
        z_last = zscore_series.iloc[-1] if not pd.isna(zscore_series.iloc[-1]) else 0.0
        signals["rolling_zscore"] = float(z_last)
        signals["rolling_zscore_signal"] = float(np.clip(-z_last / self.z_entry, -1, 1))

        # ── Hurst Exponent ────────────────────────────────────────────────────
        if len(close) >= 40:
            hurst = hurst_exponent(close.iloc[-self.lookback :], max_lag=min(20, self.lookback // 3))
            signals["hurst"] = hurst
            # H < 0.5 = mean reverting → amplify signal
            # H > 0.5 = trending → reduce signal
            hurst_scale = max(0.0, (0.5 - hurst) / 0.5 + 0.5)  # [0.5, 1.0] when H ∈ [0, 0.5]
        else:
            signals["hurst"] = 0.5
            hurst_scale = 0.5

        # ── ADF Stationarity Test ─────────────────────────────────────────────
        if len(close) >= 30:
            adf_pval = adf_test_pvalue(log_price.iloc[-self.lookback :])
            signals["adf_pvalue"] = adf_pval
            adf_confidence = max(0.0, 1.0 - adf_pval / 0.1)  # Full confidence at p=0, zero at p>0.1
        else:
            signals["adf_pvalue"] = 0.5
            adf_confidence = 0.5

        # ── Bollinger Band Reversion ──────────────────────────────────────────
        window = min(20, len(close) - 1)
        bb_mean = close.rolling(window).mean()
        bb_std = close.rolling(window).std()
        bb_z = ((close - bb_mean) / bb_std).iloc[-1]
        if pd.isna(bb_z):
            bb_z = 0.0
        signals["bb_zscore"] = float(bb_z)
        signals["bb_signal"] = float(np.clip(-bb_z / 2.0, -1, 1))

        # ── RSI Extreme Reversion ─────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi_last = rsi.iloc[-1]
        if pd.isna(rsi_last):
            signals["mr_rsi"] = 0.0
        elif rsi_last < 20:
            signals["mr_rsi"] = (20 - rsi_last) / 20  # +1 at RSI=0
        elif rsi_last > 80:
            signals["mr_rsi"] = -(rsi_last - 80) / 20  # -1 at RSI=100
        else:
            signals["mr_rsi"] = 0.0

        # ── Composite Mean Reversion Signal ───────────────────────────────────
        # Weight by Hurst (lower H = stronger mean reversion) and ADF confidence
        regime_weight = hurst_scale * (0.5 + 0.5 * adf_confidence)

        sub_signals = {
            "ou_signal": 1.5,
            "rolling_zscore_signal": 1.0,
            "bb_signal": 1.0,
            "mr_rsi": 0.8,
        }
        total_w = sum(sub_signals.values())
        raw_composite = sum(
            signals.get(k, 0.0) * w for k, w in sub_signals.items()
        ) / total_w

        # Scale by regime suitability
        signals["mr_composite"] = float(np.clip(raw_composite * regime_weight * 2, -1, 1))

        return signals
