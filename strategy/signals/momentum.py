"""
Multi-timeframe momentum signal generator.

Theory: Returns tend to persist over short horizons (price momentum) and
reverse over long horizons (mean reversion). This module captures the
cross-sectional and time-series momentum effects.

Signals:
  - Time-series momentum (TSM) over multiple windows
  - Exponentially weighted momentum
  - Rate of Change (ROC) multi-period
  - Momentum factor Z-score
  - Skip-month momentum (skips most recent bar to avoid microstructure noise)
  - Dual-momentum (absolute + relative)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


def time_series_momentum(returns: pd.Series, lookback: int) -> float:
    """
    Time-series momentum: sign of cumulative return over lookback period.
    Returns in [-1, +1]: +1 if positive momentum, -1 if negative.
    """
    if len(returns) < lookback:
        return 0.0
    cum_return = (1 + returns.iloc[-lookback:]).prod() - 1
    # Scale by magnitude, clip to [-1, +1]
    return float(np.clip(cum_return * 5, -1, 1))


def exponential_momentum(returns: pd.Series, halflife: int) -> float:
    """
    Exponentially weighted momentum — more recent returns weighted higher.
    halflife: number of bars for weight to decay to 50%.
    """
    if len(returns) < halflife:
        return 0.0
    weights = 0.5 ** (np.arange(len(returns)) / halflife)
    weights = weights[::-1]  # Most recent has highest weight
    weights /= weights.sum()
    weighted_return = (returns.values * weights).sum()
    return float(np.clip(weighted_return * 50, -1, 1))


def rate_of_change(prices: pd.Series, period: int) -> float:
    """ROC = (price / price[n]) - 1. Classic momentum measure."""
    if len(prices) <= period:
        return 0.0
    roc = prices.iloc[-1] / prices.iloc[-period] - 1
    return float(np.clip(roc * 5, -1, 1))


def skip_month_momentum(returns: pd.Series, lookback: int = 60, skip: int = 1) -> float:
    """
    Skip-month momentum: cumulative return from bar [-lookback] to bar [-skip].
    Skips most recent bar(s) to avoid short-term reversal bias.
    """
    if len(returns) < lookback + skip:
        return 0.0
    subset = returns.iloc[-(lookback + skip) : -skip] if skip > 0 else returns.iloc[-lookback:]
    cum_return = (1 + subset).prod() - 1
    return float(np.clip(cum_return * 5, -1, 1))


def momentum_zscore(prices: pd.Series, windows: List[int]) -> float:
    """
    Compute momentum signal as Z-score of multi-horizon returns.
    Averages Z-scores across all lookback windows.
    """
    if len(prices) < max(windows) + 1:
        return 0.0
    returns = prices.pct_change()
    zscores = []
    for w in windows:
        if len(returns) < w + 1:
            continue
        cum_ret = (1 + returns.iloc[-w:]).prod() - 1
        hist_returns = [(1 + returns.iloc[max(0, i - w) : i]).prod() - 1 for i in range(w, len(returns), w // 2 or 1)]
        if len(hist_returns) < 5:
            continue
        mu = np.mean(hist_returns)
        sigma = np.std(hist_returns)
        if sigma > 0:
            zscores.append((cum_ret - mu) / sigma)
    if not zscores:
        return 0.0
    avg_z = np.mean(zscores)
    return float(np.clip(avg_z / 2.0, -1, 1))


def acceleration_momentum(returns: pd.Series, short: int = 5, long: int = 20) -> float:
    """
    Momentum acceleration = short-term mom minus long-term mom.
    Captures whether momentum is increasing or decreasing.
    """
    if len(returns) < long:
        return 0.0
    short_mom = (1 + returns.iloc[-short:]).prod() - 1
    long_mom = (1 + returns.iloc[-long:]).prod() - 1
    accel = short_mom - long_mom
    return float(np.clip(accel * 10, -1, 1))


def momentum_consistency(returns: pd.Series, window: int = 20) -> float:
    """
    Fraction of positive return bars over the window.
    High consistency (>0.7) = strong momentum; low (<0.3) = reversal candidate.
    Signal: maps [0,1] → [-1,+1] centred at 0.5.
    """
    if len(returns) < window:
        return 0.0
    recent = returns.iloc[-window:]
    consistency = (recent > 0).sum() / window
    return float((consistency - 0.5) * 2)


class MomentumSignalGenerator:
    """
    Multi-timeframe momentum signals combining multiple lookback windows.

    Uses the Moskowitz, Ooi, Pedersen (2012) TSMOM framework:
    - Signal = sign(r_{t-12m, t-1m}) with magnitude scaling
    - Extended with multiple timeframes and exponential weighting
    """

    def __init__(self, windows: List[int] = None) -> None:
        # Windows in number of bars (primary timeframe = 15min)
        # 4 bars = 1h, 16 bars = 4h, 32 = 8h, 96 = 1d, 480 = 5d
        self.windows = windows or [4, 8, 16, 32, 48, 96, 192, 480]

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all momentum signals. Returns dict with keys:
        'momentum_{window}', 'momentum_composite', 'momentum_acceleration',
        'momentum_consistency', 'momentum_zscore'
        """
        if len(df) < 2:
            return {"momentum_composite": 0.0}

        close = df["close"]
        returns = close.pct_change().dropna()
        signals = {}

        # ── Per-window time-series momentum ───────────────────────────────────
        window_signals = {}
        for w in self.windows:
            if len(returns) >= w:
                sig = time_series_momentum(returns, w)
                signals[f"momentum_{w}"] = sig
                window_signals[w] = sig

        # ── Exponential momentum (halflife = 1d = 96 bars at 15min) ──────────
        for halflife in [24, 48, 96, 192]:
            if len(returns) >= halflife:
                signals[f"exp_momentum_{halflife}"] = exponential_momentum(returns, halflife)

        # ── Skip-bar momentum (skip 1 bar = 15min microstructure noise) ──────
        for w in [32, 96, 192]:
            if len(returns) >= w + 2:
                signals[f"skip_momentum_{w}"] = skip_month_momentum(returns, w, skip=2)

        # ── Momentum Z-score ──────────────────────────────────────────────────
        available_windows = [w for w in self.windows if len(returns) >= w]
        if available_windows:
            signals["momentum_zscore"] = momentum_zscore(close, available_windows)

        # ── Acceleration ──────────────────────────────────────────────────────
        if len(returns) >= 20:
            signals["momentum_acceleration"] = acceleration_momentum(returns, 5, 20)

        # ── Consistency ───────────────────────────────────────────────────────
        if len(returns) >= 20:
            signals["momentum_consistency"] = momentum_consistency(returns, 20)

        # ── ROC signals ───────────────────────────────────────────────────────
        for w in [4, 16, 96]:
            if len(close) > w:
                signals[f"roc_{w}"] = rate_of_change(close, w)

        # ── Composite: inverse-variance weighted combination ──────────────────
        # More stable windows get higher weight (less variance in IC)
        if window_signals:
            weights = {w: 1.0 / (np.log(w + 1) + 1e-6) for w in window_signals}
            total_w = sum(weights.values())
            composite = sum(window_signals[w] * weights[w] for w in window_signals) / total_w
            signals["momentum_composite"] = float(np.clip(composite, -1, 1))
        else:
            signals["momentum_composite"] = 0.0

        return signals
