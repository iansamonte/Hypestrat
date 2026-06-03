"""
Technical analysis signal generator.

Signals (all normalised to [-1, +1]):
  - RSI divergence / overbought-oversold
  - MACD histogram crossover
  - Bollinger Band squeeze / breakout
  - EMA triple crossover (9/21/55)
  - ADX trend strength filter
  - Supertrend direction
  - Stochastic %K/%D
  - Williams %R
  - CCI (Commodity Channel Index)
  - DMI (Directional Movement Index)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """RMA = Wilder's smoothing (used in RSI, ATR)."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-12)
    bandwidth = (upper - lower) / mid
    return upper, mid, lower, pct_b, bandwidth


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    atr = compute_atr(high, low, close, period)
    plus_di = 100 * _rma(plus_dm, period) / atr
    minus_di = 100 * _rma(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = _rma(dx, period)
    return adx, plus_di, minus_di


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    atr = compute_atr(high, low, close, atr_period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        prev_upper = upper_band.iloc[i - 1] if not np.isnan(upper_band.iloc[i - 1]) else upper_band.iloc[i]
        prev_lower = lower_band.iloc[i - 1] if not np.isnan(lower_band.iloc[i - 1]) else lower_band.iloc[i]

        upper_band.iloc[i] = min(upper_band.iloc[i], prev_upper) if close.iloc[i - 1] <= prev_upper else upper_band.iloc[i]
        lower_band.iloc[i] = max(lower_band.iloc[i], prev_lower) if close.iloc[i - 1] >= prev_lower else lower_band.iloc[i]

        prev_st = supertrend.iloc[i - 1]
        if np.isnan(prev_st) or prev_st == prev_upper:
            if close.iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
        else:
            if close.iloc[i] >= lower_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

    return supertrend, direction


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k, d


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-12)


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    typical = (high + low + close) / 3
    mean_dev = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical - typical.rolling(period).mean()) / (0.015 * mean_dev + 1e-12)


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    typical = (high + low + close) / 3
    cum_vol = volume.rolling(period).sum()
    cum_tpv = (typical * volume).rolling(period).sum()
    return cum_tpv / cum_vol.replace(0, np.nan)


class TechnicalSignalGenerator:
    """
    Computes all technical signals and combines into a composite score.
    Each sub-signal is normalised to [-1, +1] before combination.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_ob: float = 70,
        rsi_os: float = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_periods: Tuple[int, int, int] = (9, 21, 55),
        atr_period: int = 14,
        adx_period: int = 14,
    ) -> None:
        self.rsi_period = rsi_period
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_short, self.ema_medium, self.ema_long = ema_periods
        self.atr_period = atr_period
        self.adx_period = adx_period

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all technical signals on the given OHLCV DataFrame.
        Returns dict of signal_name → float in [-1, +1].
        The final 'technical_composite' is the equal-weighted mean.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(1.0, index=df.index))

        signals = {}

        # ── RSI Signal ────────────────────────────────────────────────────────
        rsi = compute_rsi(close, self.rsi_period)
        rsi_last = rsi.iloc[-1]
        if pd.isna(rsi_last):
            signals["rsi"] = 0.0
        else:
            # -1 when overbought (sell), +1 when oversold (buy)
            # Scale: [0,30] → [+1,0], [70,100] → [0,-1]
            if rsi_last <= 50:
                signals["rsi"] = (50 - rsi_last) / 50  # 0 at 50, +1 at 0
            else:
                signals["rsi"] = -(rsi_last - 50) / 50  # 0 at 50, -1 at 100
            # Amplify signal near extremes
            signals["rsi"] = np.clip(signals["rsi"] * 1.5, -1, 1)

        # ── MACD Signal ───────────────────────────────────────────────────────
        macd_line, macd_sig, macd_hist = compute_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal_period
        )
        hist_last = macd_hist.iloc[-1]
        hist_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
        # Signal based on histogram sign change (momentum of momentum)
        if pd.isna(hist_last):
            signals["macd"] = 0.0
        else:
            # Normalise histogram by close price scale
            normalised_hist = hist_last / (close.iloc[-1] * 0.01 + 1e-12)
            signals["macd"] = float(np.clip(normalised_hist * 5, -1, 1))
            # Bonus if crossover happened this bar
            if hist_prev < 0 < hist_last:
                signals["macd"] = min(signals["macd"] + 0.3, 1.0)
            elif hist_prev > 0 > hist_last:
                signals["macd"] = max(signals["macd"] - 0.3, -1.0)

        # ── Bollinger Band Signal ─────────────────────────────────────────────
        upper, mid, lower, pct_b, bandwidth = compute_bollinger(close, self.bb_period, self.bb_std)
        pct_b_last = pct_b.iloc[-1]
        if pd.isna(pct_b_last):
            signals["bollinger"] = 0.0
        else:
            # pct_b: 0 = at lower band (buy), 1 = at upper band (sell)
            signals["bollinger"] = float(np.clip((0.5 - pct_b_last) * 2, -1, 1))

        # ── EMA Triple Crossover ──────────────────────────────────────────────
        ema_s = _ema(close, self.ema_short).iloc[-1]
        ema_m = _ema(close, self.ema_medium).iloc[-1]
        ema_l = _ema(close, self.ema_long).iloc[-1]
        if pd.isna(ema_s) or pd.isna(ema_l):
            signals["ema_cross"] = 0.0
        else:
            # +1 if fully aligned bullish (short > medium > long)
            # -1 if fully aligned bearish
            bull_score = 0.0
            if ema_s > ema_m:
                bull_score += 0.5
            if ema_m > ema_l:
                bull_score += 0.5
            signals["ema_cross"] = bull_score * 2 - 1  # maps [0,1] → [-1,+1]

        # ── ADX Trend Strength + Direction ───────────────────────────────────
        adx, plus_di, minus_di = compute_adx(high, low, close, self.adx_period)
        adx_last = adx.iloc[-1]
        plus_di_last = plus_di.iloc[-1]
        minus_di_last = minus_di.iloc[-1]
        if pd.isna(adx_last):
            signals["adx"] = 0.0
        else:
            strength = min(adx_last / 50.0, 1.0)  # 0 at no trend, 1 at strong trend
            direction = 1 if plus_di_last > minus_di_last else -1
            signals["adx"] = direction * strength

        # ── Supertrend ────────────────────────────────────────────────────────
        _, st_direction = compute_supertrend(high, low, close)
        signals["supertrend"] = float(st_direction.iloc[-1])  # +1 or -1

        # ── Stochastic ────────────────────────────────────────────────────────
        stoch_k, stoch_d = compute_stochastic(high, low, close)
        k_last = stoch_k.iloc[-1]
        if pd.isna(k_last):
            signals["stochastic"] = 0.0
        else:
            signals["stochastic"] = float(np.clip((50 - k_last) / 50 * 1.5, -1, 1))

        # ── Williams %R ───────────────────────────────────────────────────────
        wr = compute_williams_r(high, low, close)
        wr_last = wr.iloc[-1]
        if pd.isna(wr_last):
            signals["williams_r"] = 0.0
        else:
            # -100 = oversold (buy), 0 = overbought (sell)
            signals["williams_r"] = float(np.clip(-(wr_last + 50) / 50 * 1.5, -1, 1))

        # ── VWAP Deviation ────────────────────────────────────────────────────
        vwap = compute_vwap(high, low, close, volume)
        vwap_last = vwap.iloc[-1]
        close_last = close.iloc[-1]
        if pd.isna(vwap_last) or vwap_last == 0:
            signals["vwap_dev"] = 0.0
        else:
            dev = (close_last - vwap_last) / vwap_last
            # Mean-reverting: price above VWAP → short bias
            signals["vwap_dev"] = float(np.clip(-dev * 50, -1, 1))

        # ── CCI ───────────────────────────────────────────────────────────────
        cci = compute_cci(high, low, close)
        cci_last = cci.iloc[-1]
        if pd.isna(cci_last):
            signals["cci"] = 0.0
        else:
            signals["cci"] = float(np.clip(-cci_last / 200, -1, 1))

        # ── ATR (used externally for position sizing, not a directional signal) ─
        atr = compute_atr(high, low, close, self.atr_period)
        signals["_atr"] = float(atr.iloc[-1])  # Prefixed _ = metadata, not signal

        # ── Composite Technical Score ─────────────────────────────────────────
        directional_signals = {k: v for k, v in signals.items() if not k.startswith("_")}
        weights = {
            "rsi": 1.0,
            "macd": 1.2,
            "bollinger": 0.8,
            "ema_cross": 1.5,
            "adx": 1.3,
            "supertrend": 1.4,
            "stochastic": 0.8,
            "williams_r": 0.7,
            "vwap_dev": 0.9,
            "cci": 0.7,
        }
        total_w = sum(weights.values())
        composite = sum(directional_signals[k] * weights.get(k, 1.0) for k in directional_signals if k in weights)
        signals["technical_composite"] = float(np.clip(composite / total_w, -1, 1))

        return signals
