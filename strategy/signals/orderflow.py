"""
Order flow signal generator.

Order flow provides the most direct measure of supply/demand imbalance.
Institutional traders leave footprints in the tape that can be detected.

Signals:
  - Cumulative Volume Delta (CVD) — net buying vs selling pressure
  - VWAP deviation with mean-reversion and trend-following modes
  - Order book imbalance (bid depth vs ask depth)
  - Volume profile (Point of Control, Value Area)
  - Aggressive trade ratio (market orders vs limit orders)
  - Absorption detection (large orders with minimal price move)
  - Liquidation cascade signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def estimate_cvd(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Estimate Cumulative Volume Delta from OHLCV.
    Uses a heuristic: if close > open → buying volume, else selling.
    More accurate with tick data, but OHLCV approximation is sufficient.
    """
    delta = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
    cvd = pd.Series(delta, index=df.index).rolling(window).sum()
    return cvd


def tick_rule_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Tick rule CVD: classify each bar as buy or sell based on price direction.
    Up-tick → buy; Down-tick → sell; no change → previous direction.
    """
    direction = np.sign(df["close"].diff().fillna(0))
    # Fill zeros with previous direction (uptick rule)
    direction = direction.replace(0, np.nan).ffill().fillna(1)
    return (direction * df["volume"]).cumsum()


def compute_vwap_bands(
    df: pd.DataFrame,
    window: int = 20,
    std_multiples: List[float] = None,
) -> Dict[str, pd.Series]:
    """
    Rolling VWAP with standard deviation bands.
    Similar to Bollinger Bands but volume-weighted.
    """
    if std_multiples is None:
        std_multiples = [1.0, 2.0, 3.0]
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]
    cum_tpv = (typical * vol).rolling(window).sum()
    cum_vol = vol.rolling(window).sum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)

    # VWAP standard deviation (volume-weighted)
    deviation_sq = ((typical - vwap) ** 2 * vol).rolling(window).sum()
    vwap_std = np.sqrt(deviation_sq / cum_vol.replace(0, np.nan))

    result = {"vwap": vwap, "vwap_std": vwap_std}
    for m in std_multiples:
        result[f"vwap_upper_{m}"] = vwap + m * vwap_std
        result[f"vwap_lower_{m}"] = vwap - m * vwap_std
    return result


def compute_book_imbalance(bids: List[List[float]], asks: List[List[float]], depth: int = 5) -> float:
    """
    Order book imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume).
    Range: [-1, +1]. +1 = all bids (buy pressure), -1 = all asks (sell pressure).
    """
    bid_vol = sum(b[1] for b in bids[:depth]) if bids else 0
    ask_vol = sum(a[1] for a in asks[:depth]) if asks else 0
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def compute_volume_profile(
    df: pd.DataFrame,
    n_bins: int = 50,
) -> Tuple[float, float, float]:
    """
    Volume Profile: Point of Control (POC), Value Area High, Value Area Low.
    POC = price level with highest traded volume.
    Value Area = price range containing ~70% of volume.
    Returns (poc_price, vah, val)
    """
    if df.empty:
        close = df["close"].iloc[-1] if not df.empty else 0.0
        return close, close, close

    price_min = df["low"].min()
    price_max = df["high"].max()
    if price_min >= price_max:
        poc = df["close"].iloc[-1]
        return poc, poc, poc

    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vol_profile = np.zeros(n_bins)

    for _, row in df.iterrows():
        mask = (bin_centers >= row["low"]) & (bin_centers <= row["high"])
        if mask.any():
            vol_profile[mask] += row["volume"] / mask.sum()

    poc_idx = np.argmax(vol_profile)
    poc = bin_centers[poc_idx]

    # Value Area: expand from POC until 70% of total volume is covered
    total_vol = vol_profile.sum()
    target_vol = 0.70 * total_vol
    covered = vol_profile[poc_idx]
    low_idx = poc_idx
    high_idx = poc_idx

    while covered < target_vol and (low_idx > 0 or high_idx < n_bins - 1):
        add_low = vol_profile[low_idx - 1] if low_idx > 0 else 0
        add_high = vol_profile[high_idx + 1] if high_idx < n_bins - 1 else 0
        if add_high >= add_low:
            high_idx = min(high_idx + 1, n_bins - 1)
            covered += add_high
        else:
            low_idx = max(low_idx - 1, 0)
            covered += add_low

    return float(poc), float(bin_centers[high_idx]), float(bin_centers[low_idx])


def detect_absorption(
    df: pd.DataFrame,
    volume_threshold_std: float = 2.0,
    price_move_threshold: float = 0.002,
) -> pd.Series:
    """
    Absorption: high volume bar with tiny price move.
    Suggests large orders absorbing supply/demand — often precedes reversal.
    Returns series of +1 (buy absorption), -1 (sell absorption), 0 (none).
    """
    volume_mean = df["volume"].rolling(20).mean()
    volume_std = df["volume"].rolling(20).std()
    high_vol = df["volume"] > volume_mean + volume_threshold_std * volume_std

    price_move = (df["close"] - df["open"]).abs() / df["open"]
    small_move = price_move < price_move_threshold

    bar_direction = np.sign(df["close"] - df["open"])

    absorption = pd.Series(0, index=df.index)
    # High vol + small move + closing near bottom = selling absorbed → buy
    near_low = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-12) < 0.3
    near_high = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-12) > 0.7

    absorption[high_vol & small_move & near_low] = 1   # Buy absorption
    absorption[high_vol & small_move & near_high] = -1  # Sell absorption

    return absorption


class OrderFlowSignalGenerator:
    """
    Generates trading signals from order flow analysis.
    Primary alpha: CVD momentum + book imbalance + volume profile deviation.
    """

    def __init__(
        self,
        cvd_window: int = 50,
        vwap_window: int = 20,
        book_depth: int = 10,
    ) -> None:
        self.cvd_window = cvd_window
        self.vwap_window = vwap_window
        self.book_depth = book_depth

    def compute(
        self,
        df: pd.DataFrame,
        bids: Optional[List[List[float]]] = None,
        asks: Optional[List[List[float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute all order flow signals.
        df: OHLCV DataFrame
        bids/asks: current L2 book [[price, size], ...]
        """
        if len(df) < 20:
            return {"orderflow_composite": 0.0}

        signals = {}
        close = df["close"]

        # ── Cumulative Volume Delta ───────────────────────────────────────────
        cvd = estimate_cvd(df, self.cvd_window)
        cvd_last = cvd.iloc[-1]
        cvd_prev = cvd.iloc[-2] if len(cvd) > 1 else cvd_last

        # Normalise CVD by total volume
        total_vol = df["volume"].rolling(self.cvd_window).sum().iloc[-1]
        if total_vol > 0:
            normalised_cvd = cvd_last / total_vol
            signals["cvd"] = float(np.clip(normalised_cvd * 2, -1, 1))
        else:
            signals["cvd"] = 0.0

        # CVD momentum (rate of change)
        cvd_series = estimate_cvd(df, self.cvd_window)
        if len(cvd_series) >= 5:
            cvd_roc = (cvd_series.iloc[-1] - cvd_series.iloc[-5]) / (total_vol + 1e-9)
            signals["cvd_momentum"] = float(np.clip(cvd_roc * 5, -1, 1))
        else:
            signals["cvd_momentum"] = 0.0

        # ── VWAP Signals ──────────────────────────────────────────────────────
        vwap_data = compute_vwap_bands(df, self.vwap_window)
        vwap = vwap_data["vwap"].iloc[-1]
        vwap_std = vwap_data["vwap_std"].iloc[-1]
        current_price = float(close.iloc[-1])

        if not pd.isna(vwap) and not pd.isna(vwap_std) and vwap > 0 and vwap_std > 0:
            vwap_deviation = (current_price - vwap) / vwap_std
            # Mean-reversion from VWAP extremes
            signals["vwap_mr"] = float(np.clip(-vwap_deviation / 2.0, -1, 1))
            # Trend-following: if price clearly above VWAP → long bias
            signals["vwap_trend"] = float(np.clip(vwap_deviation / 2.0, -1, 1))
        else:
            signals["vwap_mr"] = 0.0
            signals["vwap_trend"] = 0.0

        signals["_vwap"] = float(vwap) if not pd.isna(vwap) else current_price

        # ── Order Book Imbalance ──────────────────────────────────────────────
        if bids and asks:
            imbalance = compute_book_imbalance(bids, asks, self.book_depth)
            signals["book_imbalance"] = float(np.clip(imbalance * 1.5, -1, 1))
            # Spread normalised
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid = (best_bid + best_ask) / 2
                spread_bps = (best_ask - best_bid) / mid * 10000
                signals["_spread_bps"] = spread_bps
        else:
            signals["book_imbalance"] = 0.0

        # ── Volume Profile ────────────────────────────────────────────────────
        if len(df) >= 50:
            poc, vah, val = compute_volume_profile(df.iloc[-96:], n_bins=30)  # Last 1 day
            signals["_poc"] = poc
            signals["_vah"] = vah
            signals["_val"] = val

            # Price vs value area: below val = buy, above vah = sell
            if val < current_price < vah:
                signals["volume_profile"] = 0.0  # Inside value area = neutral
            elif current_price < val:
                dist = (val - current_price) / (vah - val + 1e-9)
                signals["volume_profile"] = float(np.clip(dist, 0, 1))
            else:
                dist = (current_price - vah) / (vah - val + 1e-9)
                signals["volume_profile"] = float(np.clip(-dist, -1, 0))
        else:
            signals["volume_profile"] = 0.0

        # ── Absorption Signal ─────────────────────────────────────────────────
        absorption = detect_absorption(df)
        recent_absorption = absorption.iloc[-5:].values
        if (recent_absorption == 1).any():
            signals["absorption"] = 0.5  # Buy absorption seen recently
        elif (recent_absorption == -1).any():
            signals["absorption"] = -0.5  # Sell absorption
        else:
            signals["absorption"] = 0.0

        # ── Volume Ratio (relative volume) ────────────────────────────────────
        current_vol = df["volume"].iloc[-1]
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = current_vol / (avg_vol + 1e-9)
        signals["volume_ratio"] = float(vol_ratio)  # metadata

        # ── Composite Order Flow Signal ───────────────────────────────────────
        weights = {
            "cvd": 1.5,
            "cvd_momentum": 1.0,
            "book_imbalance": 1.2,
            "volume_profile": 0.8,
            "absorption": 0.6,
        }
        # Use trend-following or mean-reverting VWAP based on CVD
        if signals.get("cvd", 0) > 0.2:
            weights["vwap_trend"] = 0.7
        else:
            weights["vwap_mr"] = 0.7

        total_w = sum(weights.values())
        composite = sum(
            signals.get(k, 0.0) * w for k, w in weights.items()
        ) / total_w

        # Volume amplification: high volume confirms the signal
        if vol_ratio > 1.5:
            composite = np.clip(composite * 1.2, -1, 1)

        signals["orderflow_composite"] = float(np.clip(composite, -1, 1))

        return signals
