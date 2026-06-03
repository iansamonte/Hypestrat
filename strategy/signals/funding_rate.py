"""
Funding rate signal generator for Hyperliquid perpetual swaps.

Funding rate is the mechanism that keeps perpetual futures prices anchored
to the spot price. When funding is very positive, longs pay shorts — crowded
long positioning often precedes a reversal. Negative funding suggests crowded
shorts.

Signals:
  - Current funding rate extreme signal (contrarian)
  - Funding rate trend / acceleration
  - Cumulative funding cost impact on position P&L
  - Basis (premium/discount to implied spot)
  - Open interest changes (OI growing with price = confirmation)
  - Funding rate carry strategy (earn funding by being on the paid side)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


class FundingRateSignalGenerator:
    """
    Generates signals based on Hyperliquid perpetual funding dynamics.

    Key insight: extreme funding rates are mean-reverting because they create
    economic incentive to hold the opposite position, causing natural unwinding.
    """

    def __init__(
        self,
        long_threshold: float = -0.0005,   # 8h rate < this → strong buy signal
        short_threshold: float = 0.0005,    # 8h rate > this → strong sell signal
        extreme_multiplier: float = 2.0,    # Signal strength multiplier at extremes
        lookback_periods: int = 24,         # Number of 8h periods to analyse
    ) -> None:
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.extreme_multiplier = extreme_multiplier
        self.lookback_periods = lookback_periods

    def compute(
        self,
        current_funding: float,
        funding_history: Optional[pd.DataFrame] = None,
        open_interest: float = 0.0,
        price_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Compute all funding rate signals.

        current_funding: current 8h funding rate (e.g. 0.0001 = 0.01%)
        funding_history: DataFrame with ['funding_rate', 'premium', 'open_interest']
        open_interest: current OI in native token
        price_df: OHLCV price data (for basis calculation)
        """
        signals = {}
        signals["_current_funding"] = current_funding

        # ── Current Funding Rate Signal ────────────────────────────────────────
        # Contrarian: very high funding → longs paying a lot → crowded longs → sell
        # Very low (negative) funding → shorts paying → crowded shorts → buy
        if current_funding <= self.long_threshold:
            # Negative funding: magnitude determines strength
            strength = min(abs(current_funding) / abs(self.long_threshold), 2.0)
            signals["funding_current"] = float(min(strength, 1.0))
        elif current_funding >= self.short_threshold:
            strength = min(current_funding / self.short_threshold, 2.0)
            signals["funding_current"] = float(-min(strength, 1.0))
        else:
            # In neutral zone: small proportional signal
            mid = (self.long_threshold + self.short_threshold) / 2
            signals["funding_current"] = float(np.clip(
                -(current_funding - mid) / (self.short_threshold - mid), -0.3, 0.3
            ))

        # ── Annualised Funding Rate ───────────────────────────────────────────
        # 3 funding periods per day × 365 days = annualised
        annualised = current_funding * 3 * 365
        signals["_annualised_funding"] = annualised
        # If annualised > 100% APY, heavily penalise long positions
        signals["_funding_carry_cost"] = annualised

        # ── Historical Funding Analysis ───────────────────────────────────────
        if funding_history is not None and len(funding_history) >= 3:
            hist = funding_history["funding_rate"].dropna()

            # Funding rate Z-score (is current rate extreme?)
            hist_mean = hist.mean()
            hist_std = hist.std()
            if hist_std > 0:
                funding_zscore = (current_funding - hist_mean) / hist_std
                signals["funding_zscore"] = float(funding_zscore)
                # More extreme Z → stronger contrarian signal
                signals["funding_zscore_signal"] = float(np.clip(-funding_zscore / 2.0, -1, 1))
            else:
                signals["funding_zscore"] = 0.0
                signals["funding_zscore_signal"] = 0.0

            # Funding rate trend (is it increasing or decreasing?)
            if len(hist) >= 6:
                recent = hist.iloc[-3:].mean()
                older = hist.iloc[-6:-3].mean()
                trend = recent - older
                signals["funding_trend"] = float(np.clip(-trend / (abs(self.short_threshold) + 1e-9), -1, 1))
            else:
                signals["funding_trend"] = 0.0

            # Cumulative funding over lookback
            recent_hist = hist.iloc[-self.lookback_periods:]
            cum_funding = recent_hist.sum()
            signals["_cum_funding"] = float(cum_funding)
            # Very positive cumulative → longs have paid a lot → likely to unwind
            signals["cum_funding_signal"] = float(np.clip(-cum_funding / 0.01, -1, 1))

            # Funding rate acceleration
            if len(hist) >= 4:
                accel = hist.iloc[-1] - hist.iloc[-4]
                signals["funding_acceleration"] = float(np.clip(-accel / self.short_threshold, -1, 1))
            else:
                signals["funding_acceleration"] = 0.0

        else:
            signals["funding_zscore"] = 0.0
            signals["funding_zscore_signal"] = 0.0
            signals["funding_trend"] = 0.0
            signals["cum_funding_signal"] = 0.0
            signals["funding_acceleration"] = 0.0

        # ── Open Interest Signal ──────────────────────────────────────────────
        if funding_history is not None and "open_interest" in funding_history.columns:
            oi = funding_history["open_interest"].dropna()
            if len(oi) >= 5:
                oi_change = (oi.iloc[-1] - oi.iloc[-5]) / (oi.iloc[-5] + 1e-9)
                signals["_oi_change_pct"] = float(oi_change)

                # OI rising with positive funding = crowded longs → bearish
                # OI rising with negative funding = crowded shorts → bullish
                oi_signal = oi_change * (-np.sign(current_funding))
                signals["oi_signal"] = float(np.clip(oi_signal * 2, -1, 1))
            else:
                signals["oi_signal"] = 0.0
        else:
            signals["oi_signal"] = 0.0

        # ── Basis Signal (premium/discount) ──────────────────────────────────
        if funding_history is not None and "premium" in funding_history.columns:
            premium = funding_history["premium"].dropna()
            if len(premium) > 0:
                current_premium = float(premium.iloc[-1])
                signals["_premium"] = current_premium
                # Large positive premium (perp trading above spot) → sell signal
                signals["basis_signal"] = float(np.clip(-current_premium * 50, -1, 1))
            else:
                signals["basis_signal"] = 0.0
        else:
            signals["basis_signal"] = 0.0

        # ── Funding Carry Strategy Signal ─────────────────────────────────────
        # When |funding| is large enough to justify being on the paid side as a carry
        # Earn funding if expected price move is less than funding income
        if abs(current_funding) > 0.0003:  # > 0.03% per 8h = meaningful carry
            carry_signal = np.sign(current_funding) * min(abs(current_funding) / 0.001, 1.0)
            # Reverse: if positive funding, go short to EARN funding
            signals["funding_carry"] = float(-carry_signal)
        else:
            signals["funding_carry"] = 0.0

        # ── Composite Funding Signal ──────────────────────────────────────────
        weights = {
            "funding_current": 2.0,
            "funding_zscore_signal": 1.5,
            "funding_trend": 0.8,
            "cum_funding_signal": 1.0,
            "funding_acceleration": 0.6,
            "oi_signal": 0.8,
            "basis_signal": 0.5,
            "funding_carry": 0.7,
        }
        total_w = sum(weights.values())
        composite = sum(
            signals.get(k, 0.0) * w for k, w in weights.items()
        ) / total_w
        signals["funding_composite"] = float(np.clip(composite, -1, 1))

        return signals

    def compute_funding_adjusted_pnl(
        self,
        position_size: float,
        is_long: bool,
        entry_time: float,
        current_time: float,
        avg_funding_rate: float,
    ) -> float:
        """
        Compute the funding cost/income for a position held over time.
        Returns positive if funding was earned, negative if paid.
        """
        hours_held = (current_time - entry_time) / 3600
        funding_periods = hours_held / 8
        direction = 1 if is_long else -1
        # If funding positive and long → pay; if funding positive and short → earn
        funding_pnl = -direction * avg_funding_rate * funding_periods * position_size
        return float(funding_pnl)
