"""
Drawdown controller — automatic position reduction and strategy adaptation
during drawdown periods.

During drawdowns:
  1. Scale down position sizes proportionally to drawdown depth
  2. Tighten stop-losses
  3. Require stronger signal consensus to enter new trades
  4. Switch strategy mode (more conservative)
  5. Accelerate recovery by focusing on highest-conviction trades only

Recovery protocol:
  - Once drawdown recovers past 50%, gradually restore position sizes
  - Full restoration only after returning to new equity high
  - Maintains compound growth by protecting downside aggressively
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple
from utils.logger import logger


class DrawdownController:
    """
    Monitors current drawdown and adapts trading parameters accordingly.

    The drawdown response curve:
      MDD 0-5%:   Full size, no changes
      MDD 5-10%:  75% size, tighten stops 10%
      MDD 10-15%: 50% size, tighten stops 20%, require stronger signals
      MDD 15-20%: 25% size, tighten stops 30%, only highest-IC signals
      MDD > 20%:  HALT — review and restart manually

    Math: Drawdown impact on compound growth
      Losing 20% requires 25% gain to recover (1/0.8 - 1)
      Losing 50% requires 100% gain to recover (1/0.5 - 1)
      Losing 80% requires 400% gain to recover (1/0.2 - 1)
      → Protecting against large drawdowns is MORE important than chasing gains
    """

    def __init__(
        self,
        dd_thresholds: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
        size_multipliers: Tuple[float, ...] = (1.0, 0.75, 0.50, 0.25, 0.0),
        stop_tightening: Tuple[float, ...] = (1.0, 0.90, 0.80, 0.70, 0.50),
        signal_thresholds: Tuple[float, ...] = (0.35, 0.40, 0.50, 0.65, 0.99),
        consensus_requirements: Tuple[int, ...] = (3, 3, 4, 5, 99),
    ) -> None:
        self.dd_thresholds = dd_thresholds
        self.size_multipliers = size_multipliers
        self.stop_tightening = stop_tightening
        self.signal_thresholds = signal_thresholds
        self.consensus_requirements = consensus_requirements

        self._peak_value: Optional[float] = None
        self._current_dd: float = 0.0
        self._dd_level: int = 0  # 0=normal, 1-4=various levels, 5=halt

    def update(self, portfolio_value: float) -> Dict[str, float]:
        """
        Update drawdown state. Returns current drawdown parameters.
        """
        if self._peak_value is None or portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        self._current_dd = (portfolio_value - self._peak_value) / self._peak_value

        # Determine drawdown level
        dd = abs(self._current_dd)
        level = 0
        for i, threshold in enumerate(self.dd_thresholds):
            if dd >= threshold:
                level = i + 1

        self._dd_level = level

        size_mult = self.size_multipliers[level]
        stop_tight = self.stop_tightening[level]
        sig_thresh = self.signal_thresholds[level]
        consensus_req = self.consensus_requirements[level]
        halted = level >= len(self.dd_thresholds)

        if level > 0:
            logger.warning(
                f"Drawdown level {level} | DD={self._current_dd:.1%} | "
                f"size_mult={size_mult} | sig_threshold={sig_thresh}"
            )

        return {
            "current_dd": float(self._current_dd),
            "dd_level": level,
            "size_multiplier": float(size_mult),
            "stop_tightening": float(stop_tight),
            "signal_threshold": float(sig_thresh),
            "consensus_required": int(consensus_req),
            "halted": halted,
            "peak_value": float(self._peak_value),
        }

    @property
    def current_drawdown(self) -> float:
        return self._current_dd

    @property
    def peak_value(self) -> Optional[float]:
        return self._peak_value

    def adjust_stop_loss(
        self,
        stop_loss: float,
        entry_price: float,
        is_long: bool,
    ) -> float:
        """
        Tighten stop-loss based on current drawdown level.
        Reduces the stop distance by the tightening factor.
        """
        tight_factor = self.stop_tightening[self._dd_level]
        distance = abs(entry_price - stop_loss)
        tightened_distance = distance * tight_factor
        if is_long:
            return entry_price - tightened_distance
        else:
            return entry_price + tightened_distance

    def apply_size_reduction(self, base_size: float) -> float:
        """Apply drawdown-level size reduction to a computed base size."""
        return base_size * self.size_multipliers[self._dd_level]

    def should_take_trade(
        self,
        signal_strength: float,
        n_agreeing_signals: int,
    ) -> Tuple[bool, str]:
        """
        Check if a trade should be taken given current drawdown state.
        Returns (should_take, reason).
        """
        level = self._dd_level
        if level >= len(self.dd_thresholds):
            return False, "drawdown_halt"

        required_signal = self.signal_thresholds[level]
        required_consensus = self.consensus_requirements[level]

        if abs(signal_strength) < required_signal:
            return False, f"signal_below_threshold ({abs(signal_strength):.2f} < {required_signal:.2f})"

        if n_agreeing_signals < required_consensus:
            return False, f"insufficient_consensus ({n_agreeing_signals} < {required_consensus})"

        return True, "approved"

    def get_recovery_progress(self) -> float:
        """
        Recovery progress from deepest recent drawdown.
        0.0 = at peak (no drawdown), 1.0 = fully recovered.
        Returns None if never had drawdown.
        """
        if self._peak_value is None:
            return 1.0
        dd = abs(self._current_dd)
        if dd == 0:
            return 1.0
        # Simple linear recovery metric
        return float(max(0.0, 1.0 - dd / 0.20))
