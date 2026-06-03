"""
SignalOptimizer — IC-based adaptive weight updates + parameter calibration.

Every UPDATE_FREQ trades (default 50):
  1. Compute rolling IC per signal
  2. Convert IC → weights via softmax (signals with negative IC get near-zero weight)
  3. Calibrate threshold: raise if WR at current threshold < breakeven
  4. Calibrate stop multiplier: widen if >70% of exits are stop_loss
  5. Calibrate take-profit: tighten if TPs are very rare

All calibrated params are persisted and returned to the strategy on demand.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


# Default weights — match run_backtest.py composite weights
DEFAULT_WEIGHTS: Dict[str, float] = {
    "technical":       0.25,
    "momentum":        0.22,
    "mean_reversion":  0.18,
    "orderflow":       0.22,
    "funding":         0.13,
}

DEFAULT_PARAMS = {
    "signal_threshold":  0.08,   # minimum |composite| to enter
    "stop_atr_mult":     2.0,    # SL = entry ± stop_atr_mult * ATR
    "tp_atr_mult":       5.0,    # TP = entry ± tp_atr_mult * ATR
    "trailing_atr_mult": 1.5,    # trailing stop distance
    "kelly_fraction":    0.25,   # quarter-Kelly
    "max_position_pct":  0.30,   # max 30% of equity in single trade
}


class SignalOptimizer:
    """Adaptive weight + parameter optimizer with persistent state."""

    UPDATE_FREQ = 50       # retrain every N closed trades
    WINDOW      = 200      # IC computed on last N trades
    IC_TEMP     = 4.0      # softmax temperature for IC→weight mapping
    MIN_WEIGHT  = 0.02     # floor weight so no signal is fully silenced
    BREAKEVEN_WR = 1/3.5   # breakeven win rate at 2.5:1 RR ratio (approx 0.286)

    def __init__(self, path: str = "state/optimizer.json"):
        self._path = path
        self.weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        self.params:  Dict[str, float] = dict(DEFAULT_PARAMS)
        self._ic_history: List[Dict[str, float]] = []
        self._n_updates = 0
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self.weights     = data.get("weights", dict(DEFAULT_WEIGHTS))
                self.params      = {**DEFAULT_PARAMS, **data.get("params", {})}
                self._ic_history = data.get("ic_history", [])
                self._n_updates  = data.get("n_updates", 0)
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        payload = {
            "weights":    self.weights,
            "params":     self.params,
            "ic_history": self._ic_history[-50:],  # keep last 50 snapshots
            "n_updates":  self._n_updates,
        }
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self._path)

    # ── Weight update ─────────────────────────────────────────────────────────

    def update(self, journal) -> bool:
        """
        Called by AdaptiveLearner after each trade.
        Returns True if weights were actually updated this cycle.
        """
        n_closed = len(journal.closed)
        if n_closed < 20 or n_closed % self.UPDATE_FREQ != 0:
            return False

        diag = journal.full_diagnostic(n_recent=self.WINDOW)
        ic   = diag["signal_ic"]
        self._ic_history.append(ic)

        # ── Compute rolling-average IC (EWMA across update snapshots) ─────────
        if len(self._ic_history) >= 2:
            alpha = 0.3
            avg_ic = {}
            for sig in DEFAULT_WEIGHTS:
                vals = [snap.get(sig, 0.0) for snap in self._ic_history[-10:]]
                avg_ic[sig] = float(np.mean(vals))
        else:
            avg_ic = ic

        # ── IC → weights via shifted softmax ─────────────────────────────────
        # Shift so that IC=0 still gets some weight; only strong negative IC gets penalised
        shifted = {k: max(v + 0.05, 0.0) for k, v in avg_ic.items()
                   if k in DEFAULT_WEIGHTS}
        if sum(shifted.values()) < 1e-9:
            shifted = {k: 1.0 for k in DEFAULT_WEIGHTS}

        exps    = {k: np.exp(v * self.IC_TEMP) for k, v in shifted.items()}
        total_e = sum(exps.values())
        new_w   = {k: max(exps[k] / total_e, self.MIN_WEIGHT) for k in DEFAULT_WEIGHTS}
        # Normalise to sum=1
        s = sum(new_w.values())
        self.weights = {k: v/s for k, v in new_w.items()}

        # ── Parameter calibration ─────────────────────────────────────────────
        self._calibrate_params(diag)

        self._n_updates += 1
        self.save()
        return True

    def _calibrate_params(self, diag: Dict):
        stop_diag = diag.get("stop_analysis", {})
        strength  = diag.get("win_rate_by_strength", {})
        exits     = diag.get("exit_reasons", {})

        # 1. Threshold calibration
        #    If weak-signal trades win < breakeven, raise threshold
        weak_wr = strength.get("weak", {}).get("win_rate", self.BREAKEVEN_WR)
        if weak_wr < self.BREAKEVEN_WR - 0.03:
            self.params["signal_threshold"] = min(
                self.params["signal_threshold"] * 1.10, 0.25)
        elif weak_wr > self.BREAKEVEN_WR + 0.05:
            self.params["signal_threshold"] = max(
                self.params["signal_threshold"] * 0.92, 0.04)

        # 2. Stop distance calibration
        sl_rate = stop_diag.get("stop_loss_rate", 0.0)
        if sl_rate > 0.75:
            # Too many stops hit → widen
            self.params["stop_atr_mult"] = min(
                self.params["stop_atr_mult"] * 1.10, 4.0)
        elif sl_rate < 0.35:
            # Very few stops → may be able to tighten
            self.params["stop_atr_mult"] = max(
                self.params["stop_atr_mult"] * 0.95, 1.0)

        # 3. Take-profit calibration
        tp_rate = stop_diag.get("take_profit_rate", 0.0)
        if tp_rate < 0.05 and sl_rate > 0.65:
            # TP never reached, SL hit often → lower TP target
            self.params["tp_atr_mult"] = max(
                self.params["tp_atr_mult"] * 0.90, 2.0)
        elif tp_rate > 0.40:
            # TP reached frequently → let winners run
            self.params["tp_atr_mult"] = min(
                self.params["tp_atr_mult"] * 1.05, 10.0)

        # 4. Kelly fraction: reduce when in a losing regime
        n  = diag.get("n_trades", 0)
        if n > 20:
            recent_r = [t.r_multiple for t in [].__class__()]  # placeholder
        # (Full Kelly calibration via journal.recent() is done in AdaptiveLearner)

    def describe_changes(self, old_w: Dict[str, float]) -> str:
        lines = ["  Signal weight changes (IC-based):"]
        for sig in sorted(self.weights):
            delta = self.weights[sig] - old_w.get(sig, DEFAULT_WEIGHTS.get(sig, 0))
            arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "~")
            lines.append(f"    {arrow} {sig:<18}: {old_w.get(sig,0):.3f} → {self.weights[sig]:.3f}"
                         f"  (Δ{delta:+.3f})")
        lines.append(f"  Params: threshold={self.params['signal_threshold']:.3f}  "
                     f"SL={self.params['stop_atr_mult']:.1f}×ATR  "
                     f"TP={self.params['tp_atr_mult']:.1f}×ATR")
        return "\n".join(lines)
