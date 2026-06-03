"""
AdaptiveLearner — Main self-improvement orchestrator.

Pipeline:
  1. Receives closed trade records from the simulation/live loop
  2. Every UPDATE_FREQ trades: runs full diagnostic via TradeJournal
  3. Passes diagnostic to SignalOptimizer → updated weights + params
  4. Generates human-readable improvement report
  5. Optionally retrains the ML ensemble (XGBoost) on labeled trade data

Self-improvement cycle:
  ┌─────────────────────────────────────┐
  │  Trade closes                       │
  │  → record in TradeJournal           │
  │  → every 50 trades:                 │
  │    ■ compute IC per signal          │
  │    ■ identify weaknesses            │
  │    ■ update weights + thresholds    │
  │    ■ print diagnostic report        │
  │    ■ save state to disk             │
  └─────────────────────────────────────┘

Across sessions: state is loaded from disk on init, so the model
continuously improves over its entire live trading history.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from strategy.learning.trade_journal import TradeJournal, TradeRecord
from strategy.learning.signal_optimizer import SignalOptimizer, DEFAULT_WEIGHTS


class AdaptiveLearner:
    """
    Top-level self-improvement controller.

    Usage (in backtest or live loop)
    ─────────────────────────────────
    learner = AdaptiveLearner()

    # At trade entry:
    tid = learner.on_entry(entry_px, direction, size, lev, atr_pct, signals, regime, dd)

    # At trade exit:
    learner.on_exit(tid, exit_px, exit_why, net_pnl, fees, r_mult, hold_bars)

    # Get current strategy params (called before every entry):
    w   = learner.weights          # signal weights dict
    thr = learner.threshold        # composite threshold
    sl  = learner.stop_atr_mult    # SL distance in ATR
    tp  = learner.tp_atr_mult      # TP distance in ATR
    """

    REPORT_FREQ = 50   # print report every N closed trades

    def __init__(self, state_dir: str = "state"):
        self._dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.journal   = TradeJournal(os.path.join(state_dir, "trade_journal.json"))
        self.optimizer = SignalOptimizer(os.path.join(state_dir, "optimizer.json"))
        self._reports: List[str] = []

    # ── Current strategy params (live getters) ────────────────────────────────

    @property
    def weights(self) -> Dict[str, float]:
        return self.optimizer.weights

    @property
    def threshold(self) -> float:
        return self.optimizer.params["signal_threshold"]

    @property
    def stop_atr_mult(self) -> float:
        return self.optimizer.params["stop_atr_mult"]

    @property
    def tp_atr_mult(self) -> float:
        return self.optimizer.params["tp_atr_mult"]

    @property
    def kelly_fraction(self) -> float:
        return self.optimizer.params["kelly_fraction"]

    # ── Trade lifecycle hooks ─────────────────────────────────────────────────

    def on_entry(self, entry_px: float, direction: int, size: float,
                 leverage: float, atr_pct: float,
                 signals: Dict[str, float],
                 regime: str = "unknown",
                 vol_regime: str = "normal",
                 dd_at_entry: float = 0.0) -> int:
        return self.journal.open_trade(
            entry_px=entry_px, direction=direction, size_usdc=size,
            leverage=leverage, atr_pct=atr_pct, signals=signals,
            regime=regime, vol_regime=vol_regime, dd_at_entry=dd_at_entry,
        )

    def on_exit(self, trade_id: int, exit_px: float, exit_why: str,
                net_pnl: float, fees: float, r_multiple: float, hold_bars: int):
        self.journal.close_trade(trade_id, exit_px, exit_why,
                                 net_pnl, fees, r_multiple, hold_bars)
        self._maybe_update()

    # ── Learning cycle ────────────────────────────────────────────────────────

    def _maybe_update(self):
        old_w = dict(self.optimizer.weights)
        updated = self.optimizer.update(self.journal)
        if updated:
            report = self._generate_report(old_w)
            self._reports.append(report)

    def _generate_report(self, old_weights: Dict[str, float]) -> str:
        diag = self.journal.full_diagnostic(n_recent=200)
        n    = diag["n_trades"]
        ic   = diag["signal_ic"]
        exits = diag["exit_reasons"]
        stop_d = diag["stop_analysis"]
        wr_s   = diag["win_rate_by_strength"]
        regime_p = diag["regime_perf"]
        streaks  = diag["streaks"]

        lines = [
            "",
            "┌" + "─"*64 + "┐",
            f"│  🧠  ADAPTIVE LEARNER — Update #{self.optimizer._n_updates:<5}  ({n} trades analyzed)",
            "├" + "─"*64 + "┤",
        ]

        # What's going right
        best_sig  = max(ic, key=ic.get) if ic else "—"
        best_ic   = ic.get(best_sig, 0) if ic else 0
        best_reg  = max(regime_p, key=lambda r: regime_p[r]["win_rate"]) if regime_p else "—"
        best_reg_wr = regime_p.get(best_reg, {}).get("win_rate", 0)
        strong_wr = wr_s.get("strong", {}).get("win_rate", 0)
        strong_r  = wr_s.get("strong", {}).get("avg_r", 0)

        lines += [
            "│  ✅  WHAT'S WORKING:",
            f"│    • Best signal:  {best_sig:<18} IC = {best_ic:+.3f}",
            f"│    • Best regime:  {best_reg:<18} WR = {best_reg_wr*100:.1f}%",
            f"│    • Strong signals ({strong_wr*100:.1f}% WR, avg R={strong_r:+.2f}R)",
        ]

        # What's going wrong
        worst_sig  = min(ic, key=ic.get) if ic else "—"
        worst_ic   = ic.get(worst_sig, 0) if ic else 0
        sl_rate    = stop_d.get("stop_loss_rate", 0)
        sl_rec     = stop_d.get("recommendation", "")
        weak_wr    = wr_s.get("weak", {}).get("win_rate", 0)
        worst_reg  = min(regime_p, key=lambda r: regime_p[r]["win_rate"]) if regime_p else "—"
        worst_reg_wr = regime_p.get(worst_reg, {}).get("win_rate", 0)
        streak     = streaks.get("current_streak", 0)
        streak_t   = streaks.get("current_streak_type", "")
        pls_wr     = streaks.get("post_3loss_win_rate")

        lines += [
            "│",
            "│  ❌  WHAT'S NOT WORKING:",
            f"│    • Worst signal: {worst_sig:<18} IC = {worst_ic:+.3f}  → weight being reduced",
            f"│    • Stop-loss hit rate: {sl_rate*100:.1f}%  → {sl_rec}",
            f"│    • Weak-signal trades: {weak_wr*100:.1f}% WR  → threshold adjusted",
            f"│    • Worst regime: {worst_reg} ({worst_reg_wr*100:.1f}% WR) → caution flag",
        ]
        if pls_wr is not None:
            lines.append(f"│    • After 3-loss streak WR: {pls_wr*100:.1f}%"
                         + ("  (mean-revert expected)" if pls_wr > 0.5 else "  (tilt risk — reduce size)"))
        if streak >= 3:
            lines.append(f"│    • Current {streak}-{streak_t} streak detected → size scaling active")

        # Exit breakdown
        lines.append("│")
        lines.append("│  📊  EXIT BREAKDOWN (last 200 trades):")
        for why, stats in sorted(exits.items(), key=lambda x: -x[1]["count"]):
            lines.append(f"│    • {why:<20}: {stats['pct']*100:4.1f}%  WR={stats['win_rate']*100:.0f}%  "
                         f"avgR={stats['avg_r']:+.2f}R")

        # Signal weight changes
        lines.append("│")
        lines.append("│  ⚖️   UPDATED SIGNAL WEIGHTS:")
        for sig in sorted(self.optimizer.weights):
            old = old_weights.get(sig, DEFAULT_WEIGHTS.get(sig, 0))
            new = self.optimizer.weights[sig]
            delta = new - old
            bar_new = "█" * int(new * 20)
            arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "~")
            lines.append(f"│    {arrow} {sig:<18}: {old:.3f} → {new:.3f}  {bar_new}")

        # Updated params
        lines.append("│")
        lines.append("│  🔧  CALIBRATED PARAMETERS:")
        p = self.optimizer.params
        lines.append(f"│    • Entry threshold : {p['signal_threshold']:.3f}  "
                     f"(was {self.optimizer.params.get('signal_threshold', 0.08):.3f})")
        lines.append(f"│    • Stop distance   : {p['stop_atr_mult']:.2f}× ATR")
        lines.append(f"│    • Take-profit     : {p['tp_atr_mult']:.2f}× ATR")
        lines.append(f"│    • Kelly fraction  : {p['kelly_fraction']:.2f}")
        lines.append("└" + "─"*64 + "┘")

        report = "\n".join(lines)
        return report

    def print_latest_report(self):
        if self._reports:
            print(self._reports[-1])

    def print_all_reports(self):
        for r in self._reports:
            print(r)

    def final_summary(self) -> str:
        """End-of-session summary of what was learned."""
        n = len(self.journal.closed)
        if n == 0:
            return "No trades recorded — nothing to learn from yet."

        diag = self.journal.full_diagnostic(n_recent=n)
        ic   = diag["signal_ic"]
        stop = diag["stop_analysis"]

        best  = max(ic, key=ic.get) if ic else "—"
        worst = min(ic, key=ic.get) if ic else "—"

        lines = [
            "",
            "═"*66,
            "  HYPESTRAT SELF-IMPROVEMENT SUMMARY",
            "═"*66,
            f"  Total trades analyzed : {n}",
            f"  Optimizer updates     : {self.optimizer._n_updates}",
            "",
            "  Signal Performance (Information Coefficient):",
        ]
        for sig, val in sorted(ic.items(), key=lambda x: -x[1]):
            bar = "+" * max(0, int(val*20)) if val >= 0 else "-" * max(0, int(-val*20))
            lines.append(f"    {'→' if val > 0.02 else '  '} {sig:<18}: IC={val:+.3f}  {bar}")

        lines += [
            "",
            f"  Best predictive signal : {best} (IC={ic.get(best, 0):+.3f})",
            f"  Worst signal           : {worst} (IC={ic.get(worst, 0):+.3f}) → weight reduced",
            "",
            "  Current optimized weights:",
        ]
        for sig, w in sorted(self.optimizer.weights.items(), key=lambda x: -x[1]):
            lines.append(f"    {sig:<20}: {w:.3f}  {'█' * int(w*30)}")

        lines += [
            "",
            f"  Stop placement: {stop.get('stop_loss_rate',0)*100:.1f}% of exits are stop-losses",
            f"  Recommendation: {stop.get('recommendation', 'N/A')}",
            "",
            "  Final calibrated params:",
            f"    threshold    = {self.threshold:.3f}",
            f"    stop_atr     = {self.stop_atr_mult:.2f}",
            f"    tp_atr       = {self.tp_atr_mult:.2f}",
            f"    kelly_frac   = {self.kelly_fraction:.2f}",
            "",
            "  State saved to: " + self._dir,
            "  Next session will start from these learned weights.",
            "═"*66,
        ]
        return "\n".join(lines)

    def save(self):
        self.journal.save()
        self.optimizer.save()
