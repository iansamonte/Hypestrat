"""
TradeJournal — Records every trade with full signal + market context.

Each entry contains:
  - Signal values at entry (per signal component)
  - Composite score and direction
  - ATR, vol regime, drawdown at entry
  - Exit reason, PnL, R-multiple, hold time
  - Regime label if available

Analysis methods compute:
  - Per-signal Information Coefficient (Spearman rank corr with outcome)
  - Win/loss rates by signal strength bucket
  - Exit reason distribution
  - Regime-conditional stats
  - Sequential / streak analysis
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TradeRecord:
    # Identification
    trade_id: int
    timestamp: float  # unix epoch at entry

    # Entry context
    entry_px: float
    direction: int          # +1 long, -1 short
    size_usdc: float
    leverage: float
    atr_pct: float          # ATR / price at entry

    # Signal snapshot at entry (each in [-1, +1])
    sig_technical: float = 0.0
    sig_momentum: float = 0.0
    sig_mean_reversion: float = 0.0
    sig_orderflow: float = 0.0
    sig_funding: float = 0.0
    sig_composite: float = 0.0

    # Market regime at entry
    regime: str = "unknown"        # bull / bear / sideways
    vol_regime: str = "normal"     # low / normal / elevated / extreme
    drawdown_at_entry: float = 0.0 # portfolio DD fraction

    # Exit result
    exit_px: float = 0.0
    exit_why: str = ""             # stop_loss / take_profit / signal_flip / end_of_data / dd_halt
    gross_pct: float = 0.0        # gross return fraction
    net_pnl: float = 0.0          # net dollar PnL
    fees: float = 0.0
    r_multiple: float = 0.0        # net PnL / risk per trade
    hold_bars: int = 0

    # Derived (filled by journal after record_exit)
    won: bool = False


class TradeJournal:
    """Persistent trade log with self-analysis capabilities."""

    SIGNAL_COLS = ["sig_technical", "sig_momentum", "sig_mean_reversion",
                   "sig_orderflow", "sig_funding", "sig_composite"]
    SIGNAL_NAMES = ["technical", "momentum", "mean_reversion",
                    "orderflow", "funding", "composite"]

    def __init__(self, path: str = "state/trade_journal.json"):
        self._path = path
        self._trades: List[TradeRecord] = []
        self._id_counter = 0
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._trades = [TradeRecord(**d) for d in data.get("trades", [])]
                self._id_counter = data.get("id_counter", 0)
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

        def _serial(obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        payload = {
            "id_counter": self._id_counter,
            "trades": [asdict(t) for t in self._trades],
        }
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=_serial)
        os.replace(tmp, self._path)

    # ── Recording ─────────────────────────────────────────────────────────────

    def open_trade(self, entry_px: float, direction: int, size_usdc: float,
                   leverage: float, atr_pct: float,
                   signals: Dict[str, float],
                   regime: str = "unknown", vol_regime: str = "normal",
                   dd_at_entry: float = 0.0) -> int:
        """Record a new trade at entry. Returns trade_id."""
        self._id_counter += 1
        rec = TradeRecord(
            trade_id=self._id_counter,
            timestamp=time.time(),
            entry_px=entry_px,
            direction=direction,
            size_usdc=size_usdc,
            leverage=leverage,
            atr_pct=atr_pct,
            sig_technical=float(signals.get("technical", 0)),
            sig_momentum=float(signals.get("momentum", 0)),
            sig_mean_reversion=float(signals.get("mean_reversion", 0)),
            sig_orderflow=float(signals.get("orderflow", 0)),
            sig_funding=float(signals.get("funding", 0)),
            sig_composite=float(signals.get("composite", 0)),
            regime=regime,
            vol_regime=vol_regime,
            drawdown_at_entry=dd_at_entry,
        )
        self._trades.append(rec)
        return self._id_counter

    def close_trade(self, trade_id: int, exit_px: float, exit_why: str,
                    net_pnl: float, fees: float, r_multiple: float,
                    hold_bars: int):
        """Fill in exit data for an open trade."""
        rec = self._find(trade_id)
        if rec is None:
            return
        rec.exit_px = exit_px
        rec.exit_why = exit_why
        rec.net_pnl = net_pnl
        rec.fees = fees
        rec.r_multiple = r_multiple
        rec.hold_bars = hold_bars
        rec.won = net_pnl > 0
        if rec.entry_px > 0:
            rec.gross_pct = rec.direction * (exit_px - rec.entry_px) / rec.entry_px

    def _find(self, trade_id: int) -> Optional[TradeRecord]:
        for t in reversed(self._trades):
            if t.trade_id == trade_id:
                return t
        return None

    # ── Closed-trade accessors ────────────────────────────────────────────────

    @property
    def closed(self) -> List[TradeRecord]:
        return [t for t in self._trades if t.exit_why]

    def recent(self, n: int = 100) -> List[TradeRecord]:
        return self.closed[-n:]

    # ── Analysis ─────────────────────────────────────────────────────────────

    def signal_ic(self, n_recent: int = 200) -> Dict[str, float]:
        """
        Information Coefficient per signal: Spearman(signal_at_entry, outcome).
        outcome = +1 if won, -1 if lost.
        IC > 0 means the signal correctly predicted direction.
        """
        trades = self.recent(n_recent)
        if len(trades) < 10:
            return {name: 0.0 for name in self.SIGNAL_NAMES}

        outcomes = np.array([1.0 if t.won else -1.0 for t in trades])
        # Scale by direction so signal polarity is correct
        outcomes_directed = np.array([t.r_multiple for t in trades])

        result = {}
        for col, name in zip(self.SIGNAL_COLS, self.SIGNAL_NAMES):
            sig_vals = np.array([getattr(t, col) for t in trades])
            if sig_vals.std() < 1e-9:
                result[name] = 0.0
                continue
            # Spearman via rank correlation
            from scipy.stats import spearmanr
            rho, _ = spearmanr(sig_vals, outcomes_directed)
            result[name] = float(rho) if not np.isnan(rho) else 0.0
        return result

    def win_rate_by_strength(self, signal: str = "sig_composite",
                             n_recent: int = 200) -> Dict[str, dict]:
        """
        Bucket trades by signal |strength| (weak/medium/strong).
        Returns win rate and avg R per bucket.
        """
        trades = self.recent(n_recent)
        if len(trades) < 10:
            return {}

        vals = np.array([abs(getattr(t, signal)) for t in trades])
        q33, q67 = np.percentile(vals, 33), np.percentile(vals, 67)
        buckets = {"weak": [], "medium": [], "strong": []}
        for t, v in zip(trades, vals):
            if v <= q33:
                buckets["weak"].append(t)
            elif v <= q67:
                buckets["medium"].append(t)
            else:
                buckets["strong"].append(t)

        result = {}
        for bname, group in buckets.items():
            if not group:
                continue
            wr = sum(1 for t in group if t.won) / len(group)
            avgr = float(np.mean([t.r_multiple for t in group]))
            result[bname] = {"n": len(group), "win_rate": wr, "avg_r": avgr}
        return result

    def exit_reason_breakdown(self, n_recent: int = 200) -> Dict[str, dict]:
        """Fraction of exits by reason + avg R per reason."""
        trades = self.recent(n_recent)
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in trades:
            groups[t.exit_why].append(t)
        result = {}
        total = len(trades) or 1
        for why, group in sorted(groups.items()):
            result[why] = {
                "count": len(group),
                "pct": len(group) / total,
                "avg_r": float(np.mean([t.r_multiple for t in group])) if group else 0.0,
                "win_rate": sum(1 for t in group if t.won) / len(group) if group else 0.0,
            }
        return result

    def regime_performance(self, n_recent: int = 200) -> Dict[str, dict]:
        """Win rate and avg R per market regime."""
        trades = self.recent(n_recent)
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in trades:
            groups[t.regime].append(t)
        result = {}
        for reg, group in sorted(groups.items()):
            if not group:
                continue
            wr = sum(1 for t in group if t.won) / len(group)
            avgr = float(np.mean([t.r_multiple for t in group]))
            result[reg] = {"n": len(group), "win_rate": wr, "avg_r": avgr}
        return result

    def streak_analysis(self, n_recent: int = 200) -> Dict:
        """Detect win/loss streaks and post-streak performance."""
        trades = self.recent(n_recent)
        if len(trades) < 5:
            return {}
        outcomes = [t.won for t in trades]
        # Current streak
        cur = 1
        for i in range(len(outcomes)-2, -1, -1):
            if outcomes[i] == outcomes[-1]:
                cur += 1
            else:
                break
        # After 3+ loss streak, what's the next trade result?
        post_loss_streak = []
        for i in range(3, len(outcomes)):
            if not any(outcomes[i-3:i]):  # 3 losses in a row
                post_loss_streak.append(outcomes[i])
        pls_wr = sum(post_loss_streak)/len(post_loss_streak) if post_loss_streak else None
        return {
            "current_streak": cur,
            "current_streak_type": "win" if outcomes[-1] else "loss",
            "post_3loss_win_rate": pls_wr,
            "total_trades_analyzed": len(trades),
        }

    def stop_distance_analysis(self, n_recent: int = 200) -> Dict:
        """
        Diagnose stop placement.
        If >70% of exits are stop_loss, stops may be too tight.
        Compute avg hold bars for winners vs losers.
        """
        trades = [t for t in self.recent(n_recent) if t.exit_why]
        if not trades:
            return {}
        stops = [t for t in trades if t.exit_why == "stop_loss"]
        tps   = [t for t in trades if t.exit_why == "take_profit"]
        wins  = [t for t in trades if t.won]
        losers= [t for t in trades if not t.won]
        return {
            "stop_loss_rate": len(stops)/len(trades),
            "take_profit_rate": len(tps)/len(trades),
            "avg_hold_winner_bars": float(np.mean([t.hold_bars for t in wins])) if wins else 0,
            "avg_hold_loser_bars":  float(np.mean([t.hold_bars for t in losers])) if losers else 0,
            "avg_atr_pct": float(np.mean([t.atr_pct for t in trades])) if trades else 0,
            "recommendation": (
                "WIDEN STOPS (>70% hit)" if len(stops)/len(trades) > 0.70
                else "TIGHTEN TP (>70% expire)" if len(tps)/len(trades) > 0.70
                else "STOPS OK"
            ),
        }

    def full_diagnostic(self, n_recent: int = 200) -> Dict:
        return {
            "n_trades": len(self.recent(n_recent)),
            "signal_ic": self.signal_ic(n_recent),
            "win_rate_by_strength": self.win_rate_by_strength(n_recent=n_recent),
            "exit_reasons": self.exit_reason_breakdown(n_recent),
            "regime_perf": self.regime_performance(n_recent),
            "streaks": self.streak_analysis(n_recent),
            "stop_analysis": self.stop_distance_analysis(n_recent),
        }
