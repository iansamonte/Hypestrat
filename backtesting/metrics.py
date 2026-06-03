"""
Comprehensive performance metrics for backtesting results.

Metrics computed:
  - Total return, CAGR, annualised volatility
  - Sharpe ratio (annualised, risk-free = 0)
  - Sortino ratio (penalises downside only)
  - Calmar ratio (CAGR / max drawdown)
  - Omega ratio (gains vs losses above threshold)
  - Maximum drawdown (magnitude, duration, recovery)
  - Win rate, average win, average loss, profit factor
  - Expectancy per trade
  - R-multiple distribution
  - Buy-and-hold comparison
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from utils.math_utils import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    max_drawdown,
    drawdown_series,
    var_historical,
    cvar_historical,
)


def compute_full_metrics(
    equity_curve: np.ndarray,
    trades: List[Any],
    initial_capital: float = 5000.0,
    bars_per_year: int = 252 * 96,
) -> Dict[str, float]:
    """
    Compute comprehensive backtest performance metrics.

    equity_curve: array of portfolio values at each bar
    trades: list of BacktestTrade objects
    initial_capital: starting portfolio value
    bars_per_year: 96 bars/day × 252 trading days (15min)
    """
    if len(equity_curve) < 2:
        return {"error": "insufficient_data"}

    metrics = {}

    # ── Return Statistics ─────────────────────────────────────────────────────
    final_value = float(equity_curve[-1])
    metrics["initial_capital"] = initial_capital
    metrics["final_capital"] = final_value
    metrics["total_return"] = (final_value - initial_capital) / initial_capital
    metrics["total_return_pct"] = metrics["total_return"] * 100
    metrics["multiple"] = final_value / initial_capital

    # CAGR
    n_bars = len(equity_curve)
    n_years = n_bars / bars_per_year
    if n_years > 0 and final_value > 0 and initial_capital > 0:
        metrics["cagr"] = (final_value / initial_capital) ** (1 / n_years) - 1
        metrics["cagr_pct"] = metrics["cagr"] * 100
    else:
        metrics["cagr"] = 0.0
        metrics["cagr_pct"] = 0.0

    # Bar-level returns
    bar_returns = np.diff(equity_curve) / equity_curve[:-1]
    metrics["annualised_vol"] = float(np.std(bar_returns) * np.sqrt(bars_per_year))
    metrics["annualised_vol_pct"] = metrics["annualised_vol"] * 100

    # ── Risk-Adjusted Return ──────────────────────────────────────────────────
    metrics["sharpe"] = sharpe_ratio(bar_returns, bars_per_year)
    metrics["sortino"] = sortino_ratio(bar_returns, bars_per_year)
    mdd, peak_idx, trough_idx = max_drawdown(equity_curve)
    metrics["max_drawdown"] = float(mdd)
    metrics["max_drawdown_pct"] = float(mdd * 100)
    metrics["calmar"] = calmar_ratio(equity_curve, bars_per_year)
    metrics["omega"] = omega_ratio(bar_returns)

    # ── Drawdown Analysis ─────────────────────────────────────────────────────
    dd = drawdown_series(equity_curve)
    # Drawdown duration (bars in drawdown)
    in_drawdown = dd < -0.001
    max_dd_duration = 0
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0
    metrics["max_drawdown_duration_bars"] = max_dd_duration
    metrics["max_drawdown_duration_days"] = max_dd_duration / 96  # 96 bars/day

    # ── VaR / CVaR ────────────────────────────────────────────────────────────
    if len(bar_returns) >= 20:
        metrics["var_95"] = float(var_historical(bar_returns, 0.95))
        metrics["var_99"] = float(var_historical(bar_returns, 0.99))
        metrics["cvar_95"] = float(cvar_historical(bar_returns, 0.95))
        metrics["cvar_99"] = float(cvar_historical(bar_returns, 0.99))

    # ── Trade Statistics ──────────────────────────────────────────────────────
    if trades:
        n_trades = len(trades)
        net_pnls = [t.net_pnl for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p < 0]

        metrics["n_trades"] = n_trades
        metrics["win_rate"] = len(wins) / n_trades if n_trades > 0 else 0
        metrics["win_rate_pct"] = metrics["win_rate"] * 100
        metrics["avg_win"] = float(np.mean(wins)) if wins else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["profit_factor"] = sum(wins) / (sum(abs(l) for l in losses) + 1e-9) if wins else 0.0
        metrics["expectancy"] = float(np.mean(net_pnls)) if net_pnls else 0.0
        metrics["expectancy_per_dollar"] = metrics["expectancy"] / initial_capital

        # R-multiple stats
        r_multiples = [t.r_multiple for t in trades if hasattr(t, "r_multiple")]
        if r_multiples:
            metrics["avg_r_multiple"] = float(np.mean(r_multiples))
            metrics["median_r_multiple"] = float(np.median(r_multiples))
            metrics["r_multiple_std"] = float(np.std(r_multiples))

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = getattr(t, "exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        metrics["exit_reasons"] = exit_reasons

        # Average hold duration
        bar_durations = [t.exit_bar - t.entry_bar for t in trades if hasattr(t, "entry_bar")]
        if bar_durations:
            metrics["avg_hold_bars"] = float(np.mean(bar_durations))
            metrics["avg_hold_hours"] = metrics["avg_hold_bars"] / 4  # 4 bars/hour at 15min

        # Total fees paid
        metrics["total_fees"] = sum(getattr(t, "fees_paid", 0) for t in trades)
        metrics["total_funding"] = sum(getattr(t, "funding_paid", 0) for t in trades)
        metrics["fees_pct_of_pnl"] = metrics["total_fees"] / (sum(wins) + 1e-9)

    # ── Compounding Target Tracking ───────────────────────────────────────────
    target = 1_000_000.0
    monthly_required = (target / initial_capital) ** (1 / 48) - 1
    metrics["monthly_required"] = monthly_required
    metrics["monthly_achieved"] = (final_value / initial_capital) ** (1 / max(n_years * 12, 1)) - 1
    metrics["on_track"] = metrics["monthly_achieved"] >= monthly_required * 0.8

    return metrics


def print_metrics_report(metrics: Dict, title: str = "Backtest Results") -> None:
    """Pretty-print performance metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Capital:      ${metrics.get('initial_capital', 0):>12,.2f} → ${metrics.get('final_capital', 0):>12,.2f}")
    print(f"  Total Return: {metrics.get('total_return_pct', 0):>+12.1f}%  ({metrics.get('multiple', 0):.2f}x)")
    print(f"  CAGR:         {metrics.get('cagr_pct', 0):>+12.1f}%")
    print(f"  Monthly:      {metrics.get('monthly_achieved', 0)*100:>+12.1f}%  (need {metrics.get('monthly_required',0)*100:+.1f}%)")
    print(f"  On Track:     {'✓ YES' if metrics.get('on_track') else '✗ NO':>13}")
    print(f"\n  Sharpe:       {metrics.get('sharpe', 0):>12.2f}")
    print(f"  Sortino:      {metrics.get('sortino', 0):>12.2f}")
    print(f"  Calmar:       {metrics.get('calmar', 0):>12.2f}")
    print(f"  Omega:        {metrics.get('omega', 0):>12.2f}")
    print(f"  Ann. Vol:     {metrics.get('annualised_vol_pct', 0):>12.1f}%")
    print(f"\n  Max Drawdown: {metrics.get('max_drawdown_pct', 0):>12.1f}%")
    print(f"  DD Duration:  {metrics.get('max_drawdown_duration_days', 0):>12.1f} days")
    print(f"  VaR 99%:      {metrics.get('var_99', 0)*100:>12.2f}%")
    print(f"  CVaR 99%:     {metrics.get('cvar_99', 0)*100:>12.2f}%")
    print(f"\n  Trades:       {metrics.get('n_trades', 0):>12,}")
    print(f"  Win Rate:     {metrics.get('win_rate_pct', 0):>12.1f}%")
    print(f"  Profit Factor:{metrics.get('profit_factor', 0):>12.2f}")
    print(f"  Avg R:        {metrics.get('avg_r_multiple', 0):>12.2f}")
    print(f"  Expectancy:   ${metrics.get('expectancy', 0):>11.2f} / trade")
    print(f"  Total Fees:   ${metrics.get('total_fees', 0):>11.2f}")
    print(f"{'='*60}\n")
