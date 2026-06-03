"""
Performance analytics and visualisation for the trading system.

Generates:
  - Equity curve plots with target path overlay
  - Drawdown analysis chart
  - Signal IC heatmap
  - Regime distribution pie chart
  - Monthly returns heatmap (calendar)
  - Trade distribution (win/loss histogram)
  - Rolling Sharpe / Sortino charts
  - Progress toward $5k → $1M goal
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from utils.math_utils import (
    drawdown_series,
    sharpe_ratio,
    sortino_ratio,
    compound_schedule,
)
from utils.logger import logger


class PerformanceAnalytics:
    """
    Comprehensive performance analytics and charting.
    """

    def __init__(
        self,
        initial_capital: float = 5000.0,
        target_capital: float = 1_000_000.0,
        target_months: int = 48,
    ) -> None:
        self.initial_capital = initial_capital
        self.target_capital = target_capital
        self.target_months = target_months

        monthly_r = (target_capital / initial_capital) ** (1 / target_months) - 1
        self._target_schedule = compound_schedule(initial_capital, monthly_r, target_months)

    # ──────────────────────────────────────────────────────────────────────────
    # Main Dashboard
    # ──────────────────────────────────────────────────────────────────────────

    def create_dashboard(
        self,
        equity_curve: np.ndarray,
        trades: List[Any],
        metrics: Dict,
        title: str = "Hypestrat HYPE/USDC Performance",
    ) -> Optional[Any]:
        """
        Create comprehensive performance dashboard.
        Returns plotly Figure or None if plotly unavailable.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("plotly not installed — skipping charts")
            return None

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Equity Curve vs Target Path",
                "Drawdown",
                "Monthly Returns",
                "Trade P&L Distribution",
                "Rolling Sharpe (30-day)",
                "Progress to Goal",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "indicator"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # 1. Equity curve
        self._add_equity_curve(fig, equity_curve, row=1, col=1)

        # 2. Drawdown
        self._add_drawdown(fig, equity_curve, row=1, col=2)

        # 3. Monthly returns
        if trades:
            self._add_trade_distribution(fig, trades, row=2, col=2)

        # 4. Rolling Sharpe
        self._add_rolling_sharpe(fig, equity_curve, row=3, col=1)

        # 5. Goal progress indicator
        self._add_goal_indicator(fig, metrics, row=3, col=2)

        fig.update_layout(
            title=title,
            template="plotly_dark",
            showlegend=True,
            height=900,
        )

        return fig

    def _add_equity_curve(self, fig, equity_curve: np.ndarray, row: int, col: int) -> None:
        x = list(range(len(equity_curve)))

        # Actual equity
        fig.add_trace(
            go.Scatter(
                x=x,
                y=equity_curve,
                name="Portfolio Value",
                line=dict(color="#00ff88", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 255, 136, 0.05)",
            ),
            row=row, col=col,
        )

        # Target path (interpolated to match bar count)
        n_months = len(equity_curve) / (96 * 20)
        monthly_r = (self.target_capital / self.initial_capital) ** (1 / self.target_months) - 1
        target_values = [
            self.initial_capital * (1 + monthly_r) ** (m * n_months / self.target_months * self.target_months / max(n_months, 1))
            for m in range(len(equity_curve))
        ]
        target_interp = np.array([
            self.initial_capital * (1 + monthly_r) ** (i / (96 * 20))
            for i in range(len(equity_curve))
        ])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=target_interp,
                name="Target Path",
                line=dict(color="#ffaa00", width=1, dash="dash"),
                opacity=0.7,
            ),
            row=row, col=col,
        )

        fig.update_yaxes(title_text="Portfolio Value ($)", row=row, col=col, tickprefix="$")

    def _add_drawdown(self, fig, equity_curve: np.ndarray, row: int, col: int) -> None:
        dd = drawdown_series(equity_curve) * 100
        x = list(range(len(dd)))

        fig.add_trace(
            go.Scatter(
                x=x,
                y=dd,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#ff4444", width=1),
                fillcolor="rgba(255, 68, 68, 0.2)",
            ),
            row=row, col=col,
        )
        fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col, ticksuffix="%")

    def _add_trade_distribution(self, fig, trades: List, row: int, col: int) -> None:
        net_pnls = [t.net_pnl for t in trades if hasattr(t, "net_pnl")]
        if not net_pnls:
            return
        colors = ["#00ff88" if p > 0 else "#ff4444" for p in net_pnls]
        fig.add_trace(
            go.Histogram(
                x=net_pnls,
                name="Trade P&L",
                marker_color=colors,
                nbinsx=30,
                opacity=0.7,
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="Net P&L ($)", row=row, col=col)

    def _add_rolling_sharpe(self, fig, equity_curve: np.ndarray, row: int, col: int) -> None:
        if len(equity_curve) < 40:
            return
        returns = np.diff(equity_curve) / equity_curve[:-1]
        window = min(96 * 20, len(returns) // 3)  # Rolling ~20-day window

        rolling_sharpe = [
            sharpe_ratio(returns[max(0, i - window):i], periods_per_year=252 * 96)
            for i in range(window, len(returns) + 1, window // 5)
        ]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(rolling_sharpe))),
                y=rolling_sharpe,
                name="Rolling Sharpe",
                line=dict(color="#44aaff", width=2),
            ),
            row=row, col=col,
        )
        fig.add_hline(y=2.0, line_dash="dash", line_color="green", opacity=0.5, row=row, col=col)
        fig.update_yaxes(title_text="Sharpe Ratio", row=row, col=col)

    def _add_goal_indicator(self, fig, metrics: Dict, row: int, col: int) -> None:
        multiple = metrics.get("multiple", 1.0)
        target_multiple = self.target_capital / self.initial_capital

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=multiple,
                delta={"reference": target_multiple, "valueformat": ".1f", "suffix": "x needed"},
                gauge={
                    "axis": {"range": [0, target_multiple]},
                    "bar": {"color": "#00ff88"},
                    "threshold": {
                        "line": {"color": "gold", "width": 4},
                        "thickness": 0.75,
                        "value": target_multiple,
                    },
                },
                title={"text": f"Progress to Goal<br>{multiple:.2f}x / {target_multiple:.0f}x"},
            ),
            row=row, col=col,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Text Reports
    # ──────────────────────────────────────────────────────────────────────────

    def generate_report(self, metrics: Dict) -> str:
        """Generate formatted text performance report."""
        lines = [
            "=" * 70,
            "  HYPESTRAT HYPE/USDC PERFORMANCE REPORT",
            "=" * 70,
            "",
            "  GROWTH TARGETS",
            f"    Initial Capital:  ${metrics.get('initial_capital', self.initial_capital):>15,.2f}",
            f"    Target Capital:   ${self.target_capital:>15,.0f}",
            f"    Target Multiple:  {self.target_capital/self.initial_capital:>14.0f}x",
            f"    Required CAGR:    {((self.target_capital/self.initial_capital)**(1/4)-1)*100:>14.1f}%",
            f"    Required Monthly: {((self.target_capital/self.initial_capital)**(1/48)-1)*100:>14.2f}%",
            "",
            "  ACHIEVED RESULTS",
            f"    Final Capital:    ${metrics.get('final_capital', 0):>15,.2f}",
            f"    Total Return:     {metrics.get('total_return_pct', 0):>+14.1f}%",
            f"    Multiple:         {metrics.get('multiple', 1):>14.2f}x",
            f"    CAGR:             {metrics.get('cagr_pct', 0):>+14.1f}%",
            f"    Monthly Return:   {metrics.get('monthly_achieved', 0)*100:>+14.2f}%",
            f"    On Track:         {'✓ YES' if metrics.get('on_track') else '✗ NO':>15}",
            "",
            "  RISK-ADJUSTED PERFORMANCE",
            f"    Sharpe Ratio:     {metrics.get('sharpe', 0):>15.2f}",
            f"    Sortino Ratio:    {metrics.get('sortino', 0):>15.2f}",
            f"    Calmar Ratio:     {metrics.get('calmar', 0):>15.2f}",
            f"    Omega Ratio:      {metrics.get('omega', 0):>15.2f}",
            f"    Ann. Volatility:  {metrics.get('annualised_vol_pct', 0):>14.1f}%",
            "",
            "  DRAWDOWN ANALYSIS",
            f"    Max Drawdown:     {metrics.get('max_drawdown_pct', 0):>14.1f}%",
            f"    Max DD Duration:  {metrics.get('max_drawdown_duration_days', 0):>14.1f} days",
            f"    VaR 99%:          {metrics.get('var_99', 0)*100:>14.2f}%",
            f"    CVaR 99%:         {metrics.get('cvar_99', 0)*100:>14.2f}%",
            "",
            "  TRADE STATISTICS",
            f"    Total Trades:     {metrics.get('n_trades', 0):>15,}",
            f"    Win Rate:         {metrics.get('win_rate_pct', 0):>14.1f}%",
            f"    Profit Factor:    {metrics.get('profit_factor', 0):>15.2f}",
            f"    Avg R-Multiple:   {metrics.get('avg_r_multiple', 0):>+14.2f}R",
            f"    Expectancy/Trade: ${metrics.get('expectancy', 0):>14.2f}",
            f"    Total Fees Paid:  ${metrics.get('total_fees', 0):>14.2f}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)

    def print_compound_schedule(self) -> None:
        """Print the target compound growth schedule."""
        monthly_r = (self.target_capital / self.initial_capital) ** (1 / self.target_months) - 1
        print("\n  TARGET COMPOUND SCHEDULE")
        print(f"  {'Month':>6}  {'Target Value':>15}  {'Multiple':>10}")
        print(f"  {'-'*35}")
        for month in [0, 3, 6, 12, 18, 24, 30, 36, 42, 48]:
            val = self.initial_capital * (1 + monthly_r) ** month
            mult = val / self.initial_capital
            print(f"  {month:>6}  ${val:>14,.0f}  {mult:>9.1f}x")
        print()
