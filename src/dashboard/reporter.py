"""
Reporter — Rich terminal dashboard for IAN.AI Ventures company metrics.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich import box

from src.models.company_state import CompanyState, Action
from src.models.metrics import MetricsCalculator

if TYPE_CHECKING:
    from src.orchestrators.company_orchestrator import CompanyOrchestrator


console = Console()


class Reporter:
    """
    Rich terminal dashboard for IAN.AI Ventures.

    Displays:
    - Company header with key KPIs
    - MRR progress bar toward $30k target
    - Agent action table
    - Pipeline summary
    - Top clients
    - Financial snapshot
    """

    def __init__(self, state: CompanyState) -> None:
        self.state = state
        self.console = Console()

    # ------------------------------------------------------------------
    # Main dashboard
    # ------------------------------------------------------------------

    def show_dashboard(
        self,
        recent_actions: Optional[List[Action]] = None,
        monthly_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Render the full company dashboard."""
        self.console.print()
        self.console.print(Rule("[bold cyan]IAN.AI VENTURES — Autonomous AI Company[/bold cyan]"))
        self.console.print()

        # Header row
        self._show_header()
        self.console.print()

        # MRR progress bar
        self._show_mrr_progress()
        self.console.print()

        # Side-by-side: Clients + Pipeline
        left = self._build_client_panel()
        right = self._build_pipeline_panel()
        self.console.print(Columns([left, right]))
        self.console.print()

        # Financial metrics
        self._show_financial_metrics()
        self.console.print()

        # Recent agent actions
        if recent_actions:
            self._show_agent_actions(recent_actions)
            self.console.print()

        # Growth history (if available)
        if monthly_history and len(monthly_history) > 1:
            self._show_growth_history(monthly_history)
            self.console.print()

        self.console.print(Rule("[dim]end of report[/dim]"))

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _show_header(self) -> None:
        state = self.state
        state.update_mrr()

        net_monthly = state.mrr - state.monthly_expenses
        status_color = "green" if net_monthly >= 0 else "yellow"
        runway = state.runway_months()
        runway_str = "∞ (profitable)" if runway == float('inf') else f"{runway:.1f} months"

        header_text = (
            f"[bold white]{state.name}[/bold white]  |  "
            f"Month [cyan]{state.month}[/cyan]  |  "
            f"Capital: [green]${state.capital:,.0f}[/green]  |  "
            f"MRR: [yellow]${state.mrr:,.0f}[/yellow] / [dim]${state.target_mrr:,.0f}[/dim]  |  "
            f"Net/mo: [{status_color}]${net_monthly:+,.0f}[/{status_color}]  |  "
            f"Runway: [cyan]{runway_str}[/cyan]"
        )
        self.console.print(Panel(header_text, box=box.DOUBLE_EDGE, padding=(0, 1)))

    def _show_mrr_progress(self) -> None:
        state = self.state
        progress_pct = state.progress_to_target()
        remaining = state.target_mrr - state.mrr
        bar_width = 60
        filled = int(bar_width * progress_pct / 100)
        empty = bar_width - filled

        # Color based on progress
        if progress_pct >= 90:
            color = "green"
        elif progress_pct >= 50:
            color = "yellow"
        elif progress_pct >= 20:
            color = "orange3"
        else:
            color = "red"

        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
        label = (
            f"[bold]MRR Progress[/bold]  "
            f"[{color}]${state.mrr:>8,.0f}[/{color}] / ${state.target_mrr:,.0f}  "
            f"[{color}]{progress_pct:>5.1f}%[/{color}]  "
            f"  (${remaining:,.0f} remaining)"
        )

        self.console.print(Panel(
            f"{label}\n\n{bar}",
            title="[bold]$30,000 MRR Target[/bold]",
            border_style=color,
            padding=(0, 1),
        ))

    def _build_client_panel(self) -> Panel:
        state = self.state
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE,
            expand=True,
        )
        table.add_column("Client", style="white", max_width=18)
        table.add_column("Tier", style="dim", width=6)
        table.add_column("MRR", justify="right", style="green", width=8)
        table.add_column("Health", justify="right", width=7)

        if not state.clients:
            table.add_row("[dim]No clients yet[/dim]", "", "", "")
        else:
            clients_sorted = sorted(state.clients, key=lambda c: c.monthly_value, reverse=True)
            for c in clients_sorted[:8]:
                health_color = "green" if c.health_score >= 75 else ("yellow" if c.health_score >= 55 else "red")
                tier_short = f"T{c.tier}"
                table.add_row(
                    c.name.split()[0] if c.name else "?",
                    tier_short,
                    f"${c.monthly_value:,.0f}",
                    f"[{health_color}]{c.health_score:.0f}[/{health_color}]",
                )
            if len(state.clients) > 8:
                table.add_row(f"[dim]+{len(state.clients)-8} more[/dim]", "", "", "")

        total_text = f"\n[bold]Total: {len(state.clients)} clients | ${state.mrr:,.0f}/mo[/bold]"
        return Panel(
            Text.assemble(("", ""), ("", "")) if not state.clients
            else table,
            title=f"[bold cyan]Active Clients ({len(state.clients)})[/bold cyan]",
            subtitle=total_text,
            padding=(0, 1),
        )

    def _build_pipeline_panel(self) -> Panel:
        state = self.state
        funnel = MetricsCalculator.calculate_conversion_funnel(state.leads)
        pipeline_value = state.pipeline_value()

        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE,
            expand=True,
        )
        table.add_column("Stage", style="white")
        table.add_column("Count", justify="right", style="cyan", width=7)
        table.add_column("Bar", width=15)

        total = max(len(state.leads), 1)
        stages_display = ["prospect", "qualified", "proposal", "negotiation", "closed_won"]
        stage_colors = {
            "prospect": "dim white",
            "qualified": "cyan",
            "proposal": "yellow",
            "negotiation": "orange3",
            "closed_won": "green",
        }
        for stage in stages_display:
            count = funnel.get(stage, 0)
            pct = count / total * 100
            bar_len = int(pct / 7)
            color = stage_colors.get(stage, "white")
            bar = f"[{color}]{'▪' * max(bar_len, 1) if count > 0 else ''}[/{color}]"
            table.add_row(stage.replace("_", " ").title(), str(count), bar)

        return Panel(
            table,
            title="[bold magenta]Sales Pipeline[/bold magenta]",
            subtitle=f"\n[bold]Pipeline Value: ${pipeline_value:,.0f}/mo | {len(state.leads)} leads[/bold]",
            padding=(0, 1),
        )

    def _show_financial_metrics(self) -> None:
        state = self.state
        metrics = MetricsCalculator.full_metrics(state)

        table = Table(show_header=False, box=box.SIMPLE, expand=False, padding=(0, 2))
        table.add_column("Metric", style="dim", width=22)
        table.add_column("Value", style="bold white", width=16)
        table.add_column("Metric", style="dim", width=22)
        table.add_column("Value", style="bold white", width=16)

        mrr = state.mrr
        arr = MetricsCalculator.calculate_arr(mrr)
        cac = metrics.get("cac", 0)
        ltv = metrics.get("ltv", 0)
        ltv_cac = metrics.get("ltv_cac_ratio", 0)
        net_cf = metrics.get("net_monthly_cash_flow", 0)
        cf_color = "green" if net_cf >= 0 else "red"

        table.add_row("ARR", f"${arr:,.0f}", "CAC", f"${cac:,.0f}")
        table.add_row("LTV", f"${ltv:,.0f}", "LTV:CAC", f"{ltv_cac:.1f}x")
        table.add_row(
            "Net Cash Flow",
            f"[{cf_color}]${net_cf:+,.0f}[/{cf_color}]",
            "Monthly Expenses",
            f"${state.monthly_expenses:,.0f}"
        )
        table.add_row(
            "Total Leads",
            str(len(state.leads)),
            "Qualified Leads",
            str(metrics.get("lead_funnel", {}).get("qualified", 0)),
        )

        self.console.print(Panel(table, title="[bold]Financial Metrics[/bold]", padding=(0, 1)))

    def _show_agent_actions(self, actions: List[Action]) -> None:
        if not actions:
            return

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.SIMPLE,
            expand=True,
        )
        table.add_column("Agent", style="cyan", width=16)
        table.add_column("Action", style="white", width=24)
        table.add_column("Result", style="dim", max_width=50)
        table.add_column("Impact", width=8)

        impact_colors = {"positive": "green", "negative": "red", "neutral": "dim"}

        for action in actions[-15:]:  # Show last 15 actions
            color = impact_colors.get(action.impact, "dim")
            icon = "+" if action.impact == "positive" else ("-" if action.impact == "negative" else "~")
            table.add_row(
                action.agent,
                action.action_type[:22],
                action.result[:48] + ("..." if len(action.result) > 48 else ""),
                f"[{color}]{icon}[/{color}]",
            )

        self.console.print(Panel(
            table,
            title=f"[bold blue]Agent Actions (last cycle — {len(actions)} total)[/bold blue]",
            padding=(0, 1),
        ))

    def _show_growth_history(self, history: List[Dict[str, Any]]) -> None:
        table = Table(
            show_header=True,
            header_style="bold green",
            box=box.SIMPLE,
            expand=False,
        )
        table.add_column("Month", style="dim", width=7)
        table.add_column("MRR", justify="right", style="green", width=10)
        table.add_column("Clients", justify="right", style="cyan", width=8)
        table.add_column("Growth", justify="right", width=8)
        table.add_column("Capital", justify="right", style="yellow", width=10)
        table.add_column("Net", justify="right", width=10)

        for snap in history[-8:]:  # Last 8 months
            growth = snap.get("growth_rate", 0)
            growth_color = "green" if growth > 0 else "red"
            net = snap.get("net_profit", 0)
            net_color = "green" if net >= 0 else "red"
            table.add_row(
                str(snap["month"]),
                f"${snap['mrr']:,.0f}",
                str(snap["clients"]),
                f"[{growth_color}]{growth:+.1f}%[/{growth_color}]",
                f"${snap['capital']:,.0f}",
                f"[{net_color}]${net:+,.0f}[/{net_color}]",
            )

        self.console.print(Panel(
            table,
            title="[bold green]Growth History[/bold green]",
            padding=(0, 1),
        ))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def print_monthly_report(self, report: Dict[str, Any]) -> None:
        """Print a monthly cycle report."""
        self.console.print()
        self.console.print(Rule(f"[bold cyan]Month {report['month']} Report[/bold cyan]"))

        achieved = report.get("target_achieved", report["mrr"] >= self.state.target_mrr)
        status = "[bold green]TARGET REACHED![/bold green]" if achieved else "[yellow]In progress[/yellow]"

        self.console.print(f"  MRR:        [bold green]${report['mrr']:>10,.0f}[/bold green]  {status}")
        self.console.print(f"  Progress:   {report['progress_pct']:>9.1f}%")
        self.console.print(f"  Clients:    {report['clients']:>10} (+{report['new_clients']} new)")
        self.console.print(f"  Capital:    [yellow]${report['capital']:>10,.0f}[/yellow]")
        self.console.print(f"  Runway:     {report['runway']:>9.1f} months")
        self.console.print(f"  MoM Growth: [cyan]{report['mrr_growth']:>+9.1f}%[/cyan]")
        self.console.print(f"  Actions:    {report['total_actions']:>10}")
        self.console.print()

    def print_sprint_report(self, sprint: Dict[str, Any]) -> None:
        """Print a revenue sprint report."""
        achieved = sprint.get("target_achieved", False)
        icon = "[green]✓[/green]" if achieved else "[yellow]~[/yellow]"
        self.console.print(
            f"{icon} Sprint '{sprint['sprint_label']}': "
            f"+${sprint['actual_mrr_increase']:,.0f} MRR | "
            f"{sprint['new_clients_acquired']} clients | "
            f"{sprint['conversion_rate']:.1f}% conversion"
        )

    def show_simulation_progress(
        self, month: int, total_months: int, mrr: float, target: float
    ) -> None:
        """Show a simple one-line simulation progress update."""
        pct = mrr / target * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = "green" if pct >= 80 else ("yellow" if pct >= 40 else "cyan")
        self.console.print(
            f"  Month {month:>2}/{total_months} | "
            f"[{color}]{bar}[/{color}] "
            f"[{color}]{pct:>5.1f}%[/{color}] | "
            f"MRR: [bold]${mrr:>8,.0f}[/bold] | "
            f"{'🎯 TARGET REACHED!' if mrr >= target else ''}"
        )
