#!/usr/bin/env python3
"""
IAN.AI Ventures — Main Entry Point

Usage:
    python main.py --mode simulate    # 12-month simulation showing growth to $30k MRR
    python main.py --mode run         # Run one company cycle interactively
    python main.py --mode report      # Show current state dashboard

Environment:
    ANTHROPIC_API_KEY must be set (or in .env file)
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load .env before anything else
from dotenv import load_dotenv

load_dotenv()

import click
from rich.console import Console
from rich.rule import Rule

from src.models.company_state import CompanyState, Client, Lead, Action
from src.models.metrics import MetricsCalculator
from src.orchestrators.company_orchestrator import CompanyOrchestrator
from src.orchestrators.revenue_orchestrator import RevenueOrchestrator
from src.orchestrators.operations_orchestrator import OperationsOrchestrator
from src.dashboard.reporter import Reporter
from src.tools.crm_tool import CRMTool, COMPANY_POOL

console = Console()

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("logs/ian_ai.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Quiet noisy libraries
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ------------------------------------------------------------------
# Simulation helpers (no API calls — for --mode simulate)
# ------------------------------------------------------------------

def _simulate_marketing(state: CompanyState, month: int) -> int:
    """
    Simulate marketing lead generation without API calls.
    Returns number of new leads added.
    """
    # Ramp up leads over time as brand grows
    base_leads = min(3 + month * 1.5, 15)
    noise = random.uniform(0.7, 1.4)
    n_leads = int(base_leads * noise)

    sources = ["organic", "outbound", "referral", "paid"]
    weights = [0.35, 0.35, 0.15, 0.15]

    for _ in range(n_leads):
        source = random.choices(sources, weights=weights)[0]
        lead = CRMTool.generate_random_lead(source=source)
        state.add_lead(lead)

    return n_leads


def _simulate_sales(state: CompanyState, month: int) -> int:
    """
    Simulate sales converting leads without API calls.
    Returns number of new clients.

    Each lead can advance multiple stages per cycle to model
    an accelerating sales motion as the team gets better.
    """
    # Conversion improves as sales process matures
    base_conversion = 0.12 + month * 0.025
    conversion_rate = min(base_conversion, 0.35)

    # Speed of pipeline advancement increases with month
    stage_advance_prob = min(0.5 + month * 0.05, 0.85)

    prospects = [l for l in state.leads if l.stage in ("prospect", "qualified", "proposal")]
    # Sort by score so best leads are worked first
    prospects.sort(key=lambda l: l.score, reverse=True)
    new_clients = 0

    for lead in prospects[:25]:  # Work up to 25 leads per cycle
        # Each lead can advance multiple stages in one cycle
        for _ in range(3):  # Max 3 stage advances per cycle
            if lead.stage == "prospect" and lead.score >= 50:
                if random.random() < stage_advance_prob:
                    lead.stage = "qualified"
                else:
                    break
            elif lead.stage == "qualified":
                if random.random() < stage_advance_prob * 0.8:
                    lead.stage = "proposal"
                else:
                    break
            elif lead.stage == "proposal":
                # Try to close
                if random.random() < conversion_rate:
                    # Determine tier based on lead value
                    if lead.value_potential >= 1500:
                        tier = 3
                    elif lead.value_potential >= 750:
                        tier = 2
                    else:
                        tier = 1

                    tier_prices = {1: 500.0, 2: 1000.0, 3: 2000.0}
                    client = Client(
                        name=lead.name,
                        company=lead.company,
                        tier=tier,
                        monthly_value=tier_prices[tier],
                        health_score=85.0 + random.uniform(-5, 10),
                        churn_risk=0.05,
                        months_active=0,
                    )
                    lead.stage = "closed_won"
                    state.add_client(client)
                    new_clients += 1
                break
            else:
                break

    return new_clients


def _simulate_customer_success(state: CompanyState, month: int) -> Dict[str, Any]:
    """
    Simulate customer success activities without API calls.
    Returns dict with churned, upsold counts.
    """
    churned = 0
    upsold = 0

    for client in list(state.clients):
        # Increment tenure
        client.months_active += 1

        # Health drift
        delta = random.uniform(-5, 8)
        client.health_score = max(30, min(100, client.health_score + delta))

        # Churn risk based on health
        if client.health_score >= 80:
            client.churn_risk = max(0.02, client.churn_risk - 0.02)
        elif client.health_score < 55:
            client.churn_risk = min(0.60, client.churn_risk + 0.08)

        # Churn event
        if random.random() < client.churn_risk * 0.6:  # dampened for simulation
            state.remove_client(client.id)
            churned += 1
            continue

        # Upsell opportunity
        if (
            client.tier < 3
            and client.health_score >= 82
            and client.months_active >= 3
            and random.random() < 0.12  # ~12% monthly upsell rate for eligible clients
        ):
            tier_prices = {1: 500.0, 2: 1000.0, 3: 2000.0}
            new_tier = client.tier + 1
            client.tier = new_tier
            client.monthly_value = tier_prices[new_tier]
            upsold += 1

    state.update_mrr()
    return {"churned": churned, "upsold": upsold}


def _simulate_finance(state: CompanyState) -> None:
    """Update financial state after a simulated month."""
    base_costs = 110.0 + len(state.clients) * 5.0
    state.monthly_expenses = base_costs

    net = state.mrr - state.monthly_expenses
    state.capital = max(0, state.capital + net)


def run_simulation(months: int = 12) -> None:
    """
    Run a 12-month simulation of IAN.AI Ventures growth.
    Uses no API calls — pure algorithmic simulation with realistic growth curves.
    """
    console.print()
    console.print(Rule("[bold cyan]IAN.AI VENTURES — 12-Month Growth Simulation[/bold cyan]"))
    console.print(
        "[dim]Simulating autonomous AI company growth from $0 to $30,000 MRR[/dim]\n"
    )

    state = CompanyState()
    reporter = Reporter(state)

    # Track monthly history
    history: List[Dict[str, Any]] = []
    all_actions: List[Action] = []

    console.print(f"  [bold]Starting conditions:[/bold]")
    console.print(f"  Capital:     [green]${state.capital:,.0f}[/green]")
    console.print(f"  MRR:         $0")
    console.print(f"  Target:      [yellow]${state.target_mrr:,.0f}/mo[/yellow]")
    console.print(f"  Clients:     0 / 36")
    console.print(f"  Tiers:       T1 $500 | T2 $1,000 | T3 $2,000")
    console.print()
    console.print(Rule("[dim]Monthly Progress[/dim]"))
    console.print()

    for month in range(1, months + 1):
        state.month = month
        month_start_mrr = state.mrr
        month_start_clients = len(state.clients)

        # 1. Marketing generates leads
        new_leads = _simulate_marketing(state, month)

        # 2. Sales converts
        new_clients = _simulate_sales(state, month)

        # 3. Customer success manages retention + upsell
        cs_result = _simulate_customer_success(state, month)

        # 4. Finance updates
        _simulate_finance(state)

        # Calculate monthly metrics
        state.update_mrr()
        growth = MetricsCalculator.calculate_growth_rate(state.mrr, month_start_mrr)

        snap = {
            "month": month,
            "mrr": state.mrr,
            "clients": len(state.clients),
            "new_clients": new_clients,
            "new_leads": new_leads,
            "churned": cs_result["churned"],
            "upsold": cs_result["upsold"],
            "capital": state.capital,
            "expenses": state.monthly_expenses,
            "net_profit": state.mrr - state.monthly_expenses,
            "growth_rate": growth,
        }
        history.append(snap)

        # Simulate a few actions for display
        sim_actions = [
            Action(
                agent="Marketing",
                action_type="add_leads_from_campaign",
                description=f"Generated {new_leads} new leads",
                result=f"Pipeline: {len(state.leads)} total leads",
                impact="positive",
            ),
            Action(
                agent="Sales",
                action_type="close_deal",
                description=f"Closed {new_clients} deals this month",
                result=f"Total clients: {len(state.clients)}",
                impact="positive" if new_clients > 0 else "neutral",
            ),
            Action(
                agent="Finance",
                action_type="update_mrr",
                description=f"MRR updated to ${state.mrr:,.0f}",
                result=f"Net: ${state.mrr - state.monthly_expenses:+,.0f}/mo",
                impact="positive" if state.mrr > month_start_mrr else "neutral",
            ),
        ]
        if cs_result["churned"] > 0:
            sim_actions.append(Action(
                agent="CustomerSuccess",
                action_type="handle_churn_risk",
                description=f"Lost {cs_result['churned']} client(s) to churn",
                result="Executing retention campaign",
                impact="negative",
            ))
        if cs_result["upsold"] > 0:
            sim_actions.append(Action(
                agent="CustomerSuccess",
                action_type="execute_upsell",
                description=f"Upsold {cs_result['upsold']} client(s) to higher tier",
                result=f"MRR expansion: +${cs_result['upsold'] * 500:,.0f}",
                impact="positive",
            ))
        all_actions.extend(sim_actions)

        # Progress display
        reporter.show_simulation_progress(month, months, state.mrr, state.target_mrr)

        # Hit target?
        if state.mrr >= state.target_mrr:
            console.print(
                f"\n  [bold green]🎯 $30,000 MRR TARGET ACHIEVED at Month {month}![/bold green]"
            )
            break

    # Final dashboard
    console.print()
    console.print(Rule("[bold cyan]Final State[/bold cyan]"))
    reporter.show_dashboard(
        recent_actions=all_actions[-20:],
        monthly_history=history,
    )

    # Summary stats
    console.print(Rule("[bold]Simulation Summary[/bold]"))
    final = history[-1] if history else {}
    console.print(f"  Months simulated:  {len(history)}")
    console.print(f"  Final MRR:         [bold green]${state.mrr:,.0f}[/bold green] / ${state.target_mrr:,.0f}")
    console.print(f"  Progress:          {state.progress_to_target():.1f}%")
    console.print(f"  Total clients:     {len(state.clients)}")
    console.print(f"  Total leads gen'd: {len(state.leads)}")
    console.print(f"  Final capital:     ${state.capital:,.0f}")
    console.print(f"  Monthly burn:      ${state.monthly_expenses:,.0f}")
    console.print(f"  Net monthly CF:    ${state.mrr - state.monthly_expenses:+,.0f}")

    if state.mrr >= state.target_mrr:
        console.print(
            f"\n  [bold green]SUCCESS: IAN.AI Ventures reached ${state.target_mrr:,.0f} MRR "
            f"in {len(history)} months![/bold green]"
        )
    else:
        gap = state.target_mrr - state.mrr
        console.print(
            f"\n  [yellow]Gap remaining: ${gap:,.0f} MRR | "
            f"Continue running cycles to close the gap.[/yellow]"
        )
    console.print()


# ------------------------------------------------------------------
# Run mode (real API calls)
# ------------------------------------------------------------------

def run_company_cycle() -> None:
    """Run one interactive company cycle using real Anthropic API calls."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[red]Error: ANTHROPIC_API_KEY not set. "
            "Copy .env.example to .env and add your API key.[/red]"
        )
        sys.exit(1)

    console.print()
    console.print(Rule("[bold cyan]IAN.AI VENTURES — Live Cycle[/bold cyan]"))
    console.print("[dim]Running real AI agents via Anthropic API...[/dim]\n")

    state = CompanyState()
    reporter = Reporter(state)

    # Show initial state
    reporter.show_dashboard()

    # Create orchestrators
    orchestrator = CompanyOrchestrator(state)
    revenue_orch = RevenueOrchestrator(state)
    ops_orch = OperationsOrchestrator(state)

    console.print(Rule("[bold]Running CEO + Marketing + Sales + Finance[/bold]"))
    console.print("[dim]This will make real API calls to claude-sonnet-4-6...[/dim]\n")

    try:
        # Run daily cycle (CEO + Marketing + Sales + Finance)
        all_actions = orchestrator.run_daily_cycle()

        # Flatten actions
        flat_actions: List[Action] = []
        for agent_name, actions in all_actions.items():
            flat_actions.extend(actions)

        # Update state
        state.update_mrr()

        console.print("\n")
        reporter.show_dashboard(recent_actions=flat_actions)

        console.print(f"\n[bold green]Cycle complete![/bold green]")
        console.print(f"  Agents run: {', '.join(all_actions.keys())}")
        console.print(f"  Total actions: {len(flat_actions)}")
        console.print(f"  MRR after cycle: [bold]${state.mrr:,.0f}[/bold]")
        console.print(f"  Clients: {len(state.clients)}")
        console.print(f"  Leads: {len(state.leads)}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cycle interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during cycle: {e}[/red]")
        logging.exception("Cycle error")
        raise


# ------------------------------------------------------------------
# Report mode
# ------------------------------------------------------------------

def show_report() -> None:
    """Show the current company state dashboard."""
    console.print()
    console.print(Rule("[bold cyan]IAN.AI VENTURES — Company Report[/bold cyan]"))

    # Load state (fresh for report mode — would load from persistence in production)
    state = CompanyState()

    # Add some sample data so the dashboard isn't empty
    _seed_demo_state(state)

    reporter = Reporter(state)
    reporter.show_dashboard()


def _seed_demo_state(state: CompanyState) -> None:
    """Seed some demo data for the report mode."""
    # Add a mix of clients
    demo_clients = [
        Client(name="Sarah Mitchell", company="BrightEdge Marketing", tier=1, monthly_value=500, health_score=92, churn_risk=0.04, months_active=4),
        Client(name="James Thornton", company="NovaTech Solutions", tier=2, monthly_value=1000, health_score=87, churn_risk=0.06, months_active=3),
        Client(name="Laura Reyes", company="Summit Consulting", tier=1, monthly_value=500, health_score=78, churn_risk=0.12, months_active=2),
        Client(name="Kevin Park", company="Horizon Retail", tier=3, monthly_value=2000, health_score=95, churn_risk=0.02, months_active=5),
        Client(name="Danielle Ford", company="BluePeak Digital", tier=2, monthly_value=1000, health_score=73, churn_risk=0.18, months_active=1),
    ]
    for c in demo_clients:
        state.clients.append(c)
    state.update_mrr()

    # Add demo leads
    from src.tools.crm_tool import COMPANY_POOL
    stages = ["prospect", "prospect", "qualified", "qualified", "proposal", "negotiation", "closed_won"]
    for i, (name, company, industry) in enumerate(COMPANY_POOL[:12]):
        lead = Lead(
            name=name, company=company, industry=industry,
            score=40 + i * 5, stage=stages[i % len(stages)],
            value_potential=random.choice([500, 500, 1000, 2000]),
            source=random.choice(["organic", "outbound", "referral"]),
        )
        state.leads.append(lead)

    state.capital = 850.0
    state.monthly_expenses = 145.0
    state.month = 2
    state.priorities = [
        "Close 3 more Tier 1 clients this month",
        "Launch LinkedIn content campaign",
        "Identify 2 upsell candidates from current client base",
    ]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["simulate", "run", "report"], case_sensitive=False),
    default="report",
    show_default=True,
    help=(
        "simulate: 12-month algorithmic simulation (no API calls) | "
        "run: one live cycle with real Anthropic API | "
        "report: show current state dashboard"
    ),
)
@click.option(
    "--months",
    default=12,
    show_default=True,
    help="Number of months for simulation mode",
)
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level",
)
def main(mode: str, months: int, log_level: str) -> None:
    """
    IAN.AI Ventures — Fully Autonomous AI Company

    Targeting $30,000 MRR from AI automation services for SMBs.

    Starting capital: $1,000
    """
    setup_logging(log_level)

    if mode == "simulate":
        run_simulation(months=months)
    elif mode == "run":
        run_company_cycle()
    elif mode == "report":
        show_report()
    else:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
