"""
Revenue Orchestrator — Coordinates marketing + sales to hit revenue milestones.
Runs focused growth sprints toward MRR targets.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from src.agents.marketing_agent import MarketingAgent
from src.agents.sales_agent import SalesAgent
from src.models.company_state import Action, CompanyState
from src.models.metrics import MetricsCalculator
from src.tools.crm_tool import CRMTool

logger = logging.getLogger(__name__)


class RevenueOrchestrator:
    """
    Focused orchestrator for revenue growth.

    Coordinates marketing and sales in sprint mode to hit
    specific MRR milestones. Tracks funnel metrics and conversion rates.
    """

    def __init__(self, state: CompanyState, model: str = "claude-sonnet-4-6") -> None:
        self.state = state
        self.model = model
        self._marketing: Optional[MarketingAgent] = None
        self._sales: Optional[SalesAgent] = None
        self._sprint_history: List[Dict[str, Any]] = []

    @property
    def marketing(self) -> MarketingAgent:
        if self._marketing is None:
            self._marketing = MarketingAgent(model=self.model)
        return self._marketing

    @property
    def sales(self) -> SalesAgent:
        if self._sales is None:
            self._sales = SalesAgent(model=self.model)
        return self._sales

    def run_growth_sprint(
        self,
        target_new_mrr: float,
        sprint_label: str = "Growth Sprint",
    ) -> Dict[str, Any]:
        """
        Run a focused growth sprint targeting a specific MRR increase.

        Steps:
        1. Marketing generates leads targeting the revenue gap
        2. Sales qualifies and converts leads
        3. Track funnel metrics
        4. Report results

        Returns a sprint report with conversion rates and MRR impact.
        """
        logger.info(
            f"=== {sprint_label} | Target: +${target_new_mrr:,.0f} MRR | "
            f"Current MRR: ${self.state.mrr:,.0f} ==="
        )

        sprint_start_mrr = self.state.mrr
        sprint_start_leads = len(self.state.leads)
        sprint_start_clients = len(self.state.clients)

        # Determine how many clients we need
        avg_client_value = self._estimate_avg_client_value()
        clients_needed = max(1, int(target_new_mrr / avg_client_value) + 1)
        # Need ~5-7x leads to get clients (funnel conversion)
        leads_needed = clients_needed * 6

        logger.info(
            f"Sprint plan: Need {clients_needed} new clients "
            f"(${avg_client_value:.0f} avg) from {leads_needed} leads"
        )

        all_actions: Dict[str, List[Action]] = {}

        # Phase 1: Marketing lead generation
        logger.info("Phase 1: Marketing lead generation...")
        # Run marketing multiple times if needed
        mkt_rounds = max(1, min(3, clients_needed // 2))
        for i in range(mkt_rounds):
            actions = self.marketing.run_cycle(self.state)
            all_actions[f"Marketing_Round_{i+1}"] = actions

        # Phase 2: Sales conversion
        logger.info("Phase 2: Sales conversion...")
        sales_rounds = max(1, min(3, clients_needed))
        for i in range(sales_rounds):
            actions = self.sales.run_cycle(self.state)
            all_actions[f"Sales_Round_{i+1}"] = actions

        # Calculate sprint results
        self.state.update_mrr()
        sprint_end_mrr = self.state.mrr
        mrr_gained = sprint_end_mrr - sprint_start_mrr
        new_leads = len(self.state.leads) - sprint_start_leads
        new_clients = len(self.state.clients) - sprint_start_clients

        funnel = MetricsCalculator.calculate_conversion_funnel(self.state.leads)
        conversion_rate = (
            new_clients / new_leads * 100
            if new_leads > 0
            else 0
        )

        total_actions = sum(len(v) for v in all_actions.values())

        sprint_report = {
            "sprint_label": sprint_label,
            "target_mrr_increase": target_new_mrr,
            "actual_mrr_increase": mrr_gained,
            "target_achieved": mrr_gained >= target_new_mrr * 0.7,  # 70% = success
            "start_mrr": sprint_start_mrr,
            "end_mrr": sprint_end_mrr,
            "new_leads_generated": new_leads,
            "new_clients_acquired": new_clients,
            "conversion_rate": round(conversion_rate, 1),
            "total_actions": total_actions,
            "funnel": funnel,
            "mrr_progress": round(self.state.progress_to_target(), 1),
        }

        self._sprint_history.append(sprint_report)

        logger.info(
            f"Sprint complete: +${mrr_gained:,.0f} MRR | "
            f"{new_clients} clients | {conversion_rate:.1f}% conversion"
        )
        return sprint_report

    def run_lead_nurture_sequence(self) -> Dict[str, Any]:
        """
        Run a nurture sequence to advance leads already in the pipeline.
        Focuses on converting existing qualified leads before generating new ones.
        """
        logger.info("Running lead nurture sequence...")

        qualified_leads = self.state.qualified_leads()
        start_clients = len(self.state.clients)

        if len(qualified_leads) < 3:
            # Need to generate more leads first
            actions = self.marketing.run_cycle(self.state)

        # Sales focuses on existing qualified leads
        actions = self.sales.run_cycle(self.state)

        self.state.update_mrr()
        new_clients = len(self.state.clients) - start_clients
        mrr_gained = self.state.mrr - (self.state.mrr - sum(
            c.monthly_value for c in self.state.clients[-new_clients:]
        ) if new_clients > 0 else 0)

        return {
            "leads_nurtured": len(qualified_leads),
            "new_clients": new_clients,
            "mrr_gained": mrr_gained,
        }

    def get_funnel_report(self) -> str:
        """Generate a detailed funnel report."""
        funnel = MetricsCalculator.calculate_conversion_funnel(self.state.leads)
        total_leads = len(self.state.leads)
        total_clients = len(self.state.clients)

        lines = [
            "=== Revenue Funnel Report ===",
            f"Total Leads: {total_leads}",
            f"Active Clients: {total_clients}",
            f"Current MRR: ${self.state.mrr:,.0f} / ${self.state.target_mrr:,.0f}",
            "",
            "Funnel Stages:",
        ]
        for stage, count in funnel.items():
            pct = count / total_leads * 100 if total_leads > 0 else 0
            bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
            lines.append(f"  {stage:<15} {count:>3} | {bar} {pct:.1f}%")

        if self._sprint_history:
            lines.append("\nSprint History:")
            for sprint in self._sprint_history[-3:]:
                achieved = "✓" if sprint["target_achieved"] else "✗"
                lines.append(
                    f"  {achieved} {sprint['sprint_label']}: "
                    f"+${sprint['actual_mrr_increase']:,.0f} MRR | "
                    f"{sprint['new_clients_acquired']} clients | "
                    f"{sprint['conversion_rate']:.1f}% conv."
                )
        return "\n".join(lines)

    def _estimate_avg_client_value(self) -> float:
        """Estimate average new client MRR based on pipeline."""
        if not self.state.leads:
            return 500.0  # Default to Tier 1
        pipeline = [
            l for l in self.state.leads
            if l.stage in ("qualified", "proposal", "negotiation")
        ]
        if not pipeline:
            return 500.0
        return sum(l.value_potential for l in pipeline) / len(pipeline)
