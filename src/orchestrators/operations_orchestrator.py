"""
Operations Orchestrator — Coordinates product + customer success for service delivery.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from src.agents.product_agent import ProductAgent
from src.agents.customer_success_agent import CustomerSuccessAgent
from src.models.company_state import Action, CompanyState
from src.models.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class OperationsOrchestrator:
    """
    Coordinates product delivery and customer success to ensure
    client retention, quality delivery, and revenue expansion.
    """

    def __init__(self, state: CompanyState, model: str = "claude-sonnet-4-6") -> None:
        self.state = state
        self.model = model
        self._product: Optional[ProductAgent] = None
        self._cs: Optional[CustomerSuccessAgent] = None
        self._delivery_history: List[Dict[str, Any]] = []

    @property
    def product(self) -> ProductAgent:
        if self._product is None:
            self._product = ProductAgent(model=self.model)
        return self._product

    @property
    def cs(self) -> CustomerSuccessAgent:
        if self._cs is None:
            self._cs = CustomerSuccessAgent(model=self.model)
        return self._cs

    def run_delivery_cycle(self) -> Dict[str, Any]:
        """
        Run a complete delivery cycle:
        1. Product sets up new clients
        2. Customer success onboards and health-checks
        3. Track retention metrics

        Returns a delivery report.
        """
        logger.info(f"=== Delivery Cycle — Month {self.state.month} ===")

        start_mrr = self.state.mrr
        start_clients = len(self.state.clients)
        at_risk_before = len([c for c in self.state.clients if c.churn_risk > 0.3])

        all_actions: Dict[str, List[Action]] = {}

        # Phase 1: Product delivery
        logger.info("Phase 1: Product delivery...")
        prod_actions = self.product.run_cycle(self.state)
        all_actions["Product"] = prod_actions

        # Phase 2: Customer success
        logger.info("Phase 2: Customer success...")
        cs_actions = self.cs.run_cycle(self.state)
        all_actions["CustomerSuccess"] = cs_actions

        # Update MRR after potential upsells
        self.state.update_mrr()

        # Calculate results
        end_mrr = self.state.mrr
        end_clients = len(self.state.clients)
        at_risk_after = len([c for c in self.state.clients if c.churn_risk > 0.3])
        avg_health = (
            sum(c.health_score for c in self.state.clients) / len(self.state.clients)
            if self.state.clients else 0
        )

        report = {
            "month": self.state.month,
            "clients_start": start_clients,
            "clients_end": end_clients,
            "mrr_start": start_mrr,
            "mrr_end": end_mrr,
            "mrr_change": end_mrr - start_mrr,
            "at_risk_before": at_risk_before,
            "at_risk_after": at_risk_after,
            "avg_health_score": round(avg_health, 1),
            "total_actions": sum(len(v) for v in all_actions.values()),
        }

        self._delivery_history.append(report)
        logger.info(
            f"Delivery cycle complete: MRR ${end_mrr:,.0f} | "
            f"Health avg: {avg_health:.0f} | "
            f"At risk: {at_risk_after}"
        )
        return report

    def run_retention_campaign(self) -> Dict[str, Any]:
        """
        Run a focused retention campaign for at-risk clients.
        """
        at_risk = [c for c in self.state.clients if c.churn_risk > 0.25]
        if not at_risk:
            return {"message": "No at-risk clients. All healthy!", "clients_saved": 0}

        logger.info(f"Running retention campaign for {len(at_risk)} at-risk clients...")
        start_mrr = self.state.mrr

        # CS agent focuses on at-risk clients
        cs_actions = self.cs.run_cycle(self.state)

        self.state.update_mrr()
        end_mrr = self.state.mrr
        still_at_risk = len([c for c in self.state.clients if c.churn_risk > 0.25])
        saved = len(at_risk) - still_at_risk

        return {
            "at_risk_before": len(at_risk),
            "at_risk_after": still_at_risk,
            "clients_saved": saved,
            "mrr_protected": start_mrr,
            "mrr_after": end_mrr,
            "actions_taken": len(cs_actions),
        }

    def run_upsell_campaign(self) -> Dict[str, Any]:
        """
        Run a focused upsell campaign for high-health clients.
        """
        candidates = [
            c for c in self.state.clients
            if c.health_score >= 80 and c.months_active >= 2 and c.tier < 3
        ]
        if not candidates:
            return {"message": "No upsell candidates available.", "revenue_gained": 0}

        logger.info(f"Running upsell campaign for {len(candidates)} candidates...")
        start_mrr = self.state.mrr

        cs_actions = self.cs.run_cycle(self.state)

        self.state.update_mrr()
        revenue_gained = self.state.mrr - start_mrr

        return {
            "candidates": len(candidates),
            "mrr_gained": revenue_gained,
            "new_mrr": self.state.mrr,
            "actions_taken": len(cs_actions),
        }

    def get_operations_report(self) -> str:
        """Generate an operations health report."""
        clients = self.state.clients
        if not clients:
            return "No active clients. Operations standing by."

        healthy = [c for c in clients if c.health_score >= 80]
        at_risk = [c for c in clients if c.churn_risk > 0.3]
        avg_health = sum(c.health_score for c in clients) / len(clients)

        lines = [
            "=== Operations Report ===",
            f"Total Clients:    {len(clients)}",
            f"Avg Health Score: {avg_health:.1f}/100",
            f"Healthy (≥80):    {len(healthy)} ({len(healthy)/len(clients)*100:.0f}%)",
            f"At Risk (>30%):   {len(at_risk)} ({len(at_risk)/len(clients)*100:.0f}%)",
            f"Current MRR:      ${self.state.mrr:,.0f}",
        ]

        if at_risk:
            lines.append("\nAt-Risk Clients (immediate action needed):")
            for c in at_risk[:3]:
                lines.append(
                    f"  {c.name} @ {c.company} | "
                    f"Health: {c.health_score:.0f} | Risk: {c.churn_risk*100:.0f}%"
                )

        if self._delivery_history:
            last = self._delivery_history[-1]
            lines.append(f"\nLast Delivery Cycle (Month {last['month']}):")
            lines.append(f"  MRR change: ${last['mrr_change']:+,.0f}")
            lines.append(f"  At-risk change: {last['at_risk_before']} → {last['at_risk_after']}")

        return "\n".join(lines)
