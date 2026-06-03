"""
Metrics calculations for IAN.AI Ventures.
MRR, churn, CAC, LTV, growth rate, and runway.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .company_state import CompanyState, Client


@dataclass
class MonthlySnapshot:
    """A monthly snapshot of company metrics."""
    month: int
    mrr: float
    capital: float
    clients: int
    new_clients: int
    churned_clients: int
    leads: int
    qualified_leads: int
    expenses: float
    net_profit: float
    growth_rate: float
    cac: float
    ltv: float
    runway: float


class MetricsCalculator:
    """Calculates key business metrics for IAN.AI Ventures."""

    @staticmethod
    def calculate_mrr(clients: List["Client"]) -> float:
        """Calculate Monthly Recurring Revenue from client list."""
        return sum(c.monthly_value for c in clients)

    @staticmethod
    def calculate_arr(mrr: float) -> float:
        """Annual Recurring Revenue."""
        return mrr * 12

    @staticmethod
    def calculate_growth_rate(current_mrr: float, previous_mrr: float) -> float:
        """Month-over-month MRR growth rate as percentage."""
        if previous_mrr == 0:
            return 100.0 if current_mrr > 0 else 0.0
        return ((current_mrr - previous_mrr) / previous_mrr) * 100

    @staticmethod
    def calculate_churn_rate(churned_clients: int, total_clients: int) -> float:
        """Monthly churn rate as percentage."""
        if total_clients == 0:
            return 0.0
        return (churned_clients / total_clients) * 100

    @staticmethod
    def calculate_cac(marketing_spend: float, sales_spend: float, new_clients: int) -> float:
        """Customer Acquisition Cost."""
        if new_clients == 0:
            return 0.0
        return (marketing_spend + sales_spend) / new_clients

    @staticmethod
    def calculate_ltv(avg_monthly_value: float, avg_months_retained: float) -> float:
        """Customer Lifetime Value."""
        return avg_monthly_value * avg_months_retained

    @staticmethod
    def calculate_ltv_cac_ratio(ltv: float, cac: float) -> float:
        """LTV:CAC ratio — should be > 3 for healthy business."""
        if cac == 0:
            return 0.0
        return ltv / cac

    @staticmethod
    def calculate_runway(capital: float, monthly_burn: float) -> float:
        """Months of runway at current burn rate."""
        if monthly_burn <= 0:
            return float('inf')
        return capital / monthly_burn

    @staticmethod
    def calculate_net_revenue_retention(
        starting_mrr: float,
        expansion_mrr: float,
        contraction_mrr: float,
        churned_mrr: float
    ) -> float:
        """Net Revenue Retention (NRR) — % of MRR retained and expanded."""
        if starting_mrr == 0:
            return 100.0
        return ((starting_mrr + expansion_mrr - contraction_mrr - churned_mrr) / starting_mrr) * 100

    @staticmethod
    def calculate_payback_period(cac: float, avg_monthly_margin: float) -> float:
        """Months to recover CAC from gross margin."""
        if avg_monthly_margin == 0:
            return float('inf')
        return cac / avg_monthly_margin

    @staticmethod
    def project_mrr_growth(
        current_mrr: float,
        monthly_growth_rate: float,
        months: int
    ) -> List[float]:
        """Project MRR for N months at a given growth rate."""
        projections = []
        mrr = current_mrr
        for _ in range(months):
            mrr = mrr * (1 + monthly_growth_rate / 100)
            projections.append(round(mrr, 2))
        return projections

    @staticmethod
    def months_to_target(
        current_mrr: float,
        target_mrr: float,
        monthly_growth_rate: float
    ) -> Optional[int]:
        """Estimate months to reach target MRR at current growth rate."""
        if current_mrr >= target_mrr:
            return 0
        if monthly_growth_rate <= 0:
            return None
        if current_mrr == 0:
            return None
        months = math.log(target_mrr / current_mrr) / math.log(1 + monthly_growth_rate / 100)
        return math.ceil(months)

    @staticmethod
    def calculate_conversion_funnel(leads: List[Any]) -> Dict[str, int]:
        """Calculate lead funnel metrics."""
        stages = ["prospect", "qualified", "proposal", "negotiation", "closed_won", "closed_lost"]
        funnel = {stage: 0 for stage in stages}
        for lead in leads:
            stage = getattr(lead, "stage", "prospect")
            if stage in funnel:
                funnel[stage] += 1
        return funnel

    @classmethod
    def full_metrics(cls, state: "CompanyState") -> Dict[str, Any]:
        """Calculate all key metrics for a CompanyState."""
        mrr = state.calculate_mrr()
        arr = cls.calculate_arr(mrr)

        avg_monthly_value = (
            mrr / len(state.clients) if state.clients else 0.0
        )
        ltv = cls.calculate_ltv(avg_monthly_value, 18.0)  # assume 18-month avg retention

        marketing_spend = state.monthly_expenses * 0.4
        sales_spend = state.monthly_expenses * 0.2
        new_clients_this_month = len([
            c for c in state.clients if c.months_active <= 1
        ])
        cac = cls.calculate_cac(marketing_spend, sales_spend, max(new_clients_this_month, 1))

        net_monthly = mrr - state.monthly_expenses
        runway = cls.calculate_runway(state.capital, max(0, -net_monthly))

        funnel = cls.calculate_conversion_funnel(state.leads)

        return {
            "mrr": round(mrr, 2),
            "arr": round(arr, 2),
            "mrr_progress_pct": round(state.progress_to_target(), 1),
            "total_clients": len(state.clients),
            "avg_revenue_per_client": round(avg_monthly_value, 2),
            "cac": round(cac, 2),
            "ltv": round(ltv, 2),
            "ltv_cac_ratio": round(cls.calculate_ltv_cac_ratio(ltv, cac), 2),
            "pipeline_value": round(state.pipeline_value(), 2),
            "total_leads": len(state.leads),
            "lead_funnel": funnel,
            "monthly_expenses": round(state.monthly_expenses, 2),
            "net_monthly_cash_flow": round(net_monthly, 2),
            "capital": round(state.capital, 2),
            "runway_months": round(runway, 1) if runway != float('inf') else 999,
            "tier_breakdown": {
                tier_name: len([c for c in state.clients if c.tier == tier])
                for tier, tier_name in CompanyState.TIER_NAMES.items()
            } if hasattr(state, 'clients') else {},
        }


# Avoid circular import
from .company_state import CompanyState  # noqa: E402
