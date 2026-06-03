"""
Company Orchestrator — Top-level orchestration for daily, weekly, and monthly cycles.
Coordinates all agents and reports company progress.
"""

from __future__ import annotations
import logging
import time
from typing import List, Dict, Any, Optional

from src.agents.ceo_agent import CEOAgent
from src.agents.marketing_agent import MarketingAgent
from src.agents.sales_agent import SalesAgent
from src.agents.product_agent import ProductAgent
from src.agents.finance_agent import FinanceAgent
from src.agents.customer_success_agent import CustomerSuccessAgent
from src.models.company_state import Action, CompanyState
from src.models.metrics import MetricsCalculator, MonthlySnapshot

logger = logging.getLogger(__name__)


class CompanyOrchestrator:
    """
    Top-level orchestrator for IAN.AI Ventures.

    Runs coordinated cycles across all agents:
    - Daily: CEO + Marketing + Sales + Finance
    - Weekly: All agents + strategy review
    - Monthly: Full cycle + comprehensive reporting + capital allocation
    """

    def __init__(self, state: CompanyState, model: str = "claude-sonnet-4-6") -> None:
        self.state = state
        self.model = model
        self.snapshots: List[MonthlySnapshot] = []

        # Initialize agents (lazy — don't create Anthropic clients until needed)
        self._ceo: Optional[CEOAgent] = None
        self._marketing: Optional[MarketingAgent] = None
        self._sales: Optional[SalesAgent] = None
        self._product: Optional[ProductAgent] = None
        self._finance: Optional[FinanceAgent] = None
        self._cs: Optional[CustomerSuccessAgent] = None

    # ------------------------------------------------------------------
    # Agent accessors (lazy init)
    # ------------------------------------------------------------------

    @property
    def ceo(self) -> CEOAgent:
        if self._ceo is None:
            self._ceo = CEOAgent(model=self.model)
        return self._ceo

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

    @property
    def product(self) -> ProductAgent:
        if self._product is None:
            self._product = ProductAgent(model=self.model)
        return self._product

    @property
    def finance(self) -> FinanceAgent:
        if self._finance is None:
            self._finance = FinanceAgent(model=self.model)
        return self._finance

    @property
    def cs(self) -> CustomerSuccessAgent:
        if self._cs is None:
            self._cs = CustomerSuccessAgent(model=self.model)
        return self._cs

    # ------------------------------------------------------------------
    # Cycle runners
    # ------------------------------------------------------------------

    def run_daily_cycle(self) -> Dict[str, List[Action]]:
        """
        Daily cycle: CEO reviews → Marketing runs → Sales runs → Finance updates.
        Returns dict of agent_name -> actions taken.
        """
        logger.info(f"=== Daily Cycle — Month {self.state.month} ===")
        all_actions: Dict[str, List[Action]] = {}

        # 1. CEO reviews metrics and sets priorities
        logger.info("Running CEO cycle...")
        ceo_actions = self.ceo.run_cycle(self.state)
        all_actions["CEO"] = ceo_actions

        # 2. Marketing generates leads
        logger.info("Running Marketing cycle...")
        mkt_actions = self.marketing.run_cycle(self.state)
        all_actions["Marketing"] = mkt_actions

        # 3. Sales converts leads
        logger.info("Running Sales cycle...")
        sales_actions = self.sales.run_cycle(self.state)
        all_actions["Sales"] = sales_actions

        # 4. Finance updates numbers
        logger.info("Running Finance cycle...")
        fin_actions = self.finance.run_cycle(self.state)
        all_actions["Finance"] = fin_actions

        total_actions = sum(len(v) for v in all_actions.values())
        logger.info(f"Daily cycle complete — {total_actions} total actions")
        return all_actions

    def run_weekly_cycle(self) -> Dict[str, List[Action]]:
        """
        Weekly cycle: Full daily cycle + Product + Customer Success + CEO strategy review.
        """
        logger.info(f"=== Weekly Cycle — Month {self.state.month} ===")

        # Run daily cycle first
        all_actions = self.run_daily_cycle()

        # 5. Product delivers for clients
        logger.info("Running Product cycle...")
        prod_actions = self.product.run_cycle(self.state)
        all_actions["Product"] = prod_actions

        # 6. Customer success manages retention
        logger.info("Running Customer Success cycle...")
        cs_actions = self.cs.run_cycle(self.state)
        all_actions["CustomerSuccess"] = cs_actions

        # 7. CEO final strategy review
        logger.info("Running CEO strategy review...")
        ceo_review = self.ceo.run_cycle(self.state)
        all_actions["CEO_Review"] = ceo_review

        total_actions = sum(len(v) for v in all_actions.values())
        logger.info(f"Weekly cycle complete — {total_actions} total actions")
        return all_actions

    def run_monthly_cycle(self) -> Dict[str, Any]:
        """
        Monthly cycle: Full weekly cycle + comprehensive reporting + capital allocation.
        Returns a monthly report dict.
        """
        logger.info(f"=== Monthly Cycle — Month {self.state.month} ===")
        month_start_mrr = self.state.mrr
        month_start_clients = len(self.state.clients)

        # Run full weekly cycle
        all_actions = self.run_weekly_cycle()

        # Update state for new month
        self.state.update_mrr()

        # Build monthly snapshot
        new_clients = len(self.state.clients) - month_start_clients
        growth_rate = MetricsCalculator.calculate_growth_rate(self.state.mrr, month_start_mrr)

        marketing_spend = self.state.monthly_expenses * 0.4
        sales_spend = self.state.monthly_expenses * 0.2
        cac = MetricsCalculator.calculate_cac(marketing_spend, sales_spend, max(new_clients, 1))
        avg_val = self.state.mrr / max(len(self.state.clients), 1)
        ltv = MetricsCalculator.calculate_ltv(avg_val, 18.0)
        burn = max(0, self.state.monthly_expenses - self.state.mrr)
        runway = MetricsCalculator.calculate_runway(self.state.capital, burn)

        snapshot = MonthlySnapshot(
            month=self.state.month,
            mrr=self.state.mrr,
            capital=self.state.capital,
            clients=len(self.state.clients),
            new_clients=new_clients,
            churned_clients=0,
            leads=len(self.state.leads),
            qualified_leads=len(self.state.qualified_leads()),
            expenses=self.state.monthly_expenses,
            net_profit=self.state.mrr - self.state.monthly_expenses,
            growth_rate=growth_rate,
            cac=cac,
            ltv=ltv,
            runway=min(runway, 999),
        )
        self.snapshots.append(snapshot)

        # Advance to next month
        self.state.month += 1
        # Reset monthly expenses for next month
        prev_expenses = self.state.monthly_expenses
        self.state.monthly_expenses = 0.0

        # Apply base infrastructure costs for new month
        base_costs = 110.0 + len(self.state.clients) * 5
        self.state.monthly_expenses = base_costs

        # Add revenue to capital (net of expenses)
        net_revenue = snapshot.mrr - prev_expenses
        if net_revenue > 0:
            self.state.capital += net_revenue
        else:
            self.state.capital = max(0, self.state.capital + net_revenue)

        report = {
            "month": snapshot.month,
            "mrr": snapshot.mrr,
            "mrr_growth": growth_rate,
            "clients": snapshot.clients,
            "new_clients": new_clients,
            "capital": snapshot.capital,
            "runway": snapshot.runway,
            "expenses": snapshot.expenses,
            "net_profit": snapshot.net_profit,
            "cac": snapshot.cac,
            "ltv": snapshot.ltv,
            "progress_pct": self.state.progress_to_target(),
            "total_actions": sum(len(v) for v in all_actions.values()),
            "agents_run": list(all_actions.keys()),
            "priorities": self.state.priorities,
        }

        logger.info(
            f"Month {snapshot.month} complete | MRR: ${snapshot.mrr:,.0f} | "
            f"Clients: {snapshot.clients} | Growth: {growth_rate:.1f}%"
        )
        return report

    def get_monthly_history(self) -> List[Dict[str, Any]]:
        """Return historical monthly snapshots as dicts."""
        return [
            {
                "month": s.month,
                "mrr": s.mrr,
                "clients": s.clients,
                "capital": s.capital,
                "growth_rate": s.growth_rate,
                "net_profit": s.net_profit,
            }
            for s in self.snapshots
        ]

    def get_current_summary(self) -> Dict[str, Any]:
        """Return a current company summary."""
        return {
            **self.state.to_summary_dict(),
            "snapshots": len(self.snapshots),
        }
