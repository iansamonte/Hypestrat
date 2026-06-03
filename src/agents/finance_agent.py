"""
Finance Agent — P&L tracking, MRR forecasting, budget allocation, and financial health.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.models.company_state import Action, CompanyState
from src.tools.finance_tool import FinanceTool
from src.tools.analytics_tool import AnalyticsTool

logger = logging.getLogger(__name__)


class FinanceAgent(BaseAgent):
    """
    The Finance Agent maintains financial health:
    - Tracks all income and expenses
    - Updates MRR from client activity
    - Forecasts growth trajectory
    - Manages budget allocation
    - Calculates runway and alerts on risks
    """

    def __init__(self, model: str = BaseAgent.DEFAULT_MODEL) -> None:
        super().__init__(
            name="Finance",
            role="Chief Financial Officer",
            model=model,
        )
        self._finance_tool: FinanceTool = None
        self._analytics_tool: AnalyticsTool = None

    def get_system_prompt(self) -> str:
        return (
            "You are the CFO of IAN.AI Ventures, managing financial health from $1,000 capital to $30k MRR.\n\n"
            "Financial structure:\n"
            "  - Starting capital: $1,000\n"
            "  - Target MRR: $30,000\n"
            "  - Fixed costs: ~$110/mo (infrastructure, tools, API)\n"
            "  - Variable costs: grow with clients (delivery, support)\n\n"
            "Your responsibilities:\n"
            "1. Update MRR from current client base.\n"
            "2. Track all expenses and ensure we don't overspend capital.\n"
            "3. Calculate runway and flag risks if < 3 months.\n"
            "4. Allocate budget wisely: prioritize revenue-generating activities.\n"
            "5. Forecast growth trajectory toward $30k MRR.\n"
            "6. Generate P&L summary for CEO review.\n\n"
            "Financial principles:\n"
            "- Every dollar spent must have a clear path to return\n"
            "- Marketing spend: max 20% of MRR or $200 (whichever is higher early on)\n"
            "- Maintain at least 3 months runway at all times\n"
            "- Reinvest revenue into growth once profitable\n\n"
            "Always: update_mrr first, then calculate_runway, then review expenses, then forecast."
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "update_mrr",
                "description": "Recalculate MRR from current clients and update financial records.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "track_expenses",
                "description": "Review and track monthly operational expenses.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "month": {"type": "integer", "description": "Month number"},
                        "include_infrastructure": {"type": "boolean"},
                        "include_marketing": {"type": "boolean"},
                    },
                    "required": [],
                },
            },
            {
                "name": "calculate_runway",
                "description": "Calculate remaining runway and flag financial risks.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "forecast_growth",
                "description": "Forecast MRR growth and time to target based on current trajectory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "months_ahead": {
                            "type": "integer",
                            "description": "How many months to forecast",
                        },
                        "new_clients_per_month": {
                            "type": "number",
                            "description": "Expected new clients per month",
                        },
                    },
                    "required": ["months_ahead"],
                },
            },
            {
                "name": "get_pl_summary",
                "description": "Get the current Profit & Loss summary.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "set_monthly_budget",
                "description": "Set the monthly budget allocation for each department.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "marketing_budget": {"type": "number"},
                        "sales_budget": {"type": "number"},
                        "operations_budget": {"type": "number"},
                        "infrastructure_budget": {"type": "number"},
                    },
                    "required": [],
                },
            },
        ]

    def handle_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any], state: CompanyState
    ) -> str:
        if self._finance_tool is None:
            self._finance_tool = FinanceTool(state)
        if self._analytics_tool is None:
            self._analytics_tool = AnalyticsTool(state)

        if tool_name == "update_mrr":
            return self._finance_tool.execute("finance_update_mrr", {})
        elif tool_name == "track_expenses":
            return self._track_expenses(tool_input, state)
        elif tool_name == "calculate_runway":
            return self._finance_tool.execute("finance_calculate_runway", {})
        elif tool_name == "forecast_growth":
            return self._finance_tool.execute(
                "finance_forecast_growth",
                {
                    "months": tool_input.get("months_ahead", 6),
                    "new_clients_per_month": tool_input.get("new_clients_per_month", 2),
                },
            )
        elif tool_name == "get_pl_summary":
            return self._finance_tool.execute("finance_get_pl_summary", {})
        elif tool_name == "set_monthly_budget":
            return self._set_budget(tool_input, state)
        return f"Unknown finance tool: {tool_name}"

    def _track_expenses(self, inp: Dict[str, Any], state: CompanyState) -> str:
        month = inp.get("month", state.month)

        # Record standard monthly expenses
        base_expenses = {
            "infrastructure": 50.0,
            "sales_tools": 30.0,
            "ai_api": 20.0,
            "operations": 10.0,
        }

        # Scale with client count
        client_scale = max(1, len(state.clients))
        variable_cost = client_scale * 5  # $5/client for delivery

        total = sum(base_expenses.values()) + variable_cost
        state.monthly_expenses = total

        lines = [
            f"=== Monthly Expenses — Month {month} ===",
            f"Fixed Costs:",
        ]
        for cat, amt in base_expenses.items():
            lines.append(f"  {cat:<20} ${amt:>6.2f}")
        lines.append(f"  variable_delivery   ${variable_cost:>6.2f}  ({client_scale} clients × $5)")
        lines.append(f"  {'TOTAL':<20} ${total:>6.2f}")
        lines.append(f"\nMRR: ${state.mrr:,.2f} | Net: ${state.mrr - total:+,.2f}")
        return "\n".join(lines)

    def _set_budget(self, inp: Dict[str, Any], state: CompanyState) -> str:
        allocations = {
            "marketing": inp.get("marketing_budget", 0),
            "sales": inp.get("sales_budget", 0),
            "operations": inp.get("operations_budget", 0),
            "infrastructure": inp.get("infrastructure_budget", 50),
        }
        total = sum(allocations.values())
        lines = ["Monthly budget set:"]
        for dept, amt in allocations.items():
            if amt > 0:
                lines.append(f"  {dept}: ${amt:.2f}")
        lines.append(f"  TOTAL: ${total:.2f}/mo")
        if total > state.capital * 0.5:
            lines.append(f"  ⚠ Warning: Budget ${total:.2f} is >50% of capital ${state.capital:.2f}")
        return "\n".join(lines)

    def run_cycle(self, state: CompanyState) -> List[Action]:
        """Run one finance cycle."""
        self.log_cycle_start(state)
        self._finance_tool = FinanceTool(state)
        self._analytics_tool = AnalyticsTool(state)

        context = {
            **state.to_summary_dict(),
            "expense_categories": {
                "infrastructure": 50.0,
                "sales_tools": 30.0,
                "ai_api": 20.0,
                "operations": 10.0,
            },
        }

        task = (
            f"Month {state.month} Finance Cycle. "
            f"Capital: ${state.capital:,.2f}. MRR: ${state.mrr:,.2f}. "
            f"Update MRR, track monthly expenses, calculate runway, "
            f"and provide a financial forecast. "
            f"Flag any risks and recommend budget allocations for next month."
        )

        result = self.run(task=task, context=context, state=state)

        state.agent_reports["Finance"] = {
            "month": state.month,
            "mrr": state.mrr,
            "capital": state.capital,
            "expenses": state.monthly_expenses,
            "output_summary": result.output[:300],
        }

        self.log_cycle_end(result.actions_taken)
        return result.actions_taken
