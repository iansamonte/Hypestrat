"""
Finance Tool — Budget tracker, expense management, MRR updates, and projections.
"""

from __future__ import annotations
import random
from typing import Dict, Any, List

from src.models.company_state import CompanyState
from src.models.metrics import MetricsCalculator


EXPENSE_CATEGORIES = {
    "infrastructure": 50.0,   # hosting, tools
    "marketing": 0.0,         # paid ads, content
    "sales_tools": 30.0,      # CRM, outreach tools
    "ai_api": 20.0,           # Anthropic API costs
    "operations": 10.0,       # misc
}


class FinanceTool:
    """Tool for financial tracking, budget allocation, and forecasting."""

    def __init__(self, state: CompanyState):
        self.state = state
        self._expense_log: List[Dict[str, Any]] = []
        self._budget_allocations: Dict[str, float] = dict(EXPENSE_CATEGORIES)
        self._monthly_snapshots: List[Dict[str, Any]] = []

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "finance_update_mrr",
                "description": "Recalculate and update MRR from current client list.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "finance_track_expense",
                "description": "Record a business expense.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category: infrastructure, marketing, sales_tools, ai_api, operations, other",
                        },
                        "amount": {"type": "number", "description": "Expense amount in USD"},
                        "description": {"type": "string", "description": "What was the expense for"},
                    },
                    "required": ["category", "amount", "description"],
                },
            },
            {
                "name": "finance_calculate_runway",
                "description": "Calculate how many months of runway remain at current burn rate.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "finance_allocate_budget",
                "description": "Allocate monthly budget across categories.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Budget category to allocate to",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Monthly allocation amount in USD",
                        },
                    },
                    "required": ["category", "amount"],
                },
            },
            {
                "name": "finance_forecast_growth",
                "description": "Generate a financial forecast for the next N months.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "months": {
                            "type": "integer",
                            "description": "Number of months to forecast",
                        },
                        "new_clients_per_month": {
                            "type": "number",
                            "description": "Estimated new clients per month",
                        },
                        "avg_tier": {
                            "type": "number",
                            "description": "Average tier of new clients (1.0-3.0)",
                        },
                    },
                    "required": ["months"],
                },
            },
            {
                "name": "finance_get_pl_summary",
                "description": "Get a Profit & Loss summary for the current period.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        dispatch = {
            "finance_update_mrr": self._update_mrr,
            "finance_track_expense": self._track_expense,
            "finance_calculate_runway": self._calculate_runway,
            "finance_allocate_budget": self._allocate_budget,
            "finance_forecast_growth": self._forecast_growth,
            "finance_get_pl_summary": self._get_pl_summary,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"Unknown finance tool: {tool_name}"
        return fn(tool_input)

    def _update_mrr(self, _inp: Dict[str, Any]) -> str:
        old_mrr = self.state.mrr
        self.state.update_mrr()
        new_mrr = self.state.mrr
        change = new_mrr - old_mrr
        sign = "+" if change >= 0 else ""
        return (
            f"MRR updated: ${old_mrr:,.2f} -> ${new_mrr:,.2f} ({sign}${change:,.2f})\n"
            f"  Progress to target: {self.state.progress_to_target():.1f}%\n"
            f"  ARR: ${MetricsCalculator.calculate_arr(new_mrr):,.2f}"
        )

    def _track_expense(self, inp: Dict[str, Any]) -> str:
        category = inp.get("category", "other")
        amount = float(inp.get("amount", 0))
        desc = inp.get("description", "")

        if amount > self.state.capital:
            return f"INSUFFICIENT CAPITAL: Need ${amount:.2f}, have ${self.state.capital:.2f}"

        self.state.capital -= amount
        self.state.monthly_expenses += amount

        record = {
            "category": category,
            "amount": amount,
            "description": desc,
            "capital_remaining": self.state.capital,
        }
        self._expense_log.append(record)
        self._budget_allocations[category] = self._budget_allocations.get(category, 0) + amount

        return (
            f"Expense recorded: ${amount:.2f} [{category}]\n"
            f"  Description: {desc}\n"
            f"  Capital remaining: ${self.state.capital:,.2f}\n"
            f"  Total expenses this month: ${self.state.monthly_expenses:,.2f}"
        )

    def _calculate_runway(self, _inp: Dict[str, Any]) -> str:
        state = self.state
        state.update_mrr()
        net = state.mrr - state.monthly_expenses
        burn = max(0, -net)

        runway = MetricsCalculator.calculate_runway(state.capital, burn)

        lines = [
            "=== Runway Analysis ===",
            f"Capital:          ${state.capital:>10,.2f}",
            f"MRR:              ${state.mrr:>10,.2f}",
            f"Monthly Expenses: ${state.monthly_expenses:>10,.2f}",
            f"Net Monthly:      ${net:>+10,.2f}",
        ]
        if runway == float('inf'):
            lines.append("Runway:           Profitable — infinite runway!")
        else:
            lines.append(f"Runway:           {runway:.1f} months")
            if runway < 3:
                lines.append("  ⚠ WARNING: Critical low runway — raise revenue urgently!")
            elif runway < 6:
                lines.append("  ⚠ Caution: Less than 6 months runway")
        return "\n".join(lines)

    def _allocate_budget(self, inp: Dict[str, Any]) -> str:
        category = inp.get("category", "other")
        amount = float(inp.get("amount", 0))
        old = self._budget_allocations.get(category, 0)
        self._budget_allocations[category] = amount
        return (
            f"Budget allocated: {category} = ${amount:.2f}/mo (was ${old:.2f}/mo)\n"
            f"  Total budgeted: ${sum(self._budget_allocations.values()):.2f}/mo"
        )

    def _forecast_growth(self, inp: Dict[str, Any]) -> str:
        months = min(inp.get("months", 6), 24)
        new_clients = inp.get("new_clients_per_month", 2.5)
        avg_tier = inp.get("avg_tier", 1.5)

        tier_prices = {1: 500.0, 2: 1000.0, 3: 2000.0}
        # Weighted average price based on avg_tier
        avg_price = 500 + (avg_tier - 1) * 750

        state = self.state
        cap = state.capital
        mrr = state.mrr
        expenses = max(state.monthly_expenses, sum(EXPENSE_CATEGORIES.values()))

        lines = [
            f"=== {months}-Month Financial Forecast ===",
            f"Starting: MRR ${mrr:,.0f} | Capital ${cap:,.0f}",
            "",
            f"{'Month':<8} {'MRR':>10} {'Clients':>8} {'Capital':>10} {'Net CF':>10}",
            "-" * 50,
        ]
        client_count = len(state.clients)
        # Assume ~5% monthly churn
        churn_rate = 0.05

        for m in range(1, months + 1):
            churned = int(client_count * churn_rate)
            client_count = client_count - churned + new_clients
            mrr = client_count * avg_price
            net = mrr - expenses
            cap = cap + net
            lines.append(
                f"{state.month + m:<8} ${mrr:>9,.0f} {int(client_count):>8} ${cap:>9,.0f} ${net:>+9,.0f}"
            )
            if mrr >= state.target_mrr:
                lines.append(f"  ** TARGET ${state.target_mrr:,.0f} REACHED at Month {state.month + m}! **")
                break

        return "\n".join(lines)

    def _get_pl_summary(self, _inp: Dict[str, Any]) -> str:
        state = self.state
        state.update_mrr()
        revenue = state.mrr
        expenses = state.monthly_expenses
        gross_profit = revenue - expenses
        margin = (gross_profit / revenue * 100) if revenue > 0 else 0

        # Expense breakdown
        exp_lines = []
        for cat, amt in self._budget_allocations.items():
            if amt > 0:
                exp_lines.append(f"  {cat:<20} ${amt:>8,.2f}")

        lines = [
            "=== Profit & Loss Summary ===",
            f"Revenue (MRR):       ${revenue:>10,.2f}",
            f"Total Expenses:      ${expenses:>10,.2f}",
            f"Gross Profit:        ${gross_profit:>+10,.2f}",
            f"Gross Margin:        {margin:>9.1f}%",
            "",
            "Expense Breakdown:",
        ] + exp_lines + [
            "",
            f"Capital Balance:     ${state.capital:>10,.2f}",
        ]
        return "\n".join(lines)
