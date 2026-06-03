"""
CEO Agent — Strategic decisions, priority setting, metric reviews, and budget approval.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.models.company_state import Action, AgentResult, CompanyState
from src.models.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class CEOAgent(BaseAgent):
    """
    The CEO Agent drives company strategy. Each cycle it:
    1. Reviews all key metrics
    2. Sets or updates company priorities
    3. Approves/adjusts budget allocations
    4. Reviews reports from other agents
    5. Issues strategic directives
    """

    def __init__(self, model: str = BaseAgent.DEFAULT_MODEL) -> None:
        super().__init__(
            name="CEO",
            role="Chief Executive Officer",
            model=model,
        )
        self._priorities: List[str] = []

    def get_system_prompt(self) -> str:
        return (
            "You are the CEO of IAN.AI Ventures, a fully autonomous AI company. "
            "Your company sells AI automation services to SMBs with three tiers:\n"
            "  - Tier 1: AI Content Engine — $500/month (target: 20 clients)\n"
            "  - Tier 2: AI Sales Autopilot — $1,000/month (target: 12 clients)\n"
            "  - Tier 3: AI Business Suite — $2,000/month (target: 4 clients)\n"
            "Goal: $30,000 MRR from 36 clients. Starting capital: $1,000.\n\n"
            "Your responsibilities:\n"
            "1. Review company metrics and identify the most critical bottleneck.\n"
            "2. Set clear priorities for the team (marketing, sales, product, CS, finance).\n"
            "3. Approve budget allocations wisely given limited capital.\n"
            "4. Make strategic decisions on go-to-market, pricing, and focus areas.\n"
            "5. Identify risks and mitigation strategies.\n\n"
            "Be decisive, data-driven, and concise. Use tools to gather data, then issue directives. "
            "Always call get_company_metrics first, then set_priorities with your conclusions."
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_company_metrics",
                "description": "Retrieve all current company KPIs: MRR, clients, pipeline, runway, growth rate.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "set_priorities",
                "description": "Set the company's top priorities for the current period.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "priorities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ordered list of priority directives (most important first)",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief explanation for these priorities",
                        },
                    },
                    "required": ["priorities"],
                },
            },
            {
                "name": "approve_budget",
                "description": "Approve or adjust budget allocation for a department or campaign.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "department": {
                            "type": "string",
                            "description": "Department: marketing, sales, product, operations",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Monthly budget amount in USD",
                        },
                        "justification": {
                            "type": "string",
                            "description": "Why this budget is approved",
                        },
                    },
                    "required": ["department", "amount"],
                },
            },
            {
                "name": "review_agent_reports",
                "description": "Review recent reports from all agents and identify action items.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "focus_area": {
                            "type": "string",
                            "description": "Area to focus review on: revenue, operations, or all",
                        }
                    },
                    "required": [],
                },
            },
        ]

    def handle_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any], state: CompanyState
    ) -> str:
        if tool_name == "get_company_metrics":
            return self._get_metrics(state)
        elif tool_name == "set_priorities":
            return self._set_priorities(tool_input, state)
        elif tool_name == "approve_budget":
            return self._approve_budget(tool_input, state)
        elif tool_name == "review_agent_reports":
            return self._review_reports(tool_input, state)
        return f"Unknown CEO tool: {tool_name}"

    def _get_metrics(self, state: CompanyState) -> str:
        metrics = MetricsCalculator.full_metrics(state)
        summary = state.to_summary_dict()

        mrr_gap = state.target_mrr - state.mrr
        clients_needed = 36 - len(state.clients)
        burn = max(0, state.monthly_expenses - state.mrr)
        runway = state.capital / burn if burn > 0 else float('inf')

        lines = [
            "=== CEO Dashboard — Key Metrics ===",
            f"Company:         {state.name}",
            f"Month:           {state.month}",
            f"Capital:         ${state.capital:,.2f}",
            f"MRR:             ${state.mrr:,.2f} / ${state.target_mrr:,.0f} ({summary['progress_pct']:.1f}%)",
            f"MRR Gap:         ${mrr_gap:,.2f} remaining to target",
            f"Clients:         {len(state.clients)} / 36 target",
            f"Clients Needed:  {clients_needed} more to reach target",
            f"Leads:           {len(state.leads)} total | {summary['qualified_leads']} qualified",
            f"Pipeline Value:  ${summary['pipeline_value']:,.2f}/mo",
            f"Monthly Burn:    ${state.monthly_expenses:,.2f}",
            f"Runway:          {'Profitable' if runway == float('inf') else f'{runway:.1f} months'}",
            "",
            "Tier Breakdown (clients):",
        ]
        for tier, name in CompanyState.TIER_NAMES.items():
            count = len([c for c in state.clients if c.tier == tier])
            target = CompanyState.TIER_TARGETS[tier]
            tier_mrr = sum(c.monthly_value for c in state.clients if c.tier == tier)
            lines.append(f"  {name}: {count}/{target} clients | ${tier_mrr:,.0f}/mo")

        if state.priorities:
            lines.append("\nCurrent Priorities:")
            for i, p in enumerate(state.priorities, 1):
                lines.append(f"  {i}. {p}")

        return "\n".join(lines)

    def _set_priorities(self, inp: Dict[str, Any], state: CompanyState) -> str:
        priorities = inp.get("priorities", [])
        rationale = inp.get("rationale", "")
        state.priorities = priorities
        self._priorities = priorities
        lines = [f"Priorities updated ({len(priorities)} items):"]
        for i, p in enumerate(priorities, 1):
            lines.append(f"  {i}. {p}")
        if rationale:
            lines.append(f"\nRationale: {rationale}")
        return "\n".join(lines)

    def _approve_budget(self, inp: Dict[str, Any], state: CompanyState) -> str:
        dept = inp.get("department", "unknown")
        amount = float(inp.get("amount", 0))
        justification = inp.get("justification", "")
        return (
            f"Budget approved: {dept} = ${amount:.2f}/mo\n"
            f"  Capital remaining: ${state.capital:,.2f}\n"
            f"  Justification: {justification}"
        )

    def _review_reports(self, inp: Dict[str, Any], state: CompanyState) -> str:
        focus = inp.get("focus_area", "all")
        reports = state.agent_reports
        if not reports:
            return "No agent reports available yet. This is the first cycle."

        lines = [f"=== Agent Reports Review (focus: {focus}) ==="]
        for agent, report in reports.items():
            if focus == "all" or focus.lower() in agent.lower():
                lines.append(f"\n[{agent}]")
                if isinstance(report, dict):
                    for k, v in report.items():
                        lines.append(f"  {k}: {str(v)[:100]}")
                else:
                    lines.append(f"  {str(report)[:200]}")
        return "\n".join(lines) if len(lines) > 1 else "No matching reports found."

    def run_cycle(self, state: CompanyState) -> List[Action]:
        """Run one CEO strategy cycle."""
        self.log_cycle_start(state)
        context = state.to_summary_dict()

        task = (
            f"Month {state.month} CEO Strategy Review. "
            f"Current MRR: ${state.mrr:,.0f} / ${state.target_mrr:,.0f} target. "
            f"Review metrics, set this month's priorities, and provide strategic directives. "
            f"We have ${state.capital:,.0f} in capital. Focus on closing the MRR gap efficiently."
        )

        result = self.run(task=task, context=context, state=state)

        # Store report for other agents
        state.agent_reports["CEO"] = {
            "month": state.month,
            "priorities": state.priorities,
            "output_summary": result.output[:300],
        }

        self.log_cycle_end(result.actions_taken)
        return result.actions_taken
