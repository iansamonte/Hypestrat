"""
Analytics Tool — Track MRR, churn, CAC, LTV, and growth metrics.
"""

from __future__ import annotations
import random
from typing import Dict, Any, List, Optional

from src.models.company_state import CompanyState
from src.models.metrics import MetricsCalculator


class AnalyticsTool:
    """Tool for querying business analytics and KPIs."""

    def __init__(self, state: CompanyState):
        self.state = state
        self._historical_mrr: List[float] = []

    def snapshot_mrr(self) -> None:
        """Save current MRR to historical record."""
        self._historical_mrr.append(self.state.mrr)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "analytics_get_mrr_report",
                "description": "Get a full MRR report including growth, churn, and progress toward target.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "analytics_get_full_metrics",
                "description": "Get all key business metrics: MRR, ARR, CAC, LTV, runway, pipeline, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "analytics_get_growth_projection",
                "description": "Project future MRR growth over the next N months.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "months": {
                            "type": "integer",
                            "description": "Number of months to project (1-24)",
                        },
                        "growth_rate": {
                            "type": "number",
                            "description": "Monthly growth rate % to use. If not provided, uses historical average.",
                        },
                    },
                    "required": ["months"],
                },
            },
            {
                "name": "analytics_get_funnel_metrics",
                "description": "Get lead funnel conversion metrics and stage breakdown.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "analytics_get_client_health_report",
                "description": "Get a health report on all clients including churn risk and upsell potential.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        dispatch = {
            "analytics_get_mrr_report": self._get_mrr_report,
            "analytics_get_full_metrics": self._get_full_metrics,
            "analytics_get_growth_projection": self._get_growth_projection,
            "analytics_get_funnel_metrics": self._get_funnel_metrics,
            "analytics_get_client_health_report": self._get_client_health_report,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"Unknown analytics tool: {tool_name}"
        return fn(tool_input)

    def _get_mrr_report(self, _inp: Dict[str, Any]) -> str:
        state = self.state
        state.update_mrr()
        mrr = state.mrr
        target = state.target_mrr
        progress = state.progress_to_target()
        remaining = target - mrr

        prev_mrr = self._historical_mrr[-1] if self._historical_mrr else 0.0
        growth = MetricsCalculator.calculate_growth_rate(mrr, prev_mrr)

        # MRR breakdown by tier
        tier_mrr = {}
        for tier, name in CompanyState.TIER_NAMES.items():
            tier_clients = [c for c in state.clients if c.tier == tier]
            tier_mrr[name] = sum(c.monthly_value for c in tier_clients)

        lines = [
            "=== MRR Report ===",
            f"Current MRR:     ${mrr:>10,.2f}",
            f"Target MRR:      ${target:>10,.2f}",
            f"Progress:        {progress:>9.1f}%",
            f"Remaining:       ${remaining:>10,.2f}",
            f"MoM Growth:      {growth:>9.1f}%",
            "",
            "MRR by Tier:",
        ]
        for name, value in tier_mrr.items():
            lines.append(f"  {name}: ${value:,.0f}")
        lines.append(f"\nTotal Clients: {len(state.clients)}")
        return "\n".join(lines)

    def _get_full_metrics(self, _inp: Dict[str, Any]) -> str:
        metrics = MetricsCalculator.full_metrics(self.state)
        lines = ["=== Full Business Metrics ==="]
        for key, val in metrics.items():
            if isinstance(val, dict):
                lines.append(f"{key}:")
                for k, v in val.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(val, float):
                lines.append(f"{key}: {val:,.2f}")
            else:
                lines.append(f"{key}: {val}")
        return "\n".join(lines)

    def _get_growth_projection(self, inp: Dict[str, Any]) -> str:
        months = min(inp.get("months", 6), 24)
        state = self.state

        # Determine growth rate
        if "growth_rate" in inp:
            rate = inp["growth_rate"]
        elif len(self._historical_mrr) >= 2:
            prev = self._historical_mrr[-1]
            curr = state.mrr
            rate = MetricsCalculator.calculate_growth_rate(curr, prev)
            rate = max(5.0, rate)  # floor at 5% for projections
        else:
            rate = 30.0  # default optimistic early-stage growth

        projections = MetricsCalculator.project_mrr_growth(state.mrr, rate, months)
        months_to_target = MetricsCalculator.months_to_target(state.mrr, state.target_mrr, rate)

        lines = [
            f"=== MRR Growth Projection ({rate:.1f}% monthly) ===",
            f"Starting MRR: ${state.mrr:,.0f}",
        ]
        for i, proj in enumerate(projections, 1):
            marker = " << TARGET" if proj >= state.target_mrr and (i == 1 or projections[i-2] < state.target_mrr) else ""
            lines.append(f"  Month {state.month + i}: ${proj:>10,.0f}{marker}")
        if months_to_target is not None:
            lines.append(f"\nEstimated months to ${state.target_mrr:,.0f} target: {months_to_target}")
        else:
            lines.append("\nNeed positive MRR to project target date.")
        return "\n".join(lines)

    def _get_funnel_metrics(self, _inp: Dict[str, Any]) -> str:
        funnel = MetricsCalculator.calculate_conversion_funnel(self.state.leads)
        total = len(self.state.leads)
        pipeline_val = self.state.pipeline_value()

        lines = ["=== Lead Funnel Metrics ==="]
        for stage, count in funnel.items():
            pct = (count / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 5)
            lines.append(f"  {stage:<15} {count:>3} ({pct:>5.1f}%) {bar}")
        lines.append(f"\nTotal Leads: {total}")
        lines.append(f"Pipeline Value: ${pipeline_val:,.0f}/mo")

        # Conversion rates
        prospects = funnel.get("prospect", 0) + funnel.get("qualified", 0)
        won = funnel.get("closed_won", 0)
        if prospects > 0:
            conv_rate = won / (prospects + won) * 100
            lines.append(f"Overall Conversion Rate: {conv_rate:.1f}%")
        return "\n".join(lines)

    def _get_client_health_report(self, _inp: Dict[str, Any]) -> str:
        clients = self.state.clients
        if not clients:
            return "No active clients to report on."

        at_risk = [c for c in clients if c.churn_risk > 0.3]
        healthy = [c for c in clients if c.health_score >= 80]
        upsell_candidates = [c for c in clients if c.health_score >= 85 and c.tier < 3]

        lines = [
            "=== Client Health Report ===",
            f"Total Clients: {len(clients)}",
            f"Healthy (score >= 80): {len(healthy)}",
            f"At-Risk (churn risk > 30%): {len(at_risk)}",
            f"Upsell Candidates: {len(upsell_candidates)}",
            "",
        ]
        if at_risk:
            lines.append("At-Risk Clients:")
            for c in at_risk:
                lines.append(f"  {c.name} @ {c.company} | Risk: {c.churn_risk*100:.0f}% | Health: {c.health_score:.0f}")
        if upsell_candidates:
            lines.append("\nUpsell Candidates:")
            for c in upsell_candidates:
                next_tier_price = CompanyState.TIER_PRICES.get(c.tier + 1, c.monthly_value)
                upsell_value = next_tier_price - c.monthly_value
                lines.append(
                    f"  {c.name} @ {c.company} | Current: {c.tier_name()} | "
                    f"Upsell value: +${upsell_value:.0f}/mo"
                )
        return "\n".join(lines)
