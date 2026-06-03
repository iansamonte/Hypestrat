"""
Customer Success Agent — Onboarding, retention, health checks, and upsell identification.
"""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.models.company_state import Action, CompanyState, Client

logger = logging.getLogger(__name__)


ONBOARDING_STEPS = [
    "Welcome call and goal-setting session",
    "Account setup and system access provisioning",
    "Kickoff questionnaire completed",
    "Initial automation configured",
    "First deliverable delivered",
    "30-day check-in scheduled",
]

CHURN_RISK_SIGNALS = [
    "Low engagement with deliverables",
    "Missed check-in call",
    "Slow response to communications",
    "Billing issue or payment delay",
    "Expressed doubts about ROI",
    "Key champion left the company",
]


class CustomerSuccessAgent(BaseAgent):
    """
    The Customer Success Agent maximizes client retention and expansion:
    - Onboards new clients smoothly
    - Runs regular health checks
    - Identifies upsell opportunities
    - Handles at-risk clients proactively
    - Reduces churn through engagement
    """

    def __init__(self, model: str = BaseAgent.DEFAULT_MODEL) -> None:
        super().__init__(
            name="CustomerSuccess",
            role="VP of Customer Success",
            model=model,
        )
        self._onboarding_records: Dict[str, Dict] = {}
        self._health_history: Dict[str, List[float]] = {}

    def get_system_prompt(self) -> str:
        return (
            "You are the VP of Customer Success at IAN.AI Ventures.\n\n"
            "Your mission: Keep clients happy, reduce churn, and grow revenue from existing clients.\n\n"
            "Key metrics you own:\n"
            "  - Net Revenue Retention (NRR) > 110%\n"
            "  - Monthly churn rate < 3%\n"
            "  - Average health score > 80/100\n"
            "  - Upsell rate: 15% of clients upgrade per quarter\n\n"
            "Your responsibilities:\n"
            "1. Onboard all new clients within 48 hours.\n"
            "2. Run health checks on all clients monthly.\n"
            "3. Identify and pursue upsell opportunities for healthy clients.\n"
            "4. Handle at-risk clients with personalized intervention.\n"
            "5. Track and report NRR and churn metrics.\n\n"
            "Health score factors:\n"
            "  - Product usage and engagement (40%)\n"
            "  - Results achieved vs. expectations (35%)\n"
            "  - Communication responsiveness (25%)\n\n"
            "Always: onboard new clients first, then run health_check on all clients, "
            "then identify_upsell for healthy ones, then handle_churn_risk for at-risk ones."
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "onboard_client",
                "description": "Run the onboarding process for a new client.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string", "description": "Client ID"},
                        "onboarding_tier": {
                            "type": "integer",
                            "description": "Service tier (1, 2, or 3)",
                        },
                    },
                    "required": ["client_id"],
                },
            },
            {
                "name": "health_check",
                "description": "Perform a health check on a client and update their health score.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {
                            "type": "string",
                            "description": "Client ID, or 'all' to check all clients",
                        },
                    },
                    "required": ["client_id"],
                },
            },
            {
                "name": "identify_upsell",
                "description": "Identify upsell opportunities among healthy clients.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "min_health_score": {
                            "type": "number",
                            "description": "Minimum health score to consider for upsell (default 80)",
                        },
                        "min_months_active": {
                            "type": "integer",
                            "description": "Minimum months active before considering upsell",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "handle_churn_risk",
                "description": "Take action to retain an at-risk client.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string"},
                        "intervention_type": {
                            "type": "string",
                            "description": "Type: executive_call, bonus_deliverable, discount_offer, feature_unlock",
                        },
                    },
                    "required": ["client_id", "intervention_type"],
                },
            },
            {
                "name": "execute_upsell",
                "description": "Execute an upsell for a client to a higher tier.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string"},
                        "new_tier": {
                            "type": "integer",
                            "description": "New tier to upgrade to (2 or 3)",
                        },
                    },
                    "required": ["client_id", "new_tier"],
                },
            },
            {
                "name": "get_cs_metrics",
                "description": "Get customer success metrics: NRR, churn, health scores, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def handle_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any], state: CompanyState
    ) -> str:
        if tool_name == "onboard_client":
            return self._onboard(tool_input, state)
        elif tool_name == "health_check":
            return self._health_check(tool_input, state)
        elif tool_name == "identify_upsell":
            return self._identify_upsell(tool_input, state)
        elif tool_name == "handle_churn_risk":
            return self._handle_churn(tool_input, state)
        elif tool_name == "execute_upsell":
            return self._execute_upsell(tool_input, state)
        elif tool_name == "get_cs_metrics":
            return self._get_metrics(state)
        return f"Unknown CS tool: {tool_name}"

    def _onboard(self, inp: Dict[str, Any], state: CompanyState) -> str:
        client_id = inp.get("client_id", "")
        tier = inp.get("onboarding_tier", 1)

        client = next((c for c in state.clients if c.id == client_id or client_id in c.name), None)
        if not client:
            return f"Client {client_id} not found."

        steps_completed = ONBOARDING_STEPS[: min(tier + 3, len(ONBOARDING_STEPS))]

        self._onboarding_records[client.id] = {
            "client_name": client.name,
            "tier": tier,
            "steps_completed": steps_completed,
            "status": "complete",
            "nps_score": random.randint(8, 10),
        }

        # Boost health score for freshly onboarded clients
        client.health_score = min(100, client.health_score + 10)
        client.churn_risk = max(0.02, client.churn_risk - 0.05)

        step_list = "\n".join(f"  ✓ {s}" for s in steps_completed)
        return (
            f"Onboarding complete for {client.name} @ {client.company}\n"
            f"  Tier: {tier} — {client.tier_name()}\n"
            f"Steps completed:\n{step_list}\n"
            f"  NPS Score: {self._onboarding_records[client.id]['nps_score']}/10\n"
            f"  Health Score: {client.health_score:.0f} | Churn Risk: {client.churn_risk*100:.0f}%"
        )

    def _health_check(self, inp: Dict[str, Any], state: CompanyState) -> str:
        client_id = inp.get("client_id", "all")

        if client_id == "all":
            clients = state.clients
        else:
            clients = [c for c in state.clients if c.id == client_id or client_id in c.name]

        if not clients:
            return "No clients to check."

        results = []
        for client in clients:
            # Simulate health score changes based on tenure and random factors
            delta = random.uniform(-8, 12)
            if client.months_active > 3:
                delta += 2  # longer tenure = more stable
            client.health_score = max(20, min(100, client.health_score + delta))

            # Update churn risk based on health
            if client.health_score >= 85:
                client.churn_risk = max(0.02, client.churn_risk - 0.03)
            elif client.health_score < 60:
                client.churn_risk = min(0.60, client.churn_risk + 0.10)
            else:
                client.churn_risk = 0.10 + (80 - client.health_score) * 0.01

            status = "HEALTHY" if client.health_score >= 75 else ("AT RISK" if client.health_score < 55 else "NEUTRAL")
            results.append(
                f"  {client.name} @ {client.company}: "
                f"Health {client.health_score:.0f}/100 [{status}] | "
                f"Churn risk: {client.churn_risk*100:.0f}%"
            )

            # Store history
            if client.id not in self._health_history:
                self._health_history[client.id] = []
            self._health_history[client.id].append(client.health_score)

        return f"Health checks complete ({len(clients)} clients):\n" + "\n".join(results)

    def _identify_upsell(self, inp: Dict[str, Any], state: CompanyState) -> str:
        min_health = inp.get("min_health_score", 80)
        min_months = inp.get("min_months_active", 2)

        candidates = [
            c for c in state.clients
            if c.health_score >= min_health
            and c.months_active >= min_months
            and c.tier < 3
        ]

        if not candidates:
            return f"No upsell candidates meeting criteria (health >= {min_health}, months >= {min_months})."

        tier_prices = {1: 500, 2: 1000, 3: 2000}
        lines = [f"Upsell candidates ({len(candidates)}):"]
        for c in candidates:
            next_tier = c.tier + 1
            next_price = tier_prices.get(next_tier, 2000)
            uplift = next_price - c.monthly_value
            tier_names = {1: "AI Content Engine", 2: "AI Sales Autopilot", 3: "AI Business Suite"}
            lines.append(
                f"  [{c.id}] {c.name} @ {c.company}\n"
                f"    Current: {c.tier_name()} (${c.monthly_value:.0f}/mo)\n"
                f"    Recommend: Tier {next_tier} — {tier_names[next_tier]} (${next_price:.0f}/mo)\n"
                f"    MRR uplift: +${uplift:.0f}/mo | Health: {c.health_score:.0f}"
            )
        return "\n".join(lines)

    def _handle_churn(self, inp: Dict[str, Any], state: CompanyState) -> str:
        client_id = inp.get("client_id", "")
        intervention = inp.get("intervention_type", "executive_call")

        client = next((c for c in state.clients if c.id == client_id or client_id in c.name), None)
        if not client:
            return f"Client {client_id} not found."

        # Intervention effects
        effects = {
            "executive_call": ("15-point health boost + relationship restored", 15, -0.15),
            "bonus_deliverable": ("Extra deliverable delivered + 10-point boost", 10, -0.10),
            "discount_offer": ("10% discount offered + 8-point boost", 8, -0.12),
            "feature_unlock": ("New feature unlocked + 12-point boost", 12, -0.10),
        }
        desc, health_boost, risk_reduction = effects.get(intervention, ("Standard check-in", 5, -0.05))

        client.health_score = min(100, client.health_score + health_boost)
        client.churn_risk = max(0.03, client.churn_risk + risk_reduction)

        return (
            f"Churn intervention executed for {client.name} @ {client.company}\n"
            f"  Intervention: {intervention}\n"
            f"  Effect: {desc}\n"
            f"  New Health Score: {client.health_score:.0f}/100\n"
            f"  New Churn Risk: {client.churn_risk*100:.0f}%"
        )

    def _execute_upsell(self, inp: Dict[str, Any], state: CompanyState) -> str:
        client_id = inp.get("client_id", "")
        new_tier = inp.get("new_tier", 2)

        client = next((c for c in state.clients if c.id == client_id or client_id in c.name), None)
        if not client:
            return f"Client {client_id} not found."

        tier_prices = {1: 500.0, 2: 1000.0, 3: 2000.0}
        tier_names = {1: "AI Content Engine", 2: "AI Sales Autopilot", 3: "AI Business Suite"}
        new_price = tier_prices.get(new_tier, 1000.0)

        # Simulate upsell acceptance
        accept_prob = 0.40 + (client.health_score - 80) / 100
        accepted = random.random() < max(0.25, min(0.75, accept_prob))

        if accepted:
            old_tier = client.tier
            old_price = client.monthly_value
            client.tier = new_tier
            client.monthly_value = new_price
            state.update_mrr()
            return (
                f"UPSELL SUCCESS! {client.name} @ {client.company}\n"
                f"  Upgraded: Tier {old_tier} (${old_price:.0f}/mo) → Tier {new_tier} — {tier_names[new_tier]} (${new_price:.0f}/mo)\n"
                f"  MRR increase: +${new_price - old_price:.0f}/mo\n"
                f"  New total MRR: ${state.mrr:,.0f}"
            )
        else:
            return (
                f"Upsell attempt for {client.name} — not accepted yet.\n"
                f"  Follow up in 30 days. Continue delivering value."
            )

    def _get_metrics(self, state: CompanyState) -> str:
        clients = state.clients
        if not clients:
            return "No clients yet — no CS metrics available."

        avg_health = sum(c.health_score for c in clients) / len(clients)
        at_risk = [c for c in clients if c.churn_risk > 0.3]
        healthy = [c for c in clients if c.health_score >= 80]
        tier_breakdown = {t: len([c for c in clients if c.tier == t]) for t in [1, 2, 3]}

        return (
            f"=== Customer Success Metrics ===\n"
            f"Total Clients:       {len(clients)}\n"
            f"Avg Health Score:    {avg_health:.1f}/100\n"
            f"Healthy (≥80):       {len(healthy)} ({len(healthy)/len(clients)*100:.0f}%)\n"
            f"At Risk (>30% churn): {len(at_risk)} ({len(at_risk)/len(clients)*100:.0f}%)\n"
            f"Tier Breakdown:      T1={tier_breakdown[1]}, T2={tier_breakdown[2]}, T3={tier_breakdown[3]}\n"
            f"Current MRR:         ${state.mrr:,.0f}"
        )

    def run_cycle(self, state: CompanyState) -> List[Action]:
        """Run one customer success cycle."""
        self.log_cycle_start(state)

        new_clients = [c for c in state.clients if c.months_active == 0]
        at_risk_clients = [c for c in state.clients if c.churn_risk > 0.25]
        upsell_candidates = [
            c for c in state.clients
            if c.health_score >= 80 and c.months_active >= 2 and c.tier < 3
        ]

        # Increment months_active for all clients
        for c in state.clients:
            c.months_active += 1

        context = {
            **state.to_summary_dict(),
            "new_clients": [c.to_dict() for c in new_clients],
            "at_risk_clients": [c.to_dict() for c in at_risk_clients[:5]],
            "upsell_candidates": [c.to_dict() for c in upsell_candidates[:5]],
            "all_client_ids": [c.id for c in state.clients[:15]],
        }

        task = (
            f"Month {state.month} Customer Success Cycle. "
            f"Total clients: {len(state.clients)}. "
            f"New clients to onboard: {len(new_clients)}. "
            f"At-risk clients: {len(at_risk_clients)}. "
            f"Upsell candidates: {len(upsell_candidates)}. "
            f"Onboard new clients, run health checks on all, "
            f"handle at-risk clients, and pursue upsell opportunities. "
            f"Use client IDs from context. Goal: reduce churn and grow MRR from existing clients."
        )

        result = self.run(task=task, context=context, state=state)

        state.agent_reports["CustomerSuccess"] = {
            "month": state.month,
            "total_clients": len(state.clients),
            "at_risk_count": len(at_risk_clients),
            "upsell_candidates": len(upsell_candidates),
            "output_summary": result.output[:300],
        }

        self.log_cycle_end(result.actions_taken)
        return result.actions_taken
