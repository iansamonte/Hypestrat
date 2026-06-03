"""
Product Agent — Service delivery, automation building, and client setup.
"""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.models.company_state import Action, CompanyState

logger = logging.getLogger(__name__)


# Simulated deliverable templates
AUTOMATION_BLUEPRINTS = {
    1: {
        "name": "AI Content Engine Setup",
        "components": [
            "Blog post generation workflow (weekly cadence)",
            "LinkedIn content calendar (3 posts/week)",
            "Email newsletter automation (bi-weekly)",
            "SEO keyword research dashboard",
            "Content performance tracking",
        ],
        "setup_time_days": 3,
        "value_delivered": "10-15 hours/week saved on content",
    },
    2: {
        "name": "AI Sales Autopilot Setup",
        "components": [
            "ICP (Ideal Customer Profile) definition",
            "Lead sourcing automation (50 leads/week)",
            "Personalized cold email sequences (5-touch)",
            "LinkedIn outreach automation",
            "CRM integration and pipeline automation",
            "Meeting booking and follow-up sequences",
        ],
        "setup_time_days": 5,
        "value_delivered": "3x more outreach, 2x response rates",
    },
    3: {
        "name": "AI Business Suite Setup",
        "components": [
            "Full content engine (Tier 1 included)",
            "Full sales autopilot (Tier 2 included)",
            "AI customer service chatbot",
            "Financial reporting dashboard",
            "Competitive intelligence monitoring",
            "Weekly AI business strategy reports",
            "Custom workflow automation (2 per month)",
        ],
        "setup_time_days": 10,
        "value_delivered": "20+ hours/week saved, full AI stack",
    },
}


class ProductAgent(BaseAgent):
    """
    The Product Agent ensures excellent service delivery:
    - Sets up client automations
    - Builds service deliverables
    - Monitors delivery status
    - Identifies product improvements
    """

    def __init__(self, model: str = BaseAgent.DEFAULT_MODEL) -> None:
        super().__init__(
            name="Product",
            role="Chief Product Officer",
            model=model,
        )
        self._deliverables: Dict[str, Dict] = {}

    def get_system_prompt(self) -> str:
        return (
            "You are the CPO of IAN.AI Ventures, responsible for service delivery and product quality.\n\n"
            "Your products:\n"
            "  - Tier 1: AI Content Engine ($500/mo) — 3-day setup\n"
            "  - Tier 2: AI Sales Autopilot ($1,000/mo) — 5-day setup\n"
            "  - Tier 3: AI Business Suite ($2,000/mo) — 10-day setup\n\n"
            "Your responsibilities:\n"
            "1. Set up automations for all new clients.\n"
            "2. Build and deliver service components on time.\n"
            "3. Check delivery status for all active client setups.\n"
            "4. Identify product improvements based on client feedback.\n"
            "5. Document what's been delivered to maintain quality standards.\n\n"
            "Focus on: fast delivery, high quality, and making clients successful quickly. "
            "Start with setup_client_automation for any new clients, then check delivery status."
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "setup_client_automation",
                "description": "Set up automation systems for a new client.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string", "description": "Client ID"},
                        "tier": {
                            "type": "integer",
                            "description": "Service tier (1, 2, or 3)",
                        },
                        "customizations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific customizations for this client",
                        },
                    },
                    "required": ["client_id", "tier"],
                },
            },
            {
                "name": "build_service_deliverable",
                "description": "Build a specific service deliverable for a client.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string"},
                        "deliverable_type": {
                            "type": "string",
                            "description": "Type: content_workflow, email_sequence, lead_gen_system, reporting_dashboard",
                        },
                        "specifications": {
                            "type": "string",
                            "description": "Specific requirements",
                        },
                    },
                    "required": ["client_id", "deliverable_type"],
                },
            },
            {
                "name": "check_delivery_status",
                "description": "Check the delivery status of all client setups.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {
                            "type": "string",
                            "description": "Specific client ID, or leave empty for all",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "get_new_clients_to_onboard",
                "description": "Get list of new clients that need product setup.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "log_product_improvement",
                "description": "Log a product improvement idea or feature request.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "improvement": {
                            "type": "string",
                            "description": "Description of the improvement",
                        },
                        "priority": {
                            "type": "string",
                            "description": "Priority: high, medium, low",
                        },
                        "source": {
                            "type": "string",
                            "description": "Source: client_feedback, internal, market_research",
                        },
                    },
                    "required": ["improvement"],
                },
            },
        ]

    def handle_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any], state: CompanyState
    ) -> str:
        if tool_name == "setup_client_automation":
            return self._setup_automation(tool_input, state)
        elif tool_name == "build_service_deliverable":
            return self._build_deliverable(tool_input, state)
        elif tool_name == "check_delivery_status":
            return self._check_status(tool_input, state)
        elif tool_name == "get_new_clients_to_onboard":
            return self._get_new_clients(state)
        elif tool_name == "log_product_improvement":
            return self._log_improvement(tool_input)
        return f"Unknown product tool: {tool_name}"

    def _setup_automation(self, inp: Dict[str, Any], state: CompanyState) -> str:
        client_id = inp.get("client_id", "")
        tier = inp.get("tier", 1)
        customizations = inp.get("customizations", [])

        # Find client
        client = next((c for c in state.clients if c.id == client_id), None)
        if not client:
            # Try partial match
            client = next((c for c in state.clients if client_id in c.id or client_id in c.name), None)
        if not client:
            return f"Client {client_id} not found. Available: {[c.id for c in state.clients[:5]]}"

        blueprint = AUTOMATION_BLUEPRINTS.get(tier, AUTOMATION_BLUEPRINTS[1])

        # Record deliverable
        self._deliverables[client.id] = {
            "client_name": client.name,
            "tier": tier,
            "blueprint": blueprint["name"],
            "components": blueprint["components"],
            "customizations": customizations,
            "status": "in_progress",
            "setup_days": blueprint["setup_time_days"],
            "days_remaining": blueprint["setup_time_days"],
        }

        comp_list = "\n".join(f"    - {c}" for c in blueprint["components"])
        custom_list = "\n".join(f"    - {c}" for c in customizations) if customizations else "    (none)"

        return (
            f"Automation setup initiated for {client.name} @ {client.company}\n"
            f"  Tier: {tier} — {blueprint['name']}\n"
            f"  Components:\n{comp_list}\n"
            f"  Customizations:\n{custom_list}\n"
            f"  Setup time: {blueprint['setup_time_days']} days\n"
            f"  Value delivered: {blueprint['value_delivered']}"
        )

    def _build_deliverable(self, inp: Dict[str, Any], state: CompanyState) -> str:
        client_id = inp.get("client_id", "")
        d_type = inp.get("deliverable_type", "content_workflow")
        specs = inp.get("specifications", "Standard implementation")

        quality_score = random.randint(85, 98)
        delivery_time = random.randint(1, 3)

        return (
            f"Deliverable built: {d_type} for client {client_id}\n"
            f"  Specifications: {specs[:100]}\n"
            f"  Quality score: {quality_score}/100\n"
            f"  Delivery time: {delivery_time} days\n"
            f"  Status: COMPLETE — ready for client review"
        )

    def _check_status(self, inp: Dict[str, Any], state: CompanyState) -> str:
        client_id = inp.get("client_id", "")

        if client_id:
            d = self._deliverables.get(client_id)
            if not d:
                return f"No active deliverables for client {client_id}."
            return (
                f"Delivery Status — {d['client_name']}:\n"
                f"  Blueprint: {d['blueprint']}\n"
                f"  Status: {d['status']}\n"
                f"  Days remaining: {d['days_remaining']}"
            )

        if not self._deliverables:
            return "No active client setups. All clients are fully onboarded."

        lines = [f"Delivery Status ({len(self._deliverables)} active):"]
        for cid, d in self._deliverables.items():
            status_icon = "✓" if d["status"] == "complete" else "⏳"
            lines.append(
                f"  {status_icon} {d['client_name']} | {d['blueprint']} | "
                f"Status: {d['status']} | Days left: {d['days_remaining']}"
            )
        return "\n".join(lines)

    def _get_new_clients(self, state: CompanyState) -> str:
        new_clients = [c for c in state.clients if c.months_active == 0]
        if not new_clients:
            return "No new clients to onboard this cycle."
        lines = [f"New clients to onboard ({len(new_clients)}):"]
        for c in new_clients:
            lines.append(
                f"  [{c.id}] {c.name} @ {c.company} | Tier: {c.tier} — {c.tier_name()} | ${c.monthly_value:.0f}/mo"
            )
        return "\n".join(lines)

    def _log_improvement(self, inp: Dict[str, Any]) -> str:
        improvement = inp.get("improvement", "")
        priority = inp.get("priority", "medium")
        source = inp.get("source", "internal")
        return (
            f"Product improvement logged:\n"
            f"  Priority: {priority.upper()}\n"
            f"  Source: {source}\n"
            f"  Description: {improvement}"
        )

    def run_cycle(self, state: CompanyState) -> List[Action]:
        """Run one product cycle."""
        self.log_cycle_start(state)

        new_clients = [c for c in state.clients if c.months_active == 0]
        all_clients = state.clients

        context = {
            **state.to_summary_dict(),
            "new_clients_to_setup": [c.to_dict() for c in new_clients],
            "all_clients": [c.to_dict() for c in all_clients[:10]],
        }

        task = (
            f"Month {state.month} Product Delivery Cycle. "
            f"Total clients: {len(all_clients)}. "
            f"New clients needing setup: {len(new_clients)}. "
            f"Set up automations for all new clients, check delivery status, "
            f"and build any outstanding deliverables. "
            f"Use the client IDs from the context to set up each new client."
        )

        result = self.run(task=task, context=context, state=state)

        state.agent_reports["Product"] = {
            "month": state.month,
            "active_setups": len(self._deliverables),
            "new_clients_onboarded": len(new_clients),
            "output_summary": result.output[:300],
        }

        self.log_cycle_end(result.actions_taken)
        return result.actions_taken
