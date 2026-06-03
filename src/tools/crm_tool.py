"""
CRM Tool — In-memory CRM for leads, contacts, and pipeline management.
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, date
from typing import Dict, Any, List, Optional

from src.models.company_state import Lead, Client, CompanyState


# Fake SMB company pool for simulation
COMPANY_POOL = [
    ("Sarah Mitchell", "BrightEdge Marketing", "marketing"),
    ("James Thornton", "NovaTech Solutions", "technology"),
    ("Laura Reyes", "Summit Consulting Group", "consulting"),
    ("Kevin Park", "Horizon Retail Co.", "retail"),
    ("Danielle Ford", "BluePeak Digital", "marketing"),
    ("Marcus Webb", "CoreLogic Systems", "technology"),
    ("Priya Sharma", "Ascend Advisory", "consulting"),
    ("Ethan Clarke", "PinnacleFit Studios", "fitness"),
    ("Olivia Grant", "Velocity E-Commerce", "ecommerce"),
    ("Tyler Hayes", "GreenLeaf Foods", "food"),
    ("Nina Rodriguez", "CloudPath IT Services", "technology"),
    ("Brandon Kim", "PulseMedia Agency", "marketing"),
    ("Sophia Turner", "NextWave HR Solutions", "hr"),
    ("Aaron Price", "Redwood Capital Partners", "finance"),
    ("Jessica Nguyen", "TrueNorth Real Estate", "real_estate"),
    ("Derek Morrison", "SilverLine Healthcare", "healthcare"),
    ("Amanda Walsh", "TechSpark Education", "education"),
    ("Ryan Bishop", "Orbit Software Labs", "technology"),
    ("Chloe Evans", "FreshMint Beauty", "beauty"),
    ("Nathan Cole", "IronBridge Manufacturing", "manufacturing"),
]


class CRMTool:
    """In-memory CRM tool for managing leads and clients."""

    def __init__(self, state: CompanyState):
        self.state = state

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return Anthropic tool definitions for CRM operations."""
        return [
            {
                "name": "crm_list_leads",
                "description": "List all leads in the CRM pipeline with their stages and scores.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "stage_filter": {
                            "type": "string",
                            "description": "Filter by stage: prospect, qualified, proposal, negotiation, closed_won, closed_lost, or 'all'",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of leads to return (default 20)",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "crm_add_lead",
                "description": "Add a new lead to the CRM pipeline.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Contact name"},
                        "company": {"type": "string", "description": "Company name"},
                        "industry": {"type": "string", "description": "Industry sector"},
                        "value_potential": {
                            "type": "number",
                            "description": "Estimated monthly value ($500, $1000, or $2000)",
                        },
                        "source": {
                            "type": "string",
                            "description": "Lead source: organic, paid, referral, outbound",
                        },
                    },
                    "required": ["name", "company"],
                },
            },
            {
                "name": "crm_update_lead_stage",
                "description": "Update a lead's pipeline stage.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID"},
                        "new_stage": {
                            "type": "string",
                            "description": "New stage: prospect, qualified, proposal, negotiation, closed_won, closed_lost",
                        },
                        "notes": {"type": "string", "description": "Notes about the stage change"},
                    },
                    "required": ["lead_id", "new_stage"],
                },
            },
            {
                "name": "crm_convert_lead_to_client",
                "description": "Convert a won lead into a paying client.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID to convert"},
                        "tier": {
                            "type": "integer",
                            "description": "Service tier: 1=Content Engine ($500), 2=Sales Autopilot ($1000), 3=Business Suite ($2000)",
                        },
                    },
                    "required": ["lead_id", "tier"],
                },
            },
            {
                "name": "crm_get_pipeline_summary",
                "description": "Get a summary of the sales pipeline including values and stage counts.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "crm_list_clients",
                "description": "List all active clients with their health scores and tier information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sort_by": {
                            "type": "string",
                            "description": "Sort by: health_score, monthly_value, months_active, churn_risk",
                        }
                    },
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a CRM tool call and return a string result."""
        dispatch = {
            "crm_list_leads": self._list_leads,
            "crm_add_lead": self._add_lead,
            "crm_update_lead_stage": self._update_lead_stage,
            "crm_convert_lead_to_client": self._convert_lead_to_client,
            "crm_get_pipeline_summary": self._get_pipeline_summary,
            "crm_list_clients": self._list_clients,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"Unknown CRM tool: {tool_name}"
        return fn(tool_input)

    def _list_leads(self, inp: Dict[str, Any]) -> str:
        stage_filter = inp.get("stage_filter", "all")
        limit = inp.get("limit", 20)
        leads = self.state.leads
        if stage_filter != "all":
            leads = [l for l in leads if l.stage == stage_filter]
        leads = leads[:limit]
        if not leads:
            return "No leads found matching the filter."
        lines = [f"Found {len(leads)} lead(s):"]
        for l in leads:
            lines.append(
                f"  [{l.id}] {l.name} @ {l.company} | Stage: {l.stage} | "
                f"Score: {l.score:.0f} | Value: ${l.value_potential:.0f}/mo | Source: {l.source}"
            )
        return "\n".join(lines)

    def _add_lead(self, inp: Dict[str, Any]) -> str:
        value = inp.get("value_potential", 500.0)
        # Snap value to nearest tier price
        if value >= 1500:
            value = 2000.0
        elif value >= 750:
            value = 1000.0
        else:
            value = 500.0

        lead = Lead(
            name=inp.get("name", "Unknown"),
            company=inp.get("company", "Unknown Co."),
            industry=inp.get("industry", "general"),
            value_potential=value,
            source=inp.get("source", "organic"),
            score=round(random.uniform(45, 80), 1),
            stage="prospect",
        )
        self.state.add_lead(lead)
        return (
            f"Lead added: {lead.name} @ {lead.company} | "
            f"ID: {lead.id} | Score: {lead.score} | Value: ${lead.value_potential:.0f}/mo"
        )

    def _update_lead_stage(self, inp: Dict[str, Any]) -> str:
        lead_id = inp.get("lead_id", "")
        new_stage = inp.get("new_stage", "")
        notes = inp.get("notes", "")
        lead = self.state.get_lead(lead_id)
        if not lead:
            return f"Lead {lead_id} not found."
        old_stage = lead.stage
        lead.stage = new_stage
        if notes:
            lead.notes = notes
        lead.last_contact = datetime.now().isoformat()
        return (
            f"Lead {lead.name} @ {lead.company} moved from '{old_stage}' to '{new_stage}'."
            + (f" Notes: {notes}" if notes else "")
        )

    def _convert_lead_to_client(self, inp: Dict[str, Any]) -> str:
        lead_id = inp.get("lead_id", "")
        tier = inp.get("tier", 1)
        lead = self.state.get_lead(lead_id)
        if not lead:
            return f"Lead {lead_id} not found."
        tier_prices = {1: 500.0, 2: 1000.0, 3: 2000.0}
        tier_names = {1: "AI Content Engine", 2: "AI Sales Autopilot", 3: "AI Business Suite"}
        price = tier_prices.get(tier, 500.0)
        client = Client(
            name=lead.name,
            company=lead.company,
            tier=tier,
            monthly_value=price,
            health_score=85.0,
            churn_risk=0.05,
            months_active=0,
        )
        lead.stage = "closed_won"
        self.state.add_client(client)
        return (
            f"CONVERSION: {lead.name} @ {lead.company} is now a client! "
            f"Tier: {tier_names[tier]} | MRR +${price:.0f} | "
            f"New MRR: ${self.state.mrr:.0f}"
        )

    def _get_pipeline_summary(self, _inp: Dict[str, Any]) -> str:
        from src.models.metrics import MetricsCalculator
        funnel = MetricsCalculator.calculate_conversion_funnel(self.state.leads)
        pipeline_value = self.state.pipeline_value()
        lines = [
            "=== Pipeline Summary ===",
            f"Total Pipeline Value: ${pipeline_value:,.0f}/mo",
        ]
        for stage, count in funnel.items():
            lines.append(f"  {stage.capitalize()}: {count}")
        lines.append(f"Total Active Clients: {len(self.state.clients)}")
        lines.append(f"Current MRR: ${self.state.mrr:,.0f}")
        return "\n".join(lines)

    def _list_clients(self, inp: Dict[str, Any]) -> str:
        sort_by = inp.get("sort_by", "monthly_value")
        clients = list(self.state.clients)
        if sort_by == "health_score":
            clients.sort(key=lambda c: c.health_score, reverse=True)
        elif sort_by == "churn_risk":
            clients.sort(key=lambda c: c.churn_risk, reverse=True)
        elif sort_by == "months_active":
            clients.sort(key=lambda c: c.months_active, reverse=True)
        else:
            clients.sort(key=lambda c: c.monthly_value, reverse=True)
        if not clients:
            return "No active clients yet."
        lines = [f"Active Clients ({len(clients)} total):"]
        for c in clients:
            risk_label = "HIGH" if c.churn_risk > 0.3 else ("MED" if c.churn_risk > 0.15 else "LOW")
            lines.append(
                f"  [{c.id}] {c.name} @ {c.company} | {c.tier_name()} | "
                f"${c.monthly_value:.0f}/mo | Health: {c.health_score:.0f} | "
                f"Churn Risk: {risk_label}"
            )
        return "\n".join(lines)

    @staticmethod
    def generate_random_lead(source: str = "organic") -> Lead:
        """Generate a realistic fake lead for simulation."""
        if COMPANY_POOL:
            person, company, industry = random.choice(COMPANY_POOL)
        else:
            person, company, industry = "John Doe", "Acme Corp", "general"
        tier_weights = [0.6, 0.3, 0.1]  # Tier 1 most common
        tier = random.choices([1, 2, 3], weights=tier_weights)[0]
        prices = {1: 500.0, 2: 1000.0, 3: 2000.0}
        return Lead(
            name=person,
            company=company,
            industry=industry,
            score=round(random.uniform(40, 85), 1),
            stage="prospect",
            value_potential=prices[tier],
            source=source,
        )
