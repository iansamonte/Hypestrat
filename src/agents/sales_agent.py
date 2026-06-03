"""
Sales Agent — Lead qualification, outreach sequences, pipeline management, and demos.
"""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.models.company_state import Action, CompanyState, Lead
from src.tools.crm_tool import CRMTool
from src.tools.outreach_tool import OutreachTool

logger = logging.getLogger(__name__)


class SalesAgent(BaseAgent):
    """
    The Sales Agent converts leads into paying clients by:
    - Qualifying inbound and outbound leads
    - Creating personalized outreach sequences
    - Running demos and handling objections
    - Closing deals and updating the pipeline
    """

    def __init__(self, model: str = BaseAgent.DEFAULT_MODEL) -> None:
        super().__init__(
            name="Sales",
            role="Chief Sales Officer",
            model=model,
        )
        self._crm: CRMTool = None
        self._outreach: OutreachTool = None

    def get_system_prompt(self) -> str:
        return (
            "You are the CSO of IAN.AI Ventures, driving revenue by converting leads to clients.\n\n"
            "Service Tiers:\n"
            "  - Tier 1: AI Content Engine — $500/mo (easiest sell, widest market)\n"
            "  - Tier 2: AI Sales Autopilot — $1,000/mo (mid-market SMBs wanting growth)\n"
            "  - Tier 3: AI Business Suite — $2,000/mo (established SMBs wanting full automation)\n\n"
            "Your responsibilities:\n"
            "1. Qualify leads in the pipeline — focus on score > 60 and relevant industries.\n"
            "2. Send personalized outreach to top prospects.\n"
            "3. Book demos with qualified leads.\n"
            "4. Move deals through the pipeline (prospect → qualified → proposal → negotiation → won).\n"
            "5. Close at least 1-2 deals per cycle.\n\n"
            "Sales principles:\n"
            "- Lead with ROI: 'Save 10 hrs/week, generate 3x more leads'\n"
            "- Start with Tier 1 if unsure — upsell later\n"
            "- Follow up 3x before giving up on a lead\n"
            "- Aim for 15-20% lead-to-client conversion rate\n\n"
            "Always: qualify_lead first, then create outreach, book demos, and close deals. "
            "Try to close at least one lead per cycle by converting qualified prospects."
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "qualify_lead",
                "description": "Evaluate and score a lead to determine if they're worth pursuing.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID to qualify"},
                        "criteria": {
                            "type": "object",
                            "description": "Qualification criteria to check",
                            "properties": {
                                "budget_fit": {"type": "boolean"},
                                "authority": {"type": "boolean"},
                                "need": {"type": "boolean"},
                                "timeline": {"type": "string"},
                            },
                        },
                    },
                    "required": ["lead_id"],
                },
            },
            {
                "name": "create_outreach_sequence",
                "description": "Create and send a personalized outreach sequence to a lead.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Target lead ID"},
                        "approach": {
                            "type": "string",
                            "description": "Outreach approach: value_prop, case_study, pain_point, referral",
                        },
                        "tier_recommendation": {
                            "type": "integer",
                            "description": "Recommended tier (1, 2, or 3)",
                        },
                    },
                    "required": ["lead_id"],
                },
            },
            {
                "name": "update_pipeline",
                "description": "Update multiple leads' stages in the pipeline.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "array",
                            "description": "List of stage updates",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "lead_id": {"type": "string"},
                                    "new_stage": {"type": "string"},
                                    "notes": {"type": "string"},
                                },
                            },
                        }
                    },
                    "required": ["updates"],
                },
            },
            {
                "name": "book_demo",
                "description": "Book a product demo with a qualified lead.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID to book demo with"},
                        "tier_focus": {
                            "type": "integer",
                            "description": "Which tier to demo (1, 2, or 3)",
                        },
                    },
                    "required": ["lead_id"],
                },
            },
            {
                "name": "close_deal",
                "description": "Convert a lead in proposal/negotiation stage to a paying client.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID to close"},
                        "tier": {
                            "type": "integer",
                            "description": "Service tier they're signing up for (1, 2, or 3)",
                        },
                        "discount_pct": {
                            "type": "number",
                            "description": "Discount percentage offered (0-20)",
                        },
                    },
                    "required": ["lead_id", "tier"],
                },
            },
            {
                "name": "get_pipeline_status",
                "description": "Get current pipeline status and identify the best opportunities.",
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
        if self._crm is None:
            self._crm = CRMTool(state)
        if self._outreach is None:
            self._outreach = OutreachTool(state)

        if tool_name == "qualify_lead":
            return self._qualify_lead(tool_input, state)
        elif tool_name == "create_outreach_sequence":
            return self._create_outreach(tool_input, state)
        elif tool_name == "update_pipeline":
            return self._update_pipeline(tool_input, state)
        elif tool_name == "book_demo":
            return self._book_demo(tool_input, state)
        elif tool_name == "close_deal":
            return self._close_deal(tool_input, state)
        elif tool_name == "get_pipeline_status":
            return self._crm.execute("crm_get_pipeline_summary", {})
        return f"Unknown sales tool: {tool_name}"

    def _qualify_lead(self, inp: Dict[str, Any], state: CompanyState) -> str:
        lead_id = inp.get("lead_id", "")
        criteria = inp.get("criteria", {})
        lead = state.get_lead(lead_id)
        if not lead:
            return f"Lead {lead_id} not found."

        # Score adjustments based on criteria
        score_boost = 0
        if criteria.get("budget_fit"):
            score_boost += 10
        if criteria.get("authority"):
            score_boost += 8
        if criteria.get("need"):
            score_boost += 12

        lead.score = min(100, lead.score + score_boost)

        qualified = lead.score >= 60
        if qualified and lead.stage == "prospect":
            lead.stage = "qualified"

        return (
            f"Lead qualified: {lead.name} @ {lead.company}\n"
            f"  Score: {lead.score:.0f}/100 | {'QUALIFIED' if qualified else 'NOT QUALIFIED'}\n"
            f"  Stage: {lead.stage}\n"
            f"  Value potential: ${lead.value_potential:,.0f}/mo\n"
            f"  Recommendation: {'Proceed with outreach' if qualified else 'Nurture or deprioritize'}"
        )

    def _create_outreach(self, inp: Dict[str, Any], state: CompanyState) -> str:
        lead_id = inp.get("lead_id", "")
        approach = inp.get("approach", "value_prop")
        tier_rec = inp.get("tier_recommendation", 1)

        tier_names = {1: "AI Content Engine ($500/mo)", 2: "AI Sales Autopilot ($1,000/mo)", 3: "AI Business Suite ($2,000/mo)"}
        tier_name = tier_names.get(tier_rec, "AI Content Engine")

        msg = (
            f"Personalized {approach} outreach for {tier_name}. "
            f"Focusing on ROI and specific pain points."
        )
        return self._outreach.execute(
            "outreach_send_sequence",
            {"lead_id": lead_id, "sequence_type": "cold_email", "message": msg},
        )

    def _update_pipeline(self, inp: Dict[str, Any], state: CompanyState) -> str:
        updates = inp.get("updates", [])
        results = []
        for upd in updates:
            res = self._crm.execute(
                "crm_update_lead_stage",
                {
                    "lead_id": upd.get("lead_id", ""),
                    "new_stage": upd.get("new_stage", ""),
                    "notes": upd.get("notes", ""),
                },
            )
            results.append(res)
        return "\n".join(results) if results else "No pipeline updates made."

    def _book_demo(self, inp: Dict[str, Any], state: CompanyState) -> str:
        lead_id = inp.get("lead_id", "")
        tier_focus = inp.get("tier_focus", 1)
        result = self._outreach.execute(
            "outreach_book_demo",
            {"lead_id": lead_id, "days_from_now": random.randint(2, 5)},
        )
        return result + f"\n  Demo focus: Tier {tier_focus}"

    def _close_deal(self, inp: Dict[str, Any], state: CompanyState) -> str:
        lead_id = inp.get("lead_id", "")
        tier = inp.get("tier", 1)
        discount = inp.get("discount_pct", 0)

        lead = state.get_lead(lead_id)
        if not lead:
            return f"Lead {lead_id} not found."

        if lead.stage not in ("qualified", "proposal", "negotiation"):
            # Allow close from any non-lost stage if score is good
            if lead.score < 55:
                return f"Lead {lead.name} is not ready to close (stage: {lead.stage}, score: {lead.score:.0f})."

        # Apply discount
        tier_prices = {1: 500.0, 2: 1000.0, 3: 2000.0}
        base_price = tier_prices.get(tier, 500.0)
        final_price = base_price * (1 - discount / 100)

        # Simulate close probability
        close_prob = 0.65 + (lead.score - 60) / 100
        closed = random.random() < max(0.4, min(0.9, close_prob))

        if closed:
            result = self._crm.execute(
                "crm_convert_lead_to_client",
                {"lead_id": lead_id, "tier": tier},
            )
            if discount > 0:
                # Adjust the last-added client's price
                if state.clients:
                    state.clients[-1].monthly_value = final_price
                    state.update_mrr()
                result += f"\n  Discount applied: {discount}% -> Final price: ${final_price:.0f}/mo"
            return result
        else:
            lead.stage = "negotiation"
            return (
                f"Deal not closed yet with {lead.name} @ {lead.company}. "
                f"Moving to negotiation stage. Follow up with objection handling."
            )

    def run_cycle(self, state: CompanyState) -> List[Action]:
        """Run one sales cycle."""
        self.log_cycle_start(state)
        self._crm = CRMTool(state)
        self._outreach = OutreachTool(state)

        # Get top leads to work
        qualified = [l for l in state.leads if l.stage in ("qualified", "proposal", "negotiation")]
        prospects = [l for l in state.leads if l.stage == "prospect"]
        top_prospects = sorted(prospects, key=lambda l: l.score, reverse=True)[:5]

        context = {
            **state.to_summary_dict(),
            "leads_to_qualify": [l.to_dict() for l in top_prospects[:3]],
            "qualified_leads": [l.to_dict() for l in qualified[:5]],
            "target_closes_this_month": max(1, int(len(qualified) * 0.3)),
        }

        task = (
            f"Month {state.month} Sales Cycle. "
            f"Current MRR: ${state.mrr:,.0f} / ${state.target_mrr:,.0f} target. "
            f"Pipeline: {len(qualified)} qualified, {len(prospects)} prospects. "
            f"Qualify the top prospects, send outreach, book demos with qualified leads, "
            f"and close at least 1-2 deals this cycle. "
            f"Use the lead IDs provided in context to work specific leads."
        )

        result = self.run(task=task, context=context, state=state)

        state.agent_reports["Sales"] = {
            "month": state.month,
            "pipeline_size": len(state.leads),
            "qualified_leads": len(qualified),
            "current_clients": len(state.clients),
            "output_summary": result.output[:300],
        }

        self.log_cycle_end(result.actions_taken)
        return result.actions_taken
