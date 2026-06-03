"""
Outreach Tool — Simulate email and LinkedIn outreach sequences.
"""

from __future__ import annotations
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.models.company_state import Lead, CompanyState


OUTREACH_RESPONSES = [
    "Thanks for reaching out! I'm interested in learning more.",
    "This looks interesting. Can we schedule a quick call?",
    "Not the right time for us, but keep us in mind.",
    "We're actually evaluating solutions right now — perfect timing!",
    "Could you send over some case studies?",
    "I've been looking for something exactly like this.",
    "Let's chat. What does your availability look like?",
    "Sounds promising. Can you walk me through the ROI?",
]


class OutreachTool:
    """Tool for managing email and LinkedIn outreach sequences."""

    def __init__(self, state: CompanyState):
        self.state = state
        self._sequences: List[Dict[str, Any]] = []
        self._sent_messages: List[Dict[str, Any]] = []

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "outreach_send_sequence",
                "description": "Send an outreach sequence to a specific lead.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID to contact"},
                        "sequence_type": {
                            "type": "string",
                            "description": "Type: cold_email, linkedin, follow_up, demo_invite",
                        },
                        "message": {
                            "type": "string",
                            "description": "Custom message or talking points",
                        },
                    },
                    "required": ["lead_id", "sequence_type"],
                },
            },
            {
                "name": "outreach_bulk_prospect",
                "description": "Launch a bulk outreach to all leads in a given stage.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "stage": {
                            "type": "string",
                            "description": "Target stage: prospect or qualified",
                        },
                        "channel": {
                            "type": "string",
                            "description": "Channel: email, linkedin, both",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum leads to contact (default 10)",
                        },
                    },
                    "required": ["stage"],
                },
            },
            {
                "name": "outreach_book_demo",
                "description": "Schedule a demo call with a qualified lead.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID"},
                        "days_from_now": {
                            "type": "integer",
                            "description": "Days from now to schedule the demo",
                        },
                    },
                    "required": ["lead_id"],
                },
            },
            {
                "name": "outreach_follow_up",
                "description": "Send a follow-up message to a lead who hasn't responded.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lead_id": {"type": "string", "description": "Lead ID"},
                        "follow_up_number": {
                            "type": "integer",
                            "description": "Which follow-up is this (1, 2, 3...)",
                        },
                    },
                    "required": ["lead_id"],
                },
            },
            {
                "name": "outreach_get_stats",
                "description": "Get outreach performance statistics.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        dispatch = {
            "outreach_send_sequence": self._send_sequence,
            "outreach_bulk_prospect": self._bulk_prospect,
            "outreach_book_demo": self._book_demo,
            "outreach_follow_up": self._follow_up,
            "outreach_get_stats": self._get_stats,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"Unknown outreach tool: {tool_name}"
        return fn(tool_input)

    def _send_sequence(self, inp: Dict[str, Any]) -> str:
        lead_id = inp.get("lead_id", "")
        seq_type = inp.get("sequence_type", "cold_email")
        message = inp.get("message", "")

        lead = self.state.get_lead(lead_id)
        if not lead:
            return f"Lead {lead_id} not found."

        lead.outreach_count += 1
        lead.last_contact = datetime.now().isoformat()

        # Simulate response probability
        response_prob = 0.15 + (lead.score / 100) * 0.20
        responded = random.random() < response_prob
        response = random.choice(OUTREACH_RESPONSES) if responded else None

        # Advance stage if responded positively
        if responded and lead.stage == "prospect":
            lead.stage = "qualified"

        record = {
            "lead_id": lead_id,
            "lead_name": lead.name,
            "company": lead.company,
            "type": seq_type,
            "sent_at": datetime.now().isoformat(),
            "responded": responded,
        }
        self._sent_messages.append(record)

        result = (
            f"Outreach sent to {lead.name} @ {lead.company} via {seq_type}\n"
            f"  Message: {message[:80] + '...' if len(message) > 80 else message or '[template]'}\n"
            f"  Outreach count: {lead.outreach_count}\n"
        )
        if responded:
            result += f"  Response received: '{response}'\n"
            result += f"  Lead advanced to: {lead.stage}"
        else:
            result += "  No response yet. Schedule follow-up in 3 days."
        return result

    def _bulk_prospect(self, inp: Dict[str, Any]) -> str:
        stage = inp.get("stage", "prospect")
        channel = inp.get("channel", "email")
        limit = min(inp.get("limit", 10), 50)

        targets = [l for l in self.state.leads if l.stage == stage][:limit]
        if not targets:
            return f"No leads in '{stage}' stage to contact."

        responded = 0
        advanced = 0
        for lead in targets:
            lead.outreach_count += 1
            lead.last_contact = datetime.now().isoformat()
            prob = 0.12 + (lead.score / 100) * 0.18
            if random.random() < prob:
                responded += 1
                if lead.stage == "prospect":
                    lead.stage = "qualified"
                    advanced += 1

        return (
            f"Bulk outreach sent via {channel} to {len(targets)} leads in '{stage}'\n"
            f"  Responses received: {responded}\n"
            f"  Leads advanced to qualified: {advanced}\n"
            f"  Response rate: {responded/len(targets)*100:.1f}%"
        )

    def _book_demo(self, inp: Dict[str, Any]) -> str:
        lead_id = inp.get("lead_id", "")
        days = inp.get("days_from_now", 3)
        lead = self.state.get_lead(lead_id)
        if not lead:
            return f"Lead {lead_id} not found."

        demo_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
        lead.stage = "proposal"
        lead.notes = f"Demo scheduled: {demo_date}"

        return (
            f"Demo booked with {lead.name} @ {lead.company}\n"
            f"  Date/Time: {demo_date}\n"
            f"  Lead stage updated to: proposal\n"
            f"  Value potential: ${lead.value_potential:,.0f}/mo"
        )

    def _follow_up(self, inp: Dict[str, Any]) -> str:
        lead_id = inp.get("lead_id", "")
        fu_num = inp.get("follow_up_number", 1)
        lead = self.state.get_lead(lead_id)
        if not lead:
            return f"Lead {lead_id} not found."

        lead.outreach_count += 1
        lead.last_contact = datetime.now().isoformat()

        # Diminishing return on follow-ups
        prob = max(0.05, 0.20 - fu_num * 0.04)
        responded = random.random() < prob

        if responded and lead.stage in ("prospect", "qualified"):
            if lead.stage == "prospect":
                lead.stage = "qualified"
            elif random.random() > 0.5:
                lead.stage = "proposal"

        result = (
            f"Follow-up #{fu_num} sent to {lead.name} @ {lead.company}\n"
            f"  Total touches: {lead.outreach_count}\n"
        )
        if responded:
            result += f"  Responded! Stage: {lead.stage}"
        else:
            result += "  No response. Consider pausing outreach after 5 touches."
        return result

    def _get_stats(self, _inp: Dict[str, Any]) -> str:
        total = len(self._sent_messages)
        responded = sum(1 for m in self._sent_messages if m.get("responded"))

        if total == 0:
            return "No outreach messages sent yet."

        rate = responded / total * 100

        lines = [
            "=== Outreach Statistics ===",
            f"Total Messages Sent: {total}",
            f"Responses Received: {responded}",
            f"Response Rate: {rate:.1f}%",
            "",
            "By Channel:",
        ]
        by_type: Dict[str, int] = {}
        for m in self._sent_messages:
            t = m.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        for ch, cnt in by_type.items():
            lines.append(f"  {ch}: {cnt}")
        return "\n".join(lines)
