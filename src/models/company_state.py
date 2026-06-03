"""
Company state dataclasses for IAN.AI Ventures.
Tracks all business entities: company, clients, leads, campaigns.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import uuid


@dataclass
class Client:
    """Represents a paying client."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    company: str = ""
    tier: int = 1  # 1=Content Engine, 2=Sales Autopilot, 3=Business Suite
    monthly_value: float = 500.0
    start_date: str = field(default_factory=lambda: date.today().isoformat())
    health_score: float = 85.0  # 0-100
    churn_risk: float = 0.1  # 0.0-1.0
    months_active: int = 0
    notes: str = ""

    def tier_name(self) -> str:
        names = {1: "AI Content Engine", 2: "AI Sales Autopilot", 3: "AI Business Suite"}
        return names.get(self.tier, "Unknown")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "company": self.company,
            "tier": self.tier,
            "tier_name": self.tier_name(),
            "monthly_value": self.monthly_value,
            "start_date": self.start_date,
            "health_score": self.health_score,
            "churn_risk": self.churn_risk,
            "months_active": self.months_active,
        }


@dataclass
class Lead:
    """Represents a potential client in the sales pipeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    company: str = ""
    email: str = ""
    industry: str = ""
    company_size: str = "SMB"  # SMB, Mid-Market, Enterprise
    score: float = 50.0  # 0-100 qualification score
    stage: str = "prospect"  # prospect, qualified, proposal, negotiation, closed_won, closed_lost
    value_potential: float = 500.0
    source: str = "organic"  # organic, paid, referral, outbound
    outreach_count: int = 0
    last_contact: Optional[str] = None
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "company": self.company,
            "email": self.email,
            "industry": self.industry,
            "score": self.score,
            "stage": self.stage,
            "value_potential": self.value_potential,
            "source": self.source,
            "outreach_count": self.outreach_count,
        }


@dataclass
class Campaign:
    """Represents a marketing or sales campaign."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    type: str = "content"  # content, email, social, paid, outbound
    status: str = "draft"  # draft, active, paused, completed
    target_audience: str = ""
    leads_generated: int = 0
    impressions: int = 0
    clicks: int = 0
    cost: float = 0.0
    revenue_attributed: float = 0.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    notes: str = ""

    def roi(self) -> float:
        if self.cost == 0:
            return 0.0
        return (self.revenue_attributed - self.cost) / self.cost * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "leads_generated": self.leads_generated,
            "cost": self.cost,
            "revenue_attributed": self.revenue_attributed,
            "roi": self.roi(),
        }


@dataclass
class Action:
    """Represents an action taken by an agent."""
    agent: str = ""
    action_type: str = ""
    description: str = ""
    result: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    impact: str = "neutral"  # positive, negative, neutral
    value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "action_type": self.action_type,
            "description": self.description,
            "result": self.result,
            "timestamp": self.timestamp,
            "impact": self.impact,
            "value": self.value,
        }


@dataclass
class AgentResult:
    """Result returned by an agent after completing a task."""
    agent_name: str = ""
    task: str = ""
    output: str = ""
    actions_taken: List[Action] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "output": self.output,
            "actions_taken": [a.to_dict() for a in self.actions_taken],
            "recommendations": self.recommendations,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class CompanyState:
    """
    Top-level company state for IAN.AI Ventures.

    Capital: $1,000 USD
    Target MRR: $30,000 USD
    """
    name: str = "IAN.AI Ventures"
    capital: float = 1000.0
    mrr: float = 0.0
    target_mrr: float = 30000.0
    month: int = 1
    clients: List[Client] = field(default_factory=list)
    leads: List[Lead] = field(default_factory=list)
    campaigns: List[Campaign] = field(default_factory=list)
    action_history: List[Action] = field(default_factory=list)
    monthly_expenses: float = 0.0
    total_revenue: float = 0.0
    churn_rate: float = 0.05
    conversion_rate: float = 0.15  # lead to client
    priorities: List[str] = field(default_factory=lambda: [
        "Generate initial leads",
        "Close first clients",
        "Build brand presence"
    ])
    agent_reports: Dict[str, Any] = field(default_factory=dict)

    # Pricing tiers
    TIER_PRICES = {1: 500.0, 2: 1000.0, 3: 2000.0}
    TIER_NAMES = {1: "AI Content Engine", 2: "AI Sales Autopilot", 3: "AI Business Suite"}
    TIER_TARGETS = {1: 20, 2: 12, 3: 4}

    def calculate_mrr(self) -> float:
        """Calculate current MRR from active clients."""
        return sum(c.monthly_value for c in self.clients)

    def update_mrr(self) -> None:
        """Update MRR field from current clients."""
        self.mrr = self.calculate_mrr()

    def progress_to_target(self) -> float:
        """Return progress percentage toward target MRR."""
        if self.target_mrr == 0:
            return 100.0
        return min(100.0, (self.mrr / self.target_mrr) * 100)

    def add_client(self, client: Client) -> None:
        """Add a new client and update MRR."""
        self.clients.append(client)
        self.update_mrr()

    def remove_client(self, client_id: str) -> Optional[Client]:
        """Remove a client by ID (churn)."""
        for i, c in enumerate(self.clients):
            if c.id == client_id:
                removed = self.clients.pop(i)
                self.update_mrr()
                return removed
        return None

    def add_lead(self, lead: Lead) -> None:
        """Add a new lead to the pipeline."""
        self.leads.append(lead)

    def get_lead(self, lead_id: str) -> Optional[Lead]:
        """Get a lead by ID."""
        for lead in self.leads:
            if lead.id == lead_id:
                return lead
        return None

    def qualified_leads(self) -> List[Lead]:
        """Get leads that are qualified or further in the pipeline."""
        return [l for l in self.leads if l.stage in ("qualified", "proposal", "negotiation")]

    def pipeline_value(self) -> float:
        """Calculate total pipeline value."""
        active_stages = ("prospect", "qualified", "proposal", "negotiation")
        return sum(l.value_potential for l in self.leads if l.stage in active_stages)

    def runway_months(self) -> float:
        """Calculate months of runway at current burn rate."""
        net_monthly = self.mrr - self.monthly_expenses
        if net_monthly >= 0:
            return float('inf')
        return self.capital / abs(net_monthly)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a summary dict for agent context."""
        return {
            "company": self.name,
            "month": self.month,
            "capital": self.capital,
            "mrr": self.mrr,
            "target_mrr": self.target_mrr,
            "progress_pct": round(self.progress_to_target(), 1),
            "total_clients": len(self.clients),
            "total_leads": len(self.leads),
            "qualified_leads": len(self.qualified_leads()),
            "pipeline_value": self.pipeline_value(),
            "monthly_expenses": self.monthly_expenses,
            "runway_months": self.runway_months(),
            "tier_breakdown": {
                str(tier): len([c for c in self.clients if c.tier == tier])
                for tier in [1, 2, 3]
            },
            "priorities": self.priorities,
        }
