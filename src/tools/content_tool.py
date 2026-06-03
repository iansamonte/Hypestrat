"""
Content Tool — Generate blog posts, emails, social content, and marketing materials.
"""

from __future__ import annotations
import random
from typing import Dict, Any, List

from src.models.company_state import Campaign, CompanyState


BLOG_TOPICS = [
    "How AI Automation Saves SMBs 20 Hours Per Week",
    "The Hidden Cost of Manual Marketing: What You're Missing",
    "5 Ways AI is Transforming Small Business Sales in 2025",
    "From 0 to $30k MRR: Our AI Automation Playbook",
    "Why Your Competitors Are Already Using AI (And How to Catch Up)",
    "AI Content Engine: Generate a Month of Content in 2 Hours",
    "The SMB Guide to AI Sales Automation",
    "Reducing Customer Churn with AI-Powered Success Monitoring",
]

EMAIL_SUBJECTS = [
    "Quick question about [Company]'s growth strategy",
    "Found something that might help [Company] save 10 hrs/week",
    "3 SMBs in [industry] that doubled leads with AI — case study",
    "Is manual outreach slowing down [Company]?",
    "[First name], saw your LinkedIn post — have an idea for you",
]

SOCIAL_TEMPLATES = [
    "We helped a {industry} company generate {leads} new leads in 30 days using AI automation. Here's the exact playbook 🧵",
    "Stop doing these 3 things manually if you run a business: 1) Content creation 2) Lead outreach 3) Client reporting. AI does it better.",
    "Our client {company} went from spending 20 hrs/week on marketing to 2 hrs using our AI Content Engine. The results? +{pct}% leads.",
    "Hot take: the SMBs winning in 2025 are the ones that treat AI as an employee, not a tool.",
]


class ContentTool:
    """Tool for generating and managing marketing content."""

    def __init__(self, state: CompanyState):
        self.state = state
        self._content_library: List[Dict[str, Any]] = []

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "content_create_blog_post",
                "description": "Generate a blog post on a given topic to drive organic traffic and demonstrate expertise.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Blog post topic or title"},
                        "target_audience": {
                            "type": "string",
                            "description": "Who this is written for (e.g., 'SMB owners in marketing')",
                        },
                        "cta": {
                            "type": "string",
                            "description": "Call to action at the end of the post",
                        },
                    },
                    "required": ["topic"],
                },
            },
            {
                "name": "content_create_email_sequence",
                "description": "Create a multi-email outreach or nurture sequence.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sequence_type": {
                            "type": "string",
                            "description": "Type: cold_outreach, follow_up, nurture, onboarding, upsell",
                        },
                        "target_segment": {
                            "type": "string",
                            "description": "Target audience segment",
                        },
                        "emails_count": {
                            "type": "integer",
                            "description": "Number of emails in the sequence (1-7)",
                        },
                    },
                    "required": ["sequence_type"],
                },
            },
            {
                "name": "content_create_social_post",
                "description": "Generate social media content for LinkedIn, Twitter, or other platforms.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "platform": {
                            "type": "string",
                            "description": "Platform: linkedin, twitter, instagram",
                        },
                        "theme": {
                            "type": "string",
                            "description": "Content theme: thought_leadership, case_study, product_demo, tips",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of posts to generate",
                        },
                    },
                    "required": ["platform", "theme"],
                },
            },
            {
                "name": "content_launch_campaign",
                "description": "Launch a content or marketing campaign and track it in the system.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Campaign name"},
                        "type": {
                            "type": "string",
                            "description": "Campaign type: content, email, social, paid, outbound",
                        },
                        "target_audience": {
                            "type": "string",
                            "description": "Target audience description",
                        },
                        "budget": {
                            "type": "number",
                            "description": "Campaign budget in USD",
                        },
                    },
                    "required": ["name", "type"],
                },
            },
            {
                "name": "content_get_library",
                "description": "List all content pieces created so far.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "Filter by type: blog, email, social, or 'all'",
                        }
                    },
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        dispatch = {
            "content_create_blog_post": self._create_blog_post,
            "content_create_email_sequence": self._create_email_sequence,
            "content_create_social_post": self._create_social_post,
            "content_launch_campaign": self._launch_campaign,
            "content_get_library": self._get_library,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"Unknown content tool: {tool_name}"
        return fn(tool_input)

    def _create_blog_post(self, inp: Dict[str, Any]) -> str:
        topic = inp.get("topic", random.choice(BLOG_TOPICS))
        audience = inp.get("target_audience", "SMB owners")
        cta = inp.get("cta", "Book a free AI audit at ian-ai.com")

        word_count = random.randint(800, 1400)
        seo_score = random.randint(72, 95)

        piece = {
            "type": "blog",
            "title": topic,
            "audience": audience,
            "word_count": word_count,
            "seo_score": seo_score,
            "cta": cta,
        }
        self._content_library.append(piece)

        return (
            f"Blog post created: '{topic}'\n"
            f"  Target Audience: {audience}\n"
            f"  Word Count: {word_count}\n"
            f"  SEO Score: {seo_score}/100\n"
            f"  CTA: {cta}\n"
            f"  Status: Ready to publish\n"
            f"  Estimated monthly organic traffic: {random.randint(50, 300)} visitors"
        )

    def _create_email_sequence(self, inp: Dict[str, Any]) -> str:
        seq_type = inp.get("sequence_type", "cold_outreach")
        segment = inp.get("target_segment", "SMB owners")
        count = min(inp.get("emails_count", 3), 7)

        subjects = random.sample(EMAIL_SUBJECTS, min(count, len(EMAIL_SUBJECTS)))

        piece = {
            "type": "email_sequence",
            "sequence_type": seq_type,
            "segment": segment,
            "emails": count,
            "subjects": subjects,
        }
        self._content_library.append(piece)

        open_rate = random.uniform(18, 42)
        reply_rate = random.uniform(3, 12)

        subject_lines = "\n".join(f"  Email {i+1}: {s}" for i, s in enumerate(subjects))
        return (
            f"Email sequence created: {seq_type} ({count} emails)\n"
            f"  Target segment: {segment}\n"
            f"Subject lines:\n{subject_lines}\n"
            f"  Expected open rate: {open_rate:.1f}%\n"
            f"  Expected reply rate: {reply_rate:.1f}%"
        )

    def _create_social_post(self, inp: Dict[str, Any]) -> str:
        platform = inp.get("platform", "linkedin")
        theme = inp.get("theme", "thought_leadership")
        count = min(inp.get("count", 3), 10)

        posts = []
        for i in range(count):
            template = random.choice(SOCIAL_TEMPLATES)
            post = template.format(
                industry=random.choice(["marketing", "retail", "consulting", "tech"]),
                leads=random.randint(20, 80),
                company=random.choice(["BluePeak Digital", "Summit Consulting", "NovaTech"]),
                pct=random.randint(35, 120),
            )
            posts.append(post)

        piece = {
            "type": "social",
            "platform": platform,
            "theme": theme,
            "count": count,
        }
        self._content_library.append(piece)

        impressions_est = count * random.randint(300, 1200)
        engagement_rate = random.uniform(2.5, 8.5)

        return (
            f"Social content created: {count} {platform} posts ({theme})\n"
            f"  Preview: '{posts[0][:100]}...'\n"
            f"  Estimated impressions: {impressions_est:,}\n"
            f"  Expected engagement rate: {engagement_rate:.1f}%"
        )

    def _launch_campaign(self, inp: Dict[str, Any]) -> str:
        name = inp.get("name", "Unnamed Campaign")
        camp_type = inp.get("type", "content")
        audience = inp.get("target_audience", "SMB owners")
        budget = inp.get("budget", 0.0)

        campaign = Campaign(
            name=name,
            type=camp_type,
            target_audience=audience,
            cost=budget,
            status="active",
            leads_generated=random.randint(3, 15),
            impressions=random.randint(500, 5000),
        )
        self.state.campaigns.append(campaign)

        if budget > 0:
            self.state.capital -= budget
            self.state.monthly_expenses += budget

        return (
            f"Campaign launched: '{name}' [{camp_type}]\n"
            f"  Target: {audience}\n"
            f"  Budget: ${budget:.0f}\n"
            f"  Status: ACTIVE\n"
            f"  Projected leads: {campaign.leads_generated}\n"
            f"  Projected impressions: {campaign.impressions:,}"
        )

    def _get_library(self, inp: Dict[str, Any]) -> str:
        content_type = inp.get("content_type", "all")
        items = self._content_library
        if content_type != "all":
            items = [c for c in items if c.get("type") == content_type]
        if not items:
            return "No content pieces in library yet."
        lines = [f"Content Library ({len(items)} items):"]
        for item in items:
            lines.append(f"  [{item['type']}] {item.get('title', item.get('sequence_type', item.get('theme', '?')))}")
        return "\n".join(lines)
