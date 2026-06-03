# IAN.AI Ventures

**Fully Autonomous AI Company** — Targeting $30,000 MRR

An AI-run company that sells AI automation services to SMBs, managed entirely by a multi-agent system built on the Anthropic API.

---

## Business Model

| Tier | Product | Price | Target Clients | Target MRR |
|------|---------|-------|----------------|------------|
| 1 | AI Content Engine | $500/mo | 20 | $10,000 |
| 2 | AI Sales Autopilot | $1,000/mo | 12 | $12,000 |
| 3 | AI Business Suite | $2,000/mo | 4 | $8,000 |
| **Total** | | | **36** | **$30,000** |

**Starting capital:** $1,000 USD

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 3. Run simulation (no API calls needed)
python main.py --mode simulate

# 4. Run one live cycle (uses Anthropic API)
python main.py --mode run

# 5. Show current dashboard
python main.py --mode report
```

---

## Modes

### `--mode simulate`
Runs a 12-month algorithmic simulation showing realistic growth from $0 to $30k MRR. No API calls — instant results with growth curves, churn, and upsells.

### `--mode run`
Executes one complete company cycle using real Anthropic API calls. All agents (CEO, Marketing, Sales, Finance, Product, CS) run their tool-use loops and produce real outputs.

### `--mode report`
Displays the Rich terminal dashboard showing company metrics, pipeline, clients, and financial health.

---

## Architecture

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| `CEOAgent` | Strategy, priorities, budget | `get_company_metrics`, `set_priorities`, `approve_budget`, `review_agent_reports` |
| `MarketingAgent` | Content, campaigns, lead gen | `research_target_market`, `create_content`, `launch_campaign`, `add_leads_from_campaign` |
| `SalesAgent` | Qualification, outreach, closing | `qualify_lead`, `create_outreach_sequence`, `book_demo`, `close_deal` |
| `ProductAgent` | Service delivery, setup | `setup_client_automation`, `build_service_deliverable`, `check_delivery_status` |
| `FinanceAgent` | P&L, runway, forecasting | `update_mrr`, `track_expenses`, `calculate_runway`, `forecast_growth` |
| `CustomerSuccessAgent` | Onboarding, retention, upsell | `onboard_client`, `health_check`, `identify_upsell`, `execute_upsell` |

### Orchestrators

- **`CompanyOrchestrator`**: Runs daily, weekly, and monthly cycles across all agents
- **`RevenueOrchestrator`**: Focused growth sprints coordinating Marketing + Sales
- **`OperationsOrchestrator`**: Delivery cycles coordinating Product + Customer Success

### Tools

- **`CRMTool`**: In-memory CRM for leads and clients
- **`ContentTool`**: Blog, email, social content generation
- **`AnalyticsTool`**: MRR, funnel, growth metrics
- **`OutreachTool`**: Email and LinkedIn outreach simulation
- **`FinanceTool`**: P&L, runway, expense tracking

---

## Project Structure

```
Hypestrat/
├── src/
│   ├── agents/          # 6 specialized AI agents
│   ├── orchestrators/   # 3 coordination layers
│   ├── tools/           # 5 tool implementations
│   ├── models/          # CompanyState, Client, Lead, Campaign dataclasses
│   └── dashboard/       # Rich terminal UI
├── config/
│   ├── company_config.yaml
│   └── agent_prompts.yaml
├── data/               # Persistent storage (future)
├── logs/               # Agent logs
├── main.py             # Entry point
├── requirements.txt
└── .env.example
```

---

## Agent Loop

Each agent inherits from `BaseAgent` and runs a tool-use loop:

```python
while iterations < MAX_ITERATIONS:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        tools=self.get_tools(),
        messages=messages,
    )
    if response.stop_reason == "end_turn":
        break
    # Execute tool calls, append results, continue loop
```

---

## Configuration

Edit `config/company_config.yaml` to change pricing, targets, and agent settings.

Key settings:
- `targets.mrr`: MRR goal (default: $30,000)
- `targets.timeline_months`: Target timeline (default: 12 months)
- `agents.model`: Anthropic model (default: `claude-sonnet-4-6`)
- `financials.base_monthly_expenses`: Operating costs

---

## Development

```bash
# Run with debug logging
python main.py --mode run --log-level DEBUG

# Simulate 6 months
python main.py --mode simulate --months 6

# Check logs
tail -f logs/ian_ai.log
```

---

## License

Internal — IAN.AI Ventures proprietary system.
