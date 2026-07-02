# Renewal Risk Intelligence Engine

An AI-powered tool that identifies at-risk SaaS account renewals by ingesting multiple data sources (CRM data, usage metrics, support tickets, CSM notes, NPS surveys, product changelog), reconciling messy data, computing weighted risk scores, and generating plain-English explanations using LLMs.

Built for the Contentstack BizOps team to replace gut-feel renewal forecasting with data-driven, explainable risk intelligence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│  accounts.csv │ usage_metrics.csv │ support_tickets.csv         │
│  csm_notes.txt │ nps_responses.csv │ changelog.md               │
│                                                                  │
│  • CSV/text parsing with type coercion                          │
│  • CSM notes: regex-based structured extraction                 │
│  • Changelog: deprecation/breaking-change detection             │
│  • Fuzzy name matching (RapidFuzz) for reconciliation           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    SIGNAL EXTRACTION LAYER                       │
│                                                                  │
│  Usage Signals ─── 6-month trend analysis (API, users, content) │
│  Ticket Signals ── P1/P2 ratio, open tickets, resolution time   │
│  NPS Signals ───── Score mapping + non-English translation (LLM)│
│  CSM Signals ───── Sentiment analysis via LLM (GPT-4o-mini)    │
│  SDK Risk ──────── Changelog correlation (v3.x deprecation)    │
│  Engagement ────── Active users vs. plan tier benchmarks        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    RISK SCORING LAYER                            │
│                                                                  │
│  Weighted composite score (0.0 – 1.0) per account:             │
│    Usage Decline:   25%    │    CSM Sentiment:   25%            │
│    Support Health:  15%    │    SDK Risk:        15%            │
│    NPS Score:       10%    │    Engagement:      10%            │
│                                                                  │
│  Risk Tiers: High (≥65%) │ Medium (≥40%) │ Low (<40%)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    LLM EXPLANATION LAYER                         │
│                                                                  │
│  • Plain-English risk briefing per at-risk account (GPT-4o-mini)│
│  • Non-obvious insight discovery across the full dataset        │
│  • CSM note sentiment/competitor/champion extraction            │
│  • Non-English NPS comment translation                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    PRESENTATION LAYER                            │
│                                                                  │
│  CLI: Rich terminal output with tables + panels                 │
│  Streamlit: Interactive dashboard with charts, drilldowns,      │
│             signal heatmaps, and insight explorer                │
│  Exports: CSV (risk scores), JSON (briefings, insights)         │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions & Tradeoffs

### What "At Risk" Means (My Definition)

An account is at-risk when **multiple independent signals** converge:

| Signal | Why It Matters | Weight |
|--------|---------------|--------|
| **Usage Decline** | Declining API calls / active users = disengagement | 25% |
| **CSM Sentiment** | Competitor mentions, champion loss, billing disputes | 25% |
| **SDK Deprecation** | Accounts on v3.x face forced migration or security risk | 15% |
| **Support Health** | High P1/P2 tickets + unresolved issues = frustration | 15% |
| **NPS** | Detractor scores + negative sentiment in comments | 10% |
| **Engagement** | Low active users vs. plan capacity = shelfware | 10% |

I weighted CSM notes equally with usage because qualitative signals (competitor evaluations, champion departures) are often the **earliest** churn indicators — preceding metric declines by 1-2 months.

### How the LLM is Used (Not a Gimmick)

The LLM (GPT-4o-mini) is used in four **meaningful** ways:

1. **CSM Note Analysis** — Extracting structured signals (sentiment, competitor names, champion status) from messy, unstructured call notes that no regex could reliably parse
2. **Non-English Translation** — NPS comments in Chinese (account 1017), French (1014), and Spanish (1013) need translation before sentiment analysis
3. **Risk Explanations** — Generating actionable, plain-English briefings that synthesize 6+ signal dimensions into a coherent narrative
4. **Insight Discovery** — Finding cross-cutting patterns (e.g., SDK deprecation cascade) that require reasoning across multiple data sources

### Data Reconciliation Approach

CSM notes are messy by design. My approach:
- **Step 1**: Extract explicit account IDs from notes (`acct 1001`, `#1007`, `account 1016`)
- **Step 2**: Extract account names via regex patterns for various date/separator formats
- **Step 3**: Fuzzy match extracted names against `accounts.csv` using RapidFuzz token_sort_ratio (threshold: 65%)
- Handles: "BritePath" → "BrightPath", "Pinacle" → "Pinnacle", "Thunderbolt Moters" → "Thunderbolt Motors"

### Non-Obvious Insights

Things a rule-based system would miss:

1. **Silent Churn Pattern** — Meridian Health has NPS=8 but CSM notes reveal they're building a homegrown replacement. The score reflects liking the *people*, not the *product*.

2. **SDK Deprecation Cascade** — Accounts on v3.x (like NovaTech, Zenith Publishing, Acme Corp) have disproportionately higher ticket volumes. The changelog reveals v3.x loses security patches April 30, 2026, creating a forced-migration-or-churn dynamic.

3. **NPS Score-Sentiment Contradictions** — Some accounts have high NPS scores but deeply negative comments (or vice versa), suggesting survey fatigue or data quality issues.

4. **Champion Loss as Leading Indicator** — Accounts where the internal champion is "at risk" (e.g., Orion Education's Director of Content nervous about post-merger role) show stable metrics *now* but are likely to decline within 1-2 months.

5. **Changelog-Aware Risk** — The changelog's breaking change (v4.2.0 response envelope format change) correlates with increased tickets from accounts on v4.0.0 and v4.1.0 who haven't upgraded.

## What I'd Do With More Time

- **Historical churn data**: Train a proper ML model (XGBoost/logistic regression) on past churned accounts to learn signal weights empirically
- **Slack/email integration**: Ingest actual communication channels for real-time sentiment
- **Time-series forecasting**: Predict *when* usage will hit critical thresholds, not just that it's declining
- **A/B test risk weights**: Run the model with different weight configurations and validate against actual outcomes
- **Automated alerting**: Trigger Slack notifications when an account crosses a risk threshold
- **CRM integration**: Push risk scores back into Salesforce for the account team
- **Feedback loop**: Let account managers mark predictions as correct/incorrect to improve the model

## What I'd Change for Production

- Replace in-memory processing with a database (PostgreSQL) for persistence
- Add authentication and role-based access to the dashboard
- Implement incremental processing (only re-score accounts with new signals)
- Add unit tests with pytest for each signal extractor
- Use async/batch LLM calls to reduce latency
- Add monitoring for LLM cost tracking
- Deploy via Docker with health checks and auto-restart

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/renewal-intelligence.git
cd renewal-intelligence

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
# Copy env_example.txt to .env and add your OpenAI API key
cp env_example.txt .env     # macOS/Linux
copy env_example.txt .env   # Windows
# Edit .env and set OPENAI_API_KEY=sk-your-key-here

# 5. Run the CLI pipeline
python run_pipeline.py

# 6. Or launch the interactive dashboard
streamlit run app.py
```

### Docker (Optional)
```bash
docker build -t renewal-intelligence .
docker run -e OPENAI_API_KEY=sk-your-key -p 8501:8501 renewal-intelligence
```

## Output Files

After running the pipeline, check the `output/` folder:

| File | Description |
|------|-------------|
| `risk_scored_accounts.csv` | All accounts with composite risk scores and tiers |
| `detailed_signals.csv` | Full signal breakdown per account |
| `account_briefings.json` | LLM-generated risk explanations per account |
| `insights.json` | Summary statistics and non-obvious insights |

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Data Processing | pandas, numpy | Industry standard for tabular data |
| Fuzzy Matching | RapidFuzz | Fast C-extension fuzzy matching for name reconciliation |
| LLM | OpenAI GPT-4o-mini | Cost-effective ($0.15/1M tokens) with strong reasoning |
| Dashboard | Streamlit + Plotly | Rapid interactive UI with zero frontend code |
| CLI Output | Rich | Beautiful terminal tables and progress indicators |

## Project Structure

```
renewal-intelligence/
├── config.py                 # Settings & .env loading
├── run_pipeline.py           # CLI entry point — runs full analysis
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
├── env_example.txt           # Environment variable template
├── src/
│   ├── data_loader.py        # Load & parse all 5 data sources + changelog
│   ├── reconciler.py         # Fuzzy-match CSM note names to account IDs
│   ├── signal_extractor.py   # Compute risk signals from each source
│   ├── risk_scorer.py        # Weighted composite scoring + tier assignment
│   ├── llm_engine.py         # All OpenAI calls (translate, analyze, explain)
│   └── insights.py           # Non-obvious pattern discovery
├── data/                     # Input data files
│   ├── accounts.csv
│   ├── usage_metrics.csv
│   ├── support_tickets.csv
│   ├── csm_notes.txt
│   ├── nps_responses.csv
│   └── changelog.md
└── output/                   # Generated analysis results
```

## License

MIT
