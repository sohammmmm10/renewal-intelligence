# Renewal Risk Intelligence Engine

Predicts which SaaS accounts will churn, explains why, and recommends actions.

Ingests 5 messy data sources + changelog, reconciles inconsistencies, extracts 6 risk signals, scores all 120 accounts, and generates LLM-powered explanations.

**Results:** 120 accounts scored | 6 High Risk | 5 Medium | $4.5M ARR at risk

---

## Quick Start

```bash
git clone https://github.com/sohammmmm10/renewal-intelligence.git
cd renewal-intelligence
python -m venv venv && venv\Scripts\activate       # Windows
pip install -r requirements.txt
copy env_example.txt .env                          # Add your OpenAI API key
python run_pipeline.py                             # CLI pipeline
streamlit run app.py                               # Dashboard
```

---

## Architecture

```
DATA INGESTION ──> SIGNAL EXTRACTION ──> RISK SCORING ──> LLM EXPLANATION
                                                          
6 data sources     6 risk signals        Weighted         Plain-English
CSVs + text +      (0.0-1.0 each)       composite        briefings +
changelog          via math + LLM        score + tier     non-obvious insights
```

### 7-Step Pipeline

| Step | What | How |
|------|------|-----|
| 1 | **Load + Validate** | Parse 5 CSVs + changelog. Validate: duplicates, negative values, missing fields |
| 2 | **Reconcile** | Match messy CSM note names to accounts. AI-first (LLM) + fuzzy fallback. 27/27 matched |
| 3a | **Usage Signals** | Compare first 2 months vs last 2 months. API calls, users, content, workflows |
| 3b | **Ticket Signals** | P1/P2 ratio, open tickets, resolution time, volume |
| 3c | **NPS Signals** | Score mapping + LLM sentiment analysis on comments + non-English translation |
| 3d | **CSM Signals** | LLM extracts sentiment, competitors, champion status from unstructured notes |
| 3e | **SDK Risk** | Cross-reference changelog with usage data. v3.x deprecated = high risk |
| 3f | **Engagement** | Active users vs plan tier benchmarks. Detects shelfware |
| 4 | **Risk Scoring** | Weighted composite (see below). Tiers: High/Medium/Low. Confidence levels |
| 5 | **Explanations** | LLM generates 3-5 sentence actionable briefing per at-risk account |
| 6 | **Insights** | LLM discovers cross-source patterns rules would miss |
| 7 | **Export** | CSV + JSON output files |

---

## Risk Scoring

| Signal | Weight | Why |
|--------|--------|-----|
| Usage Decline | 25% | Hard data: declining API calls = disengagement |
| CSM Sentiment | 25% | Earliest churn signal: competitor mentions, champion loss |
| SDK Deprecation | 15% | v3.x sunset forces migrate-or-leave decision |
| Support Health | 15% | Unresolved P1/P2 tickets = frustration |
| NPS | 10% | Detractor scores, but can be misleading alone |
| Engagement | 10% | Low users vs plan capacity = shelfware risk |

**Tiers:** High (>=60%) | Medium (>=35%) | Low (<35%)

**Calibration:** Scores are churn probabilities, not relative ranks. 0.60 = "60% likely to churn."

**Confidence:** High (4+ real signals), Medium (2-3), Low (0-1 signals available).

---

## How the LLM is Used (7 Places)

| # | Task | Why LLM, not rules? |
|---|------|---------------------|
| 1 | **CSM note entity extraction** | Notes have 5+ formats. Regex breaks on new formats; LLM handles any style |
| 2 | **Name reconciliation** | "BritePath" -> "BrightPath". LLM understands meaning, not just characters |
| 3 | **CSM sentiment analysis** | Extracts sentiment, competitors, champion status from messy human writing |
| 4 | **NPS comment sentiment** | Understands sarcasm, context. Keywords miss "execution fell off a cliff" |
| 5 | **Non-English translation** | Chinese, French, Spanish NPS comments translated before analysis |
| 6 | **Risk explanations** | Synthesizes 6 signals into 3-5 sentence actionable briefing |
| 7 | **Insight discovery** | Finds cross-source patterns: silent churn, SDK cascade, champion loss |

---

## AI Architecture Decisions

### AI-First Reconciliation (3-Tier)
```
Tier 1: Fast path    -> Has "acct 1001"? Direct ID lookup. Free, instant.
Tier 2: AI path      -> Send ALL unmatched names to LLM in 1 batch. Understands typos + context.
Tier 3: Fuzzy fallback-> If LLM fails, RapidFuzz character matching. Safety net.
```
Result: 27/27 matched (vs 26/27 with regex+fuzzy only).

### Structured Outputs
All JSON calls use `response_format={"type": "json_object"}`. No markdown stripping. Pydantic models validate every LLM response.

### Async Parallel
CSM analysis runs 5 concurrent LLM calls via `asyncio + AsyncOpenAI`. 27 notes in ~15s instead of ~60s.

### Prompt Engineering
Every prompt uses: few-shot examples, chain-of-thought ("think step by step"), and score calibration anchors.

---

## Non-Obvious Insights Found

| Insight | Example | Why rules miss it |
|---------|---------|-------------------|
| **Silent Churn** | Meridian Health: NPS=8 but building homegrown replacement | Rules trust NPS score; LLM reads CSM notes revealing migration |
| **SDK Cascade** | v3.x accounts have more tickets -> frustration -> churn | Rules check sources independently; this is a 3-step causal chain |
| **Champion Loss** | Vanguard Retail: champion left, usage stable *for now* | Rules only see current metrics; champion loss predicts future decline |
| **NPS Contradictions** | Score=8 but comment says "execution fell off a cliff" | Rules trust the number; LLM detects sentiment mismatch |
| **Pricing Pressure** | Zenith ($1.6M): demands 30% discount + Kontent.ai POC | Rules don't parse negotiation context from CSM notes |

---

## Testing

```bash
python -m pytest tests/ -v     # 53 tests, all passing
```

| Test File | Tests | What's Covered |
|-----------|-------|----------------|
| `test_data_loader.py` | 16 | Schema, types, CSM parsing, changelog |
| `test_reconciler.py` | 10 | Fuzzy matching typos, edge cases |
| `test_risk_scorer.py` | 11 | Weight math, tier thresholds, confidence |
| `test_data_validator.py` | 8 | Catches duplicates, negatives, out-of-range |
| `test_llm_models.py` | 8 | Pydantic handles malformed LLM responses |

---

## Output Files

| File | Contents |
|------|----------|
| `risk_scored_accounts.csv` | All 120 accounts with scores + tiers |
| `detailed_signals.csv` | Full 6-signal breakdown per account |
| `account_briefings.json` | LLM risk explanations per account |
| `insights.json` | Non-obvious insights + summary stats |

---

## Project Structure

```
├── run_pipeline.py           # CLI entry point (7-step pipeline)
├── app.py                    # Streamlit dashboard
├── config.py                 # Settings, weights, thresholds
├── src/
│   ├── data_loader.py        # Parse all data sources
│   ├── data_validator.py     # Validate data quality before processing
│   ├── reconciler.py         # AI-first name matching + fuzzy fallback
│   ├── signal_extractor.py   # 6 risk signal extractors
│   ├── risk_scorer.py        # Weighted composite scoring + confidence
│   ├── llm_engine.py         # All OpenAI calls (async, structured, Pydantic)
│   └── insights.py           # Non-obvious pattern discovery
├── tests/                    # 53 unit tests
├── data/                     # Input data (5 CSVs + changelog)
└── output/                   # Generated results
```

---

## Tech Stack

| What | Tool | Why |
|------|------|-----|
| LLM | GPT-4o-mini | $0.15/1M tokens, strong reasoning, structured output support |
| Data | pandas | Industry standard for tabular data |
| Matching | RapidFuzz | Fast C-extension fuzzy matching |
| Validation | Pydantic | Typed LLM output schemas with defaults |
| Async | asyncio + AsyncOpenAI | 3x speedup for parallel LLM calls |
| Dashboard | Streamlit + Plotly | Interactive UI, zero frontend code |
| CLI | Rich | Terminal tables and progress output |
| Tests | pytest | 53 tests covering all modules |

---

## What I'd Do With More Time

- Train ML model (XGBoost) on historical churn data to learn weights empirically
- Time-series forecasting: predict *when* usage hits critical threshold
- Slack alerts when account crosses risk threshold
- CRM integration: push scores to Salesforce
- Feedback loop: let account managers validate predictions

## What I'd Change for Production

- PostgreSQL for persistence instead of in-memory
- Auth + role-based access on dashboard
- Incremental processing (only re-score accounts with new signals)
- Docker deployment with health checks
- LLM cost monitoring and rate limiting
