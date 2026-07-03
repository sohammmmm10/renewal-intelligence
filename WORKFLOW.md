# How the Pipeline Works (Step-by-Step)

This document walks through exactly what happens when you run `python run_pipeline.py`.

---

## Overview

```
INPUT                          PROCESS                         OUTPUT
─────                          ───────                         ──────
accounts.csv (120)      ──>    Load + Validate          ──>    risk_scored_accounts.csv
usage_metrics.csv (720) ──>    Reconcile CSM names      ──>    detailed_signals.csv
support_tickets.csv     ──>    Extract 6 risk signals   ──>    account_briefings.json
nps_responses.csv       ──>    Score + filter to 90d    ──>    insights.json
csm_notes.txt (27)      ──>    Generate explanations
changelog.md            ──>    Discover insights
```

---

## Step 1: Load + Validate Data

**File:** `src/data_loader.py`, `src/data_validator.py`

**What happens:**
- Reads 5 CSV/text files + 1 markdown changelog
- Parses dates, casts types, fills missing values
- Validates data quality before anything else runs

**Validation catches:**
- Duplicate account IDs (removes them)
- Negative ARR values (clamps to 0)
- NPS scores outside 0-10 range (clamps)
- Unknown ticket statuses (warns: "Resolved" not in expected set)
- 11 NPS responses with empty comments (warns)

**Result:** Clean data ready for processing. 2 warnings, 0 errors.

```
120 accounts | 720 usage rows | 271 tickets | 98 NPS | 27 CSM notes
```

---

## Step 2: Reconcile CSM Note Names

**File:** `src/data_loader.py` (parsing), `src/reconciler.py` (matching)

**Problem:** CSM notes are messy. Names are misspelled, formats vary:
```
"BritePath Solutions"     (typo: missing 'g' and 'h')
"Pinacle Media"           (typo: missing 'n', missing 'Group')
"meridian health"         (lowercase)
"acct 1001"               (ID instead of name)
```

**Solution: 3-tier hybrid approach**

```
TIER 1: LLM Entity Extraction (PRIMARY)
  Send all 27 raw note blocks to ai_parse_csm_notes() in one LLM batch call.
  LLM extracts: account_name, account_id, csm_name, date from any format.
  If LLM fails -> fall through to regex.

TIER 2: Regex Fallback
  For any fields LLM missed, regex patterns try to extract:
  - Account IDs: "acct 1001", "#1007", "account 1016"
  - Names: date-separator-name patterns

TIER 3: Name Matching (Reconciler)
  Now we have 27 extracted names. Match them to the 120 canonical accounts:

  Fast path: Exact match (case-insensitive)? -> Done. Free, instant.
       |
       v  (unmatched names)
  AI path:  Send ALL unmatched to LLM in 1 batch call.
            LLM understands "BritePath" = "BrightPath". ~2s.
       |
       v  (if LLM fails or low confidence)
  Fuzzy:    RapidFuzz character-level matching. Safety net.
```

**Result:** 27/27 CSM notes matched to accounts.

---

## Step 3: Extract 6 Risk Signals

Each signal produces a 0.0-1.0 risk score per account. Higher = more risky.

### 3a: Usage Decline (math, no LLM)

**File:** `src/signal_extractor.py` -> `compute_usage_signals()`

**How:** Compare first 2 months average vs last 2 months average for each metric.

```
Example: NovaTech Industries
  API calls:  Month 1-2 avg = 45,000  |  Month 5-6 avg = 22,000  |  -51% -> risk 0.51
  Active users: 12 -> 5  |  -58% -> risk 0.58
  
  Weighted: api_calls(30%) + content(25%) + users(30%) + workflows(15%) = 0.50
```

### 3b: Support Tickets (math, no LLM)

**File:** `src/signal_extractor.py` -> `compute_ticket_signals()`

**How:** Scores 4 components:
```
P1/P2 ratio     (30%) -- more critical tickets = more risk
Open/Escalated   (30%) -- unresolved issues = frustration
Resolution time  (20%) -- >48hr avg = slow support
Volume           (20%) -- more tickets = more problems

Example: NovaTech has 14 tickets (8 P1/P2), 6 open -> ticket_risk = 0.74
```

### 3c: NPS Signals (math + LLM)

**File:** `src/signal_extractor.py` -> `compute_nps_signals()`

**How:**
1. Map NPS score to risk: 0-6 (detractor) = 0.8-1.0, 7-8 = 0.3-0.45, 9-10 = 0.0-0.1
2. Translate non-English comments via LLM (Chinese, French, Spanish)
3. Send ALL comments to LLM in one batch for sentiment analysis
4. LLM detects score-sentiment contradictions (e.g., NPS=8 but "fell off a cliff")
5. Apply risk boost/reduction from LLM sentiment

```
Example: Atlas Financial
  NPS score = 8 -> base risk = 0.30
  Comment: "execution has fallen off a cliff" -> LLM says: negative, contradicts score
  Boosted risk = 0.30 + 0.25 = 0.55
```

### 3d: CSM Sentiment (LLM, async parallel)

**File:** `src/signal_extractor.py` -> `compute_csm_signals()`

**How:**
1. Group notes by account (some accounts have multiple notes)
2. Send ALL 27 analyses to LLM in parallel batches of 5 (async)
3. LLM extracts per account: sentiment, risk_score, competitors, champion_status, concerns, actions

**Prompt includes:**
- Few-shot examples (3 examples: high/low/medium risk)
- Chain-of-thought ("Think step by step: 1. Tone? 2. Competitors? 3. Champion?")
- Score anchoring ("0.8-1.0 = active competitor POC, stated intent to leave")

```
Example: Zenith Publishing
  CSM note: "They want 30% discount or they walk. Competitor POC with Kontent.ai."
  LLM output: sentiment=negative, risk_score=0.85, competitors=["Kontent.ai"],
              champion_status=at_risk, concerns=["pricing", "competitor POC"]
```

**Speed:** 27 notes in ~15s (async) vs ~60s (sequential)

### 3e: SDK Risk (math, no LLM)

**File:** `src/signal_extractor.py` -> `compute_sdk_risk()`

**How:** Cross-reference changelog.md with each account's SDK version from usage data.

```
v3.x -> 0.70-0.85 risk (deprecated, security patches end April 30, 2026)
v4.0  -> 0.25 risk (needs upgrade for locale fix + breaking change)
v4.1  -> 0.15 risk (affected by v4.2.0 response envelope change)
v4.2+ -> 0.00 risk (current, no issues)

Example: Zenith Publishing on v3.1.2 -> sdk_risk = 0.85
```

### 3f: Engagement (math, no LLM)

**File:** `src/signal_extractor.py` -> `compute_engagement_signals()`

**How:** Compare active users to plan tier benchmarks.

```
Plan benchmarks: Starter=3, Growth=12, Scale=35, Enterprise=80 expected users

Example: An Enterprise account with 20 active users (expected 80)
  Engagement ratio = 20/80 = 0.25 -> engagement_risk = 0.90 (severe shelfware)
```

---

## Step 4: Score + Filter to 90-Day Window

**File:** `src/risk_scorer.py`

**How:**
1. Merge all 6 signal scores onto accounts
2. Compute weighted composite: `usage(25%) + csm(25%) + sdk(15%) + tickets(15%) + nps(10%) + engagement(10%)`
3. Assign tier: High (>=0.60) | Medium (>=0.35) | Low (<0.35)
4. Compute `days_until_renewal` from fixed snapshot date (2026-04-10)
5. **Filter: keep only accounts with 0 <= days_until_renewal <= 90**
6. Compute confidence level: High (4+ real signals) | Medium (2-3) | Low (0-1)

```
Example: Vanguard Retail
  usage=0.57 * 0.25 = 0.143
  csm=0.80  * 0.25 = 0.200
  sdk=0.85  * 0.15 = 0.128
  tickets=0.62 * 0.15 = 0.093
  nps=0.93  * 0.10 = 0.093
  engagement=0.70 * 0.10 = 0.070
  ─────────────────────────
  COMPOSITE = 0.72 -> HIGH RISK
  Renewing in 53 days (within 90-day window -> included)
  Confidence: High (5 real signals)
```

**Result:** 27 accounts in window | 6 High | 4 Medium | 17 Low | $4M ARR at risk

---

## Step 5: Generate LLM Explanations

**File:** `src/llm_engine.py` -> `generate_risk_explanation()`

**How:** For each High/Medium account (10 total), send all signal data to LLM. LLM writes a 3-5 sentence actionable briefing.

**Prompt includes:**
- Chain-of-thought ("1. What's the #1 reason? 2. Which signals? 3. What actions?")
- Example briefing so LLM knows the format
- Instruction: "Be direct. Use data points. No jargon."

```
Example output for Zenith Publishing:
"Zenith Publishing is at HIGH RISK with $1.6M ARR renewing in 25 days.
They're demanding a 30% discount while running a competitor POC with Kontent.ai.
Usage has declined and they're on deprecated SDK v3.1.2. Action: (1) Arrange
executive meeting within 48 hours. (2) Prepare counter-proposal with migration
support to demonstrate continued value."
```

---

## Step 6: Discover Non-Obvious Insights

**File:** `src/insights.py`, `src/llm_engine.py` -> `discover_insights()`

**Two approaches run in parallel:**

### Structured Pattern Detection (code, no LLM)
Scans for specific cross-source patterns:

| Pattern | Logic | Found |
|---------|-------|-------|
| Silent Churn | NPS >= 7 AND usage_risk >= 0.4 | 3 accounts |
| SDK Cascade | SDK v3.x AND tickets > 5 | 8 accounts |
| Champion Loss | champion_status = "at_risk" or "lost" | 9 accounts |
| NPS Contradictions | LLM flagged score-sentiment mismatch | varies |

### LLM Insight Discovery
Sends full dataset summary (top 15 accounts + SDK distribution + NPS breakdown) to LLM.
Prompt: "Find NON-OBVIOUS patterns. Look for contradictions, causal chains, cohort patterns."

```
Example insight found:
"Meridian Health has NPS=8 but CSM notes reveal they're building a homegrown
replacement. This is silent churn -- they like the people, not the product.
A rule-based system would trust the NPS score and miss this entirely."
```

---

## Step 7: Validate Weights

**File:** `src/weight_validator.py`

**How:** Tests if the chosen weights (25/15/10/25/15/10) actually work.

1. Build pseudo-ground-truth from the data:
   - Expected HIGH: negative CSM + competitors + champion lost + NPS <= 3
   - Expected LOW: positive CSM + NPS >= 9 + active champion
2. Test 5 weight configurations (current, equal, usage-heavy, CSM-heavy, no-CSM)
3. Measure: separation (high avg - low avg), precision@K, recall

```
Result: separation=0.449, precision=83%, recall=50%
Note: This is circular (CSM sentiment is both signal and label).
Real validation needs actual churn/renew outcome data.
```

---

## Step 8: Export

**Output files:**

| File | What | Rows |
|------|------|------|
| `risk_scored_accounts.csv` | All in-window accounts with scores, tiers, explanations | 27 |
| `detailed_signals.csv` | Full signal breakdown (all 6 scores + metadata) | 27 |
| `account_briefings.json` | LLM-generated risk briefings per account | 27 |
| `insights.json` | Summary stats + LLM insights + pattern counts | 1 |

---

## Data Flow Diagram

```
accounts.csv ──────────────────────────────────────────────┐
usage_metrics.csv ──> compute_usage_signals() ──> 0.0-1.0 ─┤
                  ──> compute_sdk_risk() ──────> 0.0-1.0 ─┤
                  ──> compute_engagement() ────> 0.0-1.0 ─┤
support_tickets ───> compute_ticket_signals() ─> 0.0-1.0 ─┤
nps_responses ─────> compute_nps_signals() ───> 0.0-1.0 ─┤  compute_composite_risk()
                     (LLM: translate + sentiment)         ├──> weighted score ──> filter 0-90d
csm_notes.txt ─────> ai_parse_csm_notes() [LLM]          │         │
                     ──> reconcile [LLM + fuzzy]          │         v
                     ──> compute_csm_signals() [LLM] ──> 0.0-1.0 ─┤  27 scored accounts
changelog.md ──────> parse_changelog() ───────────────────┘         │
                                                                    v
                                                    generate_risk_explanation() [LLM]
                                                    discover_insights() [LLM]
                                                    validate_weights()
                                                            │
                                                            v
                                                    CSV + JSON exports
```

---

## LLM Call Summary

| When | Function | Calls | Async? | Purpose |
|------|----------|-------|--------|---------|
| Step 2 | `ai_parse_csm_notes()` | 1 batch | No | Extract entities from 27 notes |
| Step 2 | `ai_reconcile_names()` | 1 batch | No | Match names to accounts |
| Step 3c | `translate_comments()` | 1 batch | No | Translate non-English NPS |
| Step 3c | `analyze_nps_sentiment_batch()` | 1 batch | No | Sentiment on all NPS comments |
| Step 3d | `_async_analyze_one()` x 27 | 27 (5 parallel) | Yes | CSM note sentiment analysis |
| Step 5 | `generate_risk_explanation()` x 10 | 10 | No | Risk briefings |
| Step 6 | `discover_insights()` | 1 | No | Cross-source patterns |

**Total: ~42 LLM calls per run. Cost: ~$0.05 with gpt-4o-mini.**
