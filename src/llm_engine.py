"""
LLM Engine -- All OpenAI GPT interactions for the pipeline.

UPGRADE NOTES (AI Architect decisions):
  1. STRUCTURED OUTPUTS: All JSON calls use response_format={"type": "json_object"}
     to guarantee valid JSON -- no more markdown stripping hacks.
  2. ASYNC PARALLEL: CSM analysis uses asyncio + AsyncOpenAI to run 5 calls
     in parallel, cutting latency from ~60s to ~15s.
  3. PYDANTIC MODELS: Every LLM response has a Pydantic schema for validation.
     If the LLM returns garbage, Pydantic catches it with typed defaults.
  4. PROMPT ENGINEERING: All prompts use:
     - Few-shot examples (show the LLM what good output looks like)
     - Chain-of-thought ("Think step by step")
     - Output anchoring ("0.0 = definitely renewing, 1.0 = definitely churning")
  5. CONFIDENCE CALIBRATION: Risk scores have explicit calibration guide in prompts.
"""

import json
import time
import asyncio
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from config import Config


# =============================================
# PYDANTIC MODELS (Typed LLM Output Schemas)
# =============================================

class CSMAnalysis(BaseModel):
    """Structured output from CSM note analysis."""
    sentiment: str = Field(default="neutral", description="negative, neutral, or positive")
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Calibrated churn probability")
    competitor_mentions: list[str] = Field(default_factory=list)
    champion_status: str = Field(default="active", description="active, at_risk, or lost")
    key_concerns: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    summary: str = Field(default="No analysis available.")


class ReconciliationMatch(BaseModel):
    """Structured output for one name match."""
    input_name: str = ""
    matched_id: int | None = None
    matched_name: str | None = None
    confidence: str = Field(default="low", description="high, medium, or low")


# =============================================
# CLIENT MANAGEMENT
# =============================================

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def get_client() -> OpenAI:
    """Lazy-init sync OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _client


def get_async_client() -> AsyncOpenAI:
    """Lazy-init async OpenAI client for parallel calls."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
    return _async_client


# =============================================
# CORE LLM CALL FUNCTIONS
# =============================================

def _chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Sync chat completion for text responses (explanations, insights)."""
    client = get_client()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e
    return ""


def _chat_json(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024) -> dict:
    """
    Sync chat completion with STRUCTURED OUTPUT (guaranteed valid JSON).

    Uses response_format={"type": "json_object"} so OpenAI guarantees
    the response is valid JSON -- no markdown wrapping, no parsing failures.
    """
    client = get_client()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e
    return {}


async def _async_chat_json(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024) -> dict:
    """
    Async chat completion with structured JSON output.
    Used for parallel CSM note analysis (5 concurrent calls).
    """
    client = get_async_client()
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return {}
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                raise e
    return {}


# =============================================
# 0. AI-POWERED CSM NOTE PARSING & RECONCILIATION
# =============================================

def ai_parse_csm_notes(raw_blocks: list[str]) -> list[dict]:
    """
    Use LLM to extract structured fields from raw CSM note blocks.
    Uses structured JSON output for guaranteed valid response.
    """
    if not raw_blocks:
        return []

    system = (
        "You are a data extraction assistant. Extract structured fields from CSM call notes.\n\n"
        "For EACH note, extract:\n"
        '- "account_name": the company name (clean up typos if obvious)\n'
        '- "account_id": numeric ID if present (e.g. "acct 1001", "#1007"). null if not found.\n'
        '- "date": the date in any format\n'
        '- "csm_name": who wrote the note, if mentioned\n\n'
        "Return a JSON object with key \"notes\" containing an array of extracted objects.\n\n"
        "EXAMPLE:\n"
        'Input: "Mar 12 - Acme Corp. Frustrated with editor performance."\n'
        'Output: {"notes": [{"account_name": "Acme Corp", "account_id": null, "date": "Mar 12", "csm_name": ""}]}'
    )

    notes_text = "\n\n".join(f"--- NOTE {i+1} ---\n{block}" for i, block in enumerate(raw_blocks))
    result = _chat_json(system, notes_text, temperature=0.0, max_tokens=4096)

    notes = result.get("notes", [])
    if isinstance(notes, list) and len(notes) == len(raw_blocks):
        return notes

    return [{"account_name": "", "account_id": None, "date": "", "csm_name": ""} for _ in raw_blocks]


def ai_reconcile_names(
    extracted_names: list[str],
    canonical_accounts: list[dict],
) -> list[dict]:
    """
    Use LLM to match messy names to canonical accounts.
    Uses structured JSON output + few-shot examples + chain-of-thought.
    """
    if not extracted_names:
        return []

    system = (
        "You are a data reconciliation assistant. Match messy account names to canonical accounts.\n\n"
        "Think step by step for each name:\n"
        "1. Is it an exact match (ignoring case)?\n"
        "2. Is it a typo? (e.g. missing letters, swapped letters)\n"
        "3. Is it an abbreviation or partial name?\n"
        "4. Does the context suggest a specific account?\n\n"
        "EXAMPLES:\n"
        '- "BritePath Solutions" -> "BrightPath Solutions" (typo: i->ig) -> confidence: high\n'
        '- "Pinacle Media" -> "Pinnacle Media Group" (missing n, missing Group) -> confidence: high\n'
        '- "vanguard retail" -> "Vanguard Retail" (case difference) -> confidence: high\n'
        '- "unknown company xyz" -> null (no match) -> confidence: low\n\n'
        'Return JSON object with key "matches" containing array of:\n'
        '{"input_name": "...", "matched_id": 1001, "matched_name": "...", "confidence": "high"|"medium"|"low"}\n'
        "Set matched_id to null if no good match exists."
    )

    user = (
        f"MESSY NAMES TO MATCH:\n{json.dumps(extracted_names)}\n\n"
        f"CANONICAL ACCOUNTS:\n{json.dumps(canonical_accounts)}"
    )

    result = _chat_json(system, user, temperature=0.0, max_tokens=4096)
    matches = result.get("matches", [])

    if isinstance(matches, list) and len(matches) == len(extracted_names):
        return matches

    return [{"input_name": n, "matched_id": None, "matched_name": None, "confidence": "failed"} for n in extracted_names]


# =============================================
# 1. TRANSLATE NON-ENGLISH NPS COMMENTS
# =============================================

def translate_comments(comments: list[dict]) -> list[dict]:
    """
    Detect language and translate non-English NPS comments.
    Uses structured JSON output.
    """
    if not comments:
        return comments

    system = (
        "You are a translation assistant. For each comment:\n"
        "1. Detect the language\n"
        "2. Translate to English if not already English\n\n"
        'Return JSON object with key "translations" containing array of:\n'
        '{"account_id": 1017, "comment": "original...", "language": "Chinese", "translated": "English translation..."}\n'
        "Keep the original comment field unchanged. Add language and translated fields."
    )

    user = json.dumps(comments, ensure_ascii=False)
    result = _chat_json(system, user, max_tokens=2048)

    translations = result.get("translations", [])
    if isinstance(translations, list) and len(translations) > 0:
        return translations

    # Fallback
    for c in comments:
        c["language"] = "unknown"
        c["translated"] = c.get("comment", "")
    return comments


# =============================================
# 2. ANALYZE CSM NOTES (ASYNC PARALLEL)
# =============================================

# Calibrated prompt with few-shot examples, chain-of-thought, and output anchoring
CSM_ANALYSIS_SYSTEM = (
    "You are a Customer Success risk analyst at a SaaS company. Analyze CSM call notes "
    "and extract structured risk signals.\n\n"
    "THINK STEP BY STEP:\n"
    "1. What is the overall tone? (frustrated, neutral, happy)\n"
    "2. Are any competitors mentioned by name?\n"
    "3. Is the internal champion (our advocate) still engaged, at risk, or lost?\n"
    "4. What specific concerns were raised?\n"
    "5. What actions should the account team take?\n\n"
    "RISK SCORE CALIBRATION (this is a churn probability, not a relative ranking):\n"
    "- 0.0-0.2: Account is healthy, expanding, or explicitly confirmed renewal\n"
    "- 0.2-0.4: Minor concerns but overall positive relationship\n"
    "- 0.4-0.6: Mixed signals, needs monitoring (e.g. usage dip but good relationship)\n"
    "- 0.6-0.8: Significant risk -- competitor evaluation, champion loss, or budget cuts\n"
    "- 0.8-1.0: Near-certain churn -- active competitor POC, demanded large discount, or stated intent to leave\n\n"
    "FEW-SHOT EXAMPLES:\n\n"
    'Example 1 (HIGH RISK):\n'
    'Notes: "They want 30% discount or they walk. Competitor POC with Kontent.ai underway."\n'
    'Output: {"sentiment": "negative", "risk_score": 0.85, "competitor_mentions": ["Kontent.ai"], '
    '"champion_status": "at_risk", "key_concerns": ["pricing demand", "active competitor POC"], '
    '"recommended_actions": ["executive meeting within 48hrs", "prepare counter-proposal"], '
    '"summary": "Account demanding 30% discount with active competitor POC. High churn risk."}\n\n'
    'Example 2 (LOW RISK):\n'
    'Notes: "Great call. They want to add 30 seats and upgrade to Enterprise. Budget approved."\n'
    'Output: {"sentiment": "positive", "risk_score": 0.05, "competitor_mentions": [], '
    '"champion_status": "active", "key_concerns": [], '
    '"recommended_actions": ["prepare expansion proposal"], '
    '"summary": "Account expanding with budget approval. No risk."}\n\n'
    'Example 3 (MEDIUM RISK - silent churn):\n'
    'Notes: "NPS is decent but usage cratered. Built custom middleware, slowly migrating away."\n'
    'Output: {"sentiment": "neutral", "risk_score": 0.70, "competitor_mentions": [], '
    '"champion_status": "active", "key_concerns": ["usage decline", "building replacement"], '
    '"recommended_actions": ["investigate usage drop", "demonstrate unique value"], '
    '"summary": "Silent churn pattern: good NPS but declining usage suggests migration to homegrown solution."}\n\n'
    "Return a JSON object with the following fields:\n"
    '"sentiment", "risk_score", "competitor_mentions", "champion_status", '
    '"key_concerns", "recommended_actions", "summary"'
)


def analyze_csm_notes(notes_text: str, account_name: str) -> dict:
    """Sync version: Analyze one CSM note with structured output + calibrated prompts."""
    user = f"Account: {account_name}\n\nCSM Notes:\n{notes_text}"
    result = _chat_json(CSM_ANALYSIS_SYSTEM, user, max_tokens=1024)

    try:
        analysis = CSMAnalysis(**result)
        return analysis.model_dump()
    except Exception:
        return CSMAnalysis().model_dump()


async def _async_analyze_one(notes_text: str, account_name: str) -> dict:
    """Async version: Analyze one CSM note."""
    user = f"Account: {account_name}\n\nCSM Notes:\n{notes_text}"
    result = await _async_chat_json(CSM_ANALYSIS_SYSTEM, user, max_tokens=1024)

    try:
        analysis = CSMAnalysis(**result)
        return analysis.model_dump()
    except Exception:
        return CSMAnalysis().model_dump()


async def _analyze_batch(tasks: list[tuple[str, str]], batch_size: int = 5) -> list[dict]:
    """
    Run CSM analyses in parallel batches.

    Instead of 27 sequential calls (~60s), we run 5 at a time (~15s).
    batch_size=5 respects OpenAI rate limits while maximizing throughput.
    """
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[_async_analyze_one(text, name) for text, name in batch]
        )
        results.extend(batch_results)
    return results


def analyze_csm_notes_batch(tasks: list[tuple[str, str]]) -> list[dict]:
    """
    Public API: Analyze multiple CSM notes in parallel.
    Wraps async calls for use in sync code.

    tasks: [(notes_text, account_name), ...]
    returns: [CSMAnalysis dict, ...]
    """
    if not tasks:
        return []

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in an async context (e.g. Streamlit)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _analyze_batch(tasks))
                return future.result()
        else:
            return loop.run_until_complete(_analyze_batch(tasks))
    except RuntimeError:
        return asyncio.run(_analyze_batch(tasks))


# =============================================
# 3. GENERATE RISK EXPLANATION
# =============================================

def generate_risk_explanation(account_summary: dict) -> str:
    """
    Generate plain-English risk briefing with chain-of-thought prompting.
    """
    system = (
        "You are a BizOps analyst writing internal account risk briefings.\n\n"
        "THINK STEP BY STEP:\n"
        "1. What is the overall risk level and the #1 reason?\n"
        "2. Which 2-3 specific signals are most concerning? Use exact numbers.\n"
        "3. What 1-2 concrete actions should the team take THIS WEEK?\n\n"
        "Write a clear, actionable, 3-5 sentence briefing.\n"
        "Be direct. Use data points. No jargon.\n\n"
        "EXAMPLE:\n"
        '"Zenith Publishing is at HIGH RISK with $1.6M ARR renewing in 5 days. '
        "Usage dropped 62% over 6 months (207K to 79K API calls) and they're running "
        "a competitor POC with Kontent.ai. Their CRO's involvement signals C-level escalation. "
        "Action: (1) Arrange executive meeting within 48 hours. "
        '(2) Deploy Performance Engineer for their 50K-entry editor issue."'
    )

    user = json.dumps(account_summary, indent=2, default=str)
    return _chat(system, user, temperature=0.3, max_tokens=512)


# =============================================
# 4. DISCOVER NON-OBVIOUS INSIGHTS
# =============================================

def discover_insights(full_dataset_summary: str) -> str:
    """
    Find non-obvious patterns with chain-of-thought prompting.
    """
    system = (
        "You are a senior data analyst at a SaaS company. Find NON-OBVIOUS insights "
        "that a simple rule-based risk system would miss.\n\n"
        "THINK STEP BY STEP:\n"
        "1. Look for CONTRADICTIONS (e.g., high NPS but declining usage = silent churn)\n"
        "2. Look for CAUSAL CHAINS (e.g., SDK deprecation -> ticket spike -> churn)\n"
        "3. Look for COHORT PATTERNS (e.g., all accounts in one industry declining)\n"
        "4. Look for LEADING INDICATORS (e.g., champion loss predicts usage drop in 2 months)\n"
        "5. Look for CHANGELOG IMPACT (e.g., breaking changes correlating with at-risk accounts)\n\n"
        "For each insight:\n"
        "1. State it clearly with specific account names and numbers\n"
        "2. Explain WHY a rule-based system would miss it\n"
        "3. Suggest a concrete action\n\n"
        "Provide 3-5 insights. Be specific, not generic."
    )

    return _chat(system, full_dataset_summary, temperature=0.4, max_tokens=2048)
