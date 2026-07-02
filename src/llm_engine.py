"""
LLM Engine — All OpenAI GPT interactions for the pipeline.

Handles:
1. Translating non-English NPS comments
2. Analyzing CSM notes for risk signals (sentiment, competitor mentions, etc.)
3. Generating plain-English risk explanations per account
4. Discovering non-obvious insights across the dataset
"""

import json
import time
from openai import OpenAI
from config import Config


_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy-init OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _client


def _chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Make a chat completion call with retry logic."""
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


# ─────────────────────────────────────────────
# 0. AI-POWERED CSM NOTE PARSING & RECONCILIATION
# ─────────────────────────────────────────────

def ai_parse_csm_notes(raw_blocks: list[str]) -> list[dict]:
    """
    Use LLM to extract structured fields from raw CSM note blocks.

    WHY AI instead of regex?
    - CSM notes have 5+ different date/name formats that regex can't generalize
    - Regex breaks on any new format; LLM handles ANY human writing style
    - LLM understands context: "talked to Sarah's healthcare team" → Meridian Health
    - One LLM call replaces 10+ fragile regex patterns

    Input:  ["Mar 12 - Acme Corp. They're frustrated...", ...]
    Output: [{"account_name": "Acme Corp", "account_id": null, "date": "Mar 12", "csm_name": ""}, ...]
    """
    if not raw_blocks:
        return []

    system = (
        "You are a data extraction assistant. You will receive a list of raw CSM (Customer Success Manager) "
        "call notes. Each note is messy and has inconsistent formatting.\n\n"
        "For EACH note, extract:\n"
        '- "account_name": the customer/company name mentioned (clean it up, remove typos if obvious)\n'
        '- "account_id": any numeric account ID mentioned (look for patterns like "acct 1001", "#1007", '
        '"account 1016"). Set to null if not found.\n'
        '- "date": the date of the note in any format found\n'
        '- "csm_name": the CSM/person who wrote the note, if mentioned\n\n'
        "Return a JSON array with one object per note, in the same order as input.\n"
        "Return ONLY valid JSON, no markdown."
    )

    # Send notes as numbered list for clarity
    notes_text = "\n\n".join(
        f"--- NOTE {i+1} ---\n{block}" for i, block in enumerate(raw_blocks)
    )

    raw = _chat(system, notes_text, temperature=0.0, max_tokens=4096)

    # Clean markdown wrapping
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]

    try:
        results = json.loads(raw)
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass

    # Fallback: return empty dicts
    return [{"account_name": "", "account_id": None, "date": "", "csm_name": ""} for _ in raw_blocks]


def ai_reconcile_names(
    extracted_names: list[str],
    canonical_accounts: list[dict],
) -> list[dict]:
    """
    Use LLM to match messy extracted names to canonical account names.

    WHY AI instead of fuzzy matching?
    - Fuzzy matching only compares character similarity ("Pinacle" vs "Pinnacle" = 85%)
    - LLM understands SEMANTIC similarity ("the healthcare account" → Meridian Health)
    - LLM can resolve abbreviations, nicknames, partial names
    - LLM sees the FULL list of options and picks the best one with reasoning
    - Fuzzy matching fails when names are structurally different ("Acme" vs "Acme Corporation")

    Input:
      extracted_names: ["BritePath Solutions", "Pinacle Media", "vanguard retail", ...]
      canonical_accounts: [{"account_id": 1001, "account_name": "BrightPath Solutions"}, ...]

    Output: [{"input_name": "BritePath Solutions", "matched_id": 1001,
              "matched_name": "BrightPath Solutions", "confidence": "high"}, ...]
    """
    if not extracted_names:
        return []

    system = (
        "You are a data reconciliation assistant. You will receive:\n"
        "1. A list of MESSY account names extracted from CSM notes (may have typos, wrong casing, abbreviations)\n"
        "2. A list of CANONICAL account names with their IDs from the official database\n\n"
        "Your job: match each messy name to the correct canonical account.\n\n"
        "Rules:\n"
        "- Match based on meaning, not just character similarity\n"
        '- If a name is clearly a typo (e.g. "BritePath" = "BrightPath"), match it\n'
        '- If a name is an abbreviation or partial (e.g. "Acme" = "Acme Corp"), match it\n'
        "- If you're unsure, set confidence to 'low'\n"
        "- If no match exists, set matched_id to null\n\n"
        "Return a JSON array with one object per input name:\n"
        '{"input_name": "...", "matched_id": 1001, "matched_name": "...", "confidence": "high"|"medium"|"low"}\n\n'
        "Return ONLY valid JSON, no markdown."
    )

    user = (
        f"MESSY NAMES TO MATCH:\n{json.dumps(extracted_names)}\n\n"
        f"CANONICAL ACCOUNTS:\n{json.dumps(canonical_accounts)}"
    )

    raw = _chat(system, user, temperature=0.0, max_tokens=4096)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]

    try:
        results = json.loads(raw)
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass

    return [{"input_name": n, "matched_id": None, "matched_name": None, "confidence": "failed"} for n in extracted_names]


# ─────────────────────────────────────────────
# 1. TRANSLATE NON-ENGLISH NPS COMMENTS
# ─────────────────────────────────────────────

def translate_comments(comments: list[dict]) -> list[dict]:
    """
    Detect and translate non-English NPS comments.

    Input: [{"account_id": 1017, "comment": "产品功能还可以..."}]
    Output: Same list with added "translated" and "language" fields.
    """
    if not comments:
        return comments

    system = (
        "You are a translation assistant. For each comment, detect the language and "
        "translate to English if not already English. Return a JSON array with the same "
        "structure, adding 'language' (detected language) and 'translated' (English translation, "
        "or the original if already English) fields. Return ONLY valid JSON, no markdown."
    )

    user = json.dumps(comments, ensure_ascii=False)
    raw = _chat(system, user, max_tokens=2048)

    # Clean potential markdown wrapping
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return originals with unknown language
        for c in comments:
            c["language"] = "unknown"
            c["translated"] = c.get("comment", "")
        return comments


# ─────────────────────────────────────────────
# 2. ANALYZE CSM NOTES FOR RISK SIGNALS
# ─────────────────────────────────────────────

def analyze_csm_notes(notes_text: str, account_name: str) -> dict:
    """
    Use LLM to extract structured risk signals from CSM notes for one account.

    Returns:
    {
        "sentiment": "negative" | "neutral" | "positive",
        "risk_score": 0.0 - 1.0,
        "competitor_mentions": ["Hygraph", "Strapi"],
        "champion_status": "active" | "at_risk" | "lost",
        "key_concerns": ["billing dispute", "editor performance"],
        "recommended_actions": ["schedule exec call", "offer discount"],
        "summary": "Brief 1-2 sentence summary..."
    }
    """
    system = (
        "You are a Customer Success analyst. Analyze the following CSM call notes for a "
        "SaaS account and extract structured risk signals. Be precise and data-driven.\n\n"
        "Return ONLY a valid JSON object with these fields:\n"
        '- "sentiment": "negative", "neutral", or "positive"\n'
        '- "risk_score": float 0.0 (no risk) to 1.0 (certain churn)\n'
        '- "competitor_mentions": list of competitor names mentioned\n'
        '- "champion_status": "active", "at_risk", or "lost"\n'
        '- "key_concerns": list of specific concerns raised\n'
        '- "recommended_actions": list of suggested actions\n'
        '- "summary": 1-2 sentence summary of the situation\n\n'
        "Return ONLY valid JSON, no markdown formatting."
    )

    user = f"Account: {account_name}\n\nCSM Notes:\n{notes_text}"
    raw = _chat(system, user, max_tokens=1024)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "sentiment": "neutral",
            "risk_score": 0.5,
            "competitor_mentions": [],
            "champion_status": "active",
            "key_concerns": [],
            "recommended_actions": [],
            "summary": "Could not parse CSM note analysis.",
        }


# ─────────────────────────────────────────────
# 3. GENERATE RISK EXPLANATION PER ACCOUNT
# ─────────────────────────────────────────────

def generate_risk_explanation(account_summary: dict) -> str:
    """
    Generate a plain-English explanation of why an account is at risk
    and what action the account team should consider.

    account_summary contains all computed signals for the account.
    """
    system = (
        "You are a BizOps analyst writing internal account risk briefings for the sales "
        "and customer success team. Write a clear, actionable, 3-5 sentence explanation.\n\n"
        "Structure:\n"
        "1. State the risk level and primary reason\n"
        "2. List 2-3 specific signals that contributed\n"
        "3. Recommend 1-2 concrete actions the team should take\n\n"
        "Be direct, specific, and avoid jargon. Use data points where available."
    )

    user = json.dumps(account_summary, indent=2, default=str)
    return _chat(system, user, temperature=0.3, max_tokens=512)


# ─────────────────────────────────────────────
# 4. DISCOVER NON-OBVIOUS INSIGHTS
# ─────────────────────────────────────────────

def discover_insights(full_dataset_summary: str) -> str:
    """
    Ask the LLM to identify non-obvious patterns across the entire dataset
    that a simple rule-based system would miss.
    """
    system = (
        "You are a senior data analyst at a SaaS company. You've been given a summary of "
        "account health signals across 120 accounts. Your job is to find NON-OBVIOUS insights "
        "that a simple rule-based risk system would miss.\n\n"
        "Look for:\n"
        "- Contradictions (e.g., high NPS but declining usage = silent churn)\n"
        "- Correlated risks (e.g., SDK deprecation → ticket spike → churn risk)\n"
        "- Cohort patterns (e.g., all accounts in a specific industry declining)\n"
        "- Leading indicators (e.g., champion loss precedes usage drop by 2 months)\n"
        "- Changelog impact (e.g., breaking changes correlating with at-risk accounts)\n\n"
        "Provide 3-5 specific, data-backed insights. For each:\n"
        "1. State the insight clearly\n"
        "2. Explain why a rule-based system would miss it\n"
        "3. Suggest what action to take\n\n"
        "Be specific — reference account names and numbers where possible."
    )

    return _chat(system, full_dataset_summary, temperature=0.4, max_tokens=2048)
