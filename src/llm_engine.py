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
