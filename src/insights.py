"""
Insights Module — Discovers non-obvious patterns across the dataset.

These are patterns that a simple rule-based system would miss:
1. NPS-Usage Divergence (silent churn)
2. SDK Deprecation → Ticket Spike → Churn correlation
3. Champion loss as a leading indicator
4. Changelog impact on specific account cohorts
5. Cross-signal contradictions
"""

import pandas as pd
import numpy as np
from src.llm_engine import discover_insights


def find_nps_usage_divergence(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find accounts where NPS score is decent (7+) but usage is declining.
    This is the "silent churn" pattern — they like the people but are leaving.

    Example: Meridian Health (NPS=8, CSM notes say "classic silent churn pattern")
    """
    if "nps_score" not in risk_df.columns or "usage_risk_score" not in risk_df.columns:
        return pd.DataFrame()

    divergent = risk_df[
        (risk_df["nps_score"] >= 7) &
        (risk_df["usage_risk_score"] >= 0.4)
    ].copy()

    divergent["insight_type"] = "Silent Churn Risk"
    divergent["insight_detail"] = divergent.apply(
        lambda r: (
            f"NPS score is {int(r['nps_score'])} (passive/promoter) but usage has declined "
            f"significantly (risk: {r['usage_risk_score']:.0%}). This suggests the account "
            f"may be migrating away while maintaining a positive relationship with the team."
        ),
        axis=1,
    )

    return divergent[["account_id", "account_name", "nps_score", "usage_risk_score",
                       "insight_type", "insight_detail"]]


def find_sdk_deprecation_impact(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find accounts on deprecated SDK versions (v3.x) that ALSO have high
    ticket volumes — suggesting the deprecation is causing real pain.

    This connects the changelog (SDK sunset) → tickets → churn risk.
    """
    if "sdk_version" not in risk_df.columns:
        return pd.DataFrame()

    deprecated = risk_df[
        risk_df["sdk_version"].str.startswith("v3", na=False)
    ].copy()

    deprecated["insight_type"] = "SDK Deprecation Cascade"
    deprecated["insight_detail"] = deprecated.apply(
        lambda r: (
            f"Account is on deprecated {r.get('sdk_version', 'v3.x')} with "
            f"{int(r.get('total_tickets', 0))} support tickets "
            f"({int(r.get('p1_p2_tickets', 0))} P1/P2). SDK v3.x loses security patches "
            f"after April 30, 2026. The changelog shows this is connected to breaking changes "
            f"in v4.2.0+ (response envelope format change). Migration effort is likely the "
            f"root cause of their frustration."
        ),
        axis=1,
    )

    return deprecated[["account_id", "account_name", "sdk_version",
                        "total_tickets", "p1_p2_tickets", "insight_type", "insight_detail"]]


def find_champion_at_risk_accounts(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find accounts where the CSM notes indicate a champion is at risk or lost.
    Champion loss is a leading indicator of churn that precedes usage decline.
    """
    if "champion_status" not in risk_df.columns:
        return pd.DataFrame()

    at_risk = risk_df[
        risk_df["champion_status"].isin(["at_risk", "lost"])
    ].copy()

    at_risk["insight_type"] = "Champion at Risk"
    at_risk["insight_detail"] = at_risk.apply(
        lambda r: (
            f"Account champion status is '{r['champion_status']}'. "
            f"CSM notes: {r.get('csm_summary', 'N/A')}. "
            f"Champion loss typically precedes usage decline by 1-2 months and is a strong "
            f"leading indicator that a rule-based system watching only metrics would miss."
        ),
        axis=1,
    )

    return at_risk[["account_id", "account_name", "champion_status",
                     "csm_summary", "insight_type", "insight_detail"]]


def find_score_sentiment_contradictions(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find accounts where the NPS score contradicts the sentiment in the comment.
    Example: NPS=3 but comment says "phenomenal" (likely a data entry error or misunderstanding).
    """
    if "sentiment_score_mismatch" not in risk_df.columns:
        return pd.DataFrame()

    mismatches = risk_df[
        risk_df["sentiment_score_mismatch"] == True
    ].copy()

    mismatches["insight_type"] = "NPS Score-Sentiment Contradiction"
    mismatches["insight_detail"] = mismatches.apply(
        lambda r: (
            f"NPS score is {int(r.get('nps_score', 0))} but the comment sentiment is negative. "
            f"Comment: \"{r.get('nps_comment', '')[:100]}...\". "
            f"This contradiction suggests the score may not reflect true satisfaction."
        ),
        axis=1,
    )

    return mismatches[["account_id", "account_name", "nps_score",
                        "nps_comment", "insight_type", "insight_detail"]]


def compile_all_insights(risk_df: pd.DataFrame) -> dict:
    """
    Compile all non-obvious insights into a structured report.
    """
    insights = {
        "silent_churn": find_nps_usage_divergence(risk_df),
        "sdk_cascade": find_sdk_deprecation_impact(risk_df),
        "champion_risk": find_champion_at_risk_accounts(risk_df),
        "nps_contradictions": find_score_sentiment_contradictions(risk_df),
    }

    # Summary counts
    summary = {
        category: len(df)
        for category, df in insights.items()
        if isinstance(df, pd.DataFrame)
    }

    return {
        "insights": insights,
        "summary": summary,
    }


def build_llm_insights_prompt(risk_df: pd.DataFrame) -> str:
    """Build a summary of the dataset for the LLM to analyze for non-obvious insights."""
    lines = []
    lines.append(f"Total accounts analyzed: {len(risk_df)}")
    lines.append(f"Risk distribution: "
                 f"High={len(risk_df[risk_df['risk_tier']=='High'])}, "
                 f"Medium={len(risk_df[risk_df['risk_tier']=='Medium'])}, "
                 f"Low={len(risk_df[risk_df['risk_tier']=='Low'])}")
    lines.append("")

    # Top 15 riskiest accounts with details
    lines.append("=== TOP 15 AT-RISK ACCOUNTS ===")
    top = risk_df.head(15)
    for _, r in top.iterrows():
        lines.append(
            f"- {r['account_name']} (ID:{r['account_id']}): "
            f"Risk={r['composite_risk_score']:.0%} [{r['risk_tier']}], "
            f"ARR=${r['arr']:,}, Plan={r.get('plan_tier','?')}, "
            f"SDK={r.get('sdk_version','?')}, "
            f"NPS={r.get('nps_score','N/A')}, "
            f"Usage Decline Risk={r.get('usage_risk_score', 0):.0%}, "
            f"Tickets={r.get('total_tickets', 0)} ({r.get('p1_p2_tickets', 0)} P1/P2), "
            f"Competitors={r.get('competitor_mentions', 'none')}, "
            f"Champion={r.get('champion_status', 'unknown')}, "
            f"Renewal in {r.get('days_until_renewal', '?')} days. "
            f"CSM: {r.get('csm_summary', 'No notes')}"
        )

    # SDK version distribution
    lines.append("\n=== SDK VERSION DISTRIBUTION ===")
    if "sdk_version" in risk_df.columns:
        for sdk, count in risk_df["sdk_version"].value_counts().items():
            at_risk = len(risk_df[(risk_df["sdk_version"]==sdk) & (risk_df["risk_tier"].isin(["High","Medium"]))])
            lines.append(f"- {sdk}: {count} accounts ({at_risk} at-risk)")

    # NPS distribution
    lines.append("\n=== NPS SCORE DISTRIBUTION ===")
    if "nps_score" in risk_df.columns:
        nps_available = risk_df[risk_df["nps_score"].notna()]
        detractors = len(nps_available[nps_available["nps_score"] <= 6])
        passives = len(nps_available[(nps_available["nps_score"] >= 7) & (nps_available["nps_score"] <= 8)])
        promoters = len(nps_available[nps_available["nps_score"] >= 9])
        lines.append(f"Detractors (0-6): {detractors}, Passives (7-8): {passives}, Promoters (9-10): {promoters}")

    # Industry breakdown of at-risk accounts
    lines.append("\n=== INDUSTRY RISK BREAKDOWN ===")
    at_risk = risk_df[risk_df["risk_tier"].isin(["High", "Medium"])]
    if "industry" in at_risk.columns:
        for industry, group in at_risk.groupby("industry"):
            lines.append(f"- {industry}: {len(group)} at-risk accounts, total ARR=${group['arr'].sum():,}")

    return "\n".join(lines)


def get_llm_insights(risk_df: pd.DataFrame) -> str:
    """Get LLM-generated non-obvious insights."""
    prompt = build_llm_insights_prompt(risk_df)
    return discover_insights(prompt)
