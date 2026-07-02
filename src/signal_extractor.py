"""
Signal Extractor — Computes quantitative risk signals from each data source.

Produces a normalized 0.0–1.0 risk score per signal category per account.
Higher = more risky.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from src.data_loader import CSMNote, ChangelogEntry
from src.reconciler import get_csm_notes_by_account
from src.llm_engine import analyze_csm_notes, analyze_csm_notes_batch, translate_comments, analyze_nps_sentiment_batch


# ─────────────────────────────────────────────
# 1. USAGE DECLINE SIGNALS
# ─────────────────────────────────────────────

def compute_usage_signals(usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute usage decline signals per account.

    Metrics:
    - API call trend (6-month slope, normalized)
    - Content creation trend
    - Active user trend
    - Workflow trigger trend

    Returns DataFrame with account_id and usage_risk_score (0-1).
    """
    results = []

    for account_id, group in usage_df.groupby("account_id"):
        group = group.sort_values("month")

        if len(group) < 3:
            results.append({"account_id": account_id, "usage_risk_score": 0.5})
            continue

        # Compute % change from first 2 months avg to last 2 months avg
        signals = {}
        for col in ["api_calls", "content_entries_created", "active_users", "workflows_triggered"]:
            values = group[col].values
            early_avg = np.mean(values[:2])
            late_avg = np.mean(values[-2:])

            if early_avg > 0:
                pct_change = (late_avg - early_avg) / early_avg
            else:
                pct_change = 0.0

            # Convert to risk: -50% change → 1.0 risk, +50% → 0.0 risk
            risk = max(0.0, min(1.0, -pct_change))
            signals[f"{col}_decline"] = risk

        # Weighted combination
        usage_risk = (
            signals["api_calls_decline"] * 0.30
            + signals["content_entries_created_decline"] * 0.25
            + signals["active_users_decline"] * 0.30
            + signals["workflows_triggered_decline"] * 0.15
        )

        # Also capture absolute latest values for context
        latest = group.iloc[-1]
        results.append({
            "account_id": account_id,
            "usage_risk_score": round(usage_risk, 3),
            "api_calls_latest": int(latest["api_calls"]),
            "active_users_latest": int(latest["active_users"]),
            "api_calls_decline_pct": round(signals["api_calls_decline"] * -100, 1),
            "active_users_decline_pct": round(signals["active_users_decline"] * -100, 1),
            "sdk_version": latest["sdk_version"],
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 2. SUPPORT TICKET SIGNALS
# ─────────────────────────────────────────────

def compute_ticket_signals(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute support health signals per account.

    Metrics:
    - Total ticket count (normalized by percentile)
    - P1 + P2 (high-priority) ticket ratio
    - Open/Escalated ticket ratio
    - Average resolution time vs. benchmark
    - Recent ticket velocity (last 90 days vs. prior)

    Returns DataFrame with account_id and ticket_risk_score (0-1).
    """
    results = []
    now = pd.Timestamp.now()
    ninety_days_ago = now - pd.Timedelta(days=90)

    for account_id, group in tickets_df.groupby("account_id"):
        total = len(group)
        high_priority = len(group[group["priority"].isin(["P1", "P2"])])
        open_escalated = len(group[group["status"].isin(["Open", "Escalated"])])

        # Resolution time analysis
        resolved = group[group["resolution_time_hours"].notna()]
        avg_resolution = resolved["resolution_time_hours"].mean() if len(resolved) > 0 else 0

        # Recent velocity
        recent = group[group["created_date"] >= ninety_days_ago]
        recent_count = len(recent)

        # Scoring components
        # High priority ratio: more P1/P2 = more risk
        hp_ratio = high_priority / max(total, 1)

        # Open/escalated ratio: more unresolved = more risk
        open_ratio = open_escalated / max(total, 1)

        # Resolution time risk: >48 hours avg is concerning
        resolution_risk = min(1.0, avg_resolution / 96.0) if avg_resolution > 0 else 0.3

        # Volume risk: normalize by total (will be percentile-adjusted later)
        volume_risk = min(1.0, total / 15.0)

        ticket_risk = (
            hp_ratio * 0.30
            + open_ratio * 0.30
            + resolution_risk * 0.20
            + volume_risk * 0.20
        )

        results.append({
            "account_id": account_id,
            "ticket_risk_score": round(ticket_risk, 3),
            "total_tickets": total,
            "p1_p2_tickets": high_priority,
            "open_escalated": open_escalated,
            "avg_resolution_hours": round(avg_resolution, 1),
            "recent_ticket_count": recent_count,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 3. NPS SIGNALS
# ─────────────────────────────────────────────

def compute_nps_signals(nps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NPS risk signals per account.

    Scoring:
    - NPS 0-6 (Detractor) → high risk
    - NPS 7-8 (Passive) → medium risk
    - NPS 9-10 (Promoter) → low risk
    - Also translates non-English comments and checks for sentiment-score mismatch

    Returns DataFrame with account_id and nps_risk_score (0-1).
    """
    # First, identify and translate non-English comments
    non_english = []
    for _, row in nps_df.iterrows():
        comment = row["verbatim_comment"]
        if comment and not all(ord(c) < 256 for c in comment):
            non_english.append({
                "account_id": int(row["account_id"]),
                "comment": comment,
            })

    translations = {}
    if non_english:
        try:
            translated = translate_comments(non_english)
            for item in translated:
                translations[item["account_id"]] = {
                    "translated": item.get("translated", item.get("comment", "")),
                    "language": item.get("language", "unknown"),
                }
        except Exception:
            pass

    # Build base NPS risk scores from numeric score
    base_results = []
    for _, row in nps_df.iterrows():
        account_id = row["account_id"]
        score = row["score"]
        comment = row["verbatim_comment"]

        # NPS score -> risk
        if score <= 6:
            nps_risk = 0.8 + (6 - score) * 0.033  # 0 -> 1.0, 6 -> 0.8
        elif score <= 8:
            nps_risk = 0.3 + (8 - score) * 0.15   # 7 -> 0.45, 8 -> 0.3
        else:
            nps_risk = 0.1 * (10 - score)           # 9 -> 0.1, 10 -> 0.0

        # Get translation if available
        translated_comment = comment
        detected_language = "English"
        if account_id in translations:
            translated_comment = translations[account_id]["translated"]
            detected_language = translations[account_id]["language"]

        base_results.append({
            "account_id": account_id,
            "nps_risk_score": nps_risk,
            "nps_score": score,
            "nps_comment": comment,
            "translated_comment": translated_comment,
            "detected_language": detected_language,
        })

    # Use LLM to analyze ALL comments with non-empty text (replaces keyword list)
    comments_for_llm = []
    for r in base_results:
        comment_text = r["translated_comment"] or r["nps_comment"]
        if comment_text and len(comment_text.strip()) > 5:
            comments_for_llm.append({
                "account_id": r["account_id"],
                "score": r["nps_score"],
                "comment": comment_text,
            })

    # One batch LLM call for all NPS comments
    sentiment_results = {}
    if comments_for_llm:
        try:
            llm_sentiments = analyze_nps_sentiment_batch(comments_for_llm)
            for s in llm_sentiments:
                sentiment_results[s["account_id"]] = s
        except Exception:
            pass  # Fallback: no sentiment adjustment

    # Apply LLM sentiment adjustments to base scores
    results = []
    for r in base_results:
        aid = r["account_id"]
        sentiment_data = sentiment_results.get(aid, {})
        risk_boost = sentiment_data.get("risk_boost", 0.0)
        score_contradicts = sentiment_data.get("score_contradicts", False)
        comment_sentiment = sentiment_data.get("sentiment", "neutral")

        # Apply LLM-detected risk boost
        adjusted_risk = min(1.0, max(0.0, r["nps_risk_score"] + risk_boost))

        results.append({
            "account_id": aid,
            "nps_risk_score": round(adjusted_risk, 3),
            "nps_score": r["nps_score"],
            "nps_comment": r["nps_comment"],
            "translated_comment": r["translated_comment"],
            "detected_language": r["detected_language"],
            "comment_sentiment": comment_sentiment,
            "sentiment_score_mismatch": score_contradicts,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 4. CSM SENTIMENT SIGNALS (LLM-powered)
# ─────────────────────────────────────────────

def compute_csm_signals(
    csm_notes: list[CSMNote],
    accounts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use LLM to analyze CSM notes and extract risk signals.

    UPGRADED: Uses async parallel batch processing (5 concurrent LLM calls)
    to reduce latency from ~60s to ~15s.

    Returns DataFrame with account_id and csm_risk_score (0-1) + structured analysis.
    """
    grouped = get_csm_notes_by_account(csm_notes)

    if not grouped:
        return pd.DataFrame(columns=[
            "account_id", "csm_risk_score", "csm_sentiment", "competitor_mentions",
            "champion_status", "key_concerns", "csm_recommended_actions", "csm_summary",
        ])

    # Prepare batch of (notes_text, account_name) tuples
    account_ids = []
    batch_tasks = []

    for account_id, notes in grouped.items():
        account_row = accounts_df[accounts_df["account_id"] == account_id]
        account_name = account_row["account_name"].iloc[0] if len(account_row) > 0 else f"Account {account_id}"
        combined_text = "\n\n".join(n.raw_text for n in notes)

        account_ids.append(account_id)
        batch_tasks.append((combined_text, account_name))

    # Run all analyses in parallel (5 concurrent calls)
    analyses = analyze_csm_notes_batch(batch_tasks)

    # Build results
    results = []
    for account_id, analysis in zip(account_ids, analyses):
        results.append({
            "account_id": account_id,
            "csm_risk_score": analysis.get("risk_score", 0.5),
            "csm_sentiment": analysis.get("sentiment", "neutral"),
            "competitor_mentions": ", ".join(analysis.get("competitor_mentions", [])),
            "champion_status": analysis.get("champion_status", "unknown"),
            "key_concerns": "; ".join(analysis.get("key_concerns", [])),
            "csm_recommended_actions": "; ".join(analysis.get("recommended_actions", [])),
            "csm_summary": analysis.get("summary", ""),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 5. SDK / CHANGELOG RISK SIGNALS
# ─────────────────────────────────────────────

def compute_sdk_risk(
    usage_df: pd.DataFrame,
    changelog: list[ChangelogEntry],
) -> pd.DataFrame:
    """
    Compute SDK-related risk based on:
    - Whether account is on a deprecated SDK version (v3.x)
    - Whether account is affected by breaking changes
    - Proximity to deprecation deadlines

    The changelog reveals:
    - SDK v3.x stops receiving security patches after April 30, 2026
    - REST API v2 endpoints sunset April 30, 2026
    - SDK v4.2.0+ has breaking response envelope change
    - Legacy editor removal in v4.4.0 (May 2026)
    """
    # Get latest SDK version per account
    latest_sdk = usage_df.sort_values("month").groupby("account_id").last()["sdk_version"]

    results = []
    for account_id, sdk_version in latest_sdk.items():
        risk = 0.0
        risk_factors = []

        # v3.x accounts: deprecated, no security patches after April 30, 2026
        if sdk_version.startswith("v3"):
            risk += 0.7
            risk_factors.append(f"On deprecated {sdk_version} (security patches end April 30, 2026)")

            # v3.1.x and v3.2.x are even more at risk (oldest)
            if sdk_version in ("v3.1.2", "v3.2.0"):
                risk += 0.15
                risk_factors.append("On oldest SDK version, maximum migration effort required")

        # v4.0.0 accounts: missed the locale fallback fix in v4.2, and the breaking
        # response envelope change in v4.2.0+ means they need to update
        elif sdk_version == "v4.0.0":
            risk += 0.25
            risk_factors.append("On v4.0.0, needs upgrade for locale fix and response envelope change")

        # v4.1.0 accounts: also affected by breaking response envelope change
        elif sdk_version == "v4.1.0":
            risk += 0.15
            risk_factors.append("On v4.1.0, affected by v4.2.0 response envelope breaking change")

        # v4.2.3+ and v4.3.0: current, low risk
        elif sdk_version in ("v4.2.3", "v4.3.0"):
            risk += 0.0
            risk_factors.append(f"On current {sdk_version}, no immediate SDK risk")

        results.append({
            "account_id": account_id,
            "sdk_risk_score": round(min(1.0, risk), 3),
            "sdk_version": sdk_version,
            "sdk_risk_factors": "; ".join(risk_factors),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 6. ENGAGEMENT SIGNALS
# ─────────────────────────────────────────────

def compute_engagement_signals(
    usage_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute engagement/adoption signals:
    - Active users vs. what's expected for the plan tier
    - Content creation velocity
    - Workflow adoption

    Detects "shelfware" situations (paying for seats nobody uses).
    """
    # Expected active user ranges by plan tier
    tier_benchmarks = {
        "Starter": {"min_users": 2, "expected_users": 3},
        "Growth": {"min_users": 8, "expected_users": 12},
        "Scale": {"min_users": 25, "expected_users": 35},
        "Enterprise": {"min_users": 60, "expected_users": 80},
    }

    latest_usage = usage_df.sort_values("month").groupby("account_id").last()
    merged = latest_usage.merge(
        accounts_df[["account_id", "plan_tier", "arr"]],
        on="account_id",
        how="left",
    )

    results = []
    for _, row in merged.iterrows():
        tier = row.get("plan_tier", "Growth")
        benchmark = tier_benchmarks.get(tier, {"min_users": 5, "expected_users": 10})
        active_users = row["active_users"]
        expected = benchmark["expected_users"]

        # Engagement ratio: active_users / expected
        engagement_ratio = active_users / max(expected, 1)

        # Risk scoring: below 50% of expected = high risk
        if engagement_ratio < 0.3:
            engagement_risk = 0.9  # Severe shelfware
        elif engagement_ratio < 0.5:
            engagement_risk = 0.7
        elif engagement_ratio < 0.75:
            engagement_risk = 0.4
        elif engagement_ratio < 1.0:
            engagement_risk = 0.2
        else:
            engagement_risk = 0.05  # Good adoption

        # ARR per active user — higher = more risk if low engagement
        arr = row.get("arr", 0)
        arr_per_user = arr / max(active_users, 1)

        results.append({
            "account_id": int(row["account_id"]),
            "engagement_risk_score": round(engagement_risk, 3),
            "active_users": int(active_users),
            "expected_users": expected,
            "engagement_ratio": round(engagement_ratio, 2),
            "arr_per_active_user": round(arr_per_user, 0),
        })

    return pd.DataFrame(results)
