"""
Risk Scorer -- Combines all signal scores into a composite risk score and tier.

Architecture:
    Each signal extractor produces a 0.0-1.0 risk score per account.
    This module applies configurable weights and produces:
    - composite_risk_score (0.0-1.0)
    - risk_tier ("High", "Medium", "Low")
    - confidence_level (how many signals contributed to the score)
    - signal_breakdown (which signals contributed most)

Confidence Calibration:
    The composite score is a WEIGHTED CHURN PROBABILITY, not a relative rank.
    - 0.0-0.2 = Very likely to renew (healthy, expanding)
    - 0.2-0.4 = Likely to renew with minor concerns
    - 0.4-0.6 = Uncertain -- could go either way, needs attention
    - 0.6-0.8 = Likely to churn or downgrade
    - 0.8-1.0 = Very likely to churn (active competitor POC, stated intent to leave)

    Confidence depends on signal coverage:
    - HIGH confidence: 4+ signals available (NPS + CSM + usage + tickets)
    - MEDIUM confidence: 2-3 signals available
    - LOW confidence: 0-1 signals available (mostly defaults)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import Config


def compute_composite_risk(
    accounts_df: pd.DataFrame,
    usage_signals: pd.DataFrame,
    ticket_signals: pd.DataFrame,
    nps_signals: pd.DataFrame,
    csm_signals: pd.DataFrame,
    sdk_signals: pd.DataFrame,
    engagement_signals: pd.DataFrame,
    renewal_window_days: int = Config.RENEWAL_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Merge all signal DataFrames and compute weighted composite risk score.

    Scores ALL 120 accounts regardless of contract date because:
    1. We have usage/ticket/NPS/SDK/engagement data for ALL accounts
    2. Risk doesn't start at the renewal date -- it builds over months
    3. The dataset may be run at any time; filtering by date loses insights
    4. The days_until_renewal column tells the user which ones are urgent
    5. Confidence level tells the user which scores are trustworthy

    The output is sorted by risk score (highest first) so the most
    at-risk accounts surface to the top regardless of renewal date.
    """
    now = pd.Timestamp.now()

    # Score ALL accounts -- risk exists regardless of renewal date
    renewing = accounts_df.copy()

    # Merge all signals onto renewing accounts
    merged = renewing.copy()

    for signals_df, score_col, default in [
        (usage_signals, "usage_risk_score", 0.3),
        (ticket_signals, "ticket_risk_score", 0.2),
        (nps_signals, "nps_risk_score", 0.5),       # Default 0.5 if no NPS response
        (csm_signals, "csm_risk_score", 0.3),        # Default 0.3 if no CSM notes
        (sdk_signals, "sdk_risk_score", 0.2),
        (engagement_signals, "engagement_risk_score", 0.3),
    ]:
        if not signals_df.empty:
            merged = merged.merge(signals_df, on="account_id", how="left", suffixes=("", "_dup"))
            # Drop duplicate columns
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            merged = merged.drop(columns=dup_cols)

        if score_col not in merged.columns:
            merged[score_col] = default
        else:
            merged[score_col] = merged[score_col].fillna(default)

    # Compute weighted composite score
    merged["composite_risk_score"] = (
        merged["usage_risk_score"] * Config.WEIGHT_USAGE_DECLINE
        + merged["ticket_risk_score"] * Config.WEIGHT_SUPPORT_HEALTH
        + merged["nps_risk_score"] * Config.WEIGHT_NPS
        + merged["csm_risk_score"] * Config.WEIGHT_CSM_SENTIMENT
        + merged["sdk_risk_score"] * Config.WEIGHT_SDK_RISK
        + merged["engagement_risk_score"] * Config.WEIGHT_ENGAGEMENT
    ).round(3)

    # Assign risk tier
    merged["risk_tier"] = merged["composite_risk_score"].apply(_score_to_tier)

    # Days until renewal
    merged["days_until_renewal"] = (
        merged["contract_end_date"] - now
    ).dt.days

    # Sort by risk (highest first)
    merged = merged.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)

    # Compute top risk drivers per account
    merged["top_risk_drivers"] = merged.apply(_get_top_drivers, axis=1)

    # Compute confidence level based on signal coverage
    merged["confidence_level"] = merged.apply(_compute_confidence, axis=1)

    return merged


def _score_to_tier(score: float) -> str:
    """Convert composite score to risk tier."""
    if score >= Config.HIGH_RISK_THRESHOLD:
        return "High"
    elif score >= Config.MEDIUM_RISK_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def _get_top_drivers(row: pd.Series) -> str:
    """Identify the top 3 risk drivers for an account."""
    signal_names = {
        "usage_risk_score": "Usage Decline",
        "ticket_risk_score": "Support Issues",
        "nps_risk_score": "Low NPS",
        "csm_risk_score": "CSM Sentiment",
        "sdk_risk_score": "SDK Deprecation",
        "engagement_risk_score": "Low Engagement",
    }

    scores = {name: row.get(col, 0) for col, name in signal_names.items()}
    sorted_drivers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Only include drivers that actually contribute (score > 0.3)
    top = [f"{name} ({score:.0%})" for name, score in sorted_drivers[:3] if score > 0.3]
    return "; ".join(top) if top else "No significant risk drivers"


def _compute_confidence(row: pd.Series) -> str:
    """
    Compute confidence level based on how many REAL signals (not defaults) we have.

    HIGH = 4+ real signals -> we can trust this score
    MEDIUM = 2-3 real signals -> directionally correct but incomplete
    LOW = 0-1 real signals -> mostly guessing, treat with caution
    """
    # Default values that indicate "no data" (these are our fillna defaults)
    defaults = {
        "usage_risk_score": 0.3,
        "ticket_risk_score": 0.2,
        "nps_risk_score": 0.5,
        "csm_risk_score": 0.3,
        "sdk_risk_score": 0.2,
        "engagement_risk_score": 0.3,
    }

    real_signals = 0
    for col, default_val in defaults.items():
        val = row.get(col, default_val)
        # If value differs from default, we have real data for this signal
        if abs(val - default_val) > 0.01:
            real_signals += 1

    if real_signals >= 4:
        return "High"
    elif real_signals >= 2:
        return "Medium"
    else:
        return "Low"


def get_risk_summary(risk_df: pd.DataFrame) -> dict:
    """Compute summary statistics for the risk analysis."""
    return {
        "total_accounts_analyzed": len(risk_df),
        "high_risk_count": len(risk_df[risk_df["risk_tier"] == "High"]),
        "medium_risk_count": len(risk_df[risk_df["risk_tier"] == "Medium"]),
        "low_risk_count": len(risk_df[risk_df["risk_tier"] == "Low"]),
        "total_arr_at_risk": int(risk_df[risk_df["risk_tier"].isin(["High", "Medium"])]["arr"].sum()),
        "high_risk_arr": int(risk_df[risk_df["risk_tier"] == "High"]["arr"].sum()),
        "avg_risk_score": round(risk_df["composite_risk_score"].mean(), 3),
        "median_risk_score": round(risk_df["composite_risk_score"].median(), 3),
    }
