"""
Weight Validator -- Empirical validation of risk scoring weights.

PROBLEM: We chose weights (25/15/10/25/15/10) based on domain intuition.
But are they optimal? Without historical churn labels, we can't train a model.

SOLUTION: Use the data itself as pseudo-ground-truth:
  - CSM notes with negative sentiment + competitor mentions = SHOULD be high risk
  - CSM notes with positive sentiment + no concerns = SHOULD be low risk
  - Accounts on deprecated SDK v3.x with high tickets = SHOULD be high risk

Then test: do our weights correctly rank risky accounts above healthy ones?
If yes, the weights are validated. If not, we find better ones.

This is a RANKING validation (not probability calibration), which is the
best we can do without actual churn/renew outcome labels.
"""

import pandas as pd
import numpy as np
from itertools import product


def build_ground_truth(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pseudo-ground-truth labels from strong signals in the data.

    An account is labeled "expected_high_risk" if ANY of these are true:
    - CSM sentiment is "negative" (the CSM literally says things are bad)
    - Competitors are mentioned in CSM notes (actively evaluating alternatives)
    - Champion status is "lost" (internal advocate gone)
    - NPS <= 3 (extreme detractor)
    - On deprecated SDK v3.x AND has >5 support tickets

    An account is labeled "expected_low_risk" if ALL of these are true:
    - CSM sentiment is "positive" or no CSM notes
    - No competitors mentioned
    - Champion is "active"
    - NPS >= 9
    """
    df = risk_df.copy()

    # Expected high risk indicators
    negative_csm = df.get("csm_sentiment", pd.Series(dtype=str)).isin(["negative"])
    has_competitors = df.get("competitor_mentions", pd.Series(dtype=str)).fillna("").str.len() > 0
    champion_lost = df.get("champion_status", pd.Series(dtype=str)).isin(["lost"])
    extreme_detractor = df.get("nps_score", pd.Series(dtype=float)).fillna(5) <= 3
    sdk_v3_with_tickets = (
        df.get("sdk_version", pd.Series(dtype=str)).fillna("").str.startswith("v3") &
        (df.get("total_tickets", pd.Series(dtype=float)).fillna(0) > 5)
    )

    df["expected_high"] = (
        negative_csm | has_competitors | champion_lost |
        extreme_detractor | sdk_v3_with_tickets
    )

    # Expected low risk
    positive_csm = df.get("csm_sentiment", pd.Series(dtype=str)).isin(["positive"])
    no_csm = df.get("csm_sentiment", pd.Series(dtype=str)).isna()
    no_competitors = ~has_competitors
    champion_active = df.get("champion_status", pd.Series(dtype=str)).isin(["active"]) | df.get("champion_status", pd.Series(dtype=str)).isna()
    promoter = df.get("nps_score", pd.Series(dtype=float)).fillna(5) >= 9

    df["expected_low"] = (
        (positive_csm | no_csm) & no_competitors & champion_active & promoter
    )

    return df


def compute_score_with_weights(risk_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute composite score with given weight configuration."""
    return (
        risk_df.get("usage_risk_score", 0.3) * weights["usage"]
        + risk_df.get("ticket_risk_score", 0.2) * weights["tickets"]
        + risk_df.get("nps_risk_score", 0.5) * weights["nps"]
        + risk_df.get("csm_risk_score", 0.3) * weights["csm"]
        + risk_df.get("sdk_risk_score", 0.2) * weights["sdk"]
        + risk_df.get("engagement_risk_score", 0.3) * weights["engagement"]
    )


def evaluate_weights(risk_df: pd.DataFrame, weights: dict) -> dict:
    """
    Evaluate a weight configuration against pseudo-ground-truth.

    Metrics:
    - separation: avg score of expected-high minus avg score of expected-low
      (higher = better discrimination between risky and healthy)
    - precision_at_k: what % of top-K scored accounts are expected-high
    - recall: what % of expected-high accounts are in the top tier
    - auc_proxy: area under pseudo-ROC (rank-based)
    """
    df = build_ground_truth(risk_df)
    scores = compute_score_with_weights(df, weights)

    high_mask = df["expected_high"]
    low_mask = df["expected_low"]

    high_scores = scores[high_mask]
    low_scores = scores[low_mask]

    # Separation: how well do scores distinguish expected-high from expected-low
    separation = 0.0
    if len(high_scores) > 0 and len(low_scores) > 0:
        separation = high_scores.mean() - low_scores.mean()

    # Precision at K: of the top-K accounts by score, how many are expected-high?
    k = max(1, int(high_mask.sum()))
    top_k_ids = scores.nlargest(k).index
    precision_at_k = df.loc[top_k_ids, "expected_high"].mean() if k > 0 else 0.0

    # Recall: what fraction of expected-high accounts are in the top tier (>= 0.60)?
    recall = 0.0
    if high_mask.sum() > 0:
        flagged_high = scores >= 0.60
        recall = (flagged_high & high_mask).sum() / high_mask.sum()

    return {
        "separation": round(separation, 4),
        "precision_at_k": round(precision_at_k, 4),
        "recall": round(recall, 4),
        "avg_high_score": round(high_scores.mean(), 4) if len(high_scores) > 0 else 0,
        "avg_low_score": round(low_scores.mean(), 4) if len(low_scores) > 0 else 0,
        "n_expected_high": int(high_mask.sum()),
        "n_expected_low": int(low_mask.sum()),
    }


def run_weight_sensitivity(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test multiple weight configurations to find which performs best.

    Tests 5 configurations:
    1. Current weights (25/15/10/25/15/10)
    2. Equal weights (16.7% each)
    3. Usage-heavy (40/10/10/20/10/10)
    4. CSM-heavy (15/10/10/40/15/10)
    5. Balanced-no-CSM (30/20/20/0/20/10) -- what if we had no CSM notes?
    """
    configs = {
        "Current (25/15/10/25/15/10)": {
            "usage": 0.25, "tickets": 0.15, "nps": 0.10,
            "csm": 0.25, "sdk": 0.15, "engagement": 0.10,
        },
        "Equal (16.7% each)": {
            "usage": 0.167, "tickets": 0.167, "nps": 0.167,
            "csm": 0.167, "sdk": 0.167, "engagement": 0.167,
        },
        "Usage-heavy (40/10/10/20/10/10)": {
            "usage": 0.40, "tickets": 0.10, "nps": 0.10,
            "csm": 0.20, "sdk": 0.10, "engagement": 0.10,
        },
        "CSM-heavy (15/10/10/40/15/10)": {
            "usage": 0.15, "tickets": 0.10, "nps": 0.10,
            "csm": 0.40, "sdk": 0.15, "engagement": 0.10,
        },
        "No-CSM (30/20/20/0/20/10)": {
            "usage": 0.30, "tickets": 0.20, "nps": 0.20,
            "csm": 0.00, "sdk": 0.20, "engagement": 0.10,
        },
    }

    results = []
    for name, weights in configs.items():
        metrics = evaluate_weights(risk_df, weights)
        metrics["config"] = name
        results.append(metrics)

    return pd.DataFrame(results).set_index("config")


def validate_current_weights(risk_df: pd.DataFrame) -> dict:
    """
    Run full validation of current weight configuration.
    Returns validation results + sensitivity comparison.
    """
    current_weights = {
        "usage": 0.25, "tickets": 0.15, "nps": 0.10,
        "csm": 0.25, "sdk": 0.15, "engagement": 0.10,
    }

    current_metrics = evaluate_weights(risk_df, current_weights)
    sensitivity = run_weight_sensitivity(risk_df)

    # Determine if current weights are optimal
    best_config = sensitivity["separation"].idxmax()
    current_rank = sensitivity["separation"].rank(ascending=False).loc["Current (25/15/10/25/15/10)"]

    return {
        "current_metrics": current_metrics,
        "sensitivity_table": sensitivity,
        "best_config": best_config,
        "current_rank": int(current_rank),
        "is_optimal": current_rank <= 2,  # Top 2 = good enough
    }
