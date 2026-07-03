"""Tests for weight validation framework."""

import pytest
import pandas as pd
from src.weight_validator import build_ground_truth, evaluate_weights, run_weight_sensitivity


@pytest.fixture
def mock_risk_df():
    """Create a minimal risk DataFrame for testing."""
    return pd.DataFrame({
        "account_id": [1, 2, 3, 4],
        "account_name": ["Risky Co", "Safe Co", "Mixed Co", "Unknown Co"],
        "csm_sentiment": ["negative", "positive", "neutral", None],
        "competitor_mentions": ["Kontent.ai", "", "", ""],
        "champion_status": ["lost", "active", "active", None],
        "nps_score": [2, 10, 7, 5],
        "sdk_version": ["v3.2.0", "v4.3.0", "v4.1.0", "v4.2.3"],
        "total_tickets": [12, 1, 3, 2],
        "usage_risk_score": [0.8, 0.1, 0.4, 0.3],
        "ticket_risk_score": [0.7, 0.1, 0.3, 0.2],
        "nps_risk_score": [0.9, 0.0, 0.4, 0.5],
        "csm_risk_score": [0.85, 0.1, 0.5, 0.3],
        "sdk_risk_score": [0.85, 0.0, 0.15, 0.0],
        "engagement_risk_score": [0.7, 0.05, 0.3, 0.3],
    })


class TestBuildGroundTruth:
    def test_negative_csm_is_expected_high(self, mock_risk_df):
        df = build_ground_truth(mock_risk_df)
        assert df.loc[0, "expected_high"] == True  # "Risky Co"

    def test_promoter_with_positive_csm_is_expected_low(self, mock_risk_df):
        df = build_ground_truth(mock_risk_df)
        assert df.loc[1, "expected_low"] == True  # "Safe Co"

    def test_competitor_mentions_flag_high(self, mock_risk_df):
        df = build_ground_truth(mock_risk_df)
        assert df.loc[0, "expected_high"] == True  # Has competitors


class TestEvaluateWeights:
    def test_current_weights_separate_well(self, mock_risk_df):
        weights = {"usage": 0.25, "tickets": 0.15, "nps": 0.10,
                   "csm": 0.25, "sdk": 0.15, "engagement": 0.10}
        metrics = evaluate_weights(mock_risk_df, weights)
        assert metrics["separation"] > 0, "High-risk should score higher than low-risk"

    def test_returns_all_metrics(self, mock_risk_df):
        weights = {"usage": 0.25, "tickets": 0.15, "nps": 0.10,
                   "csm": 0.25, "sdk": 0.15, "engagement": 0.10}
        metrics = evaluate_weights(mock_risk_df, weights)
        assert "separation" in metrics
        assert "precision_at_k" in metrics
        assert "recall" in metrics


class TestSensitivityAnalysis:
    def test_returns_5_configs(self, mock_risk_df):
        results = run_weight_sensitivity(mock_risk_df)
        assert len(results) == 5

    def test_current_config_included(self, mock_risk_df):
        results = run_weight_sensitivity(mock_risk_df)
        assert "Current (25/15/10/25/15/10)" in results.index
