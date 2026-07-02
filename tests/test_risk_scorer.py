"""Tests for risk scoring math and tier assignment."""

import pytest
import pandas as pd
import numpy as np
from config import Config


class TestRiskTierThresholds:
    """Verify risk tier thresholds are correctly configured."""

    def test_high_risk_threshold(self):
        assert Config.HIGH_RISK_THRESHOLD == 0.60
        assert Config.HIGH_RISK_THRESHOLD > Config.MEDIUM_RISK_THRESHOLD

    def test_medium_risk_threshold(self):
        assert Config.MEDIUM_RISK_THRESHOLD == 0.35

    def test_weights_sum_to_one(self):
        total = (
            Config.WEIGHT_USAGE_DECLINE
            + Config.WEIGHT_SUPPORT_HEALTH
            + Config.WEIGHT_NPS
            + Config.WEIGHT_CSM_SENTIMENT
            + Config.WEIGHT_SDK_RISK
            + Config.WEIGHT_ENGAGEMENT
        )
        assert abs(total - 1.0) < 0.01, f"Weights should sum to ~1.0, got {total}"


class TestScoreToTier:
    """Test the score-to-tier mapping function."""

    def test_high_risk(self):
        from src.risk_scorer import _score_to_tier
        assert _score_to_tier(0.80) == "High"
        assert _score_to_tier(0.65) == "High"
        assert _score_to_tier(0.60) == "High"

    def test_medium_risk(self):
        from src.risk_scorer import _score_to_tier
        assert _score_to_tier(0.55) == "Medium"
        assert _score_to_tier(0.40) == "Medium"
        assert _score_to_tier(0.35) == "Medium"

    def test_low_risk(self):
        from src.risk_scorer import _score_to_tier
        assert _score_to_tier(0.34) == "Low"
        assert _score_to_tier(0.20) == "Low"
        assert _score_to_tier(0.00) == "Low"


class TestCompositeScoreMath:
    """Verify the weighted composite scoring formula is correct."""

    def test_all_zeros_gives_zero(self):
        """All signals at 0 should produce composite of 0."""
        score = (
            0.0 * Config.WEIGHT_USAGE_DECLINE
            + 0.0 * Config.WEIGHT_SUPPORT_HEALTH
            + 0.0 * Config.WEIGHT_NPS
            + 0.0 * Config.WEIGHT_CSM_SENTIMENT
            + 0.0 * Config.WEIGHT_SDK_RISK
            + 0.0 * Config.WEIGHT_ENGAGEMENT
        )
        assert score == 0.0

    def test_all_ones_gives_one(self):
        """All signals at 1.0 should produce composite of ~1.0."""
        score = (
            1.0 * Config.WEIGHT_USAGE_DECLINE
            + 1.0 * Config.WEIGHT_SUPPORT_HEALTH
            + 1.0 * Config.WEIGHT_NPS
            + 1.0 * Config.WEIGHT_CSM_SENTIMENT
            + 1.0 * Config.WEIGHT_SDK_RISK
            + 1.0 * Config.WEIGHT_ENGAGEMENT
        )
        assert abs(score - 1.0) < 0.01

    def test_high_csm_and_usage_dominates(self):
        """CSM (25%) + Usage (25%) = 50% of score. High values should push to high risk."""
        score = (
            0.9 * Config.WEIGHT_USAGE_DECLINE     # 25%
            + 0.0 * Config.WEIGHT_SUPPORT_HEALTH   # 15%
            + 0.0 * Config.WEIGHT_NPS              # 10%
            + 0.9 * Config.WEIGHT_CSM_SENTIMENT    # 25%
            + 0.0 * Config.WEIGHT_SDK_RISK         # 15%
            + 0.0 * Config.WEIGHT_ENGAGEMENT       # 10%
        )
        # 0.9 * 0.25 + 0.9 * 0.25 = 0.45 -- should be Medium risk
        assert score >= Config.MEDIUM_RISK_THRESHOLD


class TestConfidenceLevel:
    """Test confidence level computation."""

    def test_high_confidence_with_many_signals(self):
        from src.risk_scorer import _compute_confidence
        row = pd.Series({
            "usage_risk_score": 0.8,     # differs from default 0.3
            "ticket_risk_score": 0.6,    # differs from default 0.2
            "nps_risk_score": 0.9,       # differs from default 0.5
            "csm_risk_score": 0.7,       # differs from default 0.3
            "sdk_risk_score": 0.85,      # differs from default 0.2
            "engagement_risk_score": 0.3, # same as default
        })
        assert _compute_confidence(row) == "High"

    def test_low_confidence_with_defaults(self):
        from src.risk_scorer import _compute_confidence
        row = pd.Series({
            "usage_risk_score": 0.3,     # default
            "ticket_risk_score": 0.2,    # default
            "nps_risk_score": 0.5,       # default
            "csm_risk_score": 0.3,       # default
            "sdk_risk_score": 0.2,       # default
            "engagement_risk_score": 0.3, # default
        })
        assert _compute_confidence(row) == "Low"
