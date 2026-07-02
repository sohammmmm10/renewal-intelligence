"""Tests for Pydantic models and LLM output validation."""

import pytest
from src.llm_engine import CSMAnalysis, ReconciliationMatch


class TestCSMAnalysisModel:
    """Test that the Pydantic model handles all edge cases."""

    def test_valid_input(self):
        analysis = CSMAnalysis(
            sentiment="negative",
            risk_score=0.85,
            competitor_mentions=["Kontent.ai"],
            champion_status="at_risk",
            key_concerns=["pricing", "competitor POC"],
            recommended_actions=["exec meeting"],
            summary="High risk account.",
        )
        assert analysis.risk_score == 0.85
        assert analysis.sentiment == "negative"
        assert len(analysis.competitor_mentions) == 1

    def test_empty_input_uses_defaults(self):
        """LLM returns empty dict -- Pydantic should use safe defaults."""
        analysis = CSMAnalysis()
        assert analysis.sentiment == "neutral"
        assert analysis.risk_score == 0.5
        assert analysis.competitor_mentions == []
        assert analysis.champion_status == "active"
        assert analysis.summary == "No analysis available."

    def test_risk_score_clamped_to_range(self):
        """Risk score > 1.0 should be rejected by Pydantic."""
        with pytest.raises(Exception):
            CSMAnalysis(risk_score=1.5)

    def test_risk_score_negative_rejected(self):
        """Negative risk score should be rejected."""
        with pytest.raises(Exception):
            CSMAnalysis(risk_score=-0.1)

    def test_partial_input_fills_defaults(self):
        """LLM returns only some fields -- others should default."""
        analysis = CSMAnalysis(sentiment="positive", risk_score=0.1)
        assert analysis.sentiment == "positive"
        assert analysis.risk_score == 0.1
        assert analysis.competitor_mentions == []  # default
        assert analysis.champion_status == "active"  # default

    def test_model_dump(self):
        """Ensure model can be serialized to dict."""
        analysis = CSMAnalysis(sentiment="negative", risk_score=0.7)
        d = analysis.model_dump()
        assert isinstance(d, dict)
        assert d["sentiment"] == "negative"
        assert d["risk_score"] == 0.7


class TestReconciliationMatchModel:
    def test_valid_match(self):
        match = ReconciliationMatch(
            input_name="BritePath Solutions",
            matched_id=1001,
            matched_name="BrightPath Solutions",
            confidence="high",
        )
        assert match.matched_id == 1001
        assert match.confidence == "high"

    def test_no_match(self):
        match = ReconciliationMatch(input_name="Unknown Corp")
        assert match.matched_id is None
        assert match.confidence == "low"
