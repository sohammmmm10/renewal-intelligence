"""Tests for account name reconciliation."""

import pytest
import pandas as pd
from src.reconciler import fuzzy_match_name, build_account_lookup, build_name_to_id


# Known canonical account names for testing
CANONICAL_NAMES = [
    "BrightPath Solutions",
    "Pinnacle Media Group",
    "Vanguard Retail",
    "Thunderbolt Motors",
    "Crescent Labs",
    "Meridian Health",
    "NovaTech Industries",
]


class TestFuzzyMatchName:
    """Test the fuzzy matching fallback logic."""

    def test_exact_match(self):
        match, score = fuzzy_match_name("BrightPath Solutions", CANONICAL_NAMES)
        assert match == "BrightPath Solutions"
        assert score == 100

    def test_case_insensitive_match(self):
        match, score = fuzzy_match_name("vanguard retail", CANONICAL_NAMES)
        assert match == "Vanguard Retail"
        assert score >= 95

    def test_typo_brightpath(self):
        """BritePath -> BrightPath (missing 'g' and 'h')."""
        match, score = fuzzy_match_name("BritePath Solutions", CANONICAL_NAMES)
        assert match == "BrightPath Solutions"
        assert score >= 65

    def test_typo_pinnacle(self):
        """Pinacle Media -> Pinnacle Media Group (missing 'n')."""
        match, score = fuzzy_match_name("Pinacle Media", CANONICAL_NAMES)
        assert match == "Pinnacle Media Group"
        assert score >= 65

    def test_typo_thunderbolt(self):
        """Thunderbolt Moters -> Thunderbolt Motors (typo in Motors)."""
        match, score = fuzzy_match_name("Thunderbolt Moters", CANONICAL_NAMES)
        assert match == "Thunderbolt Motors"
        assert score >= 65

    def test_no_match_for_unknown(self):
        match, score = fuzzy_match_name("Completely Unknown Corp", CANONICAL_NAMES)
        # Should either return None or a very low score match
        if match is not None:
            assert score < 65

    def test_empty_string_returns_none(self):
        match, score = fuzzy_match_name("", CANONICAL_NAMES)
        assert match is None
        assert score == 0

    def test_short_string_returns_none(self):
        match, score = fuzzy_match_name("AB", CANONICAL_NAMES)
        assert match is None
        assert score == 0


class TestBuildLookups:
    def test_account_lookup(self):
        df = pd.DataFrame({
            "account_id": [1001, 1002],
            "account_name": ["BrightPath Solutions", "NovaTech Industries"],
        })
        lookup = build_account_lookup(df)
        assert lookup[1001] == "BrightPath Solutions"
        assert lookup[1002] == "NovaTech Industries"

    def test_name_to_id(self):
        df = pd.DataFrame({
            "account_id": [1001, 1002],
            "account_name": ["BrightPath Solutions", "NovaTech Industries"],
        })
        mapping = build_name_to_id(df)
        assert mapping["brightpath solutions"] == 1001
        assert mapping["novatech industries"] == 1002
