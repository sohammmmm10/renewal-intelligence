"""Tests for data loading and parsing."""

import pytest
import pandas as pd
from src.data_loader import (
    load_accounts, load_usage_metrics, load_support_tickets,
    load_nps_responses, parse_csm_notes, parse_changelog, load_all_data,
)


class TestLoadAccounts:
    def test_loads_all_120_accounts(self):
        df = load_accounts()
        assert len(df) == 120

    def test_has_required_columns(self):
        df = load_accounts()
        required = ["account_id", "account_name", "arr", "contract_end_date", "plan_tier"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_account_ids_are_integers(self):
        df = load_accounts()
        assert df["account_id"].dtype in ("int64", "int32")

    def test_arr_is_positive(self):
        df = load_accounts()
        assert (df["arr"] > 0).all(), "All accounts should have positive ARR"

    def test_contract_dates_are_parsed(self):
        df = load_accounts()
        assert pd.api.types.is_datetime64_any_dtype(df["contract_end_date"])

    def test_plan_tiers_are_valid(self):
        df = load_accounts()
        valid = {"Starter", "Growth", "Scale", "Enterprise"}
        actual = set(df["plan_tier"].unique())
        assert actual.issubset(valid), f"Unexpected plan tiers: {actual - valid}"


class TestLoadUsageMetrics:
    def test_loads_720_rows(self):
        df = load_usage_metrics()
        assert len(df) == 720  # 120 accounts x 6 months

    def test_six_months_per_account(self):
        df = load_usage_metrics()
        months_per_account = df.groupby("account_id")["month"].nunique()
        assert (months_per_account == 6).all(), "Each account should have 6 months of data"

    def test_no_negative_api_calls(self):
        df = load_usage_metrics()
        assert (df["api_calls"] >= 0).all(), "API calls should never be negative"


class TestParseCSMNotes:
    def test_parses_27_notes(self):
        notes = parse_csm_notes()
        assert len(notes) == 27

    def test_each_note_has_raw_text(self):
        notes = parse_csm_notes()
        for note in notes:
            assert len(note.raw_text) > 10, "Each note should have substantial text"

    def test_some_notes_have_account_ids(self):
        notes = parse_csm_notes()
        with_ids = [n for n in notes if n.account_id is not None]
        assert len(with_ids) >= 5, "At least 5 notes should have explicit account IDs"

    def test_some_notes_have_account_names(self):
        notes = parse_csm_notes()
        with_names = [n for n in notes if n.account_name]
        assert len(with_names) >= 20, "At least 20 notes should have extracted names"


class TestParseChangelog:
    def test_parses_entries(self):
        entries = parse_changelog()
        assert len(entries) > 0, "Should parse at least some changelog entries"

    def test_has_deprecation_entries(self):
        entries = parse_changelog()
        deprecations = [e for e in entries if e.category == "deprecation"]
        assert len(deprecations) >= 1, "Should find SDK deprecation notice"


class TestLoadAllData:
    def test_returns_all_keys(self):
        data = load_all_data()
        expected_keys = {"accounts", "usage", "tickets", "nps", "csm_notes", "changelog"}
        assert set(data.keys()) == expected_keys
