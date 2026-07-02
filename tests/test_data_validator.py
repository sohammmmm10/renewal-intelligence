"""Tests for data validation layer."""

import pytest
import pandas as pd
from src.data_validator import (
    validate_accounts, validate_usage, validate_tickets, validate_nps,
)


class TestValidateAccounts:
    def test_valid_data_passes(self):
        df = pd.DataFrame({
            "account_id": [1001, 1002],
            "account_name": ["Acme", "Beta"],
            "arr": [50000, 100000],
            "contract_end_date": pd.to_datetime(["2026-09-01", "2026-10-01"]),
            "plan_tier": ["Growth", "Enterprise"],
        })
        _, result = validate_accounts(df)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_catches_duplicate_ids(self):
        df = pd.DataFrame({
            "account_id": [1001, 1001, 1002],
            "account_name": ["Acme", "Acme Dup", "Beta"],
            "arr": [50000, 50000, 100000],
            "contract_end_date": pd.to_datetime(["2026-09-01", "2026-09-01", "2026-10-01"]),
            "plan_tier": ["Growth", "Growth", "Enterprise"],
        })
        cleaned, result = validate_accounts(df)
        assert len(cleaned) == 2  # Duplicate removed
        assert any("Duplicate" in w for w in result.warnings)

    def test_catches_negative_arr(self):
        df = pd.DataFrame({
            "account_id": [1001],
            "account_name": ["Acme"],
            "arr": [-5000],
            "contract_end_date": pd.to_datetime(["2026-09-01"]),
            "plan_tier": ["Growth"],
        })
        cleaned, result = validate_accounts(df)
        assert cleaned.iloc[0]["arr"] == 0  # Clamped to 0
        assert any("negative ARR" in w for w in result.warnings)

    def test_missing_column_is_error(self):
        df = pd.DataFrame({"account_id": [1001]})
        _, result = validate_accounts(df)
        assert not result.is_valid
        assert len(result.errors) > 0


class TestValidateNPS:
    def test_catches_out_of_range_scores(self):
        df = pd.DataFrame({
            "account_id": [1001, 1002],
            "score": [15, -3],
        })
        cleaned, result = validate_nps(df)
        assert cleaned.iloc[0]["score"] == 10  # Clamped
        assert cleaned.iloc[1]["score"] == 0   # Clamped
        assert any("outside 0-10" in w for w in result.warnings)

    def test_valid_nps_passes(self):
        df = pd.DataFrame({
            "account_id": [1001, 1002],
            "score": [8, 3],
            "verbatim_comment": ["Great", "Bad"],
        })
        _, result = validate_nps(df)
        assert result.is_valid


class TestValidateUsage:
    def test_catches_negative_api_calls(self):
        df = pd.DataFrame({
            "account_id": [1001],
            "month": ["2026-01"],
            "api_calls": [-100],
            "content_entries_created": [50],
            "active_users": [10],
            "workflows_triggered": [5],
            "sdk_version": ["v4.3.0"],
        })
        cleaned, result = validate_usage(df)
        assert cleaned.iloc[0]["api_calls"] == 0  # Clamped
        assert any("negative" in w for w in result.warnings)


class TestValidateTickets:
    def test_catches_unknown_priority(self):
        df = pd.DataFrame({
            "account_id": [1001],
            "priority": ["P5"],
            "status": ["Open"],
        })
        _, result = validate_tickets(df)
        assert any("Unknown priority" in w for w in result.warnings)
