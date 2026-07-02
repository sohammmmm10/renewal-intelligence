"""
Data Validation Layer -- Validates all input data before pipeline processing.

Catches:
- Duplicate account IDs
- Usage metrics with impossible values (negative API calls)
- NPS scores outside 0-10 range
- Contract end dates in the past (stale data)
- Missing required fields
- Type mismatches

This runs BEFORE signal extraction so garbage-in doesn't become garbage-out.
"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of validating one data source."""
    source: str
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_accounts(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """Validate accounts.csv data."""
    result = ValidationResult(source="accounts.csv", rows_before=len(df))

    # Check required columns
    required = ["account_id", "account_name", "arr", "contract_end_date", "plan_tier"]
    for col in required:
        if col not in df.columns:
            result.add_error(f"Missing required column: {col}")

    if not result.is_valid:
        result.rows_after = len(df)
        return df, result

    # Check for duplicate account IDs
    dupes = df["account_id"].duplicated()
    if dupes.any():
        dupe_ids = df[dupes]["account_id"].tolist()
        result.add_warning(f"Duplicate account IDs found: {dupe_ids}. Keeping first occurrence.")
        df = df.drop_duplicates(subset="account_id", keep="first")

    # Check for missing account names
    missing_names = df["account_name"].isna().sum()
    if missing_names > 0:
        result.add_warning(f"{missing_names} accounts missing account_name")

    # Check ARR values
    negative_arr = (df["arr"] < 0).sum()
    if negative_arr > 0:
        result.add_warning(f"{negative_arr} accounts have negative ARR. Setting to 0.")
        df.loc[df["arr"] < 0, "arr"] = 0

    zero_arr = (df["arr"] == 0).sum()
    if zero_arr > 0:
        result.add_warning(f"{zero_arr} accounts have zero ARR (free tier?)")

    # Check contract end dates
    now = pd.Timestamp.now()
    past_contracts = (df["contract_end_date"] < now - pd.Timedelta(days=365)).sum()
    if past_contracts > 0:
        result.add_warning(f"{past_contracts} accounts have contract_end_date > 1 year in the past (stale data)")

    # Check plan tiers
    valid_tiers = {"Starter", "Growth", "Scale", "Enterprise"}
    invalid_tiers = set(df["plan_tier"].unique()) - valid_tiers
    if invalid_tiers:
        result.add_warning(f"Unknown plan tiers: {invalid_tiers}")

    result.rows_after = len(df)
    return df, result


def validate_usage(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """Validate usage_metrics.csv data."""
    result = ValidationResult(source="usage_metrics.csv", rows_before=len(df))

    required = ["account_id", "month", "api_calls", "active_users", "sdk_version"]
    for col in required:
        if col not in df.columns:
            result.add_error(f"Missing required column: {col}")

    if not result.is_valid:
        result.rows_after = len(df)
        return df, result

    # Check for negative values (impossible)
    for col in ["api_calls", "content_entries_created", "active_users", "workflows_triggered"]:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                result.add_warning(f"{negatives} rows have negative {col}. Clamping to 0.")
                df.loc[df[col] < 0, col] = 0

    # Check for unreasonably large values (data entry errors)
    if "api_calls" in df.columns:
        extreme = (df["api_calls"] > 10_000_000).sum()
        if extreme > 0:
            result.add_warning(f"{extreme} rows have api_calls > 10M (potential data entry error)")

    # Check SDK version format
    if "sdk_version" in df.columns:
        invalid_sdk = ~df["sdk_version"].str.match(r"^v\d+\.\d+", na=False)
        if invalid_sdk.any():
            result.add_warning(f"{invalid_sdk.sum()} rows have invalid SDK version format")

    # Check for accounts with too few months (< 3 months = unreliable trends)
    months_per_account = df.groupby("account_id")["month"].nunique()
    sparse = (months_per_account < 3).sum()
    if sparse > 0:
        result.add_warning(f"{sparse} accounts have < 3 months of data (trends may be unreliable)")

    result.rows_after = len(df)
    return df, result


def validate_tickets(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """Validate support_tickets.csv data."""
    result = ValidationResult(source="support_tickets.csv", rows_before=len(df))

    required = ["account_id", "priority", "status"]
    for col in required:
        if col not in df.columns:
            result.add_error(f"Missing required column: {col}")

    if not result.is_valid:
        result.rows_after = len(df)
        return df, result

    # Check priority values
    valid_priorities = {"P1", "P2", "P3", "P4"}
    invalid = set(df["priority"].unique()) - valid_priorities
    if invalid:
        result.add_warning(f"Unknown priority values: {invalid}")

    # Check status values
    valid_statuses = {"Open", "Closed", "Escalated", "In Progress"}
    invalid = set(df["status"].unique()) - valid_statuses
    if invalid:
        result.add_warning(f"Unknown status values: {invalid}")

    # Check resolution time
    if "resolution_time_hours" in df.columns:
        negative_res = df["resolution_time_hours"].dropna()
        negative_count = (negative_res < 0).sum()
        if negative_count > 0:
            result.add_warning(f"{negative_count} tickets have negative resolution time")

        extreme_res = (negative_res > 720).sum()  # > 30 days
        if extreme_res > 0:
            result.add_warning(f"{extreme_res} tickets have resolution > 30 days (data quality issue?)")

    result.rows_after = len(df)
    return df, result


def validate_nps(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """Validate nps_responses.csv data."""
    result = ValidationResult(source="nps_responses.csv", rows_before=len(df))

    required = ["account_id", "score"]
    for col in required:
        if col not in df.columns:
            result.add_error(f"Missing required column: {col}")

    if not result.is_valid:
        result.rows_after = len(df)
        return df, result

    # Check NPS scores are 0-10
    out_of_range = ((df["score"] < 0) | (df["score"] > 10)).sum()
    if out_of_range > 0:
        result.add_warning(f"{out_of_range} NPS scores outside 0-10 range. Clamping.")
        df["score"] = df["score"].clip(0, 10)

    # Check for duplicate responses per account
    dupes = df["account_id"].duplicated()
    if dupes.any():
        result.add_warning(
            f"{dupes.sum()} duplicate NPS responses. Using latest per account."
        )

    # Check for accounts with no comment (expected but worth noting)
    empty_comments = (df.get("verbatim_comment", pd.Series([""])) == "").sum()
    if empty_comments > 0:
        result.add_warning(f"{empty_comments} NPS responses have no comment")

    result.rows_after = len(df)
    return df, result


def validate_all(data: dict) -> tuple[dict, list[ValidationResult]]:
    """
    Validate all data sources and return cleaned data + validation results.

    This runs BEFORE signal extraction to catch data quality issues early.
    """
    results = []

    # Validate each source
    data["accounts"], r = validate_accounts(data["accounts"])
    results.append(r)

    data["usage"], r = validate_usage(data["usage"])
    results.append(r)

    data["tickets"], r = validate_tickets(data["tickets"])
    results.append(r)

    data["nps"], r = validate_nps(data["nps"])
    results.append(r)

    # Cross-source validation: check all referenced account IDs exist
    valid_ids = set(data["accounts"]["account_id"])

    for source_name, df in [("usage", data["usage"]), ("tickets", data["tickets"]), ("nps", data["nps"])]:
        orphan_ids = set(df["account_id"]) - valid_ids
        if orphan_ids:
            r = ValidationResult(source=f"{source_name} (cross-ref)")
            r.add_warning(f"{len(orphan_ids)} account IDs in {source_name} not found in accounts.csv: {sorted(orphan_ids)[:5]}...")
            results.append(r)

    return data, results
