"""
Renewal Intelligence Engine -- CLI Pipeline Runner

Orchestrates the full pipeline:
1. Load all data sources
2. Reconcile account names
3. Extract risk signals from each source
4. Compute composite risk scores
5. Generate LLM-powered explanations
6. Discover non-obvious insights
7. Export results

Usage:
    python run_pipeline.py
"""

import sys
import os
import json
import time
import pandas as pd
from pathlib import Path

# Fix Windows terminal encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import Config
from src.data_loader import load_all_data
from src.data_validator import validate_all
from src.reconciler import reconcile_csm_notes
from src.signal_extractor import (
    compute_usage_signals,
    compute_ticket_signals,
    compute_nps_signals,
    compute_csm_signals,
    compute_sdk_risk,
    compute_engagement_signals,
)
from src.risk_scorer import compute_composite_risk, get_risk_summary
from src.insights import compile_all_insights, get_llm_insights
from src.llm_engine import generate_risk_explanation

console = Console(force_terminal=True)


def _log(msg: str):
    """Print a status message, handling Windows encoding gracefully."""
    try:
        console.print(msg)
    except UnicodeEncodeError:
        print(msg)


def main():
    """Run the full Renewal Risk Intelligence pipeline."""
    _log("\n[bold cyan]===== Renewal Risk Intelligence Engine =====[/bold cyan]")
    _log("[dim]Contentstack BizOps -- Account Risk Analysis[/dim]")

    # -- Step 0: Validate config --
    try:
        Config.validate()
        Config.ensure_dirs()
    except (ValueError, FileNotFoundError) as e:
        _log(f"[red]Configuration Error:[/red] {e}")
        sys.exit(1)

    _log(f"[dim]Model: {Config.LLM_MODEL} | Window: {Config.RENEWAL_WINDOW_DAYS} days[/dim]\n")

    # -- Step 1: Load + Validate data --
    _log("[yellow]Step 1/7:[/yellow] Loading all data sources...")
    data = load_all_data()
    _log(f"  [green]>> Loaded {len(data['accounts'])} accounts, "
         f"{len(data['usage'])} usage rows, {len(data['tickets'])} tickets, "
         f"{len(data['nps'])} NPS responses, {len(data['csm_notes'])} CSM notes[/green]")

    # -- Step 1b: Validate data quality --
    _log("\n[yellow]Step 1b:[/yellow] Validating data quality...")
    data, validations = validate_all(data)
    for v in validations:
        if v.errors:
            for e in v.errors:
                _log(f"  [red]ERROR ({v.source}):[/red] {e}")
        if v.warnings:
            for w in v.warnings:
                _log(f"  [yellow]WARN ({v.source}):[/yellow] {w}")
    total_warnings = sum(len(v.warnings) for v in validations)
    total_errors = sum(len(v.errors) for v in validations)
    if total_errors == 0:
        _log(f"  [green]>> Validation passed ({total_warnings} warnings, 0 errors)[/green]")
    else:
        _log(f"  [red]>> Validation found {total_errors} errors, {total_warnings} warnings[/red]")

    accounts = data["accounts"]

    # -- Step 2: Reconcile CSM notes --
    _log("\n[yellow]Step 2/7:[/yellow] Reconciling CSM note account names (AI-first + fuzzy fallback)...")
    csm_notes = reconcile_csm_notes(data["csm_notes"], accounts)
    matched = sum(1 for n in csm_notes if n.account_id is not None)
    _log(f"  [green]>> Reconciled {matched}/{len(csm_notes)} CSM notes to accounts[/green]")

    # -- Step 3: Extract signals --
    _log("\n[yellow]Step 3/7:[/yellow] Extracting risk signals from all data sources...")

    _log("  [dim]3a) Computing usage decline signals...[/dim]")
    usage_signals = compute_usage_signals(data["usage"])
    _log(f"  [green]>> Usage signals: {len(usage_signals)} accounts[/green]")

    _log("  [dim]3b) Computing support ticket signals...[/dim]")
    ticket_signals = compute_ticket_signals(data["tickets"])
    _log(f"  [green]>> Ticket signals: {len(ticket_signals)} accounts[/green]")

    _log("  [dim]3c) Computing NPS signals (+ translating non-English comments via LLM)...[/dim]")
    nps_signals = compute_nps_signals(data["nps"])
    non_english = len(nps_signals[nps_signals["detected_language"] != "English"]) if "detected_language" in nps_signals.columns else 0
    _log(f"  [green]>> NPS signals: {len(nps_signals)} accounts ({non_english} translated)[/green]")

    _log("  [dim]3d) Analyzing CSM notes with LLM (async parallel, ~15-30s)...[/dim]")
    t0 = time.time()
    csm_signals = compute_csm_signals(csm_notes, accounts)
    elapsed = time.time() - t0
    _log(f"  [green]>> CSM signals: {len(csm_signals)} accounts via LLM ({elapsed:.1f}s)[/green]")

    _log("  [dim]3e) Computing SDK deprecation risk from changelog...[/dim]")
    sdk_signals = compute_sdk_risk(data["usage"], data["changelog"])
    v3_count = len(sdk_signals[sdk_signals["sdk_version"].str.startswith("v3", na=False)])
    _log(f"  [green]>> SDK signals: {v3_count} accounts on deprecated v3.x[/green]")

    _log("  [dim]3f) Computing engagement signals...[/dim]")
    engagement_signals = compute_engagement_signals(data["usage"], accounts)
    _log(f"  [green]>> Engagement signals: {len(engagement_signals)} accounts[/green]")

    # -- Step 4: Composite risk scoring --
    _log("\n[yellow]Step 4/7:[/yellow] Computing composite risk scores (calibrated churn probability)...")
    risk_df = compute_composite_risk(
        accounts, usage_signals, ticket_signals, nps_signals,
        csm_signals, sdk_signals, engagement_signals,
    )
    summary = get_risk_summary(risk_df)
    _log(f"  [green]>> Risk scored: {summary['high_risk_count']} High, "
         f"{summary['medium_risk_count']} Medium, {summary['low_risk_count']} Low[/green]")

    # -- Step 5: Generate explanations for at-risk accounts --
    at_risk = risk_df[risk_df["risk_tier"].isin(["High", "Medium"])].copy()
    _log(f"\n[yellow]Step 5/7:[/yellow] Generating LLM explanations for {len(at_risk)} at-risk accounts...")
    t0 = time.time()
    explanations = []
    for idx, (_, row) in enumerate(at_risk.iterrows()):
        account_data = row.to_dict()
        for k, v in account_data.items():
            if isinstance(v, (pd.Timestamp,)):
                account_data[k] = str(v)
            elif isinstance(v, (float,)) and pd.isna(v):
                account_data[k] = None
        explanation = generate_risk_explanation(account_data)
        explanations.append(explanation)
        if (idx + 1) % 5 == 0:
            _log(f"  [dim]  ... {idx + 1}/{len(at_risk)} explanations generated[/dim]")

    at_risk = at_risk.copy()
    at_risk["risk_explanation"] = explanations
    elapsed = time.time() - t0
    _log(f"  [green]>> Generated {len(explanations)} explanations ({elapsed:.1f}s)[/green]")

    # Add explanations back to the full risk_df
    risk_df = risk_df.merge(
        at_risk[["account_id", "risk_explanation"]],
        on="account_id",
        how="left",
    )
    risk_df["risk_explanation"] = risk_df["risk_explanation"].fillna("Low risk -- no immediate concerns.")

    # -- Step 6: Non-obvious insights --
    _log("\n[yellow]Step 6/7:[/yellow] Discovering non-obvious insights with LLM...")
    t0 = time.time()
    structured_insights = compile_all_insights(risk_df)
    llm_insights = get_llm_insights(risk_df)
    elapsed = time.time() - t0
    _log(f"  [green]>> Insights generated ({elapsed:.1f}s)[/green]")

    # -- Display Results --
    _log("")
    _print_summary(summary)
    _print_risk_table(risk_df)
    _print_insights(structured_insights, llm_insights)

    # -- Export --
    _export_results(risk_df, structured_insights, llm_insights, summary)

    _log("\n[bold green]Pipeline complete![/bold green]")


def _print_summary(summary: dict):
    """Print summary panel."""
    console.print(Panel(
        f"[bold]Accounts Analyzed:[/bold] {summary['total_accounts_analyzed']}\n"
        f"[bold red]High Risk:[/bold red] {summary['high_risk_count']}  "
        f"[bold yellow]Medium Risk:[/bold yellow] {summary['medium_risk_count']}  "
        f"[bold green]Low Risk:[/bold green] {summary['low_risk_count']}\n"
        f"[bold]ARR at Risk (High+Medium):[/bold] ${summary['total_arr_at_risk']:,}\n"
        f"[bold]High Risk ARR:[/bold] ${summary['high_risk_arr']:,}",
        title="Risk Summary",
        border_style="yellow",
    ))


def _print_risk_table(risk_df: pd.DataFrame):
    """Print the top at-risk accounts table."""
    table = Table(title="Top At-Risk Accounts (Renewing in Next 90 Days)")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Account", style="bold", width=24)
    table.add_column("Risk", justify="center", width=6)
    table.add_column("Tier", justify="center", width=8)
    table.add_column("ARR", justify="right", width=12)
    table.add_column("Renewal", justify="center", width=10)
    table.add_column("SDK", justify="center", width=8)
    table.add_column("Top Drivers", width=40)

    for i, (_, row) in enumerate(risk_df.head(20).iterrows()):
        tier = row["risk_tier"]
        tier_style = {"High": "bold red", "Medium": "yellow", "Low": "green"}.get(tier, "white")

        table.add_row(
            str(i + 1),
            str(row["account_name"]),
            f"{row['composite_risk_score']:.0%}",
            f"[{tier_style}]{tier}[/{tier_style}]",
            f"${row['arr']:,}",
            f"{row.get('days_until_renewal', '?')}d",
            str(row.get("sdk_version", "?")),
            str(row.get("top_risk_drivers", ""))[:40],
        )

    console.print(table)


def _print_insights(structured_insights: dict, llm_insights: str):
    """Print non-obvious insights."""
    console.print(Panel(
        llm_insights,
        title="Non-Obvious Insights (LLM Analysis)",
        border_style="magenta",
    ))

    for category, count in structured_insights.get("summary", {}).items():
        if count > 0:
            label = category.replace("_", " ").title()
            _log(f"  * {label}: [bold]{count}[/bold] accounts flagged")


def _export_results(risk_df: pd.DataFrame, insights: dict, llm_insights: str, summary: dict):
    """Export results to CSV and JSON."""
    output_dir = Config.OUTPUT_DIR

    # Export risk-scored accounts
    export_cols = [
        "account_id", "account_name", "arr", "plan_tier", "industry", "region",
        "contract_end_date", "days_until_renewal",
        "composite_risk_score", "risk_tier", "top_risk_drivers",
        "usage_risk_score", "ticket_risk_score", "nps_risk_score",
        "csm_risk_score", "sdk_risk_score", "engagement_risk_score",
        "risk_explanation",
    ]
    available_cols = [c for c in export_cols if c in risk_df.columns]
    risk_df[available_cols].to_csv(output_dir / "risk_scored_accounts.csv", index=False)
    _log(f"[dim]Exported: {output_dir / 'risk_scored_accounts.csv'}[/dim]")

    # Export detailed signals
    detail_cols = [c for c in risk_df.columns if c not in ["risk_explanation"]]
    risk_df[detail_cols].to_csv(output_dir / "detailed_signals.csv", index=False)
    _log(f"[dim]Exported: {output_dir / 'detailed_signals.csv'}[/dim]")

    # Export insights as JSON
    insights_export = {
        "summary": summary,
        "llm_insights": llm_insights,
        "structured_insight_counts": insights.get("summary", {}),
    }
    with open(output_dir / "insights.json", "w", encoding="utf-8") as f:
        json.dump(insights_export, f, indent=2, default=str)
    _log(f"[dim]Exported: {output_dir / 'insights.json'}[/dim]")

    # Export individual account risk briefings
    briefings = []
    for _, row in risk_df.iterrows():
        briefings.append({
            "account_id": int(row["account_id"]),
            "account_name": row["account_name"],
            "risk_tier": row["risk_tier"],
            "risk_score": float(row["composite_risk_score"]),
            "arr": int(row["arr"]),
            "explanation": row.get("risk_explanation", ""),
        })
    with open(output_dir / "account_briefings.json", "w", encoding="utf-8") as f:
        json.dump(briefings, f, indent=2, default=str)
    _log(f"[dim]Exported: {output_dir / 'account_briefings.json'}[/dim]")


if __name__ == "__main__":
    main()
