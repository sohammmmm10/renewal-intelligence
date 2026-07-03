"""
Renewal Intelligence Engine — Streamlit Dashboard

Interactive UI for exploring account risk scores, explanations, and insights.

Run: streamlit run app.py
"""

import sys
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from config import Config
from src.data_loader import load_all_data
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
from src.insights import compile_all_insights, get_llm_insights, build_llm_insights_prompt
from src.llm_engine import generate_risk_explanation


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Renewal Intelligence Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CACHE THE PIPELINE (runs once, then cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline():
    """Run the full analysis pipeline (cached after first run)."""
    Config.validate()
    Config.ensure_dirs()

    data = load_all_data()
    accounts = data["accounts"]

    # Reconcile CSM notes
    csm_notes = reconcile_csm_notes(data["csm_notes"], accounts)

    # Extract all signals
    usage_signals = compute_usage_signals(data["usage"])
    ticket_signals = compute_ticket_signals(data["tickets"])
    nps_signals = compute_nps_signals(data["nps"])
    csm_signals = compute_csm_signals(csm_notes, accounts)
    sdk_signals = compute_sdk_risk(data["usage"], data["changelog"])
    engagement_signals = compute_engagement_signals(data["usage"], accounts)

    # Composite risk scoring
    risk_df = compute_composite_risk(
        accounts, usage_signals, ticket_signals, nps_signals,
        csm_signals, sdk_signals, engagement_signals,
    )

    # Generate explanations for at-risk accounts
    at_risk = risk_df[risk_df["risk_tier"].isin(["High", "Medium"])].copy()
    explanations = []
    for _, row in at_risk.iterrows():
        account_data = row.to_dict()
        for k, v in account_data.items():
            if isinstance(v, (pd.Timestamp,)):
                account_data[k] = str(v)
            elif isinstance(v, float) and pd.isna(v):
                account_data[k] = None
        explanations.append(generate_risk_explanation(account_data))

    at_risk["risk_explanation"] = explanations
    risk_df = risk_df.merge(
        at_risk[["account_id", "risk_explanation"]], on="account_id", how="left"
    )
    risk_df["risk_explanation"] = risk_df["risk_explanation"].fillna(
        "Low risk — no immediate concerns identified."
    )

    # Insights
    structured_insights = compile_all_insights(risk_df)
    llm_insights = get_llm_insights(risk_df)

    summary = get_risk_summary(risk_df)

    return risk_df, summary, structured_insights, llm_insights


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with filters."""
    st.sidebar.title("Filters")

    risk_filter = st.sidebar.multiselect(
        "Risk Tier",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    plan_filter = st.sidebar.multiselect(
        "Plan Tier",
        ["Starter", "Growth", "Scale", "Enterprise"],
        default=["Starter", "Growth", "Scale", "Enterprise"],
    )

    min_arr = st.sidebar.number_input("Min ARR ($)", value=0, step=10000)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Model:** `{Config.LLM_MODEL}`\n\n"
        f"**Snapshot Date:** {Config.DATASET_DATE.date()}\n\n"
        f"**Renewal Window:** {Config.RENEWAL_WINDOW_DAYS} days\n\n"
        f"**Showing:** Accounts renewing {Config.DATASET_DATE.date()} to "
        f"{(Config.DATASET_DATE + pd.Timedelta(days=Config.RENEWAL_WINDOW_DAYS)).date()}"
    )

    return risk_filter, plan_filter, min_arr


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────

def main():
    st.title("Renewal Risk Intelligence Engine")
    st.caption("Contentstack BizOps — AI-Powered Account Risk Analysis")

    # Check config
    try:
        Config.validate()
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Configuration Error: {e}")
        st.info("Please create a `.env` file from `env_example.txt` and add your OpenAI API key.")
        st.stop()

    # Sidebar filters
    risk_filter, plan_filter, min_arr = render_sidebar()

    # Run pipeline (cached)
    with st.spinner("Running analysis pipeline (first load takes ~60s for LLM calls)..."):
        risk_df, summary, structured_insights, llm_insights = run_pipeline()

    # Apply filters
    filtered = risk_df[
        (risk_df["risk_tier"].isin(risk_filter)) &
        (risk_df["plan_tier"].isin(plan_filter)) &
        (risk_df["arr"] >= min_arr)
    ]

    # ── KPI Cards ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accounts Analyzed", summary["total_accounts_analyzed"])
    with col2:
        st.metric("High Risk", summary["high_risk_count"], delta=None)
    with col3:
        st.metric("ARR at Risk", f"${summary['total_arr_at_risk']:,}")
    with col4:
        st.metric("Avg Risk Score", f"{summary['avg_risk_score']:.0%}")

    st.markdown("---")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk Overview", "Account Deep-Dive", "Signal Analysis", "Non-Obvious Insights"
    ])

    with tab1:
        render_risk_overview(filtered, summary)

    with tab2:
        render_account_drilldown(filtered)

    with tab3:
        render_signal_analysis(filtered)

    with tab4:
        render_insights(risk_df, structured_insights, llm_insights)


# ─────────────────────────────────────────────
# TAB 1: RISK OVERVIEW
# ─────────────────────────────────────────────

def render_risk_overview(df: pd.DataFrame, summary: dict):
    st.subheader("Risk Distribution")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Risk tier donut chart
        tier_counts = df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
        fig = px.pie(
            tier_counts, values="Count", names="Risk Tier",
            color="Risk Tier", color_discrete_map=color_map,
            hole=0.4, title="Risk Tier Distribution"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    with col2:
        # ARR by risk tier
        arr_by_tier = df.groupby("risk_tier")["arr"].sum().reset_index()
        arr_by_tier.columns = ["Risk Tier", "ARR"]
        fig = px.bar(
            arr_by_tier, x="Risk Tier", y="ARR", color="Risk Tier",
            color_discrete_map=color_map, title="ARR by Risk Tier",
            text_auto=True,
        )
        fig.update_layout(height=350, showlegend=False)
        fig.update_traces(texttemplate="$%{value:,.0f}")
        st.plotly_chart(fig, width="stretch")

    # Risk scatter: ARR vs Risk Score
    st.subheader("ARR vs Risk Score")
    fig = px.scatter(
        df, x="composite_risk_score", y="arr",
        color="risk_tier", color_discrete_map=color_map,
        size="arr", size_max=50,
        hover_name="account_name",
        hover_data=["plan_tier", "days_until_renewal", "top_risk_drivers"],
        labels={"composite_risk_score": "Risk Score", "arr": "ARR ($)"},
        title="Account Risk Map — Size = ARR, Color = Risk Tier",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    # Risk table
    st.subheader("All Accounts Ranked by Risk")
    display_cols = [
        "account_name", "risk_tier", "composite_risk_score", "arr",
        "plan_tier", "days_until_renewal", "top_risk_drivers",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].style.apply(
            lambda row: [
                "background-color: #fecaca" if row.get("risk_tier") == "High"
                else "background-color: #fef3c7" if row.get("risk_tier") == "Medium"
                else ""
            ] * len(row),
            axis=1,
        ),
        width="stretch",
        height=400,
    )


# ─────────────────────────────────────────────
# TAB 2: ACCOUNT DEEP-DIVE
# ─────────────────────────────────────────────

def render_account_drilldown(df: pd.DataFrame):
    st.subheader("Account Deep-Dive")

    account_names = df["account_name"].tolist()
    selected = st.selectbox("Select Account", account_names, index=0)

    if not selected:
        return

    row = df[df["account_name"] == selected].iloc[0]

    # Account header
    tier = row["risk_tier"]
    tier_color = {"High": "red", "Medium": "orange", "Low": "green"}.get(tier, "gray")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Score", f"{row['composite_risk_score']:.0%}")
    with col2:
        st.metric("Risk Tier", tier)
    with col3:
        st.metric("ARR", f"${row['arr']:,}")
    with col4:
        st.metric("Renewal In", f"{row.get('days_until_renewal', '?')} days")

    st.markdown("---")

    # Risk explanation (LLM-generated)
    st.subheader("AI Risk Assessment")
    st.info(row.get("risk_explanation", "No explanation available."))

    # Signal breakdown
    st.subheader("Signal Breakdown")
    signal_data = {
        "Signal": ["Usage Decline", "Support Issues", "NPS", "CSM Sentiment", "SDK Risk", "Engagement"],
        "Score": [
            row.get("usage_risk_score", 0),
            row.get("ticket_risk_score", 0),
            row.get("nps_risk_score", 0),
            row.get("csm_risk_score", 0),
            row.get("sdk_risk_score", 0),
            row.get("engagement_risk_score", 0),
        ],
    }
    signal_df = pd.DataFrame(signal_data)
    fig = px.bar(
        signal_df, x="Score", y="Signal", orientation="h",
        color="Score", color_continuous_scale=["green", "yellow", "red"],
        range_color=[0, 1], title="Risk Signal Breakdown",
    )
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # Detail cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Account Details**")
        st.write(f"- **Plan:** {row.get('plan_tier', 'N/A')}")
        st.write(f"- **Industry:** {row.get('industry', 'N/A')}")
        st.write(f"- **Region:** {row.get('region', 'N/A')}")
        st.write(f"- **CSM:** {row.get('csm_name', 'N/A')}")
        st.write(f"- **SDK Version:** {row.get('sdk_version', 'N/A')}")
        st.write(f"- **Contract End:** {row.get('contract_end_date', 'N/A')}")

    with col2:
        st.markdown("**Risk Signals**")
        st.write(f"- **NPS Score:** {row.get('nps_score', 'No response')}")
        st.write(f"- **Total Tickets:** {row.get('total_tickets', 0)} ({row.get('p1_p2_tickets', 0)} P1/P2)")
        st.write(f"- **Open/Escalated:** {row.get('open_escalated', 0)}")
        st.write(f"- **Competitors Mentioned:** {row.get('competitor_mentions', 'None')}")
        st.write(f"- **Champion Status:** {row.get('champion_status', 'Unknown')}")
        st.write(f"- **Top Drivers:** {row.get('top_risk_drivers', 'N/A')}")

    # CSM notes summary
    if row.get("csm_summary"):
        st.subheader("CSM Notes Summary")
        st.warning(row["csm_summary"])

    # NPS comment
    if row.get("nps_comment"):
        st.subheader("NPS Comment")
        comment = row["nps_comment"]
        translated = row.get("translated_comment", "")
        lang = row.get("detected_language", "English")
        if lang != "English" and translated:
            st.write(f"**Original ({lang}):** {comment}")
            st.write(f"**Translated:** {translated}")
        else:
            st.write(comment)


# ─────────────────────────────────────────────
# TAB 3: SIGNAL ANALYSIS
# ─────────────────────────────────────────────

def render_signal_analysis(df: pd.DataFrame):
    st.subheader("Signal Correlation Analysis")

    # Signal heatmap for at-risk accounts
    signal_cols = [
        "usage_risk_score", "ticket_risk_score", "nps_risk_score",
        "csm_risk_score", "sdk_risk_score", "engagement_risk_score",
    ]
    available_signals = [c for c in signal_cols if c in df.columns]

    if available_signals:
        at_risk = df[df["risk_tier"].isin(["High", "Medium"])].head(20)
        if not at_risk.empty:
            heatmap_data = at_risk.set_index("account_name")[available_signals]
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale=["green", "yellow", "red"],
                labels={"color": "Risk Score"},
                title="Signal Heatmap — At-Risk Accounts",
                aspect="auto",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

    # SDK version distribution
    col1, col2 = st.columns(2)
    with col1:
        if "sdk_version" in df.columns:
            sdk_dist = df.groupby(["sdk_version", "risk_tier"]).size().reset_index(name="count")
            color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
            fig = px.bar(
                sdk_dist, x="sdk_version", y="count", color="risk_tier",
                color_discrete_map=color_map, title="SDK Version vs Risk Tier",
                barmode="stack",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")

    with col2:
        if "plan_tier" in df.columns:
            plan_dist = df.groupby(["plan_tier", "risk_tier"]).size().reset_index(name="count")
            fig = px.bar(
                plan_dist, x="plan_tier", y="count", color="risk_tier",
                color_discrete_map=color_map, title="Plan Tier vs Risk Tier",
                barmode="stack",
                category_orders={"plan_tier": ["Starter", "Growth", "Scale", "Enterprise"]},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────
# TAB 4: NON-OBVIOUS INSIGHTS
# ─────────────────────────────────────────────

def render_insights(df: pd.DataFrame, structured_insights: dict, llm_insights: str):
    st.subheader("Non-Obvious Insights")
    st.caption("Patterns that a simple rule-based system would miss")

    # LLM-generated insights
    st.markdown("### AI-Discovered Insights")
    st.markdown(llm_insights)

    st.markdown("---")

    # Structured insights
    st.markdown("### Pattern Detection Results")

    insights_data = structured_insights.get("insights", {})

    for category, insight_df in insights_data.items():
        if isinstance(insight_df, pd.DataFrame) and not insight_df.empty:
            label = category.replace("_", " ").title()
            with st.expander(f"{label} ({len(insight_df)} accounts)", expanded=True):
                for _, row in insight_df.iterrows():
                    st.markdown(f"**{row.get('account_name', 'Unknown')}** (ID: {row.get('account_id', '?')})")
                    st.write(row.get("insight_detail", ""))
                    st.markdown("---")


if __name__ == "__main__":
    main()
