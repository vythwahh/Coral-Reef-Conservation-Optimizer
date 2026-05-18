import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.insert(0, "/Users/vythu/Documents/CoralOpt")

from lp_solver_pipeline import run_lp_solver, load_risk_scores

 
# PAGE CONFIG
 
st.set_page_config(
    page_title="CoralOpt",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background-color: #1a3c6e !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #15305a !important;
    }
    .stButton > button[kind="secondary"] {
        border-color: #1a3c6e !important;
        color: #1a3c6e !important;
    }
    .stAlert { border-left-color: #1a3c6e !important; }
    h1, h2, h3 { color: #1a3c6e; }
</style>
""", unsafe_allow_html=True)


# HEADER

st.title("CoralOpt")
st.markdown("**Coral Reef Conservation Budget Optimizer**")
st.markdown(
    "Sea surface temperature (SST) and bleaching index data are automatically "
    "collected from NOAA, processed through a Kafka → Spark pipeline, and used "
    "to compute the optimal allocation of conservation resources."
)
st.markdown("---")

tab1, tab2 = st.tabs(["Reef Status", "Budget Planning"])

 
# TAB 1: REEF STATUS
 
with tab1:
    st.header("Coral Reef Status")
    st.caption("Data updated automatically every day via the Kafka → Spark pipeline")

    try:
        risk_df = load_risk_scores()

        # ── Overview metrics ──────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Zones Monitored",   len(risk_df))
        col2.metric("Avg Temperature",   f"{risk_df['avg_sst'].mean():.1f}°C")
        col3.metric("At-Risk Zones",     len(risk_df[risk_df["avg_risk_score"] > 0.02]))
        col4.metric("Highest Risk Score",f"{risk_df['avg_risk_score'].max():.3f}")

        st.markdown("---")

        # ── Status table ──────────────────────
        st.subheader("Zone Details")

        def risk_label(score):
            if score >= 0.5:    return "Critical"
            elif score >= 0.1:  return "High Risk"
            elif score >= 0.02: return "Watch"
            else:               return "Normal"

        display_df = risk_df.copy()
        display_df["Status"] = display_df["avg_risk_score"].apply(risk_label)
        display_df = display_df.rename(columns={
            "reef_id"           : "Reef ID",
            "name"              : "Zone Name",
            "avg_sst"           : "Temperature (°C)",
            "avg_risk_score"    : "Risk Score",
            "avg_bleaching_risk": "Bleaching Risk",
        })

        st.dataframe(
            display_df[["Reef ID", "Zone Name", "Temperature (°C)",
                         "Risk Score", "Bleaching Risk", "Status"]]
            .sort_values("Risk Score", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Charts  
        st.subheader("Risk Overview")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        colors = ["#d32f2f" if r > 0.02 else "#1a3c6e"
                  for r in risk_df["avg_risk_score"]]
        short_names = [n.replace("Great Barrier Reef", "GBR")
                       for n in risk_df["name"]]

        axes[0].barh(short_names, risk_df["avg_risk_score"], color=colors)
        axes[0].axvline(x=0.02, color="orange", linestyle="--",
                        alpha=0.7, label="Watch threshold")
        axes[0].set_xlabel("Risk Score")
        axes[0].set_title("Risk Score by Zone")
        axes[0].legend()

        axes[1].barh(short_names, risk_df["avg_sst"], color="#1a6e5a")
        axes[1].axvline(x=29, color="red", linestyle="--",
                        alpha=0.7, label="Bleaching threshold (29°C)")
        axes[1].set_xlabel("Temperature (°C)")
        axes[1].set_title("Sea Surface Temperature (SST)")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning("No pipeline data available. Please run the Airflow DAG first.")
        st.caption(f"Details: {e}")

 
# TAB 2: BUDGET PLANNING
 
with tab2:
    st.header("Conservation Budget Planning")
    st.markdown(
        "Enter your available budget — the system will automatically compute "
        "the optimal allocation across reef zones."
    )
    st.markdown("---")

    # Input 
    col_input, col_info = st.columns([1, 1])

    with col_input:
        st.subheader("Budget")
        budget = st.number_input(
            "Total Budget (USD)",
            min_value=10_000,
            max_value=10_000_000,
            value=500_000,
            step=50_000,
            format="%d",
            help="Enter the total budget to allocate across reef zones"
        )
        st.caption(f"≈ **${budget:,.0f} USD**")

        st.markdown("")
        run_btn = st.button(
            "🔍 Compute Optimal Plan",
            type="primary",
            use_container_width=True
        )

    with col_info:
        st.subheader("How it works")
        st.info("""
**The system optimizes based on:**
- Risk score per zone (real NOAA data)
- Average restoration cost per km²
- Available area per zone

**Objective:** Maximize total conservation impact within the given budget.
        """)

    st.markdown("---")

    #  Results 
    if run_btn:
        with st.spinner("Computing optimal allocation"):
            try:
                result_df, status, summary = run_lp_solver(budget=budget)
                if status == "Optimal" and result_df is not None:
                    st.session_state["lp_result"]  = result_df
                    st.session_state["lp_summary"] = summary
                    st.session_state["lp_ran"]     = True
                else:
                    st.error("Could not find a feasible allocation plan.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.get("lp_ran"):
        result_df = st.session_state["lp_result"]
        summary   = st.session_state["lp_summary"]

        # Summary metrics  
        st.subheader("Optimal Allocation Plan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Budget",    f"${summary['budget']:,.0f}")
        m2.metric("Amount Allocated",f"${summary['total_cost']:,.0f}")
        m3.metric("Budget Utilization", f"{summary['budget_used_pct']}%")
        m4.metric("Total Impact",    f"{summary['total_impact']:.4f}")

        st.markdown("")

        # Allocation tables  
        active_df   = result_df[result_df["area_restored"] > 0].copy()
        inactive_df = result_df[result_df["area_restored"] == 0].copy()

        if not active_df.empty:
            st.markdown("** Zones receiving investment:**")
            st.dataframe(
                active_df[["name", "area_restored", "cost_used", "budget_pct", "impact"]]
                .rename(columns={
                    "name"         : "Zone Name",
                    "area_restored": "Area Restored (km²)",
                    "cost_used"    : "Cost (USD)",
                    "budget_pct"   : "% of Budget",
                    "impact"       : "Impact Score",
                }),
                use_container_width=True,
                hide_index=True
            )

        if not inactive_df.empty:
            st.markdown("**⏸Zones not funded within this budget:**")
            st.dataframe(
                inactive_df[["name", "risk_score"]].rename(columns={
                    "name"      : "Zone Name",
                    "risk_score": "Risk Score",
                }),
                use_container_width=True,
                hide_index=True
            )

        # Charts  
        if not active_df.empty:
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

            axes2[0].pie(
                active_df["cost_used"],
                labels=active_df["name"],
                autopct="%1.1f%%",
                colors=plt.cm.Set2.colors[:len(active_df)]
            )
            axes2[0].set_title("Budget Allocation by Zone")

            axes2[1].bar(
                active_df["name"],
                active_df["area_restored"],
                color="#1a3c6e"
            )
            axes2[1].set_ylabel("Area (km²)")
            axes2[1].set_title("Reef Area to be Restored")
            axes2[1].tick_params(axis="x", rotation=15)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        st.success("Optimal allocation plan computed successfully!")

        if st.button("Recalculate with different budget",
                     type="secondary", use_container_width=True):
            st.session_state.pop("lp_ran",      None)
            st.session_state.pop("lp_result",   None)
            st.session_state.pop("lp_summary",  None)
            st.rerun()