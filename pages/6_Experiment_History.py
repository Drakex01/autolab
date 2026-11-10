import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.state import init_session_state, get_current_workspace
from core.database import list_experiment_runs, delete_experiment_run, get_experiment_run
from services.history_service import format_runs_for_comparison
import plotly.express as px

st.set_page_config(page_title="AutoLab — Experiment History", page_icon="📜", layout="wide")
init_session_state()

st.title("📜 Experiment History & Run Comparison")
st.markdown("Review longitudinal experiment records, inspect audit trails, and compare metrics across models.")

current_ws = get_current_workspace()
filter_workspace = st.sidebar.checkbox("Filter by Active Workspace Only", value=True if current_ws else False)

workspace_id = current_ws["id"] if (current_ws and filter_workspace) else None
runs = list_experiment_runs(workspace_id=workspace_id)

if not runs:
    st.info("No experiment runs found.")
    if st.button("Run an Experiment ⚡"):
        st.switch_page("pages/4_Run_Experiment.py")
    st.stop()

tab_table, tab_compare, tab_details = st.tabs(["📋 Run History Table", "⚖️ Cross-Run Comparison", "🔍 Deep Run Inspection"])

with tab_table:
    st.subheader(f"Saved Execution Log ({len(runs)} Runs)")
    df_runs = format_runs_for_comparison(runs)
    st.dataframe(df_runs, use_container_width=True)

    # Download History as CSV
    csv_bytes = df_runs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Experiment History (CSV)",
        data=csv_bytes,
        file_name="autolab_experiment_history.csv",
        mime="text/csv"
    )

with tab_compare:
    st.subheader("Side-by-Side Model & Metric Comparison")
    st.markdown("Select two or more runs to contrast performance (e.g. Logistic Regression vs Random Forest):")

    run_labels = {f"Run #{r['id']} ({r['config'].get('model_type', r['experiment_type'])} - {r['dataset_name']})": r['id'] for r in runs}
    selected_run_labels = st.multiselect(
        "Choose Runs to Compare",
        options=list(run_labels.keys()),
        default=list(run_labels.keys())[:min(3, len(run_labels))]
    )

    if len(selected_run_labels) >= 2:
        selected_ids = [run_labels[lbl] for lbl in selected_run_labels]
        comp_runs = [r for r in runs if r['id'] in selected_ids]
        df_comp = format_runs_for_comparison(comp_runs)
        st.dataframe(df_comp, use_container_width=True)

        # Plot comparison for common metrics
        numeric_metric_cols = [c for c in df_comp.columns if c.startswith("Metric:") and pd.api.types.is_numeric_dtype(df_comp[c])]
        if numeric_metric_cols:
            metric_to_plot = st.selectbox("Select Metric to Visualize", numeric_metric_cols)
            df_plot = df_comp.copy()
            df_plot["Label"] = "Run #" + df_plot["Run ID"].astype(str) + " (" + df_plot["Model"] + ")"
            fig = px.bar(
                df_plot, x="Label", y=metric_to_plot,
                color="Model",
                title=f"Comparative Benchmark: {metric_to_plot}",
                height=450
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least 2 runs above to generate comparative charts.")

with tab_details:
    st.subheader("Detailed Run Audit Record")
    chosen_run_lbl = st.selectbox("Select Run", list(run_labels.keys()))
    run_id = run_labels[chosen_run_lbl]
    run_rec = get_experiment_run(run_id)

    if run_rec:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**Run ID:** `{run_rec['id']}`")
            st.markdown(f"**Workspace ID:** `{run_rec['workspace_id']}`")
            st.markdown(f"**Dataset:** `{run_rec['dataset_name']}`")
            st.markdown(f"**Experiment Type:** `{run_rec['experiment_type']}`")
            st.markdown(f"**Timestamp:** `{run_rec['created_at']}`")
        with col_d2:
            st.markdown("#### Hyperparameters & Configuration")
            st.json(run_rec["config"])

        st.markdown("#### Computed Metrics")
        st.json(run_rec["metrics"])

        if run_rec.get("interpretation"):
            st.markdown("#### Saved Interpretation")
            try:
                st.json(json.loads(run_rec["interpretation"]))
            except Exception:
                st.write(run_rec["interpretation"])

        if run_rec.get("next_experiment_recommendation"):
            st.markdown("#### Saved Next Steps")
            try:
                st.json(json.loads(run_rec["next_experiment_recommendation"]))
            except Exception:
                st.write(run_rec["next_experiment_recommendation"])

        if st.button("🗑️ Delete this Run Record", type="secondary"):
            delete_experiment_run(run_id)
            st.warning(f"Deleted Run #{run_id}.")
            st.rerun()
