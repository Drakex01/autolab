import streamlit as st
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.state import init_session_state, get_current_workspace
from core.database import save_experiment_run, get_latest_plan
from services.experiment_service import (
    get_builtin_datasets, load_dataset,
    run_classification_experiment, run_regression_experiment,
    run_clustering_experiment, run_parameter_sweep,
    plot_confusion_matrix, plot_regression_actual_vs_predicted,
    plot_clustering_2d_pca, plot_parameter_sweep, plot_feature_importance
)

st.set_page_config(page_title="AutoLab — Run Experiment", page_icon="⚡", layout="wide")
init_session_state()

st.title("⚡ Run Experiment")
st.markdown("Execute safe, reproducible, predefined statistical and machine learning workflows on verified datasets.")

current_ws = get_current_workspace()
if not current_ws:
    st.warning("⚠️ No active workspace selected. Please select or create one in **Research Workspace** first.")
    if st.button("Go to Research Workspace"):
        st.switch_page("pages/1_Research_Workspace.py")
    st.stop()

st.info(f"**Target Workspace:** `{current_ws['title']}` | **Research Question:** *{current_ws['research_question']}*")

# 1. Dataset Selection
st.subheader("1. Dataset Selection")
data_source = st.radio("Choose Dataset Source", ["Built-in Scientific Dataset", "Upload Custom CSV File"], horizontal=True)

df = None
dataset_name = ""
default_target = None

if data_source == "Built-in Scientific Dataset":
    builtin_options = get_builtin_datasets()
    chosen_label = st.selectbox("Select Built-in Dataset", list(builtin_options.keys()))
    dataset_key = builtin_options[chosen_label]
    dataset_name = chosen_label
    df, default_target = load_dataset(dataset_key)
else:
    uploaded_file = st.file_uploader("Upload CSV Dataset (Max 20MB)", type=["csv"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name
        try:
            df, _ = load_dataset("custom", uploaded_file=uploaded_file)
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")

if df is not None:
    st.session_state.current_dataset_df = df
    st.session_state.current_dataset_name = dataset_name

    # Dataset Summary Metrics
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.metric("Total Rows", f"{len(df):,}")
    with col_d2:
        st.metric("Total Columns", len(df.columns))
    with col_d3:
        missing_count = int(df.isnull().sum().sum())
        st.metric("Missing Values", f"{missing_count:,}")
    with col_d4:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")

    with st.expander("🔍 Dataset Preview & Column Summary", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        col_summary = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Null Count": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.caption("Column Specification")
        st.dataframe(col_summary, use_container_width=True)

    # 2. Experiment Setup
    st.markdown("---")
    st.subheader("2. Experiment Configuration")

    exp_type = st.selectbox(
        "Experiment Type",
        ["Classification", "Regression", "Clustering", "Parameter Sweep"],
        help="Select a safe, predefined scientific machine learning workflow."
    )

    all_cols = list(df.columns)
    target_idx = 0
    if default_target and default_target in all_cols:
        target_idx = all_cols.index(default_target)

    if exp_type in ["Classification", "Regression", "Parameter Sweep"]:
        target_col = st.selectbox(
            "Select Target Column (Dependent Variable)",
            options=all_cols,
            index=target_idx,
            help="This column will be isolated from features to prevent data leakage."
        )
    else:
        target_col = None

    c_cfg1, c_cfg2, c_cfg3 = st.columns(3)

    with c_cfg1:
        if exp_type == "Classification":
            model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
        elif exp_type == "Regression":
            model_choice = st.selectbox("Model", ["Random Forest Regressor", "Linear Regression"])
        elif exp_type == "Clustering":
            model_choice = "K-Means"
            st.text_input("Algorithm", value="K-Means", disabled=True)
        else:  # Parameter Sweep
            model_choice = st.selectbox("Base Model for Sweep", ["Random Forest", "Logistic Regression", "Random Forest Regressor"])

    with c_cfg2:
        if exp_type in ["Classification", "Regression", "Parameter Sweep"]:
            test_split = st.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        else:
            n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=3)

    with c_cfg3:
        seed = st.number_input("Reproducibility Seed", value=42, step=1)

    # Extra parameters if parameter sweep
    sweep_param = None
    sweep_vals = None
    if exp_type == "Parameter Sweep":
        st.markdown("#### Parameter Sweep Settings")
        if "Random Forest" in model_choice:
            sweep_param = st.selectbox("Sweep Parameter", ["n_estimators", "max_depth"])
            if sweep_param == "n_estimators":
                sweep_vals = [10, 25, 50, 100, 200]
            else:
                sweep_vals = [2, 4, 6, 8, 12, 16]
        else:
            sweep_param = "C"
            sweep_vals = [0.01, 0.1, 1.0, 10.0, 50.0]
        st.caption(f"Testing values: `{sweep_vals}`")

    # 3. Execution Action
    st.markdown("---")
    st.subheader("3. Execute Workflow")

    if st.button("🚀 Run Predefined Experiment", type="primary", use_container_width=True):
        with st.spinner("Executing reproducible scikit-learn workflow..."):
            try:
                result = None
                if exp_type == "Classification":
                    result = run_classification_experiment(
                        df=df, target_col=target_col, model_type=model_choice,
                        test_size=test_split, random_state=int(seed)
                    )
                elif exp_type == "Regression":
                    result = run_regression_experiment(
                        df=df, target_col=target_col, model_type=model_choice,
                        test_size=test_split, random_state=int(seed)
                    )
                elif exp_type == "Clustering":
                    result = run_clustering_experiment(
                        df=df, n_clusters=int(n_clusters), random_state=int(seed)
                    )
                elif exp_type == "Parameter Sweep":
                    result = run_parameter_sweep(
                        df=df, target_col=target_col, model_type=model_choice,
                        param_name=sweep_param, param_values=sweep_vals,
                        test_size=test_split, random_state=int(seed)
                    )

                if result and result.get("status") == "success":
                    st.success("✅ Experiment successfully completed!")
                    st.session_state.latest_run_result = result

                    # Save to database
                    latest_plan = get_latest_plan(current_ws["id"])
                    plan_id = latest_plan["id"] if latest_plan else None

                    run_id = save_experiment_run(
                        workspace_id=current_ws["id"],
                        dataset_name=dataset_name,
                        experiment_type=exp_type.lower(),
                        config_dict=result["config"],
                        metrics_dict=result["metrics"],
                        plan_id=plan_id
                    )
                    st.session_state.latest_run_id = run_id
                    st.toast(f"Saved run #{run_id} to database!")

            except Exception as ex:
                st.error(f"❌ Execution failed: {str(ex)}")

    # 4. Display Results if available
    res = st.session_state.get("latest_run_result")
    if res:
        st.markdown("---")
        st.subheader("📊 Execution Results & Visualizations")

        # Metric Badges
        metric_cols = st.columns(len(res["metrics"]))
        for col, (m_key, m_val) in zip(metric_cols, res["metrics"].items()):
            with col:
                st.metric(m_key.replace('_', ' ').title(), f"{m_val}")

        # Render relevant charts
        if res["experiment_type"] == "classification":
            c_p1, c_p2 = st.columns([1, 1])
            with c_p1:
                cm_fig = plot_confusion_matrix(res["confusion_matrix"], res["classes"])
                st.plotly_chart(cm_fig, use_container_width=True)
            with c_p2:
                if res.get("feature_importance"):
                    fi_fig = plot_feature_importance(res["feature_importance"])
                    if fi_fig:
                        st.plotly_chart(fi_fig, use_container_width=True)

        elif res["experiment_type"] == "regression":
            c_p1, c_p2 = st.columns([1, 1])
            with c_p1:
                reg_fig = plot_regression_actual_vs_predicted(res["y_true"], res["y_pred"])
                st.plotly_chart(reg_fig, use_container_width=True)
            with c_p2:
                if res.get("feature_importance"):
                    fi_fig = plot_feature_importance(res["feature_importance"])
                    if fi_fig:
                        st.plotly_chart(fi_fig, use_container_width=True)

        elif res["experiment_type"] == "clustering":
            pca_fig = plot_clustering_2d_pca(res["pca_data"])
            st.plotly_chart(pca_fig, use_container_width=True)
            st.caption("Cluster Distribution:")
            st.json(res["cluster_counts"])

        elif res["experiment_type"] == "parameter_sweep":
            sweep_fig = plot_parameter_sweep(res["sweep_data"], res["param_name"], res["metric_name"])
            st.plotly_chart(sweep_fig, use_container_width=True)
            st.dataframe(pd.DataFrame(res["sweep_data"]), use_container_width=True)

        # Transition to Results & Insights
        st.markdown("---")
        st.markdown("#### Proceed to Scientific Interpretation")
        if st.button("🧠 Interpret Results & Generate Next Steps ➡️", type="primary"):
            st.switch_page("pages/5_Results_Insights.py")
else:
    st.info("Select a dataset above to configure and execute an experiment.")
