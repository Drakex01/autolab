import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.state import init_session_state, get_current_workspace, get_runtime_llm_settings
from core.database import (
    get_experiment_run, list_experiment_runs, update_run_interpretation,
    get_latest_plan
)
from services.analysis_service import (
    analyze_and_interpret_run,
    build_comparative_analysis,
    RULE_BASED_LABEL,
    RULE_BASED_COMPARATIVE_LABEL,
    GEMINI_LABEL,
    OLLAMA_LABEL,
)
from services.experiment_service import (
    plot_confusion_matrix, plot_regression_actual_vs_predicted,
    plot_clustering_2d_pca, plot_parameter_sweep, plot_feature_importance
)
from services.llm_service import is_llm_configured

st.set_page_config(page_title="AutoLab — Results & Insights", page_icon="📊", layout="wide")
init_session_state()

st.title("📊 Results & Scientific Insights")
st.markdown("Rigorous empirical evaluation, plain-language interpretation, and actionable follow-up recommendations.")

current_ws = get_current_workspace()
if not current_ws:
    st.warning("⚠️ No active workspace selected. Please select or create one in **Research Workspace** first.")
    if st.button("Go to Research Workspace"):
        st.switch_page("pages/1_Research_Workspace.py")
    st.stop()

# Select Run to Analyze
runs = list_experiment_runs(current_ws["id"])
if not runs:
    st.info("No experiment runs recorded for this workspace yet. Run an experiment first!")
    if st.button("Go to Run Experiment ⚡", type="primary"):
        st.switch_page("pages/4_Run_Experiment.py")
    st.stop()

run_options = {f"Run #{r['id']} — {r['experiment_type'].upper()} ({r['dataset_name']}) at {r['created_at'][:19]}": r['id'] for r in runs}

latest_id = st.session_state.get("latest_run_id")
default_idx = 0
if latest_id:
    for idx, (lbl, rid) in enumerate(run_options.items()):
        if rid == latest_id:
            default_idx = idx
            break

chosen_label = st.selectbox("Select Experiment Run to Analyze", list(run_options.keys()), index=default_idx)
selected_run_id = run_options[chosen_label]
selected_run = get_experiment_run(selected_run_id)

st.divider()

# 1. EMPIRICAL BENCHMARKS (FACTUAL METRICS)
st.subheader("1. Empirical Measurements (scikit-learn)")
st.caption("These metrics were directly calculated from the executed dataset partition. Zero fabrication.")

col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
with col_m1:
    st.markdown(f"**Dataset:** `{selected_run['dataset_name']}`")
    st.markdown(f"**Type:** `{selected_run['experiment_type'].title()}`")
with col_m2:
    st.markdown(f"**Model:** `{selected_run['config'].get('model_type', selected_run.get('experiment_type'))}`")
    st.markdown(f"**Executed:** `{selected_run['created_at'][:19]}`")
with col_m3:
    st.json(selected_run["metrics"], expanded=False)

# Metric Cards (scalar metrics only)
metrics_data = {
    k: v for k, v in (selected_run["metrics"] or {}).items()
    if isinstance(v, (int, float, str)) and not isinstance(v, bool)
}
if metrics_data:
    metric_cols = st.columns(min(4, len(metrics_data)))
    for idx, (m_name, m_val) in enumerate(metrics_data.items()):
        with metric_cols[idx % len(metric_cols)]:
            st.metric(label=m_name.replace('_', ' ').title(), value=f"{m_val}")

# 2. COMPARATIVE DISCOVERY & SCIENTIFIC INTERPRETATION
st.markdown("---")
st.subheader("2. Comparative Analysis & Scientific Interpretation")

existing_interp = selected_run.get("interpretation")
existing_recs_str = selected_run.get("next_experiment_recommendation")

preview_comp = (
    build_comparative_analysis(selected_run, workspace_runs=runs)
    if selected_run["experiment_type"] == "classification"
    else None
)

selected_eval_metrics = None
if preview_comp:
    if preview_comp.get("comparison_available"):
        sig = preview_comp["selected_signature"]
        st.success(
            f"🔗 **Comparable Runs Detected in Workspace:** Random Forest (Run #{preview_comp['rf_run_id']}) vs "
            f"Logistic Regression (Run #{preview_comp['lr_run_id']}) on `{sig['dataset_name']}` "
            f"(split=`{sig['test_size']}`, seed=`{sig['random_state']}`)."
        )
        available_metric_keys = [row["metric_key"] for row in preview_comp["metrics_table"]]
        if available_metric_keys:
            selected_eval_metrics = st.multiselect(
                "Evaluation Metrics for Hypothesis Assessment",
                options=available_metric_keys,
                default=available_metric_keys,
                help="Hypothesis support is evaluated strictly across the selected recorded metrics.",
            )
    else:
        st.info(f"ℹ️ **Comparative Context:** {preview_comp.get('reason')}")

llm_cfg = get_runtime_llm_settings()
active_provider = llm_cfg["provider"]
is_configured, status_msg = is_llm_configured()
if active_provider == "ollama":
    if is_configured:
        st.success(f"🤖 **Local Ollama Engine active:** `{status_msg}`")
    else:
        st.warning(f"⚠️ **Ollama Unavailable:** {status_msg} AutoLab will generate a deterministic rule-based interpretation if you proceed.")
else:
    if not is_configured:
        st.info(f"💡 {status_msg} AutoLab will generate a deterministic rule-based interpretation.")

col_act1, col_act2 = st.columns([3, 1])
with col_act2:
    btn_label = "🔄 Re-Analyze with AI" if existing_interp else "🧠 Generate Scientific Interpretation"
    if st.button(btn_label, type="primary", use_container_width=True):
        with st.spinner("Analyzing comparable runs, hypotheses, and evidence..."):
            try:
                plan = get_latest_plan(current_ws["id"])
                plan_dict = plan["plan"] if plan else None
                analysis_out = analyze_and_interpret_run(
                    current_ws,
                    selected_run,
                    plan_dict=plan_dict,
                    workspace_runs=runs,
                    selected_metrics=selected_eval_metrics or None,
                )

                interp_json = json.dumps(analysis_out["interpretation"])
                recs_json = json.dumps(analysis_out["recommendations"])

                update_run_interpretation(selected_run["id"], interp_json, recs_json)
                if analysis_out.get("ollama_unavailable"):
                    st.session_state["last_provider_notice"] = {
                        "run_id": selected_run["id"],
                        "kind": "ollama_unavailable",
                        "message": analysis_out.get("provider_error") or "Local Ollama is stopped or model not installed.",
                        "source": analysis_out.get("source"),
                    }
                elif analysis_out.get("quota_exhausted"):
                    st.session_state["last_provider_notice"] = {
                        "run_id": selected_run["id"],
                        "kind": "quota_exhausted",
                        "message": analysis_out.get("quota_message"),
                        "retry_after": analysis_out.get("retry_after"),
                        "source": analysis_out.get("source"),
                    }
                else:
                    st.session_state.pop("last_provider_notice", None)
                    st.success("Analysis complete and persisted to database!")
                st.rerun()
            except Exception as e:
                st.error(f"Interpretation failed: {e}")

provider_notice = st.session_state.get("last_provider_notice")
if provider_notice and provider_notice.get("run_id") == selected_run["id"]:
    if provider_notice.get("kind") == "ollama_unavailable":
        if provider_notice.get("source") == "ollama_partial":
            st.warning(
                f"⚠️ **Ollama became unavailable** while generating follow-up recommendations: {provider_notice.get('message')} "
                "Your experiment results and Ollama interpretation were preserved, and deterministic rule-based recommendations are shown below."
            )
        else:
            st.error(
                f"❌ **Local Ollama Unavailable:** {provider_notice.get('message')} "
                "All experiment results and workspace data were preserved, and a deterministic rule-based fallback was generated below."
            )
    else:
        retry_text = f" Suggested retry time: **{provider_notice['retry_after']}**." if provider_notice.get("retry_after") else ""
        if provider_notice.get("source") == "gemini_partial":
            st.warning(
                f"⚠️ **Gemini daily quota has been exhausted** while generating follow-up recommendations.{retry_text} "
                "Your experiment results and Gemini interpretation were preserved, and deterministic rule-based recommendations are shown below. "
                "You can click **Re-Analyze with AI** once quota resets."
            )
        else:
            st.warning(
                f"⚠️ **Gemini daily quota has been exhausted (HTTP 429 RESOURCE_EXHAUSTED).**{retry_text} "
                "All experiment results and workspace data have been preserved. A deterministic rule-based interpretation was generated below; "
                "you can click **Re-Analyze with AI** once API quota becomes available again."
            )

# Render Interpretation if present
if existing_interp:
    try:
        interp_data = json.loads(existing_interp)
    except Exception:
        interp_data = {"summary": existing_interp, "what_results_suggest": "", "hypothesis_status": "Inconclusive", "possible_reasons": [], "limitations": []}

    is_rule_based = (
        interp_data.get("source") == "rule_based"
        or interp_data.get("source_label") == RULE_BASED_LABEL
    )
    if (
        not (provider_notice and provider_notice.get("run_id") == selected_run["id"])
        and interp_data.get("fallback_reason") == "quota_exhausted"
    ):
        retry_hint = f" Suggested retry time: **{interp_data['retry_after']}**." if interp_data.get("retry_after") else ""
        st.warning(
            f"⚠️ This run was interpreted during a **Gemini daily quota exhaustion** event.{retry_hint} "
            "Click **Re-Analyze with AI** to generate a Gemini interpretation when quota is available."
        )
    elif (
        not (provider_notice and provider_notice.get("run_id") == selected_run["id"])
        and interp_data.get("fallback_reason") == "ollama_unavailable"
    ):
        st.warning(
            "⚠️ This run was interpreted with the rule-based fallback because **Ollama was unavailable**. "
            "Start Ollama and click **Re-Analyze with AI** for local LLM synthesis."
        )

    status_tag = interp_data.get("hypothesis_status", "Inconclusive")
    status_color = "🟢" if status_tag == "Supported" else "🔴" if status_tag == "Not Supported" else "🟡"

    with st.container(border=True):
        if is_rule_based:
            if interp_data.get("analysis_mode") == "comparative_rule_based":
                st.warning(f"📐 **{RULE_BASED_COMPARATIVE_LABEL}**")
            else:
                st.warning(f"📐 **{RULE_BASED_LABEL}**")
        elif interp_data.get("source") == "ollama":
            model_tag = interp_data.get("llm_model") or llm_cfg["ollama_model"]
            if interp_data.get("analysis_mode") == "comparative_ollama":
                st.success(f"🤖 **{OLLAMA_LABEL} — `{model_tag}` (Comparative Multi-Run Context)**")
            else:
                st.success(f"🤖 **{OLLAMA_LABEL} — `{model_tag}`**")
        elif interp_data.get("source") == "gemini":
            if interp_data.get("analysis_mode") == "comparative_gemini":
                st.success(f"✨ **{GEMINI_LABEL} (Comparative Multi-Run Context)**")
            else:
                st.success(f"✨ **{GEMINI_LABEL}**")

        st.markdown(f"### Hypothesis Assessment: {status_color} **{status_tag}**")
        st.markdown(f"**Research Question:** *{current_ws['research_question']}*")
        st.markdown(f"**Tested Hypothesis:** *{current_ws['hypothesis'] or 'No hypothesis specified'}*")

        st.markdown("#### 🔬 Findings Summary")
        st.info(interp_data.get("summary", ""))

        st.markdown("#### 📈 What the Results Suggest")
        st.write(interp_data.get("what_results_suggest", ""))

        comp_data = interp_data.get("comparison") or preview_comp
        if comp_data and comp_data.get("comparison_available") and comp_data.get("metrics_table"):
            st.markdown("#### ⚖️ Comparative Metrics Table (Random Forest vs Logistic Regression)")
            df_comp_table = pd.DataFrame([
                {
                    "Metric": row["metric"],
                    "Random Forest": row["random_forest"],
                    "Logistic Regression": row["logistic_regression"],
                    "RF (%)": f"{row['rf_pct']:.2f}%",
                    "LR (%)": f"{row['lr_pct']:.2f}%",
                    "Signed Diff (RF − LR)": f"{row['signed_diff_pp']:+.2f} pp",
                    "Absolute Diff (pp)": f"{row['abs_diff_pp']:.2f} pp",
                    "Better Model": row["winner"],
                }
                for row in comp_data["metrics_table"]
            ])
            st.dataframe(df_comp_table, use_container_width=True, hide_index=True)

            cls_comp = comp_data.get("class_level_comparison") or {}
            st.markdown("#### 🧩 Class-Specific Metrics & Confusion Matrix Comparison")
            if cls_comp.get("available"):
                df_cls = pd.DataFrame(cls_comp.get("per_class_rows", []))
                if not df_cls.empty:
                    st.dataframe(df_cls, use_container_width=True, hide_index=True)
                cm_rf = cls_comp.get("rf_confusion_matrix")
                cm_lr = cls_comp.get("lr_confusion_matrix")
                if cm_rf is not None and cm_lr is not None:
                    cm_col1, cm_col2 = st.columns(2)
                    with cm_col1:
                        st.markdown("**Random Forest Confusion Matrix**")
                        st.plotly_chart(
                            plot_confusion_matrix(cm_rf, cls_comp.get("rf_classes") or [str(i) for i in range(len(cm_rf))]),
                            use_container_width=True,
                            key="cm_rf_comp",
                        )
                    with cm_col2:
                        st.markdown("**Logistic Regression Confusion Matrix**")
                        st.plotly_chart(
                            plot_confusion_matrix(cm_lr, cls_comp.get("lr_classes") or [str(i) for i in range(len(cm_lr))]),
                            use_container_width=True,
                            key="cm_lr_comp",
                        )
            else:
                st.info(cls_comp.get("note", "Class-specific metrics and confusion matrices were not recorded."))

            completed_list = comp_data.get("completed_comparisons") or []
            if completed_list:
                st.caption("**Already Completed (Excluded from Recommendations):** " + " | ".join(completed_list))

        c_r, c_l = st.columns(2)
        with c_r:
            st.markdown("#### 🧩 Possible Reasons & Mechanisms")
            reasons = interp_data.get("possible_reasons", [])
            for r in reasons:
                st.markdown(f"• {r}")
        with c_l:
            st.markdown("#### ⚠️ Experimental Limitations & Caveats")
            lims = interp_data.get("limitations", [])
            for l in lims:
                st.markdown(f"• {l}")

    # 3. NEXT EXPERIMENT RECOMMENDATIONS
    st.markdown("---")
    st.subheader("3. Concrete Follow-Up Experiment Recommendations")
    st.markdown("Actionable next steps derived from observed empirical variances and literature evidence.")

    recs_list = []
    if existing_recs_str:
        try:
            recs_list = json.loads(existing_recs_str)
        except Exception:
            recs_list = []

    if not recs_list:
        st.caption("Click 'Generate Scientific Interpretation' above to produce next-experiment recommendations.")
    else:
        for idx, rec in enumerate(recs_list, 1):
            with st.container(border=True):
                st.markdown(f"### Recommendation #{idx}: {rec.get('title')}")
                st.markdown(f"**Scientific Rationale:** {rec.get('why_useful')}")

                c_ch, c_co, c_me = st.columns(3)
                with c_ch:
                    st.markdown(f"**What to Change:**\n> {rec.get('what_to_change')}")
                with c_co:
                    st.markdown(f"**What to Keep Constant:**\n> {rec.get('what_to_keep_constant')}")
                with c_me:
                    st.markdown(f"**What to Measure:**\n> {rec.get('what_to_measure')}")

                st.markdown(f"**Expected Hypothesis Outcome:** {rec.get('expected_support_or_weakening')}")

                if rec.get("can_run_predefined"):
                    st.caption("✅ This experiment can be executed directly using AutoLab's predefined runner.")
                    if st.button(f"⚡ Configure & Run '{rec.get('title')}'", key=f"rec_run_{idx}", type="secondary"):
                        st.switch_page("pages/4_Run_Experiment.py")

else:
    st.info("Click 'Generate Scientific Interpretation' above to synthesize AI insights for this run.")

st.markdown("---")
st.markdown("#### Review Historical Runs")
if st.button("View Full Experiment History ➡️", type="primary"):
    st.switch_page("pages/6_Experiment_History.py")
