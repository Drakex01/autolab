import streamlit as st
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.state import init_session_state, get_current_workspace
from core.database import save_experiment_plan, get_latest_plan, list_plans
from core.schemas import ExperimentPlanSchema
from services.planning_service import create_experiment_plan, build_evidence_context
from services.retrieval_service import retrieve_relevant_chunks
from services.llm_service import is_gemini_configured, GeminiQuotaExhaustedError

st.set_page_config(page_title="AutoLab — Experiment Planner", page_icon="📝", layout="wide")
init_session_state()

st.title("📝 Experiment Planner")
st.markdown("Formulate structured, evidence-grounded scientific experiment plans conforming to empirical standards.")

current_ws = get_current_workspace()
if not current_ws:
    st.warning("⚠️ No active workspace selected. Please select or create one in **Research Workspace** first.")
    if st.button("Go to Research Workspace"):
        st.switch_page("pages/1_Research_Workspace.py")
    st.stop()

# LLM Status Banner
is_configured, status_msg = is_gemini_configured()
if not is_configured:
    st.warning(f"⚠️ {status_msg} (A deterministic methodological template will be generated if you proceed without an API key).")
else:
    st.success(f"✨ Gemini LLM Engine active ({status_msg})")

# Top Context Expander
with st.expander("🔍 Verified Literature Evidence Context for this Plan", expanded=False):
    evidence_chunks = retrieve_relevant_chunks(current_ws["id"], query=current_ws["research_question"], top_k=5)
    if evidence_chunks:
        st.markdown(f"Found **{len(evidence_chunks)}** relevant literature chunks:")
        for idx, ch in enumerate(evidence_chunks, 1):
            st.markdown(f"• **Chunk #{idx}** (`{ch.source_filename}`, Page {ch.page_number}): *\"{ch.content[:180]}...\"*")
    else:
        st.info("No documents uploaded yet in Knowledge Library. The plan will be constructed from general methodology.")

col_act1, col_act2 = st.columns([3, 1])
with col_act1:
    st.markdown(f"**Target:** `{current_ws['title']}` | **Area:** `{current_ws['research_area']}`")
with col_act2:
    if st.button("⚡ Generate Structured Plan", type="primary", use_container_width=True):
        with st.spinner("Synthesizing research questions, literature evidence, and experimental protocols..."):
            try:
                plan_schema = create_experiment_plan(current_ws)
                st.session_state.latest_generated_plan = plan_schema.model_dump()
                # Save to database
                plan_id = save_experiment_plan(current_ws["id"], plan_schema.model_dump())
                st.success(f"Generated & saved new Experiment Plan #{plan_id}!")
                st.rerun()
            except GeminiQuotaExhaustedError as qe:
                st.warning(f"⚠️ {qe}")
            except Exception as e:
                st.error(f"Plan generation failed: {e}")

# Check current plan
active_plan = st.session_state.get("latest_generated_plan")
if not active_plan:
    db_plan = get_latest_plan(current_ws["id"])
    if db_plan:
        active_plan = db_plan["plan"]
        st.session_state.latest_generated_plan = active_plan

if active_plan:
    st.markdown("---")
    # Grounding status badge
    is_grounded = active_plan.get("is_grounded_in_sources", False)
    if is_grounded:
        st.success("🛡️ **Source-Verified Plan:** Grounded in uploaded evidence documents from the Knowledge Library.")
    else:
        st.info("ℹ️ **General Knowledge Plan:** No matching domain literature found; plan is based on general scientific methodology.")

    tab_structured, tab_edit, tab_history = st.tabs(["📋 Structured Plan View", "✏️ Edit & Customize", "📜 Plan Revisions"])

    with tab_structured:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 1. Objective")
            st.write(active_plan.get("objective", ""))

            st.markdown("### 2. Testable Hypothesis")
            st.info(active_plan.get("testable_hypothesis", ""))

            st.markdown("### 3. Independent Variables")
            st.write(active_plan.get("independent_variables", []))

            st.markdown("### 4. Dependent Variables")
            st.write(active_plan.get("dependent_variables", []))

            st.markdown("### 5. Controlled Variables & Assumptions")
            st.write(active_plan.get("controlled_variables", []))

            st.markdown("### 6. Dataset & Required Columns")
            st.write(active_plan.get("dataset_and_required_columns", ""))

        with c2:
            st.markdown("### 7. Methodology")
            st.write(active_plan.get("methodology", ""))

            st.markdown("### 8. Baseline Comparison")
            st.write(active_plan.get("baseline", ""))

            st.markdown("### 9. Evaluation Metrics")
            st.write(active_plan.get("evaluation_metrics", []))

            st.markdown("### 10. Expected Outputs")
            st.write(active_plan.get("expected_outputs", ""))

            st.markdown("### 11. Risks & Limitations")
            st.warning(active_plan.get("risks_and_limitations", ""))

            st.markdown("### 12. Next Steps")
            st.write(active_plan.get("next_steps", []))

        # Citations Section
        citations = active_plan.get("evidence_citations", [])
        if citations:
            st.markdown("### 📚 Grounded Evidence Citations")
            for cit in citations:
                with st.container(border=True):
                    st.markdown(f"**Source:** `{cit.get('source_file')}` ({cit.get('page_or_chunk')})")
                    st.markdown(f"> *{cit.get('factual_note')}*")

    with tab_edit:
        st.subheader("Edit Plan Elements")
        with st.form("edit_plan_form"):
            e_obj = st.text_area("Objective", value=active_plan.get("objective", ""))
            e_hyp = st.text_area("Testable Hypothesis", value=active_plan.get("testable_hypothesis", ""))
            e_method = st.text_area("Methodology", value=active_plan.get("methodology", ""), height=150)
            e_base = st.text_input("Baseline", value=active_plan.get("baseline", ""))
            e_metrics = st.text_input("Evaluation Metrics (comma-separated)", value=", ".join(active_plan.get("evaluation_metrics", [])))
            e_dataset = st.text_area("Dataset Requirements", value=active_plan.get("dataset_and_required_columns", ""))
            e_risks = st.text_area("Risks & Limitations", value=active_plan.get("risks_and_limitations", ""))

            if st.form_submit_button("Save Customized Plan"):
                active_plan["objective"] = e_obj
                active_plan["testable_hypothesis"] = e_hyp
                active_plan["methodology"] = e_method
                active_plan["baseline"] = e_base
                active_plan["evaluation_metrics"] = [m.strip() for m in e_metrics.split(",") if m.strip()]
                active_plan["dataset_and_required_columns"] = e_dataset
                active_plan["risks_and_limitations"] = e_risks

                plan_id = save_experiment_plan(current_ws["id"], active_plan)
                st.session_state.latest_generated_plan = active_plan
                st.success(f"Customized plan saved as revision #{plan_id}!")
                st.rerun()

    with tab_history:
        st.subheader("Previous Plan Revisions")
        all_plans = list_plans(current_ws["id"])
        if not all_plans:
            st.info("No prior revisions.")
        else:
            for p in all_plans:
                with st.expander(f"Revision #{p['id']} — Created: {p['created_at'][:19]}"):
                    st.json(p["plan"])

    st.markdown("---")
    st.markdown("#### Ready to Execute?")
    if st.button("Proceed to Run Experiment ➡️", type="primary"):
        st.switch_page("pages/4_Run_Experiment.py")
else:
    st.info("Click **Generate Structured Plan** above to formulate a research plan.")
