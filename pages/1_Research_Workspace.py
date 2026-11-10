import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.state import init_session_state, get_current_workspace, set_current_workspace
from core.database import (
    create_workspace, get_workspace, list_workspaces, update_workspace,
    delete_workspace, list_documents, get_latest_plan, list_experiment_runs
)

st.set_page_config(page_title="AutoLab — Research Workspace", page_icon="🧭", layout="wide")
init_session_state()

st.title("🧭 Research Workspace")
st.markdown("Define your research question, formulate your hypothesis, and manage your investigation workspace.")

# Top action bar
col_left, col_right = st.columns([3, 1])
with col_right:
    if st.button("🧪 Load Iris Demo Workspace", help="Pre-populates the faculty demo scenario for Iris ML benchmark", use_container_width=True):
        demo_id = create_workspace(
            title="Iris Species Predictability Benchmark",
            research_question="Can a Random Forest classifier predict Iris species with higher accuracy and F1-score than Logistic Regression on standardized botanical measurements?",
            hypothesis="Nonlinear decision boundaries in Random Forest will yield superior multi-class classification metrics compared to linear Logistic Regression.",
            research_area="Machine Learning / Data Science",
            objective="Empirically compare Random Forest vs Logistic Regression on the Fisher Iris dataset under reproducible train/test splits.",
            constraints="Fixed 80/20 train-test split, seed 42, no external data leakage."
        )
        set_current_workspace(demo_id)
        st.success("Loaded Faculty Demo Workspace!")
        st.rerun()

current_ws = get_current_workspace()
workspaces = list_workspaces()

tab_create, tab_manage, tab_view = st.tabs(["➕ Create Workspace", "📂 Manage & Switch", "👁️ Active Workspace Details"])

with tab_create:
    st.subheader("New Research Workspace")
    with st.form("create_workspace_form"):
        title = st.text_input("Project Title*", placeholder="e.g. Crop Yield Prediction under Variable Soil Moisture")
        research_question = st.text_area(
            "Research Question*",
            placeholder="e.g. Does adding soil sensor moisture readings significantly improve crop yield forecast accuracy over climate-only regression models?"
        )
        hypothesis = st.text_area(
            "Hypothesis",
            placeholder="e.g. Incorporating depth-specific moisture metrics will reduce mean absolute error by at least 15%."
        )

        c1, c2 = st.columns(2)
        with c1:
            research_area = st.selectbox(
                "Research Area",
                ["Data Science", "Computer Science", "Environmental Science", "Biology & Bioinformatics", "Engineering", "Other"]
            )
        with c2:
            constraints = st.text_input("Constraints", placeholder="e.g. Maximum compute 1 CPU core, sample size n < 1000")

        objective = st.text_area("Specific Objectives", placeholder="e.g. Compare Linear Regression vs Random Forest on cross-validated RMSE.")

        submitted = st.form_submit_button("Create & Activate Workspace", type="primary")
        if submitted:
            if not title.strip() or not research_question.strip():
                st.error("Please provide both a Title and a Research Question.")
            else:
                new_id = create_workspace(
                    title=title.strip(),
                    research_question=research_question.strip(),
                    hypothesis=hypothesis.strip(),
                    research_area=research_area,
                    objective=objective.strip(),
                    constraints=constraints.strip()
                )
                set_current_workspace(new_id)
                st.success(f"Workspace '{title}' created and activated!")
                st.rerun()

with tab_manage:
    st.subheader("Existing Workspaces")
    if not workspaces:
        st.info("No workspaces created yet.")
    else:
        for ws in workspaces:
            with st.container(border=True):
                c_title, c_meta, c_action = st.columns([3, 2, 2])
                with c_title:
                    st.markdown(f"### {ws['title']}")
                    st.caption(f"Area: `{ws['research_area']}` | Updated: `{ws['updated_at'][:19]}`")
                    st.markdown(f"**Question:** {ws['research_question']}")
                with c_meta:
                    docs = list_documents(ws["id"])
                    plan = get_latest_plan(ws["id"])
                    runs = list_experiment_runs(ws["id"])
                    st.markdown(f"• Evidence Docs: **{len(docs)}**")
                    st.markdown(f"• Plan: **{'Yes' if plan else 'None'}**")
                    st.markdown(f"• Experiments Run: **{len(runs)}**")
                with c_action:
                    is_active = current_ws and current_ws["id"] == ws["id"]
                    if is_active:
                        st.button("✅ Active", disabled=True, key=f"active_{ws['id']}")
                    else:
                        if st.button("Select Active", key=f"sel_{ws['id']}", use_container_width=True):
                            set_current_workspace(ws["id"])
                            st.rerun()

                    if st.button("🗑️ Delete", key=f"del_{ws['id']}", use_container_width=True):
                        delete_workspace(ws["id"])
                        if is_active:
                            st.session_state.current_workspace_id = None
                        st.warning(f"Deleted workspace {ws['title']}.")
                        st.rerun()

with tab_view:
    if not current_ws:
        st.warning("No active workspace selected. Choose or create one in the tabs above.")
    else:
        st.subheader(f"Active: {current_ws['title']}")
        st.caption(f"Workspace ID: {current_ws['id']} | Created: {current_ws['created_at']}")

        with st.form("edit_workspace_form"):
            e_title = st.text_input("Project Title", value=current_ws['title'])
            e_question = st.text_area("Research Question", value=current_ws['research_question'])
            e_hypothesis = st.text_area("Hypothesis", value=current_ws['hypothesis'])
            e_area = st.text_input("Research Area", value=current_ws['research_area'])
            e_obj = st.text_area("Objective", value=current_ws['objective'])
            e_cons = st.text_area("Constraints", value=current_ws['constraints'])

            if st.form_submit_button("Save Changes"):
                update_workspace(
                    current_ws["id"],
                    title=e_title.strip(),
                    research_question=e_question.strip(),
                    hypothesis=e_hypothesis.strip(),
                    research_area=e_area.strip(),
                    objective=e_obj.strip(),
                    constraints=e_cons.strip()
                )
                st.success("Workspace details updated successfully!")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Next Step")
        if st.button("Proceed to Knowledge Library ➡️", type="primary"):
            st.switch_page("pages/2_Knowledge_Library.py")
