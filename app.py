import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import GEMINI_API_KEY, GEMINI_MODEL
from core.state import init_session_state, get_current_workspace, set_current_workspace
from core.database import list_workspaces, list_documents, get_latest_plan, list_experiment_runs

st.set_page_config(
    page_title="AutoLab — AI Scientific Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()

# ----------------- SIDEBAR ----------------- #
with st.sidebar:
    st.markdown("## 🔬 **AutoLab**")
    st.caption("AI-Assisted Scientific Experiment Platform")
    st.divider()

    # Workspace Selector
    workspaces = list_workspaces()
    current_ws = get_current_workspace()

    if workspaces:
        ws_options = {f"{w['id']} - {w['title']}": w['id'] for w in workspaces}
        current_index = 0
        if current_ws:
            for idx, (label, wid) in enumerate(ws_options.items()):
                if wid == current_ws["id"]:
                    current_index = idx
                    break

        selected_label = st.selectbox(
            "Active Workspace",
            options=list(ws_options.keys()),
            index=current_index,
            help="Switch between existing research workspaces."
        )
        selected_id = ws_options[selected_label]
        if not current_ws or current_ws["id"] != selected_id:
            set_current_workspace(selected_id)
            st.rerun()
    else:
        st.info("No research workspace created yet.")
        if st.button("➕ Create Workspace", use_container_width=True):
            st.switch_page("pages/1_Research_Workspace.py")

    st.divider()

    # System Status
    st.markdown("### ⚙️ System Status")
    st.markdown("• **Database:** `SQLite (Online)`")
    if GEMINI_API_KEY:
        st.markdown(f"• **LLM Engine:** `Connected ({GEMINI_MODEL})` ✅")
    else:
        st.markdown("• **LLM Engine:** `API Key Missing` ⚠️")
        st.caption("Add `GEMINI_API_KEY` to `.env` to enable AI planner & insights.")

    st.markdown("• **Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("• **Vector Engine:** `FAISS CPU`")

# ----------------- MAIN OVERVIEW ----------------- #
st.title("🔬 AutoLab: AI Scientific Research Assistant")
st.markdown("""
Welcome to **AutoLab**, a specialized, software-based research laboratory designed to guide researchers
through structured scientific inquiry: from literature grounding to safe empirical execution and interpretation.
""")

current_ws = get_current_workspace()

if current_ws:
    st.success(f"**Current Active Workspace:** `{current_ws['title']}` (ID: {current_ws['id']})")

    # Workflow Progress Stepper
    docs = list_documents(current_ws["id"])
    plan = get_latest_plan(current_ws["id"])
    runs = list_experiment_runs(current_ws["id"])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("1. Workspace", "Configured ✅")
    with col2:
        st.metric("2. Evidence Library", f"{len(docs)} Docs {'✅' if docs else '⏳'}")
    with col3:
        st.metric("3. Experiment Plan", "Ready ✅" if plan else "Pending ⏳")
    with col4:
        st.metric("4. Executions", f"{len(runs)} Runs {'✅' if runs else '⏳'}")
    with col5:
        st.metric("5. Analysis", "Available ✅" if runs else "Pending ⏳")

    st.markdown("---")
    st.subheader("📋 Active Research Overview")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(f"**Research Question:**\n> {current_ws['research_question']}")
        st.markdown(f"**Hypothesis:**\n> {current_ws['hypothesis'] or '*(None stated)*'}")
        st.markdown(f"**Discipline:** `{current_ws['research_area']}`")
    with info_col2:
        st.markdown(f"**Objective:**\n{current_ws['objective'] or '*(None specified)*'}")
        st.markdown(f"**Constraints:**\n{current_ws['constraints'] or '*(None specified)*'}")

    st.markdown("### 🚀 Quick Navigation")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        if st.button("📚 Knowledge Library", use_container_width=True):
            st.switch_page("pages/2_Knowledge_Library.py")
    with nav_col2:
        if st.button("📝 Experiment Planner", use_container_width=True):
            st.switch_page("pages/3_Experiment_Planner.py")
    with nav_col3:
        if st.button("⚡ Run Experiment", use_container_width=True):
            st.switch_page("pages/4_Run_Experiment.py")
    with nav_col4:
        if st.button("📊 Results & Insights", use_container_width=True):
            st.switch_page("pages/5_Results_Insights.py")

else:
    st.info("👋 You don't have an active workspace selected. Please create or select one to begin!")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("✨ Go to Research Workspace", type="primary", use_container_width=True):
            st.switch_page("pages/1_Research_Workspace.py")

st.markdown("---")
with st.expander("ℹ️ How AutoLab Enforces Scientific Integrity"):
    st.markdown("""
    - **Safe Predefined Execution:** AutoLab never executes arbitrary Python or LLM-generated code. All runs execute tested, safe scikit-learn pipelines.
    - **Source-Grounded Evidence:** Document citations link directly to specific uploaded files and chunks.
    - **Zero Metric Fabrication:** Metrics and visualizations are computed directly from real datasets and train-test splits.
    - **Non-Causal Humility:** AutoLab highlights statistical correlations, caveats, sample-size limitations, and never claims an experiment 'proves' a hypothesis.
    """)
