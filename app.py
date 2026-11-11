import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import OLLAMA_SUPPORTED_MODELS
from core.state import (
    get_current_workspace,
    get_runtime_llm_settings,
    init_session_state,
    set_current_workspace,
)
from core.database import list_workspaces, list_documents, get_latest_plan, list_experiment_runs
from services.llm_service import (
    check_ollama_status,
    is_gemini_configured,
    test_ollama_prompt,
    unload_ollama_model,
)

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

    llm_cfg = get_runtime_llm_settings()
    active_provider = llm_cfg["provider"]
    active_model = llm_cfg["ollama_model"]

    if active_provider == "ollama":
        ollama_status = check_ollama_status(model=active_model)
        st.markdown("• **LLM Provider:** `Ollama (Local)`")
        st.markdown(f"• **Model:** `{active_model}`")
        if ollama_status["connected"]:
            st.markdown("• **Connection:** `Connected` ✅")
            if ollama_status["model_available"]:
                st.markdown("• **Model Status:** `Available` ✅")
            else:
                st.markdown("• **Model Status:** `Not Downloaded` ⚠️")
                st.caption(f"Run `ollama pull {active_model}` or switch to `qwen2.5:3b-instruct`.")
        else:
            st.markdown("• **Connection:** `Stopped / Unreachable` ❌")
            st.error(ollama_status["error"] or "Ollama is stopped. Start Ollama to run local AI.")
    else:
        gem_ok, gem_msg = is_gemini_configured()
        st.markdown("• **LLM Provider:** `Gemini (Optional)`")
        if gem_ok:
            st.markdown(f"• **Connection:** `{gem_msg}` ✅")
        else:
            st.markdown("• **Connection:** `Not Configured` ⚠️")

    st.markdown("• **Embeddings:** `all-MiniLM-L6-v2` (Local)")
    st.markdown("• **Vector Engine:** `FAISS CPU` (Local)")

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
st.subheader("🤖 Local LLM Settings & Diagnostics")
llm_cfg = get_runtime_llm_settings()

with st.container(border=True):
    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        provider_choice = st.selectbox(
            "LLM Provider",
            options=["ollama", "gemini"],
            index=0 if llm_cfg["provider"] == "ollama" else 1,
            format_func=lambda x: "Ollama (Local Offline Default)" if x == "ollama" else "Gemini (Optional Cloud)",
            help="AutoLab runs offline with local Ollama by default. Gemini is optional.",
        )
        if provider_choice != st.session_state.llm_provider:
            st.session_state.llm_provider = provider_choice
            st.rerun()

    with s_col2:
        if provider_choice == "ollama":
            ollama_info = check_ollama_status(model=llm_cfg["ollama_model"])
            model_choices = list(OLLAMA_SUPPORTED_MODELS)
            for m in ollama_info.get("installed_models", []):
                if m not in model_choices:
                    model_choices.append(m)
            current_m_idx = model_choices.index(llm_cfg["ollama_model"]) if llm_cfg["ollama_model"] in model_choices else 0

            selected_model = st.selectbox(
                "Local Ollama Model",
                options=model_choices,
                index=current_m_idx,
                help="Use qwen2.5:7b-instruct by default, or switch to qwen2.5:3b-instruct for lower RAM/VRAM usage.",
            )
            if selected_model != st.session_state.ollama_model:
                old_model = st.session_state.ollama_model
                unload_ollama_model(old_model)
                st.session_state.ollama_model = selected_model
                st.toast(f"Switched local model to {selected_model} (released {old_model}).")
                st.rerun()
        else:
            gem_ok, gem_msg = is_gemini_configured()
            st.text_input("Optional Cloud Provider", value="Gemini (Configured in .env)" if gem_ok else "Gemini (Key not set in .env)", disabled=True)

    with s_col3:
        ctx_options = [2048, 4096, 8192]
        cur_ctx = llm_cfg["ollama_num_ctx"] if llm_cfg["ollama_num_ctx"] in ctx_options else 4096
        selected_ctx = st.selectbox(
            "Context Window (num_ctx)",
            options=ctx_options,
            index=ctx_options.index(cur_ctx),
            disabled=(provider_choice != "ollama"),
            help="4096 tokens is recommended for 16 GB RAM / RTX 3050 laptops.",
        )
        if selected_ctx != st.session_state.ollama_num_ctx:
            st.session_state.ollama_num_ctx = selected_ctx
            st.rerun()

    if provider_choice == "ollama":
        status_now = check_ollama_status(model=st.session_state.ollama_model)
        if not status_now["connected"]:
            st.error(f"❌ **Ollama Stopped:** {status_now['error']}")
        elif not status_now["model_available"]:
            st.warning(f"⚠️ **Model Missing:** {status_now['error']}")
        else:
            st.success(f"✅ **Ollama Connected** (`{status_now['base_url']}`) — Active model `{st.session_state.ollama_model}` is ready.")

        b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
        with b_col1:
            if st.button("🔍 Check Ollama Connectivity", use_container_width=True):
                st.session_state["ollama_diag_result"] = check_ollama_status(model=st.session_state.ollama_model)
        with b_col2:
            if st.button("🧪 Test Sample Prompt", use_container_width=True):
                with st.spinner(f"Testing {st.session_state.ollama_model}..."):
                    try:
                        st.session_state["ollama_test_output"] = test_ollama_prompt(model=st.session_state.ollama_model)
                    except Exception as ex:
                        st.session_state["ollama_test_output"] = {"ok": False, "error": str(ex)}
        with b_col3:
            if st.button("🧹 Release Loaded Model", use_container_width=True, help="Unloads active model from GPU/CPU memory"):
                unload_ollama_model(st.session_state.ollama_model)
                st.toast(f"Released {st.session_state.ollama_model} from memory.")
                st.session_state["ollama_diag_result"] = check_ollama_status(model=st.session_state.ollama_model)

        diag = st.session_state.get("ollama_diag_result")
        if diag:
            if diag["connected"]:
                st.caption(f"Installed local models: `{diag['installed_models'] or ['None']}`")
                if diag["running_models"]:
                    for rm in diag["running_models"]:
                        st.info(
                            f"Active Memory Allocation (`{rm['name']}`): Total `{rm['size_mb']} MB` | "
                            f"GPU VRAM `{rm['vram_mb']} MB` | CPU RAM `{rm['cpu_ram_mb']} MB`"
                        )
            else:
                st.error(diag["error"])

        test_out = st.session_state.get("ollama_test_output")
        if test_out:
            if test_out.get("ok"):
                dur = f" ({test_out['eval_duration_s']}s)" if test_out.get("eval_duration_s") else ""
                st.success(f"Sample Prompt Response from `{test_out['model']}`{dur}: {test_out['response']}")
            else:
                st.error(f"Sample Prompt Failed: {test_out.get('error')}")

st.markdown("---")
with st.expander("ℹ️ How AutoLab Enforces Scientific Integrity"):
    st.markdown("""
    - **Safe Predefined Execution:** AutoLab never executes arbitrary Python or LLM-generated code. All runs execute tested, safe scikit-learn pipelines.
    - **Source-Grounded Evidence:** Document citations link directly to specific uploaded files and chunks.
    - **Zero Metric Fabrication:** Metrics and visualizations are computed directly from real datasets and train-test splits.
    - **Non-Causal Humility:** AutoLab highlights statistical correlations, caveats, sample-size limitations, and never claims an experiment 'proves' a hypothesis.
    """)
