import streamlit as st
from typing import Optional, Dict, Any
from core.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
)
from core.database import get_workspace, init_db

def init_session_state():
    """Ensure database is initialized and standard session keys exist."""
    init_db()

    if "current_workspace_id" not in st.session_state:
        st.session_state.current_workspace_id = None

    if "current_dataset_df" not in st.session_state:
        st.session_state.current_dataset_df = None

    if "current_dataset_name" not in st.session_state:
        st.session_state.current_dataset_name = None

    if "latest_run_id" not in st.session_state:
        st.session_state.latest_run_id = None

    if "latest_run_result" not in st.session_state:
        st.session_state.latest_run_result = None

    if "latest_generated_plan" not in st.session_state:
        st.session_state.latest_generated_plan = None

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = LLM_PROVIDER

    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = OLLAMA_MODEL

    if "ollama_num_ctx" not in st.session_state:
        st.session_state.ollama_num_ctx = OLLAMA_NUM_CTX


def get_runtime_llm_settings() -> Dict[str, Any]:
    """Returns active LLM provider settings from Streamlit session_state or config defaults."""
    provider = LLM_PROVIDER
    model = OLLAMA_MODEL
    num_ctx = OLLAMA_NUM_CTX
    try:
        provider = st.session_state.get("llm_provider", LLM_PROVIDER)
        model = st.session_state.get("ollama_model", OLLAMA_MODEL)
        num_ctx = int(st.session_state.get("ollama_num_ctx", OLLAMA_NUM_CTX))
    except Exception:
        pass
    return {
        "provider": str(provider or "ollama").strip().lower(),
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": str(model or OLLAMA_MODEL).strip(),
        "ollama_num_ctx": num_ctx,
        "ollama_temperature": OLLAMA_TEMPERATURE,
    }

def get_current_workspace() -> Optional[Dict[str, Any]]:
    init_session_state()
    ws_id = st.session_state.get("current_workspace_id")
    if ws_id is not None:
        ws = get_workspace(ws_id)
        if ws:
            return ws
        # If workspace was deleted from database
        st.session_state.current_workspace_id = None
    return None

def set_current_workspace(workspace_id: int):
    st.session_state.current_workspace_id = workspace_id

def clear_current_workspace():
    st.session_state.current_workspace_id = None
