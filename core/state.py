import streamlit as st
from typing import Optional, Dict, Any
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
