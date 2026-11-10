import streamlit as st
import sys
import shutil
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import UPLOADS_DIR, SAMPLE_DOCS_DIR, MAX_UPLOAD_SIZE_MB, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
from core.state import init_session_state, get_current_workspace
from core.database import add_document, list_documents, delete_document
from services.document_service import extract_text_from_file
from services.chunking_service import split_text_into_chunks
from services.retrieval_service import add_chunks_to_index, retrieve_relevant_chunks, clear_workspace_index

st.set_page_config(page_title="AutoLab — Knowledge Library", page_icon="📚", layout="wide")
init_session_state()

st.title("📚 Knowledge Library & Evidence Retrieval")
st.markdown("Upload domain literature, technical papers, and notes. AutoLab indexes and grounds your research plans in verified evidence.")

current_ws = get_current_workspace()
if not current_ws:
    st.warning("⚠️ No active workspace selected. Please select or create one in **Research Workspace** first.")
    if st.button("Go to Research Workspace"):
        st.switch_page("pages/1_Research_Workspace.py")
    st.stop()

st.info(f"**Target Workspace:** `{current_ws['title']}` (ID: {current_ws['id']})")

tab_upload, tab_search, tab_docs = st.tabs(["📤 Upload & Ingest", "🔎 Evidence Search & Inspection", "📑 Indexed Documents"])

with tab_upload:
    st.subheader("Ingest Literature into Workspace")

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn2:
        if st.button("📄 Load Sample Research PDF", help="Ingests pre-compiled 2-page Iris ML study into this workspace", use_container_width=True):
            sample_pdf = SAMPLE_DOCS_DIR / "iris_classification_study.pdf"
            if sample_pdf.exists():
                with st.spinner("Extracting, chunking and embedding sample PDF..."):
                    dest_name = f"sample_{uuid.uuid4().hex[:6]}_iris_classification_study.pdf"
                    dest_path = UPLOADS_DIR / dest_name
                    shutil.copyfile(sample_pdf, dest_path)

                    pages_data = extract_text_from_file(dest_path, "iris_classification_study.pdf")
                    doc_id = add_document(
                        workspace_id=current_ws["id"],
                        filename="iris_classification_study.pdf",
                        file_type="pdf",
                        file_path=str(dest_path),
                        chunk_count=0
                    )
                    chunks = split_text_into_chunks(
                        pages_data=pages_data,
                        source_filename="iris_classification_study.pdf",
                        document_id=doc_id,
                        workspace_id=current_ws["id"],
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    add_chunks_to_index(current_ws["id"], chunks)
                    st.success(f"Successfully ingested 'iris_classification_study.pdf' ({len(chunks)} chunks, 2 pages)!")
                    st.rerun()

    uploaded_files = st.file_uploader(
        f"Upload Scientific Literature (PDF, TXT, Markdown — Max {MAX_UPLOAD_SIZE_MB}MB per file)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Process & Ingest Uploaded Files", type="primary"):
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            for idx, file in enumerate(uploaded_files):
                if file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    st.error(f"File {file.name} exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")
                    continue

                safe_ext = Path(file.name).suffix.lower()
                safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.name).name}"
                dest_path = UPLOADS_DIR / safe_name

                with open(dest_path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    pages_data = extract_text_from_file(dest_path, file.name)
                    doc_id = add_document(
                        workspace_id=current_ws["id"],
                        filename=file.name,
                        file_type=safe_ext.replace(".", ""),
                        file_path=str(dest_path),
                        chunk_count=0
                    )
                    chunks = split_text_into_chunks(
                        pages_data=pages_data,
                        source_filename=file.name,
                        document_id=doc_id,
                        workspace_id=current_ws["id"],
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    add_chunks_to_index(current_ws["id"], chunks)
                    st.toast(f"Ingested {file.name}: {len(chunks)} chunks.")
                except Exception as ex:
                    st.error(f"Failed to process {file.name}: {ex}")

                progress_bar.progress((idx + 1) / total_files)

            st.success("All uploads ingested into local FAISS vector store!")
            st.rerun()

with tab_search:
    st.subheader("Direct Evidence Retrieval Test")
    st.markdown("Query the vector database to inspect which source excerpts will be passed as context to the AI Experiment Planner.")

    default_query = current_ws["research_question"]
    query = st.text_input("Search / Research Query", value=default_query)
    top_k = st.slider("Number of Chunks to Retrieve", min_value=2, max_value=8, value=TOP_K_RETRIEVAL)

    if st.button("🔍 Retrieve Evidence Chunks", type="primary"):
        with st.spinner("Searching local FAISS embedding index..."):
            evidence = retrieve_relevant_chunks(current_ws["id"], query=query, top_k=top_k)
            if not evidence:
                st.warning("No matching chunks found. Ensure you have uploaded and ingested documents above.")
            else:
                st.success(f"Retrieved {len(evidence)} evidence chunks matching query.")
                for i, chunk in enumerate(evidence, 1):
                    with st.container(border=True):
                        st.markdown(f"**Chunk #{i}** | 📄 Source: `{chunk.source_filename}` | 📖 Page: **{chunk.page_number}** | 🎯 Similarity: `{chunk.similarity_score:.4f}`")
                        st.markdown(f"> {chunk.content}")

with tab_docs:
    st.subheader("Documents Ingested in this Workspace")
    docs = list_documents(current_ws["id"])
    if not docs:
        st.info("No documents uploaded yet for this workspace.")
    else:
        for doc in docs:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.markdown(f"**{doc['filename']}**")
                    st.caption(f"Type: `{doc['file_type'].upper()}` | Uploaded: `{doc['uploaded_at'][:19]}`")
                with c2:
                    st.markdown(f"Path: `{Path(doc['file_path']).name}`")
                with c3:
                    if st.button("Delete Doc", key=f"deldoc_{doc['id']}", use_container_width=True):
                        delete_document(doc["id"])
                        st.toast(f"Removed {doc['filename']}")
                        st.rerun()

        if st.button("⚠️ Clear Workspace Vector Index", help="Deletes all indexed vectors for this workspace"):
            clear_workspace_index(current_ws["id"])
            st.warning("Cleared workspace vector index.")
            st.rerun()

st.markdown("---")
st.markdown("#### Next Step")
if st.button("Proceed to Experiment Planner ➡️", type="primary"):
    st.switch_page("pages/3_Experiment_Planner.py")
