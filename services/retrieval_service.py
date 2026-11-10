import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from core.config import VECTOR_STORE_DIR, TOP_K_RETRIEVAL
from core.schemas import DocumentChunk, RetrievedEvidence
from services.embedding_service import generate_embeddings

def _get_paths(workspace_id: int):
    index_path = VECTOR_STORE_DIR / f"ws_{workspace_id}.faiss"
    meta_path = VECTOR_STORE_DIR / f"ws_{workspace_id}_chunks.json"
    return index_path, meta_path

def add_chunks_to_index(workspace_id: int, new_chunks: List[DocumentChunk]):
    """Adds document chunks into the workspace vector store."""
    if not new_chunks:
        return

    index_path, meta_path = _get_paths(workspace_id)
    existing_chunks = []

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        except Exception:
            existing_chunks = []

    all_chunks = existing_chunks + [chunk.model_dump() for chunk in new_chunks]
    texts = [c["content"] for c in all_chunks]

    # Generate embeddings for all chunks in the workspace
    embeddings = generate_embeddings(texts)

    # Use FAISS
    try:
        import faiss
        dim = embeddings.shape[1]
        # Inner product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
    except Exception as e:
        # Fallback will compute via dot product or NearestNeighbors dynamically
        np.save(str(index_path.with_suffix(".npy")), embeddings)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

def retrieve_relevant_chunks(workspace_id: int, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[RetrievedEvidence]:
    """Retrieves top_k most similar chunks for a query from the workspace index."""
    if not query.strip():
        return []

    index_path, meta_path = _get_paths(workspace_id)
    if not meta_path.exists():
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        chunks_meta = json.load(f)

    if not chunks_meta:
        return []

    # Generate query embedding
    query_emb = generate_embeddings([query])

    results: List[RetrievedEvidence] = []
    k = min(top_k, len(chunks_meta))

    # Try FAISS search first
    if index_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(index_path))
            scores, indices = index.search(query_emb, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(chunks_meta):
                    continue
                chunk = chunks_meta[idx]
                results.append(RetrievedEvidence(
                    source_filename=chunk["source_filename"],
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    similarity_score=round(float(score), 4)
                ))
            return results
        except Exception:
            pass

    # Robust Scikit-learn / Numpy Cosine Fallback
    try:
        texts = [c["content"] for c in chunks_meta]
        all_embeddings = generate_embeddings(texts)
        # Cosine similarity for normalized vectors is dot product
        similarities = np.dot(all_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]

        for idx in top_indices:
            chunk = chunks_meta[idx]
            results.append(RetrievedEvidence(
                source_filename=chunk["source_filename"],
                page_number=chunk["page_number"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                similarity_score=round(float(similarities[idx]), 4)
            ))
        return results
    except Exception:
        return []

def get_indexed_chunks(workspace_id: int) -> List[Dict[str, Any]]:
    """Returns all metadata chunks stored for this workspace."""
    _, meta_path = _get_paths(workspace_id)
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def clear_workspace_index(workspace_id: int):
    """Removes stored vector index and metadata for a workspace."""
    index_path, meta_path = _get_paths(workspace_id)
    if index_path.exists():
        index_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    npy_path = index_path.with_suffix(".npy")
    if npy_path.exists():
        npy_path.unlink()
