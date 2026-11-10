import sys
import tempfile
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.config
temp_dir = tempfile.mkdtemp()
core.config.VECTOR_STORE_DIR = Path(temp_dir)

from core.schemas import DocumentChunk
from services.retrieval_service import (
    add_chunks_to_index,
    retrieve_relevant_chunks,
    clear_workspace_index
)

def test_retrieval_service():
    ws_id = 999
    clear_workspace_index(ws_id)

    chunks = [
        DocumentChunk(
            chunk_id="c1", document_id=1, workspace_id=ws_id,
            source_filename="iris_paper.pdf", page_number=1, chunk_index=0,
            content="The Iris flower dataset comprises 50 samples from each of three species: Iris setosa, Iris virginica, and Iris versicolor."
        ),
        DocumentChunk(
            chunk_id="c2", document_id=1, workspace_id=ws_id,
            source_filename="iris_paper.pdf", page_number=2, chunk_index=1,
            content="Random Forest classifiers construct an ensemble of decision trees to mitigate overfitting and capture non-linear relationships."
        ),
        DocumentChunk(
            chunk_id="c3", document_id=2, workspace_id=ws_id,
            source_filename="solar_energy.txt", page_number=1, chunk_index=0,
            content="Photovoltaic panels convert sunlight into electrical power using semiconductor materials."
        )
    ]

    add_chunks_to_index(ws_id, chunks)

    # Query for machine learning tree ensembles
    results = retrieve_relevant_chunks(ws_id, query="How do Random Forest decision trees work?", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "Random Forest" in top_hit.content
    assert top_hit.source_filename == "iris_paper.pdf"
    assert top_hit.page_number == 2
    assert top_hit.similarity_score > 0.3

    # Clean up
    clear_workspace_index(ws_id)
    assert len(retrieve_relevant_chunks(ws_id, "test")) == 0
