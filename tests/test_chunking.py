import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.chunking_service import split_text_into_chunks

def test_chunking_short_text():
    pages = [{"page_number": 1, "text": "Short scientific paper text."}]
    chunks = split_text_into_chunks(pages, "paper.txt", document_id=1, workspace_id=1, chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].source_filename == "paper.txt"
    assert chunks[0].page_number == 1
    assert chunks[0].content == "Short scientific paper text."

def test_chunking_long_text_with_overlap():
    long_text = "Word " * 500  # 2500 characters
    pages = [
        {"page_number": 1, "text": long_text[:1200]},
        {"page_number": 2, "text": long_text[1200:]}
    ]
    chunks = split_text_into_chunks(pages, "study.pdf", document_id=2, workspace_id=1, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 3
    # Check preservation of metadata
    p1_chunks = [c for c in chunks if c.page_number == 1]
    p2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(p1_chunks) >= 2
    assert len(p2_chunks) >= 2
    for c in chunks:
        assert c.source_filename == "study.pdf"
        assert len(c.content) <= 500
