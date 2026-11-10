from typing import List, Dict, Any
from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.schemas import DocumentChunk

def split_text_into_chunks(
    pages_data: List[Dict[str, Any]],
    source_filename: str,
    document_id: int,
    workspace_id: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[DocumentChunk]:
    """
    Splits extracted pages into overlapping chunks while preserving metadata.
    """
    chunks: List[DocumentChunk] = []
    chunk_counter = 0

    step = max(1, chunk_size - chunk_overlap)

    for page in pages_data:
        text = page["text"]
        page_num = page["page_number"]

        if not text:
            continue

        # If text is shorter than chunk size, create single chunk
        if len(text) <= chunk_size:
            chunk_id = f"ws{workspace_id}_doc{document_id}_p{page_num}_c{chunk_counter}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                workspace_id=workspace_id,
                source_filename=source_filename,
                page_number=page_num,
                chunk_index=chunk_counter,
                content=text
            ))
            chunk_counter += 1
            continue

        # Slide over text
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"ws{workspace_id}_doc{document_id}_p{page_num}_c{chunk_counter}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    source_filename=source_filename,
                    page_number=page_num,
                    chunk_index=chunk_counter,
                    content=chunk_text
                ))
                chunk_counter += 1

            if end >= len(text):
                break
            start += step

    return chunks
