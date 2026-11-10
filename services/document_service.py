from pathlib import Path
from typing import List, Dict, Any
import pymupdf

def extract_text_from_file(file_path: Path, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from PDF, TXT, or Markdown.
    Returns list of dicts: [{"page_number": int, "text": str}]
    """
    ext = file_path.suffix.lower()
    pages_data = []

    if ext == ".pdf":
        doc = pymupdf.open(str(file_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text
                })
        doc.close()

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
            if text:
                # Text files treated as single page (page 1)
                pages_data.append({
                    "page_number": 1,
                    "text": text
                })
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Only PDF, TXT, and MD are allowed.")

    return pages_data
