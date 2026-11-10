from typing import List
import numpy as np
from core.config import EMBEDDING_MODEL_NAME

_MODEL_INSTANCE = None

def get_embedding_model():
    """Lazily load the sentence transformer model once."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer
        _MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL_INSTANCE

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generates normalized L2 embeddings for a list of text strings."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)
