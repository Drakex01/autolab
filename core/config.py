import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
DB_PATH = DATA_DIR / "autolab.db"

# Ensure runtime directories exist
for directory in (DATA_DIR, UPLOADS_DIR, VECTOR_STORE_DIR, SAMPLE_DOCS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Document RAG Configuration
CHUNK_SIZE = int(os.getenv("AUTOLAB_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("AUTOLAB_CHUNK_OVERLAP", "120"))
EMBEDDING_MODEL_NAME = os.getenv("AUTOLAB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_UPLOAD_SIZE_MB = int(os.getenv("AUTOLAB_MAX_UPLOAD_SIZE_MB", "20"))
TOP_K_RETRIEVAL = int(os.getenv("AUTOLAB_TOP_K_RETRIEVAL", "5"))

# Reproducibility
DEFAULT_RANDOM_SEED = 42
