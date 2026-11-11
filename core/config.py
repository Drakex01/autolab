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

# LLM Provider Configuration (Ollama is the default local provider)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower() or "ollama"

# Local Ollama Settings (conservative defaults for 16 GB RAM / RTX 3050 4 GB VRAM)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
OLLAMA_SUPPORTED_MODELS = ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct"]
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "360"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m").strip()

# Optional Gemini API Configuration (only used when provider is explicitly set to 'gemini')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
# system_instruction and response_mime_type require v1beta in google-genai >= 3.0
GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta").strip() or "v1beta"

# Document RAG Configuration
CHUNK_SIZE = int(os.getenv("AUTOLAB_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("AUTOLAB_CHUNK_OVERLAP", "120"))
EMBEDDING_MODEL_NAME = os.getenv("AUTOLAB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_UPLOAD_SIZE_MB = int(os.getenv("AUTOLAB_MAX_UPLOAD_SIZE_MB", "20"))
TOP_K_RETRIEVAL = int(os.getenv("AUTOLAB_TOP_K_RETRIEVAL", "5"))

# Reproducibility
DEFAULT_RANDOM_SEED = 42
