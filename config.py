"""Centralized configuration for DocuNex.

All runtime settings are read from environment variables (loaded from a
`.env` file when present) with sensible local-development defaults, so the
same code runs unchanged on a laptop, a GPU box, or CI.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _detect_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'.

    Imported lazily so that simply importing the config does not require
    torch to be installed (e.g. in lightweight test/CI environments).
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --- Neo4j (optional; only used for GraphRAG graph storage) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- Ollama (local LLM runtime) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")

# --- Storage locations for generated stores/graphs ---
BASE_PERSIST_DIR = os.getenv("BASE_PERSIST_DIR", "./graph_db")
BASE_VECTOR_DIR = os.getenv("BASE_VECTOR_DIR", "./vector_db")

# --- Compute device ---
DEVICE = _detect_device()
