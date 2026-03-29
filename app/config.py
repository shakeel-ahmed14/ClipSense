import os
from pathlib import Path

# Project root (parent of app/)
ROOT = Path(__file__).resolve().parent.parent


class Settings:
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    embedding_model: str = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
    llm_model: str = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")
    multimodal_joblib: Path = Path(
        os.environ.get("MULTIMODAL_JOBLIB", str(ROOT / "multimodal_embeddings.joblib"))
    )
    text_weight: float = float(os.environ.get("RAG_TEXT_WEIGHT", "0.7"))
    visual_weight: float = float(os.environ.get("RAG_VISUAL_WEIGHT", "0.3"))
    default_top_k: int = int(os.environ.get("RAG_TOP_K", "5"))


settings = Settings()
