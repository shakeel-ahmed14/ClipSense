from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.rag_service import query_rag

BACKEND_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BACKEND_ROOT / "videos"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_df: pd.DataFrame | None = None


def get_dataframe() -> pd.DataFrame:
    if _df is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return _df


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _df
    path = settings.multimodal_joblib
    if not path.is_file():
        logger.error("Multimodal index missing at %s", path)
        _df = None
    else:
        _df = joblib.load(path)
        logger.info("Loaded %s rows from %s", len(_df), path)
    yield
    _df = None


app = FastAPI(
    title="ClipSense API",
    description="Multimodal RAG over video chunks: retrieve segments and optional LLM answer with timestamps.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question about the videos")
    top_k: int | None = Field(None, ge=1, le=50, description="Number of segments to retrieve")
    generate_answer: bool = Field(True, description="If false, only return ranked segments (no LLM)")


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_path: str
    rows: int | None = None


@app.get("/health", response_model=HealthResponse)
def health():
    path = str(settings.multimodal_joblib)
    if _df is None:
        return HealthResponse(status="degraded", index_loaded=False, index_path=path, rows=None)
    return HealthResponse(status="ok", index_loaded=True, index_path=path, rows=len(_df))


@app.post("/api/query")
def api_query(body: QueryBody):
    df = get_dataframe()
    top_k = body.top_k if body.top_k is not None else settings.default_top_k
    try:
        return query_rag(df, body.query, top_k, body.generate_answer)
    except requests.RequestException as e:
        logger.exception("Ollama request failed")
        raise HTTPException(
            status_code=502,
            detail=f"Upstream Ollama error: {e!s}. Ensure Ollama is running at {settings.ollama_base_url}.",
        ) from e


def _video_entries_from_disk() -> list[dict]:
    exts = {".mp4", ".webm", ".mkv", ".mov"}
    entries: list[dict] = []
    if not VIDEOS_DIR.is_dir():
        return entries
    for p in sorted(VIDEOS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            stem = p.stem
            entries.append(
                {
                    "file": p.name,
                    "ragTitle": stem,
                    "headline": stem.replace("_", " "),
                    "description": "",
                }
            )
    return entries


def _load_video_manifest_list() -> list[dict]:
    if not VIDEOS_DIR.is_dir():
        return []
    manifest_path = VIDEOS_DIR / "manifest.json"
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "videos" in raw:
            return list(raw["videos"])
    return _video_entries_from_disk()


@app.get("/api/videos")
def list_videos():
    """Videos served from backend/videos; optional manifest.json maps files to RAG titles."""
    entries = _load_video_manifest_list()
    out = []
    for e in entries:
        file = e.get("file")
        if not file:
            continue
        row = {**e, "src": f"/videos/{file}"}
        out.append(row)
    return {"videos": out}


if VIDEOS_DIR.is_dir():
    app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
else:
    logger.warning("Video directory missing at %s — create it and add manifest.json or media files.", VIDEOS_DIR)
