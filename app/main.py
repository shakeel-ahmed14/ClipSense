from __future__ import annotations

import logging
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.rag_service import query_rag

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
