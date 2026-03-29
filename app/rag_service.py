from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    r = requests.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.embedding_model, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data["embeddings"]


def _ollama_generate(prompt: str) -> str:
    r = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "top_p": 1, "seed": 42},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]


def _segment_row_to_dict(row: pd.Series, score: float) -> dict[str, Any]:
    captions = row.get("visual_captions")
    if captions is None or (isinstance(captions, float) and math.isnan(captions)):
        captions_out: list[str] | None = []
    else:
        captions_out = [str(c) for c in list(captions)]

    def _num(v: Any) -> float | int:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0
        if isinstance(v, (np.integer, np.floating)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    return {
        "title": str(row.get("title", "")),
        "number": _num(row.get("number")),
        "start": float(_num(row.get("start"))),
        "end": float(_num(row.get("end"))),
        "text": str(row.get("text", "")),
        "visual_captions": captions_out,
        "score": float(score),
    }


def retrieve_top_k(df: pd.DataFrame, query: str, top_k: int) -> tuple[pd.DataFrame, np.ndarray]:
    question_embedding = _ollama_embed([query])[0]

    text_similarities = cosine_similarity(
        np.vstack(df["embedding"].values),
        [question_embedding],
    ).flatten()

    visual_similarities: list[float] = []
    for visual_emb_list in df["visual_embeddings"]:
        if visual_emb_list is None or len(visual_emb_list) == 0:
            visual_similarities.append(0.0)
        else:
            try:
                visual_emb_array = np.array(visual_emb_list)
                sim = cosine_similarity(visual_emb_array, [question_embedding]).max()
                visual_similarities.append(float(sim))
            except Exception:
                visual_similarities.append(0.0)

    visual_similarities_arr = np.array(visual_similarities)
    final_similarities = (
        settings.text_weight * text_similarities + settings.visual_weight * visual_similarities_arr
    )

    k = min(top_k, len(df))
    max_idx = final_similarities.argsort()[::-1][:k]
    ranked = df.iloc[max_idx].reset_index(drop=True)
    scores = final_similarities[max_idx]
    return ranked, scores


def build_prompt(incoming_query: str, segments_df: pd.DataFrame) -> str:
    payload = segments_df[
        ["title", "number", "start", "end", "text", "visual_captions"]
    ].to_json(orient="records", indent=2)
    return f"""You are an AI assistant helping users find events in cooking tutorial videos.

Below are relevant video segments containing:

- video title
- timestamps
- transcript
- visual descriptions of frames

VIDEO SEGMENTS:
{payload}

--------------------------------

USER QUESTION:
"{incoming_query}"

INSTRUCTIONS:
- Identify the most relevant video segment
- Use transcript AND visual descriptions
- Mention exact timestamp range
- Mention video title
- Explain clearly and naturally
- If not found, say "This event was not found in the video"
- ONLY use the provided video data
- DO NOT guess or assume anything
- If exact event not present, say "This event was not found in the video"
- DO NOT invent timestamps
"""


def query_rag(df: pd.DataFrame, user_query: str, top_k: int, generate_answer: bool) -> dict[str, Any]:
    ranked, scores = retrieve_top_k(df, user_query.strip(), top_k)
    segments = [
        _segment_row_to_dict(ranked.iloc[i], float(scores[i])) for i in range(len(ranked))
    ]
    out: dict[str, Any] = {"query": user_query, "segments": segments}
    if generate_answer:
        prompt = build_prompt(user_query, ranked)
        out["answer"] = _ollama_generate(prompt)
    return out
