"""
Embedding service — sentence-transformers (local) with OpenRouter API fallback.
"""
from __future__ import annotations
import logging
import time
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import httpx

from .config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE,
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE,
)

log = logging.getLogger("cluster_engine.embeddings")

# ── Cache directory ────────────────────────────────────────────────────

CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", "/tmp/cluster_engine_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Local model (sentence-transformers) ───────────────────────────────

LOCAL_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading local embedding model: {LOCAL_MODEL_NAME}")
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
        log.info(f"Local model loaded, dim={_local_model.get_sentence_embedding_dimension()}")
    return _local_model


def _embed_batch_local(texts: list[str]) -> list[np.ndarray]:
    model = _get_local_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [emb.astype(np.float32) for emb in embeddings]


# ── API calls (fallback) ─────────────────────────────────────────────

def _embed_batch_api(texts: list[str], retries: int = 3) -> list[np.ndarray | None]:
    """Embed a batch via OpenRouter API."""
    if not texts:
        return []
    if not OPENROUTER_API_KEY:
        return [None] * len(texts)

    for attempt in range(retries):
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(
                    f"{OPENROUTER_BASE}/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": texts,
                    },
                )

            if r.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code != 200:
                log.warning(f"Embedding API error {r.status_code}: {r.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return [None] * len(texts)

            data = r.json()
            if "error" in data:
                log.warning(f"Embedding API error: {data['error']}")
                return [None] * len(texts)

            results: list[np.ndarray | None] = [None] * len(texts)
            for item in data["data"]:
                idx = item["index"]
                results[idx] = np.array(item["embedding"], dtype=np.float32)
            return results

        except Exception as e:
            log.warning(f"Embedding API exception (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return [None] * len(texts)


# ── Public API ─────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    show_progress: bool = True,
) -> list[np.ndarray | None]:
    """
    Embed texts using local sentence-transformers model.
    Falls back to OpenRouter API if local model fails.
    """
    if not texts:
        return []

    total = len(texts)
    batch_size = 256

    # Try local model first
    try:
        _get_local_model()
        log.info(f"Using local embedding model: {LOCAL_MODEL_NAME}")
        all_embeddings: list[np.ndarray | None] = [None] * total

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            results = _embed_batch_local(batch)
            for j, emb in enumerate(results):
                all_embeddings[i + j] = emb
            if show_progress and (i + batch_size) % 500 < batch_size:
                log.info(f"  Embedded {min(i + batch_size, total)}/{total} texts")

        success = sum(1 for e in all_embeddings if e is not None)
        log.info(f"Embedded {success}/{total} texts successfully (local)")
        return all_embeddings

    except Exception as e:
        log.warning(f"Local model failed: {e}, falling back to API")

    # Fallback to API
    all_embeddings = [None] * total
    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        results = _embed_batch_api(batch)
        for j, emb in enumerate(results):
            all_embeddings[i + j] = emb
        if show_progress and (i + EMBEDDING_BATCH_SIZE) % 500 < EMBEDDING_BATCH_SIZE:
            log.info(f"  Embedded {min(i + EMBEDDING_BATCH_SIZE, total)}/{total} texts")
        if i + EMBEDDING_BATCH_SIZE < total:
            time.sleep(0.1)

    success = sum(1 for e in all_embeddings if e is not None)
    log.info(f"Embedded {success}/{total} texts successfully")
    return all_embeddings


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_embedding_matrix(
    products_with_embeddings: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, list[int]]:
    """
    Build a dense matrix from (pk_id, embedding) pairs.
    Returns (matrix [N x dim], pk_id_list).
    """
    if not products_with_embeddings:
        return np.empty((0, 0)), []

    pk_ids = [pk for pk, _ in products_with_embeddings]
    dim = products_with_embeddings[0][1].shape[0]
    matrix = np.zeros((len(pk_ids), dim), dtype=np.float32)
    for i, (_, emb) in enumerate(products_with_embeddings):
        matrix[i] = emb

    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    return matrix, pk_ids
