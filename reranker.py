"""
Cross-encoder reranker for grey-zone resolution.

Takes (product_name, cluster_main_name) pairs and returns a relevance score
in [0, 1]. Used as a cheap local alternative to LLM arbitration for grey
products — high confidence auto-confirms or auto-quarantines, ambiguous
pairs fall through to the LLM.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

log = logging.getLogger("cluster_engine.reranker")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "256"))

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        log.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        _model = CrossEncoder(RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH)
        log.info("Cross-encoder loaded")
    return _model


def score_pairs(pairs: Iterable[tuple[str, str]]) -> list[float]:
    """Score a list of (text_a, text_b) pairs. Returns sigmoid-normalised
    relevance in [0, 1]. Empty list returns empty list.
    """
    pairs = [(a or "", b or "") for a, b in pairs]
    if not pairs:
        return []
    model = _get_model()
    scores = model.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
        activation_fn=None,  # raw logits; apply sigmoid below
    )
    import numpy as np
    arr = np.asarray(scores, dtype="float32")
    probs = 1.0 / (1.0 + np.exp(-arr))
    return [float(x) for x in probs]
