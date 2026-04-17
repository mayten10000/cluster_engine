#!/usr/bin/env python3
"""
Calibrate per-niche / per-category SCORE_OK thresholds.

Without a labelled eval set, we can't directly optimize precision/recall.
This script uses cluster-internal coherence as a proxy:

  • For each niche, compute mean intra-cluster cosine similarity of items in
    clusters with size >= MIN_CLUSTER_SIZE.
  • High coherence (>= 0.85): clusters in this niche are semantically tight,
    we can RELAX SCORE_OK toward 0.70 to avoid sending easy items to LLM.
  • Medium (0.75 .. 0.85): keep default 0.75.
  • Low (< 0.75): clusters are noisy, TIGHTEN to 0.80 so more borderline
    items go to LLM and get sanity-checked.

Per-category threshold = mean of its niches' thresholds.

Usage:
  cd /opt && python3 -m cluster_engine_v2.calibrate_thresholds \
    --categories 37 42 44 60 \
    --output /var/cache/cluster_engine/score_thresholds.json
"""
from __future__ import annotations
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import text

from .db import get_engine
from .embeddings import embed_texts

log = logging.getLogger("calibrate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS_PER_NICHE = 30  # cap embedding work
DEFAULT_SCORE_OK = 0.75
TIGHT_SCORE_OK = 0.80
LOOSE_SCORE_OK = 0.70


def load_clusters(category_ids: list[int]) -> dict:
    """Return: {niche_key: {category_id, clusters: {gid: [names]}}}"""
    eng = get_engine()
    placeholders = ",".join(str(int(c)) for c in category_ids)
    sql = f"""
        SELECT
            CAST(p.niche_key AS SIGNED) AS niche_key,
            n.category_id,
            pc.cluster_gid,
            p.name
        FROM mpstats_products p
        JOIN mpstats_niches n ON n.niche_key = CAST(p.niche_key AS SIGNED)
        JOIN mpstats_product_clusters pc ON pc.pk_id = p.pk_id
        WHERE n.category_id IN ({placeholders})
          AND p.name IS NOT NULL AND p.name != ''
          AND pc.cluster_gid IS NOT NULL
    """
    by_niche: dict = defaultdict(lambda: {"category_id": None, "clusters": defaultdict(list)})
    with eng.connect() as conn:
        for row in conn.execute(text(sql)).mappings():
            nk = int(row["niche_key"])
            by_niche[nk]["category_id"] = int(row["category_id"])
            by_niche[nk]["clusters"][int(row["cluster_gid"])].append(row["name"])
    return by_niche


def coherence(names: list[str]) -> float:
    """Mean pairwise cosine similarity of name embeddings."""
    if len(names) < 2:
        return 0.0
    embs = embed_texts(names, show_progress=False)
    arr = np.array([e for e in embs if e is not None], dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    sims = arr @ arr.T
    n = len(arr)
    # Mean of upper triangle (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    return float(sims[iu].mean())


def threshold_for_coherence(c: float) -> float:
    if c >= 0.85:
        return LOOSE_SCORE_OK
    if c >= 0.75:
        return DEFAULT_SCORE_OK
    return TIGHT_SCORE_OK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    log.info(f"Loading clusters for categories {args.categories}...")
    by_niche = load_clusters(args.categories)
    log.info(f"Loaded {len(by_niche)} niches")

    per_niche: dict[int, float] = {}
    per_category_acc: dict[int, list[float]] = defaultdict(list)

    for nk, info in by_niche.items():
        cat = info["category_id"]
        clusters = info["clusters"]
        # Take up to MAX_CLUSTERS_PER_NICHE clusters of size >= MIN
        big = [(gid, names) for gid, names in clusters.items() if len(names) >= MIN_CLUSTER_SIZE]
        big.sort(key=lambda x: -len(x[1]))
        big = big[:MAX_CLUSTERS_PER_NICHE]
        if not big:
            continue
        cohs = [coherence(names[:20]) for _, names in big]
        cohs = [c for c in cohs if c > 0]
        if not cohs:
            continue
        mean_coh = float(np.mean(cohs))
        thr = threshold_for_coherence(mean_coh)
        per_niche[nk] = thr
        per_category_acc[cat].append(thr)
        log.info(
            f"  niche {nk} (cat {cat}): {len(big)} clusters, coh={mean_coh:.3f} → SCORE_OK={thr}"
        )

    per_category: dict[int, float] = {}
    for cat, vals in per_category_acc.items():
        per_category[cat] = round(float(np.mean(vals)), 3)

    out = {
        "per_niche": {str(k): v for k, v in per_niche.items()},
        "per_category": {str(k): v for k, v in per_category.items()},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"=== DONE ===")
    log.info(f"Written {len(per_niche)} per-niche + {len(per_category)} per-category thresholds")
    log.info(f"Output: {out_path}")
    for cat, v in sorted(per_category.items()):
        log.info(f"  category {cat}: SCORE_OK = {v}")


if __name__ == "__main__":
    main()
