#!/usr/bin/env python3
"""
Mining training pairs for contrastive fine-tuning of sentence embeddings.

Outputs CSV: (pk_left, pk_right, label, type)
  label = 1 (positive: same cluster) | 0 (negative)
  type  = "positive" | "hard_neg" | "easy_neg"

Strategy:
  - Positive: same cluster_gid in the latest run, cluster size 3-50.
    Up to N_POS_PER_CLUSTER pairs per cluster.
  - Hard negative: same niche, different cluster, embedding cosine in
    [HARD_NEG_MIN_SIM, HARD_NEG_MAX_SIM]. These teach the model the
    fine-grained boundary between "same product" and "similar product".
  - Easy negative: different niche entirely. Random sampling.

Usage:
  python3 -m cluster_engine_v2.mine_pairs \
    --categories 37 42 44 60 \
    --output /var/cache/cluster_engine/training/pairs_v1.csv \
    --max-pairs 10000

Run from /opt for proper imports.
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import text

from cluster_engine_v2.db import get_engine
from cluster_engine_v2.embeddings import embed_texts

log = logging.getLogger("mine_pairs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Tunables ─────────────────────────────────────────────────────────
N_POS_PER_CLUSTER = 5         # max positive pairs per cluster
HARD_NEG_MIN_SIM = 0.55       # below this — too easy
HARD_NEG_MAX_SIM = 0.90       # above this — likely a real positive (noise)
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 50
EASY_NEG_RATIO = 0.5          # share of negatives that are "easy" (different niche)


def load_products(category_ids: list[int]) -> list[dict]:
    """Load all products with their cluster assignment for given categories."""
    eng = get_engine()
    placeholders = ",".join(str(int(c)) for c in category_ids)
    sql = f"""
        SELECT
            p.pk_id, p.name, p.brand,
            CAST(p.niche_key AS SIGNED) AS niche_key,
            n.category_id,
            pc.cluster_gid
        FROM mpstats_products p
        JOIN mpstats_niches n ON n.niche_key = CAST(p.niche_key AS SIGNED)
        JOIN mpstats_product_clusters pc ON pc.pk_id = p.pk_id
        WHERE n.category_id IN ({placeholders})
          AND p.name IS NOT NULL AND p.name != ''
    """
    with eng.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]


def compute_embeddings(products: list[dict]) -> np.ndarray:
    """Compute (or load cached) embeddings for product names."""
    texts_only = [p["name"] for p in products]
    log.info(f"Embedding {len(texts_only)} product names...")
    embs = embed_texts(texts_only, show_progress=True)
    # Replace None with zero vector (failed embeds)
    dim = next((e.shape[0] for e in embs if e is not None), 384)
    arr = np.zeros((len(embs), dim), dtype=np.float32)
    for i, e in enumerate(embs):
        if e is not None:
            arr[i] = e
    # L2-normalize for cosine
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def mine_positives(
    products: list[dict],
    embs: np.ndarray,
    pk_to_idx: dict[int, int],
) -> list[tuple]:
    """Sample positive pairs from same cluster."""
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(products):
        by_cluster[p["cluster_gid"]].append(i)

    positives = []
    for gid, idxs in by_cluster.items():
        sz = len(idxs)
        if sz < MIN_CLUSTER_SIZE or sz > MAX_CLUSTER_SIZE:
            continue
        # Sample pairs (no replacement)
        random.shuffle(idxs)
        # Greedy: take consecutive pairs from shuffled list
        n_pairs = min(N_POS_PER_CLUSTER, sz // 2)
        for k in range(n_pairs):
            i, j = idxs[2 * k], idxs[2 * k + 1]
            sim = cosine(embs[i], embs[j])
            positives.append((products[i]["pk_id"], products[j]["pk_id"], 1, "positive", sim))

    log.info(f"Mined {len(positives)} positive pairs from "
             f"{sum(1 for v in by_cluster.values() if MIN_CLUSTER_SIZE<=len(v)<=MAX_CLUSTER_SIZE)} eligible clusters")
    return positives


def mine_hard_negatives(
    products: list[dict],
    embs: np.ndarray,
    target_count: int,
) -> list[tuple]:
    """Find pairs from different clusters in same niche with mid-range cosine."""
    by_niche: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(products):
        by_niche[p["niche_key"]].append(i)

    hard_negs = []
    seen = set()
    niches = [(nk, idxs) for nk, idxs in by_niche.items() if len(idxs) >= 4]
    random.shuffle(niches)

    for nk, idxs in niches:
        if len(hard_negs) >= target_count:
            break
        # Sample pairs from this niche
        attempts = 0
        max_attempts = len(idxs) * 5
        while attempts < max_attempts and len(hard_negs) < target_count:
            attempts += 1
            i, j = random.sample(idxs, 2)
            if products[i]["cluster_gid"] == products[j]["cluster_gid"]:
                continue  # same cluster — skip
            pair = tuple(sorted((products[i]["pk_id"], products[j]["pk_id"])))
            if pair in seen:
                continue
            sim = cosine(embs[i], embs[j])
            if HARD_NEG_MIN_SIM <= sim <= HARD_NEG_MAX_SIM:
                seen.add(pair)
                hard_negs.append((pair[0], pair[1], 0, "hard_neg", sim))

    log.info(f"Mined {len(hard_negs)} hard negative pairs")
    return hard_negs


def mine_easy_negatives(
    products: list[dict],
    embs: np.ndarray,
    target_count: int,
) -> list[tuple]:
    """Random pairs from different categories."""
    by_cat: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(products):
        by_cat[p["category_id"]].append(i)

    cats = list(by_cat.keys())
    if len(cats) < 2:
        log.warning("Need ≥2 categories for easy negatives")
        return []

    easy_negs = []
    seen = set()
    attempts = 0
    max_attempts = target_count * 10

    while len(easy_negs) < target_count and attempts < max_attempts:
        attempts += 1
        c1, c2 = random.sample(cats, 2)
        i = random.choice(by_cat[c1])
        j = random.choice(by_cat[c2])
        pair = tuple(sorted((products[i]["pk_id"], products[j]["pk_id"])))
        if pair in seen:
            continue
        sim = cosine(embs[i], embs[j])
        seen.add(pair)
        easy_negs.append((pair[0], pair[1], 0, "easy_neg", sim))

    log.info(f"Mined {len(easy_negs)} easy negative pairs")
    return easy_negs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-pairs", type=int, default=10000,
                    help="Total pairs (positives + negatives)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    log.info(f"Loading products for categories {args.categories}...")
    products = load_products(args.categories)
    log.info(f"Loaded {len(products)} products")

    # Build pk_id → index map
    pk_to_idx = {p["pk_id"]: i for i, p in enumerate(products)}

    # Compute embeddings
    embs = compute_embeddings(products)
    log.info(f"Embeddings shape: {embs.shape}")

    # Mine pairs
    positives = mine_positives(products, embs, pk_to_idx)

    # Negative budget = same as positives (1:1)
    n_neg_total = min(len(positives), args.max_pairs - len(positives))
    n_easy = int(n_neg_total * EASY_NEG_RATIO)
    n_hard = n_neg_total - n_easy

    hard_negs = mine_hard_negatives(products, embs, n_hard)
    easy_negs = mine_easy_negatives(products, embs, n_easy)

    all_pairs = positives + hard_negs + easy_negs
    random.shuffle(all_pairs)

    # Trim to max-pairs
    if len(all_pairs) > args.max_pairs:
        all_pairs = all_pairs[: args.max_pairs]

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pk_left", "pk_right", "label", "type", "cosine"])
        for row in all_pairs:
            w.writerow(row)

    # Stats
    n_pos = sum(1 for r in all_pairs if r[2] == 1)
    n_hard_actual = sum(1 for r in all_pairs if r[3] == "hard_neg")
    n_easy_actual = sum(1 for r in all_pairs if r[3] == "easy_neg")
    log.info(f"=== DONE ===")
    log.info(f"Output: {out_path}")
    log.info(f"Total pairs: {len(all_pairs)}")
    log.info(f"  positives:    {n_pos}")
    log.info(f"  hard negs:    {n_hard_actual}")
    log.info(f"  easy negs:    {n_easy_actual}")
    if n_pos:
        avg_pos_sim = sum(r[4] for r in all_pairs if r[2] == 1) / n_pos
        log.info(f"  avg positive cosine: {avg_pos_sim:.3f}")
    if n_hard_actual:
        avg_hard_sim = sum(r[4] for r in all_pairs if r[3] == "hard_neg") / n_hard_actual
        log.info(f"  avg hard-neg cosine: {avg_hard_sim:.3f}")


if __name__ == "__main__":
    main()
