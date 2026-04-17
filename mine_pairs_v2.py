#!/usr/bin/env python3
"""
Mining v2 — head-noun-aware hard negatives.

Improvements over mine_pairs.py:
  • Hard negatives now require DIFFERENT head-noun in the SAME niche.
    Example: "Шланг ..." vs "Набор ..." in niche 5644 (Шланги для компрессоров).
    These are exactly the cases the embedding model gets wrong because the
    niche-context word ("компрессор") dominates the embedding.
  • Cluster-confusion negatives: pairs from clusters where the LLM rejected
    a move (qc.cluster_moves with quarantine reason head-noun mismatch / P1).
  • Plus the regular positives (same cluster) and easy negatives (different
    category) for balance.

Output CSV: pk_left, pk_right, label, type, cosine
  type ∈ {positive, hard_neg_head, hard_neg_p1, easy_neg}

Usage:
  cd /opt && python3 -m cluster_engine_v2.mine_pairs_v2 \
    --categories 37 42 44 60 \
    --output /var/cache/cluster_engine/training/pairs_v2.csv \
    --max-pairs 15000
"""
from __future__ import annotations
import argparse
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import text

from .db import get_engine
from .embeddings import embed_texts
from .text_processing import normalize_text, lemmatize

log = logging.getLogger("mine_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_POS_PER_CLUSTER = 5
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 50
HARD_NEG_MIN_SIM = 0.55  # below — too easy
HARD_NEG_MAX_SIM = 0.92  # above — likely a true positive
EASY_NEG_RATIO = 0.30


def head_noun(name: str) -> str | None:
    """Same algorithm as llm_arbiter._head_noun — first non-numeric lemma ≥3 chars."""
    if not name:
        return None
    for w in normalize_text(name).split():
        if len(w) < 2 or w.isdigit():
            continue
        if any(c.isdigit() for c in w):
            continue
        lem = lemmatize(w)
        if len(lem) < 3:
            continue
        return lem
    return None


def load_products(category_ids: list[int]) -> list[dict]:
    eng = get_engine()
    placeholders = ",".join(str(int(c)) for c in category_ids)
    sql = f"""
        SELECT
            p.pk_id, p.name,
            CAST(p.niche_key AS SIGNED) AS niche_key,
            n.category_id,
            pc.cluster_gid
        FROM mpstats_products p
        JOIN mpstats_niches n ON n.niche_key = CAST(p.niche_key AS SIGNED)
        JOIN mpstats_product_clusters pc ON pc.pk_id = p.pk_id
        WHERE n.category_id IN ({placeholders})
          AND p.name IS NOT NULL AND p.name != ''
          AND pc.cluster_gid IS NOT NULL
    """
    with eng.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql)).mappings()]


def compute_embeddings(products: list[dict]) -> np.ndarray:
    names = [p["name"] for p in products]
    log.info(f"Embedding {len(names)} names...")
    embs = embed_texts(names, show_progress=True)
    dim = next((e.shape[0] for e in embs if e is not None), 384)
    arr = np.zeros((len(embs), dim), dtype=np.float32)
    for i, e in enumerate(embs):
        if e is not None:
            arr[i] = e
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def mine_positives(products, embs):
    by_cluster = defaultdict(list)
    for i, p in enumerate(products):
        by_cluster[p["cluster_gid"]].append(i)
    out = []
    for gid, idxs in by_cluster.items():
        sz = len(idxs)
        if sz < MIN_CLUSTER_SIZE or sz > MAX_CLUSTER_SIZE:
            continue
        random.shuffle(idxs)
        n_pairs = min(N_POS_PER_CLUSTER, sz // 2)
        for k in range(n_pairs):
            i, j = idxs[2 * k], idxs[2 * k + 1]
            sim = cosine(embs[i], embs[j])
            out.append((products[i]["pk_id"], products[j]["pk_id"], 1, "positive", sim))
    log.info(f"positives: {len(out)}")
    return out


def mine_hard_neg_head(products, embs, target_count):
    """Hard negatives: SAME niche but DIFFERENT head-noun (the killer case)."""
    by_niche = defaultdict(list)
    for i, p in enumerate(products):
        by_niche[p["niche_key"]].append(i)

    out = []
    seen = set()
    niches = list(by_niche.items())
    random.shuffle(niches)

    for nk, idxs in niches:
        if len(out) >= target_count:
            break
        # Group by head_noun
        by_head = defaultdict(list)
        for i in idxs:
            h = head_noun(products[i]["name"])
            if h:
                by_head[h].append(i)
        if len(by_head) < 2:
            continue
        heads = list(by_head.keys())
        attempts = 0
        max_attempts = max(50, len(idxs))
        while attempts < max_attempts and len(out) < target_count:
            attempts += 1
            h1, h2 = random.sample(heads, 2)
            i = random.choice(by_head[h1])
            j = random.choice(by_head[h2])
            pair = tuple(sorted((products[i]["pk_id"], products[j]["pk_id"])))
            if pair in seen:
                continue
            sim = cosine(embs[i], embs[j])
            if sim < HARD_NEG_MIN_SIM or sim > HARD_NEG_MAX_SIM:
                continue
            seen.add(pair)
            out.append((pair[0], pair[1], 0, "hard_neg_head", sim))
    log.info(f"hard_neg_head (same niche, different head-noun): {len(out)}")
    return out


def mine_hard_neg_p1(products, embs, target_count):
    """Hard negatives: same niche, same head-noun, different cluster
    (the classic 'similar but not same product' case)."""
    by_niche = defaultdict(list)
    for i, p in enumerate(products):
        by_niche[p["niche_key"]].append(i)
    out = []
    seen = set()
    niches = list(by_niche.items())
    random.shuffle(niches)
    for nk, idxs in niches:
        if len(out) >= target_count:
            break
        attempts = 0
        max_attempts = len(idxs) * 3
        while attempts < max_attempts and len(out) < target_count:
            attempts += 1
            if len(idxs) < 2:
                break
            i, j = random.sample(idxs, 2)
            if products[i]["cluster_gid"] == products[j]["cluster_gid"]:
                continue
            h1 = head_noun(products[i]["name"])
            h2 = head_noun(products[j]["name"])
            if h1 != h2:
                continue
            pair = tuple(sorted((products[i]["pk_id"], products[j]["pk_id"])))
            if pair in seen:
                continue
            sim = cosine(embs[i], embs[j])
            if HARD_NEG_MIN_SIM <= sim <= HARD_NEG_MAX_SIM:
                seen.add(pair)
                out.append((pair[0], pair[1], 0, "hard_neg_p1", sim))
    log.info(f"hard_neg_p1 (same head-noun, different cluster): {len(out)}")
    return out


def mine_easy_neg(products, embs, target_count):
    by_cat = defaultdict(list)
    for i, p in enumerate(products):
        by_cat[p["category_id"]].append(i)
    cats = list(by_cat.keys())
    if len(cats) < 2:
        return []
    out = []
    seen = set()
    attempts = 0
    while len(out) < target_count and attempts < target_count * 10:
        attempts += 1
        c1, c2 = random.sample(cats, 2)
        i = random.choice(by_cat[c1])
        j = random.choice(by_cat[c2])
        pair = tuple(sorted((products[i]["pk_id"], products[j]["pk_id"])))
        if pair in seen:
            continue
        sim = cosine(embs[i], embs[j])
        seen.add(pair)
        out.append((pair[0], pair[1], 0, "easy_neg", sim))
    log.info(f"easy_neg: {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-pairs", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    log.info(f"Loading products for categories {args.categories}...")
    products = load_products(args.categories)
    log.info(f"Loaded {len(products)}")

    embs = compute_embeddings(products)

    positives = mine_positives(products, embs)

    # Negative budget = 1:1 with positives, distributed:
    #   60% hard_neg_head (the new killer case)
    #   25% hard_neg_p1   (same head, different cluster)
    #   15% easy_neg      (different category — anchor)
    n_neg_total = min(len(positives), args.max_pairs - len(positives))
    n_head = int(n_neg_total * 0.60)
    n_p1 = int(n_neg_total * 0.25)
    n_easy = n_neg_total - n_head - n_p1

    hn_head = mine_hard_neg_head(products, embs, n_head)
    hn_p1 = mine_hard_neg_p1(products, embs, n_p1)
    en = mine_easy_neg(products, embs, n_easy)

    all_pairs = positives + hn_head + hn_p1 + en
    random.shuffle(all_pairs)
    if len(all_pairs) > args.max_pairs:
        all_pairs = all_pairs[: args.max_pairs]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pk_left", "pk_right", "label", "type", "cosine"])
        for row in all_pairs:
            w.writerow(row)

    log.info(f"=== DONE ===")
    log.info(f"Output: {out_path}, total {len(all_pairs)} pairs")
    for t in ("positive", "hard_neg_head", "hard_neg_p1", "easy_neg"):
        n = sum(1 for r in all_pairs if r[3] == t)
        log.info(f"  {t}: {n}")


if __name__ == "__main__":
    main()
