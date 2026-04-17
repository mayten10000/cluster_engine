"""
Blocking — candidate pair generation.

Phase 0: Deterministic EAN matching.
Phase 1: Embedding-based approximate nearest neighbor search.
Phase 2: Brand + token overlap filtering.

Output: list of (pk_a, pk_b, weight) candidate edges for the graph.
"""
from __future__ import annotations
import logging
from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import (
    EAN_MIN_OCCURRENCES, EAN_MAX_PRODUCTS,
    BLOCKING_TOP_K, BLOCKING_MIN_SIM,
    EDGE_WEIGHT_EAN, EDGE_WEIGHT_BRAND_TOKENS,
    EDGE_WEIGHT_EMBEDDING, EDGE_WEIGHT_PRICE_PENALTY,
)
from .models import Product
from .text_processing import (
    token_overlap, brand_match, price_ratio, normalize_brand,
)
from .embeddings import cosine_sim

log = logging.getLogger("cluster_engine.blocking")

# Type alias for edges
Edge = tuple[int, int, float]  # (pk_a, pk_b, weight)


def phase0_ean_match(products: list[Product]) -> list[Edge]:
    """
    Deterministic EAN matching.
    Products with the same valid EAN are connected with weight 1.0.
    """
    ean_groups: dict[str, list[int]] = defaultdict(list)

    for p in products:
        ean = (p.ean or "").strip()
        if not ean or len(ean) < 8:
            continue
        # Skip garbage EANs
        if ean == "0" * len(ean):
            continue
        if len(set(ean)) <= 2:  # e.g., "1111111111111"
            continue
        ean_groups[ean].append(p.pk_id)

    edges = []
    for ean, pks in ean_groups.items():
        if len(pks) < EAN_MIN_OCCURRENCES:
            continue
        if len(pks) > EAN_MAX_PRODUCTS:
            log.debug(f"Skipping EAN {ean}: {len(pks)} products (>{EAN_MAX_PRODUCTS})")
            continue
        # Connect all pairs in the group
        for i in range(len(pks)):
            for j in range(i + 1, len(pks)):
                edges.append((pks[i], pks[j], EDGE_WEIGHT_EAN))

    log.info(f"Phase 0 (EAN): {len(ean_groups)} valid EAN groups → {len(edges)} edges")
    return edges


def phase1_embedding_blocking(
    products: list[Product],
    embedding_matrix: np.ndarray,
    pk_id_list: list[int],
) -> list[Edge]:
    """
    Embedding-based blocking using approximate nearest neighbors.
    For each product, find top-K nearest neighbors by cosine similarity.
    """
    n = embedding_matrix.shape[0]
    if n == 0:
        return []

    k = min(BLOCKING_TOP_K + 1, n)  # +1 because self is included

    log.info(f"Phase 1 (Embedding): Building kNN index for {n} products, k={k}...")

    # sklearn NearestNeighbors with cosine metric
    # matrix is already L2-normalized, so dot product = cosine sim
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(embedding_matrix)
    distances, indices = nn.kneighbors(embedding_matrix)

    # Build product lookup for fast access
    pk_to_product: dict[int, Product] = {p.pk_id: p for p in products}
    pk_set = set(pk_id_list)

    edges = []
    seen = set()

    for i in range(n):
        pk_a = pk_id_list[i]
        prod_a = pk_to_product.get(pk_a)
        if not prod_a:
            continue

        for j_idx in range(1, k):  # skip self at index 0
            pk_b = pk_id_list[indices[i][j_idx]]
            if pk_a == pk_b:
                continue

            # Canonical edge ordering to avoid duplicates
            edge_key = (min(pk_a, pk_b), max(pk_a, pk_b))
            if edge_key in seen:
                continue

            # cosine distance → cosine similarity
            cos_sim = 1.0 - float(distances[i][j_idx])
            if cos_sim < BLOCKING_MIN_SIM:
                continue

            prod_b = pk_to_product.get(pk_b)
            if not prod_b:
                continue

            # Compute edge weight
            weight = cos_sim * EDGE_WEIGHT_EMBEDDING

            # Brand boost/penalty
            bm = brand_match(prod_a.brand, prod_b.brand)
            if bm == 1.0:
                weight = min(weight + 0.10, 1.0)  # boost for same brand
            elif bm == 0.0 and prod_a.brand and prod_b.brand:
                weight *= 0.7  # penalty for different known brands

            # Token overlap boost
            tok_ovl = token_overlap(prod_a.tokens, prod_b.tokens)
            if tok_ovl > 0.5:
                weight = min(weight + tok_ovl * 0.1, 1.0)

            # Price divergence penalty
            pr = price_ratio(prod_a.price, prod_b.price)
            if pr > 3.0:
                weight -= EDGE_WEIGHT_PRICE_PENALTY
            elif pr > 5.0:
                weight -= EDGE_WEIGHT_PRICE_PENALTY * 2

            if weight < BLOCKING_MIN_SIM:
                continue

            seen.add(edge_key)
            edges.append((pk_a, pk_b, round(weight, 4)))

    log.info(f"Phase 1 (Embedding): {len(edges)} edges (min_sim={BLOCKING_MIN_SIM})")
    return edges


def phase2_brand_token_boost(
    products: list[Product],
    existing_edges: set[tuple[int, int]],
) -> list[Edge]:
    """
    Additional edges for products with same brand + high token overlap
    that may have been missed by embedding blocking.

    Only checks within same niche to limit comparisons.
    """
    by_brand_niche: dict[tuple[str, int], list[Product]] = defaultdict(list)

    for p in products:
        nb = normalize_brand(p.brand)
        if nb and p.niche_key:
            by_brand_niche[(nb, p.niche_key)].append(p)

    edges = []
    for (brand, niche), group in by_brand_niche.items():
        if len(group) > 200:
            continue  # skip very large brand-niche groups
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                edge_key = (min(a.pk_id, b.pk_id), max(a.pk_id, b.pk_id))
                if edge_key in existing_edges:
                    continue

                tok_ovl = token_overlap(a.tokens, b.tokens)
                if tok_ovl < 0.4:
                    continue

                weight = EDGE_WEIGHT_BRAND_TOKENS * tok_ovl
                if weight >= BLOCKING_MIN_SIM:
                    edges.append((a.pk_id, b.pk_id, round(weight, 4)))

    log.info(f"Phase 2 (Brand+Token): {len(edges)} additional edges")
    return edges


def generate_candidate_edges(
    products: list[Product],
    embedding_matrix: np.ndarray,
    pk_id_list: list[int],
) -> list[Edge]:
    """
    Full blocking pipeline: EAN → Embedding kNN → Brand+Token boost.
    Returns deduplicated list of weighted edges.
    """
    # Phase 0: EAN
    ean_edges = phase0_ean_match(products)

    # Phase 1: Embedding blocking
    emb_edges = phase1_embedding_blocking(products, embedding_matrix, pk_id_list)

    # Merge and deduplicate (keep max weight)
    edge_weights: dict[tuple[int, int], float] = {}
    for a, b, w in ean_edges + emb_edges:
        key = (min(a, b), max(a, b))
        edge_weights[key] = max(edge_weights.get(key, 0.0), w)

    # Phase 2: Brand+Token (only for pairs not already connected)
    existing = set(edge_weights.keys())
    brand_edges = phase2_brand_token_boost(products, existing)
    for a, b, w in brand_edges:
        key = (min(a, b), max(a, b))
        edge_weights[key] = max(edge_weights.get(key, 0.0), w)

    # Convert to edge list
    all_edges = [(a, b, w) for (a, b), w in edge_weights.items()]

    log.info(
        f"Total candidate edges: {len(all_edges)} "
        f"(EAN={len(ean_edges)}, Emb={len(emb_edges)}, Brand={len(brand_edges)})"
    )
    return all_edges
