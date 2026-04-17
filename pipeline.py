"""
Main pipeline orchestrator.

Usage:
    python -m cluster_engine.pipeline --category_id=123
    python -m cluster_engine.pipeline --niche_key=456
    python -m cluster_engine.pipeline --category_id=123 --dry_run
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import time
import sys

import numpy as np

from .db import (
    load_products_by_category,
    load_products_by_niche,
    load_old_assignments,
    get_next_run_id,
    get_next_cluster_gid,
    write_assignments,
    get_engine,
)
from sqlalchemy import text
from .text_processing import tokenize, strip_numbers
from .embeddings import embed_texts, build_embedding_matrix
from .blocking import generate_candidate_edges
from .graph_cluster import (
    build_graph, run_leiden, membership_to_clusters,
    compute_cluster_centroids,
)
from .scoring import score_products
from .llm_arbiter import arbitrate_grey_zone
from .config import RERANKER_ENABLED, RERANKER_AUTO_OK, RERANKER_AUTO_QUARANTINE
from .diff import generate_moves
from .ocr import run_ocr
from .schema_attrs import extract_all as extract_schema_attrs

log = logging.getLogger("cluster_engine")


def _apply_reranker_to_grey(products, clusters) -> dict:
    """Run cross-encoder on grey products against their cluster's main product
    name. High relevance auto-confirms, low relevance auto-quarantines;
    middle band stays grey for the LLM.

    Mutates product.decision / decided_by / reason in place.
    Returns stats dict.
    """
    from .reranker import score_pairs

    pk_to_product = {p.pk_id: p for p in products}
    grey = [p for p in products if p.decision == "grey"]
    if not grey:
        return {"reranker_grey_in": 0}

    pairs: list[tuple[str, str]] = []
    candidates: list = []
    for p in grey:
        cl = clusters.get(p.new_cluster_gid)
        if not cl or not cl.main_pk:
            continue
        main = pk_to_product.get(cl.main_pk)
        if not main or not main.name or not p.name:
            continue
        pairs.append((p.name, main.name))
        candidates.append(p)

    if not pairs:
        return {"reranker_grey_in": len(grey), "reranker_scored": 0}

    log.info(f"  Reranker: scoring {len(pairs)} grey pairs...")
    try:
        scores = score_pairs(pairs)
    except Exception as e:
        log.warning(f"  Reranker failed: {e}")
        return {"reranker_grey_in": len(grey), "reranker_error": str(e)}

    auto_ok = 0
    auto_q = 0
    for p, s in zip(candidates, scores):
        tag = f"[RERANK={s:.3f}]"
        if s >= RERANKER_AUTO_OK:
            p.decision = "ok"
            p.decided_by = "reranker"
            p.reason = (p.reason + " | " if p.reason else "") + f"{tag} auto-ok"
            auto_ok += 1
        elif s <= RERANKER_AUTO_QUARANTINE:
            p.decision = "quarantine"
            p.decided_by = "reranker"
            p.reason = (p.reason + " | " if p.reason else "") + f"{tag} auto-quarantine"
            auto_q += 1
        else:
            p.reason = (p.reason + " | " if p.reason else "") + tag

    log.info(
        f"  Reranker: {auto_ok} auto-ok, {auto_q} auto-quarantine, "
        f"{len(pairs) - auto_ok - auto_q} still grey → LLM"
    )
    return {
        "reranker_grey_in": len(grey),
        "reranker_scored": len(pairs),
        "reranker_auto_ok": auto_ok,
        "reranker_auto_quarantine": auto_q,
        "reranker_to_llm": len(pairs) - auto_ok - auto_q,
    }


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def run_pipeline(
    category_id: int | None = None,
    niche_key: int | None = None,
    dry_run: bool = False,
    skip_llm: bool = False,
    leiden_resolution: float = 1.0,
) -> dict:
    """
    Full clustering pipeline.

    Steps:
        1. Load products from MySQL
        2. Tokenize + normalize
        3. Embed product names via OpenRouter
        4. Generate candidate edges (EAN + embedding + brand/token)
        5. Build graph → Leiden community detection
        6. Score each product in its cluster (cumulative score)
        7. LLM arbitration for grey zone
        8. Diff old vs new assignments
        9. Write results to MySQL + ClickHouse

    Returns dict with pipeline stats.
    """
    t0 = time.time()
    stats = {}

    # ── Step 1: Load products ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Loading products...")

    if category_id:
        products = load_products_by_category(category_id)
    elif niche_key:
        products = load_products_by_niche(niche_key)
    else:
        raise ValueError("Must specify either category_id or niche_key")

    if not products:
        log.warning("No products found!")
        return {"error": "no products"}

    stats["n_products"] = len(products)
    log.info(f"Loaded {len(products)} products")

    # ── Step 2: Tokenize ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Tokenizing...")

    for p in products:
        p.tokens = tokenize(p.name)

    avg_tokens = np.mean([len(p.tokens) for p in products])
    stats["avg_tokens"] = round(avg_tokens, 1)
    log.info(f"Average tokens per product: {avg_tokens:.1f}")

    # ── Step 3: Embed ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 3: Embedding product names (numbers stripped)...")

    texts = [strip_numbers(p.name) for p in products]
    embeddings = embed_texts(texts, show_progress=True)

    # Assign embeddings to products
    embedded_pairs = []
    for i, p in enumerate(products):
        if embeddings[i] is not None:
            p.embedding = embeddings[i]
            embedded_pairs.append((p.pk_id, embeddings[i]))

    stats["n_embedded"] = len(embedded_pairs)
    log.info(f"Embedded {len(embedded_pairs)}/{len(products)} products")

    if len(embedded_pairs) < len(products) * 0.5:
        log.warning("Less than 50% of products have embeddings!")

    # Build embedding matrix for blocking
    embedding_matrix, pk_id_list = build_embedding_matrix(embedded_pairs)

    # ── Step 4: Generate candidate edges ───────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4: Generating candidate edges...")

    edges = generate_candidate_edges(products, embedding_matrix, pk_id_list)
    stats["n_edges"] = len(edges)

    # ── Step 5: Graph clustering ───────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 5: Building graph and running Leiden...")

    graph, pk_to_vertex = build_graph(products, edges)
    membership = run_leiden(graph, resolution=leiden_resolution)

    gid_start = get_next_cluster_gid()
    clusters = membership_to_clusters(products, membership, gid_start)

    stats["n_clusters"] = len(clusters)
    stats["avg_cluster_size"] = round(
        np.mean([c.size for c in clusters.values()]), 1
    )

    # Compute centroids for scoring
    compute_cluster_centroids(clusters, products)

    # ── Step 5b: OCR from product card images (cache only) ──────────────
    log.info("=" * 60)
    log.info("STEP 5b: Loading OCR from cache...")
    from .ocr import _load_cached
    _ocr_cache = _load_cached([p.pk_id for p in products])
    ocr_count = 0
    for p in products:
        if p.pk_id in _ocr_cache:
            p.ocr_text = _ocr_cache[p.pk_id]["ocr_text"]
            p.ocr_attrs = _ocr_cache[p.pk_id]["ocr_attrs"]
            ocr_count += 1
    log.info(f"  OCR cache: {ocr_count}/{len(products)} products")
    stats["ocr_products"] = ocr_count

    # ── Step 5c: Extract structured attrs (regex schema, no API calls) ──
    if category_id:
        log.info("STEP 5c: Extracting structured attrs from name + OCR...")
        attr_count = extract_schema_attrs(products, category_id, niche_key=niche_key)
        stats["schema_attrs_products"] = attr_count

    # ── Step 6: Score products ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 6: Scoring products...")

    score_products(products, clusters)

    grey_count = sum(1 for p in products if p.decision == "grey")
    ok_count = sum(1 for p in products if p.decision == "ok")
    quarantine_count = sum(1 for p in products if p.decision == "quarantine")

    stats["scoring_ok"] = ok_count
    stats["scoring_grey"] = grey_count
    stats["scoring_quarantine"] = quarantine_count

    # ── Step 6b: Cross-encoder reranker (grey-zone prefilter) ──────────
    if RERANKER_ENABLED and grey_count > 0:
        log.info("=" * 60)
        log.info("STEP 6b: Cross-encoder reranker on grey zone...")
        stats.update(_apply_reranker_to_grey(products, clusters))
        grey_count = sum(1 for p in products if p.decision == "grey")

    # ── Step 7: LLM arbitration ────────────────────────────────────────
    if not skip_llm and grey_count > 0:
        log.info("=" * 60)
        log.info(f"STEP 7: LLM arbitration for {grey_count} grey products...")
        llm_stats = await arbitrate_grey_zone(products, clusters)
        stats.update(llm_stats)
    else:
        log.info("STEP 7: Skipping LLM (no grey products or --skip_llm)")

    # ── Step 8: Load old assignments and diff ──────────────────────────
    log.info("=" * 60)
    log.info("STEP 8: Computing diff with old assignments...")

    old_assignments = load_old_assignments([p.pk_id for p in products])
    for p in products:
        old = old_assignments.get(p.pk_id)
        if old:
            p.old_cluster_gid, p.old_score = old

    moves = generate_moves(products, clusters)
    stats["n_moves"] = len(moves)

    # ── Step 9: Write results ──────────────────────────────────────────
    if dry_run:
        log.info("=" * 60)
        log.info("DRY RUN — not writing to database")
        log.info(f"Would write {len(products)} assignments and {len(moves)} moves")
    else:
        log.info("=" * 60)
        log.info("STEP 9: Writing results...")

        run_id = get_next_run_id()
        assignments = []
        for p in products:
            if p.new_cluster_gid is not None and p.decision != "quarantine":
                assignments.append({
                    "pk_id": p.pk_id,
                    "cluster_gid": p.new_cluster_gid,
                    "score": p.new_score,
                    "main_product": 1 if clusters.get(p.new_cluster_gid, None)
                                        and clusters[p.new_cluster_gid].main_pk == p.pk_id
                                        else 0,
                    "niche_key": p.niche_key,
                })

        n_written = write_assignments(run_id, assignments)
        stats["n_written"] = n_written
        stats["run_id"] = run_id

        # Write moves to ClickHouse
        if moves:
            try:
                from .ch_writer import write_cluster_moves, ensure_table_exists
                ensure_table_exists()
                n_ch = write_cluster_moves(moves)
                stats["n_moves_written_ch"] = n_ch
            except Exception as e:
                log.error(f"ClickHouse write failed: {e}")
                stats["ch_error"] = str(e)
        stats["n_moves_pending"] = len(moves)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 1)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Products:     {stats['n_products']}")
    log.info(f"  Clusters:     {stats['n_clusters']}")
    log.info(f"  Avg size:     {stats['avg_cluster_size']}")
    log.info(f"  Edges:        {stats['n_edges']}")
    log.info(f"  Scoring: ok={stats['scoring_ok']} grey={stats['scoring_grey']} quarantine={stats['scoring_quarantine']}")
    log.info(f"  Moves:        {stats['n_moves']}")
    log.info(f"  Time:         {elapsed:.1f}s")
    log.info("=" * 60)

    return stats


async def run_highlight(
    category_id: int = None,
    niche_key: int = None,
    skip_llm: bool = False,
    cluster_gid: int = None,
) -> dict:
    """
    Highlight mode — review existing clusters WITHOUT re-clustering.

    Loads products with their CURRENT cluster assignments,
    scores each product against its cluster, runs LLM on grey zone,
    writes only moves to ClickHouse (does NOT touch MySQL).

    Steps:
        1. Load products with existing clusters
        2. Build ClusterResult objects + centroids
        3. Embed product names
        4. Score each product in its current cluster
        5. LLM arbitration for grey zone
        6. Write moves to ClickHouse only (created_by='cluster_ideal')
    """
    from collections import defaultdict
    from .models import ClusterResult

    t0 = time.time()
    stats = {}

    log.info("=" * 60)
    if niche_key and not category_id:
        # Resolve category_id from niche_key
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(text("SELECT category_id FROM mpstats_niches WHERE niche_key = :nk"), {"nk": niche_key}).mappings().first()
        if row:
            category_id = int(row["category_id"])
        else:
            return {"error": f"niche_key {niche_key} not found"}
    log.info(f"HIGHLIGHT MODE: category {category_id}" + (f", niche {niche_key}" if niche_key else ""))
    log.info("=" * 60)

    # Step 1: Load products with existing clusters
    log.info("STEP 1: Loading products with existing clusters...")
    if niche_key:
        products = load_products_by_niche(niche_key)
    else:
        products = load_products_by_category(category_id)
    if not products:
        return {"error": "no products"}

    # Load current assignments
    old_assignments = load_old_assignments([p.pk_id for p in products])
    for p in products:
        old = old_assignments.get(p.pk_id)
        if old:
            p.old_cluster_gid, p.old_score = old
            p.new_cluster_gid = old[0]  # keep current cluster

    # Filter: only products that have clusters
    products_with_clusters = [p for p in products if p.new_cluster_gid]
    products_without = [p for p in products if not p.new_cluster_gid]

    # Optional single-cluster filter (for debugging / targeted reruns)
    if cluster_gid is not None:
        products_with_clusters = [p for p in products_with_clusters
                                   if p.new_cluster_gid == cluster_gid]
        products_without = []
        log.info(f"  FILTER: cluster_gid={cluster_gid} → {len(products_with_clusters)} products")
        if not products_with_clusters:
            return {"error": f"no products in cluster {cluster_gid}"}
    stats["n_products"] = len(products)
    stats["n_with_clusters"] = len(products_with_clusters)
    stats["n_without_clusters"] = len(products_without)
    log.info(f"  {len(products_with_clusters)} with clusters, {len(products_without)} without")

    # Step 2: Build ClusterResult objects from existing assignments
    log.info("STEP 2: Building cluster objects...")
    by_cluster: dict[int, list] = defaultdict(list)
    for p in products_with_clusters:
        by_cluster[p.new_cluster_gid].append(p)

    clusters: dict[int, ClusterResult] = {}
    for gid, members in by_cluster.items():
        prices = [p.price for p in members if p.price > 0]
        clusters[gid] = ClusterResult(
            gid=gid,
            product_ids=[p.pk_id for p in members],
            size=len(members),
            avg_price=np.mean(prices) if prices else 0,
            main_pk=members[0].pk_id,
        )

    stats["n_clusters"] = len(clusters)
    log.info(f"  {len(clusters)} clusters")

    # Step 3: Embed
    log.info("STEP 3: Embedding product names (numbers stripped)...")
    texts = [strip_numbers(p.name) for p in products_with_clusters]
    embeddings = embed_texts(texts, show_progress=True)

    embedded_pairs = []
    for i, p in enumerate(products_with_clusters):
        if embeddings[i] is not None:
            p.embedding = embeddings[i]
            embedded_pairs.append((p.pk_id, embeddings[i]))

    # Compute centroids
    compute_cluster_centroids(clusters, products_with_clusters)

    log.info("STEP 3b: Loading OCR from cache...")
    from .ocr import _load_cached
    _ocr_cache = _load_cached([p.pk_id for p in products_with_clusters])
    ocr_count = 0
    for p in products_with_clusters:
        if p.pk_id in _ocr_cache:
            p.ocr_text = _ocr_cache[p.pk_id]["ocr_text"]
            p.ocr_attrs = _ocr_cache[p.pk_id]["ocr_attrs"]
            ocr_count += 1
    log.info(f"  OCR cache: {ocr_count}/{len(products_with_clusters)} products")
    stats["ocr_products"] = ocr_count
    stats["ocr_fixed"] = 0

    # Step 3c: Extract structured attrs
    log.info("STEP 3c: Extracting structured attrs from name + OCR...")
    attr_count = extract_schema_attrs(products_with_clusters, category_id, niche_key=niche_key)
    stats["schema_attrs_products"] = attr_count

    # Step 4: Score
    log.info("STEP 4: Scoring products in current clusters...")
    score_products(products_with_clusters, clusters, review_mode=True)

    ok_count = sum(1 for p in products_with_clusters if p.decision == "ok")
    grey_count = sum(1 for p in products_with_clusters if p.decision == "grey")
    quarantine_count = sum(1 for p in products_with_clusters if p.decision == "quarantine")
    stats.update({"scoring_ok": ok_count, "scoring_grey": grey_count, "scoring_quarantine": quarantine_count})
    log.info(f"  ok={ok_count}, grey={grey_count}, quarantine={quarantine_count}")

    log.info("STEP 4a: Anomaly-pass (flag quiet mismatches and small-cluster members)...")
    from .anomaly_pass import apply_anomaly_pass
    anomaly_stats = apply_anomaly_pass(products_with_clusters, clusters)
    stats.update(anomaly_stats)
    grey_count = sum(1 for p in products_with_clusters if p.decision == "grey")
    ok_count = sum(1 for p in products_with_clusters if p.decision == "ok")
    log.info(f"  after anomaly: ok={ok_count}, grey={grey_count}")

    # Step 4b: Cross-encoder reranker (grey-zone prefilter)
    if RERANKER_ENABLED and grey_count > 0:
        log.info("STEP 4b: Cross-encoder reranker on grey zone...")
        stats.update(_apply_reranker_to_grey(products_with_clusters, clusters))
        grey_count = sum(1 for p in products_with_clusters if p.decision == "grey")

    # Step 5: LLM
    if not skip_llm and grey_count > 0:
        log.info(f"STEP 5: LLM arbitration for {grey_count} grey products...")
        llm_stats = await arbitrate_grey_zone(products_with_clusters, clusters)
        stats.update(llm_stats)
    else:
        log.info("STEP 5: Skipping LLM")

    # Step 5b: REMOVED — LLM quarantine decisions are final
    quarantined_count = sum(1 for p in products_with_clusters if p.decision == "quarantine")
    if quarantined_count:
        log.info(f"STEP 5b: {quarantined_count} quarantined products (kept as-is, no override)")

    # Step 5b-vision: Vision-LLM function tag check on move candidates.
    # Compares the functional type of the product (what it DOES) against
    # source and target clusters. Mismatches → quarantine before CLIP runs.
    move_products = [p for p in products_with_clusters if p.decision == "move"]
    if move_products and category_id:
        try:
            from .vision_tags import get_vision_tags
            from collections import Counter

            # Resolve category name (best-effort, falls back to id)
            try:
                with get_engine().connect() as conn:
                    row = conn.execute(text(
                        "SELECT name FROM mpstats_categories WHERE id = :cid LIMIT 1"
                    ), {"cid": category_id}).mappings().first()
                category_name = str(row["name"]) if row else f"Category {category_id}"
            except Exception:
                category_name = f"Category {category_id}"

            pk_to_product = {p.pk_id: p for p in products_with_clusters}
            vision_pks: set[int] = set()
            for p in move_products:
                vision_pks.add(p.pk_id)
                src_cl = clusters.get(p.old_cluster_gid)
                tgt_cl = clusters.get(p.new_cluster_gid)
                if src_cl:
                    vision_pks.update(src_cl.product_ids[:5])
                if tgt_cl:
                    vision_pks.update(tgt_cl.product_ids[:5])

            vision_input = [pk_to_product[pk] for pk in vision_pks if pk in pk_to_product]
            log.info(f"STEP 5b-vision: function tagging for {len(vision_input)} products"
                     f" ({len(move_products)} moves)")
            tags = get_vision_tags(
                vision_input,
                category_id=category_id,
                category_name=category_name,
                sample_names=[p.name for p in products_with_clusters[:80]],
            )

            def _cluster_function(cl) -> tuple[str | None, int]:
                """Majority function slug among first 5 cluster members. Returns (slug, count)."""
                slugs = []
                for pk in cl.product_ids[:5]:
                    t = tags.get(pk)
                    if t and t.get("function") and t["function"] != "unknown":
                        slugs.append(t["function"])
                if not slugs:
                    return None, 0
                slug, n = Counter(slugs).most_common(1)[0]
                return slug, n

            vision_rejected = 0
            vision_confirmed = 0
            vision_skipped = 0
            for p in move_products:
                t = tags.get(p.pk_id)
                if not t or t.get("function") in (None, "unknown", ""):
                    vision_skipped += 1
                    continue
                p_func = t["function"]
                tgt_cl = clusters.get(p.new_cluster_gid)
                if not tgt_cl:
                    vision_skipped += 1
                    continue
                tgt_func, tgt_n = _cluster_function(tgt_cl)
                if not tgt_func or tgt_n < 2:
                    vision_skipped += 1
                    continue
                if p_func != tgt_func:
                    p.decision = "quarantine"
                    p.reason += (f" | [VISION-FUNC-REJECT] product={p_func}"
                                 f" tgt_cluster={tgt_func} (n={tgt_n})")
                    vision_rejected += 1
                    log.info(f"  VISION FUNC pk={p.pk_id}: {p_func} vs tgt={tgt_func} → quarantine")
                else:
                    vision_confirmed += 1
                    p.reason += f" | [VISION-FUNC-OK] {p_func}"

            stats["vision_function_rejected"] = vision_rejected
            stats["vision_function_confirmed"] = vision_confirmed
            stats["vision_function_skipped"] = vision_skipped
            log.info(f"  VISION FUNC: {vision_confirmed} confirmed, {vision_rejected} rejected,"
                     f" {vision_skipped} skipped (no tag) out of {len(move_products)}")
        except Exception as e:
            log.warning(f"  Vision function tagging failed: {e}", exc_info=True)

    # Recompute move_products after vision rejections
    move_products = [p for p in products_with_clusters if p.decision == "move"]

    # Step 5c: CLIP image verification — compare product with source vs target cluster
    # When CLIP is ambiguous (scores ~equal), use numeric attribute tiebreaker
    if move_products:
        log.info(f"STEP 5c: CLIP image verification for {len(move_products)} move candidates...")
        try:
            from ce_ideal.image_embed import embed_images as clip_embed_images
            from .value_canon import canon_numeric

            pk_to_product = {p.pk_id: p for p in products_with_clusters}

            need_pks = set()
            for p in move_products:
                need_pks.add(p.pk_id)
                src_cl = clusters.get(p.old_cluster_gid)
                tgt_cl = clusters.get(p.new_cluster_gid)
                if src_cl:
                    need_pks.update(src_cl.product_ids[:5])
                if tgt_cl:
                    need_pks.update(tgt_cl.product_ids[:5])

            need_products = [pk_to_product[pk] for pk in need_pks if pk in pk_to_product]
            log.info(f"  Generating CLIP embeddings for {len(need_products)} products...")
            img_embs = clip_embed_images(need_products)
            log.info(f"  CLIP embeddings: {len(img_embs)} products")

            # ── helper: numeric distance between product and cluster ──
            CLIP_AMBIGUOUS_THRESHOLD = 0.03  # |delta| < this → CLIP can't decide

            def _cluster_numeric_medians(cl, exclude_pk=None):
                """Build {canon_attr_name: median_value} for cluster's P1/P2 numeric attrs."""
                from collections import Counter
                from .scoring import _canon_key
                attr_vals: dict[str, list[float]] = {}
                for pk in cl.product_ids[:20]:
                    if pk == exclude_pk:
                        continue
                    cp = pk_to_product.get(pk)
                    if not cp or not cp.ocr_attrs:
                        continue
                    for aname, adata in cp.ocr_attrs.items():
                        priority = adata.get("priority", 99) if isinstance(adata, dict) else 99
                        if priority > 2:
                            continue
                        is_num = adata.get("numeric", False) if isinstance(adata, dict) else False
                        if not is_num:
                            continue
                        val = adata.get("value") if isinstance(adata, dict) else adata
                        cn = canon_numeric(val)
                        if cn is not None:
                            ckey = _canon_key(aname)
                            attr_vals.setdefault(ckey, []).append(cn[0])
                medians = {}
                for k, vs in attr_vals.items():
                    if len(vs) >= 2:
                        vs.sort()
                        medians[k] = vs[len(vs) // 2]
                return medians

            def _numeric_distance(product, medians):
                """Sum of |log(product_val / median)| for matching P1/P2 attrs.
                Lower = closer fit. Returns (distance, n_attrs_compared)."""
                import math
                from .scoring import _canon_key
                if not product.ocr_attrs or not medians:
                    return None, 0
                dist = 0.0
                n = 0
                for aname, adata in product.ocr_attrs.items():
                    priority = adata.get("priority", 99) if isinstance(adata, dict) else 99
                    if priority > 2:
                        continue
                    is_num = adata.get("numeric", False) if isinstance(adata, dict) else False
                    if not is_num:
                        continue
                    val = adata.get("value") if isinstance(adata, dict) else adata
                    cn = canon_numeric(val)
                    if cn is None:
                        continue
                    ckey = _canon_key(aname)
                    if ckey not in medians:
                        continue
                    pv = cn[0]
                    mv = medians[ckey]
                    if pv > 0 and mv > 0:
                        dist += abs(math.log(pv / mv))
                        n += 1
                return dist if n > 0 else None, n

            clip_rejected = 0
            clip_confirmed = 0
            clip_numeric_decided = 0
            clip_consistency_rejected = 0

            def _intra_consistency(cl, embs_dict, exclude_pk=None):
                """Mean pairwise cosine among cluster members (intra-cluster coherence)."""
                vecs = [embs_dict[pk] for pk in cl.product_ids[:10]
                        if pk in embs_dict and pk != exclude_pk]
                if len(vecs) < 2:
                    return None
                sims = []
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        sims.append(float(np.dot(vecs[i], vecs[j])))
                return sum(sims) / len(sims)

            CONSISTENCY_MIN_DELTA = 0.02  # tgt must not be this much worse than src

            for p in move_products:
                if p.pk_id not in img_embs:
                    continue
                p_emb = img_embs[p.pk_id]

                src_cl = clusters.get(p.old_cluster_gid)
                tgt_cl = clusters.get(p.new_cluster_gid)
                if not src_cl or not tgt_cl:
                    continue

                src_embs = [img_embs[pk] for pk in src_cl.product_ids[:5] if pk in img_embs and pk != p.pk_id]
                tgt_embs = [img_embs[pk] for pk in tgt_cl.product_ids[:5] if pk in img_embs]

                if not src_embs or not tgt_embs:
                    continue

                src_centroid = np.mean(src_embs, axis=0).astype(np.float32)
                norm_s = np.linalg.norm(src_centroid)
                if norm_s > 0:
                    src_centroid /= norm_s

                tgt_centroid = np.mean(tgt_embs, axis=0).astype(np.float32)
                norm_t = np.linalg.norm(tgt_centroid)
                if norm_t > 0:
                    tgt_centroid /= norm_t

                sim_source = float(np.dot(p_emb, src_centroid))
                sim_target = float(np.dot(p_emb, tgt_centroid))
                clip_delta = sim_target - sim_source

                # ── Intra-cluster consistency gate ──
                src_coh = _intra_consistency(src_cl, img_embs, exclude_pk=p.pk_id)
                tgt_coh = _intra_consistency(tgt_cl, img_embs)
                if (src_coh is not None and tgt_coh is not None
                        and tgt_coh < src_coh - CONSISTENCY_MIN_DELTA
                        and clip_delta < CLIP_AMBIGUOUS_THRESHOLD):
                    p.decision = "quarantine"
                    p.reason += (f" | [CLIP-CONSISTENCY-REJECT] src_coh={src_coh:.3f}"
                                 f" tgt_coh={tgt_coh:.3f} delta={clip_delta:+.3f}")
                    clip_consistency_rejected += 1
                    clip_rejected += 1
                    log.info(f"  CLIP CONSISTENCY pk={p.pk_id}: src_coh={src_coh:.3f}"
                             f" > tgt_coh={tgt_coh:.3f} (clip_d={clip_delta:+.3f}) → quarantine")
                    continue

                # ── CLIP is ambiguous → numeric tiebreaker ──
                if abs(clip_delta) < CLIP_AMBIGUOUS_THRESHOLD:
                    src_meds = _cluster_numeric_medians(src_cl, exclude_pk=p.pk_id)
                    tgt_meds = _cluster_numeric_medians(tgt_cl)
                    dist_src, n_src = _numeric_distance(p, src_meds)
                    dist_tgt, n_tgt = _numeric_distance(p, tgt_meds)

                    if dist_src is not None and dist_tgt is not None and (n_src + n_tgt) > 0:
                        clip_numeric_decided += 1
                        if dist_tgt <= dist_src:
                            # Numerically closer to target → confirm move
                            p.reason += (f" | [CLIP-AMBIG→NUM-OK] clip_d={clip_delta:+.3f}"
                                         f" num_src={dist_src:.3f}({n_src}) num_tgt={dist_tgt:.3f}({n_tgt})")
                            clip_confirmed += 1
                            log.info(f"  CLIP AMBIG pk={p.pk_id}: clip_d={clip_delta:+.3f}"
                                     f" → NUM tgt={dist_tgt:.3f} <= src={dist_src:.3f} → confirm")
                        else:
                            # Numerically closer to source → reject move
                            p.decision = "quarantine"
                            p.reason += (f" | [CLIP-AMBIG→NUM-REJECT] clip_d={clip_delta:+.3f}"
                                         f" num_src={dist_src:.3f}({n_src}) > num_tgt={dist_tgt:.3f}({n_tgt})")
                            clip_rejected += 1
                            log.info(f"  CLIP AMBIG pk={p.pk_id}: clip_d={clip_delta:+.3f}"
                                     f" → NUM src={dist_src:.3f} < tgt={dist_tgt:.3f} → quarantine")
                    else:
                        # No numeric data + ambiguous CLIP → reject (no evidence target is better)
                        p.decision = "quarantine"
                        p.reason += (f" | [CLIP-AMBIG-REJECT] src={sim_source:.3f} ~ tgt={sim_target:.3f}"
                                     f" (delta={clip_delta:+.3f}, no numeric data)")
                        clip_rejected += 1
                        log.info(f"  CLIP AMBIG pk={p.pk_id}: src={sim_source:.3f} ~ tgt={sim_target:.3f}"
                                 f" no numeric → quarantine")

                elif clip_delta <= 0:
                    # ── CLIP says source is same or better → reject ──
                    p.decision = "quarantine"
                    p.reason += (f" | [CLIP-REJECT] src={sim_source:.3f} >= tgt={sim_target:.3f}"
                                 f" (delta={clip_delta:+.3f})")
                    clip_rejected += 1
                    log.info(f"  CLIP REJECT pk={p.pk_id}: src={sim_source:.3f} >= tgt={sim_target:.3f} → quarantine")
                else:
                    # ── CLIP clearly says target is better (delta > 0) ──
                    p.reason += (f" | [CLIP-OK] tgt={sim_target:.3f} > src={sim_source:.3f}"
                                 f" (delta={clip_delta:+.3f})")
                    clip_confirmed += 1

            stats["clip_rejected"] = clip_rejected
            stats["clip_confirmed"] = clip_confirmed
            stats["clip_numeric_decided"] = clip_numeric_decided
            stats["clip_consistency_rejected"] = clip_consistency_rejected
            log.info(f"  CLIP: {clip_confirmed} confirmed, {clip_rejected} rejected"
                     f" ({clip_numeric_decided} numeric, {clip_consistency_rejected} consistency)"
                     f" out of {len(move_products)}")
        except ImportError:
            log.warning("  CLIP verification skipped: ce_ideal.image_embed not available")
        except Exception as e:
            log.warning(f"  CLIP verification failed: {e}")

    # Step 6: Generate moves and write to ClickHouse ONLY
    log.info("STEP 6: Writing moves to ClickHouse...")
    moves = generate_moves(products_with_clusters, clusters)
    stats["n_moves"] = len(moves)

    if moves:
        try:
            from .ch_writer import write_cluster_moves, ensure_table_exists
            ensure_table_exists()
            n_ch = write_cluster_moves(moves)
            stats["n_moves_written_ch"] = n_ch
        except Exception as e:
            log.error(f"ClickHouse write failed: {e}")
            stats["ch_error"] = str(e)

    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 1)

    log.info("=" * 60)
    log.info("HIGHLIGHT COMPLETE")
    log.info(f"  Products: {stats['n_products']} ({stats['n_with_clusters']} with clusters)")
    log.info(f"  Scoring: ok={ok_count} grey={grey_count} quarantine={quarantine_count}")
    log.info(f"  Moves to CH: {stats.get('n_moves_written_ch', 0)}")
    log.info(f"  Time: {elapsed:.1f}s")
    log.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Product Clustering Engine v2")
    parser.add_argument("--category_id", type=int, help="Category ID to cluster")
    parser.add_argument("--niche_key", type=int, help="Niche key to cluster")
    parser.add_argument("--cluster_gid", type=int, help="Highlight: limit to a single cluster_gid")
    parser.add_argument("--highlight", action="store_true",
        help="Highlight mode: review existing clusters, write only to CH")
    parser.add_argument("--dry_run", action="store_true", help="Don't write to DB")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM arbitration")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if not args.category_id and not args.niche_key:
        parser.error("Must specify --category_id or --niche_key")

    setup_logging(args.verbose)

    if args.highlight:
        if not args.category_id and not args.niche_key:
            parser.error("--highlight requires --category_id or --niche_key")
        stats = asyncio.run(run_highlight(
            category_id=args.category_id,
            niche_key=args.niche_key,
            skip_llm=args.skip_llm,
            cluster_gid=args.cluster_gid,
        ))
    else:
        stats = asyncio.run(run_pipeline(
            category_id=args.category_id,
            niche_key=args.niche_key,
            dry_run=args.dry_run,
            skip_llm=args.skip_llm,
            leiden_resolution=args.resolution,
        ))

    # Print stats as JSON for easy parsing
    import json
    print("\n--- STATS ---")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
