"""
Scoring — cascaded priority scoring for product-cluster assignments.

Priority cascade:
  1. Numbers (from niche schema) — BLOCKER: key attr mismatch → grey
     Number MATCH with cluster dominant → force ok
  2. Text embedding similarity (0..0.35)
  3. Token overlap (0..0.15)
  4. Brand (0..0.10)
  5. Price — informational (-0.05..+0.05)
"""
from __future__ import annotations
import logging
from collections import Counter

import numpy as np

from .config import SCORE_OK, SCORE_QUARANTINE
from .models import Product, ClusterResult
from .text_processing import (
    token_overlap, brand_match, extract_numeric_attrs,
    normalize_brand, price_ratio,
)
from .embeddings import cosine_sim
from .attr_weights import get_blocking_attrs
from .value_canon import canon_string, canon_numeric, canon_for_compare

log = logging.getLogger("cluster_engine.scoring")

# ── Synonym dict: loaded once, maps any key variant → canonical key ──
_synonym_reverse: dict[str, str] | None = None


def _get_synonym_map() -> dict[str, str]:
    """Load reverse synonym map (lazy, cached)."""
    global _synonym_reverse
    if _synonym_reverse is None:
        try:
            from .build_synonyms import load_synonyms, build_reverse_map
            syns = load_synonyms()
            _synonym_reverse = build_reverse_map(syns) if syns else {}
            if _synonym_reverse:
                log.info(f"Loaded synonym map: {len(_synonym_reverse)} keys → {len(syns)} canonical forms")
        except Exception:
            _synonym_reverse = {}
    return _synonym_reverse


def _canon_key(key: str) -> str:
    """Normalize attribute key via synonym dict."""
    rev = _get_synonym_map()
    return rev.get(key.lower().strip(), key)


# Cache blocking attrs per category (loaded once)
_blocking_cache: dict[int, dict] = {}

# Count-type attribute keys: discrete piece counts where ±15% is too loose.
# 118 vs 128 pieces are different SKUs, not the same product.
_COUNT_KEY_FRAGMENTS = ("количество", "предмет", "_шт", "штук", "_pcs", "items_count")


def _is_count_attr(attr_name: str) -> bool:
    name = (attr_name or "").lower()
    return any(frag in name for frag in _COUNT_KEY_FRAGMENTS)


def _numeric_tolerance(attr_name: str) -> float:
    """Return ratio threshold for numeric match. Counts: 3%; physical: 15%."""
    return 1.03 if _is_count_attr(attr_name) else 1.15


def _embedding_score(product: Product, cluster: ClusterResult) -> float:
    """Embedding similarity to cluster centroid → [0, 0.35]."""
    if product.embedding is None or cluster.centroid is None:
        return 0.15
    sim = cosine_sim(product.embedding, cluster.centroid)
    product.embedding_sim = sim
    return max(0.0, min(0.35, (sim - 0.5) * 0.7778))


def _token_score(product: Product, cluster: ClusterResult, pk_to_product: dict) -> float:
    """Token overlap with cluster exemplars → [0, 0.15]."""
    cluster_tokens: list[str] = []
    for pk in cluster.product_ids[:20]:
        p = pk_to_product.get(pk)
        if p and p.tokens:
            cluster_tokens.extend(p.tokens)

    if not cluster_tokens or not product.tokens:
        return 0.05

    n_sampled = min(len(cluster.product_ids), 20)
    token_freq = Counter(cluster_tokens)
    core_tokens = [t for t, cnt in token_freq.items()
                   if cnt >= max(2, n_sampled * 0.3)]

    if not core_tokens:
        return 0.05

    product_set = set(product.tokens)
    core_set = set(core_tokens)
    overlap = len(product_set & core_set) / len(core_set) if core_set else 0.0

    product.token_overlap_score = overlap
    return overlap * 0.15


def _brand_score(product: Product, cluster: ClusterResult) -> float:
    """Brand match with cluster dominant brand → [0, 0.10]."""
    if not product.brand or not cluster.brand:
        return 0.03

    bm = brand_match(product.brand, cluster.brand)
    product.brand_match_score = bm

    if bm == 1.0:
        return 0.10
    elif bm == 0.5:
        return 0.05
    return 0.0


def _gemini_attr_block(product: Product, cluster: ClusterResult, pk_to_product: dict,
                       category_id: int | None = None) -> tuple[bool, list[str]]:
    """Check auto-calibrated Gemini attr blocking.
    Returns (is_blocked, list_of_mismatch_descriptions).
    Handles both string (exact match) and numeric (±15%) BLOCK attrs.
    """
    if not product.ocr_attrs or not category_id:
        return False, []

    global _blocking_cache
    if category_id not in _blocking_cache:
        _blocking_cache[category_id] = get_blocking_attrs(category_id)
    blocking = _blocking_cache[category_id]
    if not blocking:
        return False, []

    # Build cluster dominant values for each Gemini attr (canonical keys)
    cluster_vals: dict[str, list] = {}
    for pk in cluster.product_ids:
        p = pk_to_product.get(pk)
        if not p or not p.ocr_attrs:
            continue
        for attr_name, attr_data in p.ocr_attrs.items():
            ckey = _canon_key(attr_name)
            val = attr_data.get("value") if isinstance(attr_data, dict) else attr_data
            if val is not None and not isinstance(val, bool):
                cluster_vals.setdefault(ckey, []).append(val)

    # Outlier filter: in clusters with 5+ products, drop singleton values
    # (likely OCR/regex noise like "663м" appearing once in a cluster of 50м items).
    if len(cluster.product_ids) >= 5:
        for attr_name, vals in list(cluster_vals.items()):
            ctr = Counter(canon_string(v) for v in vals)
            cluster_vals[attr_name] = [v for v in vals if ctr[canon_string(v)] >= 2]
            if not cluster_vals[attr_name]:
                del cluster_vals[attr_name]

    mismatches = []
    # Build product attrs by canonical key for blocking lookup
    _p_block_attrs = {_canon_key(k): v for k, v in product.ocr_attrs.items()}
    for attr_name, cfg in blocking.items():
        if not cfg.get("block"):
            continue
        ckey = _canon_key(attr_name)
        if ckey not in _p_block_attrs or ckey not in cluster_vals:
            continue

        p_attr = _p_block_attrs[ckey]
        p_val = p_attr.get("value") if isinstance(p_attr, dict) else p_attr
        if p_val is None or isinstance(p_val, bool):
            continue

        vals = cluster_vals[ckey]
        if len(vals) < 3:
            continue

        attr_type = cfg.get("type", "string")
        if attr_type == "numeric":
            # Canonicalize all values to base SI unit before comparison
            canon_vals = []
            for v in vals:
                cv = canon_numeric(v)
                if cv is not None:
                    canon_vals.append(cv[0])
            pf_res = canon_numeric(p_val)
            if not canon_vals or pf_res is None:
                continue
            sorted_v = sorted(canon_vals)
            pf = pf_res[0]
            best_group = []
            for anchor in sorted_v:
                group = [v for v in sorted_v if anchor * 0.85 <= v <= anchor * 1.15]
                if len(group) > len(best_group):
                    best_group = group
            if len(best_group) / len(sorted_v) < 0.4:
                continue  # no clear dominant
            median = sorted(best_group)[len(best_group) // 2]
            if pf > 0 and median > 0:
                ratio = max(pf, median) / min(pf, median)
                if ratio > 1.15:
                    mismatches.append(f"{attr_name}={p_val}≠{median}")
        else:
            # String compare: canonicalize (NFKD, ё→е, lowercase, strip punct)
            ctr = Counter(canon_string(v) for v in vals if canon_string(v))
            if not ctr:
                continue
            top_val, top_cnt = ctr.most_common(1)[0]
            if top_cnt / len(vals) < 0.5:
                continue  # no clear dominant
            p_str = canon_string(p_val)
            if not p_str or p_str == top_val:
                continue
            mismatches.append(f"{attr_name}={p_val!r}≠{top_val!r}")

    return bool(mismatches), mismatches


def _numeric_attr_score(product: Product, cluster: ClusterResult, pk_to_product: dict) -> tuple[float, bool, bool]:
    """
    Numeric attribute consistency → [-0.25, 0.10].
    Returns (score, is_blocker, is_match).
    blocker = key numeric attr mismatches significantly (>30%)
    match = key numeric attr matches cluster dominant (±15%)
    """
    blocker = False
    match = False

    if product.ocr_attrs:
        cluster_schema: dict[str, Counter] = {}
        # Use the full cluster, not just first 30 — otherwise the dominant
        # group is computed from a biased sample. For cluster 258695 (105
        # products) the [:30] cap was causing 50м items to be marked "ok"
        # while 100м items got blocker, depending on sample order.
        for pk in cluster.product_ids:
            p = pk_to_product.get(pk)
            if p and p.ocr_attrs:
                for attr_name, attr_data in p.ocr_attrs.items():
                    ckey = _canon_key(attr_name)
                    if ckey not in cluster_schema:
                        cluster_schema[ckey] = Counter()
                    val = attr_data.get("value") if isinstance(attr_data, dict) else attr_data
                    if val is None or isinstance(val, (list, dict, set)):
                        continue
                    cluster_schema[ckey][val] += 1

        # Outlier filter: drop singleton values in 5+ product clusters
        if len(cluster.product_ids) >= 5:
            for attr_name in list(cluster_schema.keys()):
                cluster_schema[attr_name] = Counter({
                    v: c for v, c in cluster_schema[attr_name].items() if c >= 2
                })
                if not cluster_schema[attr_name]:
                    del cluster_schema[attr_name]

        score = 0.0
        # Build product attrs indexed by canonical key
        _p_attrs_by_canon = {}
        for _ak, _av in product.ocr_attrs.items():
            _p_attrs_by_canon[_canon_key(_ak)] = _av

        for attr_name, counter in cluster_schema.items():
            if attr_name not in _p_attrs_by_canon:
                continue
            if not counter:
                continue

            p_attr = _p_attrs_by_canon[attr_name]
            p_val = p_attr.get("value") if isinstance(p_attr, dict) else p_attr
            priority = p_attr.get("priority", 99) if isinstance(p_attr, dict) else 99
            is_numeric = p_attr.get("numeric", False) if isinstance(p_attr, dict) else False

            if not is_numeric or p_val is None:
                continue

            # Group close values (±15%) to find dominant.
            # Canonicalize each via canon_numeric so "1.5 кВт"/"1500 Вт"/"15м"
            # all map to the same SI base unit before comparison.
            vals = []
            for v in counter.elements():
                cv = canon_numeric(v)
                if cv is not None:
                    vals.append(cv[0])
            if not vals:
                continue

            vals_sorted = sorted(vals)
            best_group = []
            for anchor in vals_sorted:
                group = [v for v in vals_sorted if anchor * 0.85 <= v <= anchor * 1.15]
                if len(group) > len(best_group):
                    best_group = group

            if len(best_group) / len(vals) < 0.4:
                # No clear dominant group — cluster is heterogeneous in this attr.
                # For P1 attrs, mark as blocker ONLY items that are NOT in the
                # largest group. The largest group is the de-facto kernel of
                # the cluster — its members stay grey (not blocker), everyone
                # else is split material.
                if priority <= 1:
                    unique_vals = set(vals_sorted)
                    if len(unique_vals) >= 3:
                        pf_res = canon_numeric(p_val)
                        if pf_res is None:
                            continue
                        pf_check = pf_res[0]
                        in_best = False
                        if best_group:
                            anchor = best_group[len(best_group) // 2]
                            if anchor * 0.85 <= pf_check <= anchor * 1.15:
                                in_best = True
                        if not in_best:
                            score -= 0.15
                            blocker = True
                continue

            median = sorted(best_group)[len(best_group) // 2]

            pf_res = canon_numeric(p_val)
            if pf_res is None:
                continue
            pf = pf_res[0]

            if pf > 0 and median > 0:
                ratio = max(pf, median) / min(pf, median)
                if ratio <= 1.15 and priority <= 2:
                    score += 0.05
                    match = True
                elif ratio > 1.3 and priority <= 2:
                    score -= 0.15
                    blocker = True
                elif ratio > 1.3:
                    score -= 0.05

        return max(-0.25, min(0.10, score)), blocker, match

    # Fallback: extract from name
    product_attrs = extract_numeric_attrs(product.name)
    if not product_attrs:
        return 0.0, False, False

    cluster_attrs: dict[str, list[float]] = {}
    for pk in cluster.product_ids[:30]:
        p = pk_to_product.get(pk)
        if p:
            for unit, val in extract_numeric_attrs(p.name).items():
                cluster_attrs.setdefault(unit, []).append(val)

    if not cluster_attrs:
        return 0.0, False, False

    score = 0.0
    for unit, vals in cluster_attrs.items():
        if len(vals) < 3:
            continue
        val_counts = Counter(vals)
        top_val, top_cnt = val_counts.most_common(1)[0]
        dominance = top_cnt / len(vals)
        if dominance < 0.5:
            continue

        product_val = product_attrs.get(unit)
        if product_val is None:
            continue

        if product_val == top_val:
            score += 0.05
        else:
            ratio = max(product_val / top_val, top_val / product_val) if top_val > 0 and product_val > 0 else 999
            if ratio > 3.0:
                score -= 0.10
            elif ratio > 1.5:
                score -= 0.03

    return max(-0.15, min(0.10, score)), False, False


def compute_price_anomaly(
    product_price: float,
    cluster_prices: list[float],
    niche_prices: list[float],
) -> tuple[int, int, float, int]:
    """
    IQR-based price tier with cluster→niche fallback.

    Returns (tier, src, ratio, flags):
      tier: 0=LOW (<Q1), 1=MID (Q1..Q3), 2=HIGH (>Q3), 3=SPIKE (outside Q±1.5·IQR)
      src:  0=cluster reference, 1=niche reference
      ratio: product_price / reference_median
      flags: bit 1 = SPIKE (other bits reserved for NICHE|SUBJ|BETTER)
    """
    if product_price <= 0:
        return (1, 0, 1.0, 0)

    cluster_pos = [p for p in cluster_prices if p > 0]
    if len(cluster_pos) >= 4:
        ref = cluster_pos
        src = 0
    else:
        ref = [p for p in niche_prices if p > 0]
        src = 1

    if len(ref) < 4:
        return (1, src, 1.0, 0)

    s = sorted(ref)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    iqr = q3 - q1
    median = s[n // 2]
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr

    ratio = round(product_price / median, 3) if median > 0 else 1.0

    if product_price < lo or product_price > hi:
        return (3, src, ratio, 1)
    if product_price < q1:
        return (0, src, ratio, 0)
    if product_price > q3:
        return (2, src, ratio, 0)
    return (1, src, ratio, 0)


def _price_score_legacy(product: Product, cluster: ClusterResult) -> float:
    """Legacy price consistency → [-0.05, 0.05]. Used in full pipeline mode."""
    if product.price <= 0 or cluster.avg_price <= 0:
        return 0.0

    pr = price_ratio(product.price, cluster.avg_price)
    product.price_ratio = pr

    if pr <= 1.5:
        return 0.05
    elif pr <= 3.0:
        return 0.0
    else:
        return -0.05


def _price_score_iqr(
    product: Product,
    cluster: ClusterResult,
    pk_to_product: dict,
    niche_prices_by_key: dict[int, list[float]],
) -> float:
    """IQR-based price anomaly score → [-0.05, 0.05]. Review mode only. Stores tier/src/flags on product."""
    if product.price <= 0:
        return 0.0

    cluster_prices = [
        pk_to_product[pk].price
        for pk in cluster.product_ids
        if pk in pk_to_product and pk_to_product[pk].price > 0
    ]
    niche_prices = niche_prices_by_key.get(product.niche_key, [])

    tier, src, ratio, flags = compute_price_anomaly(
        product.price, cluster_prices, niche_prices
    )
    product.price_tier = tier
    product.price_src = src
    product.price_ratio = ratio
    product.anomaly_flags |= flags

    if tier == 3:
        return -0.05
    if tier == 1:
        return 0.05
    return 0.0


def score_products(
    products: list[Product],
    clusters: dict[int, ClusterResult],
    review_mode: bool = False,
) -> None:
    """
    Compute cumulative score with priority cascade.
    Number blocker → grey (LLM decides).
    Number match → ok (skip LLM).
    """
    pk_to_product = {p.pk_id: p for p in products}

    niche_prices_by_key: dict[int, list[float]] = {}
    if review_mode:
        for _p in products:
            if _p.price > 0 and _p.niche_key:
                niche_prices_by_key.setdefault(_p.niche_key, []).append(_p.price)

    # Detect category from products for Gemini attr blocking
    _cat_id = getattr(products[0], "category_id", None) if products else None

    scored = 0
    decisions = {"ok": 0, "grey": 0, "quarantine": 0}
    gemini_blocks = 0

    for p in products:
        cluster = clusters.get(p.new_cluster_gid)
        if not cluster:
            p.cumulative_score = 0.0
            p.decision = "grey"
            p.confidence = 0.5
            decisions["grey"] += 1
            continue

        # Compute all signal scores
        emb_s = _embedding_score(p, cluster)
        tok_s = _token_score(p, cluster, pk_to_product)
        brand_s = _brand_score(p, cluster)
        attr_s, num_blocker, num_match = _numeric_attr_score(p, cluster, pk_to_product)
        if review_mode:
            price_s = _price_score_iqr(p, cluster, pk_to_product, niche_prices_by_key)
        else:
            price_s = _price_score_legacy(p, cluster)

        # Gemini attr blocking (auto-calibrated)
        cat_id = getattr(p, "category_id", _cat_id)
        gemini_blocked, gemini_mismatches = _gemini_attr_block(p, cluster, pk_to_product, cat_id)
        if gemini_blocked:
            num_blocker = True
            gemini_blocks += 1

        # Cumulative score
        raw = emb_s + tok_s + brand_s + attr_s + price_s
        score = max(0.0, min(1.0, (raw + 0.30) / 1.05))

        p.cumulative_score = round(score, 4)
        p.new_score = score
        p.confidence = score

        # Detect errors for LLM context
        errors = []

        if gemini_blocked and gemini_mismatches:
            errors.append(f"[АТРИБУТ-БЛОКЕР] {', '.join(gemini_mismatches)}")
        elif num_blocker:
            if p.ocr_attrs:
                attrs_str = ", ".join(
                    f"{v.get('label', k)}={v.get('raw_match', v.get('value'))}"
                    for k, v in p.ocr_attrs.items()
                    if isinstance(v, dict)
                )
                errors.append(f"[ЧИСЛО-БЛОКЕР] атрибуты [{attrs_str}] не совпадают с кластером")
            else:
                errors.append("[ЧИСЛО-БЛОКЕР] числовые атрибуты не совпадают с кластером")

        # Token overlap — informational only, don't add to errors
        # (was confusing LLM into moving products with matching numbers)

        # Price is informational only — extreme item-vs-mean ratio gets a soft
        # tag for the LLM to consider, but is NEVER a blocker by itself.
        # Price spread alone proved unreliable (e.g. small heterogeneous clusters
        # with 5+ unique sorts at different price points were forcing splits
        # the LLM couldn't reason about).
        item_pr = getattr(p, "price_ratio", 0) or 0
        if item_pr > 3.0:
            errors.append(f"[ЦЕНА] {item_pr:.1f}x от средней кластера")

        if len(errors) >= 2:
            errors.insert(0, f"[COMPOUND] {len(errors)} проблем")

        if errors:
            p.reason = " | ".join(errors)
            p.score_errors = errors
        else:
            p.reason = ""
            p.score_errors = []

        # Decision — scoring only filters extremes, LLM decides the rest
        if num_blocker:
            p.decision = "grey"
            decisions["grey"] += 1
        elif score >= SCORE_OK:
            p.decision = "ok"
            p.decided_by = "scoring"
            decisions["ok"] += 1
        elif score <= SCORE_QUARANTINE:
            p.decision = "quarantine"
            p.decided_by = "scoring"
            decisions["quarantine"] += 1
        else:
            p.decision = "grey"
            decisions["grey"] += 1

        scored += 1

    log.info(
        f"Scored {scored} products: "
        f"ok={decisions['ok']}, grey={decisions['grey']}, "
        f"quarantine={decisions['quarantine']}"
        f"{f', gemini_blocks={gemini_blocks}' if gemini_blocks else ''}"
    )
