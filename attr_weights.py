#!/usr/bin/env python3
"""
Auto-calibrate attribute blocking weights from existing cluster data.

For each Gemini attr key, computes "uniformity" within good clusters:
  uniformity = avg fraction of most-common value across clusters.

High uniformity (>= 0.7) → BLOCK attr (same product = same value).
Low uniformity (< 0.7) → VARIATION attr (don't block).

String attrs with uniformity >= 0.7 → exact-match BLOCK.
Numeric attrs with uniformity >= 0.7 → ratio-based BLOCK (±15%).

Output: JSON dict per category:
  {"количество_предметов": {"block": true, "type": "numeric", "uniformity": 0.92},
   "бренд": {"block": false, "type": "string", "uniformity": 0.15}, ...}

Usage:
  cd /opt && python3 -m cluster_engine_v2.attr_weights --categories 37 42 44 60
"""
from __future__ import annotations
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import text

from .db import get_engine

log = logging.getLogger("attr_weights")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MIN_CLUSTER_SIZE = 5
MAX_CLUSTER_SIZE = 100
BLOCK_THRESHOLD = 0.70  # uniformity >= this → BLOCK
OUTPUT_PATH = Path(os.getenv(
    "ATTR_WEIGHTS_JSON",
    "/var/cache/cluster_engine/attr_weights.json",
))


def calibrate(category_ids: list[int]) -> dict:
    """Compute per-attr blocking weights from cluster data + Gemini attrs."""
    eng = get_engine()

    # Load cluster assignments + Gemini attrs
    placeholders = ",".join(str(int(c)) for c in category_ids)
    sql = f"""
        SELECT
            pc.cluster_gid,
            pc.pk_id,
            n.category_id,
            o.ocr_attrs
        FROM mpstats_product_clusters pc
        JOIN mpstats_products p ON p.pk_id = pc.pk_id
        JOIN mpstats_niches n ON n.niche_key = CAST(p.niche_key AS SIGNED)
        LEFT JOIN mpstats_product_ocr o ON o.pk_id = pc.pk_id
        WHERE n.category_id IN ({placeholders})
          AND pc.cluster_gid IS NOT NULL
    """
    rows = []
    with eng.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()

    log.info(f"Loaded {len(rows)} product-cluster rows")

    # Group by (category, cluster_gid)
    clusters: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cat = int(r["category_id"])
        gid = int(r["cluster_gid"])
        attrs_raw = r["ocr_attrs"]
        if isinstance(attrs_raw, str):
            try:
                attrs_raw = json.loads(attrs_raw)
            except (json.JSONDecodeError, TypeError):
                attrs_raw = {}
        if not attrs_raw or not isinstance(attrs_raw, dict):
            continue
        # Normalize flat Gemini attrs
        first_val = next(iter(attrs_raw.values()), None)
        if not isinstance(first_val, dict):
            normalized = {}
            for k, v in attrs_raw.items():
                if v is None:
                    continue
                normalized[k] = v
            attrs_raw = normalized
        else:
            attrs_raw = {k: v.get("value") for k, v in attrs_raw.items() if v.get("value") is not None}

        clusters[cat][gid].append(attrs_raw)

    result = {}
    for cat in sorted(clusters):
        cat_clusters = clusters[cat]
        # Filter to good-size clusters
        good = {gid: items for gid, items in cat_clusters.items()
                if MIN_CLUSTER_SIZE <= len(items) <= MAX_CLUSTER_SIZE}
        log.info(f"Cat {cat}: {len(good)} clusters with size {MIN_CLUSTER_SIZE}-{MAX_CLUSTER_SIZE}")

        if not good:
            continue

        # For each attr key, compute uniformity across clusters
        attr_stats: dict[str, list[float]] = defaultdict(list)
        attr_types: dict[str, str] = {}

        for gid, items in good.items():
            # Collect values per attr in this cluster
            attr_vals: dict[str, list] = defaultdict(list)
            for item_attrs in items:
                for k, v in item_attrs.items():
                    attr_vals[k].append(v)

            for k, vals in attr_vals.items():
                if len(vals) < 3:
                    continue  # not enough data
                # Determine type
                is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
                if is_numeric:
                    attr_types.setdefault(k, "numeric")
                    # Uniformity for numeric: fraction in largest ±15% group
                    sorted_vals = sorted(float(v) for v in vals)
                    best_sz = 0
                    for anchor in sorted_vals:
                        group = [v for v in sorted_vals if anchor * 0.85 <= v <= anchor * 1.15]
                        best_sz = max(best_sz, len(group))
                    uniformity = best_sz / len(vals)
                else:
                    attr_types.setdefault(k, "string")
                    # Uniformity for string: fraction of most common value
                    ctr = Counter(str(v).lower().strip() for v in vals)
                    top_cnt = ctr.most_common(1)[0][1]
                    uniformity = top_cnt / len(vals)

                attr_stats[k].append(uniformity)

        # Average uniformity across clusters
        cat_key = f"cat_{cat}"
        cat_result = {}
        for k, uniformities in sorted(attr_stats.items()):
            if len(uniformities) < 3:
                continue  # too few clusters to judge
            avg_u = sum(uniformities) / len(uniformities)
            is_block = avg_u >= BLOCK_THRESHOLD
            cat_result[k] = {
                "block": is_block,
                "type": attr_types.get(k, "unknown"),
                "uniformity": round(avg_u, 3),
                "n_clusters": len(uniformities),
            }
            log.info(f"  {cat_key}.{k}: uniformity={avg_u:.3f} type={attr_types.get(k)} → {'BLOCK' if is_block else 'skip'}")

        result[cat_key] = cat_result

    return result


def load_weights() -> dict:
    """Load cached weights from JSON file."""
    if OUTPUT_PATH.exists():
        try:
            return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_blocking_attrs(category_id: int) -> dict[str, dict]:
    """Get blocking config for a category. Returns {attr_name: {block, type, uniformity}}."""
    weights = load_weights()
    return weights.get(f"cat_{category_id}", {})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", type=int, nargs="+", default=[37, 42, 44, 60])
    args = ap.parse_args()

    result = calibrate(args.categories)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
