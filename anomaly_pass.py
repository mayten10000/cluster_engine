from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Iterable

from .models import Product, ClusterResult
from .scoring import _canon_key
from .value_canon import canon_numeric

log = logging.getLogger("cluster_engine.anomaly_pass")

SMALL_CLUSTER_SIZE = 2
DISTANCE_THRESHOLD = 0.15
MIN_CLUSTER_FOR_MEDIAN = 3


def _numeric_medians(members: list[Product]) -> dict[str, float]:
    attr_vals: dict[str, list[float]] = {}
    for cp in members:
        if not cp.ocr_attrs:
            continue
        for aname, adata in cp.ocr_attrs.items():
            if not isinstance(adata, dict):
                continue
            priority = adata.get("priority", 99)
            if priority > 2:
                continue
            if not adata.get("numeric", False):
                continue
            cn = canon_numeric(adata.get("value"))
            if cn is None:
                continue
            attr_vals.setdefault(_canon_key(aname), []).append(cn[0])
    medians: dict[str, float] = {}
    for k, vs in attr_vals.items():
        if len(vs) >= 2:
            vs.sort()
            medians[k] = vs[len(vs) // 2]
    return medians


def _product_numeric(product: Product) -> dict[str, float]:
    out: dict[str, float] = {}
    if not product.ocr_attrs:
        return out
    for aname, adata in product.ocr_attrs.items():
        if not isinstance(adata, dict):
            continue
        if adata.get("priority", 99) > 2:
            continue
        if not adata.get("numeric", False):
            continue
        cn = canon_numeric(adata.get("value"))
        if cn is None:
            continue
        out[_canon_key(aname)] = cn[0]
    return out


def _max_log_distance(product_vals: dict[str, float], medians: dict[str, float]) -> tuple[float, str | None, int]:
    worst = 0.0
    worst_key: str | None = None
    compared = 0
    for k, mv in medians.items():
        pv = product_vals.get(k)
        if pv is None or pv <= 0 or mv <= 0:
            continue
        d = abs(math.log(pv / mv))
        compared += 1
        if d > worst:
            worst = d
            worst_key = k
    return worst, worst_key, compared


def apply_anomaly_pass(
    products: Iterable[Product],
    clusters: dict[int, ClusterResult],
    *,
    pk_to_product: dict[int, Product] | None = None,
    threshold: float = DISTANCE_THRESHOLD,
) -> dict:
    if pk_to_product is None:
        pk_to_product = {p.pk_id: p for p in products}

    small_grey = 0
    attr_grey = 0
    checked = 0

    for cl in clusters.values():
        members = [pk_to_product[pk] for pk in cl.product_ids if pk in pk_to_product]
        size = len(members)
        if size == 0:
            continue

        if size <= SMALL_CLUSTER_SIZE:
            for p in members:
                if p.decision == "ok":
                    p.decision = "grey"
                    p.reason = (p.reason + " | " if p.reason else "") + f"[ANOMALY-SMALL] cluster_size={size}"
                    small_grey += 1
            continue

        if size < MIN_CLUSTER_FOR_MEDIAN:
            continue

        for p in members:
            if p.decision != "ok":
                continue
            others = [m for m in members if m.pk_id != p.pk_id]
            medians = _numeric_medians(others)
            if not medians:
                continue
            product_vals = _product_numeric(p)
            if not product_vals:
                continue
            worst, worst_key, compared = _max_log_distance(product_vals, medians)
            if compared == 0:
                continue
            checked += 1
            if worst > threshold:
                p.decision = "grey"
                p.reason = (p.reason + " | " if p.reason else "") + (
                    f"[ANOMALY-ATTR] {worst_key}: {product_vals.get(worst_key):.3g}"
                    f" vs median {medians[worst_key]:.3g} (log_dist={worst:.2f})"
                )
                attr_grey += 1

    log.info(
        f"  ANOMALY-PASS: checked {checked} ok-products,"
        f" flagged {attr_grey} by attr distance, {small_grey} from small clusters (size≤{SMALL_CLUSTER_SIZE})"
    )
    return {
        "anomaly_small_grey": small_grey,
        "anomaly_attr_grey": attr_grey,
        "anomaly_checked": checked,
    }
