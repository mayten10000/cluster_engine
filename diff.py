"""
Diff — compare old vs new cluster assignments.

Generates cluster_moves records for ClickHouse (compatible with viewer).
Detects: moves, new clusters, quarantined products.
"""
from __future__ import annotations
import logging
import uuid
from datetime import datetime

from .models import Product, ClusterResult

log = logging.getLogger("cluster_engine.diff")


def generate_moves(
    products: list[Product],
    clusters: dict[int, ClusterResult],
    created_by: str = "cluster_ideal",
) -> list[dict]:
    """
    Compare old_cluster_gid vs new_cluster_gid for each product.
    Generate cluster_moves records for products that changed.

    Returns list of dicts ready for ClickHouse insertion.
    """
    moves = []
    stats = {"unchanged": 0, "moved": 0, "quarantined": 0, "new_assignment": 0}

    # Build pk→product map for cluster titles
    pk_to_product = {p.pk_id: p for p in products}

    # Build cluster titles from main product name
    cluster_titles: dict[int, str] = {}
    for gid, cl in clusters.items():
        if cl.main_pk and cl.main_pk in pk_to_product:
            cluster_titles[gid] = pk_to_product[cl.main_pk].name[:200]
        elif cl.product_ids:
            first = pk_to_product.get(cl.product_ids[0])
            cluster_titles[gid] = first.name[:200] if first else f"cluster_{gid}"
        else:
            cluster_titles[gid] = f"cluster_{gid}"

    for p in products:
        old_gid = p.old_cluster_gid
        new_gid = p.new_cluster_gid

        # No change
        if old_gid == new_gid and p.decision != "quarantine":
            stats["unchanged"] += 1
            continue

        # Build full reason
        reason = p.reason or ""
        if not reason:
            if p.decision == "quarantine":
                reason = f"Не подходит кластеру (score={p.cumulative_score:.2f})"
            elif p.decision == "move" and old_gid and new_gid:
                old_title = cluster_titles.get(old_gid, f"K{old_gid}")
                new_title = cluster_titles.get(new_gid, f"K{new_gid}")
                reason = f"Перенести из \"{old_title[:60]}\" в \"{new_title[:60]}\" (score={p.cumulative_score:.2f})"
            elif p.decision == "grey":
                reason = f"Неуверенное попадание (score={p.cumulative_score:.2f})"

        # Prepend score error tags if available
        if p.score_errors:
            error_prefix = " | ".join(p.score_errors)
            reason = f"{error_prefix} | {reason}" if reason else error_prefix

        # Override product reason with full explanation
        p.reason = reason

        # Quarantined — skip (no silent flags)
        if p.decision == "quarantine":
            stats["quarantined"] += 1
            continue

        # Product moved to new_cluster=0 → create new cluster
        if p.decision == "move" and (new_gid is None or new_gid == 0):
            new_title = p.reason[:200] if p.reason else "Новый кластер"
            moves.append(_make_move(
                product=p,
                target_action="create",
                new_cluster=0,
                new_cluster_title=new_title,
                created_by=created_by,
            ))
            stats["new_assignment"] += 1
            continue

        # Product moved to different cluster
        if old_gid is not None and old_gid != new_gid and new_gid is not None:
            title = cluster_titles.get(new_gid, f"cluster_{new_gid}")
            moves.append(_make_move(
                product=p,
                target_action="move",
                new_cluster=new_gid,
                new_cluster_title=title,
                created_by=created_by,
            ))
            stats["moved"] += 1
            continue

        # New assignment (product had no cluster before)
        if old_gid is None and new_gid is not None:
            title = cluster_titles.get(new_gid, f"cluster_{new_gid}")
            moves.append(_make_move(
                product=p,
                target_action="create",
                new_cluster=new_gid,
                new_cluster_title=title,
                created_by=created_by,
            ))
            stats["new_assignment"] += 1

    log.info(
        f"Diff: {stats['unchanged']} unchanged, {stats['moved']} moved, "
        f"{stats['quarantined']} quarantined, {stats['new_assignment']} new"
    )
    return moves


def _make_move(
    product: Product,
    target_action: str,
    new_cluster: int,
    new_cluster_title: str,
    created_by: str,
) -> dict:
    """Create a cluster_moves record."""
    # Extract nm_id from thumb URL or use pk_id
    nm_id = product.pk_id  # fallback

    return {
        "pk_id": product.pk_id,
        "nm_id": nm_id,
        "product_name": product.name[:200],
        "old_cluster": product.old_cluster_gid or 0,
        "new_cluster": new_cluster,
        "target_action": target_action,
        "new_cluster_title": new_cluster_title[:200],
        "niche_key_int": product.niche_key,
        "reason": product.reason[:500] if product.reason else "",
        "status": "pending",
        "created_by": created_by,
        "confidence_score": round(product.confidence, 4),
        "anomaly_flags": int(product.anomaly_flags or 0),
        "idempotency_key": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat(),
    }
