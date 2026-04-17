"""
Database layer — read products from MySQL, write results back.
"""
from __future__ import annotations
import logging
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import (
    DB_DSN, DB_SCHEMA,
    T_PRODUCTS, T_PC, T_NICHES,
    COL_PK, COL_GID, COL_SCORE, COL_RUN, COL_MAIN, COL_NICHE,
    COL_NAME, COL_BRAND, COL_SELLER, COL_PRICE_MED, COL_THUMB,
    COL_SALES, COL_REVENUE,
)
from .models import Product

log = logging.getLogger("cluster_engine.db")

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            DB_DSN,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def load_products_by_category(category_id: int) -> list[Product]:
    """Load all products for a category (via niches)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT
                    p.{COL_PK}        AS pk_id,
                    p.{COL_NAME}      AS name,
                    COALESCE(p.{COL_BRAND}, '')  AS brand,
                    COALESCE(p.{COL_SELLER}, '') AS seller,
                    CAST(p.{COL_NICHE} AS SIGNED) AS niche_key,
                    COALESCE(p.{COL_PRICE_MED}, 0)  AS price,
                    COALESCE(p.{COL_SALES}, 0)      AS sales_1m,
                    COALESCE(p.{COL_REVENUE}, 0)     AS revenue_1m,
                    COALESCE(p.{COL_THUMB}, '')      AS thumb_url
                FROM {T_PRODUCTS} p
                JOIN {T_NICHES} n ON n.niche_key = CAST(p.{COL_NICHE} AS SIGNED)
                WHERE n.category_id = :cid
                  AND p.{COL_NAME} IS NOT NULL
                  AND p.{COL_NAME} != ''
            """),
            {"cid": category_id},
        ).mappings().all()

    products = []
    for r in rows:
        products.append(Product(
            pk_id=int(r["pk_id"]),
            name=str(r["name"] or ""),
            brand=str(r["brand"] or ""),
            seller=str(r["seller"] or ""),
            niche_key=int(r["niche_key"] or 0),
            price=float(r["price"] or 0),
            sales_1m=float(r["sales_1m"] or 0),
            revenue_1m=float(r["revenue_1m"] or 0),
            thumb_url=str(r["thumb_url"] or ""),
        ))

    log.info(f"Loaded {len(products)} products for category_id={category_id}")
    return products


def load_products_by_niche(niche_key: int) -> list[Product]:
    """Load all products for a single niche."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT
                    p.{COL_PK}        AS pk_id,
                    p.{COL_NAME}      AS name,
                    COALESCE(p.{COL_BRAND}, '')  AS brand,
                    COALESCE(p.{COL_SELLER}, '') AS seller,
                    CAST(p.{COL_NICHE} AS SIGNED) AS niche_key,
                    COALESCE(p.{COL_PRICE_MED}, 0)  AS price,
                    COALESCE(p.{COL_SALES}, 0)      AS sales_1m,
                    COALESCE(p.{COL_REVENUE}, 0)     AS revenue_1m,
                    COALESCE(p.{COL_THUMB}, '')      AS thumb_url
                FROM {T_PRODUCTS} p
                WHERE CAST(p.{COL_NICHE} AS SIGNED) = :nk
                  AND p.{COL_NAME} IS NOT NULL
                  AND p.{COL_NAME} != ''
            """),
            {"nk": niche_key},
        ).mappings().all()

    products = []
    for r in rows:
        products.append(Product(
            pk_id=int(r["pk_id"]),
            name=str(r["name"] or ""),
            brand=str(r["brand"] or ""),
            seller=str(r["seller"] or ""),
            niche_key=int(r["niche_key"] or 0),
            price=float(r["price"] or 0),
            sales_1m=float(r["sales_1m"] or 0),
            revenue_1m=float(r["revenue_1m"] or 0),
            thumb_url=str(r["thumb_url"] or ""),
        ))

    log.info(f"Loaded {len(products)} products for niche_key={niche_key}")
    return products


def load_old_assignments(product_ids: list[int]) -> dict[int, tuple[int, float]]:
    """Load current cluster assignments: pk_id → (cluster_gid, score)."""
    if not product_ids:
        return {}

    engine = get_engine()
    result = {}

    # Process in chunks to avoid too-long IN clauses
    chunk_size = 5000
    for i in range(0, len(product_ids), chunk_size):
        chunk = product_ids[i:i + chunk_size]
        placeholders = ",".join(str(int(pk)) for pk in chunk)

        with engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT {COL_PK} AS pk_id,
                           {COL_GID} AS cluster_gid,
                           {COL_SCORE} AS score
                    FROM {T_PC}
                    WHERE {COL_PK} IN ({placeholders})
                      AND {COL_GID} IS NOT NULL
                    ORDER BY {COL_RUN} DESC
                """),
            ).mappings().all()

        for r in rows:
            pk = int(r["pk_id"])
            if pk not in result:  # first row = latest run
                result[pk] = (int(r["cluster_gid"]), float(r["score"] or 0))

    log.info(f"Loaded {len(result)} old assignments")
    return result


def get_next_run_id() -> int:
    """Get next available run_id."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT COALESCE(MAX({COL_RUN}), 0) AS mx FROM {T_PC}")
        ).mappings().first()

    from .config import RUN_ID_START
    current_max = int(row["mx"]) if row else 0
    return max(current_max + 1, RUN_ID_START)


def write_assignments(
    run_id: int,
    assignments: list[dict],  # [{pk_id, cluster_gid, score, main_product, niche_key}]
    batch_size: int = 2000,
) -> int:
    """Write new cluster assignments to mpstats_product_clusters."""
    if not assignments:
        return 0

    engine = get_engine()

    # Ensure run_id exists in cluster_runs
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM mpstats_cluster_runs WHERE run_id = :rid"),
            {"rid": run_id}
        ).first()
        if not exists:
            conn.execute(
                text("INSERT INTO mpstats_cluster_runs (run_id, status, note) VALUES (:rid, 'ok', :note)"),
                {"rid": run_id, "note": f"ok: {len(assignments)} assignments"}
            )

    total = 0

    for i in range(0, len(assignments), batch_size):
        batch = assignments[i:i + batch_size]
        values = []
        for a in batch:
            values.append({
                "run_id": run_id,
                "pk_id": a["pk_id"],
                "cluster_gid": a["cluster_gid"],
                "score": round(a.get("score", 0.0), 6),
                "main_product": a.get("main_product", 0),
                "niche_key": a.get("niche_key", 0),
            })

        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {T_PC}
                        ({COL_RUN}, {COL_PK}, {COL_GID}, {COL_SCORE}, {COL_MAIN}, {COL_NICHE})
                    VALUES
                        (:run_id, :pk_id, :cluster_gid, :score, :main_product, :niche_key)
                """),
                values,
            )
        total += len(batch)

        if total % 10000 == 0:
            log.info(f"  Written {total}/{len(assignments)} assignments")

    log.info(f"Written {total} assignments with run_id={run_id}")
    return total


def get_next_cluster_gid() -> int:
    """Get next available cluster_gid (above all existing ones)."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(f"""
                SELECT COALESCE(MAX({COL_GID}), 400000) AS mx
                FROM {T_PC}
                WHERE {COL_GID} IS NOT NULL
            """)
        ).mappings().first()

    return int(row["mx"]) + 1 if row else 400001
