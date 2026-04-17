"""
ClickHouse writer — write cluster_moves records compatible with the viewer.

Supports three backends (same as your viewer):
  1. clickhouse_connect
  2. clickhouse_driver
  3. HTTP interface via httpx
"""
from __future__ import annotations
import logging
import json
from datetime import datetime

import os
import httpx

from .config import CH_HOST, CH_HTTP_PORT, CH_DB, CH_USER, CH_PASS

log = logging.getLogger("cluster_engine.ch_writer")

_ch_mode = None
_ch_client = None


def _init_ch():
    global _ch_mode, _ch_client
    if _ch_mode is not None:
        return

    # 1) clickhouse-connect (exclude localhost from proxy)
    try:
        import clickhouse_connect
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
        if "127.0.0.1" not in os.environ.get("no_proxy", ""):
            os.environ["no_proxy"] = os.environ["no_proxy"] + ",127.0.0.1,localhost"
        _ch_client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_HTTP_PORT,
            username=CH_USER, password=CH_PASS, database=CH_DB,
        )
        _ch_mode = "clickhouse_connect"
        log.info("ClickHouse: using clickhouse_connect")
        return
    except Exception:
        pass

    # 2) clickhouse-driver
    try:
        from clickhouse_driver import Client
        _ch_client = Client(
            host=CH_HOST, port=9000,
            user=CH_USER, password=CH_PASS, database=CH_DB,
        )
        _ch_mode = "clickhouse_driver"
        log.info("ClickHouse: using clickhouse_driver (native)")
        return
    except Exception:
        pass

    # 3) HTTP
    _ch_mode = "http"
    _ch_client = None
    log.info("ClickHouse: using HTTP interface")


def _exec_sql(sql: str):
    """Execute raw SQL on ClickHouse."""
    _init_ch()

    if _ch_mode == "clickhouse_connect":
        _ch_client.command(sql)
    elif _ch_mode == "clickhouse_driver":
        _ch_client.execute(sql)
    elif _ch_mode == "http":
        url = f"http://{CH_HOST}:{CH_HTTP_PORT}"
        with httpx.Client(proxy=None, timeout=30.0) as _hc:
            r = _hc.post(
                url,
                params={"database": CH_DB},
                content=sql.encode("utf-8"),
                auth=(CH_USER, CH_PASS) if CH_USER else None,
            )
        if r.status_code >= 400:
            raise RuntimeError(f"ClickHouse HTTP error {r.status_code}: {r.text[:500]}")
    else:
        raise RuntimeError("ClickHouse not initialized")


def _escape(s: str) -> str:
    """Escape a string for ClickHouse SQL."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def write_cluster_moves(moves: list[dict], batch_size: int = 50) -> int:
    """
    Write cluster_moves records to ClickHouse.

    Expected fields per move:
        pk_id, nm_id, product_name, old_cluster, new_cluster,
        target_action, new_cluster_title, niche_key_int,
        reason, status, created_by, confidence_score,
        idempotency_key, created_at

    Returns number of records written.
    """
    if not moves:
        return 0

    _init_ch()
    total = 0

    for i in range(0, len(moves), batch_size):
        batch = moves[i:i + batch_size]
        rows = []

        for m in batch:
            pk_id = int(m.get("pk_id", 0))
            nm_id = int(m.get("nm_id", pk_id))
            product_name = _escape(str(m.get("product_name", ""))[:200])
            old_cluster = int(m.get("old_cluster", 0))
            new_cluster = int(m.get("new_cluster", 0))
            target_action = _escape(str(m.get("target_action", ""))[:50])
            new_cluster_title = _escape(str(m.get("new_cluster_title", ""))[:200])
            niche_key_int = int(m.get("niche_key_int", 0))
            reason = _escape(str(m.get("reason", ""))[:500])
            status = _escape(str(m.get("status", "pending"))[:20])
            created_by = _escape(str(m.get("created_by", "cluster_engine_v2"))[:50])
            confidence = round(float(m.get("confidence_score", 0.5)), 4)
            anomaly_flags = int(m.get("anomaly_flags", 0)) & 0xFF
            idem_key = _escape(str(m.get("idempotency_key", ""))[:100])

            rows.append(
                f"({pk_id}, {nm_id}, '{product_name}', "
                f"{old_cluster}, {new_cluster}, '{target_action}', "
                f"'{new_cluster_title}', {niche_key_int}, "
                f"'{reason}', '{status}', '{created_by}', "
                f"{confidence}, {anomaly_flags}, '{idem_key}', now())"
            )

        values_str = ",\n".join(rows)
        sql = f"""
            INSERT INTO {CH_DB}.cluster_moves
                (pk_id, nm_id, product_name,
                 old_cluster, new_cluster, target_action,
                 new_cluster_title, niche_key_int,
                 reason, status, created_by,
                 confidence_score, anomaly_flags, idempotency_key, created_at)
            VALUES
                {values_str}
        """

        try:
            _exec_sql(sql)
            total += len(batch)
        except Exception as e:
            log.error(f"Failed to write batch {i}-{i+len(batch)}: {e}")
            # Try individual inserts for the batch
            for row_sql in rows:
                try:
                    single = f"""
                        INSERT INTO {CH_DB}.cluster_moves
                            (pk_id, nm_id, product_name,
                             old_cluster, new_cluster, target_action,
                             new_cluster_title, niche_key_int,
                             reason, status, created_by,
                             confidence_score, idempotency_key, created_at)
                        VALUES {row_sql}
                    """
                    _exec_sql(single)
                    total += 1
                except Exception as e2:
                    log.error(f"Individual insert failed: {e2}")

        if total > 0 and total % 1000 == 0:
            log.info(f"  Written {total}/{len(moves)} moves to ClickHouse")

    log.info(f"Written {total} cluster_moves to ClickHouse")
    return total


def ensure_table_exists():
    """Create cluster_moves table if it doesn't exist."""
    sql = f"""
        CREATE TABLE IF NOT EXISTS {CH_DB}.cluster_moves (
            id              UInt64 DEFAULT generateUUIDv4(),
            pk_id           Int64,
            nm_id           Int64 DEFAULT 0,
            product_name    String DEFAULT '',
            old_cluster     Int64 DEFAULT 0,
            new_cluster     Int64 DEFAULT 0,
            target_action   String DEFAULT '',
            new_cluster_title String DEFAULT '',
            niche_key_int   Int64 DEFAULT 0,
            reason          String DEFAULT '',
            status          String DEFAULT 'pending',
            created_by      String DEFAULT '',
            reviewed_by     String DEFAULT '',
            confidence_score Float32 DEFAULT 0.5,
            idempotency_key String DEFAULT '',
            reject_reason   String DEFAULT '',
            anomaly_flags   UInt8 DEFAULT 0,
            created_at      DateTime DEFAULT now(),
            reviewed_at     DateTime DEFAULT toDateTime(0)
        )
        ENGINE = MergeTree()
        ORDER BY (old_cluster, pk_id, created_at)
    """
    try:
        _exec_sql(sql)
        log.info("cluster_moves table ensured")
    except Exception as e:
        log.warning(f"Could not create cluster_moves table: {e}")

    try:
        _exec_sql(
            f"ALTER TABLE {CH_DB}.cluster_moves "
            f"ADD COLUMN IF NOT EXISTS anomaly_flags UInt8 DEFAULT 0"
        )
    except Exception as e:
        log.warning(f"Could not add anomaly_flags column: {e}")
