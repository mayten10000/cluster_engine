#!/usr/bin/env python3
"""
build_synonyms.py — собирает все уникальные ключи атрибутов из БД,
отправляет в LLM батчами, строит synonym_dict.json.

Использование:
    python3 -m cluster_engine_v2.build_synonyms [--category_id=37] [--force]

Результат: /var/cache/cluster_engine/synonym_dict.json
    {
        "грузоподъёмность": ["грузоподъемность_т", "нагрузка_кг", "макс_вес", ...],
        "количество_инструментов": ["кол_во_инструментов", "инструментов_шт", ...],
        ...
    }
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, "/opt")

from cluster_engine_v2.config import OPENROUTER_API_KEY, OPENROUTER_BASE

log = logging.getLogger("build_synonyms")

CACHE_DIR = Path("/var/cache/cluster_engine")
SYNONYM_PATH = CACHE_DIR / "synonym_dict.json"
BATCH_SIZE = 250  # keys per LLM call


PROMPT_TEMPLATE = """Ты — эксперт по нормализации атрибутов товаров маркетплейса.

Вот список {n} уникальных имён ключей атрибутов, извлечённых из товаров:

{keys_json}

ЗАДАЧА:
Сгруппируй синонимы — ключи которые означают одно и то же.
Для каждой группы выбери одно каноническое имя (самое понятное и короткое).

ПРАВИЛА:
- Группируй ТОЛЬКО реальные синонимы (грузоподъёмность_т = нагрузка_кг = макс_вес)
- НЕ группируй разные понятия (тип ≠ тип_резца, длина ≠ длина_шины)
- Ключи с единицами измерения → синоним без единицы (мощность_вт → мощность)
- ё/е варианты → один ключ (ёмкость = емкость)
- Если ключ уникален и нет синонимов — НЕ включай в ответ

Ответь ТОЛЬКО JSON объектом:
{{
    "каноническое_имя": ["синоним1", "синоним2", ...],
    ...
}}
Без пояснений, без markdown."""


def _call_llm(prompt: str) -> str:
    """Call LLM via OpenRouter."""
    url = f"{OPENROUTER_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.1,
    }
    with httpx.Client(proxy=None, timeout=120.0) as client:
        r = client.post(url, json=body, headers=headers)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def collect_keys(category_id: int | None = None) -> list[str]:
    """Collect unique attribute keys from DB."""
    from cluster_engine_v2.db import get_engine
    from sqlalchemy import text

    engine = get_engine()
    if category_id:
        sql = text("""
            SELECT DISTINCT o.ocr_attrs
            FROM mpstats_product_ocr o
            JOIN mpstats_products p ON p.pk_id = o.pk_id
            JOIN mpstats_niches n ON n.niche_key = p.subject_id
            WHERE n.category_id = :cid AND o.ocr_attrs IS NOT NULL AND o.ocr_attrs != ''
        """)
        params = {"cid": category_id}
    else:
        sql = text("""
            SELECT ocr_attrs FROM mpstats_product_ocr
            WHERE ocr_attrs IS NOT NULL AND ocr_attrs != ''
        """)
        params = {}

    all_keys = set()
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    for r in rows:
        try:
            d = json.loads(r[0]) if isinstance(r[0], str) else r[0]
            if isinstance(d, dict):
                all_keys.update(d.keys())
        except (json.JSONDecodeError, TypeError):
            pass

    # Exclude trivial keys
    skip = {"бренд", "цвет", "материал", "гарантия", "страна", "упаковка", "артикул"}
    return sorted(all_keys - skip)


def build_synonyms(keys: list[str], existing: dict | None = None) -> dict[str, list[str]]:
    """Send keys to LLM in batches, merge results."""
    result = dict(existing) if existing else {}

    # Keys already mapped — skip them
    already_mapped = set()
    for canonical, syns in result.items():
        already_mapped.add(canonical)
        already_mapped.update(syns)

    new_keys = [k for k in keys if k not in already_mapped]
    if not new_keys:
        log.info("All keys already mapped, nothing to do")
        return result

    log.info(f"Processing {len(new_keys)} new keys in batches of {BATCH_SIZE}")

    for i in range(0, len(new_keys), BATCH_SIZE):
        batch = new_keys[i:i + BATCH_SIZE]
        log.info(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} keys...")

        prompt = PROMPT_TEMPLATE.format(
            n=len(batch),
            keys_json=json.dumps(batch, ensure_ascii=False, indent=2),
        )

        try:
            raw = _call_llm(prompt)
            # Clean markdown wrapper
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            batch_result = json.loads(text)
            if not isinstance(batch_result, dict):
                log.warning(f"  LLM returned {type(batch_result)}, skipping batch")
                continue

            # Merge into result
            for canonical, synonyms in batch_result.items():
                if not isinstance(synonyms, list):
                    continue
                canonical = canonical.strip().lower()
                synonyms = [s.strip().lower() for s in synonyms if isinstance(s, str)]
                if canonical in result:
                    existing_syns = set(result[canonical])
                    existing_syns.update(synonyms)
                    result[canonical] = sorted(existing_syns)
                else:
                    result[canonical] = sorted(set(synonyms))

            log.info(f"  Got {len(batch_result)} groups from this batch")

        except Exception as e:
            log.error(f"  Batch failed: {e}")
            continue

        time.sleep(1)  # rate limit courtesy

    return result


def save_synonyms(synonyms: dict[str, list[str]]) -> Path:
    """Save to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SYNONYM_PATH.write_text(
        json.dumps(synonyms, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    total_keys = sum(len(v) for v in synonyms.values()) + len(synonyms)
    log.info(f"Saved {len(synonyms)} groups ({total_keys} total keys) → {SYNONYM_PATH}")
    return SYNONYM_PATH


def load_synonyms() -> dict[str, list[str]]:
    """Load existing synonym dict (or empty)."""
    if SYNONYM_PATH.exists():
        try:
            return json.loads(SYNONYM_PATH.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def build_reverse_map(synonyms: dict[str, list[str]]) -> dict[str, str]:
    """Build reverse lookup: any variant → canonical key."""
    rev = {}
    for canonical, syns in synonyms.items():
        rev[canonical] = canonical
        for s in syns:
            rev[s] = canonical
    return rev


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build synonym dictionary for attribute keys")
    parser.add_argument("--category_id", type=int, help="Limit to category")
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    existing = {} if args.force else load_synonyms()
    if existing:
        log.info(f"Loaded existing: {len(existing)} groups")

    keys = collect_keys(args.category_id)
    log.info(f"Collected {len(keys)} unique attribute keys")

    synonyms = build_synonyms(keys, existing)
    save_synonyms(synonyms)

    # Print summary
    rev = build_reverse_map(synonyms)
    log.info(f"Reverse map: {len(rev)} keys → {len(synonyms)} canonical forms")


if __name__ == "__main__":
    main()
