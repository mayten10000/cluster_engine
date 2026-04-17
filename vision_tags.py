"""
Vision-LLM function tagging for products.

Two-phase like schema_attrs:
  1. ensure_function_list(category_id, ...) — one text-LLM call to build a
     closed list of functional types for the category, cached on disk.
  2. get_vision_tags(products, category_id, ...) — per-product vision call to
     gemini-flash that classifies the product image into one of those types,
     cached in MySQL table product_vision_tags.

The function tag is a domain-meaningful slug (e.g. "crimper", "stripper",
"cutter", "knife"). Equality comparison of slugs is the blocking signal in
scoring — products with different functions cannot be in the same cluster.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import text

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE
from .models import Product

log = logging.getLogger("cluster_engine.vision_tags")

VISION_MODEL = "google/gemini-2.0-flash-001"
FUNC_DIR = Path("/var/cache/cluster_engine/vision_functions")
FUNC_DIR.mkdir(parents=True, exist_ok=True)

VISION_TIMEOUT = 60.0
VISION_WORKERS = 8
MAX_FUNC_LIST_SAMPLES = 80


# ── Function list (per-category) ─────────────────────────────────────

_FUNC_LIST_PROMPT = """Ты эксперт по товарам маркетплейса. Категория «{category_name}» (id={category_id}).

Ниже названия товаров из этой категории:
{samples_str}

Задача: выдели исчерпывающий список **функциональных типов** товаров. Функция = что предмет ДЕЛАЕТ, а не как выглядит. Например в категории «Электроинструмент»: crimper (обжимает клеммы), stripper (снимает изоляцию с провода), cutter (режет провод), knife (нож для резки). Обжимной кримпер и стриппер — РАЗНЫЕ функции, даже если внешне похожи.

Правила:
- 3..15 типов. Если в категории один тип — верни 1 (`generic`).
- Slug: lowercase, ASCII, snake_case (например `wire_stripper`).
- description: 1 строка по-русски.
- examples: 2-4 ключевых слова из названий, по которым этот тип распознаётся.
- Если функцию нельзя надёжно определить из картинки — НЕ включай.

Верни СТРОГО JSON без markdown:
{{
  "functions": [
    {{"slug": "wire_stripper", "description": "снимает изоляцию с провода", "examples": ["стриппер", "съёмник изоляции"]}},
    ...
  ]
}}"""


def _func_list_path(category_id: int) -> Path:
    return FUNC_DIR / f"cat_{category_id}.json"


def load_function_list(category_id: int) -> Optional[list[dict]]:
    p = _func_list_path(category_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8")).get("functions")
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load function list {p}: {e}")
        return None


def _call_text_llm(prompt: str) -> str:
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
            },
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ensure_function_list(
    category_id: int,
    category_name: str,
    sample_names: list[str],
    force: bool = False,
) -> Optional[list[dict]]:
    """Build the closed list of functional types for the category. Cached on disk."""
    if not force:
        cached = load_function_list(category_id)
        if cached:
            return cached

    if not OPENROUTER_API_KEY:
        log.warning("No OPENROUTER_API_KEY, cannot generate function list")
        return None

    if len(sample_names) < 3:
        log.warning(f"Too few samples ({len(sample_names)}) for cat {category_id}")
        return None

    samples_str = "\n".join(f"  - {n}" for n in sample_names[:MAX_FUNC_LIST_SAMPLES])
    prompt = _FUNC_LIST_PROMPT.format(
        category_name=category_name,
        category_id=category_id,
        samples_str=samples_str,
    )

    log.info(f"Generating function list via LLM for cat {category_id} ({category_name})...")
    try:
        raw = _call_text_llm(prompt)
    except Exception as e:
        log.error(f"LLM call failed for function list cat {category_id}: {e}")
        return None

    a, b = raw.find("{"), raw.rfind("}") + 1
    if a < 0 or b <= a:
        log.error(f"No JSON in LLM response for function list cat {category_id}")
        return None
    try:
        parsed = json.loads(raw[a:b])
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in function list cat {category_id}: {e}")
        return None

    funcs = parsed.get("functions") or []
    if not funcs:
        log.error(f"Empty functions list for cat {category_id}")
        return None

    payload = {
        "category_id": category_id,
        "category_name": category_name,
        "model": VISION_MODEL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "functions": funcs,
    }
    _func_list_path(category_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
    )
    log.info(f"Function list saved for cat {category_id}: {len(funcs)} types")
    return funcs


# ── Per-product vision tagging ────────────────────────────────────────

def _ensure_table(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS product_vision_tags (
                pk_id BIGINT PRIMARY KEY,
                category_id INT NOT NULL,
                function_slug VARCHAR(64) NOT NULL,
                raw_json TEXT,
                model VARCHAR(64),
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_vt_category (category_id),
                INDEX idx_vt_function (function_slug)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))
        conn.commit()


def load_vision_tags(pk_ids: list[int]) -> dict[int, dict]:
    if not pk_ids:
        return {}
    from ce_ideal.db import get_engine
    engine = get_engine()
    out: dict[int, dict] = {}
    CHUNK = 1000
    with engine.connect() as conn:
        for i in range(0, len(pk_ids), CHUNK):
            chunk = pk_ids[i:i + CHUNK]
            placeholders = ", ".join(str(p) for p in chunk)
            rows = conn.execute(text(
                f"SELECT pk_id, category_id, function_slug, raw_json, model "
                f"FROM product_vision_tags WHERE pk_id IN ({placeholders})"
            )).fetchall()
            for row in rows:
                raw = {}
                if row[3]:
                    try:
                        raw = json.loads(row[3])
                    except json.JSONDecodeError:
                        raw = {}
                out[row[0]] = {
                    "category_id": row[1],
                    "function": row[2],
                    "raw": raw,
                    "model": row[4],
                }
    return out


def _save_tags(records: list[tuple[int, int, str, dict]]) -> None:
    if not records:
        return
    from ce_ideal.db import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        for pk_id, cat_id, slug, raw in records:
            conn.execute(text("""
                INSERT INTO product_vision_tags (pk_id, category_id, function_slug, raw_json, model)
                VALUES (:pk, :cat, :slug, :raw, :model)
                ON DUPLICATE KEY UPDATE
                    category_id   = VALUES(category_id),
                    function_slug = VALUES(function_slug),
                    raw_json      = VALUES(raw_json),
                    model         = VALUES(model),
                    computed_at   = CURRENT_TIMESTAMP
            """), {
                "pk": pk_id,
                "cat": cat_id,
                "slug": slug,
                "raw": json.dumps(raw, ensure_ascii=False),
                "model": VISION_MODEL,
            })
        conn.commit()


def _build_image_url(thumb_url: str) -> Optional[str]:
    """Use the first photo, upscaled. Mirrors ce_ideal.image_embed._build_photo_urls."""
    if not thumb_url:
        return None
    url = thumb_url.replace("//", "https://", 1) if thumb_url.startswith("//") else thumb_url
    return url.replace("/c246x328/", "/c516x688/")


_VISION_PROMPT_TEMPLATE = """На картинке товар из категории «{category_name}». Название: «{name}».

Доступные функциональные типы:
{func_list_str}

Задача: посмотри на ИЗОБРАЖЕНИЕ и выбери ОДИН slug, который описывает функцию предмета (что он делает). Опирайся на форму, рабочие части, элементы управления — не на цвет и не на упаковку. Если ни один не подходит — верни `unknown`.

Верни СТРОГО JSON без markdown:
{{"function": "<slug>", "confidence": 0.0..1.0, "reason": "одна короткая фраза"}}"""


def _vision_call(image_url: str, name: str, category_name: str, func_list_str: str) -> Optional[dict]:
    prompt = _VISION_PROMPT_TEMPLATE.format(
        category_name=category_name,
        name=name,
        func_list_str=func_list_str,
    )
    try:
        with httpx.Client(timeout=VISION_TIMEOUT) as client:
            r = client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]}],
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
            )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.debug(f"Vision call failed for {image_url}: {e}")
        return None

    a, b = raw.find("{"), raw.rfind("}") + 1
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(raw[a:b])
    except json.JSONDecodeError:
        return None


def get_vision_tags(
    products: list[Product],
    category_id: int,
    category_name: str,
    sample_names: Optional[list[str]] = None,
    force: bool = False,
) -> dict[int, dict]:
    """
    Return {pk_id: {function, raw, ...}} for products. Vision-LLM is invoked
    only for products without a cached tag.
    """
    if not products:
        return {}

    func_list = ensure_function_list(
        category_id,
        category_name,
        sample_names or [p.name for p in products[:MAX_FUNC_LIST_SAMPLES]],
        force=force,
    )
    if not func_list:
        log.warning(f"No function list for cat {category_id}, skipping vision tagging")
        return {}

    valid_slugs = {f["slug"] for f in func_list}
    func_list_str = "\n".join(
        f"  - {f['slug']}: {f['description']} (примеры: {', '.join(f.get('examples', []))})"
        for f in func_list
    )

    from ce_ideal.db import get_engine
    _ensure_table(get_engine())

    pk_ids = [p.pk_id for p in products]
    cached = {} if force else load_vision_tags(pk_ids)
    log.info(f"vision_tags: {len(cached)}/{len(pk_ids)} cached, calling vision for the rest")

    to_call = [p for p in products if p.pk_id not in cached]
    new_records: list[tuple[int, int, str, dict]] = []

    def _process(p: Product) -> Optional[tuple[int, str, dict]]:
        url = _build_image_url(p.thumb_url)
        if not url:
            return None
        result = _vision_call(url, p.name, category_name, func_list_str)
        if not result:
            return None
        slug = str(result.get("function", "")).strip().lower() or "unknown"
        if slug != "unknown" and slug not in valid_slugs:
            log.debug(f"vision returned unknown slug '{slug}' for pk={p.pk_id}, coercing to unknown")
            slug = "unknown"
        return p.pk_id, slug, result

    if to_call:
        with ThreadPoolExecutor(max_workers=VISION_WORKERS) as ex:
            futures = {ex.submit(_process, p): p for p in to_call}
            done = 0
            for fut in as_completed(futures):
                p = futures[fut]
                res = fut.result()
                done += 1
                if done % 50 == 0:
                    log.info(f"  vision_tags progress: {done}/{len(to_call)}")
                if res is None:
                    continue
                pk_id, slug, raw = res
                new_records.append((pk_id, category_id, slug, raw))
                cached[pk_id] = {
                    "category_id": category_id,
                    "function": slug,
                    "raw": raw,
                    "model": VISION_MODEL,
                }
        _save_tags(new_records)
        log.info(f"vision_tags: tagged {len(new_records)} new products")

    return cached
