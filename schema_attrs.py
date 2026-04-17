"""
Schema-based attribute extraction from product names and OCR text.

Supports two levels:
  1. Per-niche schema  — /var/cache/cluster_engine/schemas/niche_{key}.json  (preferred)
  2. Per-category schema — /var/cache/cluster_engine/schemas/category_{id}_v2.json (fallback)

Niche schemas are auto-generated on first use via LLM (one call per niche, cached).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import httpx

from .models import Product
from .config import OPENROUTER_API_KEY, OPENROUTER_BASE
from .value_canon import canon_numeric

log = logging.getLogger("cluster_engine.schema_attrs")


def _dedup_against_gemini(attrs: dict, gemini_keys: set) -> dict:
    """
    Drop regex-extracted attrs whose canonical (value, dimension) matches
    any Gemini-extracted attr. Prevents duplicate "Дальн=50м, Дальн=50M"
    in scoring reasons and stops noisy regex from competing with Gemini.

    Only drops when:
      - both have a real SI dimension (not bare number)
      - canonical values match within 5%
    """
    if not gemini_keys:
        return attrs

    # Hint dimension from key suffix: "дальность_м" → "м", "мощность_вт" → "вт"
    def _hint(key: str) -> str | None:
        m = re.search(r"_([a-zа-я]+)$", key.lower())
        return m.group(1) if m else None

    gemini_canon = {}
    for k in gemini_keys:
        if k not in attrs:
            continue
        v = attrs[k]
        raw = v.get("value") if isinstance(v, dict) else v
        c = canon_numeric(raw, hint_unit=_hint(k))
        if c and c[1]:
            gemini_canon[k] = c

    if not gemini_canon:
        return attrs

    out = dict(attrs)
    for k in list(attrs.keys()):
        if k in gemini_keys:
            continue
        v = attrs[k]
        raw = v.get("value") if isinstance(v, dict) else v
        c = canon_numeric(raw)
        if c is None or not c[1]:
            continue
        for gc in gemini_canon.values():
            if c[1] != gc[1]:
                continue
            denom = max(abs(c[0]), abs(gc[0]), 1e-9)
            if abs(c[0] - gc[0]) / denom < 0.05:
                out.pop(k, None)
                break
    return out

SCHEMA_DIR = Path("/var/cache/cluster_engine/schemas")
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_LLM_MODEL = "google/gemini-2.0-flash-001"


# ── Loading ──────────────────────────────────────────────────────────

def load_schema_for_niche(niche_key: int) -> Optional[dict]:
    """Load cached niche-level schema."""
    path = SCHEMA_DIR / f"niche_{niche_key}.json"
    if path.exists():
        try:
            schema = json.loads(path.read_text("utf-8"))
            seps = schema.get("key_separators", [])
            log.info(f"Loaded niche schema {niche_key}: {len(seps)} separators")
            return schema
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to load niche schema {path}: {e}")
    return None


def load_schema(category_id: int) -> Optional[dict]:
    """Load cached schema for category. Returns None if not cached."""
    for suffix in [f"category_{category_id}_v2.json", f"category_{category_id}.json"]:
        path = SCHEMA_DIR / suffix
        if path.exists():
            try:
                schema = json.loads(path.read_text("utf-8"))
                seps = schema.get("key_separators", [])
                log.info(f"Loaded schema for category {category_id}: {len(seps)} separators ({path.name})")
                return schema
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Failed to load schema {path}: {e}")
    return None


# ── Generation ───────────────────────────────────────────────────────

_NICHE_SCHEMA_PROMPT = """Ты — эксперт по группировке товаров маркетплейса Wildberries.

Ниша: "{niche_name}" (niche_key={niche_key})

Примеры названий товаров из этой ниши:
{samples_str}

Некоторые товары имеют [OCR с фото] — текст распознанный с фотографии карточки. Используй его для поиска дополнительных атрибутов, которых может не быть в названии.

ЗАДАЧА: Проанализируй названия (и OCR где есть) и создай схему КЛЮЧЕВЫХ АТРИБУТОВ-РАЗДЕЛИТЕЛЕЙ — по которым товары РАЗДЕЛЯЮТСЯ на разные кластеры (группы одинаковых товаров).

ГЛАВНОЕ ПРАВИЛО: Найди ВСЕ числа в названиях. Каждое число рядом с контекстом — потенциальный разделитель.

ТИПЫ РАЗДЕЛИТЕЛЕЙ — ищи ВСЕ:
- Числовые с единицами: количество (шт), размер (мм, см), объём (мл, л), вес (г, кг), мощность (вт), длина (м) и т.д.
- Числовые БЕЗ единиц: диоптрии (-2.50), оптическая сила (+1.50), концентрация (0.5%), номер тона, индекс
- Числа с минусом/плюсом: -2.50, +3.00 — НЕ игнорируй, это числовые атрибуты товара
- Дробные числа: 8.5, 14.0 — определи что это (радиус, диаметр, коэффициент) и включи
- Текстовые модификаторы: одинарный/двойной/тройной, мини/стандарт/макси, малый/средний/большой, левый/правый, мужской/женский
- Тип/подтип товара: если в нише разные подтипы (однодневные/месячные, помповая/рычажная)
- Материал: если разные материалы = разные товары
- Стандарт/типоразмер: PH1/PH2/PZ/TORX, 1/4"/1/2"

"-2.50" ≠ "-3.00" — это РАЗНЫЕ товары!
"8.5" ≠ "9.0" — если это характеристика товара, это РАЗНЫЕ товары!
"6шт" ≠ "12шт" — это РАЗНЫЕ товары!
"500мл" ≠ "1л" — это РАЗНЫЕ товары!

Для каждого атрибута дай:
1. "attr" — полное название атрибута
2. "regex" — Python regex с capturing group для значения. Case-insensitive. Должен быть ТОЧНЫМ — не захватывать чужие числа.
3. "unit" — базовая единица ("м", "л", "шт", "г") или null. Если у атрибута несколько единиц с разной шкалой (см/мм/м или мл/л/г/кг) — указывай БАЗОВУЮ.
4. "unit_factor" — словарь конверсии в базовую единицу. ОБЯЗАТЕЛЬНО если регекс ловит несколько единиц.
   Пример: {{"м": 1, "см": 0.01, "мм": 0.001}} приведёт "30 см" → 0.30 (м).
   Пример: {{"л": 1, "мл": 0.001}} приведёт "500 мл" → 0.5 (л).
   Если только одна единица → не указывай.
5. "numeric" — true/false
6. "priority" — 1 (главный разделитель) до 3
7. "label" — метка ≤6 символов
8. "example" — пример значения

🔑 КЛЮЧЕВЫЕ ПРАВИЛА REGEX:
- ВСЕ возможные написания и сокращения единиц/слов должны ловиться одним regex.
  Если в названиях встречается "штук", "шт", "шт.", "ш" — regex должен ловить ВСЕ варианты.
  Если "предметов", "предмета", "предмет", "пред", "пред.", "пр", "пр." — ВСЕ варианты.
  Используй \\b и опциональные группы:
    Плохо: (\\d+)\\s*предмет\\w*  ← пропустит "пред", "пред.", "пр."
    Хорошо: (\\d+)\\s*(?:шт\\w*|пред(?:мет\\w*)?\\b|пр\\.?\\b)
- Числа с минусом/плюсом (-2.50, +1.50) — ОБЯЗАТЕЛЬНО требуй знак: ([-+]\\d+(?:[.,]\\d+)?)
- Десятичные допускают и `.` и `,`: \\d+(?:[.,]\\d+)?
- Для атрибутов с несколькими единицами (см/мм/м) — захвати единицу второй группой и заполни unit_factor.
  Пример: "(\\d+(?:[.,]\\d+)?)\\s*(см|мм|м)\\b"  + unit_factor={{"м":1,"см":0.01,"мм":0.001}}

ОСТАЛЬНОЕ:
- Включи ВСЕ числовые атрибуты из примеров — каждое число в названии должно быть учтено
- Текстовые модификаторы (для дома/для авто, мужской/женский) — ОБЯЗАТЕЛЬНО если есть
- НЕ включай бренд, цену, маркетинговые слова
- Лучше включить лишний атрибут, чем пропустить важный

Верни ТОЛЬКО JSON:
{{
  "niche": "{niche_name}",
  "niche_key": {niche_key},
  "key_separators": [
    {{
      "attr": "Количество предметов",
      "regex": "(\\\\d+)\\\\s*(?:шт\\\\w*|пред(?:мет\\\\w*)?\\\\b|пр\\\\.?\\\\b)",
      "unit": "шт",
      "numeric": true,
      "priority": 1,
      "label": "Кол-во",
      "example": "12"
    }},
    {{
      "attr": "Длина",
      "regex": "(\\\\d+(?:[.,]\\\\d+)?)\\\\s*(см|мм|м)\\\\b",
      "unit": "м",
      "unit_factor": {{"м": 1, "см": 0.01, "мм": 0.001}},
      "numeric": true,
      "priority": 1,
      "label": "Длина",
      "example": "1.5"
    }}
  ]
}}"""


def _call_llm_sync(prompt: str) -> str:
    """Call LLM via OpenRouter."""
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": SCHEMA_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def generate_niche_schema(
    niche_key: int,
    niche_name: str,
    sample_names: list[str],
    sample_ocr: dict[str, str] = None,
    force: bool = False,
) -> Optional[dict]:
    """Generate and cache a per-niche schema via LLM.

    sample_ocr: {product_name: ocr_text} — OCR texts from product photos.
    """
    cache_path = SCHEMA_DIR / f"niche_{niche_key}.json"
    if not force and cache_path.exists():
        return load_schema_for_niche(niche_key)

    if not OPENROUTER_API_KEY:
        log.warning("No OPENROUTER_API_KEY, cannot generate niche schema")
        return None

    if len(sample_names) < 3:
        log.warning(f"Too few samples ({len(sample_names)}) for niche {niche_key}, skip schema generation")
        return None

    # Build samples with OCR
    lines = []
    for n in sample_names[:50]:
        ocr = (sample_ocr or {}).get(n, "")
        if ocr:
            lines.append(f"  - {n}  [OCR с фото: {ocr[:120]}]")
        else:
            lines.append(f"  - {n}")
    samples_str = "\n".join(lines)

    prompt = _NICHE_SCHEMA_PROMPT.format(
        niche_name=niche_name,
        niche_key=niche_key,
        samples_str=samples_str,
    )

    log.info(f"Generating niche schema via LLM for niche {niche_key} ({niche_name})...")
    try:
        raw = _call_llm_sync(prompt)
    except Exception as e:
        log.error(f"LLM call failed for niche schema {niche_key}: {e}")
        return None

    # Parse JSON
    a = raw.find("{")
    b = raw.rfind("}") + 1
    if a < 0 or b <= a:
        log.error(f"No JSON in LLM response for niche {niche_key}")
        return None

    try:
        schema = json.loads(raw[a:b])
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in niche schema {niche_key}: {e}")
        return None

    schema["niche_key"] = niche_key
    schema["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    schema["generated_by"] = SCHEMA_LLM_MODEL

    # Validate regex patterns
    valid_seps = []
    for sep in schema.get("key_separators", []):
        pattern = sep.get("regex", "")
        if not pattern:
            continue
        try:
            re.compile(pattern, re.I)
            valid_seps.append(sep)
        except re.error as e:
            log.warning(f"Invalid regex for {sep.get('attr')}: {e}. Skipping.")
    schema["key_separators"] = valid_seps

    cache_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), "utf-8")
    log.info(f"Niche schema built: {len(valid_seps)} separators for niche {niche_key}")
    return schema


# ── Extraction ───────────────────────────────────────────────────────


def extract_attrs(text: str, schema: dict) -> dict:
    """
    Apply schema regex patterns to text.
    Returns: {attr_name: {"value": ..., "unit": ..., "label": ..., "priority": ...}}
    """
    if not text or not schema:
        return {}

    result = {}
    seps = sorted(
        schema.get("key_separators", []),
        key=lambda s: s.get("priority", 99),
    )
    for sep in seps:
        attr = sep.get("attr", "")
        pattern = sep.get("regex", "")
        if not pattern:
            continue
        if attr in result:
            continue

        try:
            m = re.search(pattern, text, re.I)
        except re.error:
            continue

        if not m:
            continue

        raw_match = m.group(0)
        val = None
        for g in m.groups():
            if g is not None:
                val = g
                break
        if val is None:
            val = raw_match

        is_numeric = sep.get("numeric", False)
        if is_numeric:
            try:
                val = float(val.replace(",", "."))
            except (ValueError, AttributeError):
                continue

            # Optional unit conversion: if schema has `unit_factor` dict and the
            # regex captured a unit string (group 2), normalize value to base unit.
            # Example: regex "(\d+)\s*(см|мм|м)" with unit_factor={"м":1,"см":0.01}
            # converts "30 см" → 0.30 (in meters), "50 м" → 50.0
            unit_factor = sep.get("unit_factor")
            if unit_factor and m.lastindex and m.lastindex >= 2:
                unit_str = (m.group(2) or "").strip().lower()
                factor = unit_factor.get(unit_str)
                if factor is not None:
                    val = val * factor

        result[attr] = {
            "value": val,
            "unit": sep.get("unit"),
            "numeric": is_numeric,
            "label": sep.get("label", attr[:6]),
            "priority": sep.get("priority", 99),
            "raw_match": raw_match,
            "tolerance": sep.get("tolerance"),
        }

    return result


def extract_all(products: list[Product], category_id: int, niche_key: int = None) -> int:
    """
    Extract structured attrs from product names.

    Priority: niche schema (auto-generated per niche) > category schema (cached).
    If niche schema doesn't exist yet, generates it via LLM.
    When niche_key is None (category mode), groups products by niche and
    generates/loads schema per niche.
    """
    filled = 0

    if niche_key:
        # Single niche mode
        niches = {niche_key: products}
    else:
        # Category mode — group by niche
        niches: dict[int, list] = {}
        for p in products:
            if p.niche_key:
                niches.setdefault(p.niche_key, []).append(p)
            # Products without niche_key will be handled by category fallback below

    # Process each niche
    total_seps = 0
    for nk, niche_products in niches.items():
        schema = load_schema_for_niche(nk)
        if not schema:
            niche_name = _get_niche_name(nk)
            sample_names = list({p.name for p in niche_products})[:50]
            sample_ocr = {}
            for p in niche_products:
                if p.name in sample_ocr:
                    continue
                ocr = getattr(p, "ocr_text", "") or ""
                if ocr:
                    sample_ocr[p.name] = ocr
            schema = generate_niche_schema(nk, niche_name, sample_names, sample_ocr=sample_ocr)

        if not schema:
            # Fallback to category schema
            schema = load_schema(category_id)

        if not schema:
            continue

        total_seps = max(total_seps, len(schema.get('key_separators', [])))
        for p in niche_products:
            gemini_attrs = dict(p.ocr_attrs or {})
            attrs = extract_attrs(p.name, schema)
            ocr_text = getattr(p, "ocr_text", "") or ""
            if ocr_text:
                ocr_attrs = extract_attrs(ocr_text, schema)
                for k, v in ocr_attrs.items():
                    if k not in attrs:
                        attrs[k] = v
            for k, v in gemini_attrs.items():
                attrs[k] = v
            attrs = _dedup_against_gemini(attrs, set(gemini_attrs.keys()))
            if attrs:
                p.ocr_attrs = attrs
                filled += 1

    # Handle products without niche_key (category fallback)
    if not niche_key:
        no_niche = [p for p in products if not p.niche_key or not p.ocr_attrs]
        if no_niche:
            schema = load_schema(category_id)
            if schema:
                for p in no_niche:
                    gemini_attrs = dict(p.ocr_attrs or {})
                    attrs = extract_attrs(p.name, schema)
                    ocr_text = getattr(p, "ocr_text", "") or ""
                    if ocr_text:
                        ocr_attrs = extract_attrs(ocr_text, schema)
                        for k, v in ocr_attrs.items():
                            if k not in attrs:
                                attrs[k] = v
                    for k, v in gemini_attrs.items():
                        attrs[k] = v
                    attrs = _dedup_against_gemini(attrs, set(gemini_attrs.keys()))
                    if attrs:
                        p.ocr_attrs = attrs
                        filled += 1

    log.info(f"Schema attrs: {filled}/{len(products)} products got attrs "
             f"({len(niches)} niches, max {total_seps} separators)")
    return filled


def _get_niche_name(niche_key: int) -> str:
    """Get niche name from MySQL."""
    try:
        from .db import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT name FROM mpstats_niches WHERE niche_key = :nk"),
                {"nk": niche_key},
            ).mappings().first()
        return row["name"] if row else f"niche_{niche_key}"
    except Exception:
        return f"niche_{niche_key}"
