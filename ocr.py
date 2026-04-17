"""
OCR module — PaddleOCR-based text extraction from WB product card images.

Flow:
  1. Check cache in mpstats_product_ocr
  2. Fetch images in parallel (ThreadPool for I/O)
  3. Run PaddleOCR on pre-fetched images (sequential, not thread-safe)
  4. Save to DB every FLUSH_EVERY products (crash-safe)
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx
import numpy as np
from PIL import Image

from .config import DB_DSN, T_OCR, OCR_WORKERS
from .models import Product

log = logging.getLogger("cluster_engine.ocr")

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

_ocr = None
FLUSH_EVERY = 50  # save to DB every N products


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        log.info("Loading PaddleOCR (ru, PP-OCRv5)...")
        _ocr = PaddleOCR(lang="ru")
        log.info("PaddleOCR loaded")
    return _ocr


# ── Image fetch (I/O bound — parallelizable) ─────────────────────────

def _normalize_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if not url.startswith("http"):
        return "https://" + url
    return url


def _fetch_image_bytes(url: str) -> Optional[bytes]:
    """Fetch raw image bytes from URL."""
    try:
        with httpx.Client(timeout=8.0, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True) as client:
            r = client.get(_normalize_url(url))
        if r.status_code == 200 and r.content:
            return r.content
    except Exception as e:
        log.debug(f"Image fetch failed {url}: {e}")
    return None


def _prefetch_images(products: list[Product], workers: int = 8) -> dict[int, bytes]:
    """Download images in parallel. Returns {pk_id: image_bytes}."""
    result = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_image_bytes, p.thumb_url): p.pk_id for p in products}
        for fut in as_completed(futures):
            pk_id = futures[fut]
            try:
                data = fut.result()
                if data:
                    result[pk_id] = data
            except Exception:
                pass
    return result


# ── OCR on pre-fetched image ──────────────────────────────────────────

def _ocr_from_bytes(img_bytes: bytes) -> Optional[str]:
    """Run PaddleOCR on image bytes. Returns joined text or None."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f, format="JPEG")
            tmp_path = f.name
        try:
            ocr = _get_ocr()
            result = list(ocr.predict(tmp_path))
        finally:
            os.unlink(tmp_path)

        lines = []
        for res in result:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for txt, score in zip(texts, scores):
                if score >= 0.25 and len(txt.strip()) > 1:
                    lines.append(txt.strip())

        return " ".join(lines) if lines else None
    except Exception as e:
        log.debug(f"OCR inference failed: {e}")
        return None


# ── EasyOCR fallback ──────────────────────────────────────────────────

_easy_reader = None


def _get_easyocr():
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        log.info("Loading EasyOCR (ru+en)...")
        _easy_reader = easyocr.Reader(["ru", "en"], gpu=False)
        log.info("EasyOCR loaded")
    return _easy_reader


def _ocr_easyocr(img_bytes: bytes) -> Optional[str]:
    """Run EasyOCR on image bytes."""
    try:
        reader = _get_easyocr()
        results = reader.readtext(img_bytes)
        lines = [text.strip() for _, text, conf in results if conf >= 0.3 and len(text.strip()) > 1]
        return " ".join(lines) if lines else None
    except Exception as e:
        log.debug(f"EasyOCR failed: {e}")
        return None


# ── Validation: compare OCR numbers with product name ─────────────────

import re as _re
_RE_NUMS = _re.compile(r"\d+")


def _numbers_match(name: str, ocr_text: str) -> bool:
    """Check if key numbers from product name appear in OCR text."""
    name_nums = set(_RE_NUMS.findall(name))
    ocr_nums = set(_RE_NUMS.findall(ocr_text))
    # Filter meaningful numbers (≥2 digits, not years)
    name_nums = {n for n in name_nums if 2 <= len(n) <= 4 and n not in ("2024", "2025", "2026")}
    if not name_nums:
        return True  # no numbers to check
    # At least one key number from name should appear in OCR
    return bool(name_nums & ocr_nums)


def validate_ocr(product_name: str, ocr_text: str, thumb_url: str = None) -> str:
    """
    Validate PaddleOCR result against product name.
    If numbers don't match — re-run with EasyOCR.
    Returns validated OCR text.
    """
    if not ocr_text:
        return ocr_text or ""

    if _numbers_match(product_name, ocr_text):
        return ocr_text

    # Numbers mismatch — try EasyOCR
    if not thumb_url:
        return ocr_text

    log.info(f"OCR mismatch for '{product_name[:50]}': paddle='{ocr_text[:60]}' — retrying EasyOCR")
    img_bytes = _fetch_image_bytes(thumb_url)
    if not img_bytes:
        return ocr_text

    easy_text = _ocr_easyocr(img_bytes)
    if not easy_text:
        return ocr_text

    if _numbers_match(product_name, easy_text):
        log.info(f"  EasyOCR fixed: '{easy_text[:60]}'")
        return easy_text

    # Both disagree — prefer the one with more number matches
    name_nums = set(_RE_NUMS.findall(product_name))
    paddle_match = len(name_nums & set(_RE_NUMS.findall(ocr_text)))
    easy_match = len(name_nums & set(_RE_NUMS.findall(easy_text)))
    if easy_match > paddle_match:
        log.info(f"  EasyOCR better ({easy_match} vs {paddle_match} matches): '{easy_text[:60]}'")
        return easy_text

    return ocr_text


def validate_and_fix_ocr(products) -> int:
    """
    Check all products: if OCR numbers mismatch name, re-run EasyOCR once.
    Saves fixed result to DB so next pipeline run uses cache.
    Returns count of fixed products.
    """
    to_fix = []
    for p in products:
        if not p.ocr_text or not p.thumb_url:
            continue
        if not _numbers_match(p.name, p.ocr_text):
            to_fix.append(p)

    if not to_fix:
        return 0

    log.info(f"  OCR mismatch detected: {len(to_fix)} products — fixing with EasyOCR...")
    fixed = 0

    for p in to_fix:
        new_text = validate_ocr(p.name, p.ocr_text, thumb_url=p.thumb_url)
        if new_text != p.ocr_text:
            p.ocr_text = new_text
            # Save to DB
            try:
                from sqlalchemy import create_engine, text as sa_text
                eng = create_engine(DB_DSN, pool_pre_ping=True)
                with eng.begin() as conn:
                    conn.execute(
                        sa_text(f"""
                            UPDATE {T_OCR}
                            SET ocr_text = :txt, ocr_engine = 'easyocr_fix', updated_at = NOW()
                            WHERE pk_id = :pk
                        """),
                        {"pk": p.pk_id, "txt": new_text},
                    )
                eng.dispose()
                fixed += 1
            except Exception as e:
                log.warning(f"Failed to save fixed OCR for pk={p.pk_id}: {e}")

    if fixed:
        log.info(f"  Fixed {fixed}/{len(to_fix)} OCR results (saved to DB)")
    return fixed


# ── Single product OCR (fetch + infer, for one-off use) ───────────────

def ocr_product(thumb_url: str) -> Optional[dict]:
    """Fetch image and run PaddleOCR. Returns {"ocr_text": ...} or None."""
    img_bytes = _fetch_image_bytes(thumb_url)
    if not img_bytes:
        return None
    text = _ocr_from_bytes(img_bytes)
    if not text:
        return None
    return {"ocr_text": text}


# ── DB cache ───────────────────────────────────────────────────────────

def _load_cached(product_ids: list[int]) -> dict[int, dict]:
    """Load cached OCR results from mpstats_product_ocr."""
    if not product_ids:
        return {}

    from sqlalchemy import create_engine, text
    eng = create_engine(DB_DSN, pool_pre_ping=True)
    result = {}

    chunk_size = 5000
    for i in range(0, len(product_ids), chunk_size):
        chunk = product_ids[i:i + chunk_size]
        pks = ",".join(str(int(pk)) for pk in chunk)
        with eng.connect() as conn:
            rows = conn.execute(
                text(f"SELECT pk_id, ocr_text, ocr_attrs FROM {T_OCR} WHERE pk_id IN ({pks})")
            ).mappings().all()
        for r in rows:
            attrs = r["ocr_attrs"]
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
            # Normalize flat Gemini attrs ({"key": value}) into scoring
            # format ({"key": {"value": v, "label": k, "numeric": bool}})
            if attrs and isinstance(attrs, dict):
                first_val = next(iter(attrs.values()), None)
                if not isinstance(first_val, dict):
                    # Flat format — convert
                    normalized = {}
                    for k, v in attrs.items():
                        if v is None:
                            continue
                        is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                        normalized[k] = {
                            "value": v,
                            "label": k,
                            "numeric": is_num,
                            "priority": 1 if is_num else 2,
                        }
                    attrs = normalized
            result[int(r["pk_id"])] = {
                "ocr_text": r["ocr_text"] or "",
                "ocr_attrs": attrs or {},
            }

    eng.dispose()
    return result


def _flush_to_db(records: list[dict]) -> int:
    """Save OCR results to mpstats_product_ocr. Returns count saved."""
    if not records:
        return 0

    from sqlalchemy import create_engine, text
    eng = create_engine(DB_DSN, pool_pre_ping=True)

    saved = 0
    with eng.begin() as conn:
        for rec in records:
            conn.execute(
                text(f"""
                    INSERT INTO {T_OCR} (pk_id, ocr_text, ocr_attrs, ocr_engine)
                    VALUES (:pk_id, :ocr_text, :ocr_attrs, 'paddleocr')
                    ON DUPLICATE KEY UPDATE
                        ocr_text = VALUES(ocr_text),
                        ocr_attrs = VALUES(ocr_attrs),
                        ocr_engine = VALUES(ocr_engine),
                        updated_at = NOW()
                """),
                {
                    "pk_id": rec["pk_id"],
                    "ocr_text": rec["ocr_text"],
                    "ocr_attrs": json.dumps(rec.get("ocr_attrs", {}), ensure_ascii=False),
                },
            )
            saved += 1
    eng.dispose()
    return saved


# ── Batch extraction ───────────────────────────────────────────────────

def run_ocr(products: list[Product], workers: int = None) -> int:
    """
    Run PaddleOCR on products with thumb_url.
    1. Load cache from DB
    2. Prefetch images in parallel (I/O)
    3. Run OCR sequentially on prefetched images
    4. Flush to DB every FLUSH_EVERY products
    Returns count of products with OCR data.
    """
    if workers is None:
        workers = OCR_WORKERS

    candidates = [p for p in products if p.thumb_url]
    if not candidates:
        return 0

    # Load cache
    pk_ids = [p.pk_id for p in candidates]
    cached = _load_cached(pk_ids)
    log.info(f"OCR: {len(cached)} cached, {len(candidates) - len(cached)} to process")

    # Apply cached
    filled = 0
    to_process = []
    for p in candidates:
        if p.pk_id in cached:
            p.ocr_text = cached[p.pk_id]["ocr_text"]
            p.ocr_attrs = cached[p.pk_id]["ocr_attrs"]
            filled += 1
        else:
            to_process.append(p)

    if not to_process:
        log.info(f"OCR: all {filled} from cache")
        return filled

    # Warm up model once
    _get_ocr()

    # Process in chunks: prefetch images → OCR → flush to DB
    chunk_size = FLUSH_EVERY
    total_saved = 0

    for chunk_start in range(0, len(to_process), chunk_size):
        chunk = to_process[chunk_start:chunk_start + chunk_size]

        # 1. Prefetch images in parallel
        images = _prefetch_images(chunk, workers=min(workers, 8))

        # 2. Run OCR sequentially on fetched images
        batch_records = []
        for p in chunk:
            img_bytes = images.get(p.pk_id)
            if not img_bytes:
                continue
            try:
                text = _ocr_from_bytes(img_bytes)
                if text:
                    p.ocr_text = text
                    batch_records.append({
                        "pk_id": p.pk_id,
                        "ocr_text": text,
                        "ocr_attrs": {},
                    })
                    filled += 1
            except Exception as e:
                log.debug(f"OCR error pk_id={p.pk_id}: {e}")

        # 3. Flush to DB
        if batch_records:
            saved = _flush_to_db(batch_records)
            total_saved += saved

        done = min(chunk_start + chunk_size, len(to_process))
        log.info(f"  OCR: {done}/{len(to_process)} done, {filled} with data, {total_saved} saved to DB")

    log.info(f"OCR: done — {filled}/{len(candidates)} products have OCR data")
    return filled
