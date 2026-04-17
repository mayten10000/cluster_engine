"""Regenerate the schema for one niche.

Usage:
    python3 -m cluster_engine_v2.regen_schema 7182
    python3 -m cluster_engine_v2.regen_schema 655
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regen_schema")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m cluster_engine_v2.regen_schema <niche_key>")
        sys.exit(1)

    niche_key = int(sys.argv[1])

    from .db import get_engine, load_products_by_niche
    from .schema_attrs import generate_niche_schema, SCHEMA_DIR
    from sqlalchemy import text as _t

    cache_path = Path(SCHEMA_DIR) / f"niche_{niche_key}.json"
    if cache_path.exists():
        backup = cache_path.with_suffix(".json.bak")
        cache_path.rename(backup)
        log.info(f"Backed up old schema → {backup}")

    # Get niche name
    with get_engine().connect() as c:
        row = c.execute(_t("SELECT name FROM mpstats_niches WHERE niche_key=:nk"), {"nk": niche_key}).first()
    niche_name = row[0] if row else f"niche_{niche_key}"
    log.info(f"Regenerating schema for niche {niche_key}: {niche_name!r}")

    # Load sample products
    products = load_products_by_niche(niche_key)
    sample_names = [p.name for p in products[:120]]
    sample_ocr = {}
    for p in products[:120]:
        ocr = getattr(p, "ocr_text", "") or ""
        if ocr:
            sample_ocr[p.name] = ocr

    log.info(f"Calling LLM with {len(sample_names)} sample names, {len(sample_ocr)} OCR snippets...")
    schema = generate_niche_schema(niche_key, niche_name, sample_names, sample_ocr=sample_ocr)
    if schema:
        log.info(f"NEW schema saved to {cache_path}")
        log.info(f"Separators: {len(schema.get('key_separators', []))}")
        for sep in schema.get("key_separators", []):
            tag = "P1" if sep.get("priority", 9) <= 1 else f"P{sep.get('priority',9)}"
            log.info(f"  [{tag}] {sep.get('label')}: {sep.get('regex')!r}")
    else:
        log.error("Schema generation FAILED")
        sys.exit(2)


if __name__ == "__main__":
    main()
