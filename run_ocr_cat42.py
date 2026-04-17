"""One-shot OCR runner for cat42 products."""
import logging
from cluster_engine_v2.db import load_products_by_category
from cluster_engine_v2.ocr import run_ocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

products = load_products_by_category(42)
# Niche 7182 (гирлянды) first — to verify the 258695 effect quickly
products.sort(key=lambda p: (p.niche_key != 7182, p.niche_key or 0))
print(f"Loaded {len(products)} cat42 products, niche 7182 prioritized")
filled = run_ocr(products)
print(f"OCR filled: {filled}")
