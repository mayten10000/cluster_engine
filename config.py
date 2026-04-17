"""
Configuration for the clustering engine.
All DB connections, API keys, and tunable parameters.
"""
import os

# ── MySQL ──────────────────────────────────────────────────────────────
DB_DSN = os.getenv(
    "QC_DB_DSN",
    "mysql+pymysql://root:CHANGE_ME_ROOT_PASS@127.0.0.1:3306/mpstats?charset=utf8mb4",
)
DB_SCHEMA = os.getenv("QC_DB_SCHEMA", "mpstats")

# Table names (match your viewer)
T_PRODUCTS = os.getenv("QC_T_MP_PRODUCTS", "mpstats_products")
T_PC = os.getenv("QC_T_PC", "mpstats_product_clusters")
T_CLUSTERS = os.getenv("QC_T_CLUSTERS", "mpstats_clusters")
T_NICHES = "mpstats_niches"
T_CATEGORIES = "mpstats_categories"
T_OCR = "mpstats_product_ocr"

# Column names
COL_PK = "pk_id"
COL_GID = "cluster_gid"
COL_SCORE = "score"
COL_RUN = "run_id"
COL_MAIN = "main_product"
COL_NICHE = "niche_key"
COL_NAME = "name"
COL_BRAND = "brand"
COL_SELLER = "seller"
COL_PRICE_MED = "final_price_median"
COL_THUMB = "thumb_middle"
COL_SALES = "sales"
COL_REVENUE = "revenue"

# ── ClickHouse ─────────────────────────────────────────────────────────
CH_HOST = os.getenv("QC_CH_HOST", "127.0.0.1")
CH_HTTP_PORT = int(os.getenv("QC_CH_HTTP_PORT", "8123"))
CH_DB = os.getenv("QC_CH_DB", "qc")
CH_USER = os.getenv("QC_CH_USER", "default")
CH_PASS = os.getenv("QC_CH_PASS", "")

# ── OpenRouter API (embeddings + LLM) ─────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")

# Embedding model via OpenRouter
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# LLM for arbitration
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "50"))

# ── Graph clustering parameters ───────────────────────────────────────

# EAN pre-match
EAN_MIN_OCCURRENCES = 2  # EAN must appear in ≥2 products to be useful
EAN_MAX_PRODUCTS = 50    # skip EAN that maps to >50 products (likely garbage)

# Blocking (candidate generation)
BLOCKING_TOP_K = 15       # candidates per product from embedding search
BLOCKING_MIN_SIM = 0.60   # minimum cosine sim to form a candidate pair

# Graph edge weights
EDGE_WEIGHT_EAN = 1.0           # EAN exact match
EDGE_WEIGHT_BRAND_TOKENS = 0.9  # brand match + high token overlap
EDGE_WEIGHT_EMBEDDING = 0.85    # weight multiplier for embedding similarity
EDGE_WEIGHT_PRICE_PENALTY = 0.15  # max penalty for price divergence

# Leiden resolution (higher = more smaller clusters)
LEIDEN_RESOLUTION = 1.0

# ── Scoring thresholds ────────────────────────────────────────────────
SCORE_OK = 0.75            # cumulative score ≥ this → auto-ok
SCORE_QUARANTINE = 0.30    # cumulative score ≤ this → auto-quarantine
# between these → LLM arbitration

# ── Cross-encoder reranker (grey-zone prefilter before LLM) ───────────
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1") not in ("0", "false", "False", "")
RERANKER_AUTO_OK = float(os.getenv("RERANKER_AUTO_OK", "0.85"))
RERANKER_AUTO_QUARANTINE = float(os.getenv("RERANKER_AUTO_QUARANTINE", "0.20"))

# ── OCR ───────────────────────────────────────────────────────────────
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddleocr")  # paddleocr (PP 3.2.0 + OCR 3.4)

# ── Run management ────────────────────────────────────────────────────
RUN_ID_START = 100  # new run IDs start here (avoid collision with old runs)
