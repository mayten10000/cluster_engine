"""
Cluster Engine v2 — graph-based product clustering.

Architecture:
    1. Load products from MySQL (mpstats_products)
    2. Tokenize + normalize (pymorphy3 lemmatization)
    3. Embed names via OpenRouter API (text-embedding-3-small)
    4. Generate candidate edges:
       - Phase 0: EAN exact match
       - Phase 1: Embedding kNN blocking
       - Phase 2: Brand + token overlap boost
    5. Graph clustering: Leiden community detection (igraph)
    6. Cumulative scoring (embedding sim + tokens + brand + attrs + price)
    7. LLM arbitration for grey zone (Gemini Flash)
    8. Diff old vs new → cluster_moves (ClickHouse)
    9. Write results to mpstats_product_clusters (new run_id)

Usage:
    python -m cluster_engine --category_id=123
    python -m cluster_engine --niche_key=456 --dry_run
    python -m cluster_engine --category_id=123 --skip_llm --resolution=1.2
"""
