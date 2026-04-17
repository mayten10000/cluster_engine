# cluster_engine

Marketplace product clustering pipeline. Given a category of products already
assigned to clusters, re-evaluates each assignment and proposes moves, splits,
and quarantines. Designed for long-tail e-commerce where cluster quality drifts
as new SKUs arrive.

## Pipeline

1. **Load** — products + existing cluster assignments from MySQL
2. **Embed** — sentence-transformers (multilingual-e5); numbers stripped before encoding so variants like `14 шт` vs `16 шт` don't collapse
3. **OCR attrs** — cached PaddleOCR + schema-driven attribute extraction (brand, quantity, dimensions, function)
4. **Score** — per-product cumulative score vs its current cluster (embedding sim + brand + attr overlap)
5. **Anomaly pass** — flags "quiet mismatches": ok-scored products whose P1 numeric attrs deviate >15% from cluster median, plus all members of size≤2 clusters
6. **Reranker** — cross-encoder (bge-reranker-v2-m3) filters the grey zone
7. **LLM arbitration** — Gemini Flash decides `ok` / `move` / `split` / `quarantine` per product, sees niche-wide cluster candidates for move targets. Includes a relative P1 validator that allows a move when the target cluster median is closer to the product value than the source
8. **Vision** — function-level check on rejected moves (gemini-flash on product thumb)
9. **CLIP** — image-side consistency check on move candidates
10. **Write** — `qc.cluster_moves` in ClickHouse for review

## Key modules

| File | Role |
|---|---|
| `pipeline.py` | Orchestrator |
| `scoring.py` | Cumulative product↔cluster score |
| `anomaly_pass.py` | Quiet-mismatch detection |
| `reranker.py` | Cross-encoder grey-zone filter |
| `llm_arbiter.py` | LLM prompt, niche candidates, relative P1 validator |
| `vision_tags.py` | Vision function check |
| `schema_attrs.py` | Attr extraction from name + OCR |
| `blocking.py` | Candidate edge generation |
| `graph_cluster.py` | Leiden clustering (for initial build) |
| `ch_writer.py` | ClickHouse move writer |

## Config

Everything via env vars (see `config.py`). Expected:

```
QC_DB_DSN              MySQL connection
QC_CH_HOST/PORT/DB     ClickHouse
OPENROUTER_API_KEY     LLM + fallback embeddings
LOCAL_EMBEDDING_MODEL  path or HF id (default intfloat/multilingual-e5-small)
LLM_MODEL              default google/gemini-2.0-flash-001
```

## Run

```bash
# Full re-clustering build
python -m cluster_engine_v2 --category_id=37

# Highlight mode: score existing assignments, propose moves only
python -m cluster_engine_v2 --highlight --category_id=37 -v
```
