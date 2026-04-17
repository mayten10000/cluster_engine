#!/usr/bin/env python3
"""
Fine-tune sentence-transformers embedding model on mined pairs.

Uses ContrastiveLoss: positive pairs pulled together, negative pairs pushed
apart. Output is a new model directory that can be loaded by sentence-
transformers.SentenceTransformer like the base model.

Usage:
  cd /opt
  python3 -m cluster_engine_v2.finetune_embeddings \
    --pairs /var/cache/cluster_engine/training/pairs_v1.csv \
    --base intfloat/multilingual-e5-small \
    --output /var/cache/cluster_engine/models/cluster_engine_e5_finetuned_v1 \
    --epochs 3 \
    --batch-size 64

Requires sentence-transformers >= 2.0
"""
from __future__ import annotations
import argparse
import csv
import logging
import math
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

log = logging.getLogger("finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_pairs(csv_path: Path):
    """Load pairs CSV → list of InputExample."""
    examples = []
    pk_to_name: dict[int, str] = {}

    # First we need product names — load from MySQL
    from cluster_engine_v2.db import get_engine
    from sqlalchemy import text

    eng = get_engine()
    pks_needed: set[int] = set()

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        pks_needed.add(int(r["pk_left"]))
        pks_needed.add(int(r["pk_right"]))

    log.info(f"Loading names for {len(pks_needed)} unique products...")
    pks_list = list(pks_needed)
    chunk_size = 5000
    with eng.connect() as conn:
        for i in range(0, len(pks_list), chunk_size):
            chunk = pks_list[i : i + chunk_size]
            placeholders = ",".join(str(int(p)) for p in chunk)
            res = conn.execute(text(
                f"SELECT pk_id, name FROM mpstats_products "
                f"WHERE pk_id IN ({placeholders})"
            )).mappings().all()
            for row in res:
                pk_to_name[int(row["pk_id"])] = (row["name"] or "").strip()

    log.info(f"Resolved names for {len(pk_to_name)} products")

    skipped = 0
    for r in rows:
        l_pk = int(r["pk_left"])
        r_pk = int(r["pk_right"])
        label = int(r["label"])
        l_name = pk_to_name.get(l_pk)
        r_name = pk_to_name.get(r_pk)
        if not l_name or not r_name:
            skipped += 1
            continue
        # multilingual-e5 expects "query: " or "passage: " prefix for best results
        # but for similarity tasks "passage: " on both sides works fine
        examples.append(InputExample(
            texts=[f"passage: {l_name}", f"passage: {r_name}"],
            label=float(label),
        ))

    log.info(f"Loaded {len(examples)} training examples (skipped {skipped})")
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Path to pairs CSV")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--output", required=True, help="Output model directory")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    log.info(f"Loading base model: {args.base}")
    model = SentenceTransformer(args.base)
    log.info(f"Model device: {model.device}")

    pairs_path = Path(args.pairs)
    examples = load_pairs(pairs_path)
    if not examples:
        raise SystemExit("No examples loaded — aborting")

    random.shuffle(examples)
    train_loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)

    # ContrastiveLoss: pulls positives close (label=1), pushes negatives apart (label=0)
    train_loss = losses.ContrastiveLoss(model)

    n_steps_per_epoch = len(train_loader)
    total_steps = n_steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    log.info(
        f"Training: {args.epochs} epochs × {n_steps_per_epoch} steps = {total_steps} steps, "
        f"warmup {warmup_steps}, batch {args.batch_size}"
    )

    out_dir = Path(args.output)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=True,
    )

    log.info(f"=== DONE ===")
    log.info(f"Model saved to: {out_dir}")
    log.info(f"To use it set env: EMBEDDING_MODEL_NAME={out_dir}")


if __name__ == "__main__":
    main()
