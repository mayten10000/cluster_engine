"""
Data models for the clustering engine.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Product:
    pk_id: int
    name: str
    brand: str = ""
    seller: str = ""
    niche_key: int = 0
    price: float = 0.0
    sales_1m: float = 0.0
    revenue_1m: float = 0.0
    thumb_url: str = ""
    ean: str = ""

    # Computed during pipeline
    name_normalized: str = ""
    brand_normalized: str = ""
    tokens: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    # OCR from product card image
    ocr_text: str = ""          # raw OCR text
    ocr_attrs: dict = field(default_factory=dict)  # structured attrs from OCR

    # Old cluster assignment (from previous run)
    old_cluster_gid: Optional[int] = None
    old_score: float = 0.0

    # New cluster assignment (from this run)
    new_cluster_gid: Optional[int] = None
    new_score: float = 0.0

    # Scoring components
    ean_match: bool = False
    brand_match_score: float = 0.0
    token_overlap_score: float = 0.0
    embedding_sim: float = 0.0
    price_ratio: float = 1.0
    cumulative_score: float = 0.0

    price_tier: int = 1
    price_src: int = 0
    anomaly_flags: int = 0

    # Decision
    decision: str = ""  # ok / quarantine / move / grey
    decided_by: str = ""
    reason: str = ""
    score_errors: list = field(default_factory=list)  # [S2], [S4], etc.
    confidence: float = 0.5


@dataclass
class ClusterResult:
    gid: int
    product_ids: list[int] = field(default_factory=list)
    main_pk: Optional[int] = None
    centroid: Optional[np.ndarray] = None
    avg_price: float = 0.0
    brand: str = ""
    size: int = 0
