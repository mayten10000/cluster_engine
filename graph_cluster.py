"""
Graph clustering — Leiden community detection on weighted product graph.

Builds an igraph Graph from candidate edges, runs Leiden algorithm,
and produces cluster assignments.
"""
from __future__ import annotations
import logging
from collections import defaultdict

import numpy as np
import igraph as ig
import leidenalg

from .config import LEIDEN_RESOLUTION
from .models import Product, ClusterResult

log = logging.getLogger("cluster_engine.graph")


def build_graph(
    products: list[Product],
    edges: list[tuple[int, int, float]],
) -> tuple[ig.Graph, dict[int, int]]:
    """
    Build an igraph Graph from products and weighted edges.

    Returns:
        graph: igraph.Graph with edge weights
        pk_to_vertex: mapping from pk_id to vertex index
    """
    # Create vertex mapping
    pk_to_vertex: dict[int, int] = {}
    for i, p in enumerate(products):
        pk_to_vertex[p.pk_id] = i

    # Build edge list (only for products that exist)
    ig_edges = []
    ig_weights = []
    skipped = 0

    for pk_a, pk_b, weight in edges:
        va = pk_to_vertex.get(pk_a)
        vb = pk_to_vertex.get(pk_b)
        if va is None or vb is None:
            skipped += 1
            continue
        ig_edges.append((va, vb))
        ig_weights.append(weight)

    g = ig.Graph(n=len(products), edges=ig_edges, directed=False)
    g.es["weight"] = ig_weights

    # Store pk_ids as vertex attributes
    g.vs["pk_id"] = [p.pk_id for p in products]
    g.vs["name_str"] = [p.name[:80] for p in products]

    log.info(
        f"Graph: {g.vcount()} vertices, {g.ecount()} edges "
        f"(skipped {skipped} edges with missing vertices)"
    )
    return g, pk_to_vertex


def run_leiden(
    graph: ig.Graph,
    resolution: float = LEIDEN_RESOLUTION,
) -> list[int]:
    """
    Run Leiden community detection on the graph.
    Returns list of community IDs (one per vertex).
    """
    log.info(f"Running Leiden (resolution={resolution})...")

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,  # run until convergence
        seed=42,
    )

    n_communities = len(set(partition.membership))
    modularity = partition.quality()

    log.info(
        f"Leiden result: {n_communities} communities, "
        f"modularity={modularity:.4f}"
    )
    return partition.membership


def membership_to_clusters(
    products: list[Product],
    membership: list[int],
    gid_start: int,
) -> dict[int, ClusterResult]:
    """
    Convert Leiden membership to ClusterResult objects.
    Assigns sequential cluster_gid starting from gid_start.

    Singletons (community of 1) get their own cluster but are flagged.
    """
    # Group products by community
    community_products: dict[int, list[Product]] = defaultdict(list)
    for i, p in enumerate(products):
        community_products[membership[i]].append(p)

    # Sort communities by size (largest first) for stable GID assignment
    sorted_communities = sorted(
        community_products.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    clusters: dict[int, ClusterResult] = {}
    current_gid = gid_start

    for community_id, members in sorted_communities:
        gid = current_gid
        current_gid += 1

        # Pick main product: highest revenue, or highest sales
        main = max(members, key=lambda p: (p.revenue_1m, p.sales_1m, -p.pk_id))

        # Compute average price
        prices = [p.price for p in members if p.price > 0]
        avg_price = np.mean(prices) if prices else 0.0

        # Most common brand
        brand_counts = defaultdict(int)
        for p in members:
            if p.brand:
                brand_counts[p.brand.lower()] += 1
        top_brand = max(brand_counts, key=brand_counts.get) if brand_counts else ""

        cluster = ClusterResult(
            gid=gid,
            product_ids=[p.pk_id for p in members],
            main_pk=main.pk_id,
            avg_price=avg_price,
            brand=top_brand,
            size=len(members),
        )
        clusters[gid] = cluster

        # Assign to products
        for p in members:
            p.new_cluster_gid = gid

    log.info(
        f"Created {len(clusters)} clusters "
        f"(min size={min(c.size for c in clusters.values())}, "
        f"max size={max(c.size for c in clusters.values())}, "
        f"singletons={sum(1 for c in clusters.values() if c.size == 1)})"
    )
    return clusters


def compute_cluster_centroids(
    clusters: dict[int, ClusterResult],
    products: list[Product],
) -> None:
    """Compute centroid embeddings for each cluster."""
    pk_to_product = {p.pk_id: p for p in products}

    for gid, cluster in clusters.items():
        embeddings = []
        for pk in cluster.product_ids:
            p = pk_to_product.get(pk)
            if p and p.embedding is not None:
                embeddings.append(p.embedding)

        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            cluster.centroid = centroid
