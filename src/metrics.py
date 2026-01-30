#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Metrics

Implements ranking metrics for evaluating recommendation quality:
- nDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Precision@K
- Recall@K
"""

import numpy as np
from typing import Union, List

# Small constant to avoid division by zero
ATOL = 1e-12


def dcg_at_k(rels: np.ndarray) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    DCG@K = Σ_{i=1}^{K} rel_i / log2(i + 1)

    Args:
        rels: Relevance scores in ranking order (non-negative)

    Returns:
        DCG score

    Example:
        >>> dcg_at_k(np.array([3, 2, 1, 0]))  # Graded relevance
        5.69...
    """
    if rels.size == 0:
        return 0.0
    ranks = np.arange(1, rels.size + 1)
    return float(np.sum(rels / np.log2(ranks + 1)))


def ndcg_at_k(
    gains_sorted: np.ndarray,
    gains_ideal_sorted: np.ndarray
) -> float:
    """
    Compute Normalized DCG at K.

    nDCG@K = DCG@K / IDCG@K

    where IDCG@K is the DCG of the ideal (best possible) ranking.

    Args:
        gains_sorted: Relevance gains in system's ranking order
        gains_ideal_sorted: Relevance gains in ideal (descending) order

    Returns:
        nDCG score in [0, 1]

    Example:
        >>> gains = np.array([2, 0, 1])  # System ranking
        >>> ideal = np.array([2, 1, 0])  # Ideal ranking
        >>> ndcg_at_k(gains, ideal)
        0.919...
    """
    dcg = dcg_at_k(gains_sorted)
    idcg = dcg_at_k(gains_ideal_sorted) if gains_ideal_sorted.size else 0.0
    return float(dcg / (idcg + ATOL))


def average_precision_at_k(rels_binary: np.ndarray) -> float:
    """
    Compute Average Precision at K.

    AP@K = (1/R) Σ_{i=1}^{K} P(i) × rel(i)

    where R is the number of relevant items and P(i) is precision at i.

    Args:
        rels_binary: Binary relevance (0/1) in ranking order

    Returns:
        AP score in [0, 1]

    Example:
        >>> ap = average_precision_at_k(np.array([1, 0, 1, 0, 1]))
        >>> print(f"{ap:.3f}")  # (1/3)(1/1 + 2/3 + 3/5)
        0.756
    """
    if rels_binary.size == 0:
        return 0.0

    hits = 0
    s = 0.0
    for i, r in enumerate(rels_binary, 1):
        if r > 0:
            hits += 1
            s += hits / i

    total_rel = int(rels_binary.sum())
    return float(s / max(1, total_rel))


def mrr_at_k(rels_binary: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank at K.

    MRR@K = 1 / rank_of_first_relevant

    Args:
        rels_binary: Binary relevance (0/1) in ranking order

    Returns:
        MRR score in [0, 1], or 0 if no relevant item in top K

    Example:
        >>> mrr_at_k(np.array([0, 0, 1, 0]))  # First relevant at position 3
        0.333...
    """
    pos = np.flatnonzero(rels_binary > 0)
    return float(1.0 / (pos[0] + 1)) if pos.size > 0 else 0.0


def precision_at_k(rels_binary: np.ndarray) -> float:
    """
    Compute Precision at K.

    P@K = |relevant ∩ top-K| / K

    Args:
        rels_binary: Binary relevance (0/1) in ranking order

    Returns:
        Precision score in [0, 1]

    Example:
        >>> precision_at_k(np.array([1, 0, 1, 0, 0]))  # 2 relevant in top 5
        0.4
    """
    K = max(1, rels_binary.size)
    return float(np.sum(rels_binary > 0) / K)


def recall_at_k(rels_binary: np.ndarray, total_rel: int) -> float:
    """
    Compute Recall at K.

    R@K = |relevant ∩ top-K| / |all relevant|

    Args:
        rels_binary: Binary relevance (0/1) in ranking order
        total_rel: Total number of relevant items (for the query)

    Returns:
        Recall score in [0, 1]

    Example:
        >>> recall_at_k(np.array([1, 0, 1, 0, 0]), total_rel=5)  # 2 of 5 found
        0.4
    """
    if total_rel <= 0:
        return 0.0
    return float(np.sum(rels_binary > 0) / total_rel)


def hit_rate_at_k(rels_binary: np.ndarray) -> float:
    """
    Compute Hit Rate at K.

    HR@K = 1 if any relevant in top-K, else 0

    Args:
        rels_binary: Binary relevance (0/1) in ranking order

    Returns:
        1.0 if at least one relevant item, 0.0 otherwise
    """
    return float(np.any(rels_binary > 0))


def evaluate_ranking(
    ranked_items: np.ndarray,
    relevant_items: Union[np.ndarray, set, List],
    gains: np.ndarray = None,
    k: int = None
) -> dict:
    """
    Compute all ranking metrics for a single query.

    Args:
        ranked_items: Array of item indices in ranked order
        relevant_items: Set/array of relevant item indices
        gains: Optional graded relevance (for nDCG). If None, uses binary.
        k: Number of items to consider (default: len(ranked_items))

    Returns:
        Dict with all metrics

    Example:
        >>> ranked = np.array([5, 2, 8, 1, 3])
        >>> relevant = {2, 3, 7}
        >>> metrics = evaluate_ranking(ranked, relevant)
    """
    if isinstance(relevant_items, (set, list)):
        relevant_set = set(relevant_items)
    else:
        relevant_set = set(relevant_items.tolist())

    if k is not None:
        ranked_items = ranked_items[:k]

    # Binary relevance
    rels_binary = np.array([1.0 if x in relevant_set else 0.0 for x in ranked_items])

    # Graded relevance for nDCG
    if gains is not None:
        rels_graded = gains
        ideal_graded = np.sort(gains)[::-1]
    else:
        rels_graded = rels_binary
        ideal_graded = np.sort(rels_binary)[::-1]

    total_rel = len(relevant_set)

    return {
        'ndcg': ndcg_at_k(rels_graded, ideal_graded),
        'map': average_precision_at_k(rels_binary),
        'mrr': mrr_at_k(rels_binary),
        'precision': precision_at_k(rels_binary),
        'recall': recall_at_k(rels_binary, total_rel),
        'hit_rate': hit_rate_at_k(rels_binary),
    }
