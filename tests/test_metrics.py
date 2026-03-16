"""Tests for src.metrics -- all 6 ranking metric functions."""

import numpy as np
import pytest

from src.metrics import (
    dcg_at_k,
    ndcg_at_k,
    average_precision_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------- DCG ----------

def test_dcg_at_k_basic():
    """Known DCG value: [3, 2, 1, 0] -> 3/log2(2) + 2/log2(3) + 1/log2(4)."""
    rels = np.array([3, 2, 1, 0], dtype=np.float64)
    expected = 3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4) + 0.0
    assert abs(dcg_at_k(rels) - expected) < 1e-8


def test_dcg_at_k_empty():
    """Empty relevance array returns 0."""
    assert dcg_at_k(np.array([])) == 0.0


# ---------- nDCG ----------

def test_ndcg_perfect_ranking():
    """Perfect ranking (ideal order) gives nDCG = 1.0."""
    ideal = np.array([3, 2, 1, 0], dtype=np.float64)
    assert abs(ndcg_at_k(ideal, ideal) - 1.0) < 1e-8


def test_ndcg_worst_ranking():
    """All-zero gains give nDCG = 0.0."""
    zeros = np.array([0, 0, 0], dtype=np.float64)
    result = ndcg_at_k(zeros, zeros)
    assert abs(result) < 1e-6


# ---------- Average Precision ----------

def test_ap_at_k_basic():
    """AP@5 for [1,0,1,0,1]: (1/3)(1/1 + 2/3 + 3/5) = 0.7555..."""
    rels = np.array([1, 0, 1, 0, 1], dtype=np.float64)
    expected = (1.0 / 3) * (1.0 / 1 + 2.0 / 3 + 3.0 / 5)
    assert abs(average_precision_at_k(rels) - expected) < 1e-6


def test_ap_at_k_all_relevant():
    """All 1s gives AP = 1.0."""
    rels = np.array([1, 1, 1, 1], dtype=np.float64)
    assert abs(average_precision_at_k(rels) - 1.0) < 1e-8


# ---------- MRR ----------

def test_mrr_at_k_first():
    """First item relevant gives MRR = 1.0."""
    rels = np.array([1, 0, 0, 0], dtype=np.float64)
    assert mrr_at_k(rels) == 1.0


def test_mrr_at_k_third():
    """Third item relevant gives MRR = 1/3."""
    rels = np.array([0, 0, 1, 0], dtype=np.float64)
    assert abs(mrr_at_k(rels) - 1.0 / 3) < 1e-8


def test_mrr_at_k_none():
    """No relevant items gives MRR = 0.0."""
    rels = np.array([0, 0, 0, 0], dtype=np.float64)
    assert mrr_at_k(rels) == 0.0


# ---------- Precision ----------

def test_precision_at_k():
    """2 relevant out of 5 gives P@5 = 0.4."""
    rels = np.array([1, 0, 1, 0, 0], dtype=np.float64)
    assert abs(precision_at_k(rels) - 0.4) < 1e-8


# ---------- Recall ----------

def test_recall_at_k():
    """2 relevant found out of 5 total gives R@5 = 0.4."""
    rels = np.array([1, 0, 1, 0, 0], dtype=np.float64)
    assert abs(recall_at_k(rels, total_rel=5) - 0.4) < 1e-8


def test_recall_at_k_zero_total():
    """total_rel=0 gives recall = 0.0 (avoid division by zero)."""
    rels = np.array([1, 0, 1], dtype=np.float64)
    assert recall_at_k(rels, total_rel=0) == 0.0
