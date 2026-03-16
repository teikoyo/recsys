"""Tests for src.pair_batch_utils -- pair generation and batching."""

import pytest
import torch
import numpy as np

from src.pair_batch_utils import iter_pairs_from_corpus, batch_pairs_and_negs_fast
from src.sampling_utils import build_alias_on_device, build_ns_dist_from_deg


def _make_corpus(n_sents=5, sent_len=6):
    """Create a simple synthetic corpus (list of lists of string token ids)."""
    return [[str(i * sent_len + j) for j in range(sent_len)] for i in range(n_sents)]


# ---------- iter_pairs_from_corpus ----------

def test_iter_pairs_produces_tuples():
    """Each yielded element is a (int, int) tuple."""
    corpus = _make_corpus(n_sents=3, sent_len=4)
    for center, context in iter_pairs_from_corpus(corpus, window=2):
        assert isinstance(center, int)
        assert isinstance(context, int)
        break  # one check is enough


def test_iter_pairs_window_limit():
    """Context tokens are within window distance of the center."""
    corpus = [["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]]
    window = 2
    for center, context in iter_pairs_from_corpus(corpus, window=window, seed=0):
        # Dynamic window is in [1, window], so max distance is window
        assert abs(center - context) <= window


def test_iter_pairs_forward_only():
    """With forward_only=True, context index > center index (within a sentence)."""
    corpus = [["0", "1", "2", "3", "4"]]
    for center, context in iter_pairs_from_corpus(
        corpus, window=3, forward_only=True, seed=42
    ):
        assert context > center


def test_iter_pairs_max_sents():
    """max_sents limits how many sentences are processed."""
    corpus = _make_corpus(n_sents=10, sent_len=4)
    pairs_limited = list(iter_pairs_from_corpus(corpus, window=2, max_sents=2))
    pairs_full = list(iter_pairs_from_corpus(corpus, window=2, max_sents=10, seed=42))
    # Same seed means same pairs; limited should have fewer
    assert len(pairs_limited) < len(pairs_full)


# ---------- batch_pairs_and_negs_fast ----------

def test_batch_shapes():
    """Batch tensors have correct shapes: centers [B], contexts [B]."""
    vocab_size = 30
    deg = np.ones(vocab_size, dtype=np.float64)
    ns_dist = build_ns_dist_from_deg(deg)
    device = torch.device("cpu")
    prob_t, alias_t = build_alias_on_device(ns_dist, device)

    corpus = _make_corpus(n_sents=3, sent_len=6)
    pair_iter = iter_pairs_from_corpus(corpus, window=2)
    batch_size = 8
    negK = 4

    for centers, contexts, negs in batch_pairs_and_negs_fast(
        pair_iter, batch_size, negK, prob_t, alias_t, device
    ):
        assert centers.dim() == 1
        assert contexts.dim() == 1
        assert centers.shape == contexts.shape
        break  # check first batch


def test_batch_negatives_shape():
    """Negative samples have shape (B, K)."""
    vocab_size = 30
    deg = np.ones(vocab_size, dtype=np.float64)
    ns_dist = build_ns_dist_from_deg(deg)
    device = torch.device("cpu")
    prob_t, alias_t = build_alias_on_device(ns_dist, device)

    corpus = _make_corpus(n_sents=5, sent_len=8)
    pair_iter = iter_pairs_from_corpus(corpus, window=3)
    batch_size = 16
    negK = 7

    for centers, contexts, negs in batch_pairs_and_negs_fast(
        pair_iter, batch_size, negK, prob_t, alias_t, device
    ):
        B = centers.shape[0]
        assert negs.shape == (B, negK)
        break
