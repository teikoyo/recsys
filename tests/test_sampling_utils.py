"""Tests for src.sampling_utils -- negative sampling distribution and alias table."""

import numpy as np
import pytest
import torch

from src.sampling_utils import build_ns_dist_from_deg, build_alias_on_device, sample_alias_gpu


# ---------- Negative sampling distribution ----------

def test_ns_dist_sums_to_one():
    """Distribution from build_ns_dist_from_deg sums to 1.0."""
    deg = np.array([10, 50, 200, 5, 30], dtype=np.float64)
    dist = build_ns_dist_from_deg(deg)
    assert abs(dist.sum() - 1.0) < 1e-12


def test_ns_dist_with_zeros():
    """Zero-degree nodes are handled (clamped to 1 internally)."""
    deg = np.array([0, 0, 100, 50, 0], dtype=np.float64)
    dist = build_ns_dist_from_deg(deg)
    assert abs(dist.sum() - 1.0) < 1e-12
    # Zero-degree nodes still get a small nonzero probability
    assert all(d > 0 for d in dist)


# ---------- Alias table ----------

def test_alias_table_shape():
    """prob and alias tensors have correct shape (n,)."""
    n = 100
    probs = np.ones(n, dtype=np.float64) / n
    device = torch.device("cpu")
    prob_t, alias_t = build_alias_on_device(probs, device)
    assert prob_t.shape == (n,)
    assert alias_t.shape == (n,)


def test_alias_sampling_in_range():
    """All alias-sampled indices are in [0, n)."""
    n = 50
    probs = np.random.dirichlet(np.ones(n))
    device = torch.device("cpu")
    prob_t, alias_t = build_alias_on_device(probs, device)
    samples = sample_alias_gpu(prob_t, alias_t, size=(1000,), device=device)
    assert samples.min().item() >= 0
    assert samples.max().item() < n


def test_alias_sampling_distribution():
    """Uniform input distribution produces roughly uniform samples (tolerance check)."""
    n = 10
    probs = np.ones(n, dtype=np.float64) / n
    device = torch.device("cpu")
    prob_t, alias_t = build_alias_on_device(probs, device)

    num_samples = 100_000
    samples = sample_alias_gpu(prob_t, alias_t, size=(num_samples,), device=device)
    counts = torch.bincount(samples, minlength=n).float()
    expected = num_samples / n

    # Each bin should be within 10% of expected for uniform
    for c in counts:
        assert abs(c.item() - expected) / expected < 0.10, (
            f"Count {c.item()} deviates too much from expected {expected}"
        )
