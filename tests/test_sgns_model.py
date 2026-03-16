"""Tests for src.sgns_model -- SGNS model forward, backward, and embeddings."""

import pytest
import torch

from src.sgns_model import SGNS


@pytest.fixture
def model():
    """Small SGNS model for testing."""
    return SGNS(vocab_size=100, dim=32, sparse=False)


@pytest.fixture
def batch():
    """Synthetic batch: center [B], pos [B], neg [B, K]."""
    B, K = 8, 5
    center = torch.randint(0, 100, (B,))
    pos = torch.randint(0, 100, (B,))
    neg = torch.randint(0, 100, (B, K))
    return center, pos, neg


# ---------- Forward ----------

def test_forward_returns_scalar(model, batch):
    """Forward returns a 1-element tensor."""
    center, pos, neg = batch
    loss = model(center, pos, neg)
    assert loss.shape == (1,)


def test_forward_positive_loss(model, batch):
    """Loss should be positive (softplus is always >= 0)."""
    center, pos, neg = batch
    loss = model(center, pos, neg)
    assert loss.item() > 0.0


# ---------- Backward ----------

def test_backward_runs(model, batch):
    """Backward pass computes gradients on both embeddings."""
    center, pos, neg = batch
    loss = model(center, pos, neg)
    loss.backward()
    assert model.in_emb.weight.grad is not None
    assert model.out_emb.weight.grad is not None


# ---------- Embeddings ----------

def test_get_embeddings_shape(model):
    """get_embeddings returns (vocab_size, dim) shape."""
    emb = model.get_embeddings(normalize=True)
    assert emb.shape == (100, 32)


def test_get_embeddings_normalized(model):
    """Normalized embeddings have L2 norm ~1.0 for each row."""
    emb = model.get_embeddings(normalize=True)
    norms = torch.norm(emb, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_get_embeddings_unnormalized(model):
    """normalize=False returns raw embeddings (not unit-normed)."""
    emb = model.get_embeddings(normalize=False)
    norms = torch.norm(emb, dim=1)
    # Raw init is uniform in [-0.5/dim, 0.5/dim]; norms should NOT all be 1
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
