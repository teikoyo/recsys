"""Tests for src.csr_utils -- CSR transpose and torch conversion."""

import numpy as np
import pytest
import torch
from scipy import sparse

from src.csr_utils import csr_T, csr_rowview_torch


# ---------- csr_T ----------

def test_csr_T_shape(small_csr):
    """Transpose swaps (M, N) -> (N, M)."""
    t = csr_T(small_csr)
    assert t.shape == (small_csr.shape[1], small_csr.shape[0])


def test_csr_T_values(small_csr):
    """Transpose preserves all values (same set of nonzeros)."""
    t = csr_T(small_csr)
    orig_vals = sorted(small_csr.data.tolist())
    trans_vals = sorted(t.data.tolist())
    assert np.allclose(orig_vals, trans_vals)


# ---------- csr_rowview_torch ----------

def test_csr_rowview_torch_device(small_csr, cpu_device):
    """All returned tensors are on the specified device."""
    indptr, indices, data = csr_rowview_torch(small_csr, cpu_device)
    assert indptr.device == cpu_device
    assert indices.device == cpu_device
    assert data.device == cpu_device


def test_csr_rowview_torch_shapes(small_csr, cpu_device):
    """indptr has n_rows+1 elements; indices and data have nnz elements."""
    indptr, indices, data = csr_rowview_torch(small_csr, cpu_device)
    n_rows = small_csr.shape[0]
    nnz = small_csr.nnz
    assert indptr.shape == (n_rows + 1,)
    assert indices.shape == (nnz,)
    assert data.shape == (nnz,)
