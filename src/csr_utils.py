#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSR Matrix Utilities

Provides functions for loading and manipulating CSR (Compressed Sparse Row)
matrices, particularly for loading from Parquet files and converting to
PyTorch tensors for GPU operations.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import sparse


def load_csr_triplet_parquet(
    path: Union[str, Path],
    shape: Tuple[int, int],
    dtype: np.dtype = np.float32
) -> sparse.csr_matrix:
    """
    Load a CSR matrix from a Parquet file containing triplet (row, col, val) format.

    The Parquet file should have columns: 'row', 'col', 'val'.

    Args:
        path: Path to the Parquet file
        shape: Tuple (n_rows, n_cols) for the matrix shape
        dtype: Data type for matrix values

    Returns:
        scipy.sparse.csr_matrix

    Example:
        >>> DT_ppmi = load_csr_triplet_parquet("tmp/DT_ppmi.parquet", shape=(N, T))
    """
    df = pd.read_parquet(path, engine="fastparquet")
    coo = sparse.coo_matrix(
        (df["val"].astype(dtype), (df["row"], df["col"])),
        shape=shape,
        dtype=dtype
    )
    return coo.tocsr()


def csr_rowview_torch(
    mat: sparse.csr_matrix,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert CSR matrix to PyTorch tensors for GPU row-wise access.

    Returns the CSR components (indptr, indices, data) as PyTorch tensors
    on the specified device for efficient row-wise operations.

    Args:
        mat: Scipy CSR matrix
        device: Target device (cuda:X or cpu)

    Returns:
        Tuple of (indptr, indices, data) tensors:
            - indptr: Row pointer array [n_rows + 1]
            - indices: Column indices for non-zeros
            - data: Non-zero values

    Example:
        >>> indptr, indices, data = csr_rowview_torch(DT_ppmi, device)
        >>> # Access row r: indices[indptr[r]:indptr[r+1]]
    """
    mat = mat.tocsr()
    indptr = torch.from_numpy(mat.indptr.astype(np.int64)).to(device)
    indices = torch.from_numpy(mat.indices.astype(np.int64)).to(device)
    data = torch.from_numpy(mat.data.astype(np.float32)).to(device)
    return indptr, indices, data


def csr_T(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Transpose a CSR matrix and return as CSR.

    Args:
        mat: Input CSR matrix

    Returns:
        Transposed CSR matrix
    """
    return mat.transpose().tocsr()
