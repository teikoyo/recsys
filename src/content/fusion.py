#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-View Fusion with Rho-Adaptive Weights

Implements CONTENT_VIEW_EXTENSION.md section 5.1 (row information
concentration rho and adaptive alpha), section 5.2 (consistency
adjustment), and section 5.3 (fusion formula with top-K trimming).
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def compute_rho(S_csr):
    """Compute row information concentration rho[i] = sum_j S[i,j]^2.

    Args:
        S_csr: scipy CSR sparse matrix.

    Returns:
        (N,) float64 array of per-row rho values.
    """
    return np.array(S_csr.multiply(S_csr).sum(axis=1)).flatten().astype(np.float64)


def compute_adaptive_alpha(rho_dict, views):
    """Normalise rho values into per-row adaptive weights (summing to 1).

    Args:
        rho_dict: {view_name: (N,) rho array}.
        views: Ordered list of view names.

    Returns:
        {view_name: (N,) alpha weight array}.
    """
    N = len(next(iter(rho_dict.values())))
    rho_stack = np.stack([rho_dict[v] for v in views], axis=0)  # (V, N)
    rho_sum = np.maximum(rho_stack.sum(axis=0, keepdims=True), 1e-12)
    alpha_stack = rho_stack / rho_sum

    return {v: alpha_stack[vi] for vi, v in enumerate(views)}


def apply_consistency_adjustment(alpha, c_scores_df, N, beta=0.5):
    """Adjust tabcontent alpha weights using consistency scores.

    For documents where metadata and content views agree strongly
    (high c[i]), the content view weight is reduced since the metadata
    already captures that signal.

    g(i) = beta + (1 - beta) * (1 - c[i])

    Non-D_content documents have c=0, so g=1 (unaffected).

    Args:
        alpha: {view_name: (N,) alpha array} (will not be mutated).
        c_scores_df: DataFrame with 'doc_idx' and 'weighted_consistency'.
        N: Total number of nodes.
        beta: Base retention coefficient (default 0.5).

    Returns:
        Adjusted and re-normalised alpha dict.
    """
    c_arr = np.zeros(N, dtype=np.float64)
    for _, row in c_scores_df.iterrows():
        idx = int(row["doc_idx"])
        if 0 <= idx < N:
            c_arr[idx] = float(row["weighted_consistency"])

    g = beta + (1.0 - beta) * (1.0 - c_arr)

    alpha_adj = {v: arr.copy() for v, arr in alpha.items()}
    alpha_adj["tabcontent"] = alpha_adj["tabcontent"] * g

    # Re-normalise
    total = sum(alpha_adj[v] for v in alpha_adj)
    total = np.maximum(total, 1e-12)
    for v in alpha_adj:
        alpha_adj[v] = alpha_adj[v] / total

    return alpha_adj


def fuse_views(S_dict, alpha_dict, views, N, K=50, batch_size=10000):
    """Row-wise weighted fusion of multiple view matrices.

    For each row, computes a weighted sum of view scores, applies top-K
    trimming, and L1 row-normalises.

    Args:
        S_dict: {view_name: CSR matrix}.
        alpha_dict: {view_name: (N,) alpha array}.
        views: Ordered list of view names.
        N: Matrix dimension.
        K: Top-K edges to retain per row.
        batch_size: Rows processed per progress report.

    Returns:
        (rows, cols, vals) COO triplet arrays.
    """
    all_rows = []
    all_cols = []
    all_vals = []

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)

        for i in range(batch_start, batch_end):
            fused = defaultdict(float)

            for v in views:
                a_v = alpha_dict[v][i]
                if a_v < 1e-9:
                    continue

                S_v = S_dict[v]
                row_start = S_v.indptr[i]
                row_end = S_v.indptr[i + 1]
                cols_v = S_v.indices[row_start:row_end]
                vals_v = S_v.data[row_start:row_end]

                for jj in range(len(cols_v)):
                    fused[int(cols_v[jj])] += a_v * float(vals_v[jj])

            if not fused:
                continue

            # Top-K trimming
            if len(fused) > K:
                items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:K]
            else:
                items = list(fused.items())

            # L1 row normalisation
            total = sum(v for _, v in items)
            if total < 1e-12:
                continue

            for j, val in items:
                all_rows.append(i)
                all_cols.append(j)
                all_vals.append(val / total)

    return (
        np.array(all_rows, dtype=np.int64),
        np.array(all_cols, dtype=np.int64),
        np.array(all_vals, dtype=np.float32),
    )
