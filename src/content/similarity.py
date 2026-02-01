#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparse Similarity Graph Construction and I/O

Implements CONTENT_VIEW_EXTENSION.md section 3.6 (content view similarity
graph) and provides shared utilities for loading/saving partitioned COO
edge files with manifest.json metadata.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def sym_and_rownorm(rows, cols, vals, N):
    """Symmetrise (max) and L1 row-normalise a sparse matrix.

    Args:
        rows: Row indices (int64 array).
        cols: Column indices (int64 array).
        vals: Edge values (float32 array).
        N: Matrix dimension.

    Returns:
        (coo_rows, coo_cols, coo_vals) of the result.
    """
    S = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    S_T = S.T.tocsr()

    # Element-wise max symmetrisation
    S_sym = S.maximum(S_T).tocsr()

    # Remove diagonal
    S_sym.setdiag(0)
    S_sym.eliminate_zeros()

    # L1 row normalisation
    row_sums = np.array(S_sym.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, 1e-12)
    D_inv = sparse.diags(1.0 / row_sums)
    S_norm = (D_inv @ S_sym).tocoo()

    return (
        S_norm.row.astype(np.int64),
        S_norm.col.astype(np.int64),
        S_norm.data.astype(np.float32),
    )


def save_partitioned_edges(rows, cols, vals, N, prefix, k, output_dir,
                           part_size=2_000_000, note=""):
    """Save COO sparse matrix as partitioned parquet files + manifest.json.

    Args:
        rows/cols/vals: COO triplet arrays.
        N: Matrix dimension.
        prefix: Name prefix (e.g. 'S_tabcontent_symrow').
        k: Target k for the manifest metadata.
        output_dir: Output directory path.
        part_size: Maximum edges per partition file.
        note: Optional description for the manifest.

    Returns:
        Path to the manifest.json file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = pd.DataFrame({"row": rows, "col": cols, "val": vals})
    n_parts = max(1, (len(edges) + part_size - 1) // part_size)

    part_files = []
    for p in range(n_parts):
        start = p * part_size
        end = min((p + 1) * part_size, len(edges))
        part_name = f"{prefix}_k{k}_part{p:04d}.parquet"
        edges.iloc[start:end].to_parquet(
            output_dir / part_name, engine="fastparquet", index=False
        )
        part_files.append(part_name)

    manifest = {
        "view": prefix,
        "k_target": k,
        "nodes": int(N),
        "nnz": int(len(edges)),
        "parts": part_files,
        "note": note or f"row-stochastic; {prefix}",
    }
    manifest_path = output_dir / f"{prefix}_k{k}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def load_manifest_flexible(prefix, base_dir, k=50):
    """Load a manifest JSON, tolerating key name variants.

    Tries ``{prefix}_k{k}_manifest.json`` then ``{prefix}_manifest.json``.
    Normalises the parts list key ('parts', 'files', 'part_files').

    Args:
        prefix: View name prefix.
        base_dir: Directory containing the manifest.
        k: k value for filename pattern.

    Returns:
        (manifest_dict, parts_list, parent_dir)

    Raises:
        FileNotFoundError: If no matching manifest is found.
    """
    base_dir = Path(base_dir)
    candidates = [
        base_dir / f"{prefix}_k{k}_manifest.json",
        base_dir / f"{prefix}_manifest.json",
    ]
    for cand in candidates:
        if cand.exists():
            with open(cand) as f:
                man = json.load(f)
            parts = man.get("parts") or man.get("files") or man.get("part_files") or []
            return man, parts, cand.parent
    raise FileNotFoundError(f"No manifest found for {prefix} in {base_dir}")


def load_edges_from_manifest(prefix, base_dir, k=50):
    """Load all edge partitions from a manifest and concatenate.

    Args:
        prefix: View name prefix.
        base_dir: Directory containing the manifest.
        k: k value for filename pattern.

    Returns:
        (edges_df, manifest_dict)
    """
    man, parts, manifest_dir = load_manifest_flexible(prefix, base_dir, k=k)
    dfs = []
    for part_file in parts:
        pf = manifest_dir / part_file
        if pf.exists():
            dfs.append(pd.read_parquet(pf, engine="fastparquet"))
    if not dfs:
        return pd.DataFrame(columns=["row", "col", "val"]), man
    return pd.concat(dfs, ignore_index=True), man


def build_neighbor_dict(edges_df, doc_set=None):
    """Build a {doc_idx: set(neighbor_indices)} dictionary from edges.

    Args:
        edges_df: DataFrame with 'row' and 'col' columns.
        doc_set: If given, only include rows in this set.

    Returns:
        Dict mapping doc_idx to set of neighbor indices.
    """
    nbr = {}
    for _, r in edges_df.iterrows():
        row_i = int(r["row"])
        col_j = int(r["col"])
        if doc_set is not None and row_i not in doc_set:
            continue
        if row_i not in nbr:
            nbr[row_i] = set()
        nbr[row_i].add(col_j)
    return nbr


def load_csr_from_manifest(prefix, N, base_dir, k=50):
    """Load partitioned COO edges and return a scipy CSR matrix.

    Args:
        prefix: View name prefix.
        N: Matrix dimension.
        base_dir: Directory containing the manifest.
        k: k value for filename pattern.

    Returns:
        scipy CSR matrix of shape (N, N).
    """
    man, parts, manifest_dir = load_manifest_flexible(prefix, base_dir, k=k)
    dfs = []
    for pf in parts:
        path = manifest_dir / pf
        if path.exists():
            dfs.append(pd.read_parquet(path, engine="fastparquet"))
    if not dfs:
        return sparse.csr_matrix((N, N))
    edges = pd.concat(dfs, ignore_index=True)
    S = sparse.coo_matrix(
        (edges["val"].values, (edges["row"].values, edges["col"].values)),
        shape=(N, N)
    ).tocsr()
    return S
