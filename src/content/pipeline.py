#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content Feature Pipeline

Extracts the full content-view construction pipeline from NB02 into a
callable function.  Supports configurable MAX_ROWS / MAX_COLS for ablation
experiments and automatic CUDA detection for GPU-ready operation.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .sampling import (
    sample_table,
    profile_table,
    col_to_description,
)
from .encoding import aggregate_dataset_vector
from .similarity import (
    save_partitioned_edges,
    sym_and_rownorm,
    load_csr_from_manifest,
)
from .fusion import fuse_views


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return ``"cuda"`` if a CUDA GPU is available, else ``"cpu"``."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Full content pipeline
# ---------------------------------------------------------------------------

def run_content_pipeline(
    d_content_path,
    main_tables_path,
    tab_raw_dir,
    output_dir,
    *,
    max_rows: int = 1024,
    max_cols: int = 60,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_dim: int = 384,
    k_sim: int = 50,
    n_total: int = 521735,
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, Path]:
    """Run the full content-view construction pipeline.

    Steps:
      1. Load d_content + main_tables, merge
      2. sample_table -> profile -> col_to_description for each dataset
      3. Sentence-Transformer encode all descriptions
      4. aggregate_dataset_vector -> Z_tabcontent
      5. FAISS kNN -> COO edges
      6. sym_and_rownorm -> save S_tabcontent

    Args:
        d_content_path: Path to ``d_content.parquet``.
        main_tables_path: Path to ``main_tables.parquet``.
        tab_raw_dir: Root of tabular raw data (unused directly but paths
            are already absolute in main_tables).
        output_dir: Where to write outputs.
        max_rows: Maximum rows to sample per table.
        max_cols: Maximum columns to keep per table.
        embed_model: Sentence-transformer model name.
        embed_dim: Embedding dimension (must match model).
        k_sim: Number of nearest neighbors for similarity graph.
        n_total: Total corpus size (for sparse matrix shape).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = detect_device()

    outputs: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # 1. Load and merge
    # ------------------------------------------------------------------
    d_content = pd.read_parquet(d_content_path, engine="fastparquet")
    main_tables = pd.read_parquet(main_tables_path, engine="fastparquet")
    mt_ren = main_tables.rename(columns={"DatasetId": "Id"})
    work = d_content.merge(mt_ren, on=["Id", "doc_idx"], how="inner")
    work = work[work["main_table_path"] != ""].copy()
    print(f"[pipeline] Datasets with tables: {len(work)}")

    # ------------------------------------------------------------------
    # 2. Profile and generate descriptions
    # ------------------------------------------------------------------
    all_profiles = []
    all_descriptions = []
    n_success = 0

    for i, (_, row) in enumerate(work.iterrows()):
        ds_id = int(row["Id"])
        doc_idx = int(row["doc_idx"])
        table_path = row["main_table_path"]

        df = sample_table(table_path, max_rows, max_cols)
        if df is None or df.empty:
            continue

        profiles = profile_table(df)
        n_success += 1

        for cs in profiles:
            desc = col_to_description(cs)
            all_profiles.append({
                "DatasetId": ds_id,
                "doc_idx": doc_idx,
                "col_name": cs.name,
                "dtype": cs.dtype,
                "missing_pct": cs.missing_pct,
                "n_unique": cs.n_unique,
                "unique_pct": cs.unique_pct,
            })
            all_descriptions.append({
                "DatasetId": ds_id,
                "doc_idx": doc_idx,
                "col_name": cs.name,
                "description": desc,
                "dtype": cs.dtype,
                "missing_pct": cs.missing_pct,
                "unique_pct": cs.unique_pct,
            })

        if (i + 1) % 200 == 0:
            print(f"[pipeline] Profiled {i + 1}/{len(work)} datasets")

    col_profiles_df = pd.DataFrame(all_profiles)
    col_descriptions_df = pd.DataFrame(all_descriptions)

    col_profiles_df.to_parquet(output_dir / "col_profiles.parquet", engine="fastparquet")
    col_descriptions_df.to_parquet(output_dir / "col_descriptions.parquet", engine="fastparquet")
    outputs["col_profiles"] = output_dir / "col_profiles.parquet"
    outputs["col_descriptions"] = output_dir / "col_descriptions.parquet"
    print(f"[pipeline] Profiled {n_success} datasets, {len(col_profiles_df)} columns")

    # ------------------------------------------------------------------
    # 3. Encode descriptions
    # ------------------------------------------------------------------
    descriptions = col_descriptions_df["description"].tolist()

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        print(f"[pipeline] Encoding {len(descriptions)} descriptions on {device}")
        model = SentenceTransformer(embed_model, device=device)
        embeddings = model.encode(
            descriptions,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        print("[pipeline] sentence-transformers not available, using random embeddings")
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(len(descriptions), embed_dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

    # Save col embeddings
    emb_data = col_descriptions_df[["DatasetId", "doc_idx", "col_name"]].copy()
    emb_cols_df = pd.DataFrame(embeddings, columns=[f"e{j}" for j in range(embed_dim)])
    emb_df = pd.concat([emb_data.reset_index(drop=True), emb_cols_df], axis=1)
    emb_df.to_parquet(output_dir / "col_embeddings.parquet", engine="fastparquet")
    outputs["col_embeddings"] = output_dir / "col_embeddings.parquet"

    # ------------------------------------------------------------------
    # 4. Aggregate to dataset vectors
    # ------------------------------------------------------------------
    emb_cols = [f"e{j}" for j in range(embed_dim)]

    merged = emb_df.merge(
        col_profiles_df[["DatasetId", "col_name", "dtype", "missing_pct", "unique_pct"]],
        on=["DatasetId", "col_name"],
        how="left",
        suffixes=("", "_prof"),
    )

    dtype_col = "dtype_prof" if "dtype_prof" in merged.columns else "dtype"
    miss_col = "missing_pct_prof" if "missing_pct_prof" in merged.columns else "missing_pct"
    uniq_col = "unique_pct_prof" if "unique_pct_prof" in merged.columns else "unique_pct"

    z_rows = []
    for ds_id in merged["DatasetId"].unique():
        sub = merged[merged["DatasetId"] == ds_id]
        doc_idx = int(sub["doc_idx"].iloc[0])
        col_embs = sub[emb_cols].values.astype(np.float32)
        col_stats = [
            {"dtype": r[dtype_col], "missing_pct": r[miss_col], "unique_pct": r[uniq_col]}
            for _, r in sub.iterrows()
        ]
        z = aggregate_dataset_vector(col_embs, col_stats)
        row_dict = {"doc_idx": doc_idx}
        for j in range(embed_dim):
            row_dict[f"f{j}"] = float(z[j])
        z_rows.append(row_dict)

    Z_df = pd.DataFrame(z_rows)
    Z_df["doc_idx"] = Z_df["doc_idx"].astype(int)
    Z_df = Z_df.sort_values("doc_idx").reset_index(drop=True)
    Z_df.to_parquet(output_dir / "Z_tabcontent.parquet", engine="fastparquet")
    outputs["Z_tabcontent"] = output_dir / "Z_tabcontent.parquet"
    print(f"[pipeline] Z_tabcontent: {Z_df.shape}")

    # ------------------------------------------------------------------
    # 5. FAISS kNN -> COO edges
    # ------------------------------------------------------------------
    doc_indices = Z_df["doc_idx"].values.astype(np.int64)
    feat_cols = [f"f{j}" for j in range(embed_dim)]
    Z = Z_df[feat_cols].values.astype(np.float32)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(norms, 1e-12)
    Z = np.ascontiguousarray(Z)
    B_actual = len(Z)

    try:
        import faiss
        index = faiss.IndexFlatIP(embed_dim)
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[pipeline] Using FAISS GPU index")
        except Exception:
            print("[pipeline] Using FAISS CPU index")
        index.add(Z)
        k_search = min(k_sim + 1, B_actual)
        scores, idxs = index.search(Z, k_search)
    except ImportError:
        print("[pipeline] FAISS not available, using sklearn NearestNeighbors fallback")
        k_search = min(k_sim + 1, B_actual)
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(
                n_neighbors=k_search, metric="cosine", algorithm="brute",
            )
            nn.fit(Z)
            distances, idxs = nn.kneighbors(Z)
            scores = 1.0 - distances  # cosine similarity
        except ImportError:
            print("[pipeline] sklearn not available, using numpy dense fallback")
            sim_matrix = Z @ Z.T
            idxs = np.argsort(-sim_matrix, axis=1)[:, :k_search]
            scores = np.take_along_axis(sim_matrix, idxs, axis=1)

    rows_list, cols_list, vals_list = [], [], []
    for i in range(B_actual):
        global_i = int(doc_indices[i])
        for j_pos in range(k_search):
            local_j = int(idxs[i, j_pos])
            if local_j == i:
                continue
            global_j = int(doc_indices[local_j])
            val = float(scores[i, j_pos])
            if val > 0:
                rows_list.append(global_i)
                cols_list.append(global_j)
                vals_list.append(val)

    rows_arr = np.array(rows_list, dtype=np.int64)
    cols_arr = np.array(cols_list, dtype=np.int64)
    vals_arr = np.array(vals_list, dtype=np.float32)
    print(f"[pipeline] COO edges: {len(rows_arr)}")

    # ------------------------------------------------------------------
    # 6. Symmetrise, row-normalise, save
    # ------------------------------------------------------------------
    sym_rows, sym_cols, sym_vals = sym_and_rownorm(rows_arr, cols_arr, vals_arr, n_total)
    manifest_path = save_partitioned_edges(
        sym_rows, sym_cols, sym_vals, n_total,
        prefix="S_tabcontent_symrow", k=k_sim, output_dir=output_dir,
        note=f"content view; max_rows={max_rows}, max_cols={max_cols}",
    )
    outputs["S_tabcontent_manifest"] = manifest_path
    print(f"[pipeline] Saved S_tabcontent: {len(sym_rows)} edges")

    return outputs


# ---------------------------------------------------------------------------
# Naive fusion builder
# ---------------------------------------------------------------------------

def build_naive_fusion(
    S_fused3_dir,
    S_tabcontent_dir,
    output_dir,
    N: int,
    k_sim: int = 50,
) -> Path:
    """Build naive 4-view fusion: 0.5*S_fused3 + 0.5*S_tabcontent.

    Args:
        S_fused3_dir: Directory containing S_fused3_symrow manifest.
        S_tabcontent_dir: Directory containing S_tabcontent_symrow manifest.
        output_dir: Where to save the fused matrix.
        N: Corpus size.
        k_sim: k for manifest filenames and top-K trimming.

    Returns:
        Path to the saved manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    S_fused3 = load_csr_from_manifest("S_fused3_symrow", N, S_fused3_dir, k=k_sim)
    S_tabcontent = load_csr_from_manifest("S_tabcontent_symrow", N, S_tabcontent_dir, k=k_sim)

    naive_alpha = {
        "fused3": np.full(N, 0.5, dtype=np.float64),
        "tabcontent": np.full(N, 0.5, dtype=np.float64),
    }
    naive_S_dict = {"fused3": S_fused3, "tabcontent": S_tabcontent}
    naive_views = ["fused3", "tabcontent"]

    rows, cols, vals = fuse_views(naive_S_dict, naive_alpha, naive_views, N, K=k_sim)
    manifest_path = save_partitioned_edges(
        rows, cols, vals, N,
        prefix="S_naive4_symrow", k=k_sim, output_dir=output_dir,
        note="0.5*S_fused3 + 0.5*S_tabcontent; top-K + L1 norm",
    )
    print(f"[pipeline] Naive fusion saved: {len(rows)} edges")
    return manifest_path
