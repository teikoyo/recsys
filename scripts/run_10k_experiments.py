#!/usr/bin/env python3
"""
10K Fusion Method Improvement Experiments

Five experiments to improve fusion quality on the 10K content subset:
  1. weight_sweep  – Naive weight α scan over {0.3..0.9}
  2. selective     – Selective fusion (content docs fused, rest Meta-only)
  3. quality       – Content quality filtering before fusion
  4. grid          – Fixed 4-view weight grid (ablation)
  5. k_sweep       – k_sim parameter sweep for content similarity

Usage:
    python scripts/run_10k_experiments.py --experiments weight_sweep selective quality grid k_sweep
    python scripts/run_10k_experiments.py --experiments weight_sweep  # run just one
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.content.evaluation import load_silver_standards
from src.content.similarity import (
    load_csr_from_manifest,
    save_partitioned_edges,
    sym_and_rownorm,
)
from src.metrics import (
    average_precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
W_TAG, W_DESC, W_CRE = 0.5, 0.3, 0.2
DESC_THRESHOLD = 0.2
N_TOTAL = 521735


# ---------------------------------------------------------------------------
# Core evaluation (operates on pre-built nbr dicts, avoids CSR reload)
# ---------------------------------------------------------------------------

def evaluate_from_nbrs(nbr_idx, method_name, subset, standards, k_eval=20):
    """Evaluate one method given pre-computed neighbor dicts.

    Same logic as eval_at_scale.evaluate_single_method_csr but accepts
    nbr_idx directly instead of a CSR matrix.
    """
    if not nbr_idx:
        print(f"  WARNING: No neighbors for {method_name}")
        return None

    doc_tags = standards.doc_tags
    idf_map = standards.idf_map
    S_bm25 = standards.S_bm25
    creator_ids = standards.creator_ids
    creator_counts = standards.creator_counts
    N = standards.N

    results = {"method": method_name}

    # --- Tag ---
    tag_ndcgs = []
    tag_covered = 0
    for doc_i in subset:
        if doc_i not in nbr_idx or doc_i not in doc_tags:
            continue
        tags_i = set(doc_tags[doc_i])
        if not tags_i:
            continue
        tag_covered += 1
        neighbors = nbr_idx[doc_i]
        gains, binary = [], []
        for j in neighbors[:k_eval]:
            j = int(j)
            tags_j = set(doc_tags.get(j, []))
            inter = tags_i & tags_j
            union = tags_i | tags_j
            gain = (sum(idf_map.get(t, 1.0) for t in inter)
                    / sum(idf_map.get(t, 1.0) for t in union)) if union else 0.0
            gains.append(gain)
            binary.append(1.0 if len(inter) >= 1 else 0.0)
        gains = np.array(gains, dtype=np.float64)
        binary = np.array(binary, dtype=np.float64)
        ideal = np.sort(gains)[::-1]
        tag_ndcgs.append(ndcg_at_k(gains, ideal))

    results["tag_ndcg"] = np.mean(tag_ndcgs) if tag_ndcgs else 0.0
    results["tag_n"] = tag_covered

    # --- Desc ---
    desc_ndcgs = []
    desc_covered = 0
    for doc_i in subset:
        if doc_i not in nbr_idx:
            continue
        neighbors = nbr_idx[doc_i]
        row_start = S_bm25.indptr[doc_i]
        row_end = S_bm25.indptr[doc_i + 1]
        if row_end - row_start == 0:
            continue
        desc_covered += 1
        bm25_cols = S_bm25.indices[row_start:row_end]
        bm25_vals = S_bm25.data[row_start:row_end]
        bm25_lookup = dict(zip(bm25_cols.astype(int), bm25_vals.astype(float)))
        gains, binary = [], []
        for j in neighbors[:k_eval]:
            j = int(j)
            sim = bm25_lookup.get(j, 0.0)
            gains.append(sim)
            binary.append(1.0 if sim > DESC_THRESHOLD else 0.0)
        gains = np.array(gains, dtype=np.float64)
        binary = np.array(binary, dtype=np.float64)
        ideal = np.sort(gains)[::-1]
        desc_ndcgs.append(ndcg_at_k(gains, ideal))

    results["desc_ndcg"] = np.mean(desc_ndcgs) if desc_ndcgs else 0.0
    results["desc_n"] = desc_covered

    # --- Creator ---
    cre_ndcgs = []
    cre_covered = 0
    for doc_i in subset:
        if doc_i not in nbr_idx:
            continue
        neighbors = nbr_idx[doc_i]
        cid_i = creator_ids[doc_i] if doc_i < N else 0
        if cid_i == 0:
            continue
        cre_covered += 1
        gains, binary = [], []
        for j in neighbors[:k_eval]:
            j = int(j)
            cid_j = creator_ids[j] if j < N else 0
            match = 1.0 if (cid_j == cid_i and cid_j > 0) else 0.0
            gains.append(match)
            binary.append(match)
        gains = np.array(gains, dtype=np.float64)
        binary = np.array(binary, dtype=np.float64)
        ideal = np.sort(gains)[::-1]
        cre_ndcgs.append(ndcg_at_k(gains, ideal))

    results["cre_ndcg"] = np.mean(cre_ndcgs) if cre_ndcgs else 0.0
    results["cre_n"] = cre_covered

    results["unified_ndcg"] = (
        W_TAG * results["tag_ndcg"]
        + W_DESC * results["desc_ndcg"]
        + W_CRE * results["cre_ndcg"]
    )
    print(
        f"  {method_name}: unified={results['unified_ndcg']:.4f} "
        f"(tag={results['tag_ndcg']:.4f}, desc={results['desc_ndcg']:.4f}, "
        f"cre={results['cre_ndcg']:.4f})"
    )
    return results


# ---------------------------------------------------------------------------
# On-the-fly fusion: fuse two CSR rows without materialising full matrix
# ---------------------------------------------------------------------------

def build_topk_fused_onthefly(S_meta, S_content, alpha_meta, doc_indices, k_eval=20):
    """Row-wise on-the-fly fusion of S_meta and S_content for selected docs.

    S_fused[i] = alpha_meta * S_meta[i] + (1-alpha_meta) * S_content[i],
    then top-K and L1 normalise.

    Returns nbr_idx dict mapping doc_i -> np.array of neighbor indices (sorted by score).
    """
    alpha_content = 1.0 - alpha_meta
    nbr_idx = {}

    for doc_i in doc_indices:
        fused = defaultdict(float)

        # Meta row
        m_start, m_end = S_meta.indptr[doc_i], S_meta.indptr[doc_i + 1]
        m_cols = S_meta.indices[m_start:m_end]
        m_vals = S_meta.data[m_start:m_end]
        for jj in range(len(m_cols)):
            fused[int(m_cols[jj])] += alpha_meta * float(m_vals[jj])

        # Content row
        c_start, c_end = S_content.indptr[doc_i], S_content.indptr[doc_i + 1]
        c_cols = S_content.indices[c_start:c_end]
        c_vals = S_content.data[c_start:c_end]
        for jj in range(len(c_cols)):
            fused[int(c_cols[jj])] += alpha_content * float(c_vals[jj])

        if not fused:
            continue

        # Top-K
        if len(fused) > k_eval:
            items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_eval]
        else:
            items = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        nbr_idx[doc_i] = np.array([j for j, _ in items], dtype=np.int64)

    return nbr_idx


def build_topk_from_csr(S_csr, k_eval, subset):
    """Extract top-K neighbors from CSR matrix for docs in subset."""
    nbr_idx = {}
    for doc_i in subset:
        start = S_csr.indptr[doc_i]
        end = S_csr.indptr[doc_i + 1]
        if end == start:
            continue
        cols = S_csr.indices[start:end]
        vals = S_csr.data[start:end]
        if len(cols) <= k_eval:
            order = np.argsort(-vals)
            nbr_idx[doc_i] = cols[order].astype(np.int64)
        else:
            top_k = np.argpartition(vals, -k_eval)[-k_eval:]
            top_k = top_k[np.argsort(vals[top_k])[::-1]]
            nbr_idx[doc_i] = cols[top_k].astype(np.int64)
    return nbr_idx


def build_selective_nbrs(S_meta, S_content, alpha_meta, d_content_set,
                         full_subset, k_eval=20):
    """Selective fusion: fused neighbors for content docs, meta-only for rest.

    Args:
        S_meta: CSR meta matrix.
        S_content: CSR content matrix.
        alpha_meta: Weight for meta in fusion.
        d_content_set: Set of doc indices with content.
        full_subset: Full evaluation subset.
        k_eval: Top-K neighbors.

    Returns:
        nbr_idx dict for all docs in full_subset.
    """
    # Content docs: on-the-fly fusion
    content_docs = [d for d in full_subset if d in d_content_set]
    non_content_docs = [d for d in full_subset if d not in d_content_set]

    nbr_fused = build_topk_fused_onthefly(
        S_meta, S_content, alpha_meta, content_docs, k_eval
    )
    nbr_meta = build_topk_from_csr(S_meta, k_eval, non_content_docs)

    # Merge
    nbr_fused.update(nbr_meta)
    return nbr_fused


# ---------------------------------------------------------------------------
# 4-view on-the-fly fusion
# ---------------------------------------------------------------------------

def build_topk_4view_onthefly(S_dict, weights, doc_indices, k_eval=20):
    """Fuse 4 views with fixed weights for given docs.

    Args:
        S_dict: Dict of view_name -> CSR matrix.
        weights: Dict of view_name -> float weight.
        doc_indices: Iterable of doc indices.
        k_eval: Top-K to retain.

    Returns:
        nbr_idx dict.
    """
    views = list(weights.keys())
    nbr_idx = {}

    for doc_i in doc_indices:
        fused = defaultdict(float)
        for v in views:
            w = weights[v]
            if w < 1e-9:
                continue
            S = S_dict[v]
            start, end = S.indptr[doc_i], S.indptr[doc_i + 1]
            cols = S.indices[start:end]
            vals = S.data[start:end]
            for jj in range(len(cols)):
                fused[int(cols[jj])] += w * float(vals[jj])

        if not fused:
            continue

        if len(fused) > k_eval:
            items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_eval]
        else:
            items = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        nbr_idx[doc_i] = np.array([j for j, _ in items], dtype=np.int64)

    return nbr_idx


# ---------------------------------------------------------------------------
# Quality score computation
# ---------------------------------------------------------------------------

def compute_content_quality_scores(scale_dir):
    """Compute per-dataset quality scores from col_profiles.parquet.

    Quality = n_cols * numeric_ratio * (1 - mean_missing) * mean_unique_capped

    Returns DataFrame with columns: doc_idx, quality, n_cols, numeric_ratio,
    mean_missing, mean_unique.
    """
    prof_path = Path(scale_dir) / "col_profiles.parquet"
    if not prof_path.exists():
        raise FileNotFoundError(f"col_profiles.parquet not found in {scale_dir}")

    profiles = pd.read_parquet(prof_path, engine="fastparquet")
    rows = []
    for doc_idx, grp in profiles.groupby("doc_idx"):
        n_cols = len(grp)
        numeric_ratio = (grp["dtype"] == "numeric").mean()
        mean_missing = grp["missing_pct"].mean() / 100.0
        mean_unique = min(grp["unique_pct"].mean() / 100.0, 0.95)

        quality = n_cols * max(numeric_ratio, 0.1) * (1.0 - mean_missing) * max(mean_unique, 0.01)
        rows.append({
            "doc_idx": int(doc_idx),
            "quality": quality,
            "n_cols": n_cols,
            "numeric_ratio": numeric_ratio,
            "mean_missing": mean_missing,
            "mean_unique": mean_unique,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Experiment runners
# ===========================================================================

def run_weight_sweep(S_meta, S_content, d_content_set, standards, k_eval,
                     output_dir):
    """Experiment 1: Naive weight α sweep."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Naive Weight Sweep")
    print("=" * 70)

    alphas = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    results = []

    for alpha in alphas:
        name = f"Naive-α={alpha:.2f}"
        print(f"\n[exp1] Testing α_meta={alpha:.2f} (α_content={1-alpha:.2f})")
        t0 = time.time()

        nbr_idx = build_topk_fused_onthefly(
            S_meta, S_content, alpha, list(d_content_set), k_eval
        )
        res = evaluate_from_nbrs(nbr_idx, name, d_content_set, standards, k_eval)
        if res:
            res["alpha_meta"] = alpha
            res["alpha_content"] = 1.0 - alpha
            results.append(res)
        print(f"  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp1_weight_sweep.csv", index=False)

    # Find best
    if not df.empty:
        best = df.loc[df["unified_ndcg"].idxmax()]
        print(f"\n[exp1] Best: α_meta={best['alpha_meta']:.2f} → "
              f"unified={best['unified_ndcg']:.4f}")
        return float(best["alpha_meta"])

    return 0.5


def run_selective_fusion(S_meta, S_content, d_content_set, standards,
                         best_alpha, k_eval, output_dir, seed=42):
    """Experiment 2: Selective fusion on D_content and dilution subsets."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Selective Fusion")
    print("=" * 70)

    N = standards.N
    rng = np.random.RandomState(seed)
    all_indices = np.arange(N)
    non_content = np.array([i for i in all_indices if i not in d_content_set],
                           dtype=np.int64)

    subsets = [
        ("D_content_only", d_content_set, 1.0),
    ]
    for size in [50000, 100000]:
        n_extra = min(size - len(d_content_set), len(non_content))
        sampled = rng.choice(non_content, n_extra, replace=False)
        subset = d_content_set | set(sampled.tolist())
        ratio = len(d_content_set) / len(subset)
        subsets.append((f"dilution_{size}", subset, ratio))

    results = []
    for subset_name, subset, ratio in subsets:
        print(f"\n[exp2] {subset_name}: {len(subset)} docs, "
              f"coverage={ratio:.2%}, α_meta={best_alpha:.2f}")
        t0 = time.time()

        nbr_idx = build_selective_nbrs(
            S_meta, S_content, best_alpha, d_content_set, subset, k_eval
        )
        name = f"Selective-α={best_alpha:.2f}"
        res = evaluate_from_nbrs(nbr_idx, name, subset, standards, k_eval)
        if res:
            res["subset_type"] = subset_name
            res["subset_size"] = len(subset)
            res["content_ratio"] = ratio
            res["alpha_meta"] = best_alpha
            results.append(res)
        print(f"  ({time.time() - t0:.1f}s)")

        # Also evaluate meta-only baseline on same subset for comparison
        print(f"  [baseline] Meta-only on {subset_name}")
        nbr_meta = build_topk_from_csr(S_meta, k_eval, subset)
        res_meta = evaluate_from_nbrs(
            nbr_meta, "Meta-only", subset, standards, k_eval
        )
        if res_meta:
            res_meta["subset_type"] = subset_name
            res_meta["subset_size"] = len(subset)
            res_meta["content_ratio"] = ratio
            results.append(res_meta)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp2_selective_fusion.csv", index=False)

    # Show improvement summary
    for subset_name, _, _ in subsets:
        sub = df[df["subset_type"] == subset_name]
        meta_row = sub[sub["method"] == "Meta-only"]
        sel_row = sub[sub["method"].str.startswith("Selective")]
        if not meta_row.empty and not sel_row.empty:
            meta_u = meta_row.iloc[0]["unified_ndcg"]
            sel_u = sel_row.iloc[0]["unified_ndcg"]
            pct = (sel_u - meta_u) / abs(meta_u) * 100 if meta_u != 0 else 0
            print(f"  {subset_name}: Selective={sel_u:.4f} vs Meta={meta_u:.4f} "
                  f"({pct:+.1f}%)")

    return df


def run_quality_filter(S_meta, S_content, d_content_set, standards,
                       best_alpha, k_eval, scale_dir, output_dir):
    """Experiment 3: Content quality filtering."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Content Quality Filtering")
    print("=" * 70)

    quality_df = compute_content_quality_scores(scale_dir)
    print(f"[exp3] Quality scores computed for {len(quality_df)} datasets")
    print(f"  Quality stats: min={quality_df['quality'].min():.3f}, "
          f"median={quality_df['quality'].median():.3f}, "
          f"max={quality_df['quality'].max():.3f}")

    # Save quality scores
    quality_df.to_csv(output_dir / "exp3_quality_scores.csv", index=False)

    # Try different thresholds: p25, median, p75
    thresholds = {
        "no_filter": 0.0,
        "p10": quality_df["quality"].quantile(0.10),
        "p25": quality_df["quality"].quantile(0.25),
        "median": quality_df["quality"].quantile(0.50),
        "p75": quality_df["quality"].quantile(0.75),
    }

    results = []
    quality_doc_set = set(quality_df["doc_idx"].values)

    for thresh_name, thresh_val in thresholds.items():
        high_quality = set(
            quality_df[quality_df["quality"] >= thresh_val]["doc_idx"].values
        )
        # Docs in d_content but not in quality_df (no profiles) are excluded
        # from content fusion and fall back to meta-only
        fused_set = high_quality & d_content_set
        fallback_set = d_content_set - fused_set

        print(f"\n[exp3] Threshold={thresh_name} ({thresh_val:.3f}): "
              f"{len(fused_set)} fused, {len(fallback_set)} fallback to meta")

        # Build selective: high-quality content docs get fusion, rest meta-only
        nbr_fused = build_topk_fused_onthefly(
            S_meta, S_content, best_alpha, list(fused_set), k_eval
        )
        nbr_fallback = build_topk_from_csr(S_meta, k_eval, fallback_set)
        nbr_fused.update(nbr_fallback)

        name = f"Quality-{thresh_name}"
        res = evaluate_from_nbrs(nbr_fused, name, d_content_set, standards, k_eval)
        if res:
            res["threshold_name"] = thresh_name
            res["threshold_value"] = thresh_val
            res["n_fused"] = len(fused_set)
            res["n_fallback"] = len(fallback_set)
            res["alpha_meta"] = best_alpha
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp3_quality_filter.csv", index=False)

    if not df.empty:
        best = df.loc[df["unified_ndcg"].idxmax()]
        print(f"\n[exp3] Best: {best['method']} → unified={best['unified_ndcg']:.4f}")

    return df


def run_4view_grid(S_tag, S_text, S_beh, S_content, d_content_set,
                   standards, k_eval, output_dir):
    """Experiment 4: Fixed 4-view weight grid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Fixed 4-View Weight Grid")
    print("=" * 70)

    S_dict = {"tag": S_tag, "text": S_text, "beh": S_beh, "content": S_content}

    configs = {
        "Beh-heavy":    {"tag": 0.10, "text": 0.10, "beh": 0.50, "content": 0.30},
        "Equal-4":      {"tag": 0.25, "text": 0.25, "beh": 0.25, "content": 0.25},
        "Content-heavy": {"tag": 0.10, "text": 0.10, "beh": 0.30, "content": 0.50},
        "Beh+Content":  {"tag": 0.05, "text": 0.05, "beh": 0.45, "content": 0.45},
        "Tag+Content":  {"tag": 0.30, "text": 0.10, "beh": 0.30, "content": 0.30},
    }

    results = []
    for config_name, weights in configs.items():
        print(f"\n[exp4] {config_name}: {weights}")
        t0 = time.time()

        nbr_idx = build_topk_4view_onthefly(
            S_dict, weights, list(d_content_set), k_eval
        )
        res = evaluate_from_nbrs(nbr_idx, config_name, d_content_set, standards, k_eval)
        if res:
            res.update({f"w_{k}": v for k, v in weights.items()})
            results.append(res)
        print(f"  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp4_4view_grid.csv", index=False)

    if not df.empty:
        best = df.loc[df["unified_ndcg"].idxmax()]
        print(f"\n[exp4] Best: {best['method']} → unified={best['unified_ndcg']:.4f}")

    return df


def run_k_sweep(scale_dir, d_content_set, standards, k_eval, output_dir):
    """Experiment 5: k_sim parameter sweep for content similarity.

    Rebuilds S_tabcontent from Z_tabcontent for each k value, then
    evaluates content-only on D_content.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: k_sim Parameter Sweep")
    print("=" * 70)

    scale_dir = Path(scale_dir)
    Z_path = scale_dir / "Z_tabcontent.parquet"
    if not Z_path.exists():
        print("[exp5] Z_tabcontent.parquet not found, skipping")
        return pd.DataFrame()

    Z_df = pd.read_parquet(Z_path, engine="fastparquet")
    doc_indices = Z_df["doc_idx"].values.astype(np.int64)
    feat_cols = [c for c in Z_df.columns if c.startswith("f")]
    embed_dim = len(feat_cols)
    Z = Z_df[feat_cols].values.astype(np.float32)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(norms, 1e-12)
    Z = np.ascontiguousarray(Z)
    B = len(Z)
    N = standards.N

    print(f"[exp5] Loaded Z_tabcontent: {B} vectors, dim={embed_dim}")

    # Build FAISS index once
    try:
        import faiss
        index = faiss.IndexFlatIP(embed_dim)
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[exp5] Using FAISS GPU")
        except Exception:
            print("[exp5] Using FAISS CPU")
        index.add(Z)
    except ImportError:
        print("[exp5] FAISS not available, using sklearn fallback")
        index = None

    k_values = [20, 30, 50, 75, 100]
    results = []

    for k_sim in k_values:
        print(f"\n[exp5] k_sim={k_sim}")
        t0 = time.time()

        k_search = min(k_sim + 1, B)
        if index is not None:
            scores, idxs = index.search(Z, k_search)
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k_search, metric="cosine",
                                  algorithm="brute")
            nn.fit(Z)
            distances, idxs = nn.kneighbors(Z)
            scores = 1.0 - distances

        # Build COO edges
        rows_list, cols_list, vals_list = [], [], []
        for i in range(B):
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

        # Symmetrise and row-normalise
        sym_rows, sym_cols, sym_vals = sym_and_rownorm(
            rows_arr, cols_arr, vals_arr, N
        )

        # Build CSR directly (don't save to disk for sweep)
        S_content = sparse.coo_matrix(
            (sym_vals, (sym_rows, sym_cols)), shape=(N, N)
        ).tocsr()

        # Evaluate content-only
        nbr_idx = build_topk_from_csr(S_content, k_eval, d_content_set)
        res = evaluate_from_nbrs(
            nbr_idx, f"Content-k={k_sim}", d_content_set, standards, k_eval
        )
        if res:
            res["k_sim"] = k_sim
            res["nnz"] = S_content.nnz
            results.append(res)

        print(f"  nnz={S_content.nnz}, ({time.time() - t0:.1f}s)")
        del S_content
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp5_k_sweep.csv", index=False)

    if not df.empty:
        best = df.loc[df["unified_ndcg"].idxmax()]
        print(f"\n[exp5] Best: k_sim={best['k_sim']:.0f} → "
              f"unified={best['unified_ndcg']:.4f}")

    return df


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="10K Fusion Improvement Experiments",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=["weight_sweep", "selective", "quality", "grid", "k_sweep"],
        default=["weight_sweep", "selective", "quality", "grid", "k_sweep"],
        help="Which experiments to run",
    )
    parser.add_argument("--k-eval", type=int, default=20)
    parser.add_argument("--k-sim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-total", type=int, default=N_TOTAL)
    args = parser.parse_args()

    TMP_DIR = ROOT / "tmp"
    SCALE_DIR = TMP_DIR / "content" / "scale_10000"
    OUTPUT_DIR = SCALE_DIR / "experiments"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    N = args.n_total

    # ------------------------------------------------------------------
    # Load D_content
    # ------------------------------------------------------------------
    dc_path = SCALE_DIR / "d_content.parquet"
    d_content = pd.read_parquet(dc_path, engine="fastparquet")
    d_content_set = set(d_content["doc_idx"].astype(int).values)
    print(f"[main] D_content: {len(d_content_set)} docs")

    # ------------------------------------------------------------------
    # Load silver standards
    # ------------------------------------------------------------------
    print("[main] Loading silver standards...")
    t0 = time.time()
    standards = load_silver_standards(TMP_DIR, N)
    print(f"[main] Standards loaded ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Load shared matrices
    # ------------------------------------------------------------------
    S_meta = None
    S_content = None
    S_tag = S_text = S_beh = None

    needs_meta = any(e in args.experiments
                     for e in ["weight_sweep", "selective", "quality", "grid"])
    needs_content = any(e in args.experiments
                        for e in ["weight_sweep", "selective", "quality", "grid"])
    needs_single = "grid" in args.experiments

    if needs_meta:
        print("[main] Loading S_fused3 (meta)...")
        t0 = time.time()
        S_meta = load_csr_from_manifest("S_fused3_symrow", N, TMP_DIR, k=args.k_sim)
        print(f"  nnz={S_meta.nnz} ({time.time() - t0:.1f}s)")

    if needs_content:
        print("[main] Loading S_tabcontent (content)...")
        t0 = time.time()
        S_content = load_csr_from_manifest(
            "S_tabcontent_symrow", N, SCALE_DIR, k=args.k_sim
        )
        print(f"  nnz={S_content.nnz} ({time.time() - t0:.1f}s)")

    if needs_single:
        print("[main] Loading single-view matrices for grid experiment...")
        t0 = time.time()
        S_tag = load_csr_from_manifest("S_tag_symrow", N, TMP_DIR, k=args.k_sim)
        S_text = load_csr_from_manifest("S_text_symrow", N, TMP_DIR, k=args.k_sim)
        S_beh = load_csr_from_manifest("S_beh_symrow", N, TMP_DIR, k=args.k_sim)
        print(f"  Loaded tag/text/beh ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    best_alpha = 0.5  # default; updated by exp1

    if "weight_sweep" in args.experiments:
        best_alpha = run_weight_sweep(
            S_meta, S_content, d_content_set, standards, args.k_eval, OUTPUT_DIR
        )
        # Save best alpha for other experiments
        with open(OUTPUT_DIR / "best_alpha.json", "w") as f:
            json.dump({"best_alpha_meta": best_alpha}, f)
    else:
        # Try to load from previous run
        alpha_path = OUTPUT_DIR / "best_alpha.json"
        if alpha_path.exists():
            best_alpha = json.loads(alpha_path.read_text())["best_alpha_meta"]
            print(f"[main] Loaded best_alpha={best_alpha} from previous run")

    if "selective" in args.experiments:
        run_selective_fusion(
            S_meta, S_content, d_content_set, standards,
            best_alpha, args.k_eval, OUTPUT_DIR, seed=args.seed
        )

    if "quality" in args.experiments:
        run_quality_filter(
            S_meta, S_content, d_content_set, standards,
            best_alpha, args.k_eval, SCALE_DIR, OUTPUT_DIR
        )

    if "grid" in args.experiments:
        run_4view_grid(
            S_tag, S_text, S_beh, S_content, d_content_set,
            standards, args.k_eval, OUTPUT_DIR
        )

    if "k_sweep" in args.experiments:
        run_k_sweep(
            SCALE_DIR, d_content_set, standards, args.k_eval, OUTPUT_DIR
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")

    # Collect all experiment CSVs
    all_csvs = sorted(OUTPUT_DIR.glob("exp*.csv"))
    for csv_path in all_csvs:
        print(f"  {csv_path.name}")


if __name__ == "__main__":
    main()
