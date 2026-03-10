#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Content Pipeline and Evaluation at Scale

For a given target scale (10K/50K/100K), runs the full content-view pipeline
and evaluates all 8 methods on multiple evaluation subsets.

Usage:
    python scripts/run_content_at_scale.py --target 10000 --seed 42 --device auto

Inputs:
    tmp/content/scale_{N}/d_content.parquet
    tmp/content/scale_{N}/main_tables.parquet

Outputs:
    tmp/content/scale_{N}/  (pipeline intermediates)
    tmp/content/scale_{N}/results_content_only.csv
    tmp/content/scale_{N}/results_all_subsets.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.content.evaluation import (
    METHODS_CONFIG,
    evaluate_all_methods,
    evaluate_method_on_subset,
    load_silver_standards,
)
from src.content.fusion import (
    apply_consistency_adjustment,
    compute_adaptive_alpha,
    compute_rho,
    fuse_views,
)
from src.content.pipeline import (
    build_naive_fusion,
    detect_device,
    run_content_pipeline,
)
from src.content.similarity import (
    load_csr_from_manifest,
    save_partitioned_edges,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run content pipeline and evaluation at scale",
    )
    p.add_argument(
        "--target", type=int, required=True,
        help="Target scale (e.g. 10000, 50000, 100000)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--k-eval", type=int, default=20)
    p.add_argument("--k-sim", type=int, default=50)
    p.add_argument("--n-total", type=int, default=521735)
    p.add_argument(
        "--max-rows", type=int, default=1024,
        help="MAX_ROWS for table sampling",
    )
    p.add_argument(
        "--max-cols", type=int, default=60,
        help="MAX_COLS for table sampling",
    )
    p.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip pipeline if S_tabcontent already exists",
    )
    p.add_argument(
        "--skip-fusion", action="store_true",
        help="Skip fusion stages if manifests already exist",
    )
    p.add_argument(
        "--dilution-sizes", type=int, nargs="*", default=None,
        help="Additional subset sizes for dilution evaluation "
             "(e.g. 50000 100000). D_content-only is always evaluated.",
    )
    return p.parse_args()


def build_adaptive_fusion(
    scale_dir: Path,
    tmp_dir: Path,
    N: int,
    k_sim: int = 50,
) -> Path:
    """Build adaptive 4-view fusion using rho-based weights.

    Returns path to S_fused4_symrow manifest.
    """
    # Load all 4 view matrices
    S_tag = load_csr_from_manifest("S_tag_symrow", N, tmp_dir, k=k_sim)
    S_text = load_csr_from_manifest("S_text_symrow", N, tmp_dir, k=k_sim)
    S_beh = load_csr_from_manifest("S_beh_symrow", N, tmp_dir, k=k_sim)
    S_tabcontent = load_csr_from_manifest("S_tabcontent_symrow", N, scale_dir, k=k_sim)

    # Compute rho for each view
    rho_dict = {
        "tag": compute_rho(S_tag),
        "text": compute_rho(S_text),
        "beh": compute_rho(S_beh),
        "tabcontent": compute_rho(S_tabcontent),
    }
    views = ["tag", "text", "beh", "tabcontent"]
    alpha = compute_adaptive_alpha(rho_dict, views)

    S_dict = {
        "tag": S_tag,
        "text": S_text,
        "beh": S_beh,
        "tabcontent": S_tabcontent,
    }

    rows, cols, vals = fuse_views(S_dict, alpha, views, N, K=k_sim)
    manifest_path = save_partitioned_edges(
        rows, cols, vals, N,
        prefix="S_fused4_symrow", k=k_sim, output_dir=scale_dir,
        note="4-view rho-adaptive fusion; top-K + L1 norm",
    )
    print(f"[scale] Adaptive fusion saved: {len(rows)} edges")
    return manifest_path


def _compute_consistency_from_csr(S_meta, S_cont, d_content_ids):
    """Compute per-doc Jaccard + weighted consistency directly from CSR matrices.

    This avoids the slow iterrows approach of the original
    compute_jaccard_and_consistency, which expects pre-built neighbor dicts.

    Args:
        S_meta: CSR matrix (N x N) for metadata fusion view.
        S_cont: CSR matrix (N x N) for content view.
        d_content_ids: List of doc_idx values in D_content.

    Returns:
        DataFrame with columns: doc_idx, jaccard, weighted_consistency,
        n_meta, n_cont, n_intersect.
    """
    results = []
    for doc_i in d_content_ids:
        # Extract neighbors and weights from CSR for meta view
        m_start, m_end = S_meta.indptr[doc_i], S_meta.indptr[doc_i + 1]
        meta_cols = S_meta.indices[m_start:m_end]
        meta_vals = S_meta.data[m_start:m_end]
        meta_set = set(meta_cols.tolist())
        meta_w = {int(meta_cols[j]): float(meta_vals[j]) for j in range(len(meta_cols))}

        # Extract neighbors and weights from CSR for content view
        c_start, c_end = S_cont.indptr[doc_i], S_cont.indptr[doc_i + 1]
        cont_cols = S_cont.indices[c_start:c_end]
        cont_vals = S_cont.data[c_start:c_end]
        cont_set = set(cont_cols.tolist())
        cont_w = {int(cont_cols[j]): float(cont_vals[j]) for j in range(len(cont_cols))}

        # Jaccard
        inter = meta_set & cont_set
        union = meta_set | cont_set
        jaccard = len(inter) / max(len(union), 1)

        # Weighted consistency
        w_sum = 0.0
        for j in inter:
            w_sum += min(meta_w.get(j, 0.0), cont_w.get(j, 0.0))

        meta_total = sum(meta_w.values())
        cont_total = sum(cont_w.values())
        w_max = min(meta_total, cont_total) if (meta_total > 0 and cont_total > 0) else 0.0
        consistency = w_sum / max(w_max, 1e-12) if w_max > 0 else 0.0
        consistency = min(consistency, 1.0)

        results.append({
            "doc_idx": doc_i,
            "jaccard": jaccard,
            "weighted_consistency": consistency,
            "n_meta": len(meta_set),
            "n_cont": len(cont_set),
            "n_intersect": len(inter),
        })

    return pd.DataFrame(results)


def build_adaptive_cons_fusion(
    scale_dir: Path,
    tmp_dir: Path,
    N: int,
    k_sim: int = 50,
) -> Path:
    """Build adaptive + consistency-adjusted 4-view fusion.

    Returns path to S_fused4c_symrow manifest.
    """
    S_tag = load_csr_from_manifest("S_tag_symrow", N, tmp_dir, k=k_sim)
    S_text = load_csr_from_manifest("S_text_symrow", N, tmp_dir, k=k_sim)
    S_beh = load_csr_from_manifest("S_beh_symrow", N, tmp_dir, k=k_sim)
    S_tabcontent = load_csr_from_manifest("S_tabcontent_symrow", N, scale_dir, k=k_sim)

    rho_dict = {
        "tag": compute_rho(S_tag),
        "text": compute_rho(S_text),
        "beh": compute_rho(S_beh),
        "tabcontent": compute_rho(S_tabcontent),
    }
    views = ["tag", "text", "beh", "tabcontent"]
    alpha = compute_adaptive_alpha(rho_dict, views)

    # Load or compute consistency scores
    cons_path = scale_dir / "consistency_meta_content.parquet"
    if cons_path.exists():
        c_scores = pd.read_parquet(cons_path, engine="fastparquet")
    else:
        # Compute meta-content consistency from CSR matrices
        S_fused3 = load_csr_from_manifest("S_fused3_symrow", N, tmp_dir, k=k_sim)
        d_content = pd.read_parquet(scale_dir / "d_content.parquet", engine="fastparquet")
        d_content_ids = d_content["doc_idx"].astype(int).tolist()
        c_scores = _compute_consistency_from_csr(S_fused3, S_tabcontent, d_content_ids)
        c_scores.to_parquet(cons_path, engine="fastparquet")
        print(f"[scale] Consistency scores computed and saved")

    alpha_adj = apply_consistency_adjustment(alpha, c_scores, N)

    S_dict = {
        "tag": S_tag,
        "text": S_text,
        "beh": S_beh,
        "tabcontent": S_tabcontent,
    }

    rows, cols, vals = fuse_views(S_dict, alpha_adj, views, N, K=k_sim)
    manifest_path = save_partitioned_edges(
        rows, cols, vals, N,
        prefix="S_fused4c_symrow", k=k_sim, output_dir=scale_dir,
        note="4-view rho-adaptive + consistency; top-K + L1 norm",
    )
    print(f"[scale] Adaptive+Cons fusion saved: {len(rows)} edges")
    return manifest_path


def main():
    args = parse_args()
    target = args.target

    if args.device == "auto":
        args.device = detect_device()
    print(f"[scale] Device: {args.device}")

    TMP_DIR = ROOT / "tmp"
    CONTENT_DIR = TMP_DIR / "content"
    SCALE_DIR = CONTENT_DIR / f"scale_{target}"
    TAB_RAW_DIR = ROOT / "data" / "tabular_raw"

    dc_path = SCALE_DIR / "d_content.parquet"
    mt_path = SCALE_DIR / "main_tables.parquet"

    if not dc_path.exists() or not mt_path.exists():
        print(f"[scale] ERROR: Run expand_content_coverage.py --target {target} first")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Run content pipeline
    # ------------------------------------------------------------------
    manifest_check = SCALE_DIR / f"S_tabcontent_symrow_k{args.k_sim}_manifest.json"
    if args.skip_pipeline and manifest_check.exists():
        print("[scale] Skipping pipeline (S_tabcontent already exists)")
    else:
        print(f"\n{'=' * 60}")
        print(f"[scale] Running content pipeline for target={target}")
        print(f"{'=' * 60}")
        t0 = time.time()

        outputs = run_content_pipeline(
            d_content_path=dc_path,
            main_tables_path=mt_path,
            tab_raw_dir=TAB_RAW_DIR,
            output_dir=SCALE_DIR,
            max_rows=args.max_rows,
            max_cols=args.max_cols,
            k_sim=args.k_sim,
            n_total=args.n_total,
            device=args.device,
            seed=args.seed,
        )
        pipeline_time = time.time() - t0
        print(f"[scale] Pipeline completed in {pipeline_time:.1f}s")

    # ------------------------------------------------------------------
    # 2. Build Naive Fusion
    # ------------------------------------------------------------------
    naive_manifest = SCALE_DIR / f"S_naive4_symrow_k{args.k_sim}_manifest.json"
    if args.skip_fusion and naive_manifest.exists():
        print("[scale] Skipping Naive Fusion (manifest exists)")
    else:
        print(f"\n{'=' * 60}")
        print("[scale] Building Naive Fusion...")
        print(f"{'=' * 60}")
        t1 = time.time()
        build_naive_fusion(
            S_fused3_dir=TMP_DIR,
            S_tabcontent_dir=SCALE_DIR,
            output_dir=SCALE_DIR,
            N=args.n_total,
            k_sim=args.k_sim,
        )
        print(f"[scale] Naive fusion built in {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------
    # 3. Build Adaptive Fusion
    # ------------------------------------------------------------------
    adaptive_manifest = SCALE_DIR / f"S_fused4_symrow_k{args.k_sim}_manifest.json"
    if args.skip_fusion and adaptive_manifest.exists():
        print("[scale] Skipping Adaptive Fusion (manifest exists)")
    else:
        print(f"\n{'=' * 60}")
        print("[scale] Building Adaptive Fusion...")
        print(f"{'=' * 60}")
        t2 = time.time()
        build_adaptive_fusion(SCALE_DIR, TMP_DIR, args.n_total, args.k_sim)
        print(f"[scale] Adaptive fusion built in {time.time() - t2:.1f}s")

    # ------------------------------------------------------------------
    # 4. Build Adaptive+Cons Fusion
    # ------------------------------------------------------------------
    cons_manifest = SCALE_DIR / f"S_fused4c_symrow_k{args.k_sim}_manifest.json"
    if args.skip_fusion and cons_manifest.exists():
        print("[scale] Skipping Adaptive+Cons Fusion (manifest exists)")
    else:
        print(f"\n{'=' * 60}")
        print("[scale] Building Adaptive+Cons Fusion...")
        print(f"{'=' * 60}")
        t3 = time.time()
        build_adaptive_cons_fusion(SCALE_DIR, TMP_DIR, args.n_total, args.k_sim)
        print(f"[scale] Adaptive+Cons fusion built in {time.time() - t3:.1f}s")

    # ------------------------------------------------------------------
    # 5. Load silver standards
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("[scale] Loading silver standards...")
    print(f"{'=' * 60}")
    standards = load_silver_standards(TMP_DIR, args.n_total)

    # Load D_content set
    d_content = pd.read_parquet(dc_path, engine="fastparquet")
    d_content_set = set(d_content["doc_idx"].astype(int).values)
    n_content = len(d_content_set)
    print(f"[scale] D_content: {n_content} docs")

    # ------------------------------------------------------------------
    # 6. Evaluate on D_content-only subset (100% coverage)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[scale] Evaluating on D_content subset ({n_content} docs, 100% coverage)")
    print(f"{'=' * 60}")

    # Build methods config with scale-specific dirs
    scale_methods = {
        "Meta-only": {
            "prefix": "S_fused3_symrow",
            "dir_key": "tmp",
            "group": "Metadata Fusion",
        },
        "Content-only": {
            "prefix": "S_tabcontent_symrow",
            "dir_key": "scale",
            "group": "Content View",
        },
        "Naive-Fusion": {
            "prefix": "S_naive4_symrow",
            "dir_key": "scale",
            "group": "4-View Fusion",
        },
        "Adaptive-Fusion": {
            "prefix": "S_fused4_symrow",
            "dir_key": "scale",
            "group": "4-View Fusion",
        },
        "Adaptive+Cons": {
            "prefix": "S_fused4c_symrow",
            "dir_key": "scale",
            "group": "4-View Fusion",
        },
        "Tag-only": {
            "prefix": "S_tag_symrow",
            "dir_key": "tmp",
            "group": "Single View",
        },
        "Text-only": {
            "prefix": "S_text_symrow",
            "dir_key": "tmp",
            "group": "Single View",
        },
        "Beh-only": {
            "prefix": "S_beh_symrow",
            "dir_key": "tmp",
            "group": "Single View",
        },
    }

    dirs = {"tmp": TMP_DIR, "content": CONTENT_DIR, "scale": SCALE_DIR}

    content_metrics = evaluate_all_methods(
        scale_methods,
        d_content_set,
        standards,
        dirs,
        k_eval=args.k_eval,
        k_sim=args.k_sim,
    )
    content_metrics["subset_size"] = n_content
    content_metrics["content_ratio"] = 1.0
    content_metrics["n_content_docs"] = n_content
    content_metrics["subset_type"] = "D_content_only"

    content_path = SCALE_DIR / "results_content_only.csv"
    content_metrics.to_csv(content_path, index=False)
    print(f"\n[scale] Saved: {content_path.name}")

    all_results = [content_metrics]

    # ------------------------------------------------------------------
    # 7. Evaluate on dilution subsets
    # ------------------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    all_indices = np.arange(args.n_total)
    non_content_indices = np.array(
        [i for i in all_indices if i not in d_content_set], dtype=np.int64,
    )

    dilution_sizes = args.dilution_sizes
    if dilution_sizes is None:
        # Default: D_content + 50K and D_content + 100K
        dilution_sizes = [50000, 100000]

    for size in sorted(dilution_sizes):
        if size <= n_content:
            print(f"\n[scale] Skipping dilution size {size} <= D_content ({n_content})")
            continue

        print(f"\n{'=' * 60}")
        print(f"[scale] Evaluating dilution subset: size={size}")
        print(f"{'=' * 60}")

        n_extra = min(size - n_content, len(non_content_indices))
        sampled_extra = rng.choice(non_content_indices, n_extra, replace=False)
        subset = d_content_set | set(sampled_extra.tolist())
        content_ratio = n_content / len(subset)
        print(f"[scale] Subset: {n_content} content + {n_extra} non-content = "
              f"{len(subset)} docs (coverage: {content_ratio:.2%})")

        metrics_df = evaluate_all_methods(
            scale_methods,
            subset,
            standards,
            dirs,
            k_eval=args.k_eval,
            k_sim=args.k_sim,
        )
        metrics_df["subset_size"] = len(subset)
        metrics_df["content_ratio"] = content_ratio
        metrics_df["n_content_docs"] = n_content
        metrics_df["subset_type"] = f"dilution_{size}"

        dilution_path = SCALE_DIR / f"results_dilution_{size}.csv"
        metrics_df.to_csv(dilution_path, index=False)
        print(f"[scale] Saved: {dilution_path.name}")

        all_results.append(metrics_df)

    # ------------------------------------------------------------------
    # 8. Combined results
    # ------------------------------------------------------------------
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = SCALE_DIR / "results_all_subsets.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\n[scale] Saved combined results: {combined_path.name}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"SCALE EXPERIMENT SUMMARY: target={target}")
    print(f"{'=' * 80}")

    for subset_type in combined["subset_type"].unique():
        sub = combined[combined["subset_type"] == subset_type]
        print(f"\n--- {subset_type} (n={sub['subset_size'].iloc[0]}, "
              f"coverage={sub['content_ratio'].iloc[0]:.2%}) ---")
        print(f"{'Method':<20s} {'Unified':>10s} {'Tag':>10s} {'Desc':>10s} {'Creator':>10s}")
        for _, row in sub.sort_values("unified_ndcg", ascending=False).iterrows():
            print(f"{row['method']:<20s} {row['unified_ndcg']:>10.4f} "
                  f"{row['tag_ndcg']:>10.4f} {row['desc_ndcg']:>10.4f} "
                  f"{row['cre_ndcg']:>10.4f}")


if __name__ == "__main__":
    main()
