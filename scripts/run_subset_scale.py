#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scale Experiment: Dilution Effect Test

Creates evaluation subsets of varying sizes (each containing all 1000
D_content documents + randomly sampled non-content documents) and evaluates
all 8 methods on each subset.  This tests how content coverage dilution
affects fusion quality.

Usage:
    python scripts/run_subset_scale.py \\
        --sizes 1000 5000 10000 50000 \\
        --seed 42 --k-eval 20 \\
        --output-dir tmp/content/scale_experiments
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from src.constants import CORPUS_SIZE_DEFAULT
from src.content.evaluation import (
    METHODS_CONFIG,
    evaluate_all_methods,
    load_silver_standards,
)


def parse_args():
    p = argparse.ArgumentParser(description="Scale experiment: dilution effect")
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 50000],
        help="Subset sizes to evaluate",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-eval", type=int, default=20)
    p.add_argument("--k-sim", type=int, default=50)
    p.add_argument("--n-total", type=int, default=CORPUS_SIZE_DEFAULT)
    p.add_argument(
        "--output-dir", type=str,
        default=str(ROOT / "tmp" / "content" / "scale_experiments"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    TMP_DIR = ROOT / "tmp"
    CONTENT_DIR = TMP_DIR / "content"
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dirs = {"tmp": TMP_DIR, "content": CONTENT_DIR}

    # ------------------------------------------------------------------
    # 1. Load silver standards (once)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading silver standards...")
    t0 = time.time()
    standards = load_silver_standards(TMP_DIR, args.n_total)
    print(f"Silver standards loaded in {time.time() - t0:.1f}s")
    print(f"  Tags: {len(standards.doc_tags)} docs")
    print(f"  BM25: nnz={standards.S_bm25.nnz:,}")
    print(f"  Creators: {(standards.creator_ids > 0).sum():,} docs")

    # ------------------------------------------------------------------
    # 2. Load D_content set
    # ------------------------------------------------------------------
    d_content = pd.read_parquet(CONTENT_DIR / "d_content.parquet", engine="fastparquet")
    d_content_set = set(d_content["doc_idx"].astype(int).values)
    n_content = len(d_content_set)
    print(f"\nD_content: {n_content} docs")

    # All non-content doc indices
    all_indices = np.arange(args.n_total)
    non_content_indices = np.array(
        [i for i in all_indices if i not in d_content_set], dtype=np.int64
    )
    print(f"Non-content pool: {len(non_content_indices):,} docs")

    # ------------------------------------------------------------------
    # 3. Run evaluation at each scale
    # ------------------------------------------------------------------
    all_scale_results = []

    for size in sorted(args.sizes):
        print(f"\n{'=' * 60}")
        print(f"Scale experiment: size={size}")
        print(f"{'=' * 60}")

        if size <= n_content:
            # Subset is just D_content (or smaller)
            subset = d_content_set
            print(f"  Using D_content only ({n_content} docs, requested {size})")
        else:
            # D_content + random non-content sample
            n_extra = size - n_content
            if n_extra > len(non_content_indices):
                n_extra = len(non_content_indices)
                print(f"  WARNING: Capping extra docs to {n_extra}")
            sampled_extra = rng.choice(non_content_indices, n_extra, replace=False)
            subset = d_content_set | set(sampled_extra.tolist())
            print(f"  Subset: {n_content} content + {n_extra} non-content = {len(subset)} docs")

        content_ratio = n_content / len(subset)
        print(f"  Content coverage: {content_ratio:.2%}")

        t1 = time.time()
        metrics_df = evaluate_all_methods(
            METHODS_CONFIG,
            subset,
            standards,
            dirs,
            k_eval=args.k_eval,
            k_sim=args.k_sim,
        )
        elapsed = time.time() - t1
        print(f"  Evaluated {len(metrics_df)} methods in {elapsed:.1f}s")

        # Add scale metadata
        metrics_df["subset_size"] = len(subset)
        metrics_df["content_ratio"] = content_ratio
        metrics_df["n_content_docs"] = n_content

        # Save per-scale CSV
        scale_path = OUT_DIR / f"scale_{len(subset)}_metrics.csv"
        metrics_df.to_csv(scale_path, index=False)
        print(f"  Saved: {scale_path.name}")

        all_scale_results.append(metrics_df)

    # ------------------------------------------------------------------
    # 4. Combine and save all results
    # ------------------------------------------------------------------
    combined = pd.concat(all_scale_results, ignore_index=True)
    combined_path = OUT_DIR / "scale_all_metrics.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined results: {combined_path}")
    print(f"Total: {len(combined)} rows ({len(args.sizes)} sizes x {len(METHODS_CONFIG)} methods)")

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("SCALE EXPERIMENT SUMMARY: Unified@nDCG20 by Method and Subset Size")
    print(f"{'=' * 80}")

    pivot = combined.pivot_table(
        index="method", columns="subset_size", values="unified_ndcg",
    )
    # Sort by the smallest subset's score
    first_col = pivot.columns[0]
    pivot = pivot.sort_values(first_col, ascending=False)

    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # Content coverage ratio
    print("Content coverage by subset size:")
    for size in sorted(combined["subset_size"].unique()):
        ratio = combined[combined["subset_size"] == size]["content_ratio"].iloc[0]
        print(f"  {size:>6d}: {ratio:.2%}")


if __name__ == "__main__":
    main()
