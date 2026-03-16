#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content Feature Ablation Experiment

Tests the impact of MAX_ROWS and MAX_COLS on content-view quality by
re-running the content pipeline with different configurations and evaluating
Content-only and Naive-Fusion on the D_content subset.

Design:
  - Row ablation: fixed MAX_COLS=60, vary MAX_ROWS in {64,128,256,512,1024}
  - Col ablation: fixed MAX_ROWS=1024, vary MAX_COLS in {5,10,20,30,60}
  - 9 independent configs (1024x60 is the shared baseline)

Usage:
    python scripts/run_content_ablation.py \\
        --row-ablation 64 128 256 512 1024 \\
        --col-ablation 5 10 20 30 60 \\
        --seed 42 --device auto \\
        --output-dir tmp/content/ablation_experiments
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from src.constants import CORPUS_SIZE_DEFAULT
from src.content.evaluation import (
    evaluate_method_on_subset,
    load_silver_standards,
)
from src.content.pipeline import (
    build_naive_fusion,
    detect_device,
    run_content_pipeline,
)


def parse_args():
    p = argparse.ArgumentParser(description="Content feature ablation experiment")
    p.add_argument(
        "--row-ablation", type=int, nargs="+", default=[64, 128, 256, 512, 1024],
        help="MAX_ROWS values (cols fixed at 60)",
    )
    p.add_argument(
        "--col-ablation", type=int, nargs="+", default=[5, 10, 20, 30, 60],
        help="MAX_COLS values (rows fixed at 1024)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--k-eval", type=int, default=20)
    p.add_argument("--k-sim", type=int, default=50)
    p.add_argument("--n-total", type=int, default=CORPUS_SIZE_DEFAULT)
    p.add_argument(
        "--output-dir", type=str,
        default=str(ROOT / "tmp" / "content" / "ablation_experiments"),
    )
    return p.parse_args()


def run_ablation_config(
    max_rows, max_cols, label, config_dir, args, d_content_path, main_tables_path,
    d_content_set, standards, S_fused3_dir,
):
    """Run pipeline + evaluate for one ablation configuration.

    Returns a list of result dicts (one per evaluated method).
    """
    print(f"\n--- Config: {label} (MAX_ROWS={max_rows}, MAX_COLS={max_cols}) ---")
    config_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: Run content pipeline
    outputs = run_content_pipeline(
        d_content_path=d_content_path,
        main_tables_path=main_tables_path,
        tab_raw_dir=ROOT / "data" / "tabular_raw",
        output_dir=config_dir,
        max_rows=max_rows,
        max_cols=max_cols,
        k_sim=args.k_sim,
        n_total=args.n_total,
        device=args.device,
        seed=args.seed,
    )
    pipeline_time = time.time() - t0
    print(f"  Pipeline completed in {pipeline_time:.1f}s")

    # Step 2: Build naive fusion
    t1 = time.time()
    build_naive_fusion(
        S_fused3_dir=S_fused3_dir,
        S_tabcontent_dir=config_dir,
        output_dir=config_dir,
        N=args.n_total,
        k_sim=args.k_sim,
    )
    fusion_time = time.time() - t1
    print(f"  Naive fusion built in {fusion_time:.1f}s")

    # Step 3: Evaluate Content-only and Naive-Fusion
    results = []
    for method_name, prefix in [
        ("Content-only", "S_tabcontent_symrow"),
        ("Naive-Fusion", "S_naive4_symrow"),
    ]:
        res, _ = evaluate_method_on_subset(
            prefix=prefix,
            method_name=method_name,
            base_dir=config_dir,
            subset=d_content_set,
            standards=standards,
            k_eval=args.k_eval,
            k_sim=args.k_sim,
        )
        if res is not None:
            res["max_rows"] = max_rows
            res["max_cols"] = max_cols
            res["label"] = label
            results.append(res)

    return results


def main():
    args = parse_args()

    if args.device == "auto":
        args.device = detect_device()
    print(f"Device: {args.device}")

    TMP_DIR = ROOT / "tmp"
    CONTENT_DIR = TMP_DIR / "content"
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d_content_path = CONTENT_DIR / "d_content.parquet"
    main_tables_path = CONTENT_DIR / "main_tables.parquet"

    # ------------------------------------------------------------------
    # 1. Load silver standards (once)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading silver standards...")
    t0 = time.time()
    standards = load_silver_standards(TMP_DIR, args.n_total)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # D_content set
    d_content = pd.read_parquet(d_content_path, engine="fastparquet")
    d_content_set = set(d_content["doc_idx"].astype(int).values)
    print(f"D_content: {len(d_content_set)} docs")

    # ------------------------------------------------------------------
    # 2. Build unique configuration list (avoid duplicate 1024x60)
    # ------------------------------------------------------------------
    configs = []

    # Row ablation: fixed cols=60, vary rows
    for rows in args.row_ablation:
        configs.append((rows, 60, f"R{rows}_C60"))

    # Col ablation: fixed rows=1024, vary cols
    for cols in args.col_ablation:
        key = (1024, cols)
        # Avoid duplicating the shared baseline (1024, 60)
        if not any(c[0] == key[0] and c[1] == key[1] for c in configs):
            configs.append((1024, cols, f"R1024_C{cols}"))

    print(f"\nTotal configurations: {len(configs)}")
    for r, c, label in configs:
        print(f"  {label}: MAX_ROWS={r}, MAX_COLS={c}")

    # ------------------------------------------------------------------
    # 3. Run all configurations
    # ------------------------------------------------------------------
    all_results = []

    for max_rows, max_cols, label in configs:
        config_dir = OUT_DIR / label
        results = run_ablation_config(
            max_rows, max_cols, label, config_dir,
            args, d_content_path, main_tables_path,
            d_content_set, standards, TMP_DIR,
        )
        all_results.extend(results)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    all_df = pd.DataFrame(all_results)

    # Split into row ablation and col ablation tables
    row_ablation = all_df[all_df["max_cols"] == 60].copy()
    col_ablation = all_df[all_df["max_rows"] == 1024].copy()

    row_ablation.to_csv(OUT_DIR / "ablation_rows.csv", index=False)
    col_ablation.to_csv(OUT_DIR / "ablation_cols.csv", index=False)
    all_df.to_csv(OUT_DIR / "ablation_all.csv", index=False)

    print(f"\n{'=' * 80}")
    print("ABLATION RESULTS SAVED")
    print(f"{'=' * 80}")
    print(f"  ablation_rows.csv: {len(row_ablation)} rows")
    print(f"  ablation_cols.csv: {len(col_ablation)} rows")
    print(f"  ablation_all.csv:  {len(all_df)} rows")

    # ------------------------------------------------------------------
    # 5. Summary tables
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("ROW ABLATION (fixed MAX_COLS=60)")
    print(f"{'=' * 80}")
    if len(row_ablation) > 0:
        for method in ["Content-only", "Naive-Fusion"]:
            sub = row_ablation[row_ablation["method"] == method].sort_values("max_rows")
            if len(sub) > 0:
                print(f"\n  {method}:")
                print(f"  {'MAX_ROWS':>10s}  {'unified_ndcg':>14s}  {'tag_ndcg':>10s}  {'desc_ndcg':>11s}  {'cre_ndcg':>10s}")
                for _, r in sub.iterrows():
                    print(
                        f"  {int(r['max_rows']):>10d}  {r['unified_ndcg']:>14.4f}  "
                        f"{r['tag_ndcg']:>10.4f}  {r['desc_ndcg']:>11.4f}  {r['cre_ndcg']:>10.4f}"
                    )

    print(f"\n{'=' * 80}")
    print("COL ABLATION (fixed MAX_ROWS=1024)")
    print(f"{'=' * 80}")
    if len(col_ablation) > 0:
        for method in ["Content-only", "Naive-Fusion"]:
            sub = col_ablation[col_ablation["method"] == method].sort_values("max_cols")
            if len(sub) > 0:
                print(f"\n  {method}:")
                print(f"  {'MAX_COLS':>10s}  {'unified_ndcg':>14s}  {'tag_ndcg':>10s}  {'desc_ndcg':>11s}  {'cre_ndcg':>10s}")
                for _, r in sub.iterrows():
                    print(
                        f"  {int(r['max_cols']):>10d}  {r['unified_ndcg']:>14.4f}  "
                        f"{r['tag_ndcg']:>10.4f}  {r['desc_ndcg']:>11.4f}  {r['cre_ndcg']:>10.4f}"
                    )


if __name__ == "__main__":
    main()
