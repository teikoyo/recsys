#!/usr/bin/env python3
"""
Standalone evaluation for scale experiments.

Runs evaluation for a single method at a time, saving results incrementally.
Uses CSR matrices directly for much faster neighbor extraction.

Usage:
    python scripts/eval_at_scale.py --target 10000 --seed 42
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent

from src.constants import (
    W_TAG_EVAL as W_TAG,
    W_DESC_EVAL as W_DESC,
    W_CREATOR_EVAL as W_CRE,
    DESC_SIM_THRESHOLD as DESC_THRESHOLD,
    CORPUS_SIZE_DEFAULT,
)
from src.content.evaluation import (
    SilverStandards,
    evaluate_method_on_subset,
    load_silver_standards,
)
from src.content.similarity import load_csr_from_manifest


def build_topk_from_csr(S_csr, k_eval, subset):
    """Extract top-K neighbors from CSR matrix for docs in subset.

    Much faster than loading parquet edges + groupby.
    """
    nbr_idx = {}
    nbr_w = {}
    for doc_i in subset:
        start = S_csr.indptr[doc_i]
        end = S_csr.indptr[doc_i + 1]
        if end == start:
            continue
        cols = S_csr.indices[start:end]
        vals = S_csr.data[start:end]
        if len(cols) <= k_eval:
            nbr_idx[doc_i] = cols.astype(np.int64)
            nbr_w[doc_i] = vals.astype(np.float32)
        else:
            top_k = np.argpartition(vals, -k_eval)[-k_eval:]
            top_k = top_k[np.argsort(vals[top_k])[::-1]]
            nbr_idx[doc_i] = cols[top_k].astype(np.int64)
            nbr_w[doc_i] = vals[top_k].astype(np.float32)
    return nbr_idx, nbr_w


def evaluate_single_method_csr(
    S_csr, method_name, subset, standards, k_eval=20,
):
    """Evaluate one method using pre-loaded CSR matrix."""
    nbr_idx, nbr_w = build_topk_from_csr(S_csr, k_eval, subset)
    if not nbr_idx:
        print(f"  WARNING: No neighbors for {method_name}")
        return None

    doc_tags = standards.doc_tags
    idf_map = standards.idf_map
    S_bm25 = standards.S_bm25
    creator_ids = standards.creator_ids
    creator_counts = standards.creator_counts
    N = standards.N

    from src.metrics import (
        average_precision_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    results = {"method": method_name}

    # Tag dimension
    tag_ndcgs, tag_maps, tag_mrrs, tag_precs, tag_recs = [], [], [], [], []
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
            gain = sum(idf_map.get(t, 1.0) for t in inter) / sum(idf_map.get(t, 1.0) for t in union) if union else 0.0
            gains.append(gain)
            binary.append(1.0 if len(inter) >= 1 else 0.0)
        gains = np.array(gains, dtype=np.float64)
        binary = np.array(binary, dtype=np.float64)
        ideal = np.sort(gains)[::-1]
        tag_ndcgs.append(ndcg_at_k(gains, ideal))
        tag_maps.append(average_precision_at_k(binary))
        tag_mrrs.append(mrr_at_k(binary))
        tag_precs.append(precision_at_k(binary))
        tag_recs.append(float(binary.sum()) / max(k_eval, 1))

    results["tag_ndcg"] = np.mean(tag_ndcgs) if tag_ndcgs else 0.0
    results["tag_map"] = np.mean(tag_maps) if tag_maps else 0.0
    results["tag_mrr"] = np.mean(tag_mrrs) if tag_mrrs else 0.0
    results["tag_prec"] = np.mean(tag_precs) if tag_precs else 0.0
    results["tag_rec"] = np.mean(tag_recs) if tag_recs else 0.0
    results["tag_coverage"] = tag_covered / max(len(subset), 1)
    results["tag_n"] = tag_covered

    # Desc dimension
    desc_ndcgs, desc_maps, desc_mrrs, desc_precs, desc_recs = [], [], [], [], []
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
        desc_maps.append(average_precision_at_k(binary))
        desc_mrrs.append(mrr_at_k(binary))
        desc_precs.append(precision_at_k(binary))
        desc_recs.append(float(binary.sum()) / max(k_eval, 1))

    results["desc_ndcg"] = np.mean(desc_ndcgs) if desc_ndcgs else 0.0
    results["desc_map"] = np.mean(desc_maps) if desc_maps else 0.0
    results["desc_mrr"] = np.mean(desc_mrrs) if desc_mrrs else 0.0
    results["desc_prec"] = np.mean(desc_precs) if desc_precs else 0.0
    results["desc_rec"] = np.mean(desc_recs) if desc_recs else 0.0
    results["desc_coverage"] = desc_covered / max(len(subset), 1)
    results["desc_n"] = desc_covered

    # Creator dimension
    cre_ndcgs, cre_maps, cre_mrrs, cre_precs, cre_recs = [], [], [], [], []
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
        total_rel = max(creator_counts.get(cid_i, 1) - 1, 1)
        cre_ndcgs.append(ndcg_at_k(gains, ideal))
        cre_maps.append(average_precision_at_k(binary))
        cre_mrrs.append(mrr_at_k(binary))
        cre_precs.append(precision_at_k(binary))
        cre_recs.append(recall_at_k(binary, total_rel))

    results["cre_ndcg"] = np.mean(cre_ndcgs) if cre_ndcgs else 0.0
    results["cre_map"] = np.mean(cre_maps) if cre_maps else 0.0
    results["cre_mrr"] = np.mean(cre_mrrs) if cre_mrrs else 0.0
    results["cre_prec"] = np.mean(cre_precs) if cre_precs else 0.0
    results["cre_rec"] = np.mean(cre_recs) if cre_recs else 0.0
    results["cre_coverage"] = cre_covered / max(len(subset), 1)
    results["cre_n"] = cre_covered

    results["unified_ndcg"] = (
        W_TAG * results["tag_ndcg"]
        + W_DESC * results["desc_ndcg"]
        + W_CRE * results["cre_ndcg"]
    )
    print(
        f"  {method_name}: unified_nDCG={results['unified_ndcg']:.4f} "
        f"(tag={results['tag_ndcg']:.4f}, desc={results['desc_ndcg']:.4f}, "
        f"cre={results['cre_ndcg']:.4f})"
    )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-eval", type=int, default=20)
    p.add_argument("--k-sim", type=int, default=50)
    p.add_argument("--n-total", type=int, default=CORPUS_SIZE_DEFAULT)
    p.add_argument("--dilution-sizes", type=int, nargs="*", default=None)
    args = p.parse_args()

    TMP_DIR = ROOT / "tmp"
    SCALE_DIR = TMP_DIR / "content" / f"scale_{args.target}"
    N = args.n_total

    # Load D_content
    dc_path = SCALE_DIR / "d_content.parquet"
    d_content = pd.read_parquet(dc_path, engine="fastparquet")
    d_content_set = set(d_content["doc_idx"].astype(int).values)
    n_content = len(d_content_set)
    print(f"[eval] D_content: {n_content} docs")

    # Load silver standards
    print("[eval] Loading silver standards...")
    t0 = time.time()
    standards = load_silver_standards(TMP_DIR, N)
    print(f"[eval] Silver standards loaded in {time.time() - t0:.1f}s")

    # Method configs: (name, prefix, dir)
    methods = [
        ("Meta-only", "S_fused3_symrow", TMP_DIR),
        ("Content-only", "S_tabcontent_symrow", SCALE_DIR),
        ("Naive-Fusion", "S_naive4_symrow", SCALE_DIR),
        ("Adaptive-Fusion", "S_fused4_symrow", SCALE_DIR),
        ("Adaptive+Cons", "S_fused4c_symrow", SCALE_DIR),
        ("Tag-only", "S_tag_symrow", TMP_DIR),
        ("Text-only", "S_text_symrow", TMP_DIR),
        ("Beh-only", "S_beh_symrow", TMP_DIR),
    ]

    groups = {
        "Meta-only": "Metadata Fusion",
        "Content-only": "Content View",
        "Naive-Fusion": "4-View Fusion",
        "Adaptive-Fusion": "4-View Fusion",
        "Adaptive+Cons": "4-View Fusion",
        "Tag-only": "Single View",
        "Text-only": "Single View",
        "Beh-only": "Single View",
    }

    # Build evaluation subsets
    rng = np.random.RandomState(args.seed)
    all_indices = np.arange(N)
    non_content = np.array([i for i in all_indices if i not in d_content_set], dtype=np.int64)

    dilution_sizes = args.dilution_sizes or [50000, 100000]
    subsets = [("D_content_only", d_content_set, 1.0)]
    for size in sorted(dilution_sizes):
        if size <= n_content:
            continue
        n_extra = min(size - n_content, len(non_content))
        sampled = rng.choice(non_content, n_extra, replace=False)
        subset = d_content_set | set(sampled.tolist())
        ratio = n_content / len(subset)
        subsets.append((f"dilution_{size}", subset, ratio))

    # Load CSR matrices and evaluate
    all_results = []
    for method_name, prefix, base_dir in methods:
        print(f"\n[eval] Loading CSR: {method_name} ({prefix})")
        t0 = time.time()
        S = load_csr_from_manifest(prefix, N, base_dir, k=args.k_sim)
        print(f"[eval] Loaded in {time.time() - t0:.1f}s, nnz={S.nnz}")

        for subset_name, subset, ratio in subsets:
            print(f"\n[eval] {method_name} on {subset_name} ({len(subset)} docs, {ratio:.2%} coverage)")
            t1 = time.time()
            res = evaluate_single_method_csr(
                S, method_name, subset, standards, k_eval=args.k_eval,
            )
            if res:
                res["group"] = groups[method_name]
                res["subset_type"] = subset_name
                res["subset_size"] = len(subset)
                res["content_ratio"] = ratio
                res["n_content_docs"] = n_content
                all_results.append(res)
            print(f"  Completed in {time.time() - t1:.1f}s")

        # Save incrementally
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(SCALE_DIR / "results_all_subsets.csv", index=False)
            print(f"[eval] Incremental save: {len(all_results)} results")

        del S
        gc.collect()

    # Final save
    combined = pd.DataFrame(all_results)
    combined.to_csv(SCALE_DIR / "results_all_subsets.csv", index=False)
    print(f"\n[eval] Final results saved: {SCALE_DIR / 'results_all_subsets.csv'}")

    # Also save content-only results
    content_df = combined[combined["subset_type"] == "D_content_only"]
    content_df.to_csv(SCALE_DIR / "results_content_only.csv", index=False)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SCALE EXPERIMENT SUMMARY: target={args.target}")
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
