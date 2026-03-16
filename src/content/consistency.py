#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-Content Consistency Metrics

Implements CONTENT_VIEW_EXTENSION.md section 4.1 (Jaccard set overlap)
and section 4.2 (weighted consistency c[i]).
"""

import pandas as pd

from .similarity import load_edges_from_manifest


def compute_jaccard_and_consistency(d_content_ids, N_meta, N_cont,
                                    cont_edges_df=None, fused3_dir=None,
                                    k=50):
    """Compute per-document Jaccard J(i) and weighted consistency c(i).

    Args:
        d_content_ids: Iterable of doc_idx values in D_content.
        N_meta: {doc_idx: set(neighbors)} from the metadata fusion view.
        N_cont: {doc_idx: set(neighbors)} from the content view.
        cont_edges_df: Optional DataFrame of content edges (row, col, val)
            for weighted consistency.
        fused3_dir: Optional directory containing S_fused3_symrow manifest
            for metadata edge weights.
        k: k value for manifest lookup.

    Returns:
        DataFrame with columns: doc_idx, jaccard, weighted_consistency,
        n_meta, n_cont, n_intersect.
    """
    # Build weight lookup tables for weighted consistency
    meta_w = {}
    cont_w = {}

    if cont_edges_df is not None:
        for _, r in cont_edges_df.iterrows():
            cont_w[(int(r["row"]), int(r["col"]))] = float(r["val"])

    if fused3_dir is not None:
        try:
            fused3_edges, _ = load_edges_from_manifest(
                "S_fused3_symrow", fused3_dir, k=k
            )
            d_set = set(d_content_ids)
            for _, r in fused3_edges.iterrows():
                ri = int(r["row"])
                if ri in d_set:
                    meta_w[(ri, int(r["col"]))] = float(r["val"])
            del fused3_edges
        except (FileNotFoundError, KeyError, ValueError):
            pass

    results = []
    for doc_i in d_content_ids:
        meta_set = N_meta.get(doc_i, set())
        cont_set = N_cont.get(doc_i, set())

        inter = meta_set & cont_set
        union = meta_set | cont_set

        jaccard = len(inter) / max(len(union), 1)

        # Weighted consistency
        w_sum = 0.0
        w_max_possible = 0.0
        if meta_w and cont_w:
            for j in inter:
                wm = meta_w.get((doc_i, j), 0.0)
                wc = cont_w.get((doc_i, j), 0.0)
                w_sum += min(wm, wc)

            meta_total = sum(meta_w.get((doc_i, j), 0.0) for j in meta_set)
            cont_total = sum(cont_w.get((doc_i, j), 0.0) for j in cont_set)
            w_max_possible = min(meta_total, cont_total) if (meta_total > 0 and cont_total > 0) else 0.0

        consistency = w_sum / max(w_max_possible, 1e-12) if w_max_possible > 0 else 0.0
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
