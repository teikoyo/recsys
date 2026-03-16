#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproducible Evaluation Module

Extracts the subset evaluation logic from NB04 into a stateless, reusable
module.  All silver-standard data and method configurations are encapsulated
so that evaluation can be driven by scripts without notebook global state.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from ..metrics import (
    average_precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ..constants import (
    W_TAG_EVAL as W_TAG,
    W_DESC_EVAL as W_DESC,
    W_CREATOR_EVAL as W_CRE,
    DESC_SIM_THRESHOLD as DESC_THRESHOLD,
)
from .similarity import load_csr_from_manifest, load_manifest_flexible
from ..log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CSR matrix cache (avoids reloading the same large matrix multiple times)
# ---------------------------------------------------------------------------

_csr_cache: Dict[str, Any] = {}


def _load_csr_cached(prefix: str, N: int, base_dir, k: int = 50):
    """Load CSR matrix with caching by (prefix, base_dir) key."""
    cache_key = f"{base_dir}/{prefix}_k{k}"
    if cache_key not in _csr_cache:
        _csr_cache[cache_key] = load_csr_from_manifest(prefix, N, base_dir, k=k)
    return _csr_cache[cache_key]

# Default method configurations: prefix -> (dir_key, display_name, group)
#   dir_key is resolved at runtime via a dirs dict {"tmp": ..., "content": ...}
METHODS_CONFIG: Dict[str, Dict[str, str]] = {
    "Meta-only": {
        "prefix": "S_fused3_symrow",
        "dir_key": "tmp",
        "group": "Metadata Fusion",
    },
    "Content-only": {
        "prefix": "S_tabcontent_symrow",
        "dir_key": "content",
        "group": "Content View",
    },
    "Naive-Fusion": {
        "prefix": "S_naive4_symrow",
        "dir_key": "content",
        "group": "4-View Fusion",
    },
    "Adaptive-Fusion": {
        "prefix": "S_fused4_symrow",
        "dir_key": "content",
        "group": "4-View Fusion",
    },
    "Adaptive+Cons": {
        "prefix": "S_fused4c_symrow",
        "dir_key": "content",
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


# ---------------------------------------------------------------------------
# Silver Standards container
# ---------------------------------------------------------------------------

@dataclass
class SilverStandards:
    """Encapsulates all silver-standard data needed for evaluation."""

    doc_tags: Dict[int, List[str]]
    idf_map: Dict[str, float]
    S_bm25: Any  # scipy CSR
    creator_ids: np.ndarray  # (N,) int64
    creator_counts: Counter
    N: int


def load_silver_standards(tmp_dir, N: int) -> SilverStandards:
    """Load tag / BM25-desc / creator silver standards from *tmp_dir*.

    Args:
        tmp_dir: Path to the ``tmp/`` directory.
        N: Total number of documents in the corpus.

    Returns:
        A populated :class:`SilverStandards` instance.
    """
    tmp_dir = Path(tmp_dir)

    # --- Tags ---
    tag_docs = pd.read_parquet(tmp_dir / "relevance_tag_docs.parquet", engine="fastparquet")
    tag_idf = pd.read_parquet(tmp_dir / "relevance_tag_idf.parquet", engine="fastparquet")

    doc_tags: Dict[int, List[str]] = {}
    for _, row in tag_docs.iterrows():
        idx = int(row["doc_idx"])
        tags = row["tags"]
        if isinstance(tags, list) and len(tags) > 0:
            doc_tags[idx] = tags
        elif isinstance(tags, str) and tags:
            doc_tags[idx] = [t.strip() for t in tags.split(",") if t.strip()]

    idf_map = dict(zip(tag_idf["tag"], tag_idf["idf"]))

    # --- BM25 (Desc) ---
    S_bm25 = load_csr_from_manifest("S_textbm25_topk", N, tmp_dir)

    # --- Creator ---
    beh_base = pd.read_parquet(tmp_dir / "beh_base.parquet", engine="fastparquet")
    creator_ids = np.zeros(N, dtype=np.int64)
    for _, row in beh_base.iterrows():
        idx = int(row["doc_idx"])
        cid = int(row["CreatorUserId"])
        if 0 <= idx < N:
            creator_ids[idx] = cid

    creator_counts = Counter(creator_ids[creator_ids > 0])

    return SilverStandards(
        doc_tags=doc_tags,
        idf_map=idf_map,
        S_bm25=S_bm25,
        creator_ids=creator_ids,
        creator_counts=creator_counts,
        N=N,
    )


# ---------------------------------------------------------------------------
# Neighbor loading
# ---------------------------------------------------------------------------

def build_topk_for_method(
    prefix: str,
    k_eval: int,
    base_dir,
    k_sim: int = 50,
    subset: Optional[Set[int]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Load similarity edges and return per-doc top-K neighbors.

    Uses CSR matrix for efficient row-wise access instead of pandas groupby.

    Args:
        prefix: Manifest prefix (e.g. ``S_fused3_symrow``).
        k_eval: Number of neighbors to keep per document.
        base_dir: Directory containing the manifest.
        k_sim: k value used in the manifest filename.
        subset: If provided, only extract neighbors for these doc indices.

    Returns:
        ``(nbr_idx, nbr_w)`` dicts mapping doc_idx to neighbor arrays.
    """
    man, _, _ = load_manifest_flexible(prefix, base_dir, k=k_sim)
    N = man.get("nodes", 0)
    if N == 0:
        return {}, {}

    S = _load_csr_cached(prefix, N, base_dir, k=k_sim)

    nbr_idx: Dict[int, np.ndarray] = {}
    nbr_w: Dict[int, np.ndarray] = {}

    rows_to_check = subset if subset is not None else range(N)
    for row_i in rows_to_check:
        if row_i < 0 or row_i >= N:
            continue
        start = S.indptr[row_i]
        end = S.indptr[row_i + 1]
        if end == start:
            continue
        cols = S.indices[start:end]
        vals = S.data[start:end]
        if len(cols) <= k_eval:
            order = np.argsort(-vals)
            nbr_idx[row_i] = cols[order].astype(np.int64)
            nbr_w[row_i] = vals[order].astype(np.float32)
        else:
            top_k = np.argpartition(-vals, k_eval)[:k_eval]
            top_k = top_k[np.argsort(-vals[top_k])]
            nbr_idx[row_i] = cols[top_k].astype(np.int64)
            nbr_w[row_i] = vals[top_k].astype(np.float32)
    return nbr_idx, nbr_w


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_method_on_subset(
    prefix: str,
    method_name: str,
    base_dir,
    subset: Set[int],
    standards: SilverStandards,
    k_eval: int = 20,
    k_sim: int = 50,
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate a single method on a document subset.

    Args:
        prefix: Manifest prefix.
        method_name: Human-readable method label.
        base_dir: Directory containing the manifest.
        subset: Set of doc_idx to evaluate on.
        standards: Silver standard data.
        k_eval: Top-K for evaluation.
        k_sim: k value for manifest filenames.
        verbose: Print progress.

    Returns:
        ``(results_dict, per_doc_records)`` or ``(None, [])`` on failure.
    """
    nbr_idx, nbr_w = build_topk_for_method(prefix, k_eval, base_dir, k_sim, subset=subset)
    if not nbr_idx:
        if verbose:
            logger.warning(f"No neighbors loaded for {method_name}")
        return None, []

    doc_tags = standards.doc_tags
    idf_map = standards.idf_map
    S_bm25 = standards.S_bm25
    creator_ids = standards.creator_ids
    creator_counts = standards.creator_counts
    N = standards.N

    results: Dict[str, Any] = {"method": method_name}
    per_doc: List[Dict[str, Any]] = []

    # ---- Tag dimension ----
    tag_ndcgs, tag_maps, tag_mrrs, tag_precs, tag_recs = [], [], [], [], []
    tag_covered = 0

    for doc_i in subset:
        if doc_i not in nbr_idx:
            continue
        neighbors = nbr_idx[doc_i]
        if doc_i not in doc_tags:
            continue
        tags_i = set(doc_tags[doc_i])
        if not tags_i:
            continue
        tag_covered += 1

        gains = []
        binary = []
        for j in neighbors[:k_eval]:
            j = int(j)
            tags_j = set(doc_tags.get(j, []))
            inter = tags_i & tags_j
            union = tags_i | tags_j
            if union:
                idf_inter = sum(idf_map.get(t, 1.0) for t in inter)
                idf_union = sum(idf_map.get(t, 1.0) for t in union)
                gain = idf_inter / idf_union
            else:
                gain = 0.0
            gains.append(gain)
            binary.append(1.0 if len(inter) >= 1 else 0.0)

        gains = np.array(gains, dtype=np.float64)
        binary = np.array(binary, dtype=np.float64)
        ideal = np.sort(gains)[::-1]

        t_ndcg = ndcg_at_k(gains, ideal)
        t_map = average_precision_at_k(binary)
        t_mrr = mrr_at_k(binary)
        t_prec = precision_at_k(binary)
        t_rec = float(binary.sum()) / max(k_eval, 1)

        tag_ndcgs.append(t_ndcg)
        tag_maps.append(t_map)
        tag_mrrs.append(t_mrr)
        tag_precs.append(t_prec)
        tag_recs.append(t_rec)

        per_doc.append({
            "doc_idx": doc_i, "method": method_name,
            "tag_ndcg": t_ndcg, "tag_map": t_map,
        })

    results["tag_ndcg"] = np.mean(tag_ndcgs) if tag_ndcgs else 0.0
    results["tag_map"] = np.mean(tag_maps) if tag_maps else 0.0
    results["tag_mrr"] = np.mean(tag_mrrs) if tag_mrrs else 0.0
    results["tag_prec"] = np.mean(tag_precs) if tag_precs else 0.0
    results["tag_rec"] = np.mean(tag_recs) if tag_recs else 0.0
    results["tag_coverage"] = tag_covered / max(len(subset), 1)
    results["tag_n"] = tag_covered

    # ---- Desc dimension ----
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

        gains = []
        binary = []
        for j in neighbors[:k_eval]:
            j = int(j)
            sim = bm25_lookup.get(j, 0.0)
            gains.append(sim)
            binary.append(1.0 if sim > DESC_THRESHOLD else 0.0)

        gains = np.array(gains, dtype=np.float64)
        binary = np.array(binary, dtype=np.float64)
        ideal = np.sort(gains)[::-1]

        d_ndcg = ndcg_at_k(gains, ideal)
        d_map = average_precision_at_k(binary)
        d_mrr = mrr_at_k(binary)
        d_prec = precision_at_k(binary)
        d_rec = float(binary.sum()) / max(k_eval, 1)

        desc_ndcgs.append(d_ndcg)
        desc_maps.append(d_map)
        desc_mrrs.append(d_mrr)
        desc_precs.append(d_prec)
        desc_recs.append(d_rec)

        for rec in per_doc:
            if rec["doc_idx"] == doc_i:
                rec["desc_ndcg"] = d_ndcg
                rec["desc_map"] = d_map
                break

    results["desc_ndcg"] = np.mean(desc_ndcgs) if desc_ndcgs else 0.0
    results["desc_map"] = np.mean(desc_maps) if desc_maps else 0.0
    results["desc_mrr"] = np.mean(desc_mrrs) if desc_mrrs else 0.0
    results["desc_prec"] = np.mean(desc_precs) if desc_precs else 0.0
    results["desc_rec"] = np.mean(desc_recs) if desc_recs else 0.0
    results["desc_coverage"] = desc_covered / max(len(subset), 1)
    results["desc_n"] = desc_covered

    # ---- Creator dimension ----
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

        gains = []
        binary = []
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

        c_ndcg = ndcg_at_k(gains, ideal)
        c_map = average_precision_at_k(binary)
        c_mrr = mrr_at_k(binary)
        c_prec = precision_at_k(binary)
        c_rec = recall_at_k(binary, total_rel)

        cre_ndcgs.append(c_ndcg)
        cre_maps.append(c_map)
        cre_mrrs.append(c_mrr)
        cre_precs.append(c_prec)
        cre_recs.append(c_rec)

        for rec in per_doc:
            if rec["doc_idx"] == doc_i:
                rec["cre_ndcg"] = c_ndcg
                rec["cre_map"] = c_map
                break

    results["cre_ndcg"] = np.mean(cre_ndcgs) if cre_ndcgs else 0.0
    results["cre_map"] = np.mean(cre_maps) if cre_maps else 0.0
    results["cre_mrr"] = np.mean(cre_mrrs) if cre_mrrs else 0.0
    results["cre_prec"] = np.mean(cre_precs) if cre_precs else 0.0
    results["cre_rec"] = np.mean(cre_recs) if cre_recs else 0.0
    results["cre_coverage"] = cre_covered / max(len(subset), 1)
    results["cre_n"] = cre_covered

    # Unified nDCG
    results["unified_ndcg"] = (
        W_TAG * results["tag_ndcg"]
        + W_DESC * results["desc_ndcg"]
        + W_CRE * results["cre_ndcg"]
    )

    if verbose:
        logger.info(
            f"{method_name}: unified_nDCG={results['unified_ndcg']:.4f} "
            f"(tag={results['tag_ndcg']:.4f}, desc={results['desc_ndcg']:.4f}, "
            f"cre={results['cre_ndcg']:.4f}) "
            f"[tag_n={tag_covered}, desc_n={desc_covered}, cre_n={cre_covered}]"
        )

    return results, per_doc


# ---------------------------------------------------------------------------
# Multi-method evaluation
# ---------------------------------------------------------------------------

def evaluate_all_methods(
    methods_config: Dict[str, Dict[str, str]],
    subset: Set[int],
    standards: SilverStandards,
    dirs: Dict[str, Path],
    k_eval: int = 20,
    k_sim: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate multiple methods on a document subset.

    Args:
        methods_config: Dict keyed by method name, each value has
            ``prefix``, ``dir_key``, and ``group``.
        subset: Set of doc_idx to evaluate.
        standards: Silver standard data.
        dirs: Mapping from dir_key (e.g. ``"tmp"``, ``"content"``) to
            actual :class:`Path`.
        k_eval: Top-K for evaluation.
        k_sim: k value for manifest filenames.
        verbose: Print progress.

    Returns:
        DataFrame with one row per successfully evaluated method.
    """
    all_results: List[Dict[str, Any]] = []

    for method_name, config in methods_config.items():
        base_dir = dirs[config["dir_key"]]
        if verbose:
            logger.info(f"Evaluating: {method_name}")
        try:
            res, _ = evaluate_method_on_subset(
                config["prefix"],
                method_name,
                base_dir,
                subset,
                standards,
                k_eval=k_eval,
                k_sim=k_sim,
                verbose=verbose,
            )
            if res is not None:
                res["group"] = config.get("group", "")
                all_results.append(res)
        except (FileNotFoundError, KeyError, ValueError) as e:
            if verbose:
                logger.error(f"Failed to evaluate {method_name}: {e}")

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Improvement over baseline
# ---------------------------------------------------------------------------

def compute_improvement_over_baseline(
    metrics_df: pd.DataFrame,
    baseline: str = "Meta-only",
) -> pd.DataFrame:
    """Compute percentage improvement of each method over a baseline.

    Args:
        metrics_df: Output of :func:`evaluate_all_methods`.
        baseline: Name of the baseline method.

    Returns:
        DataFrame with improvement percentages for unified metrics.
    """
    unified_cols = [
        "unified_ndcg", "tag_ndcg", "desc_ndcg", "cre_ndcg",
        "tag_map", "desc_map", "cre_map",
    ]

    baseline_row = metrics_df[metrics_df["method"] == baseline]
    if len(baseline_row) == 0:
        return pd.DataFrame()

    base = baseline_row.iloc[0]
    rows = []
    for _, row in metrics_df.iterrows():
        if row["method"] == baseline:
            continue
        imp: Dict[str, Any] = {
            "method": row["method"],
            "group": row.get("group", ""),
        }
        for col in unified_cols:
            if col in row.index and col in base.index:
                bv = base[col]
                nv = row[col]
                if bv != 0:
                    imp[f"{col}_pct"] = ((nv - bv) / abs(bv)) * 100
                else:
                    imp[f"{col}_pct"] = float("nan")
        rows.append(imp)

    return pd.DataFrame(rows)
