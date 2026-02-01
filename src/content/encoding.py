#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding Encoding and Dataset Vector Aggregation

Implements CONTENT_VIEW_EXTENSION.md section 3.4 (column vector encoding)
and section 3.5 (dataset vector aggregation Z_tabcontent).
"""

import numpy as np

# Type weights for dataset vector aggregation (section 3.5)
W_TYPE = {"numeric": 1.0, "categorical": 1.0, "datetime": 0.8, "text": 1.2}


def aggregate_dataset_vector(col_embs, col_stats_list):
    """Aggregate column embeddings into a single dataset vector.

    Computes a weighted mean of column embeddings where each column's
    weight is ``w_type * (1 - missing_pct/100) * min(unique_pct/100, 0.95)``,
    then L2-normalises the result.

    Args:
        col_embs: (N_cols, embed_dim) array of column embeddings.
        col_stats_list: List of dicts with keys 'dtype', 'missing_pct',
            'unique_pct' for each column.

    Returns:
        L2-normalised (embed_dim,) dataset vector as float32.
    """
    weights = []
    for cs in col_stats_list:
        w_type = W_TYPE.get(cs["dtype"], 1.0)
        w_quality = (1.0 - cs["missing_pct"] / 100.0) * min(cs["unique_pct"] / 100.0, 0.95)
        weights.append(w_type * max(w_quality, 1e-6))

    weights = np.array(weights, dtype=np.float32)
    w_sum = weights.sum()
    if w_sum < 1e-12:
        return np.zeros(col_embs.shape[1], dtype=np.float32)

    z = (col_embs * weights[:, None]).sum(axis=0) / w_sum
    norm = np.linalg.norm(z)
    if norm > 1e-12:
        z = z / norm
    return z.astype(np.float32)
