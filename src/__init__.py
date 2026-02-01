#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src - Core modules for WS-SGNS recommendation system

This package provides reusable components for training Skip-gram with
Negative Sampling (SGNS) models using random walks on bipartite graphs.

Modules:
    - ddp_utils: Distributed Data Parallel utilities
    - csr_utils: CSR sparse matrix I/O operations
    - sampling_utils: Negative sampling and alias method
    - pair_batch_utils: Pair generation and batching
    - random_walk: Random walk corpus generation
    - sgns_model: SGNS model definition
    - metrics: Evaluation metrics (nDCG, MAP, MRR, etc.)

Subpackages:
    - content: Content view extension (table sampling, encoding, similarity,
               consistency, and multi-view fusion)
"""

from .ddp_utils import init_ddp, barrier, log0
from .csr_utils import load_csr_triplet_parquet, csr_rowview_torch, csr_T
from .sampling_utils import build_ns_dist_from_deg, build_alias_on_device, sample_alias_gpu
from .pair_batch_utils import iter_pairs_from_corpus, batch_pairs_and_negs_fast
from .random_walk import build_corpus, TorchWalkCorpus
from .sgns_model import SGNS
from .metrics import (
    dcg_at_k, ndcg_at_k, average_precision_at_k, mrr_at_k,
    precision_at_k, recall_at_k
)

__all__ = [
    # DDP utilities
    'init_ddp', 'barrier', 'log0',
    # CSR utilities
    'load_csr_triplet_parquet', 'csr_rowview_torch', 'csr_T',
    # Sampling utilities
    'build_ns_dist_from_deg', 'build_alias_on_device', 'sample_alias_gpu',
    # Pair/batch utilities
    'iter_pairs_from_corpus', 'batch_pairs_and_negs_fast',
    # Random walk
    'build_corpus', 'TorchWalkCorpus',
    # Model
    'SGNS',
    # Metrics
    'dcg_at_k', 'ndcg_at_k', 'average_precision_at_k', 'mrr_at_k',
    'precision_at_k', 'recall_at_k',
]

__version__ = '1.0.0'
