#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src.content - Content view extension modules

Provides reusable components for building a fourth (tabular content) view
on top of the existing three-view recommendation system.

Modules:
    - sampling: Table sampling, column profiling, and description generation
    - encoding: Sentence-transformer encoding and dataset vector aggregation
    - similarity: Sparse similarity graph construction and I/O
    - consistency: Meta-content consistency metrics (Jaccard, weighted)
    - fusion: Multi-view fusion with rho-adaptive weights and consistency adjustment
"""

from .sampling import (
    ColStats,
    read_by_ext,
    sample_table,
    profile_column,
    profile_table,
    col_to_description,
    TABULAR_EXTS,
    BLACKLIST_PATTERNS,
    select_main_table,
)
from .encoding import (
    W_TYPE,
    aggregate_dataset_vector,
)
from .similarity import (
    sym_and_rownorm,
    save_partitioned_edges,
    load_manifest_flexible,
    load_edges_from_manifest,
    build_neighbor_dict,
    load_csr_from_manifest,
)
from .consistency import compute_jaccard_and_consistency
from .fusion import (
    compute_rho,
    compute_adaptive_alpha,
    apply_consistency_adjustment,
    fuse_views,
)
from .evaluation import (
    SilverStandards,
    load_silver_standards,
    build_topk_for_method,
    evaluate_method_on_subset,
    evaluate_all_methods,
    compute_improvement_over_baseline,
    METHODS_CONFIG,
)
from .pipeline import (
    detect_device,
    run_content_pipeline,
    build_naive_fusion,
)
from .acquisition import (
    TABULAR_TAGS,
    BROAD_TAGS,
    FILTER_PRESETS,
    SEARCH_MARGIN,
    get_filter_preset,
    filter_candidates as filter_candidates_acq,
    check_kaggle_api,
    search_kaggle_slug,
    match_slug_to_ref,
    download_dataset,
    backfill_non_tabular,
    check_integrity,
)

__all__ = [
    # sampling
    'ColStats', 'read_by_ext', 'sample_table',
    'profile_column', 'profile_table', 'col_to_description',
    'TABULAR_EXTS', 'BLACKLIST_PATTERNS', 'select_main_table',
    # encoding
    'W_TYPE', 'aggregate_dataset_vector',
    # similarity
    'sym_and_rownorm', 'save_partitioned_edges',
    'load_manifest_flexible', 'load_edges_from_manifest',
    'build_neighbor_dict', 'load_csr_from_manifest',
    # consistency
    'compute_jaccard_and_consistency',
    # fusion
    'compute_rho', 'compute_adaptive_alpha',
    'apply_consistency_adjustment', 'fuse_views',
    # evaluation
    'SilverStandards', 'load_silver_standards',
    'build_topk_for_method', 'evaluate_method_on_subset',
    'evaluate_all_methods', 'compute_improvement_over_baseline',
    'METHODS_CONFIG',
    # pipeline
    'detect_device', 'run_content_pipeline', 'build_naive_fusion',
    # acquisition
    'TABULAR_TAGS', 'BROAD_TAGS', 'FILTER_PRESETS', 'SEARCH_MARGIN',
    'get_filter_preset', 'filter_candidates_acq',
    'check_kaggle_api', 'search_kaggle_slug', 'match_slug_to_ref',
    'download_dataset', 'backfill_non_tabular', 'check_integrity',
]
