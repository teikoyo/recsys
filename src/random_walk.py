#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Walk Corpus Generation

Provides classes and functions for generating random walk sentences on
bipartite graphs (Document-Tag or Document-Word). The walks are used
as input "sentences" for Skip-gram training.
"""

from typing import Iterator, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .csr_utils import csr_rowview_torch, csr_T


def _row_neighbors(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    data: torch.Tensor,
    r: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Get neighbors and weights for row r in CSR format."""
    a = indptr[r]
    b = indptr[r + 1]
    if (b - a).item() <= 0:
        return None, None
    sl = slice(a.item(), b.item())
    return indices[sl], data[sl]


def _sample_pos_by_weights(w: torch.Tensor, g: torch.Generator) -> int:
    """Sample a position from weighted distribution using CDF."""
    if w is None or w.numel() == 0:
        return -1
    w = torch.clamp(w, min=0)
    s = w.sum()
    if not torch.isfinite(s) or s.item() <= 0:
        return -1
    cdf = torch.cumsum(w, dim=0)
    u = torch.rand((), generator=g, device=w.device) * cdf[-1]
    pos = torch.searchsorted(cdf, u, right=False).item()
    return min(pos, cdf.numel() - 1)


class TorchWalkCorpus:
    """
    GPU-accelerated random walk corpus generator.

    Generates random walks on bipartite graphs (D-X-D pattern) where
    D is document nodes and X is intermediate nodes (tags or words).

    The walks follow the pattern:
        D0 -> X1 -> D1 -> X2 -> D2 -> ...

    But only document IDs are yielded, producing sentences like:
        [D0, D1, D2, D3, ...]

    Args:
        starts_np: Array of starting document indices
        DX: CSR row view (indptr, indices, data) for D->X edges
        XD: CSR row view for X->D edges (transpose of D->X)
        base_seed: Base random seed
        split_shards: Number of shards for DDP partitioning
        view_name: Name of this view (e.g., "tag" or "text")
        rw_params: Dict of random walk parameters
    """

    def __init__(
        self,
        starts_np: np.ndarray,
        DX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        XD: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        base_seed: int,
        split_shards: int = 64,
        view_name: str = "",
        rw_params: Optional[dict] = None
    ):
        self.starts_np = starts_np
        self.DX = DX
        self.XD = XD
        self.base_seed = base_seed
        self.split_shards = split_shards
        self.view_name = view_name
        self._iters = 0

        # Default random walk parameters
        self.params = {
            'walks_per_doc': 3,
            'l_docs_per_sent': 30,
            'seed_base': base_seed,
            'avoid_backtrack': True,
            'restart_prob': 0.15,
            'x_degree_pow': 0.0,
            'x_no_repeat_last': 1,
        }
        if rw_params:
            self.params.update(rw_params)

    def __len__(self) -> int:
        return int(len(self.starts_np) * self.params['walks_per_doc'])

    def iterate(
        self,
        device: torch.device,
        is_ddp: bool = False,
        rank: int = 0,
        world: int = 1
    ) -> Iterator[List[str]]:
        """
        Iterate over random walk sentences.

        Args:
            device: Target device for computation
            is_ddp: Whether running in DDP mode
            rank: Current process rank
            world: Total number of processes

        Yields:
            List of document IDs as strings (sentence)
        """
        self._iters += 1
        rng = np.random.default_rng(
            self.params['seed_base'] + 31 * self._iters + rank * 1009
        )
        starts = self.starts_np.copy()
        rng.shuffle(starts)
        shards = np.array_split(starts, max(1, self.split_shards))

        indptr_D, indices_D, data_D = self.DX
        indptr_X, indices_X, data_X = self.XD

        for sid, shard in enumerate(shards):
            # DDP: each rank processes its subset of shards
            if is_ddp and (sid % world) != rank:
                continue

            g = torch.Generator(device=device)
            g.manual_seed(self.base_seed + 7919 * (self._iters + sid + rank * 101))

            # Precompute X degree factor
            x_factor = None
            if abs(self.params['x_degree_pow']) > 1e-12:
                x_deg = (indptr_X[1:] - indptr_X[:-1]).to(torch.float32)
                x_factor = torch.clamp(x_deg, min=1.0).pow(self.params['x_degree_pow'])

            for d0 in shard:
                for _ in range(self.params['walks_per_doc']):
                    seq = [int(d0)]
                    prev_d = None
                    cur_d = int(d0)
                    last_x = -1

                    for _step in range(self.params['l_docs_per_sent'] - 1):
                        # Sample X from current D
                        r = torch.tensor(cur_d, dtype=torch.long, device=device)
                        x_cols, x_w = _row_neighbors(indptr_D, indices_D, data_D, r)
                        if x_cols is None:
                            break

                        w = x_w.clone()
                        if x_factor is not None:
                            w = w * x_factor[x_cols]

                        # Avoid repeating last X
                        if self.params['x_no_repeat_last'] > 0 and last_x >= 0:
                            m = (x_cols == last_x)
                            if m.any():
                                w[m] = 0.0

                        px = _sample_pos_by_weights(w, g)
                        if px < 0:
                            break
                        x = int(x_cols[px].item())

                        # Sample D from X
                        xr = torch.tensor(x, dtype=torch.long, device=device)
                        d_rows, d_w = _row_neighbors(indptr_X, indices_X, data_X, xr)
                        if d_rows is None:
                            break

                        # Avoid backtracking
                        if self.params['avoid_backtrack'] and prev_d is not None and d_rows.numel() > 1:
                            m = (d_rows == prev_d)
                            if m.any():
                                d_w = d_w.clone()
                                d_w[m] = 0.0

                        pdx = _sample_pos_by_weights(d_w, g)
                        if pdx < 0:
                            break
                        next_d = int(d_rows[pdx].item())

                        # Random restart
                        if torch.rand((), generator=g, device=device).item() < self.params['restart_prob']:
                            next_d = int(d0)

                        seq.append(next_d)
                        prev_d, cur_d, last_x = cur_d, next_d, x

                    if len(seq) >= 2:
                        yield [str(s) for s in seq]


def build_corpus(
    doc_df: pd.DataFrame,
    DT_ppmi: sparse.csr_matrix,
    DW_bm25: sparse.csr_matrix,
    rw_params: pd.Series,
    device: torch.device,
    is_ddp: bool,
    rank: int,
    world: int
) -> Tuple[TorchWalkCorpus, TorchWalkCorpus, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Tag and Text view random walk corpora.

    Args:
        doc_df: Document dataframe
        DT_ppmi: Document-Tag PPMI matrix
        DW_bm25: Document-Word BM25 matrix
        rw_params: Random walk parameters from config
        device: Target device
        is_ddp: Whether running in DDP mode
        rank: Current process rank
        world: Total number of processes

    Returns:
        Tuple of:
            - tag_corpus: TorchWalkCorpus for tag view
            - text_corpus: TorchWalkCorpus for text view
            - start_tag: Array of starting docs for tag view
            - start_txt: Array of starting docs for text view
            - degD_tag: Document degrees in tag graph
            - degD_txt: Document degrees in text graph
    """
    N = len(doc_df)

    # Find valid starting documents (those with at least one connection)
    degD_tag = np.diff(DT_ppmi.indptr)
    start_tag = np.where(degD_tag > 0)[0].astype(np.int64)

    degD_txt = np.diff(DW_bm25.indptr)
    start_txt = np.where(degD_txt > 0)[0].astype(np.int64)

    # Parse random walk parameters
    params = {
        'walks_per_doc': int(rw_params["RW_WALKS_PER_DOC"]),
        'l_docs_per_sent': int(rw_params["RW_L_DOCS_PER_SENT"]),
        'seed_base': int(rw_params["RW_SEED_BASE"]),
        'avoid_backtrack': bool(rw_params["RW_AVOID_BACKTRACK"]),
        'restart_prob': float(rw_params["RW_RESTART_PROB"]),
        'x_degree_pow': float(rw_params["RW_X_DEGREE_POW"]),
        'x_no_repeat_last': int(rw_params["RW_X_NO_REPEAT_LAST"]),
    }

    # Build CSR row views on device
    DX_tag = csr_rowview_torch(DT_ppmi, device)
    XD_tag = csr_rowview_torch(csr_T(DT_ppmi), device)
    DX_txt = csr_rowview_torch(DW_bm25, device)
    XD_txt = csr_rowview_torch(csr_T(DW_bm25), device)

    # Create corpus objects
    tag_corpus = TorchWalkCorpus(
        start_tag, DX_tag, XD_tag,
        base_seed=params['seed_base'] + 11,
        split_shards=64,
        view_name="tag",
        rw_params=params
    )

    text_corpus = TorchWalkCorpus(
        start_txt, DX_txt, XD_txt,
        base_seed=params['seed_base'] + 23,
        split_shards=64,
        view_name="text",
        rw_params=params
    )

    return tag_corpus, text_corpus, start_tag, start_txt, degD_tag, degD_txt
