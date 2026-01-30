#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pair Generation and Batching Utilities

Provides functions for generating (center, context) pairs from random walk
sentences and batching them with negative samples for SGNS training.
"""

import random
from typing import Iterator, Optional, Tuple, List

import torch

from .sampling_utils import sample_alias_gpu


def iter_pairs_from_corpus(
    corpus: Iterator[List[str]],
    window: int,
    max_sents: Optional[int] = None,
    seed: int = 42,
    keep_prob: float = 1.0,
    forward_only: bool = False,
    ctx_cap: int = 0
) -> Iterator[Tuple[int, int]]:
    """
    Generate (center, context) pairs from a sentence corpus.

    Implements Skip-gram pair generation with dynamic window size and
    optional subsampling strategies.

    Args:
        corpus: Iterator yielding sentences (lists of token strings)
        window: Maximum window size (actual size is random in [1, window])
        max_sents: Maximum number of sentences to process (None for all)
        seed: Random seed for reproducibility
        keep_prob: Probability of keeping each context pair (for subsampling)
        forward_only: If True, only use forward context (i+1 to i+w)
        ctx_cap: Maximum contexts per center word (0 for unlimited)

    Yields:
        (center_id, context_id) tuples

    Example:
        >>> for center, context in iter_pairs_from_corpus(corpus, window=5):
        ...     # Process pair
    """
    rng = random.Random(seed)
    sent_count = 0

    for sent in corpus:
        if max_sents is not None and sent_count >= max_sents:
            break

        s = [int(x) for x in sent]
        L = len(s)

        for i in range(L):
            # Dynamic window size
            w = rng.randint(1, window)
            l = max(0, i - w)
            r = min(L - 1, i + w)

            # Build candidate contexts
            if forward_only:
                cand = list(range(i + 1, r + 1))
            else:
                cand = list(range(l, r + 1))
                if i in cand:
                    cand.remove(i)

            # Context cap (limit contexts per center)
            if ctx_cap and len(cand) > ctx_cap:
                rng.shuffle(cand)
                cand = cand[:ctx_cap]

            # Subsampling
            if keep_prob < 1.0:
                cand = [j for j in cand if rng.random() < keep_prob]

            for j in cand:
                yield s[i], s[j]

        sent_count += 1


def batch_pairs_and_negs_fast(
    pair_iter: Iterator[Tuple[int, int]],
    batch_size_pairs: int,
    negK: int,
    ns_prob_t: torch.Tensor,
    ns_alias_t: torch.Tensor,
    device: torch.device
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Batch (center, context) pairs and generate negative samples.

    Collects pairs into batches and generates GPU-based negative samples
    using the alias method for efficient SGNS training.

    Args:
        pair_iter: Iterator of (center, context) tuples
        batch_size_pairs: Number of pairs per batch
        negK: Number of negative samples per positive pair
        ns_prob_t: Alias probability table (from build_alias_on_device)
        ns_alias_t: Alias table (from build_alias_on_device)
        device: Target device for tensors

    Yields:
        (centers, contexts, negatives) tensor tuples:
            - centers: [B] tensor of center indices
            - contexts: [B] tensor of context indices
            - negatives: [B, K] tensor of negative sample indices

    Example:
        >>> for centers, contexts, negs in batch_pairs_and_negs_fast(
        ...         pair_iter, 32768, 10, prob_t, alias_t, device):
        ...     loss = model(centers, contexts, negs)
    """
    centers, contexts = [], []

    for c, x in pair_iter:
        centers.append(c)
        contexts.append(x)

        if len(centers) >= batch_size_pairs:
            B = len(centers)
            negs_t = sample_alias_gpu(ns_prob_t, ns_alias_t, size=(B, negK), device=device)
            centers_t = torch.tensor(centers, dtype=torch.long, device=device)
            contexts_t = torch.tensor(contexts, dtype=torch.long, device=device)
            yield centers_t, contexts_t, negs_t
            centers.clear()
            contexts.clear()

    # Yield remaining pairs
    if centers:
        B = len(centers)
        negs_t = sample_alias_gpu(ns_prob_t, ns_alias_t, size=(B, negK), device=device)
        centers_t = torch.tensor(centers, dtype=torch.long, device=device)
        contexts_t = torch.tensor(contexts, dtype=torch.long, device=device)
        yield centers_t, contexts_t, negs_t
