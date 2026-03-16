#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sampling Utilities

Provides functions for negative sampling in SGNS training:
- build_ns_dist_from_deg: Create negative sampling distribution from degree
- build_alias_on_device: Build alias table for O(1) sampling on GPU
- sample_alias_gpu: Fast GPU-based alias sampling
"""

from typing import Tuple

import numpy as np
import torch

from .constants import NS_POWER_DEFAULT


def build_ns_dist_from_deg(deg: np.ndarray, power: float = NS_POWER_DEFAULT) -> np.ndarray:
    """
    Build negative sampling distribution from node degrees.

    Uses the word2vec convention: P(w) ∝ freq(w)^power
    where power=0.75 is the default to reduce sampling bias toward
    very frequent nodes.

    Args:
        deg: Array of node degrees (counts/frequencies)
        power: Exponent for frequency smoothing (default 0.75)

    Returns:
        Normalized probability distribution

    Example:
        >>> ns_dist = build_ns_dist_from_deg(degD, power=0.75)
        >>> assert np.isclose(ns_dist.sum(), 1.0)
    """
    p = np.power(np.maximum(deg, 1), power).astype(np.float64)
    p = p / p.sum()
    return p


def build_alias_on_device(
    probs_np: np.ndarray,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build alias table on GPU for O(1) sampling.

    Implements the alias method (Vose 1991) which allows sampling from
    a discrete distribution in O(1) time after O(n) preprocessing.

    Args:
        probs_np: Probability distribution (must sum to 1)
        device: Target device for tensors

    Returns:
        Tuple of (prob_table, alias_table) tensors for use with sample_alias_gpu

    Reference:
        Walker, A.J. "An efficient method for generating discrete random
        variables with general distributions." ACM TOMS, 1977.
    """
    p = probs_np.astype(np.float64, copy=True)
    n = p.size
    p = p / p.sum()

    prob = np.zeros(n, dtype=np.float32)
    alias = np.zeros(n, dtype=np.int32)
    scaled = p * n

    small = [i for i, x in enumerate(scaled) if x < 1.0]
    large = [i for i, x in enumerate(scaled) if x >= 1.0]

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = (scaled[l] + scaled[s]) - 1.0
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for i in large + small:
        prob[i] = 1.0
        alias[i] = i

    prob_t = torch.tensor(prob, dtype=torch.float32, device=device)
    alias_t = torch.tensor(alias, dtype=torch.int32, device=device)
    return prob_t, alias_t


def sample_alias_gpu(
    prob_t: torch.Tensor,
    alias_t: torch.Tensor,
    size: Tuple[int, ...],
    device: torch.device
) -> torch.Tensor:
    """
    Sample from alias table on GPU in O(1) per sample.

    Uses the pre-built alias table to draw samples efficiently.
    Fully vectorized for GPU execution.

    Args:
        prob_t: Probability table from build_alias_on_device
        alias_t: Alias table from build_alias_on_device
        size: Shape of output samples (e.g., (batch_size, num_negatives))
        device: Target device

    Returns:
        Tensor of sampled indices with shape `size`

    Example:
        >>> prob_t, alias_t = build_alias_on_device(ns_dist, device)
        >>> neg_samples = sample_alias_gpu(prob_t, alias_t, (1024, 10), device)
    """
    n = prob_t.size(0)
    k = torch.randint(n, size, device=device)
    u = torch.rand(size, device=device)
    return torch.where(u < prob_t[k], k, alias_t[k].to(k.dtype))
