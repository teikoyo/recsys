"""Tests for src.constants -- seed multipliers, eval weights, NS power."""

import math

import pytest

from src.constants import (
    RW_ITER_SEED_MULT,
    RW_RANK_SEED_MULT,
    RW_SHARD_SEED_MULT,
    RW_RANK_SHARD_SEED_MULT,
    W_TAG_EVAL,
    W_DESC_EVAL,
    W_CREATOR_EVAL,
    NS_POWER_DEFAULT,
)


def test_seed_multipliers_coprime():
    """All 4 random walk seed multipliers are pairwise coprime."""
    mults = [RW_ITER_SEED_MULT, RW_RANK_SEED_MULT,
             RW_SHARD_SEED_MULT, RW_RANK_SHARD_SEED_MULT]
    for i in range(len(mults)):
        for j in range(i + 1, len(mults)):
            assert math.gcd(mults[i], mults[j]) == 1, (
                f"{mults[i]} and {mults[j]} are not coprime"
            )


def test_eval_weights_sum_to_one():
    """W_TAG_EVAL + W_DESC_EVAL + W_CREATOR_EVAL == 1.0."""
    total = W_TAG_EVAL + W_DESC_EVAL + W_CREATOR_EVAL
    assert abs(total - 1.0) < 1e-12


def test_ns_power_positive():
    """NS_POWER_DEFAULT > 0."""
    assert NS_POWER_DEFAULT > 0
