"""Centralized constants for WS-SGNS recommendation system."""

# Random walk seed multipliers (coprime to avoid period collisions)
RW_ITER_SEED_MULT = 31
RW_RANK_SEED_MULT = 1009
RW_SHARD_SEED_MULT = 7919
RW_RANK_SHARD_SEED_MULT = 101

# View seed offsets
TAG_VIEW_SEED_OFFSET = 11
TEXT_VIEW_SEED_OFFSET = 23

# Negative sampling
NS_POWER_DEFAULT = 0.75  # word2vec convention: P(w) ~ freq(w)^0.75

# Column profiling thresholds
TAU_NUMERIC = 0.95
TAU_DATETIME = 0.90
TEXT_AVG_CHAR_LEN = 30
ID_UNIQUE_RATIO = 0.95

# Evaluation weights
W_TAG_EVAL = 0.5
W_DESC_EVAL = 0.3
W_CREATOR_EVAL = 0.2
DESC_SIM_THRESHOLD = 0.2

# Defaults
TOP_K_DEFAULT = 50
CORPUS_SIZE_DEFAULT = 521735
