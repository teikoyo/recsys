# WS-SGNS: Multi-View Dataset Recommendation System

A recommendation system based on random-walk Skip-Gram with Negative Sampling (SGNS).
It learns dataset embeddings from two complementary views -- tags and textual content --
and fuses them to produce high-quality dataset-to-dataset recommendations.

## Installation

```bash
# Core install
pip install -e .

# With content-view dependencies (transformers, sentence-transformers, etc.)
pip install -e ".[content]"

# Development / testing
pip install -e ".[dev]"
```

## Quick Start

### Train embeddings

```bash
# Dual-view training (Tag + Text)
python scripts/train_sgns.py --views tag,text --epochs 4

# Text view only
python scripts/train_sgns.py --views text

# Tag view only
python scripts/train_sgns.py --views tag

# Multi-GPU training with DDP
torchrun --nproc_per_node=2 scripts/train_sgns.py --views tag,text
```

### Build an ANN index

```bash
python scripts/build_ann_index.py --k 50 --use_gpu true
```

### Analyze random walks

```bash
python scripts/analyze_walks.py --base_dir ./tmp --output_dir ./analysis_outputs
```

## Project Structure

```
recsys-new/
├── pyproject.toml          # Package configuration
├── src/                    # Core library
│   ├── __init__.py
│   ├── constants.py        # Centralized constants
│   ├── config.py           # Dataclass configuration
│   ├── log.py              # Structured logging
│   ├── ddp_utils.py        # DDP utilities
│   ├── csr_utils.py        # CSR matrix I/O
│   ├── sampling_utils.py   # Negative sampling
│   ├── pair_batch_utils.py # Pair generation and batching
│   ├── random_walk.py      # Random walk corpus generation
│   ├── sgns_model.py       # SGNS model
│   ├── metrics.py          # Evaluation metrics
│   └── content/            # Content view extension
├── scripts/                # Executable scripts
├── tests/                  # pytest test suite
├── notebooks/              # Research notebooks
├── docs/                   # Documentation
├── data/                   # Raw data
└── tmp/                    # Intermediate results
```

## Configuration

Training can be driven by a JSON config file:

```bash
python scripts/train_sgns.py --config config.json
```

Example `config.json`:

```json
{
  "views": ["tag", "text"],
  "epochs": 4,
  "dim": 256,
  "neg": 10,
  "lr": 0.025,
  "window_tag": 5,
  "window_text": 4,
  "batch_pairs_tag": 204800,
  "batch_pairs_text": 204800,
  "amp": true
}
```

## Core Modules

| Module | Description |
|--------|-------------|
| `src/constants.py` | Centralized project constants (paths, defaults) |
| `src/config.py` | Dataclass-based configuration with JSON support |
| `src/log.py` | Structured logging helpers |
| `src/ddp_utils.py` | DDP initialization, synchronization barriers, rank-aware logging |
| `src/csr_utils.py` | Load and save sparse CSR matrices (Parquet format) |
| `src/sampling_utils.py` | GPU alias-table construction and O(1) negative sampling |
| `src/pair_batch_utils.py` | Pair generation from walk corpora and mini-batch assembly |
| `src/random_walk.py` | GPU-accelerated random walk corpus generation for Tag and Text views |
| `src/sgns_model.py` | Skip-Gram with Negative Sampling model |
| `src/metrics.py` | Evaluation metrics: nDCG@K, MAP@K, MRR@K, Precision@K, Recall@K |
| `src/content/` | Content-view extension (transformer-based text features) |

## License

MIT License
