#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS ANN Index Builder

Builds k-NN similarity graphs using FAISS for fast approximate nearest
neighbor search. Supports GPU acceleration and DDP parallelization.

Features:
- GPU-accelerated FAISS index building and search
- DDP support for parallel query processing
- Partitioned output for large-scale graphs
- Inner product similarity with normalized embeddings

Usage:
    # Build index for both views with 50 neighbors
    python scripts/build_ann_index.py --k 50 --use_gpu true

    # DDP mode for parallel processing
    torchrun --nproc_per_node=2 scripts/build_ann_index.py --k 100
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from src.ddp_utils import init_ddp, barrier, log0, cleanup_ddp

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not available. Install with: pip install faiss-gpu")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="FAISS ANN Graph Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--tmp_dir", type=str, default="./tmp",
                   help="Directory containing embeddings")
    p.add_argument("--k", type=int, default=50,
                   help="Number of nearest neighbors to find")
    p.add_argument("--batch_q", type=int, default=8192,
                   help="Query batch size for searching")
    p.add_argument("--part_size", type=int, default=2_000_000,
                   help="Partition size for saving edges")

    # FAISS parameters
    p.add_argument("--use_gpu", type=lambda s: s.lower() in ["true", "1", "yes"],
                   default=True, help="Use GPU for FAISS")
    p.add_argument("--index_type", type=str, default="flat_ip",
                   choices=["flat_ip", "ivf_ip"],
                   help="FAISS index type")
    p.add_argument("--ivf_nlist", type=int, default=4096,
                   help="Number of IVF clusters")
    p.add_argument("--ivf_nprobe", type=int, default=64,
                   help="Number of clusters to probe during search")

    # Input/output paths
    p.add_argument("--tag_in", type=str, default="Z_tag.parquet",
                   help="Tag embedding input file")
    p.add_argument("--text_in", type=str, default="Z_text.parquet",
                   help="Text embedding input file")
    p.add_argument("--tag_out_prefix", type=str, default="S_tag_topk",
                   help="Tag similarity output prefix")
    p.add_argument("--text_out_prefix", type=str, default="S_text_topk",
                   help="Text similarity output prefix")

    return p.parse_args()


def save_parquet_df(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to Parquet."""
    df.to_parquet(path, engine="fastparquet", index=False)


def load_parquet_df(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    return pd.read_parquet(path, engine="fastparquet")


def build_and_search_faiss(
    Z: np.ndarray,
    k: int,
    batch_q: int,
    use_gpu: bool,
    index_type: str,
    nlist: int,
    nprobe: int,
    is_ddp: bool = False,
    rank: int = 0,
    world: int = 1
):
    """
    Build FAISS index and search for k-NN.

    Args:
        Z: Embedding matrix [N, d]
        k: Number of neighbors
        batch_q: Query batch size
        use_gpu: Whether to use GPU
        index_type: Index type (flat_ip or ivf_ip)
        nlist: Number of IVF clusters
        nprobe: Number of probes for IVF
        is_ddp: Whether in DDP mode
        rank: Current rank
        world: World size

    Returns:
        Tuple of (rows, cols, scores) arrays
    """
    if not HAS_FAISS:
        raise ImportError("faiss is required for ANN search")

    N, d = Z.shape
    Z = Z.astype(np.float32)

    # L2 normalize for inner product search
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norm = Z / np.maximum(norms, 1e-12)

    # Build index
    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(d)
    elif index_type == "ivf_ip":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(Z_norm)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Move to GPU if available
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu_id = rank if is_ddp else 0
        if gpu_id < faiss.get_num_gpus():
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    # Add all vectors
    index.add(Z_norm)

    # Set nprobe for IVF
    if index_type == "ivf_ip":
        index.nprobe = nprobe

    # DDP: each rank processes a subset of queries
    if is_ddp:
        queries_per_rank = (N + world - 1) // world
        start_idx = rank * queries_per_rank
        end_idx = min(start_idx + queries_per_rank, N)
        query_indices = range(start_idx, end_idx)
    else:
        query_indices = range(N)

    # Search in batches
    all_rows, all_cols, all_scores = [], [], []

    query_list = list(query_indices)
    for batch_start in range(0, len(query_list), batch_q):
        batch_end = min(batch_start + batch_q, len(query_list))
        batch_indices = query_list[batch_start:batch_end]

        query = Z_norm[batch_indices]
        scores, indices = index.search(query, k + 1)  # +1 to exclude self

        for i, (row_scores, row_indices) in enumerate(zip(scores, indices)):
            query_idx = batch_indices[i]
            # Filter out self and invalid indices
            mask = (row_indices != query_idx) & (row_indices >= 0) & (row_indices < N)
            valid_cols = row_indices[mask][:k]
            valid_scores = row_scores[mask][:k]

            for col, score in zip(valid_cols, valid_scores):
                all_rows.append(query_idx)
                all_cols.append(int(col))
                all_scores.append(float(score))

    return (np.array(all_rows), np.array(all_cols),
            np.array(all_scores, dtype=np.float32))


def gather_edges_ddp(rows, cols, vals, is_ddp, rank, world):
    """Gather edges from all ranks to rank 0."""
    if not is_ddp:
        return rows, cols, vals

    device = f"cuda:{rank}"

    # Gather sizes first
    local_size = torch.tensor([len(rows)], dtype=torch.long, device=device)
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world)]
    dist.all_gather(sizes, local_size)
    sizes = [s.item() for s in sizes]

    # Convert to tensors
    rows_t = torch.from_numpy(rows).long().to(device)
    cols_t = torch.from_numpy(cols).long().to(device)
    vals_t = torch.from_numpy(vals).float().to(device)

    if rank == 0:
        all_rows = [torch.zeros(s, dtype=torch.long, device=device) for s in sizes]
        all_cols = [torch.zeros(s, dtype=torch.long, device=device) for s in sizes]
        all_vals = [torch.zeros(s, dtype=torch.float, device=device) for s in sizes]
    else:
        all_rows = all_cols = all_vals = None

    # Gather
    dist.gather(rows_t, all_rows if rank == 0 else None, dst=0)
    dist.gather(cols_t, all_cols if rank == 0 else None, dst=0)
    dist.gather(vals_t, all_vals if rank == 0 else None, dst=0)

    if rank == 0:
        rows_cat = torch.cat(all_rows).cpu().numpy()
        cols_cat = torch.cat(all_cols).cpu().numpy()
        vals_cat = torch.cat(all_vals).cpu().numpy()
        return rows_cat, cols_cat, vals_cat
    else:
        return None, None, None


def save_partitioned_edges(rows, cols, vals, N, prefix, k, part_size, tmp_dir):
    """Save edges in partitions with manifest."""
    df = pd.DataFrame({'row': rows, 'col': cols, 'val': vals})

    # Split into partitions
    num_parts = (len(df) + part_size - 1) // part_size
    manifest = {
        'N': int(N),
        'k': int(k),
        'total_edges': len(df),
        'num_parts': num_parts,
        'part_files': []
    }

    for i in range(num_parts):
        start = i * part_size
        end = min((i + 1) * part_size, len(df))
        part_df = df.iloc[start:end]

        part_file = f"{prefix}_k{k}_part{i:04d}.parquet"
        part_path = tmp_dir / part_file
        save_parquet_df(part_df, part_path)
        manifest['part_files'].append(part_file)

    # Save manifest
    manifest_path = tmp_dir / f"{prefix}_k{k}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def process_view(view_name, in_path, out_prefix, args, tmp_dir,
                 is_ddp, rank, world):
    """Process a single view (tag or text)."""
    if not in_path.exists():
        log0(is_ddp, rank, f"[{view_name}] Input not found: {in_path}")
        return

    log0(is_ddp, rank, f"[{view_name}] Loading embeddings from {in_path.name}")

    # Load embeddings
    Z_df = load_parquet_df(in_path)
    Z = Z_df[[c for c in Z_df.columns if c.startswith('f')]].values
    N = len(Z)

    log0(is_ddp, rank, f"[{view_name}] Building FAISS index (N={N:,}, k={args.k})")

    # Build and search
    rows, cols, vals = build_and_search_faiss(
        Z, args.k, args.batch_q, args.use_gpu,
        args.index_type, args.ivf_nlist, args.ivf_nprobe,
        is_ddp, rank, world
    )

    log0(is_ddp, rank, f"[{view_name}] Rank {rank}: Found {len(rows):,} edges")

    # Gather to rank 0 if DDP
    if is_ddp:
        rows, cols, vals = gather_edges_ddp(rows, cols, vals, is_ddp, rank, world)

    # Save (rank 0 only)
    if (not is_ddp) or rank == 0:
        manifest = save_partitioned_edges(
            rows, cols, vals, N, out_prefix,
            args.k, args.part_size, tmp_dir
        )
        log0(is_ddp, rank, f"[{view_name}] Saved {len(rows):,} edges to {manifest.name}")


def main():
    args = parse_args()
    is_ddp, rank, world, local, device = init_ddp("nccl")

    log0(is_ddp, rank, f"[DDP] enabled={is_ddp}, rank={rank}/{world}, device={device}")

    TMP = Path(args.tmp_dir)
    assert TMP.exists(), f"tmp_dir not found: {TMP}"

    # Process Tag view
    process_view("Tag", TMP / args.tag_in, args.tag_out_prefix,
                 args, TMP, is_ddp, rank, world)

    barrier(is_ddp)

    # Process Text view
    process_view("Text", TMP / args.text_in, args.text_out_prefix,
                 args, TMP, is_ddp, rank, world)

    barrier(is_ddp)
    log0(is_ddp, rank, "[Complete] ANN index building finished.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
