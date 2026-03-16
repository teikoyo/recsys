#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 7 · FAISS ANN Graph Construction (DDP)
- 使用FAISS构建k-NN相似度图
- 支持GPU加速和DDP并行查询
- 分区存储大规模图数据

启动示例：
  torchrun --nproc_per_node=2 ddp_scripts/step7_faiss_ann_ddp.py \
    --tmp_dir ./tmp --k 50 --batch_q 8192 \
    --use_gpu true --index_type flat_ip
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not available")

# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("FAISS ANN Graph Builder (DDP)")

    p.add_argument("--tmp_dir", type=str, default="./tmp")
    p.add_argument("--k", type=int, default=50, help="Number of nearest neighbors")
    p.add_argument("--batch_q", type=int, default=8192, help="Query batch size")
    p.add_argument("--part_size", type=int, default=2_000_000, help="Partition size for saving edges")

    # FAISS parameters
    p.add_argument("--use_gpu", type=lambda s: s.lower() in ["true","1","yes"], default=True)
    p.add_argument("--index_type", type=str, default="flat_ip", choices=["flat_ip", "ivf_ip"])
    p.add_argument("--ivf_nlist", type=int, default=4096)
    p.add_argument("--ivf_nprobe", type=int, default=64)

    # Input/output
    p.add_argument("--tag_in", type=str, default="Z_tag.parquet")
    p.add_argument("--text_in", type=str, default="Z_text.parquet")
    p.add_argument("--tag_out_prefix", type=str, default="S_tag_topk")
    p.add_argument("--text_out_prefix", type=str, default="S_text_topk")

    return p.parse_args()

# ---------------------------
# DDP init
# ---------------------------
def init_ddp(backend="nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        return True, rank, world, local, dev
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, dev

def barrier(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.barrier()

def log0(is_ddp: bool, rank: int, msg: str):
    if (not is_ddp) or rank == 0:
        print(msg, flush=True)

# ---------------------------
# I/O helpers
# ---------------------------
def save_parquet_df(df, path):
    df.to_parquet(path, engine="fastparquet", index=False)

def load_parquet_df(path):
    return pd.read_parquet(path, engine="fastparquet")

# ---------------------------
# FAISS ANN search
# ---------------------------
def build_and_search_faiss(Z, k, batch_q, use_gpu, index_type, nlist, nprobe,
                          is_ddp=False, rank=0, world=1):
    """
    Build FAISS index and search for k-NN
    DDP: Each rank processes a subset of queries
    """
    if not HAS_FAISS:
        raise ImportError("faiss is required for ANN search")

    N, d = Z.shape
    Z = Z.astype(np.float32)

    # Normalize for inner product search
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norm = Z / np.maximum(norms, 1e-12)

    # Build index (all ranks build the same index)
    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(d)
    elif index_type == "ivf_ip":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(Z_norm)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Move to GPU if requested
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu_id = rank if is_ddp else 0
        if gpu_id < faiss.get_num_gpus():
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    # Add vectors (all ranks add all vectors for now, could optimize)
    index.add(Z_norm)

    # Set nprobe for IVF
    if index_type == "ivf_ip":
        index.nprobe = nprobe

    # DDP: Each rank processes a subset of queries
    if is_ddp:
        # Divide queries among ranks
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

    return np.array(all_rows), np.array(all_cols), np.array(all_scores, dtype=np.float32)

def gather_edges_ddp(rows, cols, vals, is_ddp, rank, world):
    """Gather edges from all ranks to rank 0"""
    if not is_ddp:
        return rows, cols, vals

    # Convert to tensors
    rows_t = torch.from_numpy(rows).long()
    cols_t = torch.from_numpy(cols).long()
    vals_t = torch.from_numpy(vals).float()

    # Gather sizes first
    local_size = torch.tensor([len(rows)], dtype=torch.long, device=f"cuda:{rank}")
    sizes = [torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}") for _ in range(world)]
    dist.all_gather(sizes, local_size)
    sizes = [s.item() for s in sizes]

    if rank == 0:
        # Prepare buffers
        all_rows = [torch.zeros(s, dtype=torch.long, device=f"cuda:{rank}") for s in sizes]
        all_cols = [torch.zeros(s, dtype=torch.long, device=f"cuda:{rank}") for s in sizes]
        all_vals = [torch.zeros(s, dtype=torch.float, device=f"cuda:{rank}") for s in sizes]
    else:
        all_rows = None
        all_cols = None
        all_vals = None

    # Move to GPU
    rows_t = rows_t.to(f"cuda:{rank}")
    cols_t = cols_t.to(f"cuda:{rank}")
    vals_t = vals_t.to(f"cuda:{rank}")

    # Gather
    dist.gather(rows_t, all_rows if rank == 0 else None, dst=0)
    dist.gather(cols_t, all_cols if rank == 0 else None, dst=0)
    dist.gather(vals_t, all_vals if rank == 0 else None, dst=0)

    if rank == 0:
        # Concatenate
        rows_cat = torch.cat(all_rows).cpu().numpy()
        cols_cat = torch.cat(all_cols).cpu().numpy()
        vals_cat = torch.cat(all_vals).cpu().numpy()
        return rows_cat, cols_cat, vals_cat
    else:
        return None, None, None

def save_partitioned_edges(rows, cols, vals, N, prefix, k, part_size, tmp_dir):
    """Save edges in partitions with manifest"""
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

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    is_ddp, rank, world, local, device = init_ddp("nccl")

    log0(is_ddp, rank, f"[DDP] enabled={is_ddp}, rank={rank}/{world}, device={device}")

    TMP = Path(args.tmp_dir)
    assert TMP.exists(), f"tmp_dir not found: {TMP}"

    # Process Tag view
    tag_in_path = TMP / args.tag_in
    if tag_in_path.exists():
        log0(is_ddp, rank, f"[Tag] Loading embeddings from {tag_in_path.name}")

        # All ranks load the embeddings
        Z_tag_df = load_parquet_df(tag_in_path)
        Z_tag = Z_tag_df[[c for c in Z_tag_df.columns if c.startswith('f')]].values
        N = len(Z_tag)

        log0(is_ddp, rank, f"[Tag] Building FAISS index (N={N}, k={args.k})")
        rows, cols, vals = build_and_search_faiss(
            Z_tag, args.k, args.batch_q, args.use_gpu,
            args.index_type, args.ivf_nlist, args.ivf_nprobe,
            is_ddp, rank, world
        )

        log0(is_ddp, rank, f"[Tag] Rank {rank}: Found {len(rows):,} edges")

        # Gather to rank 0 if DDP
        if is_ddp:
            rows, cols, vals = gather_edges_ddp(rows, cols, vals, is_ddp, rank, world)

        # Save (rank 0 only)
        if (not is_ddp) or rank == 0:
            manifest = save_partitioned_edges(
                rows, cols, vals, N, args.tag_out_prefix,
                args.k, args.part_size, TMP
            )
            log0(is_ddp, rank, f"[Tag] Saved {len(rows):,} edges to {manifest.name}")
    else:
        log0(is_ddp, rank, f"[Tag] Input not found: {tag_in_path}")

    barrier(is_ddp)

    # Process Text view
    text_in_path = TMP / args.text_in
    if text_in_path.exists():
        log0(is_ddp, rank, f"[Text] Loading embeddings from {text_in_path.name}")

        Z_text_df = load_parquet_df(text_in_path)
        Z_text = Z_text_df[[c for c in Z_text_df.columns if c.startswith('f')]].values
        N = len(Z_text)

        log0(is_ddp, rank, f"[Text] Building FAISS index (N={N}, k={args.k})")
        rows, cols, vals = build_and_search_faiss(
            Z_text, args.k, args.batch_q, args.use_gpu,
            args.index_type, args.ivf_nlist, args.ivf_nprobe,
            is_ddp, rank, world
        )

        log0(is_ddp, rank, f"[Text] Rank {rank}: Found {len(rows):,} edges")

        # Gather to rank 0 if DDP
        if is_ddp:
            rows, cols, vals = gather_edges_ddp(rows, cols, vals, is_ddp, rank, world)

        # Save (rank 0 only)
        if (not is_ddp) or rank == 0:
            manifest = save_partitioned_edges(
                rows, cols, vals, N, args.text_out_prefix,
                args.k, args.part_size, TMP
            )
            log0(is_ddp, rank, f"[Text] Saved {len(rows):,} edges to {manifest.name}")
    else:
        log0(is_ddp, rank, f"[Text] Input not found: {text_in_path}")

    barrier(is_ddp)
    log0(is_ddp, rank, "[Step 7] Complete")

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()
