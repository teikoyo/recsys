#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified WS-SGNS Training Script

Trains Skip-gram with Negative Sampling embeddings using random walks on
bipartite graphs. Supports Tag view, Text view, or both (dual-view).

Features:
- DDP (Distributed Data Parallel) support via torchrun
- AMP (Automatic Mixed Precision) + TF32 acceleration
- GPU-based alias negative sampling (O(1) per sample)
- Flexible per-view parameters (window, keep_prob, etc.)
- Gradient accumulation for large effective batch sizes
- Per-epoch checkpointing and lightweight evaluation

Usage:
    # Dual-view (tag + text)
    torchrun --nproc_per_node=2 scripts/train_sgns.py --views tag,text

    # Text-only
    python scripts/train_sgns.py --views text --epochs 4

    # Tag-only with custom parameters
    python scripts/train_sgns.py --views tag --window_tag 8 --batch_pairs_tag 100000
"""

import os
import gc
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ddp_utils import init_ddp, barrier, log0, cleanup_ddp
from src.csr_utils import load_csr_triplet_parquet
from src.sampling_utils import build_ns_dist_from_deg, build_alias_on_device
from src.pair_batch_utils import iter_pairs_from_corpus, batch_pairs_and_negs_fast
from src.random_walk import build_corpus
from src.sgns_model import SGNS


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="WS-SGNS Trainer (Unified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core settings
    p.add_argument("--tmp_dir", type=str, default="./tmp",
                   help="Directory containing preprocessed data")
    p.add_argument("--views", type=str, default="tag,text",
                   help="Views to train: 'tag', 'text', or 'tag,text'")
    p.add_argument("--epochs", type=int, default=4,
                   help="Number of training epochs per view")
    p.add_argument("--dim", type=int, default=256,
                   help="Embedding dimension")
    p.add_argument("--neg", type=int, default=10,
                   help="Number of negative samples per positive")
    p.add_argument("--lr", type=float, default=0.025,
                   help="Learning rate")
    p.add_argument("--ns_power", type=float, default=0.75,
                   help="Power for negative sampling distribution")
    p.add_argument("--seed", type=int, default=2025,
                   help="Random seed")
    p.add_argument("--accum", type=int, default=1,
                   help="Gradient accumulation steps")

    # Optimizer settings
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "sparse_adam", "adagrad"],
                   help="Optimizer type")
    p.add_argument("--sparse", type=lambda s: s.lower() in ["true", "1", "yes"],
                   default=False, help="Use sparse embeddings")

    # Numeric precision
    p.add_argument("--amp", type=lambda s: s.lower() in ["true", "1", "yes"],
                   default=True, help="Enable AMP (mixed precision)")
    p.add_argument("--tf32", type=lambda s: s.lower() in ["true", "1", "yes"],
                   default=True, help="Enable TF32 acceleration")

    # Tag view parameters
    p.add_argument("--window_tag", type=int, default=5,
                   help="Window size for tag view")
    p.add_argument("--keep_prob_tag", type=float, default=1.0,
                   help="Context keep probability for tag view")
    p.add_argument("--forward_only_tag",
                   type=lambda s: s.lower() in ["true", "1", "yes"],
                   default=False, help="Forward-only context for tag view")
    p.add_argument("--ctx_cap_tag", type=int, default=0,
                   help="Context cap per center for tag view (0=unlimited)")
    p.add_argument("--batch_pairs_tag", type=int, default=204800,
                   help="Batch size (pairs) for tag view")
    p.add_argument("--max_pairs_tag", type=int, default=20_000_000,
                   help="Max pairs per epoch for tag view")
    p.add_argument("--max_sents_tag", type=int, default=None,
                   help="Max sentences for tag view (None=unlimited)")

    # Text view parameters
    p.add_argument("--window_text", type=int, default=4,
                   help="Window size for text view")
    p.add_argument("--keep_prob_text", type=float, default=0.35,
                   help="Context keep probability for text view")
    p.add_argument("--forward_only_text",
                   type=lambda s: s.lower() in ["true", "1", "yes"],
                   default=True, help="Forward-only context for text view")
    p.add_argument("--ctx_cap_text", type=int, default=4,
                   help="Context cap per center for text view (0=unlimited)")
    p.add_argument("--batch_pairs_text", type=int, default=204800,
                   help="Batch size (pairs) for text view")
    p.add_argument("--max_pairs_text", type=int, default=20_000_000,
                   help="Max pairs per epoch for text view")
    p.add_argument("--max_sents_text", type=int, default=None,
                   help="Max sentences for text view (None=unlimited)")

    # Logging and checkpointing
    p.add_argument("--log_every", type=int, default=200,
                   help="Log every N steps")
    p.add_argument("--eval_samples_per_view", type=int, default=3,
                   help="Number of samples for lightweight eval")
    p.add_argument("--eval_topk", type=int, default=5,
                   help="Top-K for lightweight eval")
    p.add_argument("--save_epoch_emb",
                   type=lambda s: s.lower() in ["true", "1", "yes"],
                   default=True, help="Save embeddings each epoch")
    p.add_argument("--emb_dtype", type=str, default="float32",
                   choices=["float32", "float16"],
                   help="Embedding save precision")

    return p.parse_args()


def pick_eval_samples(starts_np: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Pick random samples for lightweight evaluation."""
    if k <= 0 or len(starts_np) == 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    k = min(k, len(starts_np))
    return rng.choice(starts_np, size=k, replace=False).astype(np.int64)


def quick_eval_neighbors(model: nn.Module, sample_idx: np.ndarray,
                         topk: int, doc_ids: np.ndarray) -> dict:
    """Quick nearest neighbor evaluation."""
    E = (model.module.in_emb.weight if hasattr(model, "module")
         else model.in_emb.weight).detach().cpu().numpy()
    E = E.astype(np.float32, copy=False)

    # L2 normalize
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    mask = (nrm[:, 0] > 0)
    E[mask] = E[mask] / nrm[mask]

    results = {}
    for d in sample_idx:
        v = E[d:d + 1]
        scores = E @ v[0]
        scores[d] = -1.0  # Exclude self
        nn_idx = np.argpartition(scores, -topk)[-topk:]
        nn_idx = nn_idx[np.argsort(scores[nn_idx])][::-1]
        results[int(d)] = [(int(i), float(scores[i]), int(doc_ids[i])) for i in nn_idx]
    return results


def train_view(view_name: str, N: int, start_nodes: np.ndarray,
               degD: np.ndarray, corpus, device: torch.device,
               is_ddp: bool, rank: int, args, out_path: Path,
               doc_ids: np.ndarray):
    """Train embeddings for a single view."""
    torch.manual_seed(args.seed + (11 if view_name == "tag" else 23))

    # DDP + sparse is unstable, force dense
    sparse_rt = bool(args.sparse)
    if is_ddp and sparse_rt:
        log0(is_ddp, rank, f"[Warn] DDP + sparse unstable, switching to dense.")
        sparse_rt = False

    model = SGNS(vocab_size=N, dim=args.dim, sparse=sparse_rt).to(device)

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # Setup optimizer
    if sparse_rt and args.optimizer.lower() == "sparse_adam":
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)
    elif sparse_rt and args.optimizer.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(list(model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # AMP scaler
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Build negative sampling alias table
    ns_dist = build_ns_dist_from_deg(degD, power=args.ns_power)
    ns_prob_t, ns_alias_t = build_alias_on_device(ns_dist, device)

    # Get per-view parameters
    if view_name == "tag":
        v_window = args.window_tag
        v_keep = args.keep_prob_tag
        v_forward = args.forward_only_tag
        v_cap = args.ctx_cap_tag
        v_batch = args.batch_pairs_tag
        v_max_pairs = args.max_pairs_tag
        v_max_sents = args.max_sents_tag
    else:
        v_window = args.window_text
        v_keep = args.keep_prob_text
        v_forward = args.forward_only_text
        v_cap = args.ctx_cap_text
        v_batch = args.batch_pairs_text
        v_max_pairs = args.max_pairs_text
        v_max_sents = args.max_sents_text

    eval_samples = pick_eval_samples(start_nodes, args.eval_samples_per_view, args.seed + 7)

    log0(is_ddp, rank,
         f"[Train-{view_name}] epochs={args.epochs}, dim={args.dim}, "
         f"window={v_window}, neg={args.neg}, batch={v_batch}, "
         f"accum={args.accum}, AMP={args.amp}, DDP={is_ddp}")

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        # Create corpus iterator
        corpus_iter = corpus.iterate(device, is_ddp, rank,
                                      dist.get_world_size() if is_ddp else 1)

        pair_iter = iter_pairs_from_corpus(
            corpus=corpus_iter,
            window=v_window,
            max_sents=v_max_sents,
            seed=args.seed + ep + (0 if view_name == "tag" else 1000),
            keep_prob=v_keep,
            forward_only=v_forward,
            ctx_cap=v_cap,
        )

        total_pairs = 0
        total_loss = 0.0
        step = 0
        last_t = time.time()
        pairs_since_last = 0

        model.train()
        for centers_t, contexts_t, negs_t in batch_pairs_and_negs_fast(
                pair_iter, v_batch, args.neg, ns_prob_t, ns_alias_t, device):

            optimizer.zero_grad(set_to_none=True)

            B = centers_t.size(0)
            accum = max(1, int(args.accum))
            micro = (B + accum - 1) // accum

            # Micro-batch loop for gradient accumulation
            for s in range(0, B, micro):
                c_mb = centers_t[s:s + micro]
                x_mb = contexts_t[s:s + micro]
                n_mb = negs_t[s:s + micro]

                try:
                    cm = torch.amp.autocast('cuda', enabled=args.amp)
                except TypeError:
                    cm = torch.cuda.amp.autocast(enabled=args.amp)

                with cm:
                    loss = model(c_mb, x_mb, n_mb)
                    if hasattr(loss, "dim") and loss.dim() != 0:
                        loss = loss.mean()
                    loss = loss / accum

                scaler.scale(loss).backward()
                total_loss += float(loss.detach().item()) * accum * c_mb.size(0)

            scaler.step(optimizer)
            scaler.update()

            total_pairs += B
            pairs_since_last += B
            step += 1

            if step % args.log_every == 0 and (not is_ddp or rank == 0):
                now = time.time()
                dt = max(1e-9, now - last_t)
                thr = pairs_since_last / dt
                mem = 0.0
                if device.type == "cuda":
                    try:
                        mem = torch.cuda.memory_allocated(device=device) / (1024 ** 2)
                    except Exception:
                        pass
                print(f"[{view_name}] step={step:,} throughput={thr:,.0f} pairs/s "
                      f"loss~{total_loss / max(1, total_pairs):.4f} mem~{mem:.0f}MB",
                      flush=True)
                last_t = now
                pairs_since_last = 0

            if v_max_pairs is not None and total_pairs >= v_max_pairs:
                if (not is_ddp) or rank == 0:
                    print(f"[{view_name}] early stop epoch {ep}: "
                          f"reached max_pairs={v_max_pairs:,}", flush=True)
                break

        dt = time.time() - t0
        if (not is_ddp) or rank == 0:
            if total_pairs == 0:
                print(f"[Train-{view_name}] epoch {ep}: no pairs produced")
            else:
                print(f"[Train-{view_name}] epoch {ep}: pairs={total_pairs:,}, "
                      f"avg_loss={total_loss / max(1, total_pairs):.4f}, "
                      f"time={dt:.1f}s", flush=True)

            # Quick evaluation
            if len(eval_samples) > 0:
                nn_res = quick_eval_neighbors(model, eval_samples,
                                              topk=args.eval_topk, doc_ids=doc_ids)
                print(f"[Eval-{view_name}] samples={list(eval_samples)} "
                      f"top{args.eval_topk}:", flush=True)
                for q in eval_samples:
                    if q in nn_res:
                        pretty = ", ".join([f"(idx={i},Id={did},s={s:.3f})"
                                            for (i, s, did) in nn_res[q]])
                        print(f"  q(idx={q},Id={int(doc_ids[q])}) → {pretty}",
                              flush=True)

            # Per-epoch checkpoint
            if args.save_epoch_emb:
                E = (model.module.in_emb.weight if hasattr(model, "module")
                     else model.in_emb.weight).detach().cpu().numpy()
                Z = E.astype(np.float16 if args.emb_dtype == "float16"
                             else np.float32, copy=True)
                nrm = np.linalg.norm(Z, axis=1, keepdims=True)
                mask = (nrm[:, 0] > 0)
                Z[mask] = Z[mask] / nrm[mask]
                part_path = out_path.parent / f"Z_{view_name}_epoch{ep}.parquet"
                df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
                df.insert(0, "doc_idx", np.arange(N, dtype=np.int64))
                df.to_parquet(part_path, engine="fastparquet", index=False)
                print(f"[Checkpoint-{view_name}] saved {part_path.name}", flush=True)

        barrier(is_ddp)

    # Final export
    if (not is_ddp) or rank == 0:
        E = (model.module.in_emb.weight if hasattr(model, "module")
             else model.in_emb.weight).detach().cpu().numpy()
        Z = E.astype(np.float32, copy=True)
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        mask = (nrm[:, 0] > 0)
        Z[mask] = Z[mask] / nrm[mask]
        df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
        df.insert(0, "doc_idx", np.arange(N, dtype=np.int64))
        df.to_parquet(out_path, engine="fastparquet", index=False)
        print(f"[Train-{view_name}] saved {out_path.name}; "
              f"covered={int(mask.sum())}/{N} ({mask.mean():.1%})", flush=True)

    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    barrier(is_ddp)


def main():
    args = parse_args()
    is_ddp, rank, world, local, device = init_ddp("nccl")
    log0(is_ddp, rank,
         f"[DDP] enabled={is_ddp}, rank={rank}/{world}, device={device}")

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    TMP = Path(args.tmp_dir)
    assert TMP.exists(), f"tmp_dir not found: {TMP}"

    # Parse view selection
    views = [v.strip().lower() for v in args.views.split(",")]
    log0(is_ddp, rank, f"[Config] Training views: {views}")

    # Load data
    doc_df = pd.read_parquet(TMP / "doc_clean.parquet", engine="fastparquet")
    rw_params = pd.read_parquet(TMP / "rw_params.parquet", engine="fastparquet").iloc[0]
    N = len(doc_df)
    doc_ids = doc_df["Id"].to_numpy()

    # Load matrices for requested views
    DT_ppmi = None
    DW_bm25 = None

    if "tag" in views:
        tag_vocab = pd.read_parquet(TMP / "tag_vocab.parquet", engine="fastparquet")
        T = len(tag_vocab)
        DT_ppmi = load_csr_triplet_parquet(TMP / "DT_ppmi.parquet", shape=(N, T))

    if "text" in views:
        text_vocab = pd.read_parquet(TMP / "text_vocab.parquet", engine="fastparquet")
        W = len(text_vocab)
        DW_bm25 = load_csr_triplet_parquet(TMP / "DW_bm25.parquet", shape=(N, W))

    # Build corpora
    from scipy import sparse
    if DT_ppmi is None:
        DT_ppmi = sparse.csr_matrix((N, 1))  # Dummy
    if DW_bm25 is None:
        DW_bm25 = sparse.csr_matrix((N, 1))  # Dummy

    tag_corpus, text_corpus, start_tag, start_txt, degD_tag, degD_txt = build_corpus(
        doc_df, DT_ppmi, DW_bm25, rw_params, device, is_ddp, rank, world
    )

    # Train requested views
    if "tag" in views:
        log0(is_ddp, rank, "\n" + "=" * 60)
        log0(is_ddp, rank, "Training TAG view")
        log0(is_ddp, rank, "=" * 60)
        train_view("tag", N, start_tag, degD_tag, tag_corpus, device,
                   is_ddp, rank, args, TMP / "Z_tag.parquet", doc_ids)

    if "text" in views:
        log0(is_ddp, rank, "\n" + "=" * 60)
        log0(is_ddp, rank, "Training TEXT view")
        log0(is_ddp, rank, "=" * 60)
        train_view("text", N, start_txt, degD_txt, text_corpus, device,
                   is_ddp, rank, args, TMP / "Z_text.parquet", doc_ids)

    log0(is_ddp, rank, "\n[Done] Training complete.")
    cleanup_ddp()


if __name__ == "__main__":
    main()
