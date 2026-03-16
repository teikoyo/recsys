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

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist

from src.config import TrainConfig
from src.constants import TAG_VIEW_SEED_OFFSET, TEXT_VIEW_SEED_OFFSET
from src.log import get_logger, log_rank0
from src.ddp_utils import init_ddp, barrier, cleanup_ddp
from src.csr_utils import load_csr_triplet_parquet
from src.sampling_utils import build_ns_dist_from_deg, build_alias_on_device
from src.pair_batch_utils import iter_pairs_from_corpus, batch_pairs_and_negs_fast
from src.random_walk import build_corpus
from src.sgns_model import SGNS


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
               is_ddp: bool, rank: int, cfg: TrainConfig, out_path: Path,
               doc_ids: np.ndarray):
    """Train embeddings for a single view."""
    logger = get_logger("train_sgns")

    seed_offset = TAG_VIEW_SEED_OFFSET if view_name == "tag" else TEXT_VIEW_SEED_OFFSET
    torch.manual_seed(cfg.seed + seed_offset)

    # DDP + sparse is unstable, force dense
    sparse_rt = bool(cfg.sparse)
    if is_ddp and sparse_rt:
        log_rank0(logger, is_ddp, rank,
                  f"[Warn] DDP + sparse unstable, switching to dense.")
        sparse_rt = False

    model = SGNS(vocab_size=N, dim=cfg.dim, sparse=sparse_rt).to(device)

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # Setup optimizer
    if sparse_rt and cfg.optimizer.lower() == "sparse_adam":
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=cfg.lr)
    elif sparse_rt and cfg.optimizer.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(list(model.parameters()), lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    # AMP scaler
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Build negative sampling alias table
    ns_dist = build_ns_dist_from_deg(degD, power=cfg.ns_power)
    ns_prob_t, ns_alias_t = build_alias_on_device(ns_dist, device)

    # Get per-view parameters
    vp = cfg.view_params(view_name)
    v_window = vp.window
    v_keep = vp.keep_prob
    v_forward = vp.forward_only
    v_cap = vp.ctx_cap
    v_batch = vp.batch_pairs
    v_max_pairs = vp.max_pairs
    v_max_sents = vp.max_sents

    eval_samples = pick_eval_samples(start_nodes, cfg.eval_samples_per_view, cfg.seed + 7)

    log_rank0(logger, is_ddp, rank,
              f"[Train-{view_name}] epochs={cfg.epochs}, dim={cfg.dim}, "
              f"window={v_window}, neg={cfg.neg}, batch={v_batch}, "
              f"accum={cfg.accum}, AMP={cfg.amp}, DDP={is_ddp}")

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)

    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Create corpus iterator
        corpus_iter = corpus.iterate(device, is_ddp, rank,
                                      dist.get_world_size() if is_ddp else 1)

        pair_iter = iter_pairs_from_corpus(
            corpus=corpus_iter,
            window=v_window,
            max_sents=v_max_sents,
            seed=cfg.seed + ep + (0 if view_name == "tag" else 1000),
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
                pair_iter, v_batch, cfg.neg, ns_prob_t, ns_alias_t, device):

            optimizer.zero_grad(set_to_none=True)

            B = centers_t.size(0)
            accum = max(1, int(cfg.accum))
            micro = (B + accum - 1) // accum

            # Micro-batch loop for gradient accumulation
            for s in range(0, B, micro):
                c_mb = centers_t[s:s + micro]
                x_mb = contexts_t[s:s + micro]
                n_mb = negs_t[s:s + micro]

                try:
                    cm = torch.amp.autocast('cuda', enabled=cfg.amp)
                except TypeError:
                    cm = torch.cuda.amp.autocast(enabled=cfg.amp)

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

            if step % cfg.log_every == 0 and (not is_ddp or rank == 0):
                now = time.time()
                dt = max(1e-9, now - last_t)
                thr = pairs_since_last / dt
                mem = 0.0
                if device.type == "cuda":
                    try:
                        mem = torch.cuda.memory_allocated(device=device) / (1024 ** 2)
                    except RuntimeError:
                        pass
                logger.info(
                    f"[{view_name}] step={step:,} throughput={thr:,.0f} pairs/s "
                    f"loss~{total_loss / max(1, total_pairs):.4f} mem~{mem:.0f}MB")
                last_t = now
                pairs_since_last = 0

            if v_max_pairs is not None and total_pairs >= v_max_pairs:
                if (not is_ddp) or rank == 0:
                    logger.info(
                        f"[{view_name}] early stop epoch {ep}: "
                        f"reached max_pairs={v_max_pairs:,}")
                break

        dt = time.time() - t0
        if (not is_ddp) or rank == 0:
            if total_pairs == 0:
                logger.info(f"[Train-{view_name}] epoch {ep}: no pairs produced")
            else:
                logger.info(
                    f"[Train-{view_name}] epoch {ep}: pairs={total_pairs:,}, "
                    f"avg_loss={total_loss / max(1, total_pairs):.4f}, "
                    f"time={dt:.1f}s")

            # Quick evaluation
            if len(eval_samples) > 0:
                nn_res = quick_eval_neighbors(model, eval_samples,
                                              topk=cfg.eval_topk, doc_ids=doc_ids)
                logger.info(
                    f"[Eval-{view_name}] samples={list(eval_samples)} "
                    f"top{cfg.eval_topk}:")
                for q in eval_samples:
                    if q in nn_res:
                        pretty = ", ".join([f"(idx={i},Id={did},s={s:.3f})"
                                            for (i, s, did) in nn_res[q]])
                        logger.info(
                            f"  q(idx={q},Id={int(doc_ids[q])}) -> {pretty}")

            # Per-epoch checkpoint
            if cfg.save_epoch_emb:
                E = (model.module.in_emb.weight if hasattr(model, "module")
                     else model.in_emb.weight).detach().cpu().numpy()
                Z = E.astype(np.float16 if cfg.emb_dtype == "float16"
                             else np.float32, copy=True)
                nrm = np.linalg.norm(Z, axis=1, keepdims=True)
                mask = (nrm[:, 0] > 0)
                Z[mask] = Z[mask] / nrm[mask]
                part_path = out_path.parent / f"Z_{view_name}_epoch{ep}.parquet"
                df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
                df.insert(0, "doc_idx", np.arange(N, dtype=np.int64))
                df.to_parquet(part_path, engine="fastparquet", index=False)
                logger.info(f"[Checkpoint-{view_name}] saved {part_path.name}")

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
        logger.info(
            f"[Train-{view_name}] saved {out_path.name}; "
            f"covered={int(mask.sum())}/{N} ({mask.mean():.1%})")

    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    barrier(is_ddp)


def main():
    cfg = TrainConfig.from_args()
    logger = get_logger("train_sgns")
    logger.setLevel(cfg.log_level)

    is_ddp, rank, world, local, device = init_ddp("nccl")
    log_rank0(logger, is_ddp, rank,
              f"[DDP] enabled={is_ddp}, rank={rank}/{world}, device={device}")

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)

    TMP = Path(cfg.tmp_dir)
    assert TMP.exists(), f"tmp_dir not found: {TMP}"

    # Parse view selection
    views = cfg.views
    log_rank0(logger, is_ddp, rank, f"[Config] Training views: {views}")

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
        log_rank0(logger, is_ddp, rank, "\n" + "=" * 60)
        log_rank0(logger, is_ddp, rank, "Training TAG view")
        log_rank0(logger, is_ddp, rank, "=" * 60)
        train_view("tag", N, start_tag, degD_tag, tag_corpus, device,
                   is_ddp, rank, cfg, TMP / "Z_tag.parquet", doc_ids)

    if "text" in views:
        log_rank0(logger, is_ddp, rank, "\n" + "=" * 60)
        log_rank0(logger, is_ddp, rank, "Training TEXT view")
        log_rank0(logger, is_ddp, rank, "=" * 60)
        train_view("text", N, start_txt, degD_txt, text_corpus, device,
                   is_ddp, rank, cfg, TMP / "Z_text.parquet", doc_ids)

    log_rank0(logger, is_ddp, rank, "\n[Done] Training complete.")
    cleanup_ddp()


if __name__ == "__main__":
    main()
