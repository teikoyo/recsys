#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========== Imports (all at top) ==========
import os
import re
import math
import time
import json
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ========== Global Config ==========
TMP_DIR = Path("./tmp")
PARQUET_ENGINE = "fastparquet"

# 训练参数（高吞吐 & 显存友好）
SGNS_DIM        = 256
SGNS_WINDOW     = 4
SGNS_NEG        = 6            # 关键加速：从12降到6
LR              = 2e-3
WEIGHT_DECAY    = 0.0
MAX_GRAD_NORM   = 2.0
BATCH_PAIRS     = 131072       # 大批次，accum=1
EPOCHS          = 2

# 语料规模（按“总量”；DDP 时各 rank 均分）
TEXT_SENT_PER_EPOCH_TOTAL = 500_000

# AMP（Ada 支持 bf16，稳定且快）
USE_BF16 = True

# 随机游走（训练期覆写以提速）
# 若 Step 5 中 L=40，这里训练阶段下调到 30（进一步加速）
RW_L_DOCS_PER_SENT_TRAIN = 30

SEED = 2025

# ========== Utils ==========
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_parquet(path: Path):
    return pd.read_parquet(path, engine=PARQUET_ENGINE)

def load_csr_from_triplets(path: Path, shape: Tuple[int,int], dtype=np.float32) -> sparse.csr_matrix:
    df = load_parquet(path)
    coo = sparse.coo_matrix((df["val"].astype(dtype), (df["row"], df["col"])), shape=shape, dtype=dtype)
    return coo.tocsr()

def save_embeddings_parquet(mat: np.ndarray, path: Path):
    df = pd.DataFrame(mat, columns=[f"f{i}" for i in range(mat.shape[1])])
    df.insert(0, "doc_idx", np.arange(mat.shape[0], dtype=np.int64))
    df.to_parquet(path, engine=PARQUET_ENGINE, index=False)

def l2_normalize_rows(x: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

# ========== DDP Init ==========
def ddp_is_active() -> bool:
    keys = ("RANK", "WORLD_SIZE")
    return all(k in os.environ for k in keys)

def ddp_setup(backend: str = "nccl") -> Tuple[int,int,int,torch.device]:
    if ddp_is_active():
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world, local_rank, device

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ========== CSR RowView to Torch ==========
def csr_to_rowview_torch(mat: sparse.csr_matrix, device: torch.device):
    mat = mat.tocsr()
    indptr  = torch.from_numpy(mat.indptr.astype(np.int64)).to(device)
    indices = torch.from_numpy(mat.indices.astype(np.int64)).to(device)
    data    = torch.from_numpy(mat.data.astype(np.float32)).to(device)
    return indptr, indices, data

# ========== Random Walk (Text view only) ==========
@torch.no_grad()
def walk_sentences_text(
    starts: np.ndarray,
    DX_row: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],   # D->W (indptr, indices, data)
    XD_row: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],   # W->D
    sentences_target: int,
    L_docs: int,
    seed_base: int,
    rank: int,
    world: int,
    device: torch.device,
    restart_prob: float = 0.15,
    avoid_backtrack: bool = True,
    x_no_repeat_last: int = 1
):
    """
    仅 Text 视图：在线生成 D-only 序列（list[str]），分配给各rank。
    """
    indptr_D, indices_D, data_D = DX_row
    indptr_X, indices_X, data_X = XD_row

    # 本 rank 负责的子集（简单均匀切）
    starts_rank = starts[rank::world]
    g = torch.Generator(device=device)
    g.manual_seed(seed_base + 10007 * (rank + 1))

    produced = 0
    for d0 in starts_rank:
        d0 = int(d0)
        for _ in range(10):  # 每起点默认生成 ~10条句子（与 Step 5 一致）
            seq = [d0]
            prev_d, cur_d, last_x = None, d0, -1
            for _step in range(L_docs - 1):
                a, b = indptr_D[cur_d], indptr_D[cur_d + 1]
                if (b - a).item() <= 0:
                    break
                x_cols = indices_D[a:b]
                x_w    = data_D[a:b].clone()

                # 不重复上一个 X
                if x_no_repeat_last > 0 and last_x >= 0:
                    mask = (x_cols == last_x)
                    if mask.any(): x_w[mask] = 0.0

                # 采样 X
                x_w.clamp_(min=0)
                cdf = torch.cumsum(x_w, dim=0)
                tot = cdf[-1]
                if not torch.isfinite(tot) or tot.item() <= 0:
                    break
                u = torch.rand((), generator=g, device=device) * tot
                pos = torch.searchsorted(cdf, u, right=False).item()
                x = int(x_cols[pos].item())

                # X -> D
                a2, b2 = indptr_X[x], indptr_X[x + 1]
                if (b2 - a2).item() <= 0:
                    break
                d_rows = indices_X[a2:b2]
                d_w    = data_X[a2:b2].clone()

                if avoid_backtrack and prev_d is not None and d_rows.numel() > 1:
                    d_w[d_rows == prev_d] = 0.0

                d_w.clamp_(min=0)
                cdf2 = torch.cumsum(d_w, dim=0)
                tot2 = cdf2[-1]
                if not torch.isfinite(tot2) or tot2.item() <= 0:
                    break
                u2 = torch.rand((), generator=g, device=device) * tot2
                pos2 = torch.searchsorted(cdf2, u2, right=False).item()
                next_d = int(d_rows[pos2].item())

                # 重启
                if torch.rand((), generator=g, device=device).item() < restart_prob:
                    next_d = d0

                seq.append(next_d)
                prev_d, cur_d, last_x = cur_d, next_d, x

            if len(seq) >= 2:
                yield [str(s) for s in seq]
                produced += 1
                if produced >= sentences_target:
                    return

# 将句子转为 (center, context) 正样本对（动态窗口）
_rng_py = random.Random(SEED)
def sentence_to_pairs(sent_tokens: List[str], window: int):
    ids = [int(x) for x in sent_tokens]
    L = len(ids)
    for i in range(L):
        w = _rng_py.randint(1, window)
        left  = max(0, i - w)
        right = min(L - 1, i + w)
        ci = ids[i]
        for j in range(left, right + 1):
            if j == i: continue
            yield ci, ids[j]

def batch_pairs_from_corpus(corpus_iter, batch_pairs: int):
    buf_c, buf_o = [], []
    for sent in corpus_iter:
        for ci, cj in sentence_to_pairs(sent, SGNS_WINDOW):
            buf_c.append(ci); buf_o.append(cj)
            if len(buf_c) >= batch_pairs:
                yield np.asarray(buf_c, dtype=np.int64), np.asarray(buf_o, dtype=np.int64)
                buf_c.clear(); buf_o.clear()
    if buf_c:
        yield np.asarray(buf_c, dtype=np.int64), np.asarray(buf_o, dtype=np.int64)

# ========== SGNS Model ==========
class SGNSModel(nn.Module):
    def __init__(self, num_tokens: int, dim: int, neg_probs: torch.Tensor):
        super().__init__()
        self.in_emb  = nn.Embedding(num_tokens, dim)
        self.out_emb = nn.Embedding(num_tokens, dim)
        nn.init.normal_(self.in_emb.weight,  mean=0.0, std=0.02)
        nn.init.normal_(self.out_emb.weight, mean=0.0, std=0.02)
        self.register_buffer("neg_probs", neg_probs)  # 按 rank 各自持有

    def forward(self, center_idx: torch.Tensor, context_idx: torch.Tensor, neg_k: int):
        B = center_idx.size(0)
        neg = torch.multinomial(self.neg_probs, num_samples=B * neg_k, replacement=True)\
                .view(B, neg_k).to(center_idx.device, non_blocking=True)

        u = self.in_emb(center_idx)        # (B, D)
        v = self.out_emb(context_idx)      # (B, D)
        pos_score = (u * v).sum(dim=-1)    # (B,)

        v_neg = self.out_emb(neg)          # (B, K, D)
        neg_score = torch.bmm(v_neg, u.unsqueeze(2)).squeeze(2)  # (B,K)

        loss = -(F.logsigmoid(pos_score).sum() + F.logsigmoid(-neg_score).sum()) / B
        return loss

# ========== Main Training ==========
def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    rank, world, local_rank, device = ddp_setup()
    if rank == 0:
        print(f"[Env] world={world}, device={device}, TMP_DIR={TMP_DIR.resolve()}")

    # 读取必要数据
    doc_df     = load_parquet(TMP_DIR / "doc_clean.parquet")
    text_vocab = load_parquet(TMP_DIR / "text_vocab.parquet")
    N, W = len(doc_df), len(text_vocab)

    DW_bm25 = load_csr_from_triplets(TMP_DIR / "DW_bm25.parquet", shape=(N, W))
    XD_bm25 = DW_bm25.transpose().tocsr()

    # 起点（有邻接的 doc）
    start_txt = np.where(np.diff(DW_bm25.indptr) > 0)[0].astype(np.int64)

    # RowView 放到各 rank 的设备
    DX_row = csr_to_rowview_torch(DW_bm25, device)
    XD_row = csr_to_rowview_torch(XD_bm25, device)

    # 负采样分布（度^0.75）
    deg_txt = np.diff(DW_bm25.indptr).astype(np.float64)
    p = np.power(np.maximum(deg_txt, 1.0), 0.75)
    p = p / (p.sum() if p.sum() > 0 else 1.0)
    neg_probs = torch.from_numpy(p.astype(np.float32)).to(device)

    # 模型
    model = SGNSModel(num_tokens=N, dim=SGNS_DIM, neg_probs=neg_probs).to(device)
    if ddp_is_active():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, broadcast_buffers=False
        )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 每 rank 的句子预算
    sent_per_rank = math.ceil(TEXT_SENT_PER_EPOCH_TOTAL / world)

    for ep in range(1, EPOCHS + 1):
        if rank == 0:
            print(f"[Train-text] epoch={ep}/{EPOCHS} | dim={SGNS_DIM}, window={SGNS_WINDOW}, neg={SGNS_NEG}, "
                  f"batch_pairs={BATCH_PAIRS}, bf16={USE_BF16}, sent/epoch(total)={TEXT_SENT_PER_EPOCH_TOTAL}")

        # 在线语料（仅本 rank 的份额）
        corpus_iter = walk_sentences_text(
            starts=start_txt,
            DX_row=DX_row, XD_row=XD_row,
            sentences_target=sent_per_rank,
            L_docs=RW_L_DOCS_PER_SENT_TRAIN,
            seed_base=SEED + 1000 * ep,
            rank=rank, world=world, device=device,
            restart_prob=0.15, avoid_backtrack=True, x_no_repeat_last=1
        )

        # 训练
        model.train()
        seen_pairs = 0
        steps = 0
        t0 = time.time()

        for bc, bo in batch_pairs_from_corpus(corpus_iter, BATCH_PAIRS):
            centers = torch.from_numpy(bc).to(device, non_blocking=True)
            ctxts   = torch.from_numpy(bo).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if USE_BF16 and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(centers, ctxts, SGNS_NEG)
            else:
                loss = model(centers, ctxts, SGNS_NEG)

            loss.backward()
            if MAX_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()

            seen_pairs += int(len(bc))
            steps += 1
            if steps % 20 == 0 and rank == 0:
                dt = time.time() - t0
                thr = seen_pairs / max(dt, 1e-6)
                print(f"[Throughput-text] {thr:,.0f} pairs/s over {steps} steps | loss={float(loss):.4f}")

        # 同步（DDP）
        if ddp_is_active():
            dist.barrier()

        if rank == 0:
            dt = time.time() - t0
            print(f"[Epoch-text] {ep}/{EPOCHS} done: pairs≈{seen_pairs:,} in {dt:.1f}s "
                  f"(per-rank; total≈{seen_pairs*world:,})")

    # 仅 rank 0 保存
    if rank == 0:
        emb = model.module.in_emb.weight.detach().cpu().numpy() if isinstance(model, nn.parallel.DistributedDataParallel) \
              else model.in_emb.weight.detach().cpu().numpy()
        emb = l2_normalize_rows(emb.astype(np.float32))
        out_path = TMP_DIR / "Z_text.parquet"
        save_embeddings_parquet(emb, out_path)
        print(f"[Save] Z_text → {out_path.as_posix()}")

    ddp_cleanup()

if __name__ == "__main__":
    main()
