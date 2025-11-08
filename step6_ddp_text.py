#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6 (Text only) · WS-SGNS with DDP & speed-focused tweaks
- Text 视图仅训练
- DDP (torchrun) 多卡
- AMP + TF32
- GPU 别名负采样（O(1) 设备端采样，无 H2D）
- 大 batch（默认 196_608 ≈ 20万对/step）
- 成对子生成：forward-only + keep_prob + ctx_cap
- 每 epoch 最多对数阈值，到点提前结束
- 训练结束导出 Z_text.parquet（rank0）

Launch example:
  torchrun --nproc_per_node=2 step6_ddp_text.py \
    --tmp_dir ./tmp --epochs 4 --dim 256 --neg 10 --lr 0.025 \
    --amp true --tf32 true \
    --window_text 4 --keep_prob_text 0.35 --forward_only_text true --ctx_cap_text 4 \
    --batch_pairs_text 196608 --max_pairs_text 20000000 \
    --optimizer sgd --sparse false
"""

import os, gc, time, math, random, argparse
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from scipy import sparse
from pathlib import Path


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("WS-SGNS DDP (Text only)")

    # I/O & seed
    p.add_argument("--tmp_dir", type=str, default="./tmp")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=2025)

    # 模型/训练
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--neg", type=int, default=10)       # 建议 8–12；越小显存越省
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--ns_power", type=float, default=0.75)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","sparse_adam","adagrad"])
    p.add_argument("--sparse", type=lambda s: s.lower() in ["true","1","yes"], default=False)

    # 数值后端
    p.add_argument("--amp", type=lambda s: s.lower() in ["true","1","yes"], default=True)
    p.add_argument("--tf32", type=lambda s: s.lower() in ["true","1","yes"], default=True)

    # —— Text 视图的成对子策略（强烈影响 pairs/epoch）——
    p.add_argument("--window_text", type=int, default=4)         # 动态窗口上限（越小对数越少）
    p.add_argument("--keep_prob_text", type=float, default=0.35) # 对的保留率（0.3~0.5 常用）
    p.add_argument("--forward_only_text", type=lambda s: s.lower() in ["true","1","yes"], default=True)
    p.add_argument("--ctx_cap_text", type=int, default=4)        # 每中心最多上下文（0=不限）
    p.add_argument("--max_sents_text", type=int, default=None)   # 若也想限制句子数
    p.add_argument("--max_pairs_text", type=int, default=20_000_000)  # 每 epoch 最多对数，到点提前结束

    # 批量 & 日志
    p.add_argument("--batch_pairs_text", type=int, default=196_608)   # ~20万/step
    p.add_argument("--log_every", type=int, default=200)

    # 导出
    p.add_argument("--save_epoch_emb", type=lambda s: s.lower() in ["true","1","yes"], default=False) # 为极致速度默认 False
    p.add_argument("--emb_dtype", type=str, default="float32", choices=["float32","float16"])

    return p.parse_args()


# ---------------------------
# DDP init
# ---------------------------
def init_ddp(backend="nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return True, rank, world_size, local_rank, device
    return False, 0, 1, 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def barrier(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.barrier()


def log0(is_ddp: bool, rank: int, msg: str):
    if (not is_ddp) or rank == 0:
        print(msg, flush=True)


# ---------------------------
# CSR utils
# ---------------------------
def load_csr_triplet_parquet(path: Path, shape, dtype=np.float32) -> sparse.csr_matrix:
    df = pd.read_parquet(path, engine="fastparquet")
    coo = sparse.coo_matrix((df["val"].astype(dtype), (df["row"], df["col"])), shape=shape, dtype=dtype)
    return coo.tocsr()

def csr_rowview_torch(mat: sparse.csr_matrix, device: torch.device):
    mat = mat.tocsr()
    indptr = torch.from_numpy(mat.indptr.astype(np.int64)).to(device)
    indices= torch.from_numpy(mat.indices.astype(np.int64)).to(device)
    data   = torch.from_numpy(mat.data.astype(np.float32)).to(device)
    return indptr, indices, data

def csr_T(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    return mat.transpose().tocsr()


# ---------------------------
# Random walk corpus (Text only), sharded by rank
# ---------------------------
def build_text_corpus(doc_df: pd.DataFrame,
                      DW_bm25: sparse.csr_matrix,
                      rw_params: pd.Series,
                      device: torch.device,
                      is_ddp: bool, rank: int, world_size: int):

    N = len(doc_df)
    degD_txt = np.diff(DW_bm25.indptr)
    start_txt = np.where(degD_txt > 0)[0].astype(np.int64)

    RW_WALKS_PER_DOC    = int(rw_params["RW_WALKS_PER_DOC"])
    RW_L_DOCS_PER_SENT  = int(rw_params["RW_L_DOCS_PER_SENT"])
    RW_SEED_BASE        = int(rw_params["RW_SEED_BASE"])
    RW_AVOID_BACKTRACK  = bool(rw_params["RW_AVOID_BACKTRACK"])
    RW_RESTART_PROB     = float(rw_params["RW_RESTART_PROB"])
    RW_X_DEGREE_POW     = float(rw_params["RW_X_DEGREE_POW"])
    RW_X_NO_REPEAT_LAST = int(rw_params["RW_X_NO_REPEAT_LAST"])

    indptr_D, indices_D, data_D = csr_rowview_torch(DW_bm25, device)       # D->W
    indptr_W, indices_W, data_W = csr_rowview_torch(csr_T(DW_bm25), device) # W->D

    def row_neighbors(indptr, indices, data, r: torch.Tensor):
        a = indptr[r]; b = indptr[r+1]
        if (b-a).item() <= 0: return None, None
        sl = slice(a.item(), b.item())
        return indices[sl], data[sl]

    def sample_pos_by_weights(w: torch.Tensor, g: torch.Generator) -> int:
        if w is None or w.numel()==0: return -1
        w = torch.clamp(w, min=0)
        s = w.sum()
        if not torch.isfinite(s) or s.item()<=0: return -1
        cdf = torch.cumsum(w, dim=0)
        u = torch.rand((), generator=g, device=w.device) * cdf[-1]
        pos = torch.searchsorted(cdf, u, right=False).item()
        return min(pos, cdf.numel()-1)

    class TorchWalkCorpusText:
        def __init__(self, starts_np, split_shards=64):
            self.starts_np = starts_np
            self.split_shards = split_shards
            self._iters = 0

        def __len__(self):
            return int(len(self.starts_np) * RW_WALKS_PER_DOC)

        def __iter__(self):
            self._iters += 1
            rng = np.random.default_rng(RW_SEED_BASE + 31*self._iters + rank*1009)
            starts = self.starts_np.copy(); rng.shuffle(starts)
            shards = np.array_split(starts, max(1, self.split_shards))

            for shard_id, shard in enumerate(shards):
                if is_ddp and (shard_id % world_size) != rank:
                    continue
                g = torch.Generator(device=device)
                g.manual_seed(RW_SEED_BASE + 7919*(self._iters + shard_id + rank*101))

                w_factor = None
                if abs(RW_X_DEGREE_POW) > 1e-12:
                    w_deg = (indptr_W[1:] - indptr_W[:-1]).to(torch.float32)
                    w_factor = torch.clamp(w_deg, min=1.0).pow(RW_X_DEGREE_POW)

                for d0 in shard:
                    for _ in range(RW_WALKS_PER_DOC):
                        seq = [int(d0)]
                        prev_d=None; cur_d=int(d0); last_w=-1
                        for _step in range(RW_L_DOCS_PER_SENT-1):
                            r = torch.tensor(cur_d, dtype=torch.long, device=device)
                            w_cols, w_w = row_neighbors(indptr_D, indices_D, data_D, r)
                            if w_cols is None: break
                            ww = w_w.clone()
                            if w_factor is not None: ww = ww * w_factor[w_cols]
                            if RW_X_NO_REPEAT_LAST>0 and last_w>=0:
                                m = (w_cols==last_w)
                                if m.any(): ww[m]=0.0
                            pos_w = sample_pos_by_weights(ww, g)
                            if pos_w < 0: break
                            w = int(w_cols[pos_w].item())

                            wr = torch.tensor(w, dtype=torch.long, device=device)
                            d_rows, d_w = row_neighbors(indptr_W, indices_W, data_W, wr)
                            if d_rows is None: break
                            if RW_AVOID_BACKTRACK and prev_d is not None and d_rows.numel()>1:
                                m = (d_rows==prev_d)
                                if m.any(): d_w=d_w.clone(); d_w[m]=0.0
                            pos_d = sample_pos_by_weights(d_w, g)
                            if pos_d < 0: break
                            next_d = int(d_rows[pos_d].item())

                            if torch.rand((), generator=g, device=device).item() < RW_RESTART_PROB:
                                next_d = int(d0)

                            seq.append(next_d)
                            prev_d, cur_d, last_w = cur_d, next_d, w
                        if len(seq)>=2:
                            yield [str(s) for s in seq]

    return TorchWalkCorpusText(start_txt, split_shards=64), start_txt, degD_txt


# ---------------------------
# SGNS
# ---------------------------
class SGNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int, sparse: bool = False):
        super().__init__()
        self.in_emb  = nn.Embedding(vocab_size, dim, sparse=sparse)
        self.out_emb = nn.Embedding(vocab_size, dim, sparse=sparse)
        nn.init.uniform_(self.in_emb.weight,  -0.5/dim, 0.5/dim)
        nn.init.uniform_(self.out_emb.weight, -0.5/dim, 0.5/dim)

    def forward(self, center, pos, neg):
        v = self.in_emb(center)                # [B,d]
        u = self.out_emb(pos)                  # [B,d]
        pos_logit = torch.sum(v*u, dim=1)      # [B]
        neg_u = self.out_emb(neg)              # [B,K,d]
        neg_logit = torch.einsum("bd,bkd->bk", v, neg_u)
        pos_loss = torch.nn.functional.softplus(-pos_logit)
        neg_loss = torch.nn.functional.softplus(neg_logit).sum(dim=1)
        return (pos_loss + neg_loss).mean().unsqueeze(0)  # [1]


# ---------------------------
# Alias sampling (GPU)
# ---------------------------
def build_ns_dist_from_deg(deg: np.ndarray, power=0.75):
    p = np.power(np.maximum(deg, 1), power).astype(np.float64)
    p = p / p.sum()
    return p

def build_alias_on_device(probs_np: np.ndarray, device: torch.device):
    p = probs_np.astype(np.float64, copy=True)
    n = p.size
    p = p / p.sum()
    prob = np.zeros(n, dtype=np.float32)
    alias = np.zeros(n, dtype=np.int32)
    scaled = p * n
    small = [i for i,x in enumerate(scaled) if x < 1.0]
    large = [i for i,x in enumerate(scaled) if x >= 1.0]
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = (scaled[l] + scaled[s]) - 1.0
        if scaled[l] < 1.0: small.append(l)
        else:                large.append(l)
    for i in large + small:
        prob[i] = 1.0
        alias[i] = i
    prob_t  = torch.tensor(prob,  dtype=torch.float32, device=device)
    alias_t = torch.tensor(alias, dtype=torch.int32,   device=device)
    return prob_t, alias_t

def sample_alias_gpu(prob_t: torch.Tensor, alias_t: torch.Tensor, size: Tuple[int, ...], device: torch.device):
    n = prob_t.size(0)
    k = torch.randint(n, size, device=device)
    u = torch.rand(size, device=device)
    return torch.where(u < prob_t[k], k, alias_t[k].to(k.dtype))


# ---------------------------
# Pair iterator & batch maker（含 Text 下采样策略）
# ---------------------------
def iter_pairs_from_corpus_text(corpus,
                                window: int,
                                max_sents: Optional[int],
                                seed: int,
                                keep_prob: float = 1.0,
                                forward_only: bool = False,
                                ctx_cap: int = 0):
    rng = random.Random(seed)
    sent_count = 0
    for sent in corpus:
        if max_sents is not None and sent_count >= max_sents:
            break
        s = [int(x) for x in sent]
        L = len(s)
        for i in range(L):
            w = rng.randint(1, window)
            l = max(0, i - w); r = min(L - 1, i + w)
            if forward_only:
                cand = list(range(i+1, r+1))
            else:
                cand = list(range(l, r+1))
                if i in cand: cand.remove(i)
            if ctx_cap and len(cand) > ctx_cap:
                rng.shuffle(cand)
                cand = cand[:ctx_cap]
            if keep_prob < 1.0:
                thr = keep_prob
                cand = [j for j in cand if rng.random() < thr]
            for j in cand:
                yield s[i], s[j]
        sent_count += 1

def batch_pairs_and_negs_fast(pair_iter,
                              batch_size_pairs: int,
                              negK: int,
                              ns_prob_t: torch.Tensor,
                              ns_alias_t: torch.Tensor,
                              device: torch.device):
    centers, contexts = [], []
    for c, x in pair_iter:
        centers.append(c); contexts.append(x)
        if len(centers) >= batch_size_pairs:
            B = len(centers)
            negs_t    = sample_alias_gpu(ns_prob_t, ns_alias_t, size=(B, negK), device=device)
            centers_t = torch.tensor(centers, dtype=torch.long, device=device)
            contexts_t= torch.tensor(contexts, dtype=torch.long, device=device)
            yield centers_t, contexts_t, negs_t
            centers.clear(); contexts.clear()
    if centers:
        B = len(centers)
        negs_t    = sample_alias_gpu(ns_prob_t, ns_alias_t, size=(B, negK), device=device)
        centers_t = torch.tensor(centers, dtype=torch.long, device=device)
        contexts_t= torch.tensor(contexts, dtype=torch.long, device=device)
        yield centers_t, contexts_t, negs_t


# ---------------------------
# Train (Text)
# ---------------------------
def train_text(N: int,
               start_nodes: np.ndarray,
               degD_txt: np.ndarray,
               corpus,
               device: torch.device,
               is_ddp: bool, rank: int,
               args,
               out_path: Path):

    torch.manual_seed(args.seed + 23)

    # DDP + 稀疏梯度不稳：强制 dense
    sparse_runtime = bool(args.sparse)
    if is_ddp and sparse_runtime:
        log0(is_ddp, rank, "[Warn] DDP + sparse gradients may be unstable; switching to dense embeddings.")
        sparse_runtime = False

    model = SGNS(vocab_size=N, dim=args.dim, sparse=sparse_runtime).to(device)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type=="cuda" else None,
            output_device=device.index if device.type=="cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # optimizer（dense 用 SGD 最快；sparse 时可用 SparseAdam/Adagrad）
    if sparse_runtime and args.optimizer.lower()=="sparse_adam":
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)
    elif sparse_runtime and args.optimizer.lower()=="adagrad":
        optimizer = torch.optim.Adagrad(list(model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # AMP / TF32
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    # NS alias on device
    ns_dist = build_ns_dist_from_deg(degD_txt, power=args.ns_power)
    ns_prob_t, ns_alias_t = build_alias_on_device(ns_dist, device)

    # 打印配置
    log0(is_ddp, rank,
         f"[Train-text] epochs={args.epochs}, dim={args.dim}, neg={args.neg}, "
         f"batch_pairs={args.batch_pairs_text}, AMP={args.amp}, TF32={args.tf32}, "
         f"sparse={sparse_runtime}, opt={args.optimizer}, DDP={is_ddp}, "
         f"window={args.window_text}, keep_prob={args.keep_prob_text}, forward_only={args.forward_only_text}, ctx_cap={args.ctx_cap_text}, "
         f"max_pairs_epoch={args.max_pairs_text}")

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        pair_iter = iter_pairs_from_corpus_text(
            corpus=corpus,
            window=args.window_text,
            max_sents=args.max_sents_text,
            seed=args.seed + ep + 1000,
            keep_prob=args.keep_prob_text,
            forward_only=args.forward_only_text,
            ctx_cap=args.ctx_cap_text,
        )

        total_pairs = 0
        total_loss  = 0.0
        step = 0
        last_t = time.time()
        pairs_since_last = 0

        model.train()
        while True:
            # 生成一个批次；若 pair_iter 耗尽则跳出
            try:
                batch = next(batch_pairs_and_negs_fast(
                    pair_iter, args.batch_pairs_text, args.neg, ns_prob_t, ns_alias_t, device))
            except StopIteration:
                break

            centers_t, contexts_t, negs_t = batch
            optimizer.zero_grad(set_to_none=True)

            # autocast 兼容写法
            try:
                cm = torch.amp.autocast('cuda', enabled=args.amp)
            except TypeError:
                cm = torch.cuda.amp.autocast(enabled=args.amp)

            with cm:
                loss = model(centers_t, contexts_t, negs_t)
                if hasattr(loss, "dim") and loss.dim()!=0:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            b = centers_t.size(0)
            total_pairs += b
            total_loss  += float(loss.detach().item()) * b
            pairs_since_last += b
            step += 1

            if step % args.log_every == 0 and ((not is_ddp) or rank==0):
                now = time.time(); dt = max(1e-9, now-last_t)
                thr = pairs_since_last / dt
                mem = 0.0
                if device.type=="cuda":
                    try: mem = torch.cuda.memory_allocated(device=device)/(1024**2)
                    except Exception: pass
                print(f"[text] step={step:,} pairs/step={b} throughput={thr:,.0f} pairs/s "
                      f"loss={loss.item():.4f} mem~{mem:.0f}MB", flush=True)
                last_t = now; pairs_since_last = 0

            # 到达本 epoch 目标上限 → 提前结束
            if args.max_pairs_text is not None and total_pairs >= args.max_pairs_text:
                if (not is_ddp) or rank==0:
                    print(f"[text] early stop epoch {ep}: reached max_pairs_per_epoch={args.max_pairs_text:,}")
                break

        dt = time.time() - t0
        if (not is_ddp) or rank==0:
            if total_pairs == 0:
                print(f"[Train-text] epoch {ep}: no pairs produced")
            else:
                print(f"[Train-text] epoch {ep}: pairs={total_pairs:,}, "
                      f"avg_loss={total_loss/total_pairs:.4f}, time={dt:.1f}s", flush=True)

            # 可选：每 epoch 保存一次（默认关闭以追求速度）
            if args.save_epoch_emb:
                E = (model.module.in_emb.weight if hasattr(model, "module") else model.in_emb.weight).detach().cpu().numpy()
                Z = E.astype(np.float16 if args.emb_dtype=="float16" else np.float32, copy=True)
                nrm = np.linalg.norm(Z, axis=1, keepdims=True); mask = (nrm[:,0] > 0)
                Z[mask] = Z[mask] / nrm[mask]
                part_path = out_path.parent / f"Z_text_epoch{ep}.parquet"
                part_df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
                part_df.insert(0, "doc_idx", np.arange(N, dtype=np.int64))
                part_df.to_parquet(part_path, engine="fastparquet", index=False)
                print(f"[Checkpoint-text] saved {part_path.name} ({args.emb_dtype})", flush=True)

        barrier(is_ddp)

    # Final export（rank0）
    if (not is_ddp) or rank==0:
        E = (model.module.in_emb.weight if hasattr(model, "module") else model.in_emb.weight).detach().cpu().numpy()
        Z = E.astype(np.float32, copy=True)
        nrm = np.linalg.norm(Z, axis=1, keepdims=True); mask = (nrm[:,0] > 0)
        Z[mask] = Z[mask] / nrm[mask]
        emb_df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
        emb_df.insert(0, "doc_idx", np.arange(N, dtype=np.int64))
        emb_df.to_parquet(out_path, engine="fastparquet", index=False)
        print(f"[Train-text] saved {out_path.name}; covered_docs={int(mask.sum())}/{N} ({mask.mean():.1%})", flush=True)

    del model, optimizer; gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()

    barrier(is_ddp)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    is_ddp, rank, world_size, local_rank, device = init_ddp("nccl")
    log0(is_ddp, rank, f"[DDP] enabled={is_ddp}, rank={rank}/{world_size}, local_rank={local_rank}, device={device}")

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    # 读取中间件
    TMP_DIR = Path(args.tmp_dir)
    assert TMP_DIR.exists(), f"tmp_dir not found: {TMP_DIR}"

    doc_df     = pd.read_parquet(TMP_DIR / "doc_clean.parquet", engine="fastparquet")
    text_vocab = pd.read_parquet(TMP_DIR / "text_vocab.parquet", engine="fastparquet")
    rw_params  = pd.read_parquet(TMP_DIR / "rw_params.parquet", engine="fastparquet").iloc[0]

    N = len(doc_df); W = len(text_vocab)
    DW_bm25 = load_csr_triplet_parquet(TMP_DIR / "DW_bm25.parquet", shape=(N, W))

    # 构建 Text 语料（按 rank 分片）
    text_corpus, start_txt, degD_txt = build_text_corpus(
        doc_df, DW_bm25, rw_params, device, is_ddp, rank, world_size
    )

    # 训练（Text）
    train_text(N, start_txt, degD_txt, text_corpus, device, is_ddp, rank, args, TMP_DIR / "Z_text.parquet")

    log0(is_ddp, rank, "[Step 6 · Text] Done. Saved: Z_text.parquet")

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()
