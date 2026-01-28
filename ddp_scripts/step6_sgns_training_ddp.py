#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6 · WS-SGNS (DDP) — 高吞吐实现（约20万对/步）
- DDP（torchrun）
- AMP + TF32
- GPU 别名负采样（O(1) 设备端采样，无 numpy.choice/H→D）
- 视图分参：window / keep_prob / forward_only / ctx_cap / batch_pairs / max_pairs_per_epoch
- 梯度累积（--accum）适配超大 batch
- 每 epoch 轻量评估与 checkpoint，最终导出 Z_tag / Z_text

启动示例：
  torchrun --nproc_per_node=2 step6_ddp.py \
    --tmp_dir ./tmp --epochs 4 --dim 256 --neg 10 --amp true --tf32 true \
    --optimizer sparse_adam --sparse false \
    --window_tag 5 --keep_prob_tag 1.0 --forward_only_tag false --ctx_cap_tag 0 --batch_pairs_tag 204800 --max_pairs_tag 20000000 \
    --window_text 4 --keep_prob_text 0.35 --forward_only_text true --ctx_cap_text 4 --batch_pairs_text 204800 --max_pairs_text 20000000 \
    --accum 1
"""

import os, gc, time, math, random, argparse
from typing import Tuple, List, Optional
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
    p = argparse.ArgumentParser("WS-SGNS DDP Trainer (200k pairs/step)")

    p.add_argument("--tmp_dir", type=str, default="./tmp")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--neg", type=int, default=10)        # 先降到10，换更大batch
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--ns_power", type=float, default=0.75)
    p.add_argument("--optimizer", type=str, default="sparse_adam", choices=["sparse_adam","adagrad","sgd"])
    p.add_argument("--sparse", type=lambda s: s.lower() in ["true","1","yes"], default=False)
    p.add_argument("--amp", type=lambda s: s.lower() in ["true","1","yes"], default=True)
    p.add_argument("--tf32", type=lambda s: s.lower() in ["true","1","yes"], default=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--accum", type=int, default=1)       # 梯度累积步数

    # —— 视图分参 —— #
    p.add_argument("--window_tag", type=int, default=5)
    p.add_argument("--window_text", type=int, default=4)

    p.add_argument("--keep_prob_tag", type=float, default=1.0)
    p.add_argument("--keep_prob_text", type=float, default=0.35)

    p.add_argument("--forward_only_tag", type=lambda s: s.lower() in ["true","1","yes"], default=False)
    p.add_argument("--forward_only_text", type=lambda s: s.lower() in ["true","1","yes"], default=True)

    p.add_argument("--ctx_cap_tag", type=int, default=0)     # 0=不限
    p.add_argument("--ctx_cap_text", type=int, default=4)

    p.add_argument("--batch_pairs_tag", type=int, default=204800)   # ≈20万/步
    p.add_argument("--batch_pairs_text", type=int, default=204800)  # ≈20万/步

    p.add_argument("--max_pairs_tag", type=int, default=20_000_000)   # 每epoch对数上限
    p.add_argument("--max_pairs_text", type=int, default=20_000_000)

    # 监控/评估/存盘
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--eval_samples_per_view", type=int, default=3)
    p.add_argument("--eval_topk", type=int, default=5)
    p.add_argument("--eval_chunk", type=int, default=200_000)
    p.add_argument("--save_epoch_emb", type=lambda s: s.lower() in ["true","1","yes"], default=True)
    p.add_argument("--emb_dtype", type=str, default="float32", choices=["float32","float16"])
    p.add_argument("--max_sents_tag", type=int, default=None)
    p.add_argument("--max_sents_text", type=int, default=None)

    return p.parse_args()

# ---------------------------
# DDP init
# ---------------------------
def init_ddp(backend="nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank(); world = dist.get_world_size()
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
def load_csr_triplet_parquet(path: Path, shape, dtype=np.float32) -> sparse.csr_matrix:
    df = pd.read_parquet(path, engine="fastparquet")
    coo = sparse.coo_matrix((df["val"].astype(dtype), (df["row"], df["col"])), shape=shape, dtype=dtype)
    return coo.tocsr()

def csr_rowview_torch(mat: sparse.csr_matrix, device: torch.device):
    mat = mat.tocsr()
    return (
        torch.from_numpy(mat.indptr.astype(np.int64)).to(device),
        torch.from_numpy(mat.indices.astype(np.int64)).to(device),
        torch.from_numpy(mat.data.astype(np.float32)).to(device),
    )

def csr_T(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    return mat.transpose().tocsr()

# ---------------------------
# Random walk corpus with DDP shard
# ---------------------------
def build_corpus(doc_df: pd.DataFrame,
                 DT_ppmi: sparse.csr_matrix,
                 DW_bm25: sparse.csr_matrix,
                 rw_params: pd.Series,
                 device: torch.device,
                 is_ddp: bool, rank: int, world: int):
    N = len(doc_df)
    degD_tag = np.diff(DT_ppmi.indptr); start_tag = np.where(degD_tag > 0)[0].astype(np.int64)
    degD_txt = np.diff(DW_bm25.indptr); start_txt = np.where(degD_txt > 0)[0].astype(np.int64)

    RW_WALKS_PER_DOC    = int(rw_params["RW_WALKS_PER_DOC"])
    RW_L_DOCS_PER_SENT  = int(rw_params["RW_L_DOCS_PER_SENT"])
    RW_SEED_BASE        = int(rw_params["RW_SEED_BASE"])
    RW_AVOID_BACKTRACK  = bool(rw_params["RW_AVOID_BACKTRACK"])
    RW_RESTART_PROB     = float(rw_params["RW_RESTART_PROB"])
    RW_X_DEGREE_POW     = float(rw_params["RW_X_DEGREE_POW"])
    RW_X_NO_REPEAT_LAST = int(rw_params["RW_X_NO_REPEAT_LAST"])

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

    class TorchWalkCorpus:
        def __init__(self, starts_np, DX, XD, base_seed, split_shards=64, view_name=""):
            self.starts_np = starts_np
            self.DX = DX; self.XD = XD
            self.base_seed = base_seed; self.split_shards = split_shards
            self._iters = 0; self.view_name = view_name

        def __len__(self): return int(len(self.starts_np) * RW_WALKS_PER_DOC)

        def __iter__(self):
            self._iters += 1
            rng = np.random.default_rng(RW_SEED_BASE + 31*self._iters + rank*1009)
            starts = self.starts_np.copy(); rng.shuffle(starts)
            shards = np.array_split(starts, max(1, self.split_shards))

            for sid, shard in enumerate(shards):
                if is_ddp and (sid % world) != rank:
                    continue
                indptr_D, indices_D, data_D = self.DX
                indptr_X, indices_X, data_X = self.XD
                g = torch.Generator(device=device)
                g.manual_seed(self.base_seed + 7919*(self._iters + sid + rank*101))

                x_factor = None
                if abs(RW_X_DEGREE_POW) > 1e-12:
                    x_deg = (indptr_X[1:]-indptr_X[:-1]).to(torch.float32)
                    x_factor = torch.clamp(x_deg, min=1.0).pow(RW_X_DEGREE_POW)

                for d0 in shard:
                    for _ in range(RW_WALKS_PER_DOC):
                        seq = [int(d0)]
                        prev_d=None; cur_d=int(d0); last_x=-1
                        for _step in range(RW_L_DOCS_PER_SENT-1):
                            r = torch.tensor(cur_d, dtype=torch.long, device=device)
                            x_cols, x_w = row_neighbors(indptr_D, indices_D, data_D, r)
                            if x_cols is None: break
                            w = x_w.clone()
                            if x_factor is not None: w = w * x_factor[x_cols]
                            if RW_X_NO_REPEAT_LAST>0 and last_x>=0:
                                m = (x_cols==last_x)
                                if m.any(): w[m]=0.0
                            px = sample_pos_by_weights(w, g)
                            if px < 0: break
                            x = int(x_cols[px].item())

                            xr = torch.tensor(x, dtype=torch.long, device=device)
                            d_rows, d_w = row_neighbors(indptr_X, indices_X, data_X, xr)
                            if d_rows is None: break
                            if RW_AVOID_BACKTRACK and prev_d is not None and d_rows.numel()>1:
                                m = (d_rows==prev_d)
                                if m.any(): d_w=d_w.clone(); d_w[m]=0.0
                            pdx = sample_pos_by_weights(d_w, g)
                            if pdx < 0: break
                            next_d = int(d_rows[pdx].item())

                            if torch.rand((), generator=g, device=device).item() < RW_RESTART_PROB:
                                next_d = int(d0)

                            seq.append(next_d)
                            prev_d, cur_d, last_x = cur_d, next_d, x
                        if len(seq)>=2:
                            yield [str(s) for s in seq]

    DX_tag = csr_rowview_torch(DT_ppmi, device)
    XD_tag = csr_rowview_torch(csr_T(DT_ppmi), device)
    DX_txt = csr_rowview_torch(DW_bm25, device)
    XD_txt = csr_rowview_torch(csr_T(DW_bm25), device)

    tag_corpus  = TorchWalkCorpus(start_tag, DX_tag, XD_tag, base_seed=int(rw_params["RW_SEED_BASE"])+11, split_shards=64, view_name="tag")
    text_corpus = TorchWalkCorpus(start_txt, DX_txt, XD_txt, base_seed=int(rw_params["RW_SEED_BASE"])+23, split_shards=64, view_name="text")

    return tag_corpus, text_corpus, start_tag, start_txt, degD_tag, degD_txt

# ---------------------------
# SGNS model
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
# Pair iterator & batch maker
# ---------------------------
def iter_pairs_from_corpus(corpus,
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
            # 候选上下文
            if forward_only:
                cand = list(range(i+1, r+1))
            else:
                cand = list(range(l, r+1))
                if i in cand:
                    cand.remove(i)
            # 上下文封顶
            if ctx_cap and len(cand) > ctx_cap:
                rng.shuffle(cand)
                cand = cand[:ctx_cap]
            # 下采样
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
# Lightweight eval
# ---------------------------
def pick_eval_samples(starts_np: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 0 or len(starts_np) == 0: return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    k = min(k, len(starts_np))
    return rng.choice(starts_np, size=k, replace=False).astype(np.int64)

def quick_eval_neighbors(model: nn.Module, sample_idx: np.ndarray, topk: int, chunk: int, doc_ids: np.ndarray):
    E = (model.module.in_emb.weight if hasattr(model, "module") else model.in_emb.weight).detach().cpu().numpy()
    E = E.astype(np.float32, copy=False)
    nrm = np.linalg.norm(E, axis=1, keepdims=True); mask = (nrm[:,0] > 0)
    E[mask] = E[mask] / nrm[mask]

    results = {}
    for d in sample_idx:
        v = E[d:d+1]
        scores = np.empty(E.shape[0], dtype=np.float32)
        off = 0
        while off < E.shape[0]:
            blk = E[off: off+chunk]
            scores[off: off+blk.shape[0]] = (blk @ v[0]).astype(np.float32)
            off += blk.shape[0]
        scores[d] = -1.0
        nn_idx = np.argpartition(scores, -topk)[-topk:]
        nn_idx = nn_idx[np.argsort(scores[nn_idx])][::-1]
        results[int(d)] = [(int(i), float(scores[i]), int(doc_ids[i])) for i in nn_idx]
    return results

# ---------------------------
# Train one view
# ---------------------------
def train_view(view_name: str,
               N: int,
               start_nodes: np.ndarray,
               degD: np.ndarray,
               corpus,
               device: torch.device,
               is_ddp: bool, rank: int,
               args,
               out_path: Path,
               doc_ids: np.ndarray):
    torch.manual_seed(args.seed + (11 if view_name=="tag" else 23))

    # DDP + 稀疏不稳定：强制dense
    sparse_rt = bool(args.sparse)
    if is_ddp and sparse_rt:
        log0(is_ddp, rank, "[Warn] DDP + sparse embeddings 可能不稳定，已切换为 dense。")
        sparse_rt = False

    model = SGNS(vocab_size=N, dim=args.dim, sparse=sparse_rt).to(device)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type=="cuda" else None,
            output_device=device.index if device.type=="cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # Optimizer
    if sparse_rt and args.optimizer.lower()=="sparse_adam":
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)
    elif sparse_rt and args.optimizer.lower()=="adagrad":
        optimizer = torch.optim.Adagrad(list(model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # GradScaler
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ns alias
    ns_dist = build_ns_dist_from_deg(degD, power=args.ns_power)
    ns_prob_t, ns_alias_t = build_alias_on_device(ns_dist, device)

    # per-view params
    if view_name == "tag":
        v_window, v_keep, v_forward, v_cap = args.window_tag, args.keep_prob_tag, args.forward_only_tag, args.ctx_cap_tag
        v_batch, v_max_pairs, v_max_sents = args.batch_pairs_tag, args.max_pairs_tag, args.max_sents_tag
    else:
        v_window, v_keep, v_forward, v_cap = args.window_text, args.keep_prob_text, args.forward_only_text, args.ctx_cap_text
        v_batch, v_max_pairs, v_max_sents = args.batch_pairs_text, args.max_pairs_text, args.max_sents_text

    eval_samples = pick_eval_samples(start_nodes, args.eval_samples_per_view, args.seed+7)

    log0(is_ddp, rank,
         f"[Train-{view_name}] epochs={args.epochs}, dim={args.dim}, window={v_window}, neg={args.neg}, "
         f"batch_pairs={v_batch}, accum={args.accum}, AMP={args.amp}, TF32={args.tf32}, "
         f"sparse={sparse_rt}, optimizer={args.optimizer}, DDP={is_ddp}")

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        pair_iter = iter_pairs_from_corpus(
            corpus=corpus,
            window=v_window,
            max_sents=v_max_sents,
            seed=args.seed + ep + (0 if view_name=="tag" else 1000),
            keep_prob=v_keep,
            forward_only=v_forward,
            ctx_cap=v_cap,
        )

        total_pairs = 0
        total_loss  = 0.0
        step = 0
        last_t = time.time()
        pairs_since_last = 0

        model.train()
        for centers_t, contexts_t, negs_t in batch_pairs_and_negs_fast(
                pair_iter, v_batch, args.neg, ns_prob_t, ns_alias_t, device):

            optimizer.zero_grad(set_to_none=True)

            B = centers_t.size(0)
            accum = max(1, int(args.accum))
            micro = (B + accum - 1) // accum  # 向上取整的 micro-batch 大小

            # micro-batch 循环，显存更稳
            for s in range(0, B, micro):
                c_mb = centers_t[s:s+micro]
                x_mb = contexts_t[s:s+micro]
                n_mb = negs_t[s:s+micro]

                try:
                    cm = torch.amp.autocast('cuda', enabled=args.amp)
                except TypeError:
                    cm = torch.cuda.amp.autocast(enabled=args.amp)

                with cm:
                    loss = model(c_mb, x_mb, n_mb)
                    if hasattr(loss, "dim") and loss.dim()!=0:
                        loss = loss.mean()
                    # 梯度累积等价缩放
                    loss = loss / accum

                scaler.scale(loss).backward()

                # 统计（用未缩放前的数值更直观，这里乘回去）
                total_loss += float(loss.detach().item()) * accum * c_mb.size(0)

            scaler.step(optimizer)
            scaler.update()

            total_pairs += B
            pairs_since_last += B
            step += 1

            if step % args.log_every == 0 and (not is_ddp or rank==0):
                now = time.time(); dt = max(1e-9, now-last_t)
                thr = pairs_since_last / dt
                mem = 0.0
                if device.type=="cuda":
                    try: mem = torch.cuda.memory_allocated(device=device)/(1024**2)
                    except Exception: pass
                print(f"[{view_name}] step={step:,} pairs/step={B} accum={accum} throughput={thr:,.0f} pairs/s "
                      f"loss(avg-step)~{(total_loss/max(1,total_pairs)):.4f} mem~{mem:.0f}MB", flush=True)
                last_t = now; pairs_since_last = 0

            # 提前结束本 epoch
            if v_max_pairs is not None and total_pairs >= v_max_pairs:
                if (not is_ddp) or rank==0:
                    print(f"[{view_name}] early stop epoch {ep}: reached max_pairs_per_epoch={v_max_pairs:,}", flush=True)
                break

        dt = time.time() - t0
        if (not is_ddp) or rank==0:
            if total_pairs == 0:
                print(f"[Train-{view_name}] epoch {ep}: no pairs produced")
            else:
                print(f"[Train-{view_name}] epoch {ep}: pairs={total_pairs:,}, "
                      f"avg_loss={total_loss/max(1,total_pairs):.4f}, time={dt:.1f}s", flush=True)

            # 轻量评估
            if len(eval_samples) > 0:
                nn_res = quick_eval_neighbors(model, eval_samples, topk=args.eval_topk,
                                              chunk=args.eval_chunk, doc_ids=doc_ids)
                print(f"[Eval-{view_name}] samples={list(eval_samples)} top{args.eval_topk}:", flush=True)
                for q in eval_samples:
                    if q in nn_res:
                        pretty = ", ".join([f"(doc_idx={i},Id={did},s={s:.3f})"
                                            for (i,s,did) in nn_res[q]])
                        print(f"  q(doc_idx={q},Id={int(doc_ids[q])}) → {pretty}", flush=True)

            # 每 epoch checkpoint
            if args.save_epoch_emb:
                E = (model.module.in_emb.weight if hasattr(model, "module") else model.in_emb.weight).detach().cpu().numpy()
                Z = E.astype(np.float16 if args.emb_dtype=="float16" else np.float32, copy=True)
                nrm = np.linalg.norm(Z, axis=1, keepdims=True); mask = (nrm[:,0] > 0)
                Z[mask] = Z[mask] / nrm[mask]
                part_path = out_path.parent / f"Z_{view_name}_epoch{ep}.parquet"
                df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
                df.insert(0, "doc_idx", np.arange(N, dtype=np.int64))
                df.to_parquet(part_path, engine="fastparquet", index=False)
                print(f"[Checkpoint-{view_name}] saved {part_path.name} ({args.emb_dtype})", flush=True)

        barrier(is_ddp)

    # final export（rank0）
    if (not is_ddp) or rank==0:
        E = (model.module.in_emb.weight if hasattr(model, "module") else model.in_emb.weight).detach().cpu().numpy()
        Z = E.astype(np.float32, copy=True)
        nrm = np.linalg.norm(Z, axis=1, keepdims=True); mask = (nrm[:,0] > 0)
        Z[mask] = Z[mask] / nrm[mask]
        df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
        df.insert(0, "doc_idx", np.arange(N, dtype=np.int64))
        df.to_parquet(out_path, engine="fastparquet", index=False)
        print(f"[Train-{view_name}] saved {out_path.name}; covered_docs={int(mask.sum())}/{N} ({mask.mean():.1%})", flush=True)

    del model, optimizer; gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    barrier(is_ddp)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    is_ddp, rank, world, local, device = init_ddp("nccl")
    log0(is_ddp, rank, f"[DDP] enabled={is_ddp}, rank={rank}/{world}, local_rank={local}, device={device}")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    TMP = Path(args.tmp_dir)
    assert TMP.exists(), f"tmp_dir not found: {TMP}"

    # 读中间件
    doc_df     = pd.read_parquet(TMP / "doc_clean.parquet", engine="fastparquet")
    tag_vocab  = pd.read_parquet(TMP / "tag_vocab.parquet", engine="fastparquet")
    text_vocab = pd.read_parquet(TMP / "text_vocab.parquet", engine="fastparquet")
    rw_params  = pd.read_parquet(TMP / "rw_params.parquet", engine="fastparquet").iloc[0]
    N = len(doc_df); T = len(tag_vocab); W = len(text_vocab)

    DT_ppmi = load_csr_triplet_parquet(TMP / "DT_ppmi.parquet", shape=(N, T))
    DW_bm25 = load_csr_triplet_parquet(TMP / "DW_bm25.parquet", shape=(N, W))

    tag_corpus, text_corpus, start_tag, start_txt, degD_tag, degD_txt = build_corpus(
        doc_df, DT_ppmi, DW_bm25, rw_params, device, is_ddp, rank, world
    )

    doc_ids = doc_df["Id"].to_numpy()

    # 顺序训练两视图（避免峰值叠加）
    train_view("tag",  N, start_tag,  degD_tag,  tag_corpus,  device, is_ddp, rank, args, TMP / "Z_tag.parquet",  doc_ids)
    train_view("text", N, start_txt,  degD_txt,  text_corpus, device, is_ddp, rank, args, TMP / "Z_text.parquet", doc_ids)

    log0(is_ddp, rank, "[Step 6] Done. Saved: Z_tag.parquet / Z_text.parquet")

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()
