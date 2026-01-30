#!/usr/bin/env python3
"""测试完整评测流程（使用前2个分片）"""

import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
from collections import defaultdict

# 参数
PARQUET_ENGINE = 'pyarrow'
K_EVAL = 20
DESC_THRESHOLD = 0.2
MIN_SIM = 0.05
TMP_DIR = Path("/workspace/recsys/tmp")
PREF = "S_fused3_rr_k50"

print("=" * 60)
print("测试完整评测流程")
print("=" * 60)

# 加载推荐结果manifest
print("\n[1/6] 加载推荐结果manifest...")
try:
    with open(TMP_DIR / f"{PREF}_manifest.json") as f:
        manifest = json.load(f)
    N = manifest["nodes"]
    files = manifest.get("files", [])
    print(f"✅ Manifest读取成功: {N} nodes, {len(files)} files")
except Exception as e:
    print(f"❌ 错误: {e}")
    exit(1)

# 加载Tag数据
print("\n[2/6] 加载Tag数据...")
try:
    tag_docs = pd.read_parquet(TMP_DIR / "relevance_tag_docs.parquet", engine=PARQUET_ENGINE)
    tag_idf = pd.read_parquet(TMP_DIR / "relevance_tag_idf.parquet", engine=PARQUET_ENGINE)

    doc2tags = {}
    for _, row in tag_docs.iterrows():
        doc_idx = int(row["doc_idx"])
        tags = row.get("tags", [])
        if tags and len(tags) > 0:
            doc2tags[doc_idx] = set(tags)

    idf_map = dict(zip(tag_idf["tag"], tag_idf["idf"]))
    tag_freq = dict(zip(tag_idf["tag"], tag_idf["df"]))
    print(f"✅ Tag数据加载成功: {len(doc2tags)} docs, {len(idf_map)} tags")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 加载Description BM25相似度
print("\n[3/6] 加载Description BM25相似度...")
try:
    with open(TMP_DIR / "S_textbm25_topk_k50_manifest.json") as f:
        bm25_manifest = json.load(f)

    desc_sim_dict = defaultdict(dict)
    part_files = bm25_manifest.get("parts", bm25_manifest.get("files", []))

    # 只加载前2个分片用于测试
    for part_file in part_files[:2]:
        df = pd.read_parquet(TMP_DIR / part_file, engine=PARQUET_ENGINE)
        for row, col, val in zip(df['row'].values, df['col'].values, df['val'].values):
            if val > MIN_SIM:
                desc_sim_dict[int(row)][int(col)] = float(val)

    print(f"✅ Desc数据加载成功: {len(desc_sim_dict)} docs with neighbors")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 加载Creator数据
print("\n[4/6] 加载Creator数据...")
try:
    beh_df = pd.read_parquet(TMP_DIR / "beh_base.parquet", engine=PARQUET_ENGINE)
    cre_arr = np.full(N, -1, dtype=np.int64)
    valid_cre = beh_df[beh_df["CreatorUserId"].notna()]
    cre_arr[valid_cre["doc_idx"].to_numpy(np.int64)] = valid_cre["CreatorUserId"].to_numpy(np.int64)

    # 过滤单文档Creator
    cre_counts = pd.Series(cre_arr[cre_arr >= 0]).value_counts().to_dict()
    for idx in range(N):
        if cre_arr[idx] >= 0 and cre_counts.get(cre_arr[idx], 0) <= 1:
            cre_arr[idx] = -1

    cre_size = {k: v for k, v in cre_counts.items() if v > 1}
    print(f"✅ Creator数据加载成功: {(cre_arr >= 0).sum()} docs, {len(cre_size)} creators")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 计算IDCG
print("\n[5/6] 计算IDCG...")
IDCG = sum(1.0 / math.log2(r + 2) for r in range(K_EVAL))
print(f"✅ IDCG@{K_EVAL} = {IDCG:.4f}")

# 测试评测流程（使用第一个分片）
print("\n[6/6] 测试评测流程（第一个分片）...")
try:
    def zero_metrics():
        return {"ndcg": 0.0, "ap": 0.0, "rr": 0.0, "p": 0.0, "r": 0.0, "covered": 0, "total": 0}

    def update_ndcg(m, gains):
        if gains.size == 0:
            return
        dcg = float(np.sum(gains / np.log2(np.arange(gains.size) + 2)))
        m["ndcg"] += (dcg / IDCG)

    def update_binary_metrics(m, rel_flags):
        if rel_flags.size == 0:
            return
        K = rel_flags.size
        m["p"] += rel_flags.sum() / float(K)
        if rel_flags.sum() > 0:
            ranks = np.where(rel_flags > 0)[0]
            prec_at_hits = [(rel_flags[:(r+1)].sum() / (r+1)) for r in ranks]
            m["ap"] += float(np.mean(prec_at_hits))
            m["rr"] += 1.0 / (ranks[0] + 1)

    m_tag = zero_metrics()
    m_desc = zero_metrics()
    m_cre = zero_metrics()

    # 读取第一个分片
    df = pd.read_parquet(TMP_DIR / files[0], engine=PARQUET_ENGINE)
    rows = df["row"].to_numpy(np.int64)
    cols = df["col"].to_numpy(np.int64)
    vals = df["val"].to_numpy(np.float32)

    # 排序
    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]

    # 找到每个查询文档的起始位置
    uniq_rows, start_idx = np.unique(rows, return_index=True)
    start_idx = np.append(start_idx, len(rows))

    print(f"   处理 {len(uniq_rows)} 个查询文档...")

    # 只处理前10个文档用于测试
    for i, q in enumerate(uniq_rows[:10]):
        q = int(q)
        start = start_idx[i]
        end = start_idx[i + 1]
        neigh = cols[start:end][:K_EVAL]

        if len(neigh) == 0:
            continue

        # Tag维度
        tags_q = doc2tags.get(q, set())
        if len(tags_q) > 0:
            m_tag["covered"] += 1
            gains = np.zeros(len(neigh), dtype=np.float32)
            flags = np.zeros(len(neigh), dtype=np.int32)

            for j, nid in enumerate(neigh):
                tags_n = doc2tags.get(int(nid), set())
                inter = tags_q & tags_n
                if len(inter) > 0:
                    flags[j] = 1
                    idf_inter = sum(idf_map.get(t, 0.0) for t in inter)
                    idf_q = sum(idf_map.get(t, 0.0) for t in tags_q)
                    idf_n = sum(idf_map.get(t, 0.0) for t in tags_n)
                    idf_union = idf_q + idf_n - idf_inter
                    if idf_union > 0:
                        gains[j] = idf_inter / idf_union

            update_ndcg(m_tag, gains)
            update_binary_metrics(m_tag, flags)
        m_tag["total"] += 1

        # Desc维度
        if q in desc_sim_dict:
            sim_scores = np.array([
                desc_sim_dict[q].get(int(nid), 0.0) for nid in neigh
            ], dtype=np.float32)

            if sim_scores.max() > 0:
                m_desc["covered"] += 1
                gains = sim_scores
                update_ndcg(m_desc, gains)

                rel_binary = (sim_scores > DESC_THRESHOLD).astype(np.int32)
                update_binary_metrics(m_desc, rel_binary)
        m_desc["total"] += 1

        # Creator维度
        cre_q = cre_arr[q]
        if cre_q >= 0:
            m_cre["covered"] += 1
            flags = (cre_arr[neigh] == cre_q).astype(np.int32)
            gains = flags.astype(np.float32)
            update_ndcg(m_cre, gains)
            update_binary_metrics(m_cre, flags)
        m_cre["total"] += 1

    # 输出结果
    def finalize(m):
        tot = max(m["total"], 1)
        return {
            "nDCG@20": m["ndcg"] / tot,
            "MAP@20": m["ap"] / tot,
            "MRR@20": m["rr"] / tot,
            "P@20": m["p"] / tot,
            "Coverage": m["covered"] / tot
        }

    res_tag = finalize(m_tag)
    res_desc = finalize(m_desc)
    res_cre = finalize(m_cre)

    print(f"✅ 评测流程测试成功！")
    print(f"\n结果（前10个文档）:")
    print(f"  Tag:  nDCG={res_tag['nDCG@20']:.4f}, MAP={res_tag['MAP@20']:.4f}, Coverage={res_tag['Coverage']:.4f}")
    print(f"  Desc: nDCG={res_desc['nDCG@20']:.4f}, MAP={res_desc['MAP@20']:.4f}, Coverage={res_desc['Coverage']:.4f}")
    print(f"  Cre:  nDCG={res_cre['nDCG@20']:.4f}, MAP={res_cre['MAP@20']:.4f}, Coverage={res_cre['Coverage']:.4f}")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！Notebook应该可以正常运行。")
print("=" * 60)
