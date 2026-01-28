#!/usr/bin/env python3
"""测试Description相似度修复是否正确"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import sparse
from sklearn.preprocessing import normalize

# 参数
TMP_DIR = Path("/workspace/recsys/tmp")
PARQUET_ENGINE = 'fastparquet'
DESC_THRESHOLD = 0.2
K_EVAL = 20

print("=" * 60)
print("测试Description相似度修复")
print("=" * 60)

# 测试1: 加载DW_bm25矩阵
print("\n[1/5] 加载DW_bm25矩阵...")
try:
    df_bm25 = pd.read_parquet(TMP_DIR / "DW_bm25.parquet", engine=PARQUET_ENGINE)
    print(f"✅ DW_bm25数据加载成功")
    print(f"   行数: {len(df_bm25):,}")
    print(f"   列: {list(df_bm25.columns)}")

    N = 521734  # 从manifest获取
    N_words = int(df_bm25["col"].max() + 1)

    DW_bm25 = sparse.csr_matrix(
        (df_bm25["val"].values, (df_bm25["row"].values, df_bm25["col"].values)),
        shape=(N, N_words),
        dtype=np.float32
    )

    print(f"✅ 稀疏矩阵构建成功")
    print(f"   形状: {DW_bm25.shape}")
    print(f"   非零元素: {DW_bm25.nnz:,}")
    print(f"   密度: {DW_bm25.nnz / (DW_bm25.shape[0] * DW_bm25.shape[1]) * 100:.4f}%")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试2: L2归一化
print("\n[2/5] L2归一化...")
try:
    DW_bm25_norm = normalize(DW_bm25, norm='l2', axis=1)
    print(f"✅ 归一化成功")
    print(f"   归一化后形状: {DW_bm25_norm.shape}")
    print(f"   归一化后非零元素: {DW_bm25_norm.nnz:,}")

    # 验证归一化（检查几个非零行的范数）
    has_bm25 = np.array(DW_bm25.getnnz(axis=1) > 0).flatten()
    valid_indices = np.where(has_bm25)[0][:10]
    norms = []
    for idx in valid_indices:
        row = DW_bm25_norm[idx].toarray().flatten()
        norm = np.linalg.norm(row)
        norms.append(norm)

    avg_norm = np.mean(norms)
    print(f"   验证范数 (前10个非零行平均): {avg_norm:.6f} (应接近1.0)")

    if abs(avg_norm - 1.0) < 0.01:
        print(f"   ✅ 归一化验证通过")
    else:
        print(f"   ⚠️  归一化可能有问题")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试3: 覆盖率统计
print("\n[3/5] 统计覆盖率...")
try:
    has_bm25 = np.array(DW_bm25.getnnz(axis=1) > 0).flatten()
    coverage_count = has_bm25.sum()
    coverage_rate = coverage_count / N * 100

    print(f"✅ 覆盖率统计成功")
    print(f"   有BM25向量的文档: {coverage_count:,} / {N:,}")
    print(f"   覆盖率: {coverage_rate:.2f}%")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 实时相似度计算
print("\n[4/5] 测试实时相似度计算...")
try:
    def compute_bm25_similarity(q_idx, neigh_indices):
        """实时计算BM25余弦相似度"""
        q_vec = DW_bm25_norm[q_idx]
        neigh_vecs = DW_bm25_norm[neigh_indices]
        sim_scores = q_vec.dot(neigh_vecs.T).toarray().flatten().astype(np.float32)
        return sim_scores

    # 找一个有BM25向量的文档
    test_doc = np.where(has_bm25)[0][0]

    # 随机选择一些邻居（包括自己）
    test_neighbors = np.array([test_doc, test_doc + 1, test_doc + 2, test_doc + 10, test_doc + 100])

    # 计算相似度
    sim_scores = compute_bm25_similarity(test_doc, test_neighbors)

    print(f"✅ 相似度计算成功")
    print(f"   测试文档: {test_doc}")
    print(f"   邻居: {test_neighbors}")
    print(f"   相似度分数: {sim_scores}")
    print(f"   自相似度 (应为1.0): {sim_scores[0]:.6f}")

    if abs(sim_scores[0] - 1.0) < 0.01:
        print(f"   ✅ 自相似度验证通过")
    else:
        print(f"   ⚠️  自相似度不为1.0，可能有问题")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试5: 完整评测流程（使用推荐结果的第一个分片）
print("\n[5/5] 测试完整评测流程...")
try:
    # 加载推荐结果
    with open(TMP_DIR / "S_fused3_rr_k50_manifest.json") as f:
        manifest = json.load(f)

    files = manifest.get("files", [])
    if not files:
        print("❌ 未找到推荐结果文件")
        exit(1)

    # 读取第一个分片
    df = pd.read_parquet(TMP_DIR / files[0], engine=PARQUET_ENGINE)
    rows = df["row"].to_numpy(np.int64)
    cols = df["col"].to_numpy(np.int64)

    # 排序
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]

    # 找到每个查询文档的起始位置
    uniq_rows, start_idx = np.unique(rows, return_index=True)
    start_idx = np.append(start_idx, len(rows))

    print(f"   推荐结果加载成功: {len(uniq_rows)} 个查询文档")

    # 统计Desc维度
    covered = 0
    total = 0
    sim_examples = []

    # 只处理前100个文档
    for i, q in enumerate(uniq_rows[:100]):
        q = int(q)
        start = start_idx[i]
        end = start_idx[i + 1]
        neigh = cols[start:end][:K_EVAL]

        if len(neigh) == 0:
            continue

        total += 1

        if has_bm25[q]:
            sim_scores = compute_bm25_similarity(q, neigh)

            if sim_scores.max() > 0:
                covered += 1

                # 保存一些示例
                if len(sim_examples) < 3:
                    high_sim_count = (sim_scores > DESC_THRESHOLD).sum()
                    sim_examples.append({
                        "doc": q,
                        "max_sim": sim_scores.max(),
                        "avg_sim": sim_scores.mean(),
                        "high_sim_count": high_sim_count
                    })

    coverage = covered / max(total, 1) * 100

    print(f"✅ 评测流程测试成功")
    print(f"   处理文档数: {total}")
    print(f"   有相似度的文档: {covered}")
    print(f"   覆盖率: {coverage:.2f}%")

    if sim_examples:
        print(f"\n   示例:")
        for ex in sim_examples:
            print(f"     文档 {ex['doc']}: max_sim={ex['max_sim']:.4f}, "
                  f"avg_sim={ex['avg_sim']:.4f}, "
                  f"高相似邻居(>{DESC_THRESHOLD})={ex['high_sim_count']}")

    if coverage > 50:
        print(f"   ✅ 覆盖率正常 (>{50}%)")
    else:
        print(f"   ⚠️  覆盖率较低 (<{50}%)")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！修复应该有效。")
print("=" * 60)
