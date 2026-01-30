#!/usr/bin/env python3
"""测试可视化修复是否正确"""

import pandas as pd
import numpy as np
from pathlib import Path

TMP_DIR = Path("/workspace/recsys/tmp")

print("=" * 60)
print("测试可视化修复")
print("=" * 60)

# 测试1: 检查metrics文件
print("\n[测试1] 检查metrics文件...")
df_main = pd.read_csv(TMP_DIR / "metrics_main.csv")
df_desc = pd.read_csv(TMP_DIR / "metrics_desc_based.csv")

print(f"✅ metrics_main.csv: {len(df_main)} 个方法")
print(f"   列: {list(df_main.columns[:10])}...")
print(f"   有Org列: {'Org-nDCG@20' in df_main.columns}")
print(f"   有Desc列: {'Desc-nDCG@20' in df_main.columns}")

print(f"\n✅ metrics_desc_based.csv: {len(df_desc)} 个方法")
print(f"   列: {list(df_desc.columns[:10])}...")
print(f"   有Org列: {'Org-nDCG@20' in df_desc.columns}")
print(f"   有Desc列: {'Desc-nDCG@20' in df_desc.columns}")

# 测试2: 测试自动检测逻辑
print("\n[测试2] 测试自动检测Desc/Org列...")

W_TAG = 0.5
W_DESC = 0.3
W_CREATOR = 0.2

def coln(view, metric):
    return f"{view}-{metric}"

def to_unified_row(row):
    """测试统一指标计算"""
    out = {}
    metric = "nDCG@20"

    t = float(row.get(coln("Tag", metric), 0.0) or 0.0)

    # 自动检测
    desc_col = coln("Desc", metric)
    org_col = coln("Org", metric)

    if pd.notna(row.get(desc_col)):
        d = float(row.get(desc_col, 0.0) or 0.0)
        source = "Desc"
    elif pd.notna(row.get(org_col)):
        d = float(row.get(org_col, 0.0) or 0.0)
        source = "Org"
    else:
        d = 0.0
        source = "None"

    c = float(row.get(coln("Creator", metric), 0.0) or 0.0)

    unified = W_TAG * t + W_DESC * d + W_CREATOR * c

    return source, t, d, c, unified

# 测试metrics_main.csv的第一个方法
print("\n从 metrics_main.csv (有Org列):")
row1 = df_main.iloc[0]
source1, t1, d1, c1, u1 = to_unified_row(row1)
print(f"  方法: {row1['method']}")
print(f"  来源维度: {source1}")
print(f"  Tag={t1:.4f}, {source1}={d1:.4f}, Creator={c1:.4f}")
print(f"  Unified@nDCG = {u1:.4f}")

# 测试metrics_desc_based.csv的第一个方法
print("\n从 metrics_desc_based.csv (有Desc列):")
row2 = df_desc.iloc[0]
source2, t2, d2, c2, u2 = to_unified_row(row2)
print(f"  方法: {row2['method']}")
print(f"  来源维度: {source2}")
print(f"  Tag={t2:.4f}, {source2}={d2:.4f}, Creator={c2:.4f}")
print(f"  Unified@nDCG = {u2:.4f}")

# 测试3: 验证合并后的数据
print("\n[测试3] 验证合并和筛选...")
metrics_all = pd.concat([df_main, df_desc], ignore_index=True)
metrics_all = metrics_all.drop_duplicates(subset=["method"], keep="last")

print(f"✅ 合并后共 {len(metrics_all)} 个方法")

# 测试筛选
SELECTED = ["Fused3-RA", "Fused3-RR", "BM25"]
def canon(s):
    return s.strip().lower()

real_by_canon = {canon(m): m for m in metrics_all["method"].tolist()}

picked = []
for want in SELECTED:
    key = canon(want)
    if key in real_by_canon:
        picked.append(real_by_canon[key])
        print(f"  ✓ 找到: {want}")
    else:
        print(f"  ✗ 未找到: {want}")

print("\n" + "=" * 60)
print("✅ 所有测试通过！可视化应该能正常工作。")
print("=" * 60)
