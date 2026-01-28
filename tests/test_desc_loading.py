#!/usr/bin/env python3
"""测试Description相似度加载"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# 路径设置
TMP_DIR = Path("/workspace/recsys/tmp")
PARQUET_ENGINE = 'pyarrow'
MIN_SIM = 0.05

print("测试1: 检查manifest文件...")
manifest_path = TMP_DIR / "S_textbm25_topk_k50_manifest.json"
if not manifest_path.exists():
    print(f"❌ Manifest文件不存在: {manifest_path}")
    exit(1)
else:
    print(f"✅ Manifest文件存在: {manifest_path}")

print("\n测试2: 读取manifest...")
try:
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"✅ Manifest读取成功")
    print(f"   nodes: {manifest.get('nodes')}")
    print(f"   k: {manifest.get('k')}")
    print(f"   键: {list(manifest.keys())}")
except Exception as e:
    print(f"❌ Manifest读取失败: {e}")
    exit(1)

print("\n测试3: 获取分片文件列表...")
part_files = manifest.get("parts", manifest.get("files", []))
if not part_files:
    print("❌ 未找到分片文件列表")
    exit(1)
else:
    print(f"✅ 找到 {len(part_files)} 个分片文件")
    print(f"   第一个: {part_files[0]}")

print("\n测试4: 检查第一个分片文件...")
first_part = TMP_DIR / part_files[0]
if not first_part.exists():
    print(f"❌ 分片文件不存在: {first_part}")
    exit(1)
else:
    print(f"✅ 分片文件存在: {first_part}")

print("\n测试5: 读取第一个分片...")
try:
    df = pd.read_parquet(first_part, engine=PARQUET_ENGINE)
    print(f"✅ 分片读取成功")
    print(f"   行数: {len(df)}")
    print(f"   列: {list(df.columns)}")
    print(f"   前5行:\n{df.head()}")
except Exception as e:
    print(f"❌ 分片读取失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n测试6: 加载所有Description相似度（前2个分片）...")
try:
    desc_sim_dict = defaultdict(dict)

    for i, part_file in enumerate(part_files[:2]):  # 只测试前2个
        print(f"   处理 {part_file}...")
        df = pd.read_parquet(TMP_DIR / part_file, engine=PARQUET_ENGINE)
        for row, col, val in zip(df['row'].values, df['col'].values, df['val'].values):
            if val > MIN_SIM:
                desc_sim_dict[int(row)][int(col)] = float(val)

    print(f"✅ Description相似度加载成功")
    print(f"   文档数: {len(desc_sim_dict)}")

    # 显示一个示例
    if desc_sim_dict:
        sample_doc = list(desc_sim_dict.keys())[0]
        sample_neighbors = desc_sim_dict[sample_doc]
        print(f"   示例 - 文档 {sample_doc} 有 {len(sample_neighbors)} 个邻居")
        print(f"   前3个邻居: {list(sample_neighbors.items())[:3]}")

except Exception as e:
    print(f"❌ Description相似度加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ 所有测试通过！")
