#!/usr/bin/env python3
"""Fix issues in analyse_ddp_hybrid.ipynb"""
import json

with open('analyse_ddp_hybrid.ipynb', 'r') as f:
    nb = json.load(f)

# Fix Cell 3: Add scipy import check
cell3_code = """# ==================== 全局导入 ====================
import os
import re
import json
import time
import random
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

# Check scipy availability
try:
    from scipy import sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy 未安装")
    print("  安装命令: pip install scipy")

# PyTorch (用于notebook中的非DDP操作)
import torch

# ==================== 全局配置 ====================
GLOBAL_SEED = 2025
TMP_DIR = Path("./tmp")
TMP_DIR.mkdir(exist_ok=True)
PARQUET_ENGINE = "fastparquet"

# 设置随机种子
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(GLOBAL_SEED)

# I/O辅助函数
def save_parquet_df(df, path):
    df.to_parquet(path, engine=PARQUET_ENGINE, index=False)

def load_parquet_df(path):
    return pd.read_parquet(path, engine=PARQUET_ENGINE)

print("[Setup] ✓ 全局配置完成")
print(f"[Setup] TMP_DIR: {TMP_DIR.absolute()}")
print(f"[Setup] 随机种子: {GLOBAL_SEED}")
if not HAS_SCIPY:
    print("[Setup] ⚠️  scipy 未安装 - Step 3-4及后续步骤需要scipy")"""

nb['cells'][3]['source'] = cell3_code

# Fix Cell 16: Remove text=True to show real-time output
cell16_source = nb['cells'][16]['source']
if isinstance(cell16_source, list):
    cell16_source = ''.join(cell16_source)
cell16_new = cell16_source.replace(
    'result = subprocess.run(cmd, check=True, text=True)',
    'result = subprocess.run(cmd, check=True)'
)
nb['cells'][16]['source'] = cell16_new

# Fix Cell 18: Remove text=True
cell18_source = nb['cells'][18]['source']
if isinstance(cell18_source, list):
    cell18_source = ''.join(cell18_source)
cell18_new = cell18_source.replace(
    'result = subprocess.run(cmd, check=True, text=True)',
    'result = subprocess.run(cmd, check=True)'
)
nb['cells'][18]['source'] = cell18_new

# Add scipy check in cells that use sparse
for i in [9, 12, 21]:  # Cells that use scipy.sparse
    source = nb['cells'][i]['source']
    # Convert list to string if needed
    if isinstance(source, list):
        source = ''.join(source)

    if 'sparse.' in source:
        # Add check at the beginning
        check = """if not HAS_SCIPY:
    print("❌ 错误: scipy未安装，无法继续")
    print("   安装命令: pip install scipy")
else:
"""
        # Indent the rest
        lines = source.split('\n')
        indented = '\n'.join(['    ' + line if line.strip() else line for line in lines])
        nb['cells'][i]['source'] = check + indented

# Save fixed notebook
with open('analyse_ddp_hybrid.ipynb', 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✓ Fixed analyse_ddp_hybrid.ipynb")
print("  - Added scipy import check")
print("  - Fixed subprocess.run to show real-time output")
print("  - Added scipy availability checks in cells 9, 12, 21")
