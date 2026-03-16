# Content Coverage Expansion Plan: 10K / 50K / 100K

## 1. 概述

将内容特征管线从当前 1,000 个数据集（D_content）扩展到 10,000 / 50,000 / 100,000 个数据集，
然后在对应规模下评测融合效果。目标是提高内容视图的覆盖率，从而在更大评测子集上验证四视图融合的增益。

## 2. 当前状态

| 指标 | 值 |
|------|-----|
| D_content 规模 | 1,000 datasets |
| 候选池（7,393 候选，tabular tags + size 100KB-100MB + views≥1000） | top-1000 by downloads |
| 已下载数据 | 32GB（`data/tabular_raw/`，1000 个目录） |
| API 缓存 | 1,161 条（`tmp/content/api_cache/`） |
| 磁盘可用 | ~6.3TB（充足） |
| Kaggle API | 已认证 |

## 3. 每个规模的数据采集需求

### 3.1 10K 规模

| 参数 | 值 |
|------|-----|
| 目标 D_content | 10,000 datasets |
| 筛选策略 | tabular tags, size 100KB–100MB, **无** views 阈值 |
| 候选池大小 | ~29,866 |
| SEARCH_MARGIN | 1.2（搜索 12,000 个） |
| 预计新增下载 | ~9,000 datasets |
| 预计磁盘增量 | ~288GB（均值 32MB/dataset） |
| 预计累计磁盘 | ~320GB |
| 预计 API 调用 | ~11,000（扣除缓存命中） |
| 预计耗时 | API 搜索 ~1.5h + 下载 ~4h |

### 3.2 50K 规模

| 参数 | 值 |
|------|-----|
| 目标 D_content | 50,000 datasets |
| 筛选策略 | broad tags*, size 50KB–500MB, **无** views 阈值 |
| 候选池大小 | ~50,059 |
| SEARCH_MARGIN | 1.2（搜索 60,000 个） |
| 预计新增下载 | ~40,000 datasets（增量） |
| 预计磁盘增量 | ~1.3TB |
| 预计累计磁盘 | ~1.6TB |
| 预计 API 调用 | ~49,000（增量） |
| 预计耗时 | API 搜索 ~7h + 下载 ~15h |

*broad tags = tabular + data visualization + feature engineering + statistics + finance + healthcare + economics + survey + census + social science

### 3.3 100K 规模

| 参数 | 值 |
|------|-----|
| 目标 D_content | 100,000 datasets |
| 筛选策略 | **任意** tags, size 10KB–1GB, **无** views 阈值 |
| 候选池大小 | ~214,603 |
| SEARCH_MARGIN | 1.2（搜索 120,000 个） |
| 预计新增下载 | ~50,000 datasets（增量） |
| 预计磁盘增量 | ~1.6TB |
| 预计累计磁盘 | ~3.2TB |
| 预计 API 调用 | ~70,000（增量） |
| 预计耗时 | API 搜索 ~10h + 下载 ~20h |

## 4. 元数据与下载数据的一一对应保证

1. **唯一性约束**: `d_content.parquet` 中每行有唯一 `(Id, doc_idx)` 对，`main_tables.parquet` 以 `(DatasetId, doc_idx)` 为键
2. **双向验证**: 构建后检查 d_content IDs ⊆ main_tables IDs 且 main_tables IDs ⊆ d_content IDs
3. **doc_idx 校验**: 所有 doc_idx 与 `tmp/index_map.parquet` 一致
4. **目录校验**: 所有 `data/tabular_raw/{Id}/` 目录存在且非空
5. **去重**: slug_to_ref 中同一 Slug 不重复搜索

## 5. 表格格式验证机制

### 5.1 下载后验证

每个数据集下载后，调用 `select_main_table(dataset_dir)` 扫描目录：
- 仅识别 `.csv`, `.tsv`, `.parquet`, `.xlsx`, `.xls` 扩展名
- 排除 submission、dictionary、codebook、README、LICENSE 文件
- 选择最大的表格文件作为主表

### 5.2 非表格识别

`select_main_table()` 返回 `(None, 0, None)` 表示无表格文件，该数据集标记为"非表格"。

### 5.3 回填流程

非表格数据集通过两层回填替换：

1. **Tier-1**: 从已 API 匹配但未选入 d_content 的候选中，按 TotalDownloads 降序依次替换
2. **Tier-2**: 若 Tier-1 不足，从未搜索的候选池中进行 API 搜索 + 下载 + 验证

回填后重新运行完整性检查。

## 6. 筛选策略详情

### 6.1 渐进放宽策略

| 规模 | Tags | Size Range | Views | 候选池 |
|------|------|------------|-------|--------|
| ≤10K | TABULAR_TAGS | 100KB–100MB | ≥0 | ~29,866 |
| ≤50K | BROAD_TAGS | 50KB–500MB | ≥0 | ~50,059 |
| ≤100K | 任意 (None) | 10KB–1GB | ≥0 | ~214,603 |

### 6.2 Tag 集合定义

```python
TABULAR_TAGS = {
    "tabular", "classification", "regression",
    "exploratory data analysis", "data analytics",
}

BROAD_TAGS = TABULAR_TAGS | {
    "data visualization", "feature engineering", "statistics",
    "finance", "healthcare", "economics", "survey", "census",
    "social science", "business", "education", "sports",
    "geospatial analysis", "time series analysis",
}
```

### 6.3 选取策略

从候选池中按 `TotalDownloads` 降序选取 `target × SEARCH_MARGIN` 个进行 API 搜索和下载。

## 7. 预估磁盘用量

| 规模 | 原始数据 | 中间产物 | 相似度矩阵 | 合计 |
|------|---------|---------|-----------|------|
| 10K | ~320GB | ~5GB | ~2GB | ~327GB |
| 50K | ~1.6TB | ~25GB | ~10GB | ~1.635TB |
| 100K | ~3.2TB | ~50GB | ~20GB | ~3.27TB |

注：中间产物包括 col_profiles、col_descriptions、col_embeddings、Z_tabcontent。
相似度矩阵以 partitioned parquet 格式存储（COO 稀疏）。

## 8. 增量设计

- **复用已下载**: `data/tabular_raw/` 中已有目录直接跳过
- **API 缓存**: `tmp/content/api_cache/` 中已有 slug 直接读取
- **检查点**: 每 100 个数据集保存一次中间状态
- **可中断恢复**: 脚本检测已有产物，跳过已完成步骤
- **共享下载目录**: 所有规模共享 `data/tabular_raw/{Id}/`

## 9. 产出文件

每个规模 N 产出：
- `tmp/content/scale_{N}/d_content.parquet` — D_content 子集
- `tmp/content/scale_{N}/main_tables.parquet` — 主表注册表
- `tmp/content/scale_{N}/slug_to_ref.csv` — API 匹配结果
- `tmp/content/scale_{N}/results_*.csv` — 评测结果
- `data/tabular_raw/{Id}/` — 共享下载目录

## 10. 关键约束与风险

1. **Kaggle API 速率限制**: 搜索每次 0.5s 延迟，下载有隐含并发限制
2. **非表格比例**: 约 5-10% 的候选需要回填替换
3. **大文件处理**: 超大文件（>100MB）采样可能超时，设置 300s 超时
4. **100K 规模的 CPU fallback**: 需要 sklearn NearestNeighbors 替代 O(N²) 密集矩阵
5. **内存**: 100K × 384 × 4B ≈ 150MB（嵌入矩阵），可控

## 11. 执行顺序

| 步骤 | 内容 |
|------|------|
| 0 | 撰写此需求文档 |
| 1 | 创建 `src/content/acquisition.py` |
| 2 | 修复 `src/content/pipeline.py` numpy fallback |
| 3 | 更新 `src/content/__init__.py` |
| 4 | 创建 `scripts/expand_content_coverage.py` |
| 5 | 创建 `scripts/run_content_at_scale.py` |
| 6 | 运行 10K 数据采集 |
| 7 | 运行 10K content pipeline + 评测 |
| 8-11 | 重复 50K、100K |
| 12 | 更新 RESULTS_SUBSET_EVALUATION.md §12 |

## 12. 验证标准

- 10K 实验: D_content 应有 ~9,400+ 个成功 profiled 的数据集（94%+ 成功率）
- 10K 评测: Content-only 在纯 D_content 子集上的 Tag-nDCG 应保持 ~0.7+
- 融合方法在更高覆盖率下应比 1K 实验有更大增益
- 所有脚本 `--seed 42` 确保可复现
