# ⚠️ CLAUDE — 请在执行任何任务前先阅读本文件 ⚠️

> **本文件是项目进度追踪文档。** 每次新会话开始时，请优先阅读此文件以快速恢复上下文，避免重复工作或遗漏依赖。

---

## 1. 已完成任务清单

### 任务 A：创建 `00_tabular_data_download.ipynb`

- **文件**：`notebooks/04_content/00_tabular_data_download.ipynb`
- **状态**：已完成，nbformat 4.4 验证通过
- **内容**：14 cells（2 markdown + 12 code），完整的数据下载管线
- **功能**：
  - 加载 521K 条元数据，按 tag=tabular / 文件大小 100KB–100MB / views≥1000 三重过滤
  - Kaggle API 搜索（slug 精确匹配 + 模糊匹配），带 `api_cache/` 缓存
  - 批量下载数据集到 `data/tabular_raw/{Id}/`，支持断点续传（每 50 条增量保存）
  - 调用 `select_main_table()` 选取各数据集主表
  - 端到端校验：产物存在性、d_content 与 tabular_raw 交叉验证、yield funnel 输出
- **关键参数**：B_DS=1000, MIN_VIEWS=1000, SIZE_MIN=100KB, SIZE_MAX=100MB, SEARCH_MARGIN=1.2

### 任务 B：瘦身 `01_content_data_acquisition.ipynb`

- **文件**：`notebooks/04_content/01_content_data_acquisition.ipynb`
- **状态**：已完成，nbformat 4.4 验证通过
- **变更**：26 cells → 11 cells（1 markdown title + 9 code + 1 markdown summary）
- **标题更改**："01 数据采集结果验证与分析"（原为下载管线，现改为纯验证分析）
- **移除内容**：所有下载/处理 cells（API 检测、搜索函数、匹配函数、批量搜索、下载函数、批量下载、主表选取逻辑）已迁移至 NB00
- **保留功能**：
  - 加载并验证 `d_content.parquet`、`slug_to_ref.csv`、`main_tables.parquet`
  - API 匹配质量分析（confidence 分布）
  - 下载覆盖率分析（d_content IDs vs tabular_raw 目录）
  - 数据质量抽样（5 个数据集用 `read_by_ext()` 验证可读性）
  - Yield funnel 可视化

### 任务 C：nbformat 验证

- 两个 notebook 均通过 `nbformat.read(..., as_version=4)` 验证
- NB00：14 cells, nbformat=4.4
- NB01：11 cells, nbformat=4.4

---

## 2. Notebook 管线阅读指南

执行顺序：`NB00 → NB01 → NB02 → NB03 → NB04`

| 顺序 | 文件 | 职责 |
|------|------|------|
| 1st | `00_tabular_data_download.ipynb` | **数据获取**：元数据过滤 → Kaggle API 搜索 → 批量下载 → 主表选取。产出所有 `tmp/content/` 核心产物和 `data/tabular_raw/` 原始数据 |
| 2nd | `01_content_data_acquisition.ipynb` | **验证分析**：加载 NB00 产物，校验 schema，分析 API 匹配质量、下载覆盖率、数据质量抽样，生成 yield funnel |
| 3rd | `02_content_view_construction.ipynb` | **内容视图构建**（41 cells）：列采样 → 列分析 → 描述生成 → Sentence-Transformer 编码 → FAISS 近邻搜索 → 对称化 + L1 归一化 → 一致性计算 |
| 4th | `03_content_fusion_experiments.ipynb` | **四视图融合实验**（45 cells）：ρ-adaptive 加权、银标准评估（Tag/Desc/Creator）、消融实验、budget sweep |
| 5th | `04_content_subset_evaluation.ipynb` | **D_content 子集评测**（30 cells）：在 1000 D_content 文档上评测 8 种方法，消除全局稀释效应，per-doc 分析，一致性-增益关联 |

### 数据流

```
NB00: metadata_merged.csv (521K) → 过滤 → API 搜索 → 下载 → 主表选取
        ↓ 产出: candidates.parquet, slug_to_ref.csv, d_content.parquet, main_tables.parquet
NB01: 加载 NB00 产物 → 验证 + 分析 → 报告
NB02: d_content + main_tables → 列分析 → 编码 → Z_tabcontent → S_tabcontent → consistency
NB03: 4 个视图 CSR 矩阵 → ρ 计算 → adaptive α → 融合 → 评估 + 消融
```

---

## 3. 关键产物文件清单

### `tmp/content/` 目录

| 文件 | 产出自 | Schema / 说明 |
|------|--------|---------------|
| `candidates.parquet` | NB00 | Id, Slug, Title, TotalViews, TotalDownloads, TotalCompressedBytes。~1000–1200 行 |
| `slug_to_ref.csv` | NB00 | Id, Slug, Title, ref, confidence（exact/fuzzy/none） |
| `d_content.parquet` | NB00 | doc_idx, Id, Slug, ref, TotalDownloads, TotalViews。恰好 B_DS=1000 行 |
| `main_tables.parquet` | NB00 | DatasetId, doc_idx, main_table_path, file_size, extension |
| `api_cache/{slug}.json` | NB00 | 单个 slug 的 Kaggle API 搜索结果缓存 |
| `col_profiles.parquet` | NB02 | 列级统计信息 |
| `col_descriptions.parquet` | NB02 | 列的英文文本描述 |
| `col_embeddings.parquet` | NB02 | 384 维列嵌入向量 |
| `Z_tabcontent.parquet` | NB02 | 数据集向量，B_ds × 384 |
| `S_tabcontent_symrow_k50_*` | NB02 | 内容视图相似度图（分区 parquet + manifest.json） |
| `consistency_meta_content.parquet` | NB02 | doc_idx, jaccard, weighted_consistency, n_meta, n_cont, n_intersect |
| `S_naive4_symrow_k50_*` | NB03 | 朴素融合边 |
| `S_fused4_symrow_k50_*` | NB03 | Adaptive 融合边 |
| `S_fused4c_symrow_k50_*` | NB03 | Adaptive + consistency 融合边 |
| `metrics_content_experiments.csv` | NB03 | 完整评估指标表 |
| `budget_sweep_results.csv` | NB03 | 历史 budget 扫描数据点 |
| `fig_*.png` | NB02/03 | 可视化图片（method_comparison, radar, budget_curve, consistency） |

### `data/tabular_raw/` 目录

- 结构：`{Id}/` 子目录，包含从 Kaggle 解压的数据集文件
- **当前状态**：空（NB00 尚未执行）

### 前置输入文件

| 文件 | 说明 |
|------|------|
| `data/metadata_merged.csv` | 521K 条数据集元数据 |
| `data/raw_data/DatasetVersions.csv` | 版本信息及压缩大小 |
| `tmp/index_map.parquet` | 全局 doc_idx 映射表 |

---

## 4. `src/content/` 模块说明

| 模块 | 职责 |
|------|------|
| `__init__.py` | 包入口，统一导出 5 个子模块的 25 个函数/类 |
| `sampling.py` | 表 I/O（`read_by_ext`）、主表选取（`select_main_table`）、行列采样（`sample_table`）、列分析（`profile_column/table`）、描述生成（`col_to_description`） |
| `encoding.py` | 列嵌入加权聚合为数据集向量（`aggregate_dataset_vector`），权重 = w_type × (1 - missing%) × min(unique%, 95%) |
| `similarity.py` | 稀疏相似度图构建（`sym_and_rownorm`）、分区 COO 文件 I/O（`save_partitioned_edges` / `load_edges_from_manifest` / `load_csr_from_manifest`） |
| `consistency.py` | 元数据-内容一致性指标：Jaccard 集合重叠 J(i) 和加权一致性 c(i)（`compute_jaccard_and_consistency`） |
| `fusion.py` | 多视图融合：ρ 信息集中度（`compute_rho`）、adaptive α 归一化（`compute_adaptive_alpha`）、一致性调节 g(i)（`apply_consistency_adjustment`）、加权融合 + top-K 裁剪（`fuse_views`） |

---

## 5. 当前状态与下一步

### 当前状态

- NB00 已创建但 **未执行**（需要 Kaggle API 凭证：`~/.kaggle/kaggle.json`）
- NB01 已瘦身但 **无法执行**（依赖 NB00 产物）
- NB02、NB03 代码完整，等待上游产物
- `data/tabular_raw/` 为空，`tmp/content/` 仅含空的 `api_cache/` 目录
- 所有 `src/content/` 模块已完成，可导入
- Git 分支：`wssit`

### 下一步

1. **运行 NB00**：在有 Kaggle 凭证的环境中执行，预期产出 ~1000 个数据集（~10GB 存储）
2. **运行 NB01**：验证 NB00 产物完整性和质量
3. **运行 NB02**：构建内容视图（列分析 → 嵌入 → 相似度图 → 一致性）
4. **运行 NB03**：四视图融合实验和评估

---

## 6. D_content 子集专属评测实验

### 任务 D：创建并运行 `04_content_subset_evaluation.ipynb`

- **文件**：`notebooks/04_content/04_content_subset_evaluation.ipynb`
- **状态**：已完成，papermill 执行通过
- **内容**：30 cells（8 markdown + 22 code），D_content 子集评测管线
- **目的**：消除全局稀释效应（521K 文档中仅 1000 有内容视图），在 D_content 子集上直接对比方法效果

### 评测结果摘要

在 1000 D_content 文档上评测 8 种方法（K_EVAL=20）：

| 方法 | Unified nDCG | Tag nDCG | Desc nDCG | Creator nDCG |
|------|-------------|----------|-----------|-------------|
| **Naive-Fusion** | **0.4749** | 0.6449 | 0.0131 | 0.7428 |
| Beh-only | 0.4739 | 0.6416 | 0.0097 | 0.7506 |
| Adaptive-Fusion | 0.4662 | 0.6257 | 0.0113 | 0.7496 |
| Adaptive+Cons | 0.4662 | 0.6257 | 0.0113 | 0.7498 |
| Meta-only | 0.4383 | 0.5763 | 0.0077 | 0.7391 |
| Content-only | 0.3889 | **0.7472** | 0.0157 | 0.0531 |
| Tag-only | 0.1152 | 0.2302 | 0.0000 | 0.0004 |
| Text-only | 0.1127 | 0.2249 | 0.0008 | 0.0002 |

### 关键发现

1. **融合方法在子集上超越 Meta-only**：Naive-Fusion (+0.0366)、Adaptive-Fusion (+0.0279) 均优于 Meta-only 的 unified nDCG，证实全局评测中的稀释效应
2. **Content-only 在 Tag 维度最强**（0.7472），但 Creator 维度极弱（0.0531），拖累 unified 分数
3. **Naive-Fusion 是最佳方法**（unified 0.4749），简单等权融合在该子集上优于 ρ-adaptive
4. **Per-doc 分析**：68.2% 的文档 Content > Meta（Tag nDCG），但 28.8% Content < Meta
5. **一致性与融合增益正相关**（r=0.20）：高一致性文档的融合增益更好
6. **全局 vs 子集差异巨大**：Meta-only 从全局 0.3064 提升到子集 0.4383（+0.1319），Naive-Fusion 从 0.3049 到 0.4749（+0.1700）

### 产物文件

| 文件 | 位置 | 说明 |
|------|------|------|
| `metrics_subset.csv` | `tmp/content/subset_eval/` | 8 方法 × 22 指标 |
| `per_doc_scores.parquet` | `tmp/content/subset_eval/` | 每文档每方法详细分数 |
| `fig_subset_comparison.png` | `tmp/content/subset_eval/` | 方法对比柱状图 |
| `fig_meta_vs_content_scatter.png` | `tmp/content/subset_eval/` | per-doc meta vs content scatter |
| `fig_consistency_vs_gain.png` | `tmp/content/subset_eval/` | 一致性 vs 融合增益 |

---

*最后更新：2026-02-01*
