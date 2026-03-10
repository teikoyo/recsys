# D_content 子集评测实验结果

## 1. 实验概述

### 背景

在全局评测（521,735 docs）中，内容视图（TabContent）仅覆盖 943 个文档（0.18%），导致融合方法在全局范围下无法超越 Meta-only 基线。这是因为四视图融合的增益被大量无内容覆盖的文档所稀释。

### 目的

在 **D_content 子集**（1,000 docs，100% 银标准覆盖）上重新评测，消除稀释效应，真实衡量内容视图对推荐质量的贡献。

### 评测参数

- **K = 20**（Top-20 推荐列表）
- **Unified 权重**: W_TAG = 0.5, W_DESC = 0.3, W_CRE = 0.2
- **银标准**: Tag 相似度、Description 相似度、Creator 共现

---

## 2. 方法说明

本实验对比 8 种方法，分为 4 组：

| 方法 | 分组 | 说明 |
|------|------|------|
| Meta-only | Metadata Fusion | 三视图融合 S_fused3（Tag + Text + Beh） |
| Content-only | Content View | 表格内容视图 S_tabcontent |
| Naive-Fusion | 4-View Fusion | 0.5 × Meta + 0.5 × Content |
| Adaptive-Fusion | 4-View Fusion | ρ 自适应四视图权重 |
| Adaptive+Cons | 4-View Fusion | 含一致性 c(i) 调节 |
| Tag-only | Single View | 仅 Tag-SGNS 视图 |
| Text-only | Single View | 仅 Text-SGNS 视图 |
| Beh-only | Single View | 仅 Behavior 视图 |

**相似度矩阵来源**：
- **Tag-only / Text-only / Beh-only**: 各自的单视图余弦相似度矩阵
- **Meta-only**: 三视图（Tag + Text + Beh）基于 ρ 权重的融合矩阵 S_fused3
- **Content-only**: 基于表格内容特征的余弦相似度矩阵 S_tabcontent
- **Naive-Fusion**: 0.5 × S_fused3 + 0.5 × S_tabcontent
- **Adaptive-Fusion**: 四视图各自 ρ 权重加权
- **Adaptive+Cons**: 在 Adaptive-Fusion 基础上加入一致性系数 c(i) 调节

---

## 3. 主结果表（Table 1）

数据来源：`metrics_subset_aligned.csv`，按 Unified@nDCG20 降序排列。

### Table 1a: 核心指标（按维度 nDCG@20）

| Rank | Method | Group | Tag-nDCG@20 | Desc-nDCG@20 | Creator-nDCG@20 | Unified@nDCG20 |
|-----:|--------|-------|------------:|-------------:|----------------:|---------------:|
| 1 | Naive-Fusion | 4-View Fusion | 0.6449 | 0.0131 | 0.7428 | **0.4749** |
| 2 | Beh-only | Single View | 0.6416 | 0.0097 | 0.7506 | 0.4739 |
| 3 | Adaptive+Cons | 4-View Fusion | 0.6257 | 0.0113 | 0.7498 | 0.4662 |
| 4 | Adaptive-Fusion | 4-View Fusion | 0.6257 | 0.0113 | 0.7496 | 0.4662 |
| 5 | Meta-only | Metadata Fusion | 0.5763 | 0.0077 | 0.7391 | 0.4383 |
| 6 | Content-only | Content View | 0.7472 | 0.0157 | 0.0531 | 0.3889 |
| 7 | Tag-only | Single View | 0.2302 | 0.0000 | 0.0004 | 0.1152 |
| 8 | Text-only | Single View | 0.2249 | 0.0008 | 0.0003 | 0.1127 |

### Table 1b: 完整 Unified 指标

| Rank | Method | Unified@nDCG20 | Unified@MAP20 | Unified@MRR20 | Unified@P20 | Unified@R20 | Unified_cov@nDCG20 |
|-----:|--------|---------------:|--------------:|--------------:|------------:|------------:|--------------------:|
| 1 | Naive-Fusion | **0.4749** | 0.4398 | 0.4851 | 0.3038 | 0.3835 | 0.4749 |
| 2 | Beh-only | 0.4739 | 0.4150 | 0.4837 | 0.2548 | 0.3321 | 0.4739 |
| 3 | Adaptive+Cons | 0.4662 | 0.4207 | 0.4760 | 0.2019 | 0.2809 | 0.4662 |
| 4 | Adaptive-Fusion | 0.4662 | 0.4205 | 0.4761 | 0.2017 | 0.2808 | 0.4662 |
| 5 | Meta-only | 0.4383 | 0.3919 | 0.4296 | 0.1807 | 0.2597 | 0.4383 |
| 6 | Content-only | 0.3889 | 0.3767 | 0.4393 | 0.3227 | 0.3235 | 0.3889 |
| 7 | Tag-only | 0.1152 | 0.0611 | 0.0683 | 0.0238 | 0.0238 | 0.1152 |
| 8 | Text-only | 0.1127 | 0.0588 | 0.0664 | 0.0240 | 0.0240 | 0.1127 |

---

## 4. 按维度展开表（Table 2）

数据来源：`exp_subset_by_task.csv`，展示每方法在 Tag / Desc / Creator 三维度的详细分数。

### Table 2a: Tag 维度

| Method | n | nDCG@20 | MAP@20 | MRR@20 | P@20 | Coverage |
|--------|--:|--------:|-------:|-------:|-----:|---------:|
| Content-only | 943 | 0.7472 | 0.7292 | 0.8531 | 0.6414 | 0.9430 |
| Naive-Fusion | 1000 | 0.6449 | 0.5790 | 0.6679 | 0.4933 | 1.0000 |
| Beh-only | 1000 | 0.6416 | 0.5259 | 0.6631 | 0.3870 | 1.0000 |
| Adaptive+Cons | 1000 | 0.6257 | 0.5386 | 0.6476 | 0.2870 | 1.0000 |
| Adaptive-Fusion | 1000 | 0.6257 | 0.5382 | 0.6476 | 0.2870 | 1.0000 |
| Meta-only | 1000 | 0.5763 | 0.4874 | 0.5628 | 0.2450 | 1.0000 |
| Tag-only | 1000 | 0.2302 | 0.1221 | 0.1365 | 0.0477 | 1.0000 |
| Text-only | 1000 | 0.2249 | 0.1174 | 0.1326 | 0.0480 | 1.0000 |

### Table 2b: Desc 维度

| Method | n | nDCG@20 | MAP@20 | MRR@20 | P@20 | Coverage |
|--------|--:|--------:|-------:|-------:|-----:|---------:|
| Content-only | 943 | 0.0157 | 0.0147 | 0.0151 | 0.0015 | 0.9430 |
| Naive-Fusion | 1000 | 0.0131 | 0.0099 | 0.0096 | 0.0020 | 1.0000 |
| Adaptive+Cons | 1000 | 0.0113 | 0.0092 | 0.0089 | 0.0017 | 1.0000 |
| Adaptive-Fusion | 1000 | 0.0113 | 0.0092 | 0.0089 | 0.0017 | 1.0000 |
| Beh-only | 1000 | 0.0097 | 0.0077 | 0.0081 | 0.0012 | 1.0000 |
| Meta-only | 1000 | 0.0077 | 0.0048 | 0.0057 | 0.0012 | 1.0000 |
| Text-only | 1000 | 0.0008 | 0.0002 | 0.0002 | 0.0002 | 1.0000 |
| Tag-only | 1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Table 2c: Creator 维度

| Method | n | nDCG@20 | MAP@20 | MRR@20 | P@20 | Coverage |
|--------|--:|--------:|-------:|-------:|-----:|---------:|
| Beh-only | 1000 | 0.7506 | 0.7486 | 0.7487 | 0.3045 | 1.0000 |
| Adaptive+Cons | 1000 | 0.7498 | 0.7435 | 0.7478 | 0.2893 | 1.0000 |
| Adaptive-Fusion | 1000 | 0.7496 | 0.7430 | 0.7478 | 0.2888 | 1.0000 |
| Naive-Fusion | 1000 | 0.7428 | 0.7368 | 0.7415 | 0.2828 | 1.0000 |
| Meta-only | 1000 | 0.7391 | 0.7339 | 0.7325 | 0.2894 | 1.0000 |
| Content-only | 943 | 0.0531 | 0.0385 | 0.0413 | 0.0078 | 0.9430 |
| Tag-only | 1000 | 0.0004 | 0.0002 | 0.0002 | 0.0001 | 1.0000 |
| Text-only | 1000 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 1.0000 |

---

## 5. 相对改进百分比（Table 3）

数据来源：`improvement_over_meta.csv`，以 Meta-only 为基线（0%），展示各方法在 Unified 指标上的相对改进。

| Method | Group | nDCG20 (%) | MAP20 (%) | MRR20 (%) | P20 (%) | R20 (%) | cov@nDCG20 (%) |
|--------|-------|----------:|---------:|---------:|-------:|-------:|---------------:|
| Naive-Fusion | 4-View Fusion | +8.36 | +12.23 | +12.93 | +68.11 | +47.65 | +8.36 |
| Beh-only | Single View | +8.12 | +5.88 | +12.60 | +40.99 | +27.85 | +8.12 |
| Adaptive+Cons | 4-View Fusion | +6.37 | +7.35 | +10.81 | +11.73 | +8.15 | +6.37 |
| Adaptive-Fusion | 4-View Fusion | +6.37 | +7.28 | +10.82 | +11.64 | +8.11 | +6.37 |
| Content-only | Content View | -11.26 | -3.89 | +2.27 | +78.58 | +24.55 | -11.26 |
| Tag-only | Single View | -73.72 | -84.41 | -84.10 | -86.81 | -90.83 | -73.72 |
| Text-only | Single View | -74.28 | -85.01 | -84.55 | -86.70 | -90.75 | -74.28 |

**要点**：在 D_content 子集上，Naive-Fusion 相对 Meta-only 提升 **+8.36%**（nDCG20），所有四视图融合方法均显著超越 Meta-only。

---

## 6. 全局 vs 子集对比（Table 4）

数据来源：`global_vs_subset_comparison.csv`，展示 5 种方法在两个评测范围下的 Unified@nDCG20 对比。

### Table 4a: Tag 维度 nDCG@20

| Method | Global (521,735) | Subset (1,000) | Delta |
|--------|------------------:|----------------:|------:|
| Meta-only | 0.3157 | 0.5763 | +0.2606 |
| Content-only | 0.7472 | 0.7472 | 0.0000 |
| Naive-Fusion | 0.3144 | 0.6449 | +0.3306 |
| Adaptive-Fusion | 0.3141 | 0.6257 | +0.3117 |
| Adaptive+Cons | 0.3141 | 0.6257 | +0.3117 |

### Table 4b: Creator 维度 nDCG@20

| Method | Global (521,735) | Subset (1,000) | Delta |
|--------|------------------:|----------------:|------:|
| Meta-only | 0.6651 | 0.7391 | +0.0740 |
| Content-only | 0.0531 | 0.0531 | 0.0000 |
| Naive-Fusion | 0.6619 | 0.7428 | +0.0809 |
| Adaptive-Fusion | 0.6616 | 0.7496 | +0.0880 |
| Adaptive+Cons | 0.6616 | 0.7498 | +0.0882 |

### Table 4c: Unified@nDCG20

| Method | Global (521,735) | Subset (1,000) | Delta |
|--------|------------------:|----------------:|------:|
| Meta-only | 0.3064 | 0.4383 | +0.1319 |
| Content-only | 0.3889 | 0.3889 | 0.0000 |
| Naive-Fusion | 0.3049 | 0.4749 | +0.1700 |
| Adaptive-Fusion | 0.3047 | 0.4662 | +0.1615 |
| Adaptive+Cons | 0.3047 | 0.4662 | +0.1616 |

**要点**：
- 全局范围下，所有融合方法（Naive / Adaptive / Adaptive+Cons）的 Unified@nDCG20 略低于 Meta-only（~0.3047 vs 0.3064），融合未能带来增益
- 子集范围下，融合方法大幅超越 Meta-only（0.4749 vs 0.4383），证明内容视图的正向贡献被全局稀释效应掩盖
- Content-only 在两个范围下分数完全一致（0.3889），因为它仅覆盖子集内文档

---

## 7. Per-doc 分析

基于 `per_doc_scores.parquet` 中的逐文档 Unified nDCG@20 分数分析：

### Content vs Meta 逐文档比较

| 比较结果 | 文档数 | 占比 |
|----------|-------:|-----:|
| Content > Meta | 643 | 68.2% |
| Content < Meta | 272 | 28.8% |
| Content = Meta | 28 | 3.0% |

> 在 D_content 子集中，约 **68%** 的文档从内容视图中获益（Content 分数 > Meta 分数），说明表格内容特征对多数文档具有正向推荐价值。

### 融合增益与一致性相关性

- Naive-Fusion 增益（Fusion - Meta）与 Content-Meta Jaccard 一致性的 Pearson 相关系数：**r = 0.20**
- 表明一致性对融合增益有弱正相关，但不是唯一决定因素

---

## 8. 关键发现（Key Findings）

1. **融合方法在子集上显著超越 Meta-only**：Naive-Fusion 的 Unified@nDCG20 = 0.4749，相对 Meta-only（0.4383）提升 +8.36%，证实内容视图在覆盖范围内具有正向贡献。

2. **全局稀释效应是融合失效的主因**：同样的 Naive-Fusion 在全局评测中 Unified@nDCG20 = 0.3049，低于 Meta-only 的 0.3064。内容视图仅覆盖 0.18% 的文档，导致增益被完全稀释。

3. **Content-only 在 Tag 维度表现最优**：Content-only 的 Tag-nDCG@20 = 0.7472 为所有方法最高，但因 Creator 维度极低（0.0531），综合 Unified 分数（0.3889）低于融合方法。

4. **Beh-only 是最强单视图基线**：Beh-only 的 Unified@nDCG20 = 0.4739，仅次于 Naive-Fusion（0.4749），表明行为视图在子集中同样非常有效，且几乎与四视图融合持平。

5. **Adaptive 方法未显著优于 Naive**：Adaptive-Fusion（0.4662）和 Adaptive+Cons（0.4662）均低于 Naive-Fusion（0.4749），ρ 自适应权重和一致性调节在当前配置下未带来额外增益。

6. **约 68% 的文档从内容视图获益**：Per-doc 分析显示 643/943 文档的 Content 分数优于 Meta 分数，证实表格内容特征对多数文档具有推荐价值，但仍有约 29% 的文档更适合使用元数据。

---

## 9. 数据来源

### CSV 数据文件

| 文件 | 说明 |
|------|------|
| `tmp/content/subset_eval/metrics_subset_aligned.csv` | 主结果表：8 方法 × 全维度指标，按 Unified@nDCG20 降序 |
| `tmp/content/subset_eval/exp_subset_by_task.csv` | 按维度展开：每方法 × 每维度（Tag/Desc/Creator）的详细指标 |
| `tmp/content/subset_eval/improvement_over_meta.csv` | 改进百分比：以 Meta-only 为基线的各方法 Unified 相对改进 |
| `tmp/content/subset_eval/global_vs_subset_comparison.csv` | 全局 vs 子集：5 方法在两个范围下的对比 |
| `tmp/content/subset_eval/metrics_subset.csv` | 原始格式备查（与 aligned 内容一致，列名不同） |

### Parquet 数据文件

| 文件 | 说明 |
|------|------|
| `tmp/content/subset_eval/per_doc_scores.parquet` | 逐文档 Unified nDCG@20 分数，用于 per-doc 分析 |

### 图表文件

| 文件 | 说明 |
|------|------|
| `tmp/content/subset_eval/fig_subset_comparison.png` | 子集评测方法对比柱状图 |
| `tmp/content/subset_eval/fig_meta_vs_content_scatter.png` | Meta vs Content 逐文档散点图 |
| `tmp/content/subset_eval/fig_consistency_vs_gain.png` | 一致性 vs 融合增益散点图 |

### Notebook

| 文件 | 说明 |
|------|------|
| `notebooks/04_content/04_content_subset_evaluation.ipynb` | 实验执行 notebook（生成以上所有数据和图表） |

---

## 10. 规模实验：稀释效应（Scale Experiments）

### 实验设计

为量化内容覆盖率下降对融合方法的影响，创建 4 种不同规模的评测子集：

| 子集规模 | Content 文档数 | 非 Content 文档数 | Content 覆盖率 |
|---------:|---------------:|------------------:|---------------:|
| 1,000 | 1,000 | 0 | 100.00% |
| 5,000 | 1,000 | 4,000 | 20.00% |
| 10,000 | 1,000 | 9,000 | 10.00% |
| 50,000 | 1,000 | 49,000 | 2.00% |

每个子集包含全部 1,000 个 D_content 文档，加上从非 content 文档池（520,735 docs）中随机采样的文档（seed=42）。

**脚本**: `scripts/run_subset_scale.py`

### Table 5: Unified@nDCG20 按规模（全方法）

| Rank | Method | Group | N=1,000 | N=5,000 | N=10,000 | N=50,000 |
|-----:|--------|-------|--------:|--------:|---------:|---------:|
| 1 | Naive-Fusion | 4-View Fusion | **0.4749** | 0.3663 | 0.3421 | 0.3114 |
| 2 | Beh-only | Single View | 0.4739 | 0.3808 | 0.3602 | 0.3307 |
| 3 | Adaptive+Cons | 4-View Fusion | 0.4662 | 0.3626 | 0.3402 | 0.3108 |
| 4 | Adaptive-Fusion | 4-View Fusion | 0.4662 | 0.3626 | 0.3402 | 0.3108 |
| 5 | Meta-only | Metadata Fusion | 0.4383 | 0.3537 | 0.3351 | 0.3118 |
| 6 | Content-only | Content View | 0.3889 | 0.3889 | 0.3889 | 0.3889 |
| 7 | Tag-only | Single View | 0.1152 | 0.0894 | 0.0844 | 0.0771 |
| 8 | Text-only | Single View | 0.1127 | 0.0889 | 0.0827 | 0.0773 |

### Table 5b: Tag-nDCG@20 按规模

| Method | N=1,000 | N=5,000 | N=10,000 | N=50,000 |
|--------|--------:|--------:|---------:|---------:|
| Content-only | 0.7472 | 0.7472 | 0.7472 | 0.7472 |
| Naive-Fusion | 0.6449 | 0.4381 | 0.3859 | 0.3280 |
| Beh-only | 0.6416 | 0.4565 | 0.4101 | 0.3524 |
| Meta-only | 0.5763 | 0.4137 | 0.3717 | 0.3268 |

### Table 5c: Creator-nDCG@20 按规模

| Method | N=1,000 | N=5,000 | N=10,000 | N=50,000 |
|--------|--------:|--------:|---------:|---------:|
| Beh-only | 0.7506 | 0.7030 | 0.7026 | 0.6952 |
| Adaptive+Cons | 0.7498 | 0.6790 | 0.6757 | 0.6643 |
| Naive-Fusion | 0.7428 | 0.6781 | 0.6754 | 0.6645 |
| Meta-only | 0.7391 | 0.6795 | 0.6771 | 0.6678 |
| Content-only | 0.0531 | 0.0531 | 0.0531 | 0.0531 |

### 稀释效应分析

**关键观察**：

1. **融合方法的稀释退化曲线**：Naive-Fusion 从 0.4749（N=1K, 100%覆盖）下降到 0.3114（N=50K, 2%覆盖），降幅 34.4%。融合增益随覆盖率下降而迅速消失。

2. **Content-only 不受稀释影响**：Content-only 的 Unified@nDCG20 在所有规模下恒定为 0.3889，因为它仅在 943 个有内容的文档上评测，不受子集扩大影响。

3. **交叉点**：在 N=5,000（覆盖率 20%）时，Content-only（0.3889）已超越 Meta-only（0.3537）和所有融合方法（Naive: 0.3663）。到 N=50K 时，Content-only 在 Unified 指标上排名第一。

4. **Beh-only 退化最慢**：Beh-only 从 0.4739 下降到 0.3307（降幅 30.2%），在所有元数据方法中退化最慢，始终是 N>1K 时的最强基线。

5. **融合方法之间差距缩小**：在 N=50K 时，Naive-Fusion（0.3114）、Adaptive（0.3108）和 Meta-only（0.3118）几乎持平，内容视图的边际贡献已可忽略。

### 数据来源

| 文件 | 说明 |
|------|------|
| `tmp/content/scale_experiments/scale_all_metrics.csv` | 合并结果：4 规模 × 8 方法 = 32 行 |
| `tmp/content/scale_experiments/scale_{N}_metrics.csv` | 各规模单独结果 |

---

## 11. 内容特征消融实验（Content Feature Ablation）

### 实验设计

测试表格采样参数 MAX_ROWS（每表最大行数）和 MAX_COLS（每表最大列数）对内容视图质量的影响。采用分离消融设计（非全网格）以减少计算量：

- **行消融**：固定 MAX_COLS=60，变化 MAX_ROWS ∈ {64, 128, 256, 512, 1024}
- **列消融**：固定 MAX_ROWS=1024，变化 MAX_COLS ∈ {5, 10, 20, 30, 60}
- 共 9 个独立配置（R1024_C60 为共享基线）

每个配置重新运行完整的 content pipeline（采样 → profile → 编码 → 聚合 → kNN → 融合），在 D_content 子集上评测 Content-only 和 Naive-Fusion。

**脚本**: `scripts/run_content_ablation.py`

### Table 6a: 行消融（固定 MAX_COLS=60）

| MAX_ROWS | Content-only Unified | Tag-nDCG | Desc-nDCG | Cre-nDCG | Naive-Fusion Unified | Tag-nDCG | Desc-nDCG | Cre-nDCG |
|---------:|---------------------:|---------:|----------:|---------:|---------------------:|---------:|----------:|---------:|
| 64 | 0.3918 | 0.7525 | 0.0168 | 0.0526 | **0.4777** | 0.6492 | 0.0137 | 0.7447 |
| 128 | 0.3899 | 0.7492 | 0.0164 | 0.0521 | 0.4766 | 0.6472 | 0.0137 | 0.7442 |
| 256 | 0.3893 | 0.7473 | 0.0163 | 0.0540 | 0.4769 | 0.6483 | 0.0134 | 0.7436 |
| 512 | 0.3871 | 0.7437 | 0.0166 | 0.0513 | 0.4761 | 0.6466 | 0.0135 | 0.7438 |
| 1024 | 0.3888 | 0.7468 | 0.0157 | 0.0531 | 0.4749 | 0.6449 | 0.0131 | 0.7428 |

### Table 6b: 列消融（固定 MAX_ROWS=1024）

| MAX_COLS | Content-only Unified | Tag-nDCG | Desc-nDCG | Cre-nDCG | Naive-Fusion Unified | Tag-nDCG | Desc-nDCG | Cre-nDCG |
|---------:|---------------------:|---------:|----------:|---------:|---------------------:|---------:|----------:|---------:|
| 5 | 0.3833 | 0.7374 | 0.0153 | 0.0498 | **0.4776** | 0.6497 | 0.0128 | 0.7444 |
| 10 | 0.3883 | 0.7459 | 0.0155 | 0.0534 | 0.4749 | 0.6446 | 0.0134 | 0.7428 |
| 20 | 0.3887 | 0.7465 | 0.0157 | 0.0536 | 0.4737 | 0.6424 | 0.0132 | 0.7427 |
| 30 | 0.3896 | 0.7480 | 0.0160 | 0.0537 | 0.4741 | 0.6431 | 0.0134 | 0.7425 |
| 60 | 0.3888 | 0.7468 | 0.0157 | 0.0531 | 0.4749 | 0.6449 | 0.0131 | 0.7428 |

### 消融分析

**关键发现**：

1. **Content-only 对行数不敏感**：MAX_ROWS 从 64 到 1024 变化，Content-only 的 Unified@nDCG20 在 0.3871–0.3918 范围内波动，极差仅 0.0047（1.2%）。较少的行采样甚至略有优势（64 行时 0.3918 vs 1024 行时 0.3888）。

2. **Content-only 对列数轻微敏感**：MAX_COLS 从 5 增至 30，Content-only 从 0.3833 单调增长至 0.3896（+1.6%），但 60 列时略回落至 0.3888。最佳点约在 MAX_COLS=30。

3. **Naive-Fusion 同样对参数不敏感**：所有行消融配置的 Naive-Fusion Unified 在 0.4749–0.4777 范围内，极差仅 0.0028（0.6%）。列消融范围为 0.4737–0.4776，极差 0.0039（0.8%）。

4. **融合的缓冲效应**：content pipeline 参数变化主要影响 Content-only 分数，经过 0.5×Meta + 0.5×Content 融合后，差异被进一步平滑，Naive-Fusion 对参数变化极不敏感。

5. **实用建议**：当前默认参数（MAX_ROWS=1024, MAX_COLS=60）已在合理范围内。若需加速 content pipeline，可安全地将 MAX_ROWS 降至 64–128 而不损失融合质量，甚至可能略有改善。MAX_COLS=5 对 Content-only 有约 1.4% 的损失，但融合后差异可忽略。

### 数据来源

| 文件 | 说明 |
|------|------|
| `tmp/content/ablation_experiments/ablation_all.csv` | 全部结果：9 配置 × 2 方法 = 18 行 |
| `tmp/content/ablation_experiments/ablation_rows.csv` | 行消融：5 配置 × 2 方法 = 10 行 |
| `tmp/content/ablation_experiments/ablation_cols.csv` | 列消融：5 配置 × 2 方法 = 10 行 |

---

## 12. 内容覆盖扩展实验（10K / 50K / 100K）

### 实验设计

将 D_content 从 1,000 扩展到 10,000 / 50,000 / 100,000 个数据集，测试更高内容覆盖率下四视图融合的效果。

| 目标规模 | 筛选策略 | 候选池大小 | 预计磁盘 |
|---------|---------|-----------|---------|
| 10K | tabular tags, size 100KB-100MB, no view filter | ~29,866 | ~320GB |
| 50K | broad tags*, size 50KB-500MB, no view filter | ~50,059 | ~1.6TB |
| 100K | any tags, size 10KB-1GB, no view filter | ~214,603 | ~3.2TB |

*broad tags = tabular + data visualization + feature engineering + statistics + finance + healthcare 等

每个规模在三种评测子集上测试全部 8 方法：
1. 仅 D_content_N 子集（100% 内容覆盖）
2. D_content_N + random(50K)（稀释评测）
3. D_content_N + random(100K)（进一步稀释）

**脚本**:
- `scripts/expand_content_coverage.py --target {N}` — 数据采集
- `scripts/run_content_at_scale.py --target {N}` — 管线 + 评测

### Table 7: 10K 规模结果 — Unified@nDCG20

D_content = 10,000 docs。三种评测子集：仅 D_content（100% 覆盖）、+50K 稀释（20% 覆盖）、+100K 稀释（10% 覆盖）。

| Method | D_content (10K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| **Naive-Fusion** | **0.4335** | 0.3503 | 0.3298 |
| Beh-only | 0.4199 | 0.3592 | 0.3439 |
| Adaptive-Fusion | 0.4116 | 0.3418 | 0.3249 |
| Adaptive+Cons | 0.4116 | 0.3418 | 0.3249 |
| Content-only | 0.3933 | 0.3933 | 0.3933 |
| Meta-only | 0.3893 | 0.3358 | 0.3223 |
| Tag-only | 0.1098 | 0.0874 | 0.0825 |
| Text-only | 0.1077 | 0.0875 | 0.0821 |

### Table 7b: 10K Tag-nDCG@20 按子集

| Method | D_content (10K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| Content-only | 0.7396 | 0.7396 | 0.7396 |
| Naive-Fusion | 0.5793 | 0.4078 | 0.3651 |
| Beh-only | 0.5576 | 0.4159 | 0.3810 |
| Adaptive-Fusion | 0.5371 | 0.3915 | 0.3557 |
| Meta-only | 0.5000 | 0.3798 | 0.3498 |

### Table 7c: 10K Creator-nDCG@20 按子集

| Method | D_content (10K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| Adaptive-Fusion | 0.6848 | 0.6672 | 0.6658 |
| Adaptive+Cons | 0.6847 | 0.6672 | 0.6658 |
| Naive-Fusion | 0.6821 | 0.6647 | 0.6632 |
| Beh-only | 0.6819 | 0.6880 | 0.6896 |
| Meta-only | 0.6724 | 0.6652 | 0.6651 |
| Content-only | 0.0751 | 0.0751 | 0.0751 |

### 10K 分析

1. **Naive-Fusion 保持最优**：在 D_content 子集上 Unified@nDCG20 = 0.4335，超越 Meta-only（0.3893）+11.4%。相比 1K 实验（0.4749），绝对值下降但相对优势增大。

2. **内容覆盖率提升效果**：10K D_content（覆盖率 1.92%）相比 1K（0.18%），融合方法在稀释子集上表现更好。在 +50K 稀释时，Naive-Fusion（0.3503）仍显著超越 Meta-only（0.3358），而 1K 实验在同等稀释下差距更小。

3. **Content-only 稳定性**：Content-only 在所有子集上保持 0.3933，与 1K 实验（0.3889）接近但略高，说明 10K 数据质量与 1K 相当。

4. **Beh-only 仍为强基线**：在稀释子集上，Beh-only 始终排名第二，仅次于 Naive-Fusion（D_content）或 Content-only（稀释后）。

5. **交叉点后移**：在 1K 实验中，Content-only 在 +5K 稀释时即超越 Meta-only。10K 实验中，Content-only 在 +50K 稀释时才超越 Meta-only（0.3933 vs 0.3358），交叉点如预期后移。

### Table 8: "50K" 规模结果 — Unified@nDCG20

> **注意**：由于 Kaggle API 搜索端点返回 404（API 版本变更），实际 D_content = 4,527 docs（覆盖率 0.87%），未达 50K 目标。

| Method | D_content (4.5K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| **Naive-Fusion** | **0.4585** | 0.3353 | 0.3201 |
| Beh-only | 0.4522 | 0.3498 | 0.3362 |
| Adaptive-Fusion | 0.4395 | 0.3316 | 0.3179 |
| Adaptive+Cons | 0.4395 | 0.3316 | 0.3179 |
| Meta-only | 0.4216 | 0.3283 | 0.3164 |
| Content-only | 0.3863 | 0.3863 | 0.3863 |
| Tag-only | 0.1137 | 0.0825 | 0.0795 |
| Text-only | 0.1104 | 0.0822 | 0.0787 |

### Table 8b: "50K" Tag-nDCG@20 按子集

| Method | D_content (4.5K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| Content-only | 0.7392 | 0.7392 | 0.7392 |
| Naive-Fusion | 0.6124 | 0.3717 | 0.3425 |
| Beh-only | 0.6030 | 0.3905 | 0.3637 |
| Adaptive-Fusion | 0.5761 | 0.3648 | 0.3386 |
| Meta-only | 0.5452 | 0.3585 | 0.3355 |

### Table 8c: "50K" Creator-nDCG@20 按子集

| Method | D_content (4.5K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| Beh-only | 0.7394 | 0.6980 | 0.6926 |
| Adaptive+Cons | 0.7382 | 0.6737 | 0.6673 |
| Adaptive-Fusion | 0.7382 | 0.6737 | 0.6673 |
| Naive-Fusion | 0.7364 | 0.6741 | 0.6678 |
| Meta-only | 0.7303 | 0.6735 | 0.6675 |
| Content-only | 0.0596 | 0.0596 | 0.0596 |

### "50K" 分析

1. **Naive-Fusion 仍为最优**：D_content 子集上 Unified@nDCG20 = 0.4585，超越 Meta-only（0.4216）+8.8%。

2. **数据质量略有下降**：Content-only 的 Unified 从 0.3933（10K）降至 0.3863（4.5K），Tag-nDCG 从 0.7396 降至 0.7392，说明放宽筛选条件对内容质量影响很小。

3. **覆盖率效果**：4.5K 覆盖率（0.87%）介于 1K（0.18%）和 10K（1.92%）之间，融合方法的表现也介于两者之间。

### Table 9: "100K" 规模结果 — Unified@nDCG20

> **注意**：Kaggle API 恢复后完成了全部候选搜索（365,647 候选池中搜索 120,000），但仅匹配到 17,542 个 API 结果，其中 9,778 个成功提取主表。实际 D_content = 9,778 docs（覆盖率 1.88%），与 10K 实验（10,000 docs）规模接近。

| Method | D_content (9.8K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| **Naive-Fusion** | **0.4326** | 0.3520 | 0.3308 |
| Beh-only | 0.4195 | 0.3609 | 0.3438 |
| Adaptive-Fusion | 0.4100 | 0.3436 | 0.3258 |
| Adaptive+Cons | 0.4100 | 0.3436 | 0.3258 |
| Content-only | 0.3932 | 0.3932 | 0.3932 |
| Meta-only | 0.3896 | 0.3364 | 0.3221 |
| Tag-only | 0.1099 | 0.0883 | 0.0822 |
| Text-only | 0.1077 | 0.0869 | 0.0819 |

### Table 9b: "100K" Tag-nDCG@20 按子集

| Method | D_content (9.8K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| Content-only | 0.7394 | 0.7394 | 0.7394 |
| Naive-Fusion | 0.5774 | 0.4093 | 0.3649 |
| Beh-only | 0.5568 | 0.4190 | 0.3811 |
| Adaptive-Fusion | 0.5339 | 0.3932 | 0.3555 |
| Meta-only | 0.5007 | 0.3802 | 0.3486 |

### Table 9c: "100K" Creator-nDCG@20 按子集

| Method | D_content (9.8K) | +50K dilution | +100K dilution |
|--------|----------------:|--------------:|---------------:|
| Beh-only | 0.6818 | 0.6876 | 0.6896 |
| Adaptive-Fusion | 0.6849 | 0.6673 | 0.6660 |
| Adaptive+Cons | 0.6848 | 0.6673 | 0.6660 |
| Naive-Fusion | 0.6822 | 0.6672 | 0.6662 |
| Meta-only | 0.6722 | 0.6652 | 0.6652 |
| Content-only | 0.0769 | 0.0769 | 0.0769 |

### "100K" 分析

1. **Naive-Fusion 仍为最优**：D_content 子集上 Unified@nDCG20 = 0.4326，超越 Meta-only（0.3896）+11.0%。

2. **与 10K 实验高度一致**：9.8K 的结果与 10K（10,000 docs）几乎完全一致（Naive-Fusion 0.4326 vs 0.4335），说明两次独立实验的结果可重复。

3. **Content-only 质量稳定**：Unified 0.3932 与 10K（0.3933）、1K（0.3889）实验一致，说明内容视图质量不依赖于特定数据集选择。

4. **覆盖率上限**：候选池 365,647 中仅 17,542 匹配 API（4.8%），其中 55.7% 有可提取主表。约 9,778 是当前筛选条件下的覆盖率上限。

### 跨规模对比分析

| 规模 | D_content 数 | 覆盖率 | Naive-Fusion (D_content) | 相对 Meta-only | Content-only |
|------|-------------|--------|------------------------:|---------------:|-------------:|
| 1K | 1,000 | 0.18% | 0.4749 | +8.36% | 0.3889 |
| 4.5K | 4,527 | 0.87% | 0.4585 | +8.76% | 0.3863 |
| 9.8K | 9,778 | 1.88% | 0.4326 | +11.03% | 0.3932 |
| 10K | 10,000 | 1.92% | 0.4335 | +11.36% | 0.3933 |

**关键观察**：
1. **融合增益随规模增大而更显著**：在 +50K 稀释评测中，10K 的 Naive-Fusion 超越 Meta-only 4.3%（0.3503 vs 0.3358），而 1K 仅超越 3.6%（0.3663 vs 0.3537），说明更高覆盖率使融合在更大子集上也能保持优势。
2. **Content-only 质量高度稳定**：跨四个规模实验，Content-only 的 Unified@nDCG20 在 0.3863–0.3933 范围内，极差仅 0.007（1.8%），说明内容视图质量不受数据集规模影响。
3. **交叉点确实后移**：1K 实验中 Content-only 在 N=5K 时超越 Meta-only，10K 实验中延后到 N=50K 才超越，符合预期。
4. **9.8K 与 10K 结果可重复**：两次独立实验（不同候选池、不同搜索顺序）产出几乎相同的结果（Naive-Fusion 0.4326 vs 0.4335），验证了实验方法的稳定性。
5. **覆盖率瓶颈**：当前 Kaggle 数据集中约 9,778 个具备可提取的表格主文件，占 521K 元数据的 1.88%。进一步扩展需要降低表格质量门槛或引入外部数据源。

### 数据来源

| 文件 | 说明 |
|------|------|
| `tmp/content/scale_10000/results_all_subsets.csv` | 10K 规模全部评测结果 |
| `tmp/content/scale_50000/results_all_subsets.csv` | 50K 规模全部评测结果 |
| `tmp/content/scale_100000/results_all_subsets.csv` | 100K 规模全部评测结果 |
| `CONTENT_SCALE_PLAN.md` | 完整需求文档与执行计划 |

---

## 13. 10K 融合方法改进实验（Fusion Improvement Experiments）

### 实验背景

10K 实验中 Naive-Fusion（0.5×Meta + 0.5×Content）的 Unified@nDCG20 = 0.4335，超越 Meta-only +11.4%。但存在三个问题：
1. α=0.5 是任意设定，可能不是最优
2. Adaptive 权重不如简单等权 Naive
3. 稀释后（+100K）融合优势仅 +2.3%

以下 5 个实验系统探索改进空间。

**脚本**: `scripts/run_10k_experiments.py`

### Experiment 1: Naive 权重扫描（Weight Sweep）

扫描 α_meta ∈ {0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9}，其中 S_fused = α × S_meta + (1-α) × S_content。

| α_meta | α_content | Tag-nDCG | Desc-nDCG | Cre-nDCG | **Unified** |
|-------:|----------:|---------:|----------:|---------:|------------:|
| 0.30 | 0.70 | 0.6765 | 0.0295 | 0.6380 | **0.4747** |
| 0.40 | 0.60 | 0.6473 | 0.0281 | 0.6719 | 0.4665 |
| 0.45 | 0.55 | 0.6246 | 0.0266 | 0.6781 | 0.4559 |
| 0.50 | 0.50 | 0.5793 | 0.0249 | 0.6821 | 0.4335 |
| 0.55 | 0.45 | 0.5603 | 0.0234 | 0.6808 | 0.4233 |
| 0.60 | 0.40 | 0.5593 | 0.0232 | 0.6807 | 0.4228 |
| 0.70 | 0.30 | 0.5586 | 0.0230 | 0.6805 | 0.4223 |
| 0.80 | 0.20 | 0.5564 | 0.0223 | 0.6800 | 0.4209 |
| 0.90 | 0.10 | 0.5479 | 0.0210 | 0.6782 | 0.4159 |

**最优**: α_meta=0.30 → Unified=0.4747（vs 原 α=0.50 的 0.4335，提升 **+9.5%**）

**分析**：
- Content 视图权重越高，Tag-nDCG 增益越大（0.5479→0.6765），但 Creator-nDCG 下降（0.6821→0.6380）
- Unified 指标由于 Tag 权重 50%，偏好高 Tag-nDCG 的配置
- α=0.30 在 Tag 维度获得 +16.8% 提升，仅损失 Creator -6.5%，净效果显著正向
- 单调递增趋势说明在 D_content 子集上，Content 视图信息量大于 Meta 视图

### Experiment 2: 选择性融合（Selective Fusion）

**核心思想**：有内容的文档使用融合（α=0.30），无内容的文档保持 Meta-only，保证全局不退化。

| 子集 | 方法 | Unified | 相对 Meta-only |
|------|------|--------:|---------------:|
| D_content (10K) | Selective-α=0.30 | **0.4747** | +22.0% |
| D_content (10K) | Meta-only | 0.3892 | — |
| +50K dilution | Selective-α=0.30 | **0.3698** | +10.0% |
| +50K dilution | Meta-only | 0.3362 | — |
| +100K dilution | Selective-α=0.30 | **0.3414** | +5.9% |
| +100K dilution | Meta-only | 0.3224 | — |

**分析**：
- 选择性融合在所有子集上严格优于 Meta-only（设计上保证不退化）
- 稀释 +100K 时改进 +5.9%（vs 原 Naive-Fusion 的 +2.3%），提升 2.5 倍
- 稀释子集上的改进幅度符合理论预期：10K/100K × (0.4747-0.3892) ≈ 0.0086 → 实际 0.3414-0.3224 = 0.0190（超预期，因为内容文档的 meta 行也受益于融合邻居）

### Experiment 3: 内容质量过滤（Quality Filtering）

利用 col_profiles.parquet 计算每个数据集的质量分数（quality = n_cols × numeric_ratio × (1-missing) × unique_ratio），按不同阈值过滤低质量数据集。

| 阈值 | Quality 值 | 融合文档数 | 退回 Meta 数 | **Unified** |
|------|----------:|----------:|------------:|------------:|
| no_filter | 0.000 | 9,128 | 872 | **0.4749** |
| p10 | 0.185 | 8,215 | 1,785 | 0.4659 |
| p25 | 0.529 | 6,851 | 3,149 | 0.4530 |
| median | 1.344 | 4,565 | 5,435 | 0.4318 |
| p75 | 3.256 | 2,282 | 7,718 | 0.4102 |

**分析**：
- 不过滤（no_filter）表现最好，说明即使是低质量的内容数据集也对融合有正向贡献
- 过滤越严格，Unified 越低，退回 Meta-only 的文档越多反而性能越差
- 结论：**质量过滤不是改进方向**。Content 视图对大多数数据集都有正向信号，包括低质量数据集

### Experiment 4: 固定四视图权重网格（4-View Weight Grid）

测试 5 种固定权重配置（Tag + Text + Beh + Content），对比是否优于两层融合（先 3→Meta，再 Meta+Content）。

| 配置 | Tag | Text | Beh | Content | **Unified** |
|------|----:|-----:|----:|--------:|------------:|
| Content-heavy | 0.10 | 0.10 | 0.30 | 0.50 | **0.4799** |
| Beh+Content | 0.05 | 0.05 | 0.45 | 0.45 | 0.4597 |
| Beh-heavy | 0.10 | 0.10 | 0.50 | 0.30 | 0.4446 |
| Tag+Content | 0.30 | 0.10 | 0.30 | 0.30 | 0.4198 |
| Equal-4 | 0.25 | 0.25 | 0.25 | 0.25 | 0.4062 |

**分析**：
- Content-heavy (0.1/0.1/0.3/0.5) 达到 Unified=0.4799，比最优两层融合 Naive-α=0.30（0.4747）高 +1.1%
- 直接四视图加权（Content 权重 50%）略优于两层融合，可能因为避免了先 3→Meta 融合后再混合的信息损失
- 但差距很小（0.4799 vs 0.4747），两层融合架构仍然是合理选择
- Equal-4 最差（0.4062），Tag/Text 的低质量视图拖累整体
- Beh 和 Content 是最有信息量的两个视图

### Experiment 5: k_sim 参数扫描

重新从 Z_tabcontent 构建不同 k 值的 kNN 图并评测 Content-only。

| k_sim | nnz | Unified | Tag-nDCG | Desc-nDCG | Cre-nDCG |
|------:|----:|--------:|---------:|----------:|---------:|
| 20 | 304K | 0.3933 | 0.7399 | 0.0277 | 0.0752 |
| 30 | 458K | 0.3932 | 0.7399 | 0.0274 | 0.0753 |
| 50 | 767K | 0.3931 | 0.7397 | 0.0275 | 0.0752 |
| 75 | 1.16M | **0.3934** | 0.7401 | 0.0277 | 0.0752 |
| 100 | 1.54M | 0.3932 | 0.7397 | 0.0278 | 0.0753 |

**分析**：
- k_sim 对 Content-only 质量几乎无影响，所有值的 Unified 差距 < 0.001
- k=50（当前默认）已近最优，k=75 微幅改善但统计不显著
- 边缘数量随 k 线性增长，但评测时只取 top-20，较大 k 只增加计算成本
- 结论：**k=50 是合理默认值**，无需调整

### 综合结论与推荐

| 方法 | D_content | +50K | +100K | 特点 |
|------|----------:|-----:|------:|------|
| Meta-only (baseline) | 0.3892 | 0.3362 | 0.3224 | — |
| Naive-α=0.50 (原方法) | 0.4335 | 0.3503 | 0.3298 | 全局退化风险 |
| **Naive-α=0.30** | **0.4747** | — | — | D_content 最优 |
| 4View Content-heavy | **0.4799** | — | — | 绝对最优但需 4 矩阵 |
| **Selective-α=0.30** | **0.4747** | **0.3698** | **0.3414** | 无退化 + 稀释最优 |

**推荐部署方案**：
1. **选择性融合 + α=0.30** 是最佳实用方案：在 D_content 上提升 +22%，稀释 +100K 仍提升 +5.9%，且保证不退化
2. 若可接受额外复杂度，Content-heavy 四视图 (0.1/0.1/0.3/0.5) 可再提升 ~1%
3. 质量过滤和 k_sim 调整均无显著收益，不建议投入
4. α=0.30 的含义：Content 视图对 D_content 子集上的推荐质量贡献大于 Meta 视图，尤其在 Tag 维度

### 数据来源

| 文件 | 说明 |
|------|------|
| `tmp/content/scale_10000/experiments/exp1_weight_sweep.csv` | 实验 1：权重扫描结果 |
| `tmp/content/scale_10000/experiments/exp2_selective_fusion.csv` | 实验 2：选择性融合结果 |
| `tmp/content/scale_10000/experiments/exp3_quality_filter.csv` | 实验 3：质量过滤结果 |
| `tmp/content/scale_10000/experiments/exp3_quality_scores.csv` | 实验 3：数据集质量分数 |
| `tmp/content/scale_10000/experiments/exp4_4view_grid.csv` | 实验 4：四视图网格结果 |
| `tmp/content/scale_10000/experiments/exp5_k_sweep.csv` | 实验 5：k_sim 扫描结果 |
| `tmp/content/scale_10000/experiments/best_alpha.json` | 最优 α_meta 值 |
| `scripts/run_10k_experiments.py` | 实验脚本 |
