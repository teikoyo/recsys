# WS-SGNS: 基于元数据的多视图数据集推荐系统

## 项目简介

WS-SGNS（Walk-SGNS）是一个基于随机游走和 Skip-gram with Negative Sampling 的多视图数据集推荐系统。核心思想是：**仅利用数据集的元数据（标题、描述、标签、创建者等），不涉及数据集的实际内容**，通过多视图融合为用户推荐相似数据集。

系统在 521,735 个 Kaggle 数据集上验证，最佳融合方法（Fused3-Blend）达到 Unified@nDCG = 0.468，相比简单融合基线 CombSUM 提升 179%。

## 数据来源：Meta Kaggle

使用 Kaggle 官方发布的 **Meta Kaggle** 数据集，包含四张原始表：

| 文件 | 内容 | 关键字段 |
|------|------|---------|
| Datasets.csv | 数据集元信息 | Id, CreatorUserId, OwnerOrganizationId, TotalViews/Downloads/Votes |
| DatasetVersions.csv | 版本级标题/描述 | DatasetId, Title, Description |
| DatasetTags.csv | 数据集-标签映射 | DatasetId, TagId |
| Tags.csv | 标签定义 | Id, Name, FullPath |

四张表合并清洗后生成 521,735 条文档记录。**重要**：仅使用元数据字段，不解析任何数据集的实际文件内容。

## 核心研究问题

> 能否仅基于数据集的元数据（标签、文本描述、创建者/组织关系），通过多视图融合，实现高质量的数据集相似性推荐？

## 方法概览：三视图 + 融合

```
元数据 ──→ 三个独立视图 ──→ 各视图独立建模 ──→ 融合推荐
```

| 视图 | 元数据来源 | 图结构 | 建模方式 |
|------|-----------|--------|---------|
| **Tag** | DatasetTags + Tags (PPMI 加权) | 二部图 D→T→D | 随机游走 + SGNS |
| **Text** | Title + Description (BM25 加权) | 二部图 D→W→D | 随机游走 + SGNS |
| **Behavior** | CreatorUserId / OwnerOrganizationId | 直接邻接图 D→D | 相似度矩阵 |

**融合策略**：Fused3-RA（自适应行归一化） → Fused3-RR（重排序） → Fused3-Blend（混合）

## 评测体系

基于元数据构建三维度银标准（Silver Standard），无需人工标注：

| 维度 | 相关性定义 | 度量方法 | 覆盖率 |
|------|-----------|---------|--------|
| Tag-Relevance | 共享标签 | IDF 加权 Jaccard | 41.1% |
| Desc-Relevance | 描述文本相似 | BM25 余弦相似度 | 79.9% |
| Creator-Relevance | 同一创建者 | Binary 匹配 | 77.6% |

统一指标：Unified@nDCG = 0.5 × Tag + 0.3 × Desc + 0.2 × Creator

## 项目结构与文档导航

```
recsys-new/
├── src/                              # 核心代码
│   ├── sgns_model.py                 # SGNS 模型定义
│   ├── random_walk.py                # GPU 二部图随机游走
│   ├── sampling_utils.py             # O(1) 别名负采样
│   ├── pair_batch_utils.py           # Skip-gram 配对生成
│   ├── csr_utils.py                  # 稀疏矩阵工具
│   ├── ddp_utils.py                  # DDP 分布式训练
│   └── metrics.py                    # nDCG/MAP/MRR 等评测指标
│
├── data/raw_data/                    # Meta Kaggle 原始四表
├── tmp/                              # 中间结果与嵌入
│
├── docs/                             # 详细文档
│   ├── (已移至根目录 RESEARCH_PROPOSAL.md)
│   ├── EXPERIMENT_PLAN.md            # 实验设计方案
│   ├── EXPERIMENT_SUMMARY.md         # 实验结果摘要
│   ├── METRICS_EXPLANATION.md        # 评测指标详解
│   ├── SILVER_STANDARD_DESIGN.md     # 银标准设计原理
│   ├── DDP_HYBRID_GUIDE.md           # DDP 分布式训练指南
│   ├── QUICK_REFERENCE.md            # 快速参考
│   ├── DETAILED_RESULTS_ANALYSIS.md  # 详细结果分析
│   ├── EXPERIMENT_ANALYSIS.md        # 实验分析
│   ├── paper_presentation_methods.md # 论文方法呈现
│   └── archive/                      # 归档 (调试/修复类文档)
│
├── step6_ddp.py                      # DDP 训练脚本
├── README.md                         # 项目快速入门
└── RESEARCH_OVERVIEW.md              # 本文件 (研究概述入口)
```

### 快速入门

```bash
# 训练双视图嵌入
torchrun --nproc_per_node=2 step6_ddp.py --epochs 4 --dim 256 --neg 10 --amp true

# 构建 ANN 索引
python scripts/build_ann_index.py --k 50 --use_gpu true
```

详细说明请参阅 [RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md)。
