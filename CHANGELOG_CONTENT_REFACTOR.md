# 变更说明：内容视图扩展 — 结构性重构

## 概述

本轮重构将内容视图（Content View）的实现从 notebook 内联代码抽取为独立 Python 模块，
并将相关 notebook 迁移至统一目录结构，提升代码复用性与可维护性。

---

## 1. `src/content/` 模块创建

从 notebook 中抽取核心逻辑，拆分为 6 个模块：

| 模块 | 职责 | 对应规格章节 |
|------|------|-------------|
| `__init__.py` | 子包入口，声明公开接口 | — |
| `sampling.py` | 表采样、列分析（profiling）与描述生成；支持 CSV/TSV/Parquet/XLSX | §2.5, §3.1–3.3 |
| `encoding.py` | Sentence-Transformer 嵌入编码，按类型加权聚合为数据集向量 | §3.4–3.5 |
| `similarity.py` | 稀疏相似度图构建（对称化 + L1 行归一化）、分区 COO 边文件 I/O | §3.6 |
| `consistency.py` | 元数据-内容一致性指标：Jaccard 集合重叠 J(i) 与加权一致性 c(i) | §4.1–4.2 |
| `fusion.py` | 多视图融合：ρ 自适应权重计算、一致性调整、top-K 裁剪 | §5.1–5.3 |

## 2. Notebook 迁移

3 个 notebook 从项目根目录迁移至 `notebooks/04_content/`，路径与导入语句已同步更新：

| 编号 | 文件名 | 内容 |
|------|--------|------|
| NB1 | `01_content_data_acquisition.ipynb` | 数据获取与本地化：候选集筛选、Kaggle API 下载、主表选择 |
| NB2 | `02_content_view_construction.ipynb` | 内容视图构建：列分析、描述生成、嵌入编码、相似度图、一致性计算 |
| NB3 | `03_content_fusion_experiments.ipynb` | 四视图融合实验：ρ-自适应加权、银标准评估、预算实验、消融分析 |

## 3. 预算扫描自动化

NB3 Cell 38 原为单点预算记录，现重构为自动化网格扫描：
遍历预设预算梯度，自动执行融合并收集评估指标，
消除手工逐点运行的繁琐流程。

## 4. `src/__init__.py` 更新

在包级文档字符串的 `Subpackages` 节新增 content 子包说明：

```
Subpackages:
    - content: Content view extension (table sampling, encoding, similarity,
               consistency, and multi-view fusion)
```

content 模块的导出保持隔离，notebook 中通过 `from src.content import ...` 按需引入。

---

## 涉及文件清单

**新增**
- `src/content/__init__.py`
- `src/content/sampling.py`
- `src/content/encoding.py`
- `src/content/similarity.py`
- `src/content/consistency.py`
- `src/content/fusion.py`
- `notebooks/04_content/01_content_data_acquisition.ipynb`
- `notebooks/04_content/02_content_view_construction.ipynb`
- `notebooks/04_content/03_content_fusion_experiments.ipynb`

**修改**
- `src/__init__.py` — 新增 content 子包文档
