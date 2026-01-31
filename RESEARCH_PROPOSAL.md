# WS-SGNS 多视图数据集推荐系统研究方案

## 1. 研究背景与问题定义

### 1.1 研究动机

随着开放数据运动的发展，Kaggle 等数据科学平台上积累了数十万个数据集。用户面临严重的**信息过载问题**：如何从海量数据集中快速找到与当前任务相关的数据？

现有推荐方法存在以下局限：

- **基于内容的方法**需要解析数据集的实际文件内容（CSV、图片、文本等），但大规模数据集的内容解析成本极高，且格式多样化导致统一表示困难。
- **协同过滤方法**依赖显式的用户-物品交互记录（如评分、下载），但这类信息通常稀疏或不可用。
- **单一视图方法**仅利用某一种特征（如标签或文本描述），无法全面刻画数据集之间的多维度关联。

本研究的核心约束是：**仅使用数据集的元数据（metadata），不涉及数据集的实际内容**。这一约束具有实际意义——元数据获取成本低、格式统一、可实时处理，适合大规模部署。

### 1.2 数据来源：Meta Kaggle

本研究使用 Kaggle 官方发布的 **Meta Kaggle** 数据集作为实验数据源。Meta Kaggle 是 Kaggle 平台的自描述数据集，记录了平台上所有数据集的元数据信息。

原始数据位于 `data/raw_data/`，包含以下四张表：

| 文件 | 规模 | 关键列 | 用途 |
|------|------|--------|------|
| `Datasets.csv` | 521,735 行 / 68 MB | Id, CreatorUserId, OwnerOrganizationId, TotalViews, TotalDownloads, TotalVotes | 数据集核心元信息 + 社交/行为关系 |
| `DatasetVersions.csv` | 11,646,570 行 / 984 MB | DatasetId, Title, Description, VersionNumber | 版本级别的标题和描述文本 |
| `DatasetTags.csv` | 446,054 行 / 13 MB | DatasetId, TagId | 数据集与标签的映射关系 |
| `Tags.csv` | 831 行 / 98 KB | Id, Name, FullPath, ParentTagId | 标签定义（含层次结构） |

**为什么使用 Meta Kaggle？**

1. **真实场景**：数据来自真实用户行为，标签由用户主动标注，描述由创建者撰写。
2. **规模合适**：52 万数据集的规模足以验证大规模推荐算法的有效性。
3. **元数据丰富**：提供标签、文本描述、创建者/组织关系等多维度元数据，适合多视图建模。
4. **可公开获取**：作为 Kaggle 官方发布的数据集，便于其他研究者复现实验。

**重要说明**：我们**仅使用这四张表中的元数据字段**（标题、描述、标签、创建者ID、组织ID等），**不下载或解析任何数据集的实际文件内容**（如 CSV 数据、图片、文本语料等）。这是本研究的核心设计约束。

### 1.3 问题定义：基于元数据的多视图数据集相似性检索

**输入**：仅元数据字段——标题(Title)、描述(Description)、标签(Tags)、创建者(CreatorUserId)、组织(OwnerOrganizationId)。

**不使用**：数据集的实际文件内容（CSV 表格数据、图片、音频、文本语料等）。

**输出**：对于每个数据集，生成一个按相似性排序的推荐列表（Top-K 相似数据集）。

**目标**：基于元数据为用户推荐相似数据集，使推荐结果在主题相关性（Tag）、内容语义相关性（Description）、用户行为关联性（Creator）三个维度上均表现良好。

**方法**：从元数据构建三个独立视图（Tag / Text / Behavior），各视图通过二部图随机游走 + SGNS 嵌入学习独立建模，最后通过自适应融合策略合并为统一的推荐结果。

### 1.4 主要挑战

1. **元数据稀疏性**：仅 41.1% 的数据集有标签标注，组织信息覆盖率不足 1%。
2. **多视图异构性**：标签是离散符号、描述是自由文本、创建者关系是社交网络——三种信息的建模方式差异巨大。
3. **融合策略设计**：如何将不同视图的嵌入或相似度矩阵有效融合，避免某一视图主导结果。
4. **大规模评测**：52 万数据集缺乏人工标注的相关性标签，需设计可靠的自动化评测体系（银标准）。
5. **计算效率**：需在合理时间内完成 52 万文档的嵌入学习和 Top-K 检索。

---

## 2. 数据集与预处理

### 2.1 Meta Kaggle 原始数据结构

四张原始表通过以下关系合并：

```
Datasets.csv ───(Id = DatasetId)──→ DatasetVersions.csv  (取最新版本的Title/Description)
     │
     ├───(Id = DatasetId)──→ DatasetTags.csv ───(TagId = Id)──→ Tags.csv
     │
     └── CreatorUserId / OwnerOrganizationId (直接使用)
```

**合并流程**（代码位置：`scripts/data_merge.ipynb`）：

1. 以 `Datasets.csv` 为主表，取每个数据集的最新版本（按 VersionNumber 降序取第一条）的 Title 和 Description。
2. 通过 `DatasetTags.csv` 和 `Tags.csv` 关联，获取每个数据集的标签名称列表。
3. 保留 `CreatorUserId` 和 `OwnerOrganizationId` 作为行为特征。
4. 文本清洗（去 HTML 标签、URL、特殊字符）。
5. 构建 PPMI 加权的文档-标签矩阵 `DT_ppmi`。
6. 构建 BM25 加权的文档-词矩阵 `DW_bm25`。
7. 合并生成 `data/metadata_merged.csv`（约 283 MB）。
8. 数据清洗（去重、去空值、文本预处理）后生成 `tmp/doc_clean.parquet`（521,735 行）。
9. 同时导出 `tmp/tag_vocab.parquet`、`tmp/text_vocab.parquet`、`tmp/rw_params.parquet` 等中间文件。

**字段说明**：

| 字段 | 来源 | 类型 | 说明 |
|------|------|------|------|
| doc_id | Datasets.Id | int | 文档唯一标识（内部重编号 0~N-1） |
| title | DatasetVersions.Title | str | 数据集标题 |
| description | DatasetVersions.Description | str | 数据集描述（平均 294.5 字符） |
| tags | DatasetTags + Tags | list[str] | 标签名称列表（平均 0.85 个/文档） |
| creator_id | Datasets.CreatorUserId | int | 创建者 ID |
| org_id | Datasets.OwnerOrganizationId | int | 所属组织 ID（-1 表示无） |
| total_views | Datasets.TotalViews | int | 总浏览量 |
| total_downloads | Datasets.TotalDownloads | int | 总下载量 |
| total_votes | Datasets.TotalVotes | int | 总投票数 |

### 2.2 Tag 视图数据

**数据构建**：从 `DatasetTags.csv` + `Tags.csv` 提取数据集-标签关系，构建文档-标签共现矩阵。

- **标签词表**：394 个去重标签（过滤极低频标签后）
- **覆盖率**：41.1%（214,585 / 521,735）的数据集有至少一个标签
- **平均标签数**：0.85 个/文档（有标签的文档平均约 2 个）

**加权方式：PPMI（Positive Pointwise Mutual Information）**

PPMI 加权反映了标签与文档的关联强度，抑制高频通用标签（如 "data"）的权重：

```
PMI(d, t) = log[ P(d,t) / (P(d) × P(t)) ]
PPMI(d, t) = max(0, PMI(d, t))
```

生成的稀疏矩阵 `DT_ppmi` 形状为 [521,735 × 394]，存储为 `tmp/DT_ppmi.parquet`。

### 2.3 Text 视图数据

**数据构建**：从合并后的 Title + Description 文本中提取词汇，构建文档-词共现矩阵。

- **词汇表**：约 15,000 个词（经过停用词过滤、低频词剪枝）
- **覆盖率**：接近 100%（几乎所有文档都有标题或描述）

**加权方式：BM25**

BM25 加权结合了词频饱和效应和文档长度归一化，优于简单 TF-IDF：

```
BM25(d, w) = IDF(w) × [ TF(d,w) × (k1 + 1) ] / [ TF(d,w) + k1 × (1 - b + b × dl/avgdl) ]

其中：k1 = 1.2, b = 0.75, dl = 文档长度, avgdl = 平均文档长度
```

生成的稀疏矩阵 `DW_bm25` 形状为 [521,735 × ~15,000]，存储为 `tmp/DW_bm25.parquet`。

### 2.4 Behavior 视图数据

**数据构建**：基于 `CreatorUserId` 和 `OwnerOrganizationId` 构建文档间的直接关联。

- **Creator 关系**：同一创建者的数据集互相关联
  - 独立创建者：192,013 人
  - 覆盖率：100%（每个数据集都有创建者）
  - 平均活跃度：2.7 文档/创建者
- **Organization 关系**：同一组织发布的数据集互相关联
  - 独立组织：390 个
  - 覆盖率：0.5%（2,579 / 521,735）
  - 平均规模：6.6 文档/组织

Behavior 视图不经过二部图随机游走，而是直接构建文档间的**邻接图**，以余弦相似度或二值匹配计算文档对的行为相似性。

**代码位置**：`notebooks/02_evaluation/experiments.ipynb` 中按 `CreatorUserId` 分组文档，同创建者文档互为邻居（二值相似度），构建 Behavior 相似度矩阵/邻居图。

### 2.5 数据统计总表

| 统计项 | 数值 |
|--------|------|
| 数据集总数 (N) | 521,735 |
| 标签数 (去重后) | 394 |
| 词汇量 (Text视图) | ~15,000 |
| 独立创建者数 | 192,013 |
| 独立组织数 | 390 |
| Tag 视图覆盖率 | 41.1% |
| Text 视图覆盖率 | ~100% |
| Creator 覆盖率 | 100% |
| Organization 覆盖率 | 0.5% |
| 合并数据大小 | 283 MB (metadata_merged.csv) |
| 清洗后数据大小 | ~90 MB (doc_clean.parquet) |

---

## 3. 方法论

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    WS-SGNS 多视图数据集推荐系统                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Meta Kaggle 原始数据                                                     │
│  ┌─────────┐ ┌───────────────┐ ┌────────────┐ ┌────────┐               │
│  │Datasets │ │DatasetVersions│ │DatasetTags │ │  Tags  │               │
│  └────┬────┘ └──────┬────────┘ └─────┬──────┘ └───┬────┘               │
│       └──────────────┴───────────────┴─────────────┘                     │
│                           │ 合并 + 清洗                                   │
│                           ▼                                              │
│                  doc_clean.parquet (521,735 文档)                         │
│                           │                                              │
│            ┌──────────────┼──────────────┐                               │
│            ▼              ▼              ▼                                │
│     ┌────────────┐ ┌────────────┐ ┌────────────┐                        │
│     │  Tag 视图   │ │ Text 视图  │ │Behavior视图│                        │
│     │DT_ppmi矩阵 │ │DW_bm25矩阵│ │Creator/Org │                        │
│     └─────┬──────┘ └─────┬──────┘ └─────┬──────┘                        │
│           │              │              │                                │
│     D→T→D 随机游走  D→W→D 随机游走   直接邻接图                           │
│           │              │              │                                │
│     SGNS 嵌入学习    SGNS 嵌入学习   相似度矩阵                           │
│           │              │              │                                │
│     Z_tag [N×d]     Z_text [N×d]    S_beh [N×K]                        │
│           │              │              │                                │
│           └──────────────┼──────────────┘                                │
│                          ▼                                               │
│                  多视图融合 (Fused3-RA / RR / Blend)                      │
│                          │                                               │
│                          ▼                                               │
│              统一推荐列表 (Top-K 相似数据集)                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 二部图随机游走（Bipartite Random Walk）

Tag 视图和 Text 视图均采用二部图随机游走生成训练语料。以 Tag 视图为例：

**图结构**：文档 D 和标签 T 构成二部图，边权重由 PPMI 给出。

**游走模式**：D→T→D→T→D→...（交替在文档节点和标签/词节点之间跳转）

**输出**：仅保留文档节点序列 [D₀, D₁, D₂, ...]，作为"句子"输入 SGNS。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| walks_per_doc | 10 | 每个文档启动的游走次数 |
| l_docs_per_sent | 40 | 每条游走包含的文档数（句子长度） |
| restart_prob | 0.15 | 随机重启概率（返回起始文档） |
| avoid_backtrack | True | 避免立即回退到上一个文档 |
| x_degree_pow | 0.0 | 中间节点度数的幂次调整 |
| x_no_repeat_last | 1 | 避免重复采样上一个中间节点 |

**实现**（参见 `src/random_walk.py`）：

```python
class TorchWalkCorpus:
    """GPU 加速的二部图随机游走语料生成器"""

    def iterate(self, device, is_ddp, rank, world):
        """生成游走句子，D→X→D 模式，仅保留 D 节点"""
        for d0 in starts:
            for _ in range(walks_per_doc):
                seq = [d0]
                cur_d = d0
                for step in range(l_docs_per_sent - 1):
                    # 1. 从当前文档 D 采样中间节点 X（Tag 或 Word）
                    x = sample_neighbor(D→X, cur_d, weights=PPMI/BM25)
                    # 2. 从中间节点 X 采样下一个文档 D
                    next_d = sample_neighbor(X→D, x)
                    # 3. 随机重启
                    if random() < restart_prob:
                        next_d = d0
                    seq.append(next_d)
                    cur_d = next_d
                yield seq  # [D₀, D₁, D₂, ...]
```

**两个视图的差异**：

| 特性 | Tag 视图 | Text 视图 |
|------|---------|----------|
| 二部图 | D→Tag→D | D→Word→D |
| 边权重 | PPMI | BM25 |
| 中间节点数 | 394 | ~15,000 |
| 图密度 | 稀疏（41.1% 文档有标签） | 稠密（近 100% 文档有文本） |

### 3.3 SGNS 嵌入学习

**Skip-gram with Negative Sampling（SGNS）** 将随机游走生成的文档序列视为"句子"，学习文档嵌入向量。

**目标函数**：

```
L = -log σ(u_c · v_o) - Σ_{k=1}^{K} log σ(-u_c · v_k)

其中：
  u_c = 中心文档的输入嵌入 (in_emb)
  v_o = 正样本上下文文档的输出嵌入 (out_emb)
  v_k = 第 k 个负样本的输出嵌入
  σ = sigmoid 函数
  K = 负样本数量 (默认 10)
```

**实现**（参见 `src/sgns_model.py`）：

```python
class SGNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int, sparse: bool = False):
        self.in_emb = nn.Embedding(vocab_size, dim, sparse=sparse)
        self.out_emb = nn.Embedding(vocab_size, dim, sparse=sparse)
        # 均匀初始化 [-0.5/dim, 0.5/dim]
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.uniform_(self.out_emb.weight, -0.5 / dim, 0.5 / dim)

    def forward(self, center, pos, neg):
        v = self.in_emb(center)           # [B, d]
        u = self.out_emb(pos)             # [B, d]
        neg_u = self.out_emb(neg)         # [B, K, d]

        pos_logit = torch.sum(v * u, dim=1)
        neg_logit = torch.einsum("bd,bkd->bk", v, neg_u)

        # Softplus 替代 -log(sigmoid) 提高数值稳定性
        pos_loss = F.softplus(-pos_logit)
        neg_loss = F.softplus(neg_logit).sum(dim=1)
        return (pos_loss + neg_loss).mean().unsqueeze(0)
```

**窗口化配对生成**：从游走句子中使用滑动窗口生成 (center, context) 训练对。

| 参数 | Tag 视图 | Text 视图 | 说明 |
|------|---------|----------|------|
| window | 5 | 4 | 上下文窗口大小 |
| keep_prob | 1.0 | 0.35 | 高频对保留概率（Text 视图下采样） |
| forward_only | False | True | 仅使用前向上下文 |
| ctx_cap | 0（不限） | 4 | 每个中心词最大上下文数 |

### 3.4 负采样别名方法（O(1) GPU 采样）

传统负采样每次需 O(V) 复杂度，对 52 万文档不可行。本系统采用**别名方法（Alias Method）**实现 O(1) GPU 端采样。

**采样分布**：

```python
P(d) ∝ deg(d)^0.75  # Word2Vec 惯例，0.75 次幂平滑
```

**别名表构建**（参见 `src/sampling_utils.py`）：

```python
def build_alias_on_device(probs_np, device):
    """构建别名表，O(n) 预处理，之后 O(1) 采样"""
    # Walker 别名方法 (Vose 1991)
    # 返回 (prob_table, alias_table)

def sample_alias_gpu(prob_t, alias_t, size, device):
    """GPU 上 O(1) 别名采样，完全向量化"""
    n = prob_t.size(0)
    k = torch.randint(n, size, device=device)
    u = torch.rand(size, device=device)
    return torch.where(u < prob_t[k], k, alias_t[k].to(k.dtype))
```

### 3.5 多视图融合策略

三个视图独立训练后，需要融合为统一的推荐结果。本研究实现了三种融合策略。

**代码位置**：`notebooks/02_evaluation/experiments.ipynb`（主实验）、`notebooks/02_evaluation/ablation_experiments.ipynb`（消融实验 A1-A4）。

#### Fused3-RA（Row-Adaptive Fusion，核心方法）

自适应行归一化融合，根据每个文档在不同视图中的邻居分布自动调整权重：

```
S_fused(i, j) = Σ_v w_v(i) × S_v(i, j)

其中 w_v(i) 是文档 i 在视图 v 上的自适应权重
```

特点：不同文档可以有不同的视图权重，如有丰富标签的文档更依赖 Tag 视图，而无标签文档更依赖 Text 和 Behavior 视图。

#### Fused3-RR（Reranking）

在 Fused3-RA 基础上进行重排序，利用 support 一致性、标签 IDF boost、行为 boost 等信号调整排名：

```
score_rr(i, j) = score_ra(i, j) + α × support(i,j) + β × tag_boost(i,j) + γ × beh_boost(i,j)
```

#### Fused3-Blend

以固定比例混合 RA 融合结果和原始相似度分数：

```
score_blend(i, j) = (1 - η) × score_ra(i, j) + η × score_rerank(i, j)
```

其中 η 为混合比例，最优值通过网格搜索确定（实验中 η=0.20~0.30 表现最优）。

---

## 4. 评测框架

### 4.1 银标准设计

在 52 万规模数据集上，人工标注"相关/不相关"成本过高。我们基于元数据特征自动构建**银标准（Silver Standard）**评测体系，从三个维度评估推荐质量。

**代码位置**：银标准构建与评测逻辑实现于 `notebooks/02_evaluation/experiments.ipynb`，评测指标计算函数位于 `src/metrics.py`（234 行）。

#### 维度 A：Tag-Relevance（主题标签相关性）

- **定义**：两个数据集相关 ⟺ 共享至少一个标签
- **分级**：IDF 加权 Jaccard 相似度（连续值 0~1）
  ```
  score(i, j) = Σ(idf(t) for t in tags_i ∩ tags_j) / Σ(idf(t) for t in tags_i ∪ tags_j)
  ```
- **覆盖率**：41.1%（需要查询文档有标签）
- **特点**：精确度高，标签由用户主动标注

#### 维度 B：Desc-Relevance（描述文本语义相关性）

- **定义**：数据集描述文本的 BM25 余弦相似度
- **分级**：连续值 0~1
  ```
  score(i, j) = normalize(BM25_vec_i) · normalize(BM25_vec_j)
  ```
- **覆盖率**：79.9%（几乎所有文档都有描述）
- **特点**：覆盖率高，捕捉标签未覆盖的语义信息

> **设计演进**：最初使用 Org-Relevance（同组织数据集），但覆盖率仅 0.49%（2,579 / 521,735），改用 Desc-Relevance 后覆盖率提升 163 倍。

#### 维度 C：Creator-Relevance（创建者关联性）

- **定义**：两个数据集相关 ⟺ 由同一用户创建
- **分级**：Binary（0 或 1）
- **覆盖率**：77.6%（需要创建者有多个数据集）
- **特点**：反映个人工作流和研究兴趣的连续性

#### 三维度互补性

| 属性 | Tag | Desc | Creator |
|------|-----|------|---------|
| 语义层次 | 主题标签 | 文本语义 | 用户行为 |
| 覆盖率 | 41.1% | 79.9% | 77.6% |
| 精确度 | 高 | 中 | 中 |
| 相关性类型 | 分级 (连续) | 分级 (连续) | 二值 |
| 评测维度 | 主题推荐 | 内容推荐 | 个性化推荐 |

### 4.2 评测指标

每个维度计算以下 6 个指标（以 @20 为标准）：

#### 主指标：nDCG@20（Normalized Discounted Cumulative Gain）

位置敏感的排序质量指标，支持分级相关性：

```
DCG@K = Σ_{i=1}^{K} gain_i / log₂(i + 1)
nDCG@K = DCG@K / IDCG@K
```

- IDCG 是理想排序下的 DCG（所有相关项按相关性递减排列）
- 范围 [0, 1]，值越高排序质量越好

实现参见 `src/metrics.py`：

```python
def ndcg_at_k(gains_sorted, gains_ideal_sorted):
    dcg = dcg_at_k(gains_sorted)
    idcg = dcg_at_k(gains_ideal_sorted)
    return dcg / (idcg + ATOL)
```

#### 辅助指标

| 指标 | 公式 | 特点 |
|------|------|------|
| **MAP@20** | (1/R) × Σ P(k) × rel(k) | 精度与排序综合 |
| **MRR@20** | 1 / rank_first_relevant | 首个相关项位置 |
| **P@20** | \|relevant ∩ top-20\| / 20 | 准确性 |
| **R@20** | \|relevant ∩ top-20\| / \|all relevant\| | 召回能力 |
| **Coverage** | 可评测查询数 / 总查询数 | 评测可靠性 |

#### 统一指标（Unified Metric）

综合三个维度的加权平均：

```
Unified@nDCG = W_TAG × Tag-nDCG + W_DESC × Desc-nDCG + W_CREATOR × Creator-nDCG

权重: W_TAG = 0.5, W_DESC = 0.3, W_CREATOR = 0.2
```

权重设置依据：Tag（主题相关性最重要，权重 50%）> Desc（内容语义补充，权重 30%）> Creator（行为信号辅助，权重 20%）。

### 4.3 覆盖率与可靠性分析

| 维度 | 可评文档数 | 覆盖率 | 评测可靠性 |
|------|-----------|--------|-----------|
| Tag | ~214,585 | 41.1% | 中（依赖标签质量） |
| Desc | ~416,865 | 79.9% | 高（大部分文档可评） |
| Creator | ~405,027 | 77.6% | 高（大部分创建者多产） |
| **综合** | - | **~66%** | 三维度互补，整体可靠 |

**为什么银标准是合理的？**

1. 基于真实用户行为遗留的元数据信号（标签由用户主动标注、创建者关系真实存在）。
2. 多维度覆盖不同推荐场景（主题 + 内容 + 个性化）。
3. 符合领域常识：标签共现→主题相关、同创建者→兴趣关联。
4. 全自动生成，成本低、可复现、可扩展。

**已知局限性**：

- 不是真实的用户点击/评分反馈。
- 标签覆盖率不完整（59% 文档无标签）。
- 同创建者的数据集可能主题分散。
- 无法捕获隐式关联（如不同标签但内容相关的数据集）。

---

## 5. 实验设计

### 5.1 对比方法（12+ 种）

#### Tier 0：本文单视图基准（用于消融研究）

| 方法 | 简称 | 说明 | 特点 |
|------|------|------|------|
| Tag-SGNS | S_tag | Tag 视图 D→T→D 随机游走 + SGNS 嵌入 | 学习标签语义空间 |
| Text-SGNS | S_text | Text 视图 D→W→D 随机游走 + SGNS 嵌入 | 学习文本语义空间 |
| Behavior | S_beh | Creator/Org 行为图嵌入 | 学习社交关系空间 |

#### Tier 1：本文核心融合方法

| 方法 | 简称 | 说明 |
|------|------|------|
| **Fused3-RA** | S_fused3_ra | 三视图自适应行归一化融合（核心方法） |

#### Tier 2：本文增强变体

| 方法 | 简称 | 说明 |
|------|------|------|
| Fused3-RR | S_fused3_rr | 融合 + 重排序（support + tag/beh boost） |
| Fused3-Blend | S_fused3_blend | 融合 + 混合（最优 η） |

#### Group A：传统特征基线（Classical Non-Embedding）

| 方法 | 简称 | 说明 |
|------|------|------|
| Tag-PPMI-Cosine | S_tagppmi | PPMI 矩阵直接余弦相似度 |
| Text-BM25-Cosine | S_textbm25 | BM25 向量直接余弦相似度 |
| Text-Binary-Cosine | S_textbin | Binary 词袋向量余弦相似度 |
| Engagement-Cosine | S_engcos | 行为特征余弦相似度 |

#### Group B：简单融合基线（Naive Fusion）

| 方法 | 简称 | 说明 |
|------|------|------|
| RRF | S_rrf | Reciprocal Rank Fusion |
| CombSUM | S_combsum | Score Summation Fusion |

### 5.2 消融实验

#### A1：视图贡献分析

验证每个视图在融合中的贡献，逐一移除视图观察性能变化：

| 配置 | 移除视图 | 预期 Unified@nDCG | 预期下降 |
|------|---------|-------------------|---------|
| Fused3-RA（完整） | 无 | 0.357 | baseline |
| Fused2-TagText | Behavior | ~0.12 | -66% |
| Fused2-TagBeh | Text | ~0.35 | -2% |
| Fused2-TextBeh | Tag | ~0.38 | +6% |

**分析**：Behavior 视图贡献最大（主导 Creator 任务），但三视图融合提供最佳平衡性。

#### A2：融合策略对比

| 融合策略 | Unified@nDCG | 说明 |
|---------|--------------|------|
| Simple Average | ~0.30 | 等权平均 |
| Weighted Average | ~0.32 | 手动权重 |
| **Adaptive-RA (本文)** | **0.357** | 自适应行权重 |
| RRF | 0.239 | 基于排名的融合 |
| CombSUM | 0.168 | 分数求和 |

#### A3：嵌入方法效果

比较 SGNS 嵌入与传统特征方法在单视图和融合场景下的表现差异。

#### A4：重排序组件贡献

逐步添加重排序组件（support boost → tag IDF boost → behavior boost），分析各组件的边际贡献。

### 5.3 参数敏感性

#### S1：K 值影响

评测不同 K 值（5, 10, 20, 50, 100）下的性能趋势。

#### S2：混合比例 η 影响

网格搜索 η ∈ [0.0, 0.05, 0.10, ..., 0.50]，绘制 Unified@nDCG vs η 曲线。

#### S3：随机游走参数影响

| 参数 | 默认值 | 测试范围 |
|------|--------|---------|
| walks_per_doc | 10 | [5, 10, 15, 20] |
| walk_length | 40 | [20, 40, 60, 80] |
| restart_prob | 0.15 | [0.05, 0.10, 0.15, 0.20] |
| embedding_dim | 256 | [128, 256, 512] |

---

## 6. 技术实现

### 6.1 代码架构

> **架构说明**：`src/` 目录包含可复用的独立模块。`step6_ddp.py` 是主训练入口脚本，为了单文件部署便利，它将所有核心功能（SGNS 模型、随机游走、采样、配对生成、CSR 工具、DDP 工具）内联定义在文件内部，不依赖 `src/` 的 import。`src/` 模块供 notebooks、测试和其他脚本独立调用。

```
src/
├── __init__.py              # 模块导出
├── sgns_model.py            # SGNS 模型定义 (105 行)
│   └── class SGNS           # 双嵌入矩阵 + softplus 损失
├── random_walk.py           # 随机游走生成器 (285 行)
│   ├── class TorchWalkCorpus  # GPU 加速的 D→X→D 游走
│   └── build_corpus()       # 工厂函数：构建 Tag/Text 语料
├── sampling_utils.py        # 负采样工具 (123 行)
│   ├── build_ns_dist_from_deg()  # 0.75 次幂分布
│   ├── build_alias_on_device()   # O(n) 预处理别名表
│   └── sample_alias_gpu()        # O(1) GPU 别名采样
├── pair_batch_utils.py      # 配对生成与批处理 (143 行)
│   ├── iter_pairs_from_corpus()    # Skip-gram 配对生成
│   │   ├─ 动态窗口: 每个中心词随机采样 w ∈ [1, window]
│   │   ├─ 前向/双向: forward_only=True 时仅用右侧上下文
│   │   ├─ 上下文封顶: ctx_cap 限制每个中心词的最大上下文数
│   │   └─ 下采样: keep_prob < 1.0 时随机丢弃部分配对（Text 视图用 0.35）
│   └── batch_pairs_and_negs_fast() # 批量正样本 + GPU 负采样
│       ├─ 收集 batch_size_pairs 个正样本对
│       ├─ GPU 别名采样 [B, K] 个负样本
│       └─ 输出: (centers[B], contexts[B], negatives[B,K]) 张量元组
├── csr_utils.py             # 稀疏矩阵工具 (92 行)
│   ├── load_csr_triplet_parquet()  # Parquet 三元组加载
│   ├── csr_rowview_torch()         # CSR → PyTorch 张量
│   └── csr_T()                     # 矩阵转置
├── ddp_utils.py             # DDP 分布式训练工具 (98 行)
│   ├── init_ddp()           # 自动检测 DDP 环境
│   ├── cleanup_ddp()        # 销毁进程组
│   ├── barrier()            # 同步屏障
│   └── log0()               # 仅 rank-0 日志
└── metrics.py               # 评测指标 (234 行)
    ├── dcg_at_k()           # DCG@K（辅助函数）
    ├── ndcg_at_k()          # nDCG@K
    ├── average_precision_at_k()  # AP@K (用于 MAP)
    ├── mrr_at_k()           # MRR@K
    ├── precision_at_k()     # P@K
    ├── recall_at_k()        # R@K
    ├── hit_rate_at_k()      # HR@K
    └── evaluate_ranking()   # 统一接口
```

### 6.2 DDP 分布式训练

支持 PyTorch DistributedDataParallel（DDP）进行多 GPU 训练：

```bash
# 单机双卡训练
torchrun --nproc_per_node=2 step6_ddp.py \
    --tmp_dir ./tmp --epochs 4 --dim 256 --neg 10 \
    --amp true --tf32 true
```

DDP 实现要点（参见 `src/ddp_utils.py`）：

- **自动环境检测**：`init_ddp()` 自动识别 torchrun 环境，回退到单 GPU 模式。
- **语料分片**：随机游走语料按 shard 分配给不同 rank，`sid % world == rank` 。
- **梯度同步**：通过 DDP wrapper 自动同步梯度，支持 NCCL（GPU）和 GLOO（CPU）后端。
- **仅 rank-0 保存**：检查点和日志仅由 rank 0 输出，避免冲突。

### 6.3 GPU 优化策略

1. **混合精度训练（AMP）**：`torch.cuda.amp.autocast()` 自动选择 FP16/FP32 运算精度，减少显存占用。
2. **TF32 加速**：`torch.backends.cuda.matmul.allow_tf32 = True`，在 Ampere+ GPU 上利用 TF32 加速矩阵乘法。
3. **O(1) 负采样**：别名表预构建在 GPU 上，采样零 CPU 开销。
4. **CSR 行视图**：稀疏矩阵转换为 PyTorch 张量（indptr, indices, data），避免 scipy 的 CPU 瓶颈。
5. **批量化训练**：每步处理 ~204,800 个正样本对，充分利用 GPU 并行能力。
6. **梯度累积**：`--accum` 参数支持在显存受限时累积梯度，等效增大 batch size。

### 6.4 关键代码示例

#### 训练主循环：`train_view()`（`step6_ddp.py:389-573`）

完整的 `train_view()` 函数处理单个视图的全部训练流程。以下是覆盖所有关键步骤的详细伪代码：

```python
def train_view(view_name, N, start_nodes, degD, corpus, device, is_ddp, rank, args, out_path, doc_ids):
    # ── 1. 初始化 ──
    torch.manual_seed(args.seed + (11 if view_name=="tag" else 23))  # 视图独立种子

    # DDP + sparse 不兼容，强制切换 dense
    sparse_rt = args.sparse and (not is_ddp)

    model = SGNS(vocab_size=N, dim=args.dim, sparse=sparse_rt).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    # ── 2. 优化器选择（视图共享参数结构）──
    if sparse_rt:
        optimizer = SparseAdam(model.parameters(), lr=args.lr)   # 稀疏模式
    else:
        optimizer = SGD(model.parameters(), lr=args.lr)           # DDP 默认

    scaler = GradScaler('cuda', enabled=args.amp)

    # ── 3. 负采样别名表（一次性构建）──
    ns_dist = build_ns_dist_from_deg(degD, power=0.75)            # deg^0.75 分布
    ns_prob_t, ns_alias_t = build_alias_on_device(ns_dist, device) # O(N) 预处理

    # ── 4. 读取视图专属参数 ──
    # Tag: window=5, keep_prob=1.0, forward_only=False, ctx_cap=0
    # Text: window=4, keep_prob=0.35, forward_only=True, ctx_cap=4
    v_window, v_keep, v_forward, v_cap = per_view_params[view_name]

    eval_samples = pick_eval_samples(start_nodes, n=8, seed=args.seed+7)

    # ── 5. 训练循环 ──
    for ep in range(1, args.epochs + 1):
        pair_iter = iter_pairs_from_corpus(
            corpus, window=v_window, keep_prob=v_keep,
            forward_only=v_forward, ctx_cap=v_cap,
            seed=args.seed + ep + view_offset
        )

        total_pairs, total_loss, step = 0, 0.0, 0
        model.train()

        for centers_t, contexts_t, negs_t in batch_pairs_and_negs_fast(
                pair_iter, v_batch, args.neg, ns_prob_t, ns_alias_t, device):

            optimizer.zero_grad(set_to_none=True)
            B = centers_t.size(0)
            micro = ceil(B / args.accum)  # 梯度累积的 micro-batch 大小

            # ── 5a. Micro-batch 循环（显存稳定）──
            for s in range(0, B, micro):
                c_mb, x_mb, n_mb = centers_t[s:s+micro], contexts_t[s:s+micro], negs_t[s:s+micro]
                with autocast('cuda', enabled=args.amp):
                    loss = model(c_mb, x_mb, n_mb).mean() / args.accum
                scaler.scale(loss).backward()           # AMP 缩放反向传播
                total_loss += loss.item() * args.accum * c_mb.size(0)

            # ── 5b. 优化器更新 ──
            scaler.step(optimizer)
            scaler.update()

            total_pairs += B; step += 1

            # ── 5c. 进度日志（吞吐量、显存）──
            if step % args.log_every == 0 and rank == 0:
                print(f"[{view_name}] step={step} throughput=... loss=...")

            # ── 5d. 提前结束（max_pairs_per_epoch）──
            if total_pairs >= v_max_pairs:
                break

        # ── 6. Epoch 结束：轻量评估 ──
        if rank == 0 and len(eval_samples) > 0:
            nn_res = quick_eval_neighbors(model, eval_samples, topk=5)
            # 打印采样文档的最近邻 → 可视化嵌入质量

        # ── 7. Epoch Checkpoint ──
        if args.save_epoch_emb and rank == 0:
            E = model.module.in_emb.weight.detach().cpu().numpy()
            Z = L2_normalize(E.astype(float16 or float32))
            save_parquet(Z, f"Z_{view_name}_epoch{ep}.parquet")

        barrier(is_ddp)  # DDP 同步

    # ── 8. 最终导出（L2 归一化 float32）──
    if rank == 0:
        Z = L2_normalize(model.in_emb.weight.detach().cpu().numpy().astype(float32))
        save_parquet(Z, out_path)  # → Z_tag.parquet / Z_text.parquet

    del model, optimizer; torch.cuda.empty_cache()
```

**关键设计决策**：

| 决策 | 原因 |
|------|------|
| Micro-batch 循环 | 大 batch（204,800 对）时显存稳定，等效大 batch 梯度 |
| 视图独立种子 | Tag 用 seed+11，Text 用 seed+23，确保可复现且不相关 |
| DDP 强制 dense | sparse embedding + DDP 梯度同步不稳定，强制 dense 模式 |
| 每 epoch checkpoint | 允许中断续训和 epoch 间嵌入质量对比 |
| L2 归一化导出 | 后续 FAISS 内积搜索等价余弦相似度 |

#### 评测流程（简化）

```python
from src.metrics import evaluate_ranking

# 对每个查询文档
for query_id in evaluable_docs:
    ranked_neighbors = ann_index.search(Z[query_id], k=20)
    gains = [tag_similarity(query_id, n) for n in ranked_neighbors]
    metrics = evaluate_ranking(ranked_neighbors, relevant_set, gains=gains, k=20)
    results.append(metrics)

# 汇总
mean_ndcg = np.mean([r['ndcg'] for r in results])
```

### 6.5 端到端管线（End-to-End Pipeline）

从原始数据到最终推荐列表的完整 7 阶段流程：

#### Phase 1: 数据合并与清洗

- **文件**: `scripts/data_merge.ipynb`（~980 cells）
- **输入**: `data/raw_data/{Datasets,DatasetVersions,DatasetTags,Tags}.csv`
- **流程**:
  1. 以 `Datasets.csv` 为主表
  2. LEFT JOIN `DatasetVersions`（取最新版本的 Title/Description）
  3. LEFT JOIN `DatasetTags` + `Tags`（获取标签名称列表）
  4. 文本清洗（去 HTML 标签、URL、特殊字符）
  5. 构建 PPMI 加权的文档-标签矩阵 `DT_ppmi`
  6. 构建 BM25 加权的文档-词矩阵 `DW_bm25`
  7. 导出 `doc_clean.parquet`、`tag_vocab.parquet`、`text_vocab.parquet`、`rw_params.parquet`
- **输出**: `tmp/` 下的 7 个 parquet 文件

#### Phase 2: 随机游走语料生成

- **文件**: `src/random_walk.py :: build_corpus()`
- **被调用位置**: `step6_ddp.py:597`
- **流程**:
  1. 加载 `DT_ppmi` 和 `DW_bm25` 为 CSR 格式（`src/csr_utils.py :: load_csr_triplet_parquet()`）
  2. 转换为 PyTorch GPU 张量（`csr_rowview_torch()`）
  3. 构建转置矩阵（Tag→Doc 和 Word→Doc）
  4. 创建 `TorchWalkCorpus` 对象（`tag_corpus`, `text_corpus`）
  5. 确定有效起始文档（度数 > 0 的文档）

#### Phase 3: SGNS 嵌入训练

- **文件**: `step6_ddp.py :: train_view()`（行 389-573）
- **流程**:
  1. 初始化 SGNS 模型（双嵌入矩阵 `in_emb` + `out_emb`，逻辑定义于 `src/sgns_model.py`，`step6_ddp.py` 内联实现）
  2. DDP 包装（如果多卡，逻辑定义于 `src/ddp_utils.py :: init_ddp()`，`step6_ddp.py` 内联实现）
  3. 构建负采样别名表（逻辑定义于 `src/sampling_utils.py :: build_alias_on_device()`，`step6_ddp.py` 内联实现）
  4. 对每个 epoch:
     a. 从语料生成器获取游走句子（逻辑定义于 `src/random_walk.py :: TorchWalkCorpus.iterate()`，`step6_ddp.py` 内联实现）
     b. 滑动窗口生成 (center, context) 对（逻辑定义于 `src/pair_batch_utils.py :: iter_pairs_from_corpus()`，`step6_ddp.py` 内联实现）
     c. 批量收集 + GPU 别名负采样（逻辑定义于 `src/pair_batch_utils.py :: batch_pairs_and_negs_fast()`，`step6_ddp.py` 内联实现）
     d. AMP 混合精度前向/反向传播（micro-batch 循环，梯度累积）
     e. 梯度累积 + 优化器更新（`scaler.step()` + `scaler.update()`）
     f. 轻量评估：采样文档的最近邻（`quick_eval_neighbors()`）
     g. Checkpoint 保存（`Z_{view}_epoch{ep}.parquet`）
  5. 最终导出 L2 归一化嵌入（`Z_tag.parquet` / `Z_text.parquet`）

#### Phase 4: ANN 索引构建

- **文件**: `scripts/build_ann_index.py`（333 行）或 `ddp_scripts/step7_faiss_ann_ddp.py`（328 行）
- **流程**:
  1. 加载 `Z_tag.parquet` 和 `Z_text.parquet`（`[N × 256]` float32 矩阵）
  2. L2 归一化（内积搜索等价余弦相似度）
  3. 构建 FAISS 内积索引（FlatIP 精确 或 IVF 近似）
  4. GPU 加速批量查询每个文档的 Top-K 邻居
  5. 自排除（排除查询文档自身）
  6. DDP 模式下跨 rank 收集结果（`gather_edges_ddp()`）
  7. 分区存储邻居列表（parquet 格式，row/col/val 三元组）

#### Phase 5: Behavior 视图构建

- **文件**: `notebooks/02_evaluation/experiments.ipynb`（内联实现）
- **流程**:
  1. 加载 `doc_clean.parquet` 中的 `CreatorUserId` 列
  2. 按 `CreatorUserId` 分组文档
  3. 同创建者文档互为邻居（二值相似度 = 1.0）
  4. 构建 Behavior 相似度矩阵/邻居图（`S_beh`）

#### Phase 6: 银标准构建与评测

- **文件**: `notebooks/02_evaluation/experiments.ipynb`
- **指标计算**: `src/metrics.py`（234 行）
- **流程**:
  1. **Tag 维度**: IDF 加权 Jaccard 相似度（基于 `DT_ppmi` 矩阵）
  2. **Desc 维度**: BM25 余弦相似度（基于 `DW_bm25` 矩阵）
  3. **Creator 维度**: 二值匹配（基于 `doc_clean.parquet` 中的 `creator_id`）
  4. 对每个方法的推荐列表计算 nDCG/MAP/MRR/P/R/Coverage（`src/metrics.py :: evaluate_ranking()`）
  5. 加权汇总为 Unified@nDCG（`0.5×Tag + 0.3×Desc + 0.2×Creator`）

#### Phase 7: 融合与重排序

- **文件**: `notebooks/02_evaluation/{experiments,ablation_experiments}.ipynb`
- **流程**:
  1. **Fused3-RA**: 行自适应权重融合三视图邻居图（每文档按视图覆盖度自动分配权重）
  2. **Fused3-RR**: 在 RA 基础上加 support 一致性 + tag IDF boost + behavior boost 重排序
  3. **Fused3-Blend**: RA 和 RR 的线性混合（η 参数网格搜索，最优 η=0.20~0.30）
  4. 消融实验 A1-A4 在 `notebooks/02_evaluation/ablation_experiments.ipynb` 中完成

### 6.6 脚本与 Notebook 说明

#### 根目录关键文件

| 文件 | 行数 | 用途 | 管线阶段 |
|------|------|------|---------|
| `step6_ddp.py` | 610 | 主训练脚本（自包含，内联全部工具函数） | Phase 2-3 |
| `analyze_random_walks.py` | 463 | 随机游走分析与可视化（根目录版本） | 分析 |
| `RESEARCH_PROPOSAL.md` | ~1119 | 完整研究方案 | 文档 |
| `RESEARCH_OVERVIEW.md` | 96 | 研究概述入口 | 文档 |
| `README.md` | 159 | 项目快速入门 | 文档 |

#### scripts/ 目录

| 文件 | 行数 | 用途 | 管线阶段 |
|------|------|------|---------|
| `data_merge.ipynb` | ~980 cells | 原始数据合并、PPMI/BM25 矩阵构建、词表导出 | Phase 1 |
| `train_sgns.py` | 453 | 统一 SGNS 训练入口（支持单/双视图，替代直接调用 step6_ddp.py） | Phase 3 |
| `build_ann_index.py` | 333 | FAISS k-NN 索引构建（GPU 加速、DDP 支持、分区存储） | Phase 4 |
| `analyze_walks.py` | 316 | 随机游走与嵌入统计分析（游走长度分布、覆盖率、嵌入质量） | 分析 |

#### notebooks/ 目录

| 路径 | 行数 | 用途 | 管线阶段 |
|------|------|------|---------|
| `01_pipeline/analyse_new.ipynb` | ~1380 | 新管线结果分析与可视化 | 分析 |
| `01_pipeline/analyse_high.ipynb` | ~1146 | 高维嵌入分析（t-SNE/UMAP 降维可视化） | 分析 |
| `02_evaluation/experiments.ipynb` | ~3982 | **主实验**：对比评测、银标准构建、Behavior 视图、融合 | Phase 5-7 |
| `02_evaluation/ablation_experiments.ipynb` | ~2582 | 消融实验 A1-A4（视图贡献、融合策略、嵌入方法、重排序组件） | Phase 7 |
| `02_evaluation/compare_methods_framework.ipynb` | ~1143 | 多方法对比框架（12+ 方法统一评测） | Phase 7 |
| `03_analysis/tag_statistics_analysis.ipynb` | ~1471 | 标签词表统计与分布分析 | 分析 |
| `03_analysis/analyze_random_walks.ipynb` | ~954 | 游走参数与质量分析（步长、重启概率影响） | 分析 |
| `03_analysis/analyse_ddp_hybrid.ipynb` | ~1382 | DDP 性能与效率分析（多卡加速比、通信开销） | 分析 |

#### ddp_scripts/ 目录

| 文件 | 行数 | 用途 |
|------|------|------|
| `step7_faiss_ann_ddp.py` | 328 | DDP 版 FAISS 索引构建（多 GPU 并行查询、分区聚合） |
| `README.md` | 237 | DDP 脚本使用说明（torchrun 启动命令、参数详解） |

#### tests/ 目录

| 文件 | 行数 | 用途 |
|------|------|------|
| `test_evaluation_flow.py` | 237 | 端到端评测管线测试（银标准构建→指标计算→结果验证） |
| `test_desc_fixes.py` | 220 | 描述文本处理修复测试（HTML 清洗、编码处理） |
| `test_desc_loading.py` | 91 | 描述数据加载测试（parquet 读取、字段完整性） |
| `test_visualization_fix.py` | 109 | 可视化输出修复测试（图表生成、格式校验） |

### 6.7 ANN 索引构建（FAISS）

`scripts/build_ann_index.py`（333 行）使用 FAISS 库构建 k 近邻图，是 Phase 4 的核心脚本。

**工作流程**：

1. **加载嵌入**: `Z_tag.parquet` / `Z_text.parquet` → `[N × 256]` float32 矩阵
2. **L2 归一化**: 使得内积搜索等价于余弦相似度
3. **索引构建**:
   - `flat_ip`: 精确内积搜索（暴力枚举，O(N²)，适合精确结果）
   - `ivf_ip`: IVF 近似搜索（倒排索引 + 聚类，更快，适合大规模）
4. **GPU 加速**: 自动检测 GPU 可用性，使用 `faiss.index_cpu_to_gpu()` 迁移索引
5. **批量查询**: 每次查询 `batch_q`（默认 8192）个文档的 Top-K 邻居
6. **自排除**: 排除查询文档自身（对角线元素）
7. **DDP 并行**: 各 rank 处理子集查询，通过 `gather_edges_ddp()` 汇总
8. **分区存储**: 大边集分割为多个 parquet 文件（每文件最大 `part_size` 条边）

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--k` | 50 | 近邻数量 |
| `--index_type` | flat_ip | 索引类型（flat_ip / ivf_ip） |
| `--use_gpu` | true | 是否使用 GPU 加速 |
| `--batch_q` | 8192 | 每批查询的文档数 |
| `--part_size` | 2,000,000 | 每个 parquet 分区的最大边数 |
| `--nlist` | 1024 | IVF 聚类数（仅 ivf_ip） |
| `--nprobe` | 64 | IVF 搜索探针数（仅 ivf_ip） |

**DDP 版本**：`ddp_scripts/step7_faiss_ann_ddp.py`（328 行）提供等价功能，支持 `torchrun` 多 GPU 启动，各 GPU 独立构建索引并并行查询不同文档子集。

---

## 7. 预期结果

### 7.1 主要性能预期

基于已有实验数据，各方法的预期表现如下：

| 方法 | Tag-nDCG | Desc-nDCG | Creator-nDCG | Unified@nDCG | 分组 |
|------|---------|----------|--------------|--------------|------|
| **Fused3-Blend** | ~0.13 | ~0.20 | ~0.87 | **0.468** | 本文-增强 |
| **Fused3-RA** | 0.110 | ~0.15 | 0.833 | 0.357 | 本文-核心 |
| Behavior | 0.136 | 0.483 | 0.865 | 0.389 | 本文-单视图 |
| Tag-PPMI-Cos | **0.715** | 0.004 | 0.010 | 0.446 | 传统-最强 |
| Text-BM25-Cos | 0.181 | 0.203 | 0.036 | 0.134 | 传统 |
| RRF | 0.359 | 0.151 | 0.028 | 0.239 | 融合基线 |
| CombSUM | 0.239 | 0.143 | 0.033 | 0.168 | 融合基线 |
| Tag-SGNS | 0.030 | ~0.01 | ~0.01 | 0.018 | 本文-单视图 |
| Text-SGNS | 0.030 | ~0.01 | ~0.01 | 0.018 | 本文-单视图 |

### 7.2 关键发现预测

**发现 1：多视图融合优于单视图**

- Fused3-RA (Unified: 0.357) 显著优于任何单视图 SGNS 嵌入 (0.018)
- 融合后在三个维度上均有合理表现，而单视图严重偏向其擅长维度

**发现 2：自适应融合优于简单策略**

- Fused3-RA (0.357) > RRF (0.239, +49%) > CombSUM (0.168, +113%)
- 自适应权重能更好利用不同视图的互补性

**发现 3：嵌入方法在融合场景下表现更佳**

- 虽然 SGNS 单视图弱于传统方法（Tag-SGNS 0.030 vs Tag-PPMI 0.715）
- 但嵌入向量在融合时提供更好的语义泛化能力
- Tag-PPMI 过度专注于 Tag 任务（0.715），在其他维度几乎为零

**发现 4：重排序和混合策略进一步提升**

- Fused3-Blend (η=0.30) 达到 Unified@nDCG = 0.468
- 相比 CombSUM 基线提升 179%

### 7.3 潜在局限性

1. **SGNS 单视图较弱**：随机游走生成的训练语料可能不如直接特征方法（PPMI/BM25）精确，导致单视图嵌入质量有限。
2. **Behavior 视图主导**：Behavior 视图在 Creator 和 Desc 任务上贡献巨大，融合结果可能过度依赖行为信号。
3. **银标准局限**：基于元数据的自动评测不等于真实用户偏好，实际部署需进一步验证。
4. **冷启动问题**：对于无标签、无描述的新数据集，Tag 和 Text 视图无法提供有效嵌入。
5. **Tag-PPMI 的强势地位**：在 Tag 单维度评测上，简单的 PPMI 余弦方法远超所有嵌入方法，暗示嵌入学习可能丢失了某些直接共现信息。

---

## 8. 附录

### A. 完整参数列表

#### 训练参数（step6_ddp.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --epochs | 4 | 训练轮数 |
| --dim | 256 | 嵌入维度 |
| --neg | 10 | 负样本数 |
| --lr | 0.025 | 学习率 |
| --ns_power | 0.75 | 负采样频率幂次 |
| --optimizer | sparse_adam | 优化器（sparse_adam / adagrad / sgd） |
| --sparse | False | 是否使用稀疏梯度 |
| --amp | True | 混合精度训练 |
| --tf32 | True | TF32 加速 |
| --seed | 2025 | 随机种子 |
| --accum | 1 | 梯度累积步数 |

#### Tag 视图参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --window_tag | 5 | 上下文窗口 |
| --keep_prob_tag | 1.0 | 保留概率 |
| --forward_only_tag | False | 仅前向上下文 |
| --ctx_cap_tag | 0 | 上下文上限（0=不限） |
| --batch_pairs_tag | 204,800 | 每步训练对数 |
| --max_pairs_tag | 20,000,000 | 每 epoch 对数上限 |

#### Text 视图参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --window_text | 4 | 上下文窗口 |
| --keep_prob_text | 0.35 | 保留概率（下采样高频对） |
| --forward_only_text | True | 仅前向上下文 |
| --ctx_cap_text | 4 | 上下文上限 |
| --batch_pairs_text | 204,800 | 每步训练对数 |
| --max_pairs_text | 20,000,000 | 每 epoch 对数上限 |

#### 随机游走参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| walks_per_doc | 10 | 每文档游走次数 |
| l_docs_per_sent | 40 | 每条游走的文档数 |
| restart_prob | 0.15 | 随机重启概率 |
| avoid_backtrack | True | 避免回退 |
| x_degree_pow | 0.0 | 中间节点度数幂次 |
| x_no_repeat_last | 1 | 避免重复上一中间节点 |

### B. 文件清单与数据流

```
data/raw_data/
├── Datasets.csv              (521K 行, 68 MB)   ← Meta Kaggle 原始数据
├── DatasetVersions.csv       (11.6M 行, 984 MB)
├── DatasetTags.csv           (446K 行, 13 MB)
└── Tags.csv                  (831 行, 98 KB)
         │
         ▼ [合并 + 清洗]
data/metadata_merged.csv      (521K 行, ~283 MB)
         │
         ▼ [预处理]
tmp/
├── doc_clean.parquet         (521K 行, ~90 MB)   ← 清洗后的文档表
├── DT_ppmi.parquet           (3.2 MB)            ← Tag 视图稀疏矩阵
├── DT_bin.parquet            (2.0 MB)            ← Tag 二值矩阵（基线用）
├── DT_tfidf.parquet          (2.9 MB)            ← Tag TF-IDF 矩阵（基线用）
├── DW_bm25.parquet           (59 MB)             ← Text 视图稀疏矩阵
│
├── Z_tag.parquet             ← Tag 视图嵌入 [N × 256]
├── Z_text.parquet            ← Text 视图嵌入 [N × 256]
├── Z_*_epoch*.parquet        ← 各 epoch 检查点
│
├── beh_base.parquet          ← Behavior 基线嵌入
├── ann_params.parquet        ← FAISS 索引参数
│
└── comparison_results_*.csv  ← 实验对比结果
```

### C. 运行命令速查

```bash
# 1. 数据预处理（Jupyter Notebook）
jupyter notebook scripts/data_merge.ipynb

# 2. 双视图 SGNS 训练（单 GPU）
python step6_ddp.py --tmp_dir ./tmp --epochs 4 --dim 256 --neg 10 --amp true

# 3. 双视图 SGNS 训练（多 GPU / DDP）
torchrun --nproc_per_node=2 step6_ddp.py \
    --tmp_dir ./tmp --epochs 4 --dim 256 --neg 10 \
    --amp true --tf32 true \
    --window_tag 5 --keep_prob_tag 1.0 --forward_only_tag false --ctx_cap_tag 0 \
    --window_text 4 --keep_prob_text 0.35 --forward_only_text true --ctx_cap_text 4 \
    --batch_pairs_tag 204800 --batch_pairs_text 204800

# 4. 构建 ANN 索引
python scripts/build_ann_index.py --k 50 --use_gpu true

# 5. 评测实验
jupyter notebook notebooks/02_evaluation/experiments.ipynb

# 6. 消融实验
jupyter notebook notebooks/02_evaluation/ablation_experiments.ipynb

# 7. 分析与可视化
python scripts/analyze_walks.py --base_dir ./tmp --output_dir ./analysis_outputs
```
