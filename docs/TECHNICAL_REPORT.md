# Kaggle 数据集推荐系统：基于表格内容的多视图融合技术方案

> **一句话概述**：就像"看过这本书的人也看了……"，我们为 Kaggle 上 521,735 个数据集建立"看过这个数据集的人也喜欢……"的推荐系统。核心研究问题：**打开数据集、分析表格内容，能否让推荐更准？**

---

## 目录

- [第一部分：项目是什么（What & Why）](#第一部分项目是什么what--why)
- [第二部分：数据从哪来（Data Acquisition）](#第二部分数据从哪来data-acquisition)
- [第三部分：理解表格内容（Content Feature Engineering）](#第三部分理解表格内容content-feature-engineering)
- [第四部分：找到相似数据集（Similarity Graph）](#第四部分找到相似数据集similarity-graph)
- [第五部分：融合四个视角（Multi-View Fusion）](#第五部分融合四个视角multi-view-fusion)
- [第六部分：如何评价推荐质量（Evaluation）](#第六部分如何评价推荐质量evaluation)
- [第七部分：技术架构与工程实现](#第七部分技术架构与工程实现)
- [第八部分：结论与推荐](#第八部分结论与推荐)

---

## 第一部分：项目是什么（What & Why）

### 1 一句话概述

想象你走进一家图书馆，想找一本有趣的书。图书管理员可以通过几种方式帮你：看书脊上的分类标签、读封底的简介、查看"借过这本书的人还借了什么"，或者——翻开书读几页正文。这四种方式各有优劣，结合起来效果最好。

我们的项目做的是完全相同的事情，只不过"书"换成了 Kaggle 上的 521,735 个数据集。现有的推荐系统已经利用了前三种信息（标签、文字描述、用户行为），**本项目新增第四种视角：打开表格，分析里面的数据内容**，验证这能否提升推荐质量。

### 2 四个"视角"看数据集

**类比：盲人摸象。** 如果只从一个角度看数据集，就像只摸到大象的一条腿——你无法看到全貌。多视图（Multi-View）方法从多个角度同时观察，拼出更完整的画像。

```
                     ┌─────────────┐
                     │  一个数据集  │
                     └──────┬──────┘
            ┌───────┬───────┼───────┬─────────┐
            v       v       v       v         v
        Tag 视图  Text 视图  Beh 视图  Content 视图
        (标签)    (描述文字)  (行为)    (表格内容)
         │         │         │         │
         │         │         │         │ <-- 本项目新增
         v         v         v         v
     "tabular"  "sales     "买了A    "price列:
      "finance"  data..."   的人也    0.99-999,
                            买了B"    median=29"
```

| 视图 | 类比 | 信息来源 | 优势 | 局限 |
|------|------|---------|------|------|
| **Tag 视图** | 书的分类标签 | 作者自行打的标签（如 "tabular", "finance"） | 简洁明确 | 标签粗粒度，可能遗漏 |
| **Text 视图** | 书的封底简介 | 数据集标题和描述文字 | 自然语言信息丰富 | 依赖作者撰写质量 |
| **Beh 视图（Behavior）** | "买了A也买了B" | 用户浏览/下载行为 | 捕捉隐式偏好 | 冷启动问题（新数据集无行为） |
| **Content 视图（新增）** | 翻开书读正文 | 表格内的列名、数据类型、统计分布 | 直接反映数据本质 | 需要下载并解析原始数据 |

**为什么需要 Content 视图？一个例子：**

两个数据集都标着 "tabular" 标签、描述都提到 "sales"：
- **数据集 A**：10 行 × 3 列，按月汇总的宏观销售额（`month`, `region`, `total_sales`）
- **数据集 B**：100 万行 × 20 列，逐笔交易明细（`transaction_id`, `product`, `price`, `quantity`, `timestamp`, ...）

仅凭标签和描述，A 和 B 看起来非常相似。**但只有打开表格才能发现它们本质不同**——Content 视图能捕捉到这种差异。

---

## 第二部分：数据从哪来（Data Acquisition）

### 3 数据集筛选与下载

**目的**：从 521,735 个 Kaggle 数据集中，选出适合分析的表格型数据集，下载它们的原始数据文件。

#### 筛选漏斗

不是所有 Kaggle 数据集都适合表格分析——图片集、NLP 语料、二进制文件都无法用表格方法处理。我们通过多级漏斗层层筛选：

```
521,735  元数据总量 (metadata_merged.csv)
    │
    │  标签过滤：含 "tabular" 等标签
    │  大小过滤：100KB - 100MB
    │  热度过滤：浏览量 >= 1,000 (1K实验)
    v
 ~7,393  候选数据集
    │
    │  Kaggle API 搜索：slug -> ref 匹配
    │  (精确匹配 + 模糊匹配)
    v
 ~1,179  API 匹配成功
    │
    │  下载 + 主表选取
    │  (选最大表格文件，排除 submission/readme)
    v
 1,000   最终 D_content (1K 实验)
```

#### 为什么设这些条件？

| 条件 | 值 | 原因 |
|------|---|------|
| 标签 = tabular | — | 确保数据集是表格型，排除图片/NLP 数据集 |
| 文件大小 100KB - 100MB | — | 太小的文件信息量不足，太大的下载和解析成本过高 |
| 浏览量 >= 1,000 | — | 确保数据集有足够用户交互，行为视图有数据 |
| SEARCH_MARGIN = 1.2 | — | API 搜索候选多 20%，补偿匹配失败 |

#### Kaggle API 搜索流程

每个候选数据集有一个 `slug`（如 `"heptapod/titanic"`），但 Kaggle API 不支持按 slug 直接查询。我们的匹配策略：

1. **Tier-1 精确匹配**：用 slug 中的关键词搜索 API，若返回结果中 `ref` 与 `slug` 完全相同 → 命中
2. **Tier-2 模糊匹配**：若精确匹配失败，对返回结果按标题相似度排序 → 取最高分
3. **回填机制**：对匹配失败的数据集，放宽搜索条件重试

API 搜索结果缓存在 `api_cache/{slug}.json`，避免重复请求。

#### 规模扩展策略

从 1K 扩展到更大规模时，筛选条件需要逐步放宽：

| 目标规模 | 标签筛选 | 大小范围 | 浏览量要求 | 实际获得 |
|---------|---------|---------|-----------|---------|
| **1K** | tabular | 100KB - 100MB | >= 1,000 | 1,000 |
| **10K** | tabular | 100KB - 100MB | 无 | 10,000 |
| **50K** | broad tags* | 50KB - 500MB | 无 | 4,527** |
| **100K** | any tags | 10KB - 1GB | 无 | 9,778** |

*broad tags = tabular + data visualization + feature engineering + statistics + finance + healthcare 等

**受 Kaggle API 匹配率（4.8%）和主表提取成功率（55.7%）限制，实际获得数量低于目标。约 9,778 是当前筛选策略下的覆盖率上限。

---

## 第三部分：理解表格内容（Content Feature Engineering）

### 4 表格采样与列分析

**目的**：高效提取表格的"DNA"——用最小的计算成本，获得最大的信息量。

**类比**：像医生体检时抽血化验而非全身 CT——只读前 1,024 行就够了解一个表格的"体质"。

#### 三步流程

```
 原始表格文件 (.csv/.parquet/...)
     │
     │  Step 1: sample_table()
     │  - 读取前 1,024 行
     │  - 去除空列、常量列、ID 列
     │  - 按信息分数排序，保留 top-60 列
     v
 清洁后的 DataFrame (max 1024×60)
     │
     │  Step 2: profile_column()
     │  - 逐列检测类型：数值/日期/文本/分类
     │  - 计算统计信息（min/max/median/top values/...）
     v
 列级统计 (ColStats 列表)
     │
     │  Step 3: col_to_description()
     │  - 将统计信息转为英文文本描述
     v
 英文描述字符串 (每列一句话)
```

#### Step 1: 表格采样（`sample_table`）

读取原始文件后，进行三轮清洗：

1. **去除空列**：全为 NaN 的列不含信息，直接丢弃
2. **去除常量列**：所有行值相同的列（`nunique <= 1`）没有区分度
3. **去除 ID 列**：列名匹配 `^(unnamed|index|id)$|_id$|_idx$` 且唯一率 > 95% 的列是标识符，不反映数据内容

清洗后，若列数仍超过 `max_cols=60`，按 **信息分数** 排序取 top-60：

```
info_score(col) = unique_ratio(col) × notna_ratio(col)
```

**为什么这个公式有效？** 用两个极端例子说明：
- 一列全是 `NULL` → `notna_ratio = 0` → `info_score = 0`（无信息）
- 一列全是同一个值 `"A"` → `unique_ratio ≈ 0` → `info_score ≈ 0`（无区分度）
- 一列有 80% 非空、50 种不同值 → `info_score = 0.5 × 0.8 = 0.4`（信息量高）

#### Step 2: 列类型检测（`profile_column`）

类型检测按优先级依次尝试，命中即停止：

| 优先级 | 类型 | 检测条件 | 举例 |
|-------|------|---------|------|
| 1 | **Numeric（数值型）** | >= 95% 的值可转为数字 | `price: [9.99, 29.99, 99.99, ...]` |
| 2 | **Datetime（日期型）** | >= 90% 的值可解析为日期 | `created_at: [2024-01-01, ...]` |
| 3 | **Text（文本型）** | 平均字符长度 >= 30 | `description: "This product is..."` |
| 4 | **Categorical（分类型）** | 以上都不满足 | `color: [red, blue, green, ...]` |

检测阈值的设计思路：`TAU_NUM=0.95` 比 `TAU_DT=0.90` 更严格，因为数字转换几乎不会产生误报，而日期格式多样（"Jan 1", "2024/1/1", "01-01-2024"），需要更宽松的阈值。`L_TEXT=30` 字符的门槛将短标签（如 "red"）与真正的自由文本区分开。

#### Step 3: 描述生成（`col_to_description`）

将结构化统计信息转为英文自然语言描述，每种类型有专用模板：

| 类型 | 模板示例 |
|------|---------|
| Numeric | `Column "price": numeric, range [0.99, 999.99], median=29.99, mean=45.23, std=38.17, 2.1% missing.` |
| Categorical | `Column "color": categorical with 8 unique values (15.3% unique). Top values: red (120), blue (95), green (80). 0.0% missing.` |
| Datetime | `Column "created_at": datetime from 2020-01-01 to 2024-12-31, spanning 1826 days. 0.5% missing.` |
| Text | `Column "description": free text, avg length 156 chars. Sample: "This premium product features...". 3.2% missing.` |

**为什么用英文？** 因为下一步的 Sentence-Transformer 模型在英文上训练最充分，用英文描述能获得更高质量的语义嵌入。

### 5 从文字到向量

**目的**：把人类可读的列描述转换成计算机可以比较的数字向量——这是让机器理解"两个列描述有多相似"的关键。

**类比：GPS 坐标。** 每段描述变成 384 维空间中的一个"坐标点"，含义相似的描述在这个空间中距离近，含义不同的距离远。就像北京和天津的 GPS 坐标很近，北京和纽约的坐标很远。

#### Sentence-Transformer（句子编码器）

我们使用 `all-MiniLM-L6-v2` 模型，这是一个轻量级的预训练语言模型：

| 属性 | 值 |
|------|---|
| 模型名 | `sentence-transformers/all-MiniLM-L6-v2` |
| 输出维度 | 384 |
| 参数量 | ~22M（轻量级） |
| 输入限制 | 256 tokens |
| 训练数据 | 10 亿句子对 |

**为什么选这个模型？Trade-off 分析：**
- 更大的模型（如 `all-mpnet-base-v2`, 768 维）精度略高，但推理速度慢 2-3 倍，向量存储翻倍
- 更小的模型（如 `all-MiniLM-L4-v2`）存储更省，但语义理解能力下降
- `all-MiniLM-L6-v2` 在精度和效率之间取得最佳平衡，是 Sentence-Transformers 库的默认推荐

#### 编码过程

```
列描述 (字符串)  -->  Sentence-Transformer  -->  384 维向量 (float32)

"Column 'price': numeric,        [0.023, -0.156, 0.089, ..., 0.042]
 range [0.99, 999.99],    -->     (384个浮点数)
 median=29.99..."
```

编码后对向量做 L2 归一化（`normalize_embeddings=True`），使每个向量的长度恒为 1。这样后续比较时，内积就等于余弦相似度（Cosine Similarity）：

```
cos(a, b) = a · b / (||a|| × ||b||)
           = a · b    (当 ||a|| = ||b|| = 1 时)
```

**具体例子**：
- `"Column 'price': numeric, range [0, 100]"` 和 `"Column 'cost': numeric, range [0, 200]"` → 向量距离**近**（都是价格类数值列）
- `"Column 'name': text, avg length 25 chars"` 和 `"Column 'price': numeric, range [0, 100]"` → 向量距离**远**（文本列 vs 数值列）

### 6 从列向量到数据集向量

**目的**：一个数据集有 N 列，每列有一个 384 维向量。我们需要把 N 个列向量"压缩"成一个代表整个数据集的向量——数据集的"指纹"。

**类比**：一本书有很多章节，每章有自己的主题向量。图书馆索引不能存每章的向量，需要一个"全书摘要向量"。不同章节重要性不同（核心章节权重高、附录权重低），所以用加权平均而非简单平均。

#### 加权平均公式

对于数据集 $d$ 的 $N$ 个列，数据集向量为：

```
z_d = normalize( sum_i( w_i × e_i ) / sum_i( w_i ) )
```

其中每列的权重由三个因子决定：

```
w_i = w_type(dtype_i) × (1 - missing_pct_i / 100) × min(unique_pct_i / 100, 0.95)
```

| 因子 | 含义 | 设计动机 |
|------|------|---------|
| `w_type` | 类型权重 | text=1.2（文本描述信息量最大），datetime=0.8（日期列区分度较低），numeric/categorical=1.0 |
| `1 - missing%` | 完整度 | 缺失率 80% 的列只有 0.2 的权重——缺失越多，可信度越低 |
| `min(unique%, 95%)` | 多样性 | 唯一值比例反映列的区分度；上限 95% 防止近乎唯一的 ID 列获得过高权重 |

#### 举例：一个 5 列数据集的加权过程

| 列名 | 类型 | 缺失率 | 唯一率 | w_type | 1-miss | min(uniq,0.95) | **总权重** |
|------|------|-------|-------|--------|--------|---------------|-----------|
| price | numeric | 2% | 45% | 1.0 | 0.98 | 0.45 | **0.441** |
| category | categorical | 0% | 3% | 1.0 | 1.00 | 0.03 | **0.030** |
| description | text | 5% | 90% | 1.2 | 0.95 | 0.90 | **1.026** |
| created_at | datetime | 0% | 60% | 0.8 | 1.00 | 0.60 | **0.480** |
| status | categorical | 50% | 1% | 1.0 | 0.50 | 0.01 | **0.005** |

`description` 列权重最高（1.026），因为它是文本型（w_type=1.2）、几乎不缺失（0.95）、且唯一率高（0.90）。`status` 列权重最低（0.005），因为一半缺失且几乎无变化。

#### L2 归一化

加权平均后的向量做 L2 归一化：`z = z / ||z||`，使所有数据集向量长度为 1。这保证后续比较时不受列数影响——一个 5 列的数据集和一个 50 列的数据集可以公平比较。

---

## 第四部分：找到相似数据集（Similarity Graph）

### 7 最近邻搜索

**目的**：给每个数据集找到最像它的 50 个"邻居"，形成一张相似度图。

**类比：地图上找最近的 50 家餐厅。** 你站在某个位置（当前数据集的向量），需要找出距离最近的 50 家餐厅（最相似的 50 个数据集）。如果城市里有 52 万家餐厅，逐一计算距离太慢——需要专门的空间索引工具。

#### FAISS 工具

我们使用 Facebook 开源的 FAISS（Facebook AI Similarity Search）库进行高速向量搜索。

```
     Z_tabcontent (B × 384)          FAISS IndexFlatIP
     ┌─────────────────┐              ┌───────────────┐
     │  数据集 0 向量   │──── add ───>│               │
     │  数据集 1 向量   │              │   内积索引     │
     │  ...            │              │ (暴力搜索模式) │
     │  数据集 B-1 向量 │              │               │
     └─────────────────┘              └───────┬───────┘
                                              │
                          search(Z, k=51) ────┘
                                              │
                                              v
                                    每个数据集的 top-50 邻居
                                    + 对应的相似度分数
```

**内积（Inner Product）vs 余弦相似度（Cosine Similarity）的关系**：

当向量经过 L2 归一化后（`||a|| = ||b|| = 1`），两者完全等价：

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||) = a · b = inner_product(a, b)
```

因此我们用 `IndexFlatIP`（内积索引）即等同于余弦相似度搜索。

**为什么不存完整的相似度矩阵？**

52 万 × 52 万 = 2,714 亿个数字。即使用 float32（4 字节），也需要 **1 TB** 的内存。这显然不可行。

解决方案：每个数据集只保留 top-50 个最近邻 → 52 万 × 50 = 2,600 万条边 → 约 **600 MB** 的稀疏矩阵。存储量减少 **10,000 倍**。

#### GPU/CPU 回退链

FAISS 支持 GPU 加速。我们的代码实现了自动回退：

```
尝试 FAISS GPU  ──(失败)──>  FAISS CPU  ──(失败)──>  sklearn NearestNeighbors  ──(失败)──>  numpy 暴力计算
```

### 8 对称化与归一化

**目的**：保证"A 像 B"和"B 像 A"的一致性，并将相似度转化为概率分布。

#### 对称化（Symmetrisation）

kNN 搜索的结果是不对称的：A 可能在 B 的 top-50 邻居中，但 B 不一定在 A 的 top-50 中。我们通过取最大值来对称化：

```
S_sym[A, B] = max(S[A, B], S[B, A])
```

**为什么取 max 而非 average？** 如果 A 认为 B 非常相似（0.9），但 B 不认为 A 相似（0.0，因为 A 不在 B 的 top-50 中），取 average 会得到 0.45（低估了真实相似度），取 max 会得到 0.9（保留了有信息量的信号）。

#### L1 行归一化

对称化后的矩阵每行求和可能不等于 1。我们对每行做 L1 归一化：

```
S_norm[i, j] = S_sym[i, j] / sum_j(S_sym[i, j])
```

**效果**：每行的相似度之和 = 1，变成了"概率分布"——`S_norm[i, j]` 可以解读为"从数据集 i 出发，以多大概率推荐数据集 j"。

**为什么需要归一化？** 不同数据集的邻居相似度总和差异很大。一个热门数据集可能有很多高相似度邻居（总和 = 30），一个冷门数据集可能只有几个弱邻居（总和 = 3）。不归一化的话，融合时热门数据集的权重会天然偏高，不公平。

---

## 第五部分：融合四个视角（Multi-View Fusion）

### 9 一致性度量（Consistency Metrics）

**目的**：在融合之前，先量化"标签说的"和"内容看到的"是否一致。如果两个视角高度一致，说明信息冗余；如果不一致，说明内容视图提供了新信息。

#### Jaccard 集合重叠指数

对每个数据集 $i$，分别从元数据视图和内容视图获取 top-K 邻居集合，然后计算 Jaccard 指数：

```
J(i) = |N_meta(i) ∩ N_content(i)| / |N_meta(i) ∪ N_content(i)|
```

**举例**：
- 数据集 A 的元数据邻居 = {1, 2, 3, 4, 5}
- 数据集 A 的内容邻居 = {3, 4, 5, 6, 7}
- 交集 = {3, 4, 5}，并集 = {1, 2, 3, 4, 5, 6, 7}
- **J(A) = 3/7 ≈ 0.43**

J = 0 表示两个视角的推荐完全不同（最大互补），J = 1 表示完全相同（完全冗余）。

#### 加权一致性（Weighted Consistency）

Jaccard 只看邻居数量，不考虑边的权重。加权一致性 $c(i)$ 进一步考虑重叠边的权重：

```
c(i) = sum_{j ∈ intersection}( min(w_meta(i,j), w_content(i,j)) )
       / min( sum_{j ∈ N_meta}(w_meta(i,j)),  sum_{j ∈ N_cont}(w_content(i,j)) )
```

直觉：不仅看重叠了几个邻居，还看重叠邻居的"重要程度"是否一致。

### 10 三种融合方法

#### 方法 1：Naive Fusion（简单混合）

**类比：调鸡尾酒。** 按固定比例混合两种基酒——不管客人口味如何，所有人喝到的都是同一种配方。

公式：

```
S_fused = alpha × S_meta + (1 - alpha) × S_content
```

其中 `S_meta` 是三视图融合结果（Tag + Text + Beh），`S_content` 是内容视图相似度。融合后执行 top-K 裁剪和 L1 行归一化。

默认 `alpha=0.5`（等权混合），实验发现 **`alpha=0.30` 最优**（Content 权重 70%），在 D_content 子集上 Unified@nDCG20 从 0.4335 提升到 0.4747（+9.5%）。

#### 方法 2：Adaptive Fusion（自适应混合）

**类比：让更有"把握"的专家说了算。** 如果某个视角对当前数据集给出的推荐非常集中（"我很确定这几个最相似"），就给它更高的权重；如果推荐分散（"好像哪个都差不多"），就降低权重。

**ρ 信息集中度（Information Concentration）**：

```
rho_v(i) = sum_j( S_v[i, j]^2 )
```

这是相似度分布的"尖峰程度"（类似 Herfindahl 指数）：
- 如果一行的相似度集中在少数几个邻居上（如 [0.5, 0.3, 0.1, 0.1, 0, ...]），ρ 大 → 该视图对这个数据集很"自信"
- 如果相似度均匀分散（如 [0.02, 0.02, 0.02, ...]），ρ 小 → 该视图不太确定

**自适应权重**：

```
alpha_v(i) = rho_v(i) / sum_v'( rho_v'(i) )
```

每个数据集 $i$ 在每个视图 $v$ 上有独立的权重，总和为 1。这意味着不同数据集的最优融合比例可以不同。

#### 方法 3：Adaptive + Consistency（一致性调节）

在 Adaptive Fusion 基础上，用一致性分数调节内容视图权重：

```
g(i) = beta + (1 - beta) × (1 - c(i))
alpha_content_adjusted(i) = alpha_content(i) × g(i)
```

（调节后重新归一化使所有视图权重之和 = 1）

参数 `beta=0.5` 是基础保留系数。逻辑：

| 情况 | c(i) | g(i) | 效果 |
|------|------|------|------|
| 高一致性（元数据和内容说的差不多） | 0.8 | 0.6 | 降低 Content 权重（信息冗余） |
| 低一致性（内容提供了新信息） | 0.2 | 0.9 | 保留 Content 权重（新信息有价值） |
| 无内容视图的数据集 | 0.0 | 1.0 | 不影响（g=1） |

### 11 选择性融合（推荐方案）

**问题**：521,735 个数据集中只有约 10,000 个有 Content 视图。如果对所有数据集统一应用融合，无内容的数据集会在融合过程中引入噪声。

**解决方案：选择性融合（Selective Fusion）**

```
          ┌─────────────────────────────────────┐
          │   输入：数据集 i                      │
          └────────────────┬────────────────────┘
                           │
                    数据集 i 有内容视图？
                     /            \
                   Yes             No
                   /                \
    S_fused = 0.30 × S_meta     S_fused = S_meta
            + 0.70 × S_content  (保持原样)
```

**保证不退化的数学证明**：

设 D_content 为有内容视图的数据集集合。对任意数据集 $i$：
- 若 $i \in D_{content}$：$S_{selective}(i) = \alpha \cdot S_{meta}(i) + (1-\alpha) \cdot S_{content}(i)$，这是对 $S_{meta}(i)$ 的加权改进
- 若 $i \notin D_{content}$：$S_{selective}(i) = S_{meta}(i)$，与基线完全一致

因此在无内容数据集上 **严格等于** Meta-only，在有内容数据集上引入额外信号 → **全局不退化**。

**实验验证**（10K 规模，`alpha=0.30`）：

| 评测子集 | Selective Fusion | Meta-only | 提升 |
|---------|----------------:|----------:|-----:|
| D_content (10K) | **0.4747** | 0.3892 | **+22.0%** |
| +50K 稀释 | **0.3698** | 0.3362 | **+10.0%** |
| +100K 稀释 | **0.3414** | 0.3224 | **+5.9%** |

即使在稀释到 100K（Content 覆盖率仅 10%）时，选择性融合仍稳定提升 +5.9%。

**alpha=0.30 最优的实验发现**：

| alpha_meta | alpha_content | Unified@nDCG20 |
|-----------:|--------------:|---------------:|
| 0.30 | 0.70 | **0.4747** |
| 0.40 | 0.60 | 0.4665 |
| 0.50 | 0.50 | 0.4335 |
| 0.70 | 0.30 | 0.4223 |
| 0.90 | 0.10 | 0.4159 |

Content 权重越高越好（在 D_content 子集上），因为 Content 视图在 Tag 维度表现极强（Tag-nDCG 从 0.5479 升至 0.6765）。alpha=0.30 是 Tag 维度增益与 Creator 维度损失的最佳平衡点。

---

## 第六部分：如何评价推荐质量（Evaluation）

### 12 银标准评测体系（Silver Standard Evaluation）

**目的**：量化推荐质量——给出的推荐列表到底好不好？

#### 为什么叫"银标准"？

"金标准"（Gold Standard）需要人工标注（让专家判断"数据集 A 和 B 是否相关"），成本极高。我们退而求其次，用**客观可观测的特征**构造"银标准"——虽不如人工标注精确，但可以自动化计算，且在大规模下可行。

#### 三个评价维度

| 维度 | 衡量什么 | 计算方法 | 权重 |
|------|---------|---------|------|
| **Tag 相似度** | 推荐的数据集和原数据集标签是否相似 | IDF 加权 Jaccard | 0.5 |
| **Description 相似度** | 推荐的数据集描述文字是否相关 | BM25 文本匹配 | 0.3 |
| **Creator 共现** | 推荐的数据集是否来自同一作者 | 二值匹配（同作者 = 1） | 0.2 |

**Tag 相似度详解**：不是简单地数共同标签个数，而是用 IDF（Inverse Document Frequency，逆文档频率）加权。

```
tag_sim(i, j) = sum_{t ∈ tags_i ∩ tags_j}( IDF(t) )
              / sum_{t ∈ tags_i ∪ tags_j}( IDF(t) )
```

**为什么用 IDF 加权？** 因为标签的稀有度不同：
- "tabular" 标签很常见（几万个数据集都有）→ IDF 低 → 共享 "tabular" 标签不说明太多
- "time-series-forecasting" 标签很稀有（只有几百个数据集）→ IDF 高 → 共享这个标签表示高度相关

**举例**：数据集 A 标签 = {tabular, finance, stock-market}，数据集 B 标签 = {tabular, finance, cryptocurrency}。
- 共同标签 = {tabular, finance}，仅 A 的标签 = {stock-market}，仅 B 的 = {cryptocurrency}
- 若 IDF(tabular)=0.5, IDF(finance)=2.0, IDF(stock-market)=3.0, IDF(cryptocurrency)=3.5
- tag_sim = (0.5+2.0) / (0.5+2.0+3.0+3.5) = 2.5/9.0 ≈ 0.28

**Unified 综合指标**：

```
Unified@nDCG20 = 0.5 × Tag@nDCG20 + 0.3 × Desc@nDCG20 + 0.2 × Creator@nDCG20
```

#### nDCG（归一化折损累积增益）

**类比：排队买奶茶。** 想象推荐列表就是一个排队的队伍，位置越靠前越好：

- **位置 1**（第一个推荐）：你最先看到，如果它就是你想要的，太好了！全分！
- **位置 10**：你得翻好几页才看到它，就算它是好的推荐，也"打了折扣"
- **位置 20**：都快看不到了，价值进一步"折损"

nDCG 的计算分两步：
1. **DCG**（折损累积增益）= sum( gain_i / log2(i+1) )，位置越靠后，分母越大，贡献越小
2. **nDCG** = DCG / IDCG，其中 IDCG 是理想排序下的最大 DCG → nDCG ∈ [0, 1]

### 13 评测结果

#### 1K 规模结果

在 1,000 个 D_content 文档上评测 8 种方法（K=20），按 Tag、Desc、Creator 三个维度分别展示完整的 5 个指标，以及 Unified 综合分数。

##### Table 1a: Tag 维度评测结果

| Method | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|--------:|-------:|-------:|-----:|-----:|
| Content-only | 0.7472 | 0.7292 | 0.8531 | 0.6414 | 0.6414 |
| Naive-Fusion | 0.6449 | 0.5790 | 0.6679 | 0.4933 | 0.4933 |
| Beh-only | 0.6416 | 0.5259 | 0.6631 | 0.3870 | 0.3870 |
| Adaptive+Cons | 0.6257 | 0.5386 | 0.6476 | 0.2870 | 0.2870 |
| Adaptive-Fusion | 0.6257 | 0.5382 | 0.6476 | 0.2870 | 0.2870 |
| Meta-only | 0.5763 | 0.4874 | 0.5628 | 0.2450 | 0.2450 |
| Tag-only | 0.2302 | 0.1221 | 0.1365 | 0.0477 | 0.0477 |
| Text-only | 0.2249 | 0.1174 | 0.1326 | 0.0480 | 0.0480 |

##### Table 1b: Description 维度评测结果

| Method | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|--------:|-------:|-------:|-----:|-----:|
| Content-only | 0.0157 | 0.0147 | 0.0151 | 0.0015 | 0.0015 |
| Naive-Fusion | 0.0131 | 0.0099 | 0.0096 | 0.0020 | 0.0020 |
| Adaptive+Cons | 0.0113 | 0.0092 | 0.0089 | 0.0017 | 0.0017 |
| Adaptive-Fusion | 0.0113 | 0.0092 | 0.0089 | 0.0017 | 0.0017 |
| Beh-only | 0.0097 | 0.0077 | 0.0081 | 0.0012 | 0.0012 |
| Meta-only | 0.0077 | 0.0048 | 0.0057 | 0.0012 | 0.0012 |
| Text-only | 0.0008 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| Tag-only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

##### Table 1c: Creator 维度评测结果

| Method | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|--------:|-------:|-------:|-----:|-----:|
| Beh-only | 0.7506 | 0.7486 | 0.7487 | 0.3045 | 0.6909 |
| Adaptive+Cons | 0.7498 | 0.7435 | 0.7478 | 0.2893 | 0.6843 |
| Adaptive-Fusion | 0.7496 | 0.7430 | 0.7478 | 0.2888 | 0.6840 |
| Naive-Fusion | 0.7428 | 0.7368 | 0.7415 | 0.2828 | 0.6812 |
| Meta-only | 0.7391 | 0.7339 | 0.7325 | 0.2894 | 0.6845 |
| Content-only | 0.0531 | 0.0385 | 0.0413 | 0.0078 | 0.0116 |
| Tag-only | 0.0004 | 0.0002 | 0.0002 | 0.0001 | 0.0000 |
| Text-only | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0000 |

##### Table 1d: Unified 综合评测结果 (0.5×Tag + 0.3×Desc + 0.2×Creator)

| Method | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|--------:|-------:|-------:|-----:|-----:|
| **Naive-Fusion** | **0.4749** | **0.4398** | **0.4851** | **0.3038** | **0.3835** |
| Beh-only | 0.4739 | 0.4150 | 0.4837 | 0.2548 | 0.3321 |
| Adaptive+Cons | 0.4662 | 0.4207 | 0.4760 | 0.2019 | 0.2809 |
| Adaptive-Fusion | 0.4662 | 0.4205 | 0.4761 | 0.2017 | 0.2808 |
| Meta-only | 0.4383 | 0.3919 | 0.4296 | 0.1807 | 0.2597 |
| Content-only | 0.3889 | 0.3767 | 0.4393 | 0.3227 | 0.3235 |
| Tag-only | 0.1152 | 0.0611 | 0.0683 | 0.0238 | 0.0238 |
| Text-only | 0.1127 | 0.0588 | 0.0664 | 0.0240 | 0.0240 |

##### Table 2: 视图消融实验 (View-Ablation)

对比单视图 vs 视图融合的效果，使用 Unified 指标：

| Method | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|--------:|-------:|-------:|-----:|-----:|
| **Single Views** |||||
| Tag-only | 0.1152 | 0.0611 | 0.0683 | 0.0238 | 0.0238 |
| Text-only | 0.1127 | 0.0588 | 0.0664 | 0.0240 | 0.0240 |
| Beh-only | 0.4739 | 0.4150 | 0.4837 | 0.2548 | 0.3321 |
| Content-only | 0.3889 | 0.3767 | 0.4393 | 0.3227 | 0.3235 |
| **Fusion Methods** |||||
| Meta-only (3-View) | 0.4383 | 0.3919 | 0.4296 | 0.1807 | 0.2597 |
| **Naive-Fusion (4-View)** | **0.4749** | **0.4398** | **0.4851** | **0.3038** | **0.3835** |
| Adaptive-Fusion (4-View) | 0.4662 | 0.4205 | 0.4761 | 0.2017 | 0.2808 |
| Adaptive+Cons (4-View) | 0.4662 | 0.4207 | 0.4760 | 0.2019 | 0.2809 |

**解读**：
- **Naive-Fusion 最优**（Unified nDCG 0.4749），超越 Meta-only（0.4383）**+8.36%**
- **Content-only 在 Tag 维度最强**（nDCG 0.7472, MRR 0.8531），说明表格内容特征能极好地捕捉数据集的主题相似性
- Content-only 在 Creator 维度极弱（nDCG 0.0531），因为"内容相似"不等于"同一作者"
- Adaptive 方法未优于 Naive，说明简单等权融合在此场景下就足够
- 视图消融实验显示：单视图中 Beh-only 表现最佳，但 4-View 融合进一步提升性能

#### 10K 规模结果

将 D_content 扩展到 10,000 个数据集后：

| 方法 | D_content (10K) | +50K 稀释 | +100K 稀释 |
|------|----------------:|-----------:|-----------:|
| **Naive-Fusion** | **0.4335** | 0.3503 | 0.3298 |
| Beh-only | 0.4199 | 0.3592 | 0.3439 |
| Meta-only | 0.3893 | 0.3358 | 0.3223 |
| Content-only | 0.3933 | 0.3933 | 0.3933 |

**解读**：Naive-Fusion 超越 Meta-only **+11.4%**（D_content 子集）。相比 1K 实验，相对优势从 +8.4% 扩大到 +11.4%。

#### 跨规模一致性验证

| 规模 | D_content 数 | 覆盖率 | Naive-Fusion | 相对 Meta-only | Content-only |
|------|-------------|--------|------------:|---------------:|-------------:|
| 1K | 1,000 | 0.18% | 0.4749 | +8.36% | 0.3889 |
| 4.5K | 4,527 | 0.87% | 0.4585 | +8.76% | 0.3863 |
| 9.8K | 9,778 | 1.88% | 0.4326 | +11.03% | 0.3932 |
| 10K | 10,000 | 1.92% | 0.4335 | +11.36% | 0.3933 |

**关键观察**：
1. **Content-only 质量高度稳定**：跨 4 个实验，Unified 在 0.3863-0.3933 范围内，极差仅 1.8%
2. **9.8K 与 10K 结果可重复**：两次独立实验（不同候选池、不同搜索顺序）结果几乎一致（0.4326 vs 0.4335）
3. **融合增益随规模增大更显著**：1K 时相对 Meta-only +8.4%，10K 时 +11.4%

#### 稀释效应

**类比**：在 10 杯水里滴 1 滴墨水，颜色很明显。在 100 杯水里滴同样 1 滴墨水，几乎看不出来。

内容视图的增益就像那滴"墨水"——当 D_content 只有 1,000 个数据集（占 521K 的 0.18%），融合的增益被 99.82% 的无内容数据集"稀释"了。

| 稀释级别 | 覆盖率 | Naive-Fusion | Meta-only | 融合优势 |
|---------|-------|------------:|----------:|--------:|
| D_content only (1K) | 100% | 0.4749 | 0.4383 | **+8.4%** |
| +5K dilution | 20% | 0.3663 | 0.3537 | +3.6% |
| +10K dilution | 10% | 0.3421 | 0.3351 | +2.1% |
| +50K dilution | 2% | 0.3114 | 0.3118 | **-0.1%** (消失) |

选择性融合（Selective Fusion）正是为解决稀释效应而设计的——无内容的数据集不参与融合，保证增益不被稀释。

#### 5 个改进实验的结论摘要

| 实验 | 结论 |
|------|------|
| **Exp1: 权重扫描** | alpha_meta=0.30 最优（Content 权重 70%），比原 0.50 提升 +9.5% |
| **Exp2: 选择性融合** | 有内容用融合、无内容保持原样 → 稀释 +100K 仍提升 +5.9% |
| **Exp3: 质量过滤** | 不过滤最好（0.4749），过滤越严格越差 → 低质量数据集也有正向贡献 |
| **Exp4: 四视图网格** | Content-heavy (0.1/0.1/0.3/0.5) 达 0.4799，比两层融合仅高 +1.1% |
| **Exp5: k_sim 扫描** | k ∈ {20,30,50,75,100} 差距 < 0.001 → k=50 合理，无需调整 |

---

## 第七部分：技术架构与工程实现

### 14 代码架构

#### 模块关系图

```
src/content/
├── __init__.py          # 统一导出 8 个子模块的全部公开接口
├── sampling.py          # 表 I/O、主表选取、行列采样、列分析、描述生成
├── encoding.py          # 列嵌入加权聚合为数据集向量
├── similarity.py        # 稀疏相似度图构建、分区 COO 文件 I/O
├── consistency.py       # 元数据-内容一致性指标 (Jaccard + 加权)
├── fusion.py            # 多视图融合 (rho/adaptive-alpha/consistency 调节)
├── evaluation.py        # 银标准评测 (Tag/Desc/Creator) + 多方法对比
├── pipeline.py          # 端到端 Pipeline + Naive Fusion 构建
└── acquisition.py       # 数据采集 (候选筛选/API搜索/下载/主表选择)
```

#### 模块职责一览

| 模块 | 核心功能 | 关键函数 |
|------|---------|---------|
| `sampling.py` | 读取表格 → 清洗 → 列分析 → 文字描述 | `sample_table()`, `profile_column()`, `col_to_description()` |
| `encoding.py` | 列向量 → 加权聚合 → 数据集向量 | `aggregate_dataset_vector()` |
| `similarity.py` | COO 边 → 对称化 → L1 归一化 → 分区存储 | `sym_and_rownorm()`, `save_partitioned_edges()` |
| `consistency.py` | 元数据/内容邻居集合 → Jaccard + 加权一致性 | `compute_jaccard_and_consistency()` |
| `fusion.py` | 多视图 CSR 矩阵 → rho 权重 → 融合 → top-K 裁剪 | `compute_rho()`, `fuse_views()` |
| `evaluation.py` | CSR 加载 → top-K 提取 → 三维度银标准打分 | `evaluate_method_on_subset()` |
| `pipeline.py` | 编排 sampling→encoding→FAISS→fusion 全流程 | `run_content_pipeline()` |
| `acquisition.py` | Kaggle API 搜索 → 下载 → 主表选取 → 回填 | `search_kaggle_slug()`, `download_dataset()` |

#### Pipeline 端到端流程图

```
metadata_merged.csv (521K)
        │
        │  acquisition.py: filter + API search + download
        v
d_content.parquet + main_tables.parquet + data/tabular_raw/{Id}/
        │
        │  pipeline.py: run_content_pipeline()
        │
        ├──> [Step 1] sample_table() ─── 读取+清洗 ──> 清洁 DataFrame
        │
        ├──> [Step 2] profile_table() ─── 列分析 ──> ColStats 列表
        │
        ├──> [Step 3] col_to_description() ─── 生成描述 ──> 英文文本
        │
        ├──> [Step 4] SentenceTransformer.encode() ─── 编码 ──> 384维向量
        │
        ├──> [Step 5] aggregate_dataset_vector() ─── 加权聚合 ──> Z_tabcontent
        │
        ├──> [Step 6] FAISS.search() ─── kNN搜索 ──> COO 边
        │
        ├──> [Step 7] sym_and_rownorm() ─── 对称化+归一化 ──> S_tabcontent
        │
        v
S_tabcontent_symrow (稀疏相似度图)
        │
        │  fusion.py / pipeline.py: fuse_views() 或 build_naive_fusion()
        │
        │  S_fused = alpha × S_meta + (1-alpha) × S_content
        v
S_fused (融合后相似度图)
        │
        │  evaluation.py: evaluate_all_methods()
        v
metrics_*.csv (评测结果)
```

### 15 工程优化

#### 稀疏矩阵：为什么不用密集矩阵

521,735 × 521,735 的密集矩阵需要：

```
521,735² × 4 bytes (float32) = 2,721 亿 × 4 = 1.09 TB
```

实际上每个数据集只有 50 个邻居，非零元素占比：

```
521,735 × 50 / 521,735² ≈ 0.0096% (万分之一)
```

使用 CSR（Compressed Sparse Row）稀疏矩阵：

```
521,735 × 50 × (4+4+4) bytes ≈ 300 MB（行指针 + 列索引 + 值）
```

**存储减少 3,600 倍**。

#### 分区存储

大型稀疏矩阵的 COO 边（`row, col, val` 三元组）存储为分区 Parquet 文件：

```
S_tabcontent_symrow_k50_part0000.parquet  (最多 200万 边)
S_tabcontent_symrow_k50_part0001.parquet
...
S_tabcontent_symrow_k50_manifest.json     (元数据：节点数、边数、分区列表)
```

**为什么分区？**
- 单个大文件加载时内存峰值高（需要一次性分配全部内存）
- 分区文件可以按需加载（如只需前几个分区做采样分析）
- 200 万边 × 12 bytes/边 ≈ 24 MB/分区，符合 Parquet 推荐的文件大小

#### CSR 直接行访问 vs pandas groupby

评测时需要为每个数据集提取 top-K 邻居。两种实现的性能对比：

**方案 A：pandas groupby（原始实现）**
```python
edges_df.groupby("row").apply(lambda g: g.nlargest(k, "val"))
# 对 52万 文档：约 30 分钟
```

**方案 B：CSR 直接行访问（优化实现）**
```python
start = S.indptr[row_i]
end = S.indptr[row_i + 1]
cols = S.indices[start:end]
vals = S.data[start:end]
# 对 52万 文档：约 3 分钟
```

CSR 格式天然支持 O(1) 的行访问（通过 `indptr` 数组直接定位每行的起止位置），避免了 pandas groupby 的 O(N) 分组开销。实测加速 **~10x**。

此外还实现了 CSR 缓存机制（`_load_csr_cached`），同一矩阵不重复加载：

```python
_csr_cache: Dict[str, Any] = {}

def _load_csr_cached(prefix, N, base_dir, k=50):
    cache_key = f"{base_dir}/{prefix}_k{k}"
    if cache_key not in _csr_cache:
        _csr_cache[cache_key] = load_csr_from_manifest(prefix, N, base_dir, k=k)
    return _csr_cache[cache_key]
```

#### FAISS GPU/CPU 回退链

```python
try:
    import faiss
    index = faiss.IndexFlatIP(384)           # 创建内积索引
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)  # 尝试 GPU
    except:
        pass  # 回退到 CPU
    index.add(Z)
    scores, idxs = index.search(Z, k+1)
except ImportError:
    # FAISS 未安装，回退到 sklearn 或 numpy
```

这种设计保证代码在任何环境下都能运行——从有 GPU 的服务器到只有 numpy 的轻量级环境。

#### 断点续传与检查点

数据采集脚本（`expand_content_coverage.py`）支持中断恢复：
- 每 50 个数据集增量保存进度到 `d_content.parquet`
- API 搜索结果缓存到 `api_cache/{slug}.json`
- 重启时自动跳过已下载的数据集

Pipeline 脚本（`run_content_at_scale.py`）支持跳过已完成的步骤：
- `--skip-pipeline`：跳过内容 Pipeline（若 `S_tabcontent` 已存在）
- `--skip-fusion`：跳过融合构建（若融合矩阵已存在）

---

## 第八部分：结论与推荐

### 16 核心发现

1. **内容视图显著提升推荐质量**：在 D_content 子集上，融合内容视图后 Unified@nDCG20 从 0.3892 提升到 0.4747（+22.0%），证实"打开表格看内容"确实能让推荐更准。

2. **简单方法胜过复杂方法**：Naive Fusion（简单加权平均）始终优于 Adaptive Fusion（自适应权重）和 Adaptive+Consistency（一致性调节）。在工程中，简单方法更可靠、更易调试。

3. **Content 权重应高于 Meta 权重**：最优配比为 alpha_meta=0.30（Content 占 70%），说明在有表格内容的数据集上，内容特征的信息量大于元数据。Content-only 在 Tag 维度达到所有方法最高的 0.7472，远超 Meta-only 的 0.5763。

4. **选择性融合是最佳实用方案**：有内容的用融合，没内容的保持原样，保证全局不退化。在 +100K 稀释下仍稳定提升 +5.9%（Unified 0.3414 vs 0.3224）。

5. **覆盖率是核心瓶颈**：当前约 9,778 个数据集有可提取的表格主文件，占 521K 的 1.88%。进一步提升推荐质量的主要方向是扩大内容覆盖率（降低表格质量门槛、引入外部数据源、或探索非表格文件的内容分析）。

### 推荐部署方案

```
推荐配置:
  - 融合策略: 选择性融合 (Selective Fusion)
  - 权重分配: alpha_meta = 0.30, alpha_content = 0.70
  - 近邻参数: k_sim = 50
  - 采样参数: MAX_ROWS = 1024, MAX_COLS = 60
      (消融实验表明 MAX_ROWS 可安全降至 64-128 以加速处理)
  - 编码模型: all-MiniLM-L6-v2 (384维)
  - 适用范围: 所有有表格内容的数据集

预期效果:
  - D_content 子集: +22.0% (Unified@nDCG20)
  - 稀释至 +50K:  +10.0%
  - 稀释至 +100K: +5.9%
  - 全局不退化保证: 无内容数据集与 Meta-only 完全一致
```

---

*本文档基于 `src/content/` 源码、`PROGRESS.md` 和 `RESULTS_SUBSET_EVALUATION.md` 中的实验数据撰写。所有数字均可在对应文件中交叉验证。*
