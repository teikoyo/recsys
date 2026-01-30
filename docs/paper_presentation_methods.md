# 论文中代码展示方法指南
## Presentation Methods for Each Step in Academic Paper

---

## STEP 1: Environment & Utilities Setup

### 推荐展示方法：
❌ **不展示** - 这是实现细节，不适合放在论文正文中

### 可选处理：
- 在附录中简单列出：技术栈表格
- 补充材料中提供完整的依赖列表

### 示例展示：

**Table 1: Implementation Environment**
| Component | Technology | Version |
|-----------|------------|---------|
| Deep Learning | PyTorch | 2.0+ |
| ANN Search | FAISS | 1.7+ |
| Sparse Matrix | SciPy | 1.9+ |
| GPU | CUDA | 11.8+ |

---

## STEP 2: Data Preprocessing & Cleaning

### 推荐展示方法：

#### 1. **数据流程图 (Flowchart)**
```
┌─────────────┐
│  Raw CSV    │
└──────┬──────┘
       │
       ├─► Remove HTML tags
       ├─► Remove URLs
       ├─► Lowercase
       ├─► Parse tags
       │
       ▼
┌──────────────────┐
│ Cleaned Dataset  │
└──────────────────┘
```

#### 2. **示例对比表格 (Before/After Table)**

**Table 2: Text Cleaning Examples**
| Step | Before | After |
|------|--------|-------|
| HTML Removal | `<p>Machine Learning</p>` | `Machine Learning` |
| URL Removal | `Check https://example.com` | `Check` |
| Tag Parsing | `"ml\|ai\|dl"` | `["ml", "ai", "dl"]` |

#### 3. **简化的伪代码 (High-level Pseudocode)**

```
Algorithm 1: Data Preprocessing
Input: Raw documents D_raw
Output: Cleaned documents D_clean

1: for each document d in D_raw do
2:    text ← Remove_HTML(d.text)
3:    text ← Remove_URLs(text)
4:    text ← Lowercase(text)
5:    tags ← Parse_Tags(d.tags)
6:    D_clean.add(document(text, tags))
7: return D_clean
```

#### 4. **统计摘要表 (Statistics Table)**

**Table 3: Dataset Statistics**
| Metric | Value |
|--------|-------|
| Total Documents | 521,735 |
| Documents with Tags | 214,603 (41.1%) |
| Avg Text Length | 1,247 words |
| Unique Tags | 597 |

---

## STEP 3: Tag View Construction (D-T Matrix)

### 推荐展示方法：

#### 1. **数学公式 (Mathematical Formulas)**

**TF-IDF 权重:**
```latex
w_{ij}^{TFIDF} = tf_{ij} \times \log\frac{N}{df_j}
```
where $tf_{ij}$ is term frequency, $N$ is total documents, $df_j$ is document frequency.

**PPMI 权重:**
```latex
PPMI(d,t) = \max\left(0, \log\frac{P(d,t)}{P(d)P(t)}\right) = \max\left(0, \log\frac{N \cdot f_{dt}}{\sum_t f_{dt} \cdot \sum_d f_{dt}}\right)
```

#### 2. **矩阵构建示意图 (Matrix Construction Diagram)**

```
Documents:  [d1, d2, d3, ..., dN]
              ↓
Tags:      [t1, t2, t3, ..., tV]

D-T Matrix (N × V):
         t1   t2   t3  ...  tV
    d1 [ 0.3  0    0.7 ...  0  ]
    d2 [ 0    0.5  0   ...  0.2]
    d3 [ 0.1  0    0   ...  0  ]
    ...
    dN [ 0    0.4  0   ...  0  ]
```

#### 3. **算法框架 (Algorithm Framework)**

```
Algorithm 2: Multi-Weight Tag Matrix Construction
Input: Documents D, Tag vocabulary V, weighting scheme W
Output: Document-Tag matrix M ∈ ℝ^{N×V}

1: Initialize M ← sparse_matrix(N, V)
2: for each document d_i in D do
3:    for each tag t_j in d_i.tags do
4:       if W == "Binary" then
5:          M[i,j] ← 1
6:       else if W == "TF-IDF" then
7:          M[i,j] ← log(N / df_j)
8:       else if W == "PPMI" then
9:          M[i,j] ← max(0, PMI(d_i, t_j))
10: M ← RowNormalize(M, L2)
11: return M
```

#### 4. **对比实验表 (Comparison Table)**

**Table 4: Weighting Scheme Comparison**
| Scheme | Sparsity | Range | Semantic Strength |
|--------|----------|-------|-------------------|
| Binary | 99.2% | {0, 1} | Weak |
| TF-IDF | 99.2% | [0, 8.5] | Medium |
| PPMI | 99.2% | [0, 12.3] | Strong |

---

## STEP 4: Text View Construction (D-W BM25 Matrix)

### 推荐展示方法：

#### 1. **BM25 公式 (Formula)**

```latex
BM25(d,w) = IDF(w) \cdot \frac{f(d,w) \cdot (k_1 + 1)}{f(d,w) + k_1 \cdot (1-b + b \cdot \frac{|d|}{avgdl})}
```

where:
- $f(d,w)$: frequency of word $w$ in document $d$
- $|d|$: length of document $d$
- $avgdl$: average document length
- $k_1 = 1.5$, $b = 0.75$: tuning parameters

#### 2. **算法伪代码**

```
Algorithm 3: BM25 Text Matrix Construction
Input: Documents D, vocabulary V, k1=1.5, b=0.75
Output: Document-Word matrix M_BM25

1: Tokenize all documents → word counts
2: Build vocabulary V with min_df and max_df filters
3: Compute avgdl ← mean(document_lengths)
4: for each non-zero entry (d,w) do
5:    tf ← count(w in d)
6:    df ← number of docs containing w
7:    idf ← log((N - df + 0.5) / (df + 0.5))
8:    norm ← tf + k1 × (1 - b + b × |d| / avgdl)
9:    M_BM25[d,w] ← idf × (tf × (k1 + 1)) / norm
10: M_BM25 ← RowNormalize(M_BM25, L2)
11: return M_BM25
```

#### 3. **参数敏感性图 (Parameter Sensitivity Plot)**

展示 k1 和 b 参数对检索性能的影响（可选）

---

## STEP 5: GPU Random Walk Generator

### 推荐展示方法：

#### 1. **类型约束游走示意图 (Constrained Walk Diagram)**

```
Graph Structure:
    Documents (D) ←→ Tags/Words (X)

Walk Pattern: D → X → D → X → D
Example:
    doc_5 → tag_"ML" → doc_12 → tag_"AI" → doc_23
      ↓         ↓          ↓         ↓          ↓
    [   5   ,    12   ,    23   ,    ...     ]
    (only document IDs retained)
```

#### 2. **游走生成算法 (Walk Generation Algorithm)**

```
Algorithm 4: Type-Constrained Random Walk
Input: Bipartite graph G(D,X,E), walk_length L, walks_per_node K
Output: Walk sequences W

1: for each document d in D do
2:    for k = 1 to K do
3:       walk ← [d]
4:       current ← d, type ← 'D'
5:       for step = 1 to L-1 do
6:          if type == 'D' then
7:             x ← Sample_Neighbor(current, D→X, weighted)
8:             current ← x, type ← 'X'
9:          else
10:            d' ← Sample_Neighbor(current, X→D, weighted)
11:            walk.append(d')
12:            current ← d', type ← 'D'
13:       W.add(walk)
14: return W
```

#### 3. **二部图示意图 (Bipartite Graph Illustration)**

```
   Documents          Tags/Words
   ┌─────┐           ┌─────┐
   │ d1  │───────────│ t1  │
   └─────┘     ╲     └─────┘
   ┌─────┐      ╲    ┌─────┐
   │ d2  │───────────│ t2  │
   └─────┘       ╲   └─────┘
   ┌─────┐        ╲  ┌─────┐
   │ d3  │───────────│ t3  │
   └─────┘           └─────┘
```

#### 4. **超参数表 (Hyperparameters Table)**

**Table 5: Random Walk Hyperparameters**
| Parameter | Tag View | Text View | Description |
|-----------|----------|-----------|-------------|
| walk_length | 10 | 10 | Nodes per walk |
| walks_per_node | 20 | 20 | Walks per document |
| Total walks | N × 20 | N × 20 | Total sequences |

---

## STEP 6: WS-SGNS Training (Skip-Gram + Negative Sampling)

### 推荐展示方法：

#### 1. **模型架构图 (Model Architecture Diagram)**

```
┌──────────────────────────────────────────┐
│          Skip-Gram Architecture          │
├──────────────────────────────────────────┤
│                                          │
│  Input: center_id (doc)                  │
│         ↓                                │
│  ┌──────────────┐                        │
│  │ Embedding_in │  [V × D]               │
│  └──────┬───────┘                        │
│         │ center_emb [D]                 │
│         ├─────────────────┐              │
│         ↓                 ↓              │
│  ┌─────────────┐   ┌──────────────┐     │
│  │Embedding_out│   │ Embedding_out│     │
│  │(positive)   │   │ (negatives)  │     │
│  └──────┬──────┘   └──────┬───────┘     │
│         │                 │              │
│         ↓                 ↓              │
│   Dot Product       Dot Products         │
│         │                 │              │
│         ↓                 ↓              │
│    Sigmoid           Sigmoid             │
│         │                 │              │
│         └────────┬────────┘              │
│                  ↓                       │
│            NCE Loss                      │
└──────────────────────────────────────────┘
```

#### 2. **损失函数 (Loss Function)**

```latex
\mathcal{L} = -\log \sigma(\mathbf{u}_c^T \mathbf{v}_p) - \sum_{i=1}^{K} \log \sigma(-\mathbf{u}_c^T \mathbf{v}_{n_i})
```

where:
- $\mathbf{u}_c$: center node embedding (input)
- $\mathbf{v}_p$: positive context embedding (output)
- $\mathbf{v}_{n_i}$: negative sample embeddings
- $K$: number of negative samples

#### 3. **训练流程图 (Training Pipeline)**

```
Walk Sequences
      ↓
Extract (center, context) pairs
      ↓
Sample K negatives per pair
      ↓
Mini-batch: (centers, positives, negatives)
      ↓
Forward Pass → Compute Loss
      ↓
Backward Pass → Update Embeddings
      ↓
Repeat for N epochs
      ↓
Extract Input Embeddings → Z
```

#### 4. **训练配置表 (Training Configuration)**

**Table 6: SGNS Training Hyperparameters**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding Dimension | 128 | Size of embedding vectors |
| Window Size | 5 | Context window radius |
| Negative Samples | 10 | K negative samples per positive |
| Batch Size | 2048 | Training batch size |
| Learning Rate | 0.001 | Adam optimizer LR |
| Epochs | 10 | Training iterations |
| Negative Distribution | $p \propto deg^{0.75}$ | Subsampling strategy |

---

## STEP 7: ANN Graph Construction (FAISS K-NN)

### 推荐展示方法：

#### 1. **K-NN 搜索流程图 (K-NN Search Pipeline)**

```
Embeddings Z [N × D]
      ↓
Build FAISS Index (GPU)
      ↓
For each query vector z_i:
   Search K nearest neighbors
   ↓
   Return: {neighbor_ids, similarities}
      ↓
Construct Similarity Graph: (i, j, sim_ij)
      ↓
Save as Sparse Matrix (Triplets)
```

#### 2. **相似度计算公式 (Similarity Formula)**

```latex
\text{sim}(d_i, d_j) = \frac{\mathbf{z}_i^T \mathbf{z}_j}{\|\mathbf{z}_i\| \|\mathbf{z}_j\|} = \mathbf{z}_i^T \mathbf{z}_j
```
(assuming L2-normalized embeddings)

#### 3. **图构建示意图 (Graph Construction Illustration)**

```
Embedding Space (128-D)
       ┌─────────────────────────┐
       │    • d1                 │
       │         • d5            │
       │  • d3      • d2         │
       │       • d4              │
       └─────────────────────────┘
              ↓ FAISS K-NN (K=50)

Similarity Graph (Sparse)
    d1 → [d2, d5, d3, ...]  (top-50)
    d2 → [d1, d4, d5, ...]
    d3 → [d1, d4, d6, ...]
    ...
```

#### 4. **计算复杂度分析表 (Complexity Analysis)**

**Table 7: ANN Search Complexity**
| Method | Index Time | Query Time | Memory |
|--------|------------|------------|--------|
| Exact (Flat IP) | O(ND) | O(ND) | O(ND) |
| IVF-Flat | O(ND) | O(√N × D) | O(ND) |
| HNSW | O(N log N × D) | O(log N × D) | O(ND) |

---

## STEP 8: Graph Symmetrization + Row Normalization

### 推荐展示方法：

#### 1. **对称化方法对比 (Symmetrization Methods)**

```latex
\text{Max:} \quad A_{sym}[i,j] = \max(A[i,j], A[j,i])
```

```latex
\text{Avg:} \quad A_{sym}[i,j] = \frac{A[i,j] + A[j,i]}{2}
```

```latex
\text{Union:} \quad A_{sym}[i,j] = \begin{cases}
A[i,j] & \text{if } (i,j) \in E \\
A[j,i] & \text{if } (j,i) \in E, (i,j) \notin E
\end{cases}
```

#### 2. **归一化公式 (Normalization Formula)**

```latex
P[i,j] = \frac{A_{sym}[i,j]}{\sum_k A_{sym}[i,k]}
```
(Row-stochastic transition matrix)

#### 3. **图变换流程 (Graph Transformation Pipeline)**

```
Directed Graph A
      ↓
Symmetrization (max/avg/union)
      ↓
Undirected Graph A_sym
      ↓
Row Normalization: P = D^{-1} A_sym
      ↓
Transition Matrix P
```

---

## STEP 9: Multi-View Fusion (Adaptive Fusion)

### 推荐展示方法：

#### 1. **自适应融合算法 (Adaptive Fusion Algorithm)**

```
Algorithm 5: Concentration-based Adaptive Fusion
Input: Graphs A, B; top_k
Output: Fused graph F

1: for i = 1 to N do
2:    conc_A[i] ← ||A[i,:]||²₂  // Row concentration
3:    conc_B[i] ← ||B[i,:]||²₂
4:    α_A[i] ← 1 / (conc_A[i] + ε)  // Inverse weighting
5:    α_B[i] ← 1 / (conc_B[i] + ε)
6: A' ← DiagScale(α_A) × A  // Scale rows
7: B' ← DiagScale(α_B) × B
8: F ← A' + B'  // Element-wise sum
9: for i = 1 to N do
10:   Keep top-k values in F[i,:]
11:   F[i,:] ← F[i,:] / ||F[i,:]||₂  // L2 normalize
12: return F
```

#### 2. **浓度权重示意图 (Concentration Weighting Illustration)**

```
Graph A (focused):          Graph B (diffuse):
Row i: [0.9, 0.05, 0.05]   Row i: [0.2, 0.2, 0.2, 0.2, 0.2]
conc_A = 0.815              conc_B = 0.2
α_A = 1.23                  α_B = 5.0  ← Higher weight!

Intuition: Uncertain rows get more weight
           to learn from other view
```

#### 3. **多视图融合架构 (Multi-view Fusion Architecture)**

```
┌─────────────────────────────────────────┐
│       Multi-View Fusion Framework       │
├─────────────────────────────────────────┤
│                                         │
│  Tag View (S_tag)    Text View (S_text)│
│       ↓                    ↓            │
│       └────────┬───────────┘            │
│                ↓                        │
│       Adaptive Fusion                  │
│                ↓                        │
│       S_tag_text_fused                 │
│                ↓                        │
│       ┌────────┴────────┐              │
│       ↓                 ↓              │
│  S_tag_text      Behavior View         │
│                      (S_beh)            │
│       └────────┬────────┘               │
│                ↓                        │
│       Final Fusion                     │
│                ↓                        │
│       S_final (Recommendation Graph)   │
└─────────────────────────────────────────┘
```

#### 4. **融合权重可视化建议 (Fusion Weight Visualization)**

**Table 8: Adaptive Fusion Example**
| Document | Tag Conc. | Text Conc. | α_tag | α_text | Effect |
|----------|-----------|------------|-------|--------|--------|
| d₁ | 0.85 (high) | 0.12 (low) | 1.18 | 8.33 | Text view dominates |
| d₂ | 0.23 (low) | 0.67 (high) | 4.35 | 1.49 | Tag view dominates |
| d₃ | 0.45 (mid) | 0.41 (mid) | 2.22 | 2.44 | Balanced |

---

## STEP B1-B4: Behavior View Construction

### 推荐展示方法：

#### 1. **协同过滤公式 (Collaborative Filtering Formula)**

```latex
S_{ids}[i,j] = \cos(\mathbf{u}_i, \mathbf{u}_j) = \frac{\mathbf{u}_i^T \mathbf{u}_j}{||\mathbf{u}_i|| \cdot ||\mathbf{u}_j||}
```
where $\mathbf{u}_i$ is user-item interaction vector

#### 2. **行为视图流程图 (Behavior View Pipeline)**

```
User-Item Interactions
      ↓
   ┌──┴──┐
   ↓     ↓
S_ids   S_eng
(collaborative) (engagement)
   ↓     ↓
   └──┬──┘
      ↓
 Adaptive Fusion
      ↓
    S_beh
```

---

## STEP C: Three-View Total Fusion

### 推荐展示方法：

#### 1. **完整系统架构图 (Complete System Architecture)**

```
┌────────────────────────────────────────────────┐
│      Multi-View Graph Recommendation System    │
├────────────────────────────────────────────────┤
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   Tag    │  │   Text   │  │ Behavior │    │
│  │   View   │  │   View   │  │   View   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │           │
│       │  D-T Matrix │  D-W Matrix │ User-Item│
│       │  (Binary,   │  (BM25)     │ Matrix   │
│       │   TF-IDF,   │             │          │
│       │   PPMI)     │             │          │
│       ↓             ↓             ↓           │
│  Random Walks  Random Walks  Collaborative + │
│   (D→T→D)      (D→W→D)       Engagement      │
│       ↓             ↓             ↓           │
│    SGNS          SGNS         K-NN           │
│   Training      Training                     │
│       ↓             ↓             ↓           │
│   Z_tag         Z_text        S_ids, S_eng   │
│       ↓             ↓             ↓           │
│  FAISS K-NN    FAISS K-NN    Fusion          │
│       ↓             ↓             ↓           │
│   S_tag         S_text         S_beh         │
│       ↓             ↓             ↓           │
│       └──────┬──────┴──────┬──────┘          │
│              ↓             ↓                  │
│         Adaptive     Adaptive                │
│          Fusion      Fusion                  │
│              ↓             ↓                  │
│         S_tag_text ───────┘                  │
│              ↓                                │
│        Final Fusion                          │
│              ↓                                │
│      S_final (N×N)                           │
│  Recommendation Graph                        │
└────────────────────────────────────────────────┘
```

#### 2. **算法复杂度总结表 (Complexity Summary Table)**

**Table 9: Computational Complexity of Each Component**
| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Tag Matrix (TF-IDF) | O(N·T_avg) | O(N·V) sparse |
| Text Matrix (BM25) | O(N·W_avg) | O(N·W) sparse |
| Random Walks | O(N·K·L) | O(N·K·L) |
| SGNS Training | O(E·D·K_neg) | O(V·D) |
| FAISS K-NN | O(N·K·D) | O(N·D) |
| Fusion | O(N·K²) | O(N·K) sparse |
| **Total** | **O(N·K·(L+D·K_neg))** | **O(N·(V+W+D))** |

---

## 验证与评估 (Verification & Evaluation)

### 推荐展示方法：

#### 1. **邻居质量案例展示 (Neighbor Quality Examples)**

**Table 10: Sample Recommendations**
| Query Document | Top-3 Neighbors | Similarity Scores |
|----------------|-----------------|-------------------|
| "Deep Learning Tutorial" | 1. CNN Basics (0.92) | High relevance |
| | 2. PyTorch Guide (0.88) | |
| | 3. Neural Networks (0.85) | |

#### 2. **定量评估指标 (Quantitative Metrics)**

**Table 11: Evaluation Metrics**
| Metric | Value | Baseline | Improvement |
|--------|-------|----------|-------------|
| Precision@10 | 0.847 | 0.723 | +17.2% |
| Recall@50 | 0.692 | 0.581 | +19.1% |
| NDCG@20 | 0.789 | 0.694 | +13.7% |
| Coverage | 0.943 | 0.876 | +7.6% |

#### 3. **消融实验 (Ablation Study)**

**Table 12: Ablation Study Results**
| Configuration | NDCG@20 | Delta |
|---------------|---------|-------|
| Full Model | **0.789** | - |
| - Behavior View | 0.751 | -4.8% |
| - Text View | 0.723 | -8.4% |
| - Tag View | 0.698 | -11.5% |
| Only Tag View | 0.645 | -18.2% |

---

## 论文中的推荐章节结构

```
3. METHODOLOGY
   3.1 Problem Formulation
       - Mathematical notation
       - Problem definition

   3.2 Multi-View Graph Construction
       3.2.1 Tag-based View
             - Algorithm 2 (伪代码)
             - Equation for TF-IDF, PPMI (公式)
             - Table 4 (权重对比)

       3.2.2 Text-based View
             - Algorithm 3 (伪代码)
             - BM25 formula (公式)

       3.2.3 Behavior-based View
             - Collaborative filtering formula (公式)
             - Figure X (流程图)

   3.3 Graph Embedding via Random Walks
       - Algorithm 4 (游走算法)
       - Figure X (二部图示意)
       - Table 5 (超参数)

   3.4 Similarity Graph Construction
       - FAISS K-NN explanation
       - Figure X (嵌入空间示意)

   3.5 Adaptive Multi-View Fusion
       - Algorithm 5 (融合算法)
       - Equation for concentration-based weighting (公式)
       - Figure X (架构图)

4. EXPERIMENTS
   4.1 Datasets and Settings
       - Table 3 (数据集统计)
       - Table 6 (训练配置)

   4.2 Evaluation Metrics
       - Precision, Recall, NDCG definitions (公式)

   4.3 Results
       - Table 11 (评估指标)
       - Figure X (性能对比图)

   4.4 Ablation Study
       - Table 12 (消融实验)

   4.5 Case Study
       - Table 10 (案例展示)
       - Figure X (可视化示例)
```

---

## 可视化建议 (Visualization Recommendations)

### 推荐使用的图表类型：

1. **系统架构** → **Block Diagram** (方框图)
2. **算法流程** → **Flowchart** (流程图)
3. **数学公式** → **LaTeX Equations** (数学公式)
4. **复杂度分析** → **Table** (表格)
5. **超参数** → **Table** (表格)
6. **结果对比** → **Bar Chart / Line Plot** (条形图/折线图)
7. **嵌入空间** → **2D Projection (t-SNE/UMAP)** (降维可视化)
8. **邻居关系** → **Network Graph** (网络图)
9. **性能趋势** → **Line Chart** (折线图)
10. **消融实验** → **Grouped Bar Chart** (分组条形图)

---

## 工具推荐

### 绘图工具：
1. **算法流程图**: draw.io, Lucidchart, TikZ (LaTeX)
2. **架构图**: Microsoft Visio, draw.io
3. **数学公式**: LaTeX (MathJax)
4. **数据可视化**: Matplotlib, Seaborn, Plotly
5. **网络图**: NetworkX + Matplotlib, Gephi
6. **表格**: LaTeX tabular, Excel → LaTeX

### LaTeX 包推荐：
```latex
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}  % 专业表格
\usepackage{tikz}      % 绘图
```

---

## 总结

### 各步骤的核心展示方法：

| Step | 主要展示方式 | 次要展示方式 |
|------|-------------|-------------|
| STEP 1 | ❌ 不展示 | 附录：技术栈表格 |
| STEP 2 | 流程图 + 统计表 | 示例对比表 |
| STEP 3 | 公式 + 伪代码 | 对比表格 |
| STEP 4 | 公式 + 伪代码 | 参数表 |
| STEP 5 | 算法 + 示意图 | 超参数表 |
| STEP 6 | 架构图 + 公式 | 训练配置表 |
| STEP 7 | 流程图 + 公式 | 复杂度表 |
| STEP 8 | 公式对比 | 流程图 |
| STEP 9 | 算法 + 架构图 | 案例表 |
| STEP B | 公式 + 流程图 | - |
| STEP C | 完整架构图 | 复杂度总表 |
| Eval | 性能表 + 消融表 | 案例展示 |

### 优先级建议：

**必须包含 (Must-have)**:
- 完整系统架构图（STEP C）
- 核心算法伪代码（STEP 3, 4, 5, 6, 9）
- 关键数学公式（TF-IDF, PPMI, BM25, SGNS loss, Fusion）
- 评估结果表格（Metrics, Ablation）

**强烈推荐 (Highly recommended)**:
- 各视图构建流程图
- 超参数配置表
- 消融实验对比
- 案例展示

**可选 (Optional)**:
- 数据预处理流程
- 复杂度分析表
- 可视化示例（嵌入空间、网络图）
