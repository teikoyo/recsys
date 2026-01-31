# 内容视图扩展研究计划（Content View Extension）

> ⚠️ **状态：研究计划草案（未实现）**
> 本文档描述一项拟议的扩展研究，尚未开始实施。
> 主系统方案见 [RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md)。

---

## 与主系统的关系

本扩展建立在 `RESEARCH_PROPOSAL.md` 中定义的三视图推荐系统之上，涉及以下章节：

- **§1.3 问题定义**：主系统严格遵守"仅使用元数据"约束。本扩展突破该约束，在受控预算下引入少量实际表格内容。
- **§3.5 融合策略**：主系统使用 CombSUM / Fused3-RA 等三视图融合方法。本扩展将其扩展为四视图融合，并引入 ρ 自适应权重。
- **§4 评测框架**：本扩展完全复用三维度银标准和 Unified@nDCG 指标，实验结果与主系统基线直接可比。
- **§6.6 ANN 索引格式**：内容视图相似矩阵采用与现有视图一致的 COO parquet 三元组格式（`row/col/val` + `manifest.json`）。

---

## 1. 扩展目标与研究动机

§1–§7 定义的系统严格遵守"仅使用元数据"约束（§1.3），这一设计在标签覆盖率仅 41.1% 的条件下已取得 Unified@nDCG = 0.468（Fused3-Blend）。然而，元数据无法刻画数据集的**内在结构特征**——例如列名、数据类型分布、数值范围等直接反映数据集内容的信息。

本扩展的核心假设是：**即使在极低采样预算下（仅获取少量数据集的主表文件），表格结构特征也能提供元数据视图无法捕获的互补信号**。

**三个具体目标**：

1. **构建内容视图相似矩阵** `S_tabcontent_symrow`：基于表格列描述的 sentence-transformer 编码，格式与现有视图矩阵（§6.6）一致。
2. **定义元数据–内容一致性指标**：延续已有视图间一致性分析（`tmp/consistency_scores.parquet`，521,735 行 × 4 列），新增 meta-vs-content 维度。
3. **四视图融合** `S_fused4_symrow`：在现有三视图融合（§3.5 CombSUM / Fused3-RA）基础上引入内容视图，评估性能提升。

**与现有系统的关系**：

- 三视图融合结果（§3.5）作为黑盒基线输入，无需修改现有管线。
- 评测框架（§4）完全复用，包括三维度银标准和 Unified@nDCG 指标。
- 数据源仍为 Meta Kaggle（§1.2），通过 Kaggle 搜索 API 定位元数据对应的实际数据集并下载少量表格文件。
- 代码放置于已预留的 `notebooks/04_content/` 和 `src/content/` 目录。

---

## 2. 数据获取与定位

核心思路：`metadata_merged.csv` 中的每条记录都对应 Kaggle 平台上一个真实数据集。通过 Slug 和 Title 在 Kaggle 平台搜索，即可定位并下载实际数据集文件——无需额外的用户映射表或文件元信息表。

### 2.1 现有数据资产

现有 `data/raw_data/` 包含以下四张原始表及一张合并表：

| 文件 | 行数 | 关键字段 |
|------|------|----------|
| `Datasets.csv` | 521,735 | Id, CreatorUserId, OwnerUserId, TotalViews, TotalDownloads, CurrentDatasetVersionId |
| `DatasetVersions.csv` | 11,646,570 | DatasetId, Slug, Title, Description, TotalCompressedBytes |
| `DatasetTags.csv` | 446,054 | DatasetId, TagId |
| `Tags.csv` | 831 | Id, Name, FullPath |
| `metadata_merged.csv` | 521,735 | 上述合并 + Slug, Tags（逗号分隔）, TotalViews_log1p 等 |

**关键字段**：`Slug`（如 `heart-failure-clinical-data`）和 `Title`（如 `Heart Failure Prediction`）是在 Kaggle 平台定位实际数据集的主要检索键。

### 2.2 通过 Kaggle 搜索定位实际数据集

`metadata_merged.csv` 中的 Slug 仅含数据集名称部分，不含 owner 前缀。但 Kaggle API 的搜索功能可直接用 Slug 或 Title 检索，返回结果包含完整的 `{owner}/{slug}` 引用（`ref` 字段）：

```bash
kaggle datasets list -s "{slug}" --csv --max-size {size_mb}M
```

**匹配策略**：

1. **Slug 精确匹配（首选）**：搜索结果的 `ref` 字段以 `/{slug}` 结尾，即为目标数据集。大多数 Slug 具有足够的唯一性，可直接定位。
2. **Title 辅助确认**：当 Slug 匹配到多个结果时，比较 Title 文本相似度消歧。
3. **元数据交叉验证**：对匹配结果与本地元数据进行一致性检查（TotalViews 数量级、Description 前缀匹配等），排除同名但不同的数据集。

**输出**：匹配成功的记录生成映射表 `slug_to_ref.csv`：`(DatasetId, Slug, ref, match_confidence)`，缓存匹配结果以避免重复 API 调用。

### 2.3 候选数据集筛选

从 521,735 个数据集中筛选适合获取内容的候选子集：

**标签筛选——识别结构化数据集**：

| 标签范围 | 匹配数据集数 |
|----------|-------------|
| 核心标签：`tabular` | 11,621 |
| 扩展标签：`classification` / `regression` / `exploratory data analysis` / `data analytics` | 合计去重 29,866 |
| 有任意标签的数据集 | 214,603（占全量 41.1%） |

**文件大小过滤**：

`DatasetVersions.TotalCompressedBytes`（519,988 个数据集有此值）过滤合理范围：

- **下限 100KB**：排除空数据集或 toy 示例
- **上限 100MB**：排除可能含图片、视频等非结构化内容的大型数据集
- 该范围内共 209,551 个数据集

**候选池规模**：广义结构化标签 ∩ 合理文件大小 ∩ TotalViews ≥ 1,000 → tabular-tagged 约 6,391 个，广义结构化约 16,853 个，足以支撑 B_ds ∈ {500, 1000, 2000} 的预算。

### 2.4 内容子集 D_content 选择

从候选池中按 `TotalDownloads` 降序选取 top-B_ds 个数据集：

1. **预算参数** `B_ds ∈ {500, 1000, 2000}`，主实验使用 `B_ds = 1000`。
2. **覆盖率约束**：高下载量数据集与银标准评测集（有标签的 214,603 个数据集）重叠度最高，top-1000 足以覆盖高价值区间。
3. **搜索匹配**：对选中的 B_ds 个数据集执行 §2.2 的 Kaggle 搜索匹配流程，获取 `ref`。匹配失败的跳过，依次递补。

**代表性高曝光 tabular 数据集（真实数据）**：

| Id | Title | Slug | TotalViews | TotalDownloads |
|----|-------|------|------------|----------------|
| 727551 | Heart Failure Prediction | heart-failure-clinical-data | 1,103,040 | 167,625 |
| 199387 | US Accidents (2016-2023) | us-accidents | 1,018,812 | 153,799 |
| 14872 | Graduate Admission 2 | graduate-admissions | 706,533 | 120,040 |
| 522275 | Airline Passenger Satisfaction | airline-passenger-satisfaction | 648,668 | 112,097 |
| 468218 | Formula 1 World Championship | formula-1-world-championship-1950-2020 | 649,563 | 130,432 |
| 20 | Open Food Facts | world-food-facts | 517,106 | 79,782 |
| 1019790 | HR Analytics: Job Change | hr-analytics-job-change-of-data-scientists | 474,009 | 68,141 |
| 1046158 | Crop Recommendation Dataset | crop-recommendation-dataset | 396,980 | 67,410 |

### 2.5 下载与主表选择

**下载**：使用 §2.2 匹配到的 `ref` 直接下载：

```bash
kaggle datasets download -d {ref} -p data/tabular_raw/{DatasetId}/ --unzip
```

**主表选择**（下载后本地扫描）：

1. 扫描解压目录，保留 tabular 扩展名：`.csv` / `.tsv` / `.parquet` / `.xlsx` / `.xls`。
2. **黑名单过滤**：排除 `sample_submission*`、`*_submission*`、`*_dictionary*`、`*_codebook*`、`README*`、`LICENSE*`。
3. **按文件大小降序**，选取最大的 tabular 文件作为主表。
4. 若无 tabular 文件，标记为非表格类型，从 D_content 中移除，依次递补。

**统一存储**：`data/tabular_raw/{DatasetId}/main_table.{ext}`。

---

## 3. 内容视图构建方法

### 3.1 低成本表格采样

为控制存储和计算成本，对每个主表进行行列采样：

- **行采样**：`MAX_ROWS ∈ {256, 1024, 2048}`，主实验使用 1024 行。从表头开始读取（`pd.read_csv(..., nrows=MAX_ROWS)`），避免随机采样带来的 I/O 开销。
- **列过滤**：去除全空列、常数列（唯一值 ≤ 1）、明显 ID 列（列名匹配 `*_id`、`id`、`index` 且唯一率 > 95%）。
- **列预算**：`MAX_COLS ∈ {30, 60}`，按信息量（唯一值比例 × 非空比例）降序选取。

```
def sample_table(table_path, max_rows=1024, max_cols=60):
    # 按扩展名读取前 max_rows 行（csv/tsv/parquet/xlsx）
    df = read_by_ext(table_path, nrows=max_rows)

    # 列过滤：全空列、常数列、高唯一率 ID 列（*_id/index/unnamed 且 >95%）
    df = drop_empty_const_id_cols(df)

    # 列预算：按 信息量=唯一率×非空率 降序取 top-max_cols
    if len(df.columns) > max_cols:
        scores = {c: nunique_ratio(c) * notna_ratio(c) for c in df.columns}
        df = df[top_k_by_score(scores, max_cols)]
    return df
```

### 3.2 列类型识别

对采样后的每列进行类型判定：

| 类型 | 判定条件 | 采集统计量 |
|------|---------|-----------|
| **numeric** | `pd.to_numeric` 成功比例 ≥ τ_num=0.95 | min, Q1, median, Q3, max, mean, std, 缺失率 |
| **datetime** | `pd.to_datetime` 成功比例 ≥ τ_dt=0.9 | 最早日期, 最晚日期, 时间跨度 |
| **categorical** | 非 numeric/datetime 且平均字符长度 < L_text=30 | top-5 频次, 唯一值数, 唯一率 |
| **text** | 平均字符长度 ≥ L_text=30 | 平均长度, 最大长度, 样本片段 |

```
# ColStats: name, dtype("numeric"|"datetime"|"categorical"|"text"),
#           missing_pct, 及各类型特有统计量（见§3.2表格）

def profile_column(series, col_name):
    missing_pct = series.isna().mean() * 100
    non_null = series.dropna()

    # 优先级：numeric > datetime > categorical/text
    if to_numeric_ratio(non_null) >= 0.95:       # τ_num
        return ColStats("numeric", min, max, median, mean, std)
    if to_datetime_ratio(non_null) >= 0.90:       # τ_dt
        return ColStats("datetime", earliest, latest, span_days)
    if non_null.str.len().mean() >= 30:           # L_text
        return ColStats("text", avg_len, sample_text[:200])
    else:
        return ColStats("categorical", n_unique, unique_pct, top_5_values)
```

### 3.3 列描述文本生成

将每列的统计量转化为模板化英文描述（适配英文预训练 sentence-transformer 模型）：

**Numeric 模板**：
```
Column "{col_name}": numeric, range [{min}, {max}], median={median}, mean={mean:.2f}, std={std:.2f}, {missing_pct:.1f}% missing.
```

**Categorical 模板**：
```
Column "{col_name}": categorical with {n_unique} unique values ({unique_pct:.1f}% unique). Top values: {top1} ({freq1}), {top2} ({freq2}), {top3} ({freq3}). {missing_pct:.1f}% missing.
```

**Datetime 模板**：
```
Column "{col_name}": datetime from {earliest} to {latest}, spanning {span_days} days. {missing_pct:.1f}% missing.
```

**Text 模板**：
```
Column "{col_name}": free text, avg length {avg_len:.0f} chars. Sample: "{truncated_sample}". {missing_pct:.1f}% missing.
```

```
def col_to_description(cs):
    # 按 dtype 选择对应模板（见§3.3 的四种模板），填入统计量
    # 示例输出：'Column "age": numeric, range [18, 90], median=35, ...'
    return TEMPLATES[cs.dtype].format(**cs.stats_dict())
```

### 3.4 列向量编码

使用预训练 sentence-transformer 编码列描述文本：

- **模型**：`sentence-transformers/all-MiniLM-L6-v2`（输出维度 d=384）。
- **与现有嵌入的维度差异**：现有 SGNS 嵌入为 d=256（§3.3）。各视图独立运算不影响融合——融合在相似度矩阵层面（标量分数）进行，而非在嵌入空间拼接，因此无需统一维度。
- **GPU 批处理编码**：对所有列描述文本批量编码，每批 256 条。
- **L2 归一化**：编码后归一化为单位向量，与 `scripts/build_ann_index.py` 中的 `Z_norm = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)` 模式一致。

```
def encode_descriptions(descriptions, device="cuda"):
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    # 批量编码 + L2 归一化，同 build_ann_index.py 模式
    H = model.encode(descriptions, batch_size=256, normalize_embeddings=True)
    return H  # (n_total_cols, 384), float32
```

### 3.5 数据集向量聚合 Z_tabcontent

将一个数据集的多列向量聚合为单一数据集向量：

```
z_i = Σ_k (w_k · h_k) / Σ_k w_k
```

其中 `h_k` 是第 k 列的编码向量，权重 `w_k = w_type × w_quality`：

- `w_type`：类型权重（numeric=1.0, categorical=1.0, datetime=0.8, text=1.2）。
- `w_quality`：质量权重 = `(1 - missing_rate) × min(unique_rate, 0.95)`。

聚合后 L2 归一化，输出 `Z_tabcontent ∈ R^{|D_content| × 384}`。

```
W_TYPE = {"numeric": 1.0, "categorical": 1.0, "datetime": 0.8, "text": 1.2}

def aggregate_dataset_vector(col_embeddings, col_stats):
    # w_k = w_type × (1 - missing_rate) × min(unique_rate, 0.95)
    weights = [W_TYPE[cs.dtype] * (1 - cs.missing_pct/100) * min(cs.unique_pct/100, 0.95)
               for cs in col_stats]
    z = weighted_mean(col_embeddings, weights)
    return l2_normalize(z)  # (384,)

def build_Z_tabcontent(d_content_ids, dataset_vectors):
    # 输出格式同 Z_tag.parquet: columns=["doc_idx", "f0"..."f383"]
    Z = stack(dataset_vectors)  # (B_ds, 384)
    return DataFrame(Z, index=d_content_ids)  # → tmp/content/Z_tabcontent.parquet
```

### 3.6 内容视图相似图

基于 `Z_tabcontent` 构建内容视图的稀疏相似度矩阵：

1. **FAISS IndexFlatIP 构建**：内积索引（L2 归一化后等价余弦相似度），与 §6.7 的索引构建流程一致。
2. **Top-K=50 近邻检索**：为每个内容子集文档检索 50 个最近邻。
3. **对称化（max）**：`S[i,j] = max(S[i,j], S[j,i])`，确保相似度矩阵对称。
4. **行裁剪 + 行归一化**：保留每行 top-K 个非零值，L1 行归一化使每行和为 1。
5. **输出** `S_tabcontent_symrow`：格式与现有视图矩阵一致（COO parquet 三元组 `row/col/val` + `manifest.json`，参见 §6.6 ANN 索引输出格式）。

**注意**：`S_tabcontent_symrow` 仅覆盖 D_content 子集（B_ds 个数据集），未覆盖文档的内容视图分数为 0。这一稀疏性在融合阶段通过自适应权重（§5）自然处理——未覆盖文档的 ρ_tabcontent[i]=0，导致 α_tabcontent[i]=0，四视图融合退化为三视图融合，等价于现有 Fused3-RA 结果。

```
def build_content_similarity_graph(Z_path, N_total=521735, k=50):
    doc_indices, Z = load_Z(Z_path)  # Z: (B_ds, 384)

    # FAISS IndexFlatIP top-(k+1) 搜索（同 build_ann_index.py）
    index = faiss.IndexFlatIP(384)
    index.add(l2_normalize(Z))
    scores, idxs = index.search(Z, k + 1)  # 排除自身

    # 局部索引 → 全局索引，收集 COO 三元组 (rows, cols, vals)
    rows, cols, vals = local_to_global(scores, idxs, doc_indices)

    # 对称化(max) + 行归一化（复用 sym_and_rownorm）
    sym_rows, sym_cols, sym_vals = sym_and_rownorm(rows, cols, vals, N_total)
    save_partitioned_edges(sym_rows, sym_cols, sym_vals, N_total,
                           prefix="S_tabcontent_symrow", k=k)
    # → tmp/content/S_tabcontent_symrow_k50_manifest.json + _part*.parquet
```

---

## 4. 元数据–内容一致性指标

已有 `tmp/consistency_scores.parquet`（521,735 行 × 4 列）记录了视图间的 Jaccard / overlap / correlation 指标。本节延续该分析模式，新增**元数据视图与内容视图的一致性**维度。

### 4.1 Jaccard 集合重叠

对于 D_content 中的每个文档 i，比较其在元数据融合邻居集 N_meta(i)（来自 `S_fused3_symrow` 的 top-K 邻居）和内容视图邻居集 N_cont(i)（来自 `S_tabcontent_symrow` 的 top-K 邻居）的重叠度：

```
J(i) = |N_meta(i) ∩ N_cont(i)| / |N_meta(i) ∪ N_cont(i)|
```

J(i) ∈ [0, 1]，高值表示元数据和内容视图的推荐结果高度一致，低值表示两者捕获了不同维度的相关性。

```
def jaccard_overlap(d_content_ids, N_meta, N_cont):
    # N_meta/N_cont: {doc_idx: set(neighbor_indices)}，分别来自 S_fused3 和 S_tabcontent
    for i in d_content_ids:
        inter = N_meta[i] & N_cont[i]
        union = N_meta[i] | N_cont[i]
        J[i] = len(inter) / len(union)
    return DataFrame(doc_idx, jaccard, n_meta, n_cont, n_intersect)
```

### 4.2 加权一致性

考虑邻居的相似度权重：

```
C(i) = Σ_{j ∈ N_meta(i) ∩ N_cont(i)} min(w_meta(i,j), w_cont(i,j))
```

归一化后得到 `c[i] ∈ [0, 1]`，用于 §5.2 的一致性调节。

```
def weighted_consistency(d_content_ids, meta_edges, cont_edges, K=50):
    # meta_edges/cont_edges: COO parquet (row, col, val)
    meta_w = build_lookup(meta_edges)  # {(row,col): val}
    cont_w = build_lookup(cont_edges)

    for i in d_content_ids:
        common = neighbors(meta_w, i) & neighbors(cont_w, i)
        weighted_sum = sum(min(meta_w[i,j], cont_w[i,j]) for j in common)
        max_possible = min(top_K_sum(meta_w, i), top_K_sum(cont_w, i))
        c[i] = weighted_sum / max_possible   # 归一化到 [0,1]
    return c  # (B_ds,) → tmp/content/consistency_meta_content.parquet
```

---

## 5. 四视图融合

### 5.1 行自适应权重（ρ 方法）

现有三视图融合（§3.5）使用等权 CombSUM（1/3 + 1/3 + 1/3）或 Fused3-RA 的自适应行归一化。本节引入**行信息浓度**（row information concentration）`ρ` 作为更精细的权重分配基础。

这是对现有等权融合策略的改进，现有 CombSUM 可视为 ρ 退化为常数的特例。

**定义**：对于文档 i 在视图 v 上的行信息浓度：

```
ρ_v[i] = Σ_j S_v[i, j]²
```

`ρ_v[i]` 反映视图 v 对文档 i 的区分能力——若分数集中在少数邻居上（高浓度），说明该视图对此文档有清晰的判别信号。

**归一化为权重**：

```
α_v_base[i] = ρ_v[i] / Σ_{v'} ρ_{v'}[i]
```

其中求和遍历所有四个视图 v' ∈ {tag, text, beh, tabcontent}。

```
def compute_rho(S):
    # ρ_v[i] = Σ_j S_v[i,j]²（同 analyse_ddp_hybrid.ipynb 的 mat.multiply(mat).sum(1)）
    return S.multiply(S).sum(axis=1)  # (N,)

def compute_alpha(rho_dict):
    # rho_dict = {"tag": (N,), "text": (N,), "beh": (N,), "tabcontent": (N,)}
    # 非 D_content 文档 rho_tabcontent=0 → alpha_tabcontent=0，退化为 Fused3-RA
    rho_stack = stack([rho_dict[v] for v in views])  # (4, N)
    return {v: rho_stack[v] / rho_stack.sum(axis=0) for v in views}
```

### 5.2 一致性调节（可选）

利用 §4 的一致性指标 c[i] 调整内容视图权重：

```
g(i) = β + (1 - β) × (1 - c[i])
```

其中 β = 0.5 为基础保留系数。当一致性低时（c[i] → 0），`g(i) → 1`，内容视图保持全额权重；当一致性高时（c[i] → 1），`g(i) → β`，内容视图权重衰减——因为此时元数据已充分捕获了内容信号，内容视图的额外贡献有限。

调节后：`α_cont_adj[i] = α_cont_base[i] × g(i)`，然后重新归一化所有权重使其和为 1。

```
BETA = 0.5

def apply_consistency_adjustment(alpha, c_scores, d_content_ids, N):
    # g(i) = β + (1-β)×(1 - c[i])；非 D_content 文档 c=0 → g=1（不受影响）
    c_lookup = zeros(N);  c_lookup[d_content_ids] = c_scores
    g = BETA + (1 - BETA) * (1 - c_lookup)
    alpha["tabcontent"] *= g
    # 重新归一化使四视图权重和为 1
    return normalize_alpha(alpha)
```

### 5.3 融合公式

```
S_fused4_raw[i, :] = α_tag[i] · S_tag[i, :] + α_text[i] · S_text[i, :] + α_beh[i] · S_beh[i, :] + α_cont[i] · S_tabcontent[i, :]
```

后处理与现有融合流程一致：

1. 行内 top-K 裁剪（保留每行前 50 个最高分邻居）。
2. 行 L1 归一化（使每行分数和为 1）。
3. 输出 `S_fused4_symrow`（命名沿用 `_symrow` 后缀，与现有 `S_fused3_symrow` 规范一致）。

**与现有融合权重的关系**：

- 现有统一指标权重（§4.2）：`W_TAG=0.5, W_DESC=0.3, W_CREATOR=0.2`——这是**评测层面**的维度加权，不变。
- 现有 CombSUM 融合权重：`W_TAGPPMI=1/3, W_TEXTBM25=1/3, W_ENG=1/3`——这是**视图层面**的分数聚合权重。四视图融合将其替换为 ρ 自适应权重。

```
def fuse_four_views(S_views, alpha, N=521735, K=50):
    # S_fused4[i,:] = Σ_v α_v[i] · S_v[i,:]
    for i in range(N):
        fused = {}
        for v in S_views:
            if alpha[v][i] < 1e-9: continue
            for j, s in nonzero_entries(S_views[v], row=i):
                fused[j] += alpha[v][i] * s

        # top-K 裁剪 + L1 行归一化
        top = top_k(fused, K)
        emit(i, top / sum(top))

    save_partitioned_edges(rows, cols, vals, N, prefix="S_fused4_symrow", k=K)
    # → tmp/content/S_fused4_symrow_k50_manifest.json + _part*.parquet
```

---

## 6. 实验设计

### 6.1 内容预算实验

通过网格搜索评估不同预算参数对性能的影响：

| 参数 | 测试范围 | 说明 |
|------|---------|------|
| B_ds（数据集数） | {500, 1000, 2000} | 下载表格文件的数据集数量 |
| MAX_ROWS（采样行数） | {256, 1024, 2048} | 每张表读取的最大行数 |
| MAX_COLS（采样列数） | {30, 60} | 每张表保留的最大列数 |

主实验配置：B_ds=1000, MAX_ROWS=1024, MAX_COLS=60。

```
BUDGET_GRID = {"B_ds": [500, 1000, 2000], "MAX_ROWS": [256, 1024, 2048], "MAX_COLS": [30, 60]}
PRIMARY = {"B_ds": 1000, "MAX_ROWS": 1024, "MAX_COLS": 60}
```

### 6.2 对比方法

| 方法 | 说明 |
|------|------|
| **Meta-only** | `S_fused3_symrow`（现有最优三视图融合，§7 基线） |
| **Content-only** | `S_tabcontent_symrow`（仅内容视图） |
| **Naive 融合** | `0.5 × S_fused3 + 0.5 × S_tabcontent`（等权混合） |
| **Adaptive 融合** | `S_fused4_symrow`（ρ 自适应权重，§5.1） |
| **Adaptive+一致性** | `S_fused4_symrow` + c[i] 调节（§5.2） |

```
METHODS = {
    "Meta-only":       {"prefix": "S_fused3_symrow",    "dir": "tmp"},
    "Content-only":    {"prefix": "S_tabcontent_symrow", "dir": "tmp/content"},
    "Naive-Fusion":    {"prefix": "S_naive4_symrow",     "dir": "tmp/content"},
    "Adaptive-Fusion": {"prefix": "S_fused4_symrow",     "dir": "tmp/content"},
    "Adaptive+Cons":   {"prefix": "S_fused4c_symrow",    "dir": "tmp/content"},
}
```

### 6.3 评测方法

**完全复用 RESEARCH_PROPOSAL.md §4 的三维度银标准和 src/metrics.py 的评测函数。**

#### 银标准（复用已生成文件，不重新构建）

| 维度 | 银标准文件 | 增益函数 | 二元阈值 | 覆盖率 |
|------|-----------|---------|---------|--------|
| Tag-Relevance | `tmp/relevance_tag_docs.parquet` + `tmp/relevance_tag_idf.parquet` | IDF-weighted Jaccard: `idf_inter / idf_union` | 共享标签 ≥1 | 41.1% |
| Desc-Relevance | `tmp/S_textbm25_topk_k50_manifest.json` + `_part*.parquet` | BM25 余弦相似度（连续值） | `sim > 0.2` | ~79.9% |
| Creator-Relevance | `tmp/beh_base.parquet` | 二元匹配：`creator_i == creator_j` | - | 63.1% |

#### 指标集（`src/metrics.py` 中的具体函数）

| 指标 | 函数 | 增益类型 | 含义 |
|------|------|---------|------|
| nDCG@20 | `ndcg_at_k(gains, ideal)` (L44) | 分级 | 排序质量（主指标） |
| MAP@20 | `average_precision_at_k(binary)` (L73) | 二元 | 精确率-召回率均衡 |
| MRR@20 | `mrr_at_k(binary)` (L106) | 二元 | 首个相关结果排名 |
| P@20 | `precision_at_k(binary)` (L126) | 二元 | top-K 精确率 |
| R@20 | `recall_at_k(binary, total)` (L146) | 二元 | top-K 召回率 |
| Coverage | `covered / total` | - | 可评测文档比例 |

> 行号标注引用自 `src/metrics.py`，供实现时快速定位。

#### 统一指标

```
Unified_nDCG = W_TAG * tag_ndcg + W_DESC * desc_ndcg + W_CRE * cre_ndcg
# W_TAG=0.5, W_DESC=0.3, W_CRE=0.2  (RESEARCH_PROPOSAL.md §4.2)
```

**重要**：统一指标权重是**评测层面**的，不随融合策略变化。内容视图扩展改变的是**融合层面**的 α 权重，不影响评测权重。

#### 评测流程伪代码

完全复用 `compare_methods_framework.ipynb` 和 `tests/test_evaluation_flow.py` 的评测流程：

```
def evaluate_method(prefix, k_eval=20):
    # 加载相似矩阵的 top-K 邻居
    nbr_idx, nbr_w = build_topk_for_method(prefix, k_eval)

    # 三维度独立评测（复用 evaluate_ranking）
    # Tag:     gain = idf_weighted_jaccard(tags_i, tags_j)
    # Desc:    gain = bm25_sim(i,j),  binary threshold=0.2
    # Creator: gain = 1 if creator_i == creator_j else 0
    for dim in [tag, desc, creator]:
        results[dim] = aggregate(evaluate_ranking(...) for each query i)

    # 统一指标
    results["unified_ndcg"] = 0.5*tag_ndcg + 0.3*desc_ndcg + 0.2*cre_ndcg
    return results
```

#### 额外分析

1. **性能–预算曲线**：x = B_ds × MAX_ROWS（成本代理），y = Unified@nDCG，识别拐点。

2. **视图贡献消融**：逐一移除视图，其中"去Content"必须精确复现 Fused3-RA 结果（验证向后兼容性）。

3. **一致性分布**：J(i) 和 c[i] 直方图，分析互补程度。

4. **分层性能**：对比 D_content 内/外文档的 Unified@nDCG 变化（预期仅 D_content 内文档有提升，D_content 外文档性能不变，因为 α_tabcontent[i]=0 使融合退化为 Fused3-RA）。

---

## 前置依赖与实施清单

实施本扩展前需完成以下准备工作：

- [ ] 验证 `kaggle datasets list -s {slug}` 搜索匹配流程：对代表性数据集（如 `heart-failure-clinical-data`）确认 Slug 匹配可靠性
- [ ] 评估 Kaggle API 搜索速率限制，确认 B_ds=1000 批量匹配的可行性（可分批执行 + 缓存 `slug_to_ref.csv`）
- [ ] 创建 `src/content/` 模块骨架（表格采样、列统计、向量编码等子模块）
- [ ] 创建 `notebooks/04_content/` 实验 notebook（数据探索、视图构建、融合实验）
- [ ] 验证 Kaggle API 配额与下载速率限制（B_ds=1000 批量下载的可行性）
- [ ] 确认 `sentence-transformers/all-MiniLM-L6-v2` 模型可用且 GPU 环境就绪
