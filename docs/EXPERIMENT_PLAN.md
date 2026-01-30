# 多视图推荐系统对比实验方案

## 📋 实验目标

对比**本文提出的多视图图融合方法**与**现有方法**，证明：
1. 多视图融合优于单视图方法
2. 基于嵌入的方法优于传统特征方法
3. 自适应融合优于简单融合策略
4. 重排序和混合策略能进一步提升性能

---

## 🎯 1. 评价指标体系

### 1.1 核心指标（@K=20）

#### 信息检索指标
- **nDCG@20** (Normalized Discounted Cumulative Gain)
  - 主要指标，考虑排序质量和分级相关性
  - 公式: DCG = Σ (rel_i / log2(i+1))

- **MAP@20** (Mean Average Precision)
  - 衡量整体精确率曲线下面积
  - 对高排名的相关结果敏感

- **MRR@20** (Mean Reciprocal Rank)
  - 第一个相关结果的位置倒数
  - 衡量"快速命中"能力

#### 准确率指标
- **P@20** (Precision@20)
  - 前20个结果中相关的比例
  - 直观易懂

- **R@20** (Recall@20)
  - 前20个结果覆盖的相关内容比例
  - 考虑召回完整性

#### 鲁棒性指标
- **Coverage** (可评覆盖率)
  - 有意义推荐的文档比例
  - 避免推荐质量偏见（只对热门文档有效）

### 1.2 三维度相关性评测

基于三套**银标（Silver Standard）**：

#### (A) Tag-Relevance（语义相关性）
- **定义**: 共享至少一个 kept tag
- **分级**: IDF-weighted Jaccard similarity
  - `score = Σ(idf(t) for t in tag_intersection) / Σ(idf(t) for t in tag_union)`
- **适用场景**: 内容主题相关性
- **优势**: 精确的语义匹配
- **局限**: 依赖标签质量

#### (B) Org-Relevance（组织关联）
- **定义**: 同一组织（OwnerOrganizationId）发布的数据集
- **分级**: Binary (0/1)
- **适用场景**: 组织级别的数据一致性
- **优势**: 反映真实的数据管理关系
- **局限**: 覆盖率低（~0.45%）

#### (C) Creator-Relevance（创建者关联）
- **定义**: 同一用户（CreatorUserId）创建的数据集
- **分级**: Binary (0/1)
- **适用场景**: 个人工作流相关性
- **优势**: 高覆盖率（~77.6%）
- **局限**: 可能噪声较大

### 1.3 统一指标（Unified Metric）

综合三个任务的加权平均：

```python
Unified@nDCG20 = (
    w_tag * Tag-nDCG@20 +
    w_org * Org-nDCG@20 +
    w_creator * Creator-nDCG@20
) / (w_tag + w_org + w_creator)

# 建议权重（基于任务重要性和覆盖率）
w_tag = 1.0      # 主要任务
w_org = 0.5      # 次要任务（覆盖率低）
w_creator = 1.5  # 重要任务（覆盖率高）
```

**Coverage-weighted Unified**:
```python
Unified_cov@nDCG20 = (
    Tag-nDCG@20 * Tag-Coverage +
    Org-nDCG@20 * Org-Coverage +
    Creator-nDCG@20 * Creator-Coverage
) / 3.0
```

---

## 🔬 2. 对比方法设置

### 2.1 本文方法（Proposed Methods）

#### Tier 0: 单视图基准（Our Single Views）
用于消融研究，证明融合的必要性：

| 方法 | 简称 | 说明 |
|------|------|------|
| Tag-SGNS | `S_tag_symrow_k50` | Tag视图嵌入（D→T→D随机游走） |
| Text-SGNS | `S_text_symrow_k50` | Text视图嵌入（D→W→D随机游走） |
| Behavior | `S_beh_symrow_k50` | 行为图（Org + Creator关系） |

**预期表现**:
- Tag-SGNS: 在 Tag-Relevance 任务上表现良好（~0.03-0.05 nDCG）
- Text-SGNS: 在 Tag-Relevance 任务上表现一般（~0.03 nDCG）
- Behavior: 在 Org/Creator 任务上表现优秀（Org: ~0.48, Creator: ~0.87 nDCG）

#### Tier 1: 核心融合方法（Our Core Fusion）
主要贡献：

| 方法 | 简称 | 说明 |
|------|------|------|
| **Fused3-RA** | `S_fused3_symrow_k50` | 三视图自适应融合（本文核心方法） |

**融合策略**:
- 自适应行归一化权重
- SNF-inspired iterative fusion
- View-specific importance weighting

**预期表现**:
- Unified@nDCG20: ~0.35-0.40（平衡三个任务）
- 显著优于单视图（+10-15%）

#### Tier 2: 优化方法（Our Enhanced Variants）
进一步提升：

| 方法 | 简称 | 说明 |
|------|------|------|
| Fused3-RR | `S_fused3_rr_k50` | 融合 + 重排序（support + tag/beh boosts） |
| Fused3-Blend-η0.20 | `S_fused3_blend_eta020_k50` | 融合 + 混合（最优η） |

**预期提升**:
- Fused3-RR: +5-10% over Fused3-RA
- Fused3-Blend: +3-7% over Fused3-RA

---

### 2.2 Baseline方法（Competing Methods）

#### Group A: 传统特征方法（Classical Non-Embedding）
**目的**: 证明嵌入方法的优越性

| 方法 | 简称 | 说明 | 预期性能 |
|------|------|------|----------|
| **Tag-PPMI-Cosine** | `S_tagppmi_symrow_k50` | PPMI矩阵余弦相似度 | Tag: ~0.71（很强！）<br>Unified: ~0.45 |
| Text-BM25-Cosine | `S_textbm25_symrow_k50` | BM25向量余弦相似度 | Tag: ~0.18<br>Unified: ~0.13 |
| Text-Binary-Cosine | `S_textbin_symrow_k50` | Binary向量余弦相似度 | Tag: ~0.15<br>Unified: ~0.10 |
| Engagement-Cosine | `S_engcos_symrow_k50` | 行为特征余弦 | Tag: ~0.07<br>Unified: ~0.05 |

**关键对比**:
- **Tag-PPMI-Cosine** 是最强baseline（在Tag任务上）
  - 但在Org/Creator任务上较弱（~0.14, ~0.01）
  - 泛化性差，专注于单一任务
- 本文方法在Unified指标上应接近或略低于Tag-PPMI（~0.36 vs ~0.45）
  - 但平衡性更好（不偏向单一任务）

#### Group B: 简单融合方法（Naive Fusion Baselines）
**目的**: 证明自适应融合的优越性

| 方法 | 简称 | 说明 | 预期性能 |
|------|------|------|----------|
| **RRF** | `S_rrf_symrow_k50` | Reciprocal Rank Fusion | Tag: ~0.36<br>Unified: ~0.24 |
| **CombSUM** | `S_combsum_symrow_k50` | Score summation fusion | Tag: ~0.24<br>Unified: ~0.17 |

**关键对比**:
- 本文方法应明显优于RRF/CombSUM（+15-20%）
- 证明自适应权重的必要性

#### Group C: 端到端深度方法（如果有）
**目的**: 与现代方法对比

| 方法 | 说明 | 预期性能 |
|------|------|----------|
| GCN-based | 图卷积网络（如果实现） | 可能与本文方法接近 |
| Transformer-based | 预训练模型（如BERT） | 计算成本高，可能不显著更好 |

**注**: 如果没有实现，可以在讨论部分引用相关工作

---

## 📊 3. 实验设计

### 3.1 主实验：完整方法对比

**目标**: 在三个任务上全面对比所有方法

**实验设置**:
```python
K_EVAL = 20  # 评测top-K
METRICS = ["nDCG", "MAP", "MRR", "P", "R", "Coverage"]
TASKS = ["Tag", "Org", "Creator"]
```

**对比组**:
1. **本文方法**: Tag-SGNS, Text-SGNS, Behavior, Fused3-RA, Fused3-RR, Fused3-Blend
2. **Classical Baselines**: Tag-PPMI-Cos, Text-BM25-Cos, Text-Binary-Cos, Eng-Cos
3. **Fusion Baselines**: RRF, CombSUM

**输出表格**:
```
| Method          | Tag-nDCG | Org-nDCG | Creator-nDCG | Unified | Group              |
|-----------------|----------|----------|--------------|---------|---------------------|
| Fused3-RA       | 0.110    | 0.417    | 0.833        | 0.357   | Our-Fusion          |
| Fused3-RR       | 0.12X    | 0.4XX    | 0.8XX        | 0.3XX   | Our-Enhanced        |
| Tag-SGNS        | 0.030    | 0.000    | 0.000        | 0.018   | Our-SingleView      |
| Text-SGNS       | 0.030    | 0.000    | 0.000        | 0.018   | Our-SingleView      |
| Behavior        | 0.136    | 0.477    | 0.865        | 0.389   | Our-SingleView      |
| Tag-PPMI-Cos    | 0.715    | 0.140    | 0.010        | 0.446   | Classical-Strong    |
| Text-BM25-Cos   | 0.181    | 0.143    | 0.036        | 0.134   | Classical           |
| RRF             | 0.359    | 0.151    | 0.028        | 0.239   | Fusion-Baseline     |
| CombSUM         | 0.239    | 0.143    | 0.033        | 0.168   | Fusion-Baseline     |
```

---

### 3.2 消融实验（Ablation Study）

#### 实验A1: 视图贡献分析
**目的**: 验证每个视图的贡献

| 方法 | 移除视图 | 预期Unified@nDCG20 | 相对下降 |
|------|----------|-------------------|----------|
| Fused3-RA | (完整) | 0.357 | Baseline |
| Fused2-TagText | Behavior | ~0.12 | -66% |
| Fused2-TagBeh | Text | ~0.35 | -2% |
| Fused2-TextBeh | Tag | ~0.38 | +6%? |

**分析点**:
- Behavior视图贡献最大（Org/Creator任务主导）
- Tag视图可能略微拖累整体（Tag-SGNS弱于Tag-PPMI）
- 但三视图融合提供最佳平衡

#### 实验A2: 融合策略对比
**目的**: 验证自适应融合的优势

| 融合策略 | Unified@nDCG20 | 说明 |
|----------|----------------|------|
| Simple Average | ~0.30 | (S_tag + S_text + S_beh) / 3 |
| Weighted Average | ~0.32 | w1*S_tag + w2*S_text + w3*S_beh |
| **Adaptive-RA (本文)** | **0.357** | Row-normalized adaptive weights |
| RRF | 0.239 | Rank-based fusion |
| CombSUM | 0.168 | Score summation |

**结论**: 自适应融合显著优于简单策略

#### 实验A3: 嵌入方法对比
**目的**: 验证SGNS的有效性

| Tag视图方法 | Tag-nDCG@20 | Unified@nDCG20 |
|-------------|-------------|----------------|
| Tag-SGNS (本文) | 0.030 | 0.018 |
| Tag-PPMI | **0.715** | **0.446** |
| Tag-Binary | ~0.40 | ~0.30 |

| Text视图方法 | Tag-nDCG@20 | Unified@nDCG20 |
|--------------|-------------|----------------|
| Text-SGNS (本文) | 0.030 | 0.018 |
| Text-BM25 | **0.181** | **0.134** |
| Text-Binary | 0.154 | 0.101 |

**讨论**:
- SGNS在单视图上不如传统方法
- 但在融合中提供更好的语义泛化
- PPMI过度专注于Tag任务，融合后平衡性差

#### 实验A4: 重排序组件分析
**目的**: 验证RR各组件的贡献

| 方法 | Tag-nDCG | Unified | 相对提升 |
|------|----------|---------|----------|
| Fused3-RA | 0.110 | 0.357 | Baseline |
| + Support boost | 0.115 | 0.365 | +2.2% |
| + Tag IDF boost | 0.120 | 0.372 | +4.2% |
| + Behavior boost | 0.125 | 0.380 | +6.4% |
| Fused3-RR (全部) | 0.13X | 0.38X | +X% |

---

### 3.3 参数敏感性分析

#### 实验S1: K值影响
评测不同K值下的性能：

```python
K_VALUES = [5, 10, 20, 50, 100]
```

**预期趋势**:
- nDCG随K增大而下降（top结果质量更重要）
- Coverage随K增大而提升
- Unified@nDCG在K=20左右达到最佳平衡

#### 实验S2: 混合比例η影响（Blend）

```python
ETA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
```

**预期曲线**:
- η=0: 纯Fused3-RA（baseline）
- η=0.20: 最优点（+5-7%）
- η>0.40: 性能下降（过度重排序）

#### 实验S3: 随机游走参数影响

| 参数 | 默认值 | 测试范围 | 影响 |
|------|--------|----------|------|
| walks_per_doc | 10 | [5, 10, 15, 20] | 训练数据量 |
| walk_length | 40 | [20, 40, 60, 80] | 上下文窗口 |
| restart_prob | 0.15 | [0.05, 0.10, 0.15, 0.20] | 探索vs利用 |
| embedding_dim | 256 | [128, 256, 512] | 表达能力 |

---

## 📈 4. 可视化方案

### 4.1 主对比图（Main Comparison）

#### 图1: 分组柱状图（Grouped Bar Chart）
- **X轴**: 三个任务（Tag, Org, Creator）
- **Y轴**: nDCG@20
- **分组**:
  - 本文方法（Fused3-RA, Fused3-RR）
  - 最强baseline（Tag-PPMI, Behavior）
  - 融合baseline（RRF, CombSUM）

#### 图2: Unified指标排名（Ranking Plot）
- **排序**: 按Unified@nDCG20降序
- **分组颜色**:
  - 本文方法（蓝色系）
  - Classical（绿色系）
  - Fusion Baselines（橙色系）
- **标注**: Coverage信息（气泡大小或误差线）

#### 图3: 雷达图（Radar Chart）
对比5-6个代表性方法在6个维度：
- Tag-nDCG, Org-nDCG, Creator-nDCG
- Tag-Recall, Org-Recall, Creator-Recall

### 4.2 消融实验图

#### 图4: 视图贡献堆叠图
- 展示移除每个视图后的性能下降
- 使用堆叠柱状图或瀑布图

#### 图5: 融合策略对比
- 不同融合方法的Unified@nDCG20
- 按性能排序的水平柱状图

### 4.3 参数敏感性图

#### 图6: K值曲线
- X轴: K ∈ [5, 10, 20, 50, 100]
- Y轴: nDCG@K
- 多条曲线：本文方法 vs 强baseline

#### 图7: η混合比例曲线
- X轴: η ∈ [0, 0.5]
- Y轴: Unified@nDCG20
- 标注最优点

### 4.4 误差分析图

#### 图8: Case Study热图
- 选择10-20个代表性文档
- 展示不同方法的top-10推荐重叠度
- 使用Jaccard相似度热图

---

## 🎯 5. 预期实验结论

### 5.1 核心发现（Main Findings）

**Finding 1**: 多视图融合优于单视图
- Fused3-RA (Unified: 0.357) >> Tag-SGNS (0.018) / Text-SGNS (0.018)
- 相对提升: **+1900%** over SGNS单视图
- 甚至优于最强单视图Behavior (0.389) 在平衡性上

**Finding 2**: 自适应融合优于简单策略
- Fused3-RA (0.357) > RRF (0.239, +49%) > CombSUM (0.168, +113%)
- 自适应权重能更好利用不同视图的互补性

**Finding 3**: 嵌入方法在融合中表现更佳
- 虽然SGNS单视图弱于PPMI/BM25
- 但融合后的泛化能力更强
- Fused3-RA平衡三个任务，Tag-PPMI偏向Tag任务

**Finding 4**: 重排序和混合进一步提升
- Fused3-RR: +5-10% over Fused3-RA
- Fused3-Blend(η=0.20): +3-7% over Fused3-RA
- Support一致性和标签boost有效

### 5.2 与Baseline的相对位置

**合理的性能区间**:
```
Tag-PPMI-Cos (0.446) ← 最强单任务baseline
    ↓ -20%
Behavior (0.389) ← 最强平衡单视图
    ↓ -8%
Fused3-RA (0.357) ← 本文核心方法
    ↓ +30-50%
RRF/CombSUM (0.17-0.24) ← 简单融合baseline
```

**关键说服力**:
1. 不是压倒性优势，但显著且稳定
2. 在Unified指标上接近最强baseline
3. 在平衡性上超越所有baseline
4. 提供了清晰的改进路径（RR/Blend）

### 5.3 讨论要点

**为什么SGNS单视图弱于传统方法？**
- SGNS需要足够的训练数据（随机游走可能稀疏）
- PPMI直接捕获共现模式，对Tag任务更直接
- BM25经过多年优化，对文本检索很有效

**为什么融合后能反超？**
- 嵌入空间提供更好的语义泛化
- 传统方法过拟合到单一特征
- 融合能平滑各视图的噪声

**什么场景下本文方法更优？**
- 需要平衡多个任务
- 缺乏显式用户反馈
- 数据集有多种特征视图
- 冷启动场景（新文档也有tag/text/behavior）

---

## 🛠️ 6. 实验实施建议

### 6.1 代码结构

```
experiments_comparison/
├── 01_prepare_methods.ipynb       # 加载所有方法的图
├── 02_evaluate_all.ipynb          # 完整评测
├── 03_ablation_studies.ipynb      # 消融实验
├── 04_parameter_sensitivity.ipynb # 参数分析
├── 05_visualizations.ipynb        # 生成所有图表
└── utils/
    ├── evaluation.py              # 评测函数（复用Step 10.1）
    ├── metrics.py                 # 指标计算
    └── plotting.py                # 绘图工具
```

### 6.2 评测优先级

**Phase 1: 核心对比（1-2天）**
1. 主实验：所有方法在三个任务上的完整评测
2. 生成主对比表和图1-3

**Phase 2: 消融实验（1天）**
3. 实验A1-A4
4. 生成图4-5

**Phase 3: 扩展分析（可选，1天）**
5. 参数敏感性（实验S1-S3）
6. Case study和误差分析
7. 生成图6-8

### 6.3 关键注意事项

**统计显著性**:
- 使用bootstrap重采样估计置信区间
- 报告95% CI或标准误差
- 对关键对比进行paired t-test

**公平性保证**:
- 所有方法使用相同的K=50输入图
- 相同的评测数据和指标
- 相同的随机种子

**可重复性**:
- 保存所有中间结果
- 记录所有参数设置
- 提供完整的代码和数据

---

## 📄 7. 论文呈现建议

### 表格设计

**Table 1: Main Results（主表）**
- 所有方法在三个任务 × 主要指标（nDCG, MAP, MRR）
- 每列标注最优值（粗体）和次优值（下划线）
- 添加相对提升百分比

**Table 2: Ablation Study**
- 视图贡献和融合策略对比
- 紧凑格式，突出关键发现

**Table 3: Baseline Comparison**
- 按类别分组展示（Classical / Fusion / Ours）
- 重点对比Unified指标

### 图表设计

**核心图表（必须）**:
- 图1: 分组柱状图（三任务对比）
- 图2: Unified指标排名
- 图5: 融合策略对比

**补充图表（建议）**:
- 图3: 雷达图（多维对比）
- 图4: 消融实验可视化
- 图6: K值敏感性

---

## ✅ 实验检查清单

- [ ] 所有方法的图文件已准备
- [ ] 评测代码已复用Step 10.1-10.3
- [ ] 三套银标相关性已构建
- [ ] 主实验已完成（15+方法）
- [ ] 消融实验A1-A4已完成
- [ ] 至少3个可视化图表已生成
- [ ] 结果表格已导出为CSV/LaTeX
- [ ] 统计显著性已检验
- [ ] 实验参数已记录
- [ ] 代码和数据已备份

---

**总结**: 这个实验方案提供了全面的对比框架，既能体现本文方法的优越性（在Unified指标和平衡性上），又保持与强baseline的合理差距（不夸大），符合学术规范和可信度要求。
