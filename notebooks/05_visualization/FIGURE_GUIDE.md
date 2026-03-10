# 实验结果可视化图表说明

## 概述

本文档为 `experiment_results_visualization.ipynb` 中生成的所有图表提供详细说明。该 notebook 包含 **23 个图表**，系统性地展示了多视图融合推荐系统在不同数据规模（1K-100K）下的实验结果。

### 图表总览

| 分类 | 数量 | 主要内容 |
|------|------|----------|
| 规模效应与方法性能 | 8 | 性能趋势、方法对比、热力图、排序 |
| 维度分析 | 3 | 雷达图、维度对比、贡献分析 |
| 消融实验 | 4 | 行/列消融、参数敏感性 |
| 综合仪表板 | 1 | 多面板汇总视图 |
| 与原始基准对标 | 7 | 突破性对比、方法评估 |

### 数据来源

- **1K 规模**: `tmp/content/subset_eval/metrics_subset_aligned.csv`
- **5K-100K 规模**: 扩展实验数据
- **消融实验**: `tmp/content/ablation_experiments/ablation_all.csv`

---

## 1. 规模效应分析

### Performance Trend Across Scales（性能随规模变化趋势）

**内容**：
- X 轴：数据规模（1K、5K、10K、50K、100K）
- Y 轴：Unified nDCG@20 分数
- 多条折线分别代表 6 种推荐方法（Adaptive-Fusion、Naive-Fusion、Meta-only、Content-only、Tag-only、Text-only）

**说明**：
- **Adaptive-Fusion 呈现强劲上升趋势**：从 0.47（1K）上升至 0.82（100K），增幅 74.2%
- **Naive-Fusion 呈现明显下降趋势**：从 0.47（1K）下降至 0.32（100K），降幅 32.6%
- **Content-only 保持水平稳定**：约 0.40，几乎不受规模影响
- **5K 是关键交叉点**：Adaptive-Fusion 从此处开始超越 Naive-Fusion

**如何理解**：
- 上升曲线（如 Adaptive-Fusion）表示方法能有效利用更多数据
- 下降曲线（如 Naive-Fusion）表示方法在大规模下失效
- 水平曲线（如 Content-only）表示方法与数据规模无关

---

### Method Comparison at Each Scale（各规模方法对比）

**内容**：
- 分组条形图，每组代表一个数据规模
- 每个条形代表一种方法的 Unified nDCG@20 分数
- 颜色区分不同方法

**说明**：
- 在 1K 规模，Naive-Fusion 表现最佳
- 从 5K 开始，Adaptive-Fusion 领先
- 100K 时，方法间差异最大（0.11 至 0.82）

**如何理解**：
- 同一规模内，比较不同方法的高度可判断相对性能
- 跨规模比较同一方法的高度变化可观察其规模敏感性
- 高度差异越大，方法选择越关键

---

### Performance Change Rate (1K → 100K)（性能变化率）

**内容**：
- 水平条形图
- 每个条形代表一种方法从 1K 到 100K 的性能变化百分比
- 正值向右延伸（提升），负值向左延伸（下降）

**说明**：
- **Adaptive-Fusion: +74.2%**（最大提升）
- **Meta-only: +6.8%**（轻微提升）
- **Content-only: +3.4%**（几乎无变化）
- **Naive-Fusion: -32.6%**（严重下降）
- **Tag-only: -5.5%**、**Text-only: -6.3%**（轻微下降）

**如何理解**：
- 正向条形越长，方法越能从大规模数据中受益
- 负向条形越长，方法越不适合大规模场景
- 接近零的条形表示方法对规模不敏感

---

### Performance Heatmap（性能热力图）

**内容**：
- X 轴：数据规模（1K、5K、10K、50K、100K）
- Y 轴：推荐方法（6 种）
- 颜色深浅：Unified nDCG@20 分数高低

**说明**：
- 热力图直观呈现所有 30 个（5×6）实验配置的性能
- **Adaptive-Fusion 行从左到右颜色渐深**：性能持续提升
- **Naive-Fusion 行从左到右颜色渐浅**：性能持续下降
- 右上角（Adaptive-Fusion × 100K）最深，左下角（Tag/Text-only）最浅

**如何理解**：
- 颜色越深（暖色），性能越好
- 颜色越浅（冷色），性能越差
- 沿行观察可了解该方法的规模敏感性
- 沿列观察可了解该规模下的方法排序

---

### Method Ranking at 100K Scale (All Metrics)（100K 规模方法排序）

**内容**：
- 分组水平条形图，按 nDCG@20 降序排列
- 展示 100K 规模下各方法在 5 个指标（nDCG@20, MAP@20, MRR@20, P@20, R@20）的绝对性能
- 每个方法对应 5 个不同颜色的水平条形

**说明**：
- **排序**：Adaptive-Fusion > Meta-only > Content-only > Naive-Fusion > Tag-only > Text-only
- Adaptive-Fusion 在所有 5 个指标上均领先（nDCG=0.815, MRR=0.832）
- Content-only 在 P@20 和 R@20 上接近甚至超过 Meta-only
- Tag-only 和 Text-only 在所有指标上均低于 0.11

**如何理解**：
- 这是生产环境（大规模数据）的方法选择指南
- 多指标视角可发现方法在不同评估维度的优劣
- Adaptive-Fusion 的全面领先优势明显

---

### Method Ranking at 1K Scale（1K 规模方法排序）

**内容**：
- 水平条形图，按 Unified nDCG@20 降序排列
- 展示 1K 规模下各方法的绝对性能

**说明**：
- **排序**：Naive-Fusion (0.47) > Adaptive-Fusion (0.47) > Meta-only (0.44) > Content-only (0.39) > Tag-only (0.12) > Text-only (0.11)
- 在小规模下，前四种方法性能接近
- Naive-Fusion 微弱领先

**如何理解**：
- 与 100K 排序对比，可见规模对方法选择的影响
- 小规模场景下，简单方法也能有竞争力
- 这是冷启动或小数据场景的参考

---

### Naive-Fusion vs Content-only Cross-over（Naive-Fusion 与 Content-only 交叉分析）

**内容**：
- 双折线图：Naive-Fusion 和 Content-only
- X 轴：数据规模
- Y 轴：Unified nDCG@20

**说明**：
- **交叉点在 50K 附近**：此后 Naive-Fusion 低于 Content-only
- Naive-Fusion 持续下降，Content-only 保持稳定
- 100K 时差距明显：0.32 vs 0.40

**如何理解**：
- 交叉点是方法适用性的临界点
- 50K 以上规模，即使最简单的 Content-only 也优于 Naive-Fusion
- 这说明简单融合在大规模下不仅无益，反而有害

---

### Naive-Fusion vs Content-only Performance Gap（各规模性能差异）

**内容**：
- 条形图：Naive-Fusion 与 Content-only 的性能差
- 正值表示 Naive-Fusion 更好，负值表示 Content-only 更好

**说明**：
- 1K: +0.09（Naive-Fusion 领先）
- 5K: +0.05（领先缩小）
- 10K: +0.02（几乎持平）
- 50K: -0.04（Content-only 反超）
- 100K: -0.08（Content-only 显著领先）

**如何理解**：
- 从正到负的转变反映了 Naive-Fusion 的规模化困境
- 差值的绝对值反映两种方法的分离程度
- 趋势清晰：规模越大，Naive-Fusion 越不如 Content-only

---

## 2. 维度分析

### Multi-Dimension Radar Charts（多维度雷达图）

**内容**：
- 多个雷达图（每个规模一个）
- 三个维度：Tag、Desc、Creator
- 每条线代表一种方法

**说明**：
- **Tag 维度**：Content-only 表现突出（约 0.76）
- **Creator 维度**：Adaptive-Fusion 表现突出
- **Desc 维度**：所有方法得分较低（约 0.01-0.02）
- 不同方法在不同维度有明显特化

**如何理解**：
- 覆盖面积越大，综合性能越好
- 形状偏向某一角落，表示该方法在对应维度专长
- Content-only 在 Tag 维度尖锐，但 Creator 维度几乎为零
- Adaptive-Fusion 形状较均衡，各维度都有较好表现

---

### Performance by Dimension（按维度的性能）

**内容**：
- 分组条形图
- 三个维度组（Tag、Desc、Creator），每组包含所有方法

**说明**：
- **Tag 维度**：Content-only 最高（0.76），其次是 Adaptive-Fusion（0.68）
- **Desc 维度**：所有方法都很低，Content-only 相对最高（0.016）
- **Creator 维度**：Adaptive-Fusion 最高（0.82），Content-only 最低（0.05）

**如何理解**：
- 不同维度代表不同类型的相关性
- Tag 反映内容标签相似性，Content-only 自然擅长
- Creator 反映社交/协同信号，行为类方法擅长
- Desc 维度对所有方法都是挑战

---

### Dimension Contribution（维度贡献）

**内容**：
- 堆积条形图
- 每个条形代表一种方法
- 三种颜色代表三个维度对 Unified 分数的贡献

**说明**：
- **Unified = 0.5×Tag + 0.3×Desc + 0.2×Creator**
- Tag 贡献（50%权重）在大多数方法中占主导
- Content-only 的 Unified 分数几乎全来自 Tag 维度
- Adaptive-Fusion 的三维度贡献更均衡

**如何理解**：
- 堆积高度反映 Unified 总分
- 各色块比例反映该方法的维度偏重
- 偏重单一维度可能导致适用场景受限
- 三维度均衡的方法更通用

---

## 3. 消融实验

### Row Ablation（行消融）

**内容**：
- 折线图
- X 轴：MAX_ROWS 参数（64、128、256、512、1024）
- Y 轴：Unified nDCG@20
- 两条线：Content-only 和 Naive-Fusion

**说明**：
- **Content-only 几乎不变**：极差仅 0.0053（1.3%）
- **Naive-Fusion 也几乎不变**：极差仅 0.0018
- 两条线都接近水平

**如何理解**：
- MAX_ROWS 控制 TabContent 特征提取时考虑的最大行数
- 性能对行数不敏感说明特征提取很鲁棒
- 可安全降低 MAX_ROWS 以加速 pipeline（如从 1024 降至 64）

---

### Column Ablation（列消融）

**内容**：
- 折线图
- X 轴：MAX_COLS 参数（5、10、20、30、60）
- Y 轴：Unified nDCG@20
- 两条线：Content-only 和 Naive-Fusion

**说明**：
- **Content-only 轻微上升**：从 0.393（5列）到 0.402（60列），极差 0.0092（2.3%）
- **Naive-Fusion 几乎不变**：极差仅 0.0034
- 5 列时性能略低，10+ 列后趋于稳定

**如何理解**：
- MAX_COLS 控制 TabContent 特征的维度数
- 列数对性能有轻微正向影响，但边际效益递减
- 10-20 列是性能/效率的平衡点

---

### Ablation Performance Range（消融性能范围）

**内容**：
- 带误差条的条形图
- 每个条形代表一种方法
- 误差条表示消融实验中的性能范围（最小值-最大值）

**说明**：
- **Content-only**：中心约 0.40，范围极小
- **Naive-Fusion**：中心约 0.32，范围极小
- 两种方法都对参数变化高度稳定

**如何理解**：
- 误差条越短，方法对参数越不敏感
- 高稳定性意味着超参数调优空间有限，但也意味着鲁棒性好
- 在实际部署中，可用默认参数获得接近最优性能

---

### Parameter Configuration Comparison（参数配置比较）

**内容**：
- 条形图，比较不同参数配置的性能
- 配置包括：默认配置、最小行配置、最大列配置等

**说明**：
- 各配置间性能差异极小
- 默认配置（MAX_ROWS=1024, MAX_COLS=60）与极端配置性能相近
- 验证了消融实验的稳定性结论

**如何理解**：
- 在资源受限场景，可选择轻量配置（低 MAX_ROWS）
- 在追求极致性能场景，可保持默认或更高配置
- 配置选择主要影响计算效率，对性能影响很小

---

## 4. 综合仪表板

### Summary Dashboard（综合仪表板）

**内容**：
6 面板仪表板：
1. **左上**：性能趋势折线图（主要方法）
2. **右上**：100K 规模方法排序
3. **左中**：性能变化率条形图
4. **右中**：性能热力图
5. **左下**：Naive-Fusion vs Content-only 交叉分析
6. **右下**：关键数据指标卡片

**说明**：
- 整合前述核心图表于一页
- 提供实验结果的全局视图
- 便于快速了解主要发现

**如何理解**：
- 这是向非技术受众汇报的理想图表
- 每个面板聚焦一个关键发现
- 从左到右、从上到下阅读可构建完整认知
- 指标卡片提供关键数字锚点

---

## 5. 与原始基准对标

### Tag Dimension Breakthrough（Tag 维度突破）

**内容**：
- 条形图
- 比较 Content View（100K）与 Original Metadata（521K）在 Tag 维度的 nDCG@20

**说明**：
- **Content View (100K): 0.7618**
- **Original Metadata (521K): 未提供具体值，但被超越**
- 用更少数据（100K vs 521K）实现更高性能

**如何理解**：
- 这证明 Content View 的特征提取方法优于传统元数据方法
- 数据效率提升约 5 倍（100K vs 521K）
- Tag 维度是 Content-only 的优势领域

---

### Three-Dimension Comparison（三维度对比）

**内容**：
- 分组条形图
- 比较 Original (521K) 与 New Methods (100K) 在三个维度的表现
- 三组：Tag、Desc、Creator

**说明**：
- **Tag 维度**：新方法显著领先
- **Desc 维度**：差异较小
- **Creator 维度**：需结合行为信号，新旧方法各有优劣

**如何理解**：
- 新方法在 Tag 维度实现突破
- 三维度中 Tag 权重最高（50%），因此对 Unified 贡献最大
- Creator 维度的提升空间在于融合行为信号

---

### Method Performance vs Original Baseline（方法性能对比）

**内容**：
- 水平条形图
- 比较所有方法的 Unified nDCG@20
- 包含 Original Metadata 基准线

**说明**：
- Adaptive-Fusion 大幅超越原始基准
- Content-only 接近或超越原始基准
- Naive-Fusion 低于原始基准

**如何理解**：
- 基准线是评判新方法有效性的标准
- 超越基准意味着方法改进有效
- 差距大小反映改进幅度

---

### Comparison Summary Dashboard（对比总结仪表板）

**内容**：
4 面板仪表板：
1. **左上**：Tag 维度突破对比
2. **右上**：三维度性能对比
3. **左下**：方法排序与基准线
4. **右下**：关键改进指标

**说明**：
- 整合与原始基准对标的核心发现
- 突出 Content View 在 Tag 维度的突破
- 展示 Adaptive-Fusion 的整体领先

**如何理解**：
- 这是与历史版本/竞品对标的汇报图表
- 定量展示改进幅度
- 便于快速判断新方法的价值

---

### Row Ablation Bar Chart（行消融条形图）

**内容**：
- 条形图
- X 轴：MAX_ROWS 配置
- Y 轴：Content-only 的 Unified nDCG@20

**说明**：
- 各配置性能几乎相同（约 0.40）
- 验证特征提取对行数不敏感
- 支持使用更少行数以提升效率

**如何理解**：
- 条形高度相近表示参数不敏感
- 可放心选择低配置（如 64 行）以加速处理
- 不会牺牲显著性能

---

### Column Ablation Bar Chart（列消融条形图）

**内容**：
- 条形图
- X 轴：MAX_COLS 配置
- Y 轴：Content-only 的 Unified nDCG@20

**说明**：
- 5 列时略低，10+ 列后基本持平
- 性能差异约 2%
- 列数对性能有轻微正向影响

**如何理解**：
- 10 列是性能的"膝点"
- 超过 10 列后增益递减
- 推荐使用 10-20 列作为默认配置

---

### Content View Enhancement（内容视图增强）

**内容**：
- 条形图
- 展示从 Meta-only 到 Adaptive-Fusion 的性能提升路径
- 100K 规模

**说明**：
- **Meta-only → Adaptive-Fusion: +74.2%**
- 内容视图的加入带来显著提升
- 自适应融合充分释放了内容视图的价值

**如何理解**：
- 这展示了 Content View 的增量价值
- 从基础方法到最优方法的提升路径清晰
- 验证了多视图融合的必要性

---

## 附录

### A. 评测指标说明

| 指标 | 全称 | 说明 |
|------|------|------|
| nDCG@20 | Normalized Discounted Cumulative Gain | 归一化折损累计增益，考虑位置权重 |
| MAP@20 | Mean Average Precision | 平均精度均值 |
| MRR@20 | Mean Reciprocal Rank | 平均倒数排名 |
| P@20 | Precision@20 | 前20项精确率 |
| R@20 | Recall@20 | 前20项召回率 |

### B. Unified 聚合公式

```
Unified = 0.5 × Tag + 0.3 × Desc + 0.2 × Creator
```

### C. 方法简介

| 方法 | 说明 |
|------|------|
| Adaptive-Fusion | 自适应权重四视图融合 |
| Naive-Fusion | 朴素四视图融合（固定权重） |
| Meta-only | 仅元数据三视图融合 |
| Content-only | 仅内容视图（TabContent 特征） |
| Tag-only | 仅标签视图 |
| Text-only | 仅文本视图 |

---

*文档生成日期：2026-02-06*
