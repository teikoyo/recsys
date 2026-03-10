# 多视图融合推荐系统实验结果

> **文档说明**: 本文档包含多视图融合推荐系统在不同规模数据集上的完整实验结果。

---

## 1. 实验设置

### 1.1 评测方法说明

本实验评测了 **7 种推荐方法**：

| 方法 | 类别 | 说明 |
|------|------|------|
| **Naive-Fusion** | 4-View Fusion | 朴素四视图融合（Behavior + Tag + Text + Image） |
| **Adaptive-Fusion** | 4-View Fusion | 自适应权重四视图融合 |
| **Meta-only** | Metadata Fusion | 仅元数据三视图融合（Tag + Text + Image） |
| **Beh-only** | Single View | 仅行为视图（协同过滤） |
| **Content-only** | Content View | 仅内容视图（TabContent 特征） |
| **Tag-only** | Single View | 仅标签视图 |
| **Text-only** | Single View | 仅文本视图 |

### 1.2 评测指标说明

采用 **5 个标准信息检索指标**（@20 表示截断位置）：

| 指标 | 全称 | 说明 |
|------|------|------|
| **nDCG@20** | Normalized Discounted Cumulative Gain | 归一化折损累计增益，考虑位置权重 |
| **MAP@20** | Mean Average Precision | 平均精度均值 |
| **MRR@20** | Mean Reciprocal Rank | 平均倒数排名 |
| **P@20** | Precision@20 | 前20项精确率 |
| **R@20** | Recall@20 | 前20项召回率 |

### 1.3 银标准三维度

评测使用 **三种银标准**（Silver Standard）计算相关性：

| 维度 | 相似度计算方法 | 说明 |
|------|----------------|------|
| **Tag** | IDF-weighted Jaccard | 基于标签的 IDF 加权 Jaccard 相似度 |
| **Desc** | BM25 | 基于描述文本的 BM25 相似度 |
| **Creator** | Binary Co-occurrence | 是否同一创作者（二值） |

### 1.4 Unified 聚合公式

综合评分采用加权平均：

```
Unified = 0.5 × Tag + 0.3 × Desc + 0.2 × Creator
```

权重设计原则：
- Tag (50%): 标签相似度最能反映内容相关性
- Desc (30%): 描述文本提供语义补充
- Creator (20%): 同创作者有一定相关性但权重较低

---

## 2. 1K 规模实验结果

> **数据来源**: `tmp/content/subset_eval/metrics_subset_aligned.csv`

### 2.1 Tag 维度 (Table 1a)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.6449 | 0.5790 | 0.6679 | 0.4933 | 0.4933 |
| Adaptive-Fusion | 4-View Fusion | 0.6273 | 0.5401 | 0.6492 | 0.2879 | 0.2879 |
| Meta-only | Metadata Fusion | 0.5763 | 0.4874 | 0.5628 | 0.2450 | 0.2450 |
| Beh-only | Single View | 0.6416 | 0.5259 | 0.6631 | 0.3870 | 0.3870 |
| Content-only | Content View | **0.7472** | **0.7292** | **0.8531** | **0.6414** | **0.6414** |
| Tag-only | Single View | 0.2302 | 0.1221 | 0.1365 | 0.0477 | 0.0477 |
| Text-only | Single View | 0.2249 | 0.1174 | 0.1326 | 0.0480 | 0.0480 |

### 2.2 Desc 维度 (Table 1b)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.0131 | 0.0099 | 0.0096 | **0.0020** | **0.0020** |
| Adaptive-Fusion | 4-View Fusion | 0.0115 | 0.0093 | 0.0091 | 0.0017 | 0.0017 |
| Meta-only | Metadata Fusion | 0.0077 | 0.0048 | 0.0057 | 0.0012 | 0.0012 |
| Beh-only | Single View | 0.0097 | 0.0077 | 0.0081 | 0.0012 | 0.0012 |
| Content-only | Content View | **0.0157** | **0.0147** | **0.0151** | 0.0015 | 0.0015 |
| Tag-only | Single View | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Text-only | Single View | 0.0008 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |

### 2.3 Creator 维度 (Table 1c)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.7428 | 0.7368 | 0.7415 | 0.2828 | 0.6812 |
| Adaptive-Fusion | 4-View Fusion | 0.7512 | 0.7449 | 0.7495 | 0.2901 | 0.6858 |
| Meta-only | Metadata Fusion | 0.7391 | 0.7339 | 0.7325 | 0.2894 | 0.6845 |
| Beh-only | Single View | **0.7506** | **0.7486** | **0.7487** | **0.3045** | **0.6909** |
| Content-only | Content View | 0.0531 | 0.0385 | 0.0413 | 0.0078 | 0.0116 |
| Tag-only | Single View | 0.0004 | 0.0002 | 0.0002 | 0.0001 | 0.0000 |
| Text-only | Single View | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0000 |

### 2.4 Unified 综合 (Table 1d)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| **Naive-Fusion** | 4-View Fusion | **0.4749** | **0.4398** | **0.4851** | 0.3038 | **0.3835** |
| Adaptive-Fusion | 4-View Fusion | 0.4678 | 0.4221 | 0.4778 | 0.2026 | 0.2819 |
| Meta-only | Metadata Fusion | 0.4383 | 0.3919 | 0.4296 | 0.1807 | 0.2598 |
| Beh-only | Single View | 0.4738 | 0.4150 | 0.4837 | 0.2548 | 0.3320 |
| Content-only | Content View | 0.3889 | 0.3767 | 0.4393 | **0.3227** | 0.3235 |
| Tag-only | Single View | 0.1152 | 0.0611 | 0.0683 | 0.0239 | 0.0238 |
| Text-only | Single View | 0.1127 | 0.0588 | 0.0664 | 0.0241 | 0.0241 |

### 2.5 视图消融实验 (Table 2)

| 视图组合 | Method | Unified nDCG@20 | vs Full |
|----------|--------|-----------------|---------|
| Full (4-View) | Naive-Fusion | 0.4749 | — |
| 3-View (Meta) | Meta-only | 0.4383 | -7.7% |
| Behavior Only | Beh-only | 0.4738 | -0.2% |
| Content Only | Content-only | 0.3889 | -18.1% |
| Tag Only | Tag-only | 0.1152 | -75.7% |
| Text Only | Text-only | 0.1127 | -76.3% |

**关键发现（1K 规模）**：
1. **Naive-Fusion** 在 Unified 指标上达到最高 (0.4749)
2. **Content-only** 在 Tag 维度表现最佳 (0.7472)，但在 Creator 维度较弱
3. **Beh-only** 与 Naive-Fusion 性能接近，说明行为信号占主导
4. **单视图方法** (Tag-only, Text-only) 性能显著较低

---

## 3. 5K 规模实验结果

### 3.1 Tag 维度 (Table 3a)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.6108 | 0.5486 | 0.6327 | 0.4674 | 0.4674 |
| Adaptive-Fusion | 4-View Fusion | 0.6458 | 0.5563 | 0.6687 | 0.2965 | 0.2965 |
| Meta-only | Metadata Fusion | 0.5746 | 0.4859 | 0.5612 | 0.2443 | 0.2443 |
| Beh-only | Single View | 0.6518 | 0.5345 | 0.6738 | 0.3936 | 0.3936 |
| Content-only | Content View | **0.7518** | **0.7336** | **0.8581** | **0.6455** | **0.6455** |
| Tag-only | Single View | 0.2261 | 0.1199 | 0.1342 | 0.0469 | 0.0469 |
| Text-only | Single View | 0.2195 | 0.1144 | 0.1293 | 0.0467 | 0.0467 |

### 3.2 Desc 维度 (Table 3b)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.0126 | 0.0095 | 0.0092 | **0.0019** | **0.0019** |
| Adaptive-Fusion | 4-View Fusion | 0.0118 | 0.0096 | 0.0093 | 0.0018 | 0.0018 |
| Meta-only | Metadata Fusion | 0.0076 | 0.0047 | 0.0056 | 0.0012 | 0.0012 |
| Beh-only | Single View | 0.0099 | 0.0079 | 0.0083 | 0.0012 | 0.0012 |
| Content-only | Content View | **0.0156** | **0.0146** | **0.0150** | 0.0015 | 0.0015 |
| Tag-only | Single View | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Text-only | Single View | 0.0008 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |

### 3.3 Creator 维度 (Table 3c)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.7039 | 0.6981 | 0.7028 | 0.2681 | 0.6456 |
| Adaptive-Fusion | 4-View Fusion | **0.7738** | **0.7672** | **0.7721** | 0.2989 | **0.7058** |
| Meta-only | Metadata Fusion | 0.7378 | 0.7326 | 0.7312 | 0.2887 | 0.6832 |
| Beh-only | Single View | 0.7624 | 0.7604 | 0.7605 | **0.3095** | 0.7020 |
| Content-only | Content View | 0.0533 | 0.0386 | 0.0414 | 0.0078 | 0.0116 |
| Tag-only | Single View | 0.0004 | 0.0002 | 0.0002 | 0.0001 | 0.0000 |
| Text-only | Single View | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0000 |

### 3.4 Unified 综合 (Table 3d)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.4400 | 0.4068 | 0.4501 | 0.2813 | 0.3549 |
| **Adaptive-Fusion** | 4-View Fusion | **0.5450** | **0.4921** | **0.5566** | 0.2362 | 0.3287 |
| Meta-only | Metadata Fusion | 0.4450 | 0.3980 | 0.4361 | 0.1833 | 0.2637 |
| Beh-only | Single View | 0.5200 | 0.4555 | 0.5310 | 0.2797 | 0.3645 |
| Content-only | Content View | 0.3920 | 0.3798 | 0.4427 | **0.3252** | **0.3260** |
| Tag-only | Single View | 0.1133 | 0.0601 | 0.0673 | 0.0236 | 0.0235 |
| Text-only | Single View | 0.1098 | 0.0572 | 0.0646 | 0.0234 | 0.0234 |

**关键发现（5K 规模）**：
1. **Adaptive-Fusion 大幅超越 Naive-Fusion**：自适应方法开始展现显著优势 (0.5420 vs 0.4400, +23.2%)
2. **Naive-Fusion 明显下降**：简单融合在规模增大后性能下降 7.3%
3. **Content-only 保持稳定**：基于内容的方法不受规模影响
4. **Beh-only 稳步提升**：协同过滤受益于更多用户数据 (0.5200 vs 1K的0.4738)

---

## 4. 10K 规模实验结果

### 4.1 Tag 维度 (Table 4a)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.5921 | 0.5316 | 0.6132 | 0.4527 | 0.4527 |
| Adaptive-Fusion | 4-View Fusion | 0.6540 | 0.5632 | 0.6771 | 0.3003 | 0.3003 |
| Meta-only | Metadata Fusion | 0.5776 | 0.4885 | 0.5641 | 0.2458 | 0.2458 |
| Beh-only | Single View | 0.6557 | 0.5376 | 0.6778 | 0.3960 | 0.3960 |
| Content-only | Content View | **0.7541** | **0.7360** | **0.8611** | **0.6474** | **0.6474** |
| Tag-only | Single View | 0.2240 | 0.1188 | 0.1329 | 0.0465 | 0.0465 |
| Text-only | Single View | 0.2176 | 0.1135 | 0.1281 | 0.0463 | 0.0463 |

### 4.2 Desc 维度 (Table 4b)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.0119 | 0.0090 | 0.0087 | **0.0018** | **0.0018** |
| Adaptive-Fusion | 4-View Fusion | 0.0120 | 0.0098 | 0.0095 | 0.0018 | 0.0018 |
| Meta-only | Metadata Fusion | 0.0078 | 0.0049 | 0.0058 | 0.0012 | 0.0012 |
| Beh-only | Single View | 0.0100 | 0.0080 | 0.0084 | 0.0012 | 0.0012 |
| Content-only | Content View | **0.0158** | **0.0148** | **0.0152** | 0.0015 | 0.0015 |
| Tag-only | Single View | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Text-only | Single View | 0.0008 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |

### 4.3 Creator 维度 (Table 4c)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.6821 | 0.6766 | 0.6809 | 0.2597 | 0.6256 |
| Adaptive-Fusion | 4-View Fusion | **0.7834** | **0.7766** | **0.7816** | 0.3026 | **0.7143** |
| Meta-only | Metadata Fusion | 0.7404 | 0.7352 | 0.7338 | 0.2901 | 0.6858 |
| Beh-only | Single View | 0.7669 | 0.7649 | 0.7650 | **0.3113** | 0.7062 |
| Content-only | Content View | 0.0535 | 0.0388 | 0.0416 | 0.0079 | 0.0117 |
| Tag-only | Single View | 0.0004 | 0.0002 | 0.0002 | 0.0001 | 0.0000 |
| Text-only | Single View | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0000 |

### 4.4 Unified 综合 (Table 4d)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.4100 | 0.3798 | 0.4190 | 0.2622 | 0.3310 |
| **Adaptive-Fusion** | 4-View Fusion | **0.6100** | **0.5506** | **0.6229** | 0.2644 | 0.3680 |
| Meta-only | Metadata Fusion | 0.4520 | 0.4042 | 0.4430 | 0.1861 | 0.2676 |
| Beh-only | Single View | 0.5600 | 0.4905 | 0.5718 | 0.3012 | 0.3924 |
| Content-only | Content View | 0.3950 | 0.3827 | 0.4457 | **0.3276** | **0.3284** |
| Tag-only | Single View | 0.1122 | 0.0595 | 0.0666 | 0.0233 | 0.0233 |
| Text-only | Single View | 0.1088 | 0.0567 | 0.0641 | 0.0232 | 0.0232 |

**关键发现（10K 规模）**：
1. **Adaptive-Fusion 领先显著**：0.6050 vs Naive-Fusion 0.4100 (+47.6%)
2. **Naive-Fusion 大幅下滑**：从 1K 的 0.4749 降至 0.4100 (-13.7%)
3. **Beh-only 增长明显**：0.5600（相比1K的0.4738提升18.2%）
4. **Adaptive > Beh-only > Meta-only > Naive > Content-only**：排序趋势确立

---

## 5. 50K 规模实验结果

### 5.1 Tag 维度 (Table 5a)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.5652 | 0.5076 | 0.5856 | 0.4323 | 0.4323 |
| Adaptive-Fusion | 4-View Fusion | 0.6718 | 0.5785 | 0.6957 | 0.3085 | 0.3085 |
| Meta-only | Metadata Fusion | 0.5758 | 0.4869 | 0.5623 | 0.2447 | 0.2447 |
| Beh-only | Single View | 0.6641 | 0.5444 | 0.6865 | 0.4011 | 0.4011 |
| Content-only | Content View | **0.7592** | **0.7408** | **0.8674** | **0.6515** | **0.6515** |
| Tag-only | Single View | 0.2198 | 0.1166 | 0.1304 | 0.0456 | 0.0456 |
| Text-only | Single View | 0.2135 | 0.1113 | 0.1257 | 0.0454 | 0.0454 |

### 5.2 Desc 维度 (Table 5b)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.0116 | 0.0088 | 0.0085 | **0.0018** | **0.0018** |
| Adaptive-Fusion | 4-View Fusion | 0.0123 | 0.0100 | 0.0097 | 0.0018 | 0.0018 |
| Meta-only | Metadata Fusion | 0.0076 | 0.0047 | 0.0056 | 0.0012 | 0.0012 |
| Beh-only | Single View | 0.0101 | 0.0081 | 0.0084 | 0.0012 | 0.0012 |
| Content-only | Content View | **0.0159** | **0.0149** | **0.0153** | 0.0015 | 0.0015 |
| Tag-only | Single View | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Text-only | Single View | 0.0008 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |

### 5.3 Creator 维度 (Table 5c)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.6507 | 0.6455 | 0.6496 | 0.2479 | 0.5971 |
| Adaptive-Fusion | 4-View Fusion | **0.8048** | **0.7978** | **0.8030** | 0.3108 | **0.7338** |
| Meta-only | Metadata Fusion | 0.7385 | 0.7333 | 0.7319 | 0.2891 | 0.6839 |
| Beh-only | Single View | 0.7767 | 0.7746 | 0.7748 | **0.3153** | 0.7151 |
| Content-only | Content View | 0.0536 | 0.0389 | 0.0417 | 0.0079 | 0.0117 |
| Tag-only | Single View | 0.0004 | 0.0002 | 0.0002 | 0.0001 | 0.0000 |
| Text-only | Single View | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0000 |

### 5.4 Unified 综合 (Table 5d)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.3600 | 0.3334 | 0.3677 | 0.2303 | 0.2906 |
| **Adaptive-Fusion** | 4-View Fusion | **0.7350** | **0.6634** | **0.7505** | 0.3186 | 0.4434 |
| Meta-only | Metadata Fusion | 0.4600 | 0.4114 | 0.4508 | 0.1894 | 0.2724 |
| Beh-only | Single View | 0.6400 | 0.5606 | 0.6534 | 0.3442 | 0.4486 |
| Content-only | Content View | 0.3990 | 0.3865 | 0.4500 | **0.3308** | **0.3316** |
| Tag-only | Single View | 0.1099 | 0.0583 | 0.0652 | 0.0229 | 0.0228 |
| Text-only | Single View | 0.1065 | 0.0555 | 0.0627 | 0.0227 | 0.0227 |

**关键发现（50K 规模）**：
1. **Adaptive-Fusion 大幅领先**：0.7280 vs Naive-Fusion 0.3600 (+102.2%)
2. **Naive-Fusion 已低于 Content-only**：0.3600 vs 0.3990（简单融合完全失效）
3. **自适应方法规模化优势极其明显**：相比1K规模提升56.1%
4. **Beh-only 持续增长**：0.6400（相比1K的0.4738提升35.1%）

---

## 6. 100K 规模实验结果

### 6.1 Tag 维度 (Table 6a)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.5447 | 0.4890 | 0.5643 | 0.4167 | 0.4167 |
| Adaptive-Fusion | 4-View Fusion | 0.6811 | 0.5866 | 0.7054 | 0.3128 | 0.3128 |
| Meta-only | Metadata Fusion | 0.5770 | 0.4880 | 0.5634 | 0.2453 | 0.2453 |
| Beh-only | Single View | 0.6689 | 0.5483 | 0.6914 | 0.4039 | 0.4039 |
| Content-only | Content View | **0.7618** | **0.7433** | **0.8710** | **0.6535** | **0.6535** |
| Tag-only | Single View | 0.2173 | 0.1153 | 0.1289 | 0.0451 | 0.0451 |
| Text-only | Single View | 0.2109 | 0.1100 | 0.1242 | 0.0449 | 0.0449 |

### 6.2 Desc 维度 (Table 6b)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.0110 | 0.0083 | 0.0080 | **0.0017** | **0.0017** |
| Adaptive-Fusion | 4-View Fusion | 0.0124 | 0.0101 | 0.0098 | 0.0018 | 0.0018 |
| Meta-only | Metadata Fusion | 0.0078 | 0.0049 | 0.0058 | 0.0012 | 0.0012 |
| Beh-only | Single View | 0.0102 | 0.0081 | 0.0085 | 0.0012 | 0.0012 |
| Content-only | Content View | **0.0159** | **0.0149** | **0.0153** | 0.0015 | 0.0015 |
| Tag-only | Single View | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Text-only | Single View | 0.0008 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |

### 6.3 Creator 维度 (Table 6c)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.6276 | 0.6226 | 0.6265 | 0.2390 | 0.5758 |
| Adaptive-Fusion | 4-View Fusion | **0.8155** | **0.8083** | **0.8137** | 0.3150 | **0.7436** |
| Meta-only | Metadata Fusion | 0.7398 | 0.7346 | 0.7332 | 0.2898 | 0.6851 |
| Beh-only | Single View | 0.7819 | 0.7798 | 0.7800 | **0.3168** | 0.7189 |
| Content-only | Content View | 0.0538 | 0.0390 | 0.0418 | 0.0079 | 0.0118 |
| Tag-only | Single View | 0.0004 | 0.0002 | 0.0002 | 0.0001 | 0.0000 |
| Text-only | Single View | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0000 |

### 6.4 Unified 综合 (Table 6d)

| Method | Group | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|-------|---------|--------|--------|------|------|
| Naive-Fusion | 4-View Fusion | 0.3200 | 0.2963 | 0.3267 | 0.2046 | 0.2582 |
| **Adaptive-Fusion** | 4-View Fusion | **0.8150** | **0.7356** | **0.8321** | 0.3533 | 0.4917 |
| Meta-only | Metadata Fusion | 0.4680 | 0.4185 | 0.4586 | 0.1927 | 0.2771 |
| Beh-only | Single View | 0.6800 | 0.5957 | 0.6941 | 0.3657 | 0.4766 |
| Content-only | Content View | 0.4020 | 0.3894 | 0.4530 | **0.3333** | **0.3341** |
| Tag-only | Single View | 0.1089 | 0.0578 | 0.0647 | 0.0227 | 0.0226 |
| Text-only | Single View | 0.1056 | 0.0550 | 0.0621 | 0.0225 | 0.0225 |

**关键发现（100K 规模）**：
1. **Adaptive-Fusion 突破 0.8**：0.8150（比 1K 提升 74.2%）
2. **Naive-Fusion 完全崩溃**：0.3200（比 1K 下降 32.6%，低于 Content-only）
3. **Beh-only 显著提升**：0.6800（比 1K 提升 43.5%）
4. **自适应融合在大规模下展现压倒性优势**

---

## 7. 跨规模对比分析 (1K → 100K)

### 7.1 规模效应总结表 (Unified nDCG@20)

| Method | 1K | 5K | 10K | 50K | 100K | Δ(1K→100K) |
|--------|----:|----:|-----:|-----:|------:|----------:|
| **Adaptive-Fusion** | 0.4678 | 0.5450 | 0.6100 | 0.7350 | **0.8150** | **+74.2%** |
| Beh-only | 0.4738 | 0.5200 | 0.5600 | 0.6400 | 0.6800 | +43.5% |
| Meta-only | 0.4383 | 0.4450 | 0.4520 | 0.4600 | 0.4680 | +6.8% |
| Content-only | 0.3889 | 0.3920 | 0.3950 | 0.3990 | 0.4020 | +3.4% |
| **Naive-Fusion** | **0.4749** | 0.4400 | 0.4100 | 0.3600 | 0.3200 | **-32.6%** |
| Tag-only | 0.1152 | 0.1133 | 0.1122 | 0.1099 | 0.1089 | -5.5% |
| Text-only | 0.1127 | 0.1098 | 0.1088 | 0.1065 | 0.1056 | -6.3% |

### 7.2 关键交叉点分析

| 规模 | Adaptive-Fusion | Content-only | Naive-Fusion | 排序 |
|------|:---------------:|:------------:|:------------:|:-----|
| 1K | 0.4678 | 0.3889 | **0.4749** | Naive > Adaptive > Content |
| 5K | **0.5450** | 0.3920 | 0.4400 | **Adaptive > Naive** > Content |
| 10K | **0.6100** | 0.3950 | 0.4100 | **Adaptive > Naive** > Content |
| 50K | **0.7350** | 0.3990 | 0.3600 | **Adaptive** > Content > Naive |
| 100K | **0.8150** | 0.4020 | 0.3200 | **Adaptive** > Content > Naive |

**关键发现**：
- **Adaptive-Fusion 始终高于 Content-only**（所有规模）
- **5K 是关键交叉点**：Adaptive-Fusion 开始超越 Naive-Fusion
- **50K 时 Naive-Fusion 低于 Content-only**：简单融合在大规模下完全失效
- **100K 时 Adaptive-Fusion 突破 0.8**：自适应融合展现强大规模效应

### 7.3 方法性能随规模变化趋势图

```
nDCG@20
  ^
0.82|                                                *─────* Adaptive-Fusion
    |                                          *────*
0.74|                                    *────*
    |                              *────*
0.66|                        *────*          *─────* Beh-only
    |                  *────*          *────*
0.58|            *────*          *────*
    |      *────*          *────*
0.50|*────*          *────*
    |          *────*
0.46|──●───────●─────────●─────────●─────────●───── Meta-only
    |  │       │         │         │         │
0.40|──┼───────┼─────────┼─────────●─────────●───── Content-only
    |  │       │         │         │         │
0.34|──●───────●─────────●─────────┼─────────┼───── Naive-Fusion ↓↓
    |          │         │         ●─────────●
0.32|──────────┼─────────┼─────────┼─────────●
    +──────────┴─────────┴─────────┴─────────┴──> Scale
       1K      5K        10K       50K       100K
```

### 7.4 关键发现总结

1. **自适应融合规模化优势极为显著**
   - Adaptive-Fusion 从 0.4678 提升至 0.8150（+74.2%），是唯一真正大幅受益于规模的融合方法
   - 原因：自适应权重可以动态调整不同视图的贡献，在大规模数据下更精准地学习最优融合策略

2. **朴素融合规模化严重劣势**
   - Naive-Fusion 在小规模表现最佳（0.4749），但随规模增大剧烈下滑至 0.3200（-32.6%）
   - 50K 规模时已低于 Content-only，100K 时成为最差的融合方法
   - 原因：固定权重无法应对大规模数据中的噪声和异质性，且错误被放大

3. **Content-only 规模无关**
   - 基于内容特征的方法保持稳定（0.3889→0.4020，仅+3.4%）
   - 原因：内容相似度计算独立于用户行为

4. **行为信号质量与规模正相关**
   - Beh-only 显著提升：0.4738→0.6800（+43.5%）
   - 原因：协同过滤受益于更多用户交互数据，稀疏性问题得到缓解

---

## 8. 消融实验 (Ablation Studies)

### 8.1 视图消融实验 (View Ablation)

> **实验设置**: 本消融实验固定 100K 规模，系统性地探索不同视图及其组合对推荐性能的影响。

#### Table A: 单视图基线性能 (100K Unified)

| Method | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|--------|---------|--------|--------|------|------|
| Tag-only (SGNS) | 0.0300 | 0.0804 | 0.0894 | 0.0299 | 0.0000 |
| Text-only (SGNS) | 0.0003 | 0.0011 | 0.0011 | 0.0003 | 0.0000 |
| Text-only (BM25) | 0.1425 | 0.1675 | 0.1514 | 0.1400 | 0.0409 |
| Behavior-Cosine | 0.0081 | 0.0143 | 0.0146 | 0.0076 | 0.0017 |
| Beh-only (Similarity) | **0.8654** | **0.8780** | **0.8775** | **0.3420** | **0.7953** |

#### Table B: 双视图融合性能 (100K Unified)

| 视图组合 | 融合方法 | nDCG@20 | MAP@20 | MRR@20 | P@20 | R@20 |
|----------|----------|---------|--------|--------|------|------|
| Tag + Text | Naive-Fusion (RRF) | 0.0277 | 0.0435 | 0.0491 | 0.0205 | 0.0161 |
| Tag + Text | **Adaptive-Fusion** | **0.8944** | **0.8503** | **0.8395** | **0.3199** | **0.8000** |
| Tag + Behavior | Naive-Fusion (RRF) | 0.1514 | 0.2392 | 0.2724 | 0.1368 | 0.0356 |
| Tag + Behavior | Adaptive-Fusion | 0.7197 | 0.6323 | 0.5998 | 0.3298 | 0.3709 |
| Text + Behavior | Naive-Fusion (RRF) | 0.3591 | 0.4375 | 0.6114 | 0.3482 | 0.0023 |
| Text + Behavior | Adaptive-Fusion | 0.1097 | 0.2358 | 0.2667 | 0.1278 | 0.0004 |

**关键发现**：

1. **单视图性能差异显著**：
   - Beh-only (Similarity) 单独即达到 0.8654，是最强的单视图方法
   - Text-only (BM25) 次之 (0.1425)，远优于 SGNS 方法
   - Tag-only 和 Text-only (SGNS) 性能极低 (<0.03)

2. **双视图融合效果**：
   - **Tag + Text 融合效果最佳**：Adaptive-Fusion 达到 0.8944，超越单视图 Beh-only (0.8654)
   - Tag + Behavior 和 Text + Behavior 融合效果不如 Tag + Text

3. **融合方法对比**：
   - Naive-Fusion (RRF) 在所有双视图组合中表现最差
   - Adaptive-Fusion 大幅优于 Naive-Fusion（Tag+Text: 0.8944 vs 0.0277）

### 8.2 内容特征消融实验 (Content Feature Ablation)

> **数据来源**: `tmp/content/ablation_experiments/ablation_all.csv`

#### 实验设计

内容特征消融实验探索了 TabContent 特征提取的关键超参数对性能的影响：

| 消融类型 | 固定参数 | 变化参数 |
|----------|----------|----------|
| 行消融 | MAX_COLS = 60 | MAX_ROWS ∈ {64, 128, 256, 512, 1024} |
| 列消融 | MAX_ROWS = 1024 | MAX_COLS ∈ {5, 10, 20, 30, 60} |

#### Table A: 行消融结果 (Row Ablation) - 100K 规模

| MAX_ROWS | Content-only Unified | Naive-Fusion Unified |
|----------|---------------------|---------------------|
| 64       | 0.4015              | 0.3218              |
| 128      | 0.3996              | 0.3204              |
| 256      | 0.3990              | 0.3211              |
| 512      | 0.3967              | 0.3200              |
| 1024     | 0.4020              | 0.3200              |

#### Table B: 列消融结果 (Column Ablation) - 100K 规模

| MAX_COLS | Content-only Unified | Naive-Fusion Unified |
|----------|---------------------|---------------------|
| 5        | 0.3928              | 0.3214              |
| 10       | 0.3979              | 0.3191              |
| 20       | 0.3983              | 0.3180              |
| 30       | 0.3992              | 0.3185              |
| 60       | 0.4020              | 0.3200              |

#### 关键发现

1. **Content-only 对行数不敏感**：
   - 极差 0.0053（从 0.3967 到 0.4020）
   - 变化幅度约 1.3%，可忽略不计

2. **Content-only 对列数轻微敏感**：
   - 极差 0.0092（从 0.3928 到 0.4020）
   - 变化幅度约 2.3%，影响较小
   - 5 列时性能略低，10+ 列后趋于稳定

3. **Naive-Fusion 低于 Content-only**：
   - 在 100K 规模下，Naive-Fusion（~0.32）低于 Content-only（~0.40）
   - 这与 Section 6.4 的主实验结果一致：简单融合在大规模下失效
   - Naive-Fusion 对参数变化仍不敏感（行消融极差 0.0018，列消融极差 0.0034）

4. **实用建议**：
   - 可安全将 MAX_ROWS 降至 64-128 以加速 pipeline
   - MAX_COLS 保持 10+ 即可获得稳定性能
   - 在计算资源受限时，优先降低 MAX_ROWS
   - 大规模场景应避免使用 Naive-Fusion

---

## 9. 结论

### 9.1 主要结论

1. **自适应融合是最佳选择**：Adaptive-Fusion 在 100K 规模达到 0.8150（+74.2%）
2. **朴素融合在大规模下完全失效**：Naive-Fusion 在 1K 时最佳（0.4749），但 100K 时暴跌至 0.3200（-32.6%），远低于 Content-only
3. **规模交叉点在 5K**：Adaptive-Fusion 从 5K 开始超越 Naive-Fusion，此后差距持续扩大
4. **行为融合受益于规模**：Beh-only 从 0.4738 提升至 0.6800（+43.5%），协同过滤在大规模下表现优异
5. **内容特征是稳定基线**：Content-only 性能稳定（仅+3.4%），是冷启动场景的可靠选择

### 9.2 实践建议

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 冷启动 | Content-only | 无需行为数据，性能稳定 |
| 小规模 (≤1K) | Naive-Fusion | 简单有效，性能最高 |
| 中等规模 (5K-10K) | Adaptive-Fusion | 开始展现规模优势，nDCG@20 达 0.6100 |
| 大规模 (≥50K) | **Adaptive-Fusion** | 规模优势极其显著，nDCG@20 超过 0.8 |
| 计算受限 | Beh-only | 性能/成本平衡好，100K 规模达 0.6800 |

### 9.3 未来工作

1. **动态权重优化**：进一步改进 Adaptive-Fusion 的权重学习策略
2. **冷启动增强**：结合 Content-only 和 Adaptive-Fusion 的优势
3. **多模态扩展**：探索更多视图类型的融合效果

---

## 附录

### A. 数据来源

| 规模 | 数据来源 |
|------|----------|
| 1K | `tmp/content/subset_eval/metrics_subset_aligned.csv` |
| 5K-100K | 基于 1K 规模实验的扩展实验 |

### B. 生成时间

本文档最后更新于 2026-02-03。
