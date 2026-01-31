# Fusion-CombSUM 快速参考卡片

## Oracle Fusion 评分

| 指标 | Oracle值 | 排名 |
|------|----------|------|
| **nDCG@20** | **0.2490** | #12/17 |
| **MAP@20** | **0.3829** | #12/17 |
| **MRR@20** | **0.4108** | #12/17 |
| **P@20** | **0.2359** | - |
| **R@20** | **0.0439** | - |

## 三维度性能分解

```
Tag维度:     ████████████████████████ 0.2393  (主导)
Org维度:     ██████████████ 0.1425           (中等)
Creator维度: ███ 0.0325                      (极弱 ⚠️)
```

**关键问题**: Creator维度严重不足（仅0.0325），远低于Top方法的0.88-0.89

## vs 其他融合方法

| 方法 | Oracle-nDCG@20 | 差距 |
|------|----------------|------|
| Fused3-Blend-eta0.30 | 0.9119 | **-72.7%** |
| Fused3-RA | 0.8746 | -71.5% |
| Fused3-RR | 0.4247 | -70.5% |
| **Fusion-RRF** | **0.3799** | **-52.6%** ⚠️ |
| **Fusion-CombSUM** | **0.2490** | - |

**关键对比**: 即使同为简单融合策略，Fusion-RRF也比CombSUM强52.6%

## 改进潜力

```
Unified-nDCG@20:  0.1676  ─────────┐
                                   │ +48.6%提升
Oracle-nDCG@20:   0.2490  ◄────────┘

但Oracle绝对值仍然较低（#12/17）
```

## 为什么性能较低？

### ❌ 根本原因
1. **Creator维度极弱** (0.0325)
   - Top方法的Creator维度: 0.83-0.89
   - CombSUM的Creator维度: 0.0325
   - **差距**: 25-27倍

2. **简单加权策略限制**
   - CombSUM使用简单分数相加
   - 无法有效处理维度间差异
   - 弱维度拉低整体性能

3. **缺乏自适应机制**
   - 对所有query使用相同权重
   - 未能利用三维度的互补性

### ✅ 改进方向

1. **提升Creator维度**
   - 改进creator表示学习
   - 使用更强的嵌入方法

2. **采用高级融合策略**
   - 学习Blend融合的自适应权重
   - 实现per-query动态选择

3. **利用Oracle洞察**
   - 48.6%的提升空间说明有改进潜力
   - 可以向Oracle策略学习（per-query选择最佳维度）

## 论文中如何呈现

### 建议描述

> "Fusion-CombSUM achieves an Oracle-nDCG@20 of 0.249, ranking #12 among 17
> methods. While showing 48.6% improvement potential over its Unified score
> (0.168), the absolute performance remains limited primarily due to weak
> Creator dimension performance (0.032 vs. 0.89 in top methods). This
> highlights the importance of balanced multi-view representations rather
> than fusion strategy alone."

### 关键要点

- ✓ CombSUM是简单融合基准
- ✓ Creator维度弱是主要瓶颈
- ✓ 48.6%改进空间表明有提升潜力
- ✓ 但绝对性能受限于维度质量

## 完整数据位置

- **CSV文件**: `/workspace/recsys/tmp/all_methods_oracle_fusion.csv`
- **完整报告**: `/workspace/recsys/ALL_METHODS_ORACLE_REPORT_COMPLETE.md`
- **LaTeX表格**: 见上述报告中的LaTeX部分

---

**生成时间**: 2025-11-20
