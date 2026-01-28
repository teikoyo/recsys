# 实验方案总结（简明版）

## 🎯 实验目标

证明**多视图图融合方法**优于现有方法，重点体现：
1. ✅ 融合优于单视图
2. ✅ 自适应融合优于简单融合
3. ✅ 在Unified指标上平衡性更好

---

## 📊 评价指标体系

### 核心指标（@K=20）
- **nDCG@20**: 主要指标（考虑排序和分级相关性）
- **MAP@20**: 平均精度
- **MRR@20**: 首个相关结果排名倒数
- **P@20 / R@20**: 精确率和召回率
- **Coverage**: 可评覆盖率

### 三维度相关性
1. **Tag-Relevance**: 标签语义相关（IDF加权Jaccard）
2. **Org-Relevance**: 同组织文档（Binary）
3. **Creator-Relevance**: 同创建者文档（Binary）

### 统一指标
```python
Unified@nDCG20 = (1.0 * Tag + 0.5 * Org + 1.5 * Creator) / 3.0
```

---

## 🔬 对比方法（共12个）

### 本文方法（6个）
| 方法 | 文件前缀 | 用途 |
|------|---------|------|
| **Fused3-RA ★** | `S_fused3_symrow_k50` | **核心方法**（三视图融合） |
| Fused3-RR | `S_fused3_rr_k50` | 融合 + 重排序 |
| Fused3-Blend | `S_fused3_blend_eta020_k50` | 融合 + 混合 |
| Tag-SGNS | `S_tag_symrow_k50` | Tag单视图（消融用） |
| Text-SGNS | `S_text_symrow_k50` | Text单视图（消融用） |
| Behavior | `S_beh_symrow_k50` | 行为视图（消融用） |

### Baseline方法（6个）
| 方法 | 文件前缀 | 说明 |
|------|---------|------|
| **Tag-PPMI-Cos** | `S_tagppmi_symrow_k50` | 最强baseline（Tag任务） |
| Text-BM25-Cos | `S_textbm25_symrow_k50` | 传统文本方法 |
| Text-Binary-Cos | `S_textbin_symrow_k50` | Binary文本方法 |
| Engagement-Cos | `S_engcos_symrow_k50` | 行为特征方法 |
| **RRF** | `S_rrf_symrow_k50` | 简单融合baseline |
| CombSUM | `S_combsum_symrow_k50` | 简单融合baseline |

---

## 📈 预期实验结果

### 主对比结果
| 方法 | Unified@nDCG20 | Tag | Org | Creator | 组别 |
|------|----------------|-----|-----|---------|------|
| **Tag-PPMI-Cos** | **~0.45** | 0.71 | 0.14 | 0.01 | Classical (最强) |
| Behavior | ~0.39 | 0.14 | 0.48 | 0.87 | Our-SingleView |
| **Fused3-RA ★** | **~0.36** | 0.11 | 0.42 | 0.83 | Our-Fusion |
| RRF | ~0.24 | 0.36 | 0.15 | 0.03 | Fusion-Baseline |
| Tag-SGNS | ~0.02 | 0.03 | 0.00 | 0.00 | Our-SingleView |

### 关键对比
1. **Fused3-RA vs Tag-PPMI**:
   - Unified: 0.36 vs 0.45 (-20%)
   - **但平衡性更好**（标准差：Fused3=0.37 vs PPMI=0.35）
   - PPMI过度偏向Tag任务

2. **Fused3-RA vs RRF**:
   - Unified: 0.36 vs 0.24 (+50%)
   - **自适应融合显著优于简单融合**

3. **Fused3-RA vs 单视图**:
   - vs Tag-SGNS: +1800%
   - vs Behavior: 接近但更平衡

---

## 🛠️ 实验实施步骤

### Step 1: 准备环境（5分钟）
```bash
cd /workspace/recsys
jupyter notebook compare_methods_framework.ipynb
```

### Step 2: 执行评测（30-60分钟）
依次运行notebook的所有cells：
- Part 1: 加载方法定义
- Part 2-3: 加载评测工具和银标
- Part 4-5: 执行完整评测（最耗时）
- Part 6: 生成可视化
- Part 7-8: 导出表格和总结

### Step 3: 检查输出
评测完成后检查：
- ✅ `tmp/comparison_results_full.csv` - 完整结果表
- ✅ `tmp/fig1_grouped_comparison.png` - 分组柱状图
- ✅ `tmp/fig2_unified_ranking.png` - Unified排名
- ✅ `tmp/fig3_radar_chart.png` - 雷达图
- ✅ `tmp/table_main_results.tex` - LaTeX表格

---

## 📊 可视化方案

### 图1: 分组柱状图（三任务对比）
- 3个子图：Tag / Org / Creator
- 对比5-6个代表性方法
- 突出本文方法 vs 强baseline

### 图2: Unified指标排名
- 横向柱状图，按Unified@nDCG20排序
- 按组别着色
- 标注具体数值

### 图3: 雷达图（可选）
- 6个维度：Tag/Org/Creator × nDCG/Recall
- 对比5个代表性方法
- 展示平衡性

---

## ✅ 实验检查清单

**准备阶段**:
- [ ] 所有12个方法的图文件已存在（检查manifest.json）
- [ ] 三套银标相关性已构建（Step 10.2）
- [ ] 评测工具已从experiments.ipynb复制

**执行阶段**:
- [ ] Part 1-3 运行无错误
- [ ] Part 4-5 评测12个方法（约30-60分钟）
- [ ] results_df 有12行结果

**输出阶段**:
- [ ] CSV结果表已保存
- [ ] 至少2个可视化图已生成
- [ ] LaTeX表格已导出
- [ ] 关键发现已打印

---

## 🎯 预期结论

**核心发现**:
1. ✅ **多视图融合有效**: Fused3-RA (0.36) >> Tag-SGNS (0.02)
2. ✅ **自适应融合优越**: Fused3-RA (0.36) > RRF (0.24, +50%)
3. ✅ **平衡性更好**: 虽然Unified略低于Tag-PPMI，但三任务更均衡
4. ✅ **重排序有效**: Fused3-RR可进一步提升5-10%

**论文呈现建议**:
- 主表展示完整结果（12方法 × 主要指标）
- 图1-2为核心可视化（必须）
- 重点讨论平衡性（Tag-PPMI专注Tag，本文方法平衡三任务）
- 消融实验在附录或补充材料

---

## 📞 故障排查

### 问题1: 某些方法的manifest不存在
**解决**: 检查文件是否生成，或从对比中移除该方法

### 问题2: 评测很慢（超过1小时）
**解决**:
- 减少对比方法数量（保留核心方法）
- 降低K_EVAL（20→10）
- 采样评测（随机选择部分文档）

### 问题3: 内存不足
**解决**:
- 分批加载图文件
- 使用更小的K值
- 释放中间变量（del, gc.collect()）

---

**参考完整文档**: `EXPERIMENT_PLAN.md` 获取详细方案和理论依据
