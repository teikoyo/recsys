# WS-SGNS 推荐系统

基于随机游走的 Skip-gram with Negative Sampling (SGNS) 双视图推荐系统。

## 项目结构

```
recsys-new/
├── src/                          # 核心代码模块
│   ├── __init__.py               # 模块导出
│   ├── ddp_utils.py              # DDP初始化/同步/日志
│   ├── csr_utils.py              # CSR矩阵I/O
│   ├── sampling_utils.py         # 负采样/别名采样
│   ├── pair_batch_utils.py       # 配对生成/批处理
│   ├── random_walk.py            # 随机游走生成器
│   ├── sgns_model.py             # SGNS模型定义
│   └── metrics.py                # nDCG/MAP/MRR等评测指标
│
├── scripts/                      # 可执行脚本
│   ├── train_sgns.py             # 统一的SGNS训练入口
│   ├── build_ann_index.py        # FAISS索引构建
│   ├── analyze_walks.py          # 随机游走分析
│   └── data_merge.ipynb          # 数据合并notebook
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_pipeline/              # 训练管道
│   │   ├── analyse_new.ipynb
│   │   └── analyse_high.ipynb
│   ├── 02_evaluation/            # 评估实验
│   │   ├── experiments.ipynb
│   │   ├── ablation_experiments.ipynb
│   │   └── compare_methods_framework.ipynb
│   └── 03_analysis/              # 数据分析
│       ├── tag_statistics_analysis.ipynb
│       ├── analyze_random_walks.ipynb
│       └── analyse_ddp_hybrid.ipynb
│
├── tests/                        # 测试代码
│   ├── test_evaluation_flow.py
│   ├── test_desc_fixes.py
│   ├── test_desc_loading.py
│   └── test_visualization_fix.py
│
├── docs/                         # 文档
│   ├── EXPERIMENT_PLAN.md
│   ├── EXPERIMENT_SUMMARY.md
│   ├── METRICS_EXPLANATION.md
│   ├── DDP_HYBRID_GUIDE.md
│   └── ...其他文档
│
├── data/                         # 原始数据
├── tmp/                          # 中间结果
│
├── step6_ddp.py                  # 原始DDP训练脚本(参考)
├── analyze_random_walks.py       # 随机游走分析脚本(参考)
└── ddp_scripts/                  # DDP脚本目录
    ├── README.md
    └── step7_faiss_ann_ddp.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch pandas numpy scipy faiss-gpu matplotlib seaborn
```

### 2. 训练嵌入

```bash
# 双视图训练 (Tag + Text)
python scripts/train_sgns.py --views tag,text --epochs 4

# 仅Text视图
python scripts/train_sgns.py --views text

# 仅Tag视图
python scripts/train_sgns.py --views tag

# DDP多卡训练
torchrun --nproc_per_node=2 scripts/train_sgns.py --views tag,text
```

### 3. 构建ANN索引

```bash
# 构建k-NN相似度图
python scripts/build_ann_index.py --k 50 --use_gpu true
```

### 4. 分析结果

```bash
# 生成分析报告和可视化
python scripts/analyze_walks.py --base_dir ./tmp --output_dir ./analysis_outputs
```

## 核心模块

### src/ddp_utils.py
DDP分布式训练工具函数：
- `init_ddp()`: 初始化DDP环境
- `barrier()`: 同步屏障
- `log0()`: 仅rank 0打印

### src/sampling_utils.py
负采样工具：
- `build_alias_on_device()`: GPU上构建别名表
- `sample_alias_gpu()`: O(1)别名采样

### src/random_walk.py
随机游走语料生成：
- `TorchWalkCorpus`: GPU加速的随机游走生成器
- `build_corpus()`: 构建Tag和Text视图语料

### src/sgns_model.py
SGNS模型：
- `SGNS`: Skip-gram with Negative Sampling模型

### src/metrics.py
评测指标：
- `ndcg_at_k()`: nDCG@K
- `average_precision_at_k()`: AP@K (用于MAP)
- `mrr_at_k()`: MRR@K
- `precision_at_k()`, `recall_at_k()`: P@K, R@K

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--views` | tag,text | 训练视图 |
| `--epochs` | 4 | 训练轮数 |
| `--dim` | 256 | 嵌入维度 |
| `--neg` | 10 | 负样本数 |
| `--lr` | 0.025 | 学习率 |
| `--window_tag` | 5 | Tag视图窗口 |
| `--window_text` | 4 | Text视图窗口 |
| `--batch_pairs_tag/text` | 204800 | 批量大小 |
| `--amp` | true | 混合精度训练 |

## 数据格式

### 输入文件 (tmp/)
- `doc_clean.parquet`: 文档表
- `tag_vocab.parquet`: Tag词表
- `text_vocab.parquet`: 词词表
- `DT_ppmi.parquet`: Doc-Tag PPMI矩阵
- `DW_bm25.parquet`: Doc-Word BM25矩阵
- `rw_params.parquet`: 随机游走参数

### 输出文件 (tmp/)
- `Z_tag.parquet`: Tag视图嵌入
- `Z_text.parquet`: Text视图嵌入
- `Z_*_epoch*.parquet`: 各epoch检查点

## 许可证

MIT License
