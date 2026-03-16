# DDP Scripts for Multi-View Recommendation System

本目录包含用于分布式训练的DDP脚本。这些脚本可以通过`torchrun`启动，实现多GPU并行训练。

## 📁 脚本列表

### step6_sgns_training_ddp.py
**用途**: Skip-gram with Negative Sampling (SGNS) 嵌入训练
**最耗时**: ⭐⭐⭐⭐⭐ (整个pipeline中最耗时的步骤)
**输入**:
- `tmp/doc_clean.parquet` - 清洗后的文档
- `tmp/tag_vocab.parquet` - 标签词汇表
- `tmp/text_vocab.parquet` - 文本词汇表
- `tmp/DT_ppmi.parquet` - Doc-Tag PPMI矩阵
- `tmp/DW_bm25.parquet` - Doc-Word BM25矩阵
- `tmp/rw_params.parquet` - 随机游走参数

**输出**:
- `tmp/Z_tag.parquet` - Tag视图嵌入 (N × 256)
- `tmp/Z_text.parquet` - Text视图嵌入 (N × 256)
- `tmp/Z_tag_epoch{1-4}.parquet` - 每个epoch的checkpoint
- `tmp/Z_text_epoch{1-4}.parquet` - 每个epoch的checkpoint

**启动示例**:
```bash
# 使用4个GPU
torchrun --nproc_per_node=4 ddp_scripts/step6_sgns_training_ddp.py \
  --tmp_dir ./tmp \
  --epochs 4 \
  --dim 256 \
  --neg 10 \
  --lr 0.025 \
  --amp true \
  --tf32 true \
  --optimizer sgd \
  --window_tag 5 \
  --keep_prob_tag 1.0 \
  --batch_pairs_tag 204800 \
  --window_text 4 \
  --keep_prob_text 0.35 \
  --batch_pairs_text 204800 \
  --accum 1

# 查看所有参数
python ddp_scripts/step6_sgns_training_ddp.py --help
```

**关键参数**:
- `--epochs`: 训练轮数 (默认4)
- `--dim`: 嵌入维度 (默认256)
- `--neg`: 负样本数量 (默认10)
- `--amp`: 启用自动混合精度 (推荐true)
- `--batch_pairs_tag/text`: 每步的训练对数 (默认204800)
- `--accum`: 梯度累积步数 (用于模拟更大batch)

---

### step7_faiss_ann_ddp.py
**用途**: 使用FAISS构建k-NN相似度图
**输入**:
- `tmp/Z_tag.parquet` - Tag嵌入
- `tmp/Z_text.parquet` - Text嵌入

**输出**:
- `tmp/S_tag_topk_k50_part*.parquet` - Tag k-NN图 (分区存储)
- `tmp/S_tag_topk_k50_manifest.json` - Tag图元数据
- `tmp/S_text_topk_k50_part*.parquet` - Text k-NN图 (分区存储)
- `tmp/S_text_topk_k50_manifest.json` - Text图元数据

**启动示例**:
```bash
# 使用4个GPU
torchrun --nproc_per_node=4 ddp_scripts/step7_faiss_ann_ddp.py \
  --tmp_dir ./tmp \
  --k 50 \
  --batch_q 8192 \
  --use_gpu true \
  --index_type flat_ip

# 查看所有参数
python ddp_scripts/step7_faiss_ann_ddp.py --help
```

**关键参数**:
- `--k`: 最近邻数量 (默认50)
- `--batch_q`: 查询batch size (默认8192)
- `--index_type`: FAISS索引类型 (`flat_ip` 或 `ivf_ip`)
- `--use_gpu`: 是否使用GPU (推荐true)

---

## 🚀 使用方式

### 方式1: 通过Notebook调用 (推荐)

打开 `analyse_ddp_hybrid.ipynb` 并按顺序运行cells。Notebook会在适当的步骤自动调用这些DDP脚本。

### 方式2: 独立运行脚本

```bash
# 1. 确保前置步骤已完成（Step 2-5）
# 2. 运行SGNS训练
torchrun --nproc_per_node=4 ddp_scripts/step6_sgns_training_ddp.py --tmp_dir ./tmp

# 3. 运行FAISS ANN
torchrun --nproc_per_node=4 ddp_scripts/step7_faiss_ann_ddp.py --tmp_dir ./tmp
```

### 方式3: 单GPU运行（调试用）

```bash
# 不使用torchrun，直接运行
python ddp_scripts/step6_sgns_training_ddp.py --tmp_dir ./tmp --epochs 1

# 脚本会自动检测到非DDP环境，使用单GPU模式
```

---

## ⚙️ 性能优化建议

### GPU数量
- **1-2 GPU**: 基础配置，训练时间较长（~4-6小时）
- **4 GPU**: 推荐配置，训练时间适中（~1.5-2小时）
- **8 GPU**: 最佳配置，训练时间最短（~45-60分钟）

### 显存优化
如果遇到OOM (Out of Memory)错误：
1. 减小 `--batch_pairs_tag` 和 `--batch_pairs_text`
2. 增加 `--accum` (梯度累积)
3. 设置 `--amp false` (关闭混合精度，但会更慢)

### 训练速度优化
1. 确保 `--amp true` 和 `--tf32 true` 都启用
2. 使用 `--optimizer sgd` (比sparse_adam快)
3. 调整 `--batch_pairs_*` 到显存允许的最大值

---

## 📊 输出文件说明

### 嵌入文件格式
```
Z_tag.parquet / Z_text.parquet:
  列: doc_idx, f0, f1, ..., f255
  行: N个文档
  值: L2归一化的嵌入向量
```

### 图文件格式
```
S_*_topk_k50_part0000.parquet:
  列: row, col, val
  行: 边的三元组 (source, target, similarity)

S_*_topk_k50_manifest.json:
  {
    "N": 文档总数,
    "k": 邻居数量,
    "total_edges": 总边数,
    "num_parts": 分区数,
    "part_files": ["part0000.parquet", ...]
  }
```

---

## 🐛 故障排查

### 错误: "NCCL error"
**原因**: GPU之间通信失败
**解决**:
```bash
export NCCL_DEBUG=INFO  # 查看详细错误信息
export NCCL_P2P_DISABLE=1  # 禁用P2P通信（降低性能但更稳定）
```

### 错误: "RuntimeError: cuDNN error"
**原因**: cuDNN版本不兼容
**解决**:
```bash
# 禁用benchmark
# 在脚本中添加: torch.backends.cudnn.benchmark = False
```

### 错误: "Address already in use"
**原因**: 上次DDP进程未正常退出
**解决**:
```bash
pkill -9 python  # 杀死所有Python进程
# 或指定端口
export MASTER_PORT=29501  # 使用不同端口
```

### 训练速度慢
1. 检查GPU利用率: `nvidia-smi -l 1`
2. 检查是否使用了正确的GPU: 查看日志中的device信息
3. 确保数据在SSD上，不在HDD

---

## 📝 开发说明

### 添加新的DDP脚本
1. 复制现有脚本作为模板
2. 实现 `init_ddp()`, `barrier()`, `log0()` 辅助函数
3. 在主逻辑中添加rank判断:
   ```python
   if (not is_ddp) or rank == 0:
       # 只在rank 0执行的代码（如保存文件）
       save_results(...)

   barrier(is_ddp)  # 同步所有进程
   ```

### 测试脚本
```bash
# 测试单GPU模式
python ddp_scripts/step6_sgns_training_ddp.py --tmp_dir ./tmp --epochs 1

# 测试多GPU模式
torchrun --nproc_per_node=2 ddp_scripts/step6_sgns_training_ddp.py \
  --tmp_dir ./tmp --epochs 1
```

---

## 📚 参考资料

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)

---

**Version**: 1.0
**Last Updated**: 2025-11-17
