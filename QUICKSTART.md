# LCM 预训练快速开始

## 📁 核心文件

```
large_concept_model/
├── scripts/prepare_c4.py          # 数据准备核心脚本
├── INSTALL_NEW_MACHINE.sh         # 新机器环境安装
├── prepare_data.sh                # 数据准备（多GPU并行）
├── pretrain.sh                    # 模型预训练（多GPU）
├── test_super_simple.sh           # 环境测试
└── test_pretrain_100.sh           # 训练测试
```

## 🚀 完整工作流程

### 1. 新机器上安装环境

```bash
# 克隆项目（如果还没有）
git clone https://github.com/facebookresearch/large_concept_model.git
cd large_concept_model

# 运行安装脚本
bash INSTALL_NEW_MACHINE.sh

# 复制自定义脚本（从你的Delta机器）
# - scripts/prepare_c4.py
# - lcm/models/base_lcm/archs.py
# - lcm/datacards/datacards.yaml
```

### 2. 测试环境

```bash
# 快速测试（10条数据，3分钟）
bash test_super_simple.sh
```

### 3. 准备训练数据

```bash
# 准备10B tokens数据（2百万样本）
# 2张GPU: 约1-2天
# 8张GPU: 约6-12小时

bash prepare_data.sh --num_gpus=2 --num_samples=2000000

# 或者准备更多数据
bash prepare_data.sh --num_gpus=8 --num_samples=20000000  # 100B tokens

# 监控进度
tail -f logs/prepare_gpu*.log
```

### 4. 开始预训练

```bash
# 2张GPU
bash pretrain.sh --num_gpus=2

# 8张GPU
bash pretrain.sh --num_gpus=8 --max_tokens=8000

# 自定义参数
bash pretrain.sh \
    --num_gpus=4 \
    --data_dir=output/c4_10b \
    --output_dir=checkpoints/my_experiment \
    --max_tokens=7000 \
    --max_steps=200000
```

## 📊 数据规模参考

| 样本数 | Tokens | GPU数量 | 准备时间 | 建议 |
|--------|--------|---------|----------|------|
| 10 | ~50K | 1 | 3分钟 | 测试 |
| 100 | ~500K | 1 | 10分钟 | 测试 |
| 10K | ~50M | 1 | 4小时 | 小实验 |
| 100K | ~500M | 2 | 1天 | 验证 |
| 2M | ~10B | 2 | 1-2天 | 标准训练 |
| 2M | ~10B | 8 | 6-12小时 | 快速训练 |
| 20M | ~100B | 8 | 2-4天 | 大规模训练 |

## 🔧 常用命令

### 监控

```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看数据准备进度
tail -f logs/prepare_gpu*.log

# 查看训练日志
tail -f checkpoints/mse_lcm_780m/logs/*.log

# 查看已生成的数据
du -sh output/c4_10b/shard_*/
```

### 管理

```bash
# 杀死数据准备任务
pkill -f prepare_c4.py

# 检查进程
ps aux | grep prepare_c4

# 清理旧数据
rm -rf output/c4_test_*
```

## ⚠️ 注意事项

1. **GPU节点**: 数据准备和训练都需要在GPU节点上运行
2. **磁盘空间**: 10B tokens约需要20-30GB存储
3. **内存**: 建议每张GPU配80GB以上内存
4. **时间**: 预留足够的SLURM时间（数据准备1-2天，训练数天到数周）

## 🐛 问题排查

如果遇到问题：

```bash
# 运行诊断脚本
bash DEBUG_AND_FIX.sh

# 验证环境
.venv/bin/python -c "import torch, fairseq2; print('OK')"

# 重装环境
rm -rf .venv
bash INSTALL_NEW_MACHINE.sh
```

## 📞 需要帮助？

1. 查看错误日志: `tail -100 logs/*.log`
2. 检查GPU状态: `nvidia-smi`
3. 验证数据: `ls -lh output/c4_10b/`
