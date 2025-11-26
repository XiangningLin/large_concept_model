#!/bin/bash
# 验证 jianwen-dev 分支的完整流程 (2 GPU 版本 - 修复版)
set -e

# ========================================================
# 配置 & 环境变量
# ========================================================
export PROJECT_ROOT="/u/xlin5/projects/large_concept_model"
export DATA_OUTPUT_DIR="processed_data/tom_tracking_test"
export CACHE_DIR="./preprocessed_data"
export CHECKPOINT_DIR="checkpoints/tom_tracking_test_2gpu"
export EXPERIMENT_NAME="tom_tracking_test_2gpu"
# ========================================================

cd "$PROJECT_ROOT"

# 设置 MKL 库路径
export LD_LIBRARY_PATH=$PWD/.venv/lib:$LD_LIBRARY_PATH

# 禁用 Python 调试断点
export PYTHONBREAKPOINT=0

echo "=========================================="
echo "Step 1: 准备 Tom Tracking 数据 (Local 模式)"
echo "=========================================="

# 准备数据（使用 local 模式，不提交到 SLURM）
uv run python scripts/prepare_tom_tracking.py \
    --output_dir="$DATA_OUTPUT_DIR" \
    --split=tom_tracking_0.5k \
    --num_shards=1 \
    --batch_size=10 \
    --cache_dir="$CACHE_DIR" \
    --use_natural_language=True \
    --add_split_column=True \
    --train_ratio=0.8

echo ""
echo "✓ 数据准备完成！"
echo ""

# 检查数据是否生成
if [ -d "$DATA_OUTPUT_DIR" ]; then
    echo "✓ 数据目录存在: $DATA_OUTPUT_DIR"
    echo "文件列表:"
    ls -lh "$DATA_OUTPUT_DIR/" | head -10
else
    echo "❌ 数据目录不存在！"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: 训练模型 (2 GPU Standalone 模式)"
echo "=========================================="

# 创建日志目录
mkdir -p logs checkpoints

echo "使用 2 GPU 训练..."
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    -m lcm.train launcher=standalone \
    +post_training=tom_tracking_4GPU \
    ++trainer.output_dir="$CHECKPOINT_DIR" \
    ++trainer.experiment_name="$EXPERIMENT_NAME" \
    +trainer.use_submitit=false \
    ++trainer.max_steps=500 \
    ++trainer.validate_every_n_steps=100 \
    ++trainer.checkpoint_every_n_steps=100 \
    ++trainer.data_loading_config.max_tokens=1500

echo ""
echo "=========================================="
echo "✓ 验证完成！"
echo "=========================================="
echo "数据目录: $DATA_OUTPUT_DIR"
echo "检查点目录: $CHECKPOINT_DIR"
echo ""
