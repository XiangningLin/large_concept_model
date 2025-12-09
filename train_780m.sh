#!/bin/bash
# 训练 780M 模型（使用预处理好的数据）
set -e

# ========================================================
# 配置
# ========================================================
# 自动检测项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 使用大空间目录存储数据和检查点
export DATA_DIR="/work/hdd/bfaq/jlyu3/lcm/processed_data/tom_tracking_data"
export CHECKPOINT_DIR="/work/hdd/bfaq/jlyu3/lcm/checkpoints/tom_tracking_780M_2gpu"
export EXPERIMENT_NAME="tom_tracking_780M_2gpu"

# 设置环境变量 - 使用大空间目录
export UV_CACHE_DIR=/work/hdd/bfaq/jlyu3/lcm/uv_cache
export HF_HOME=/work/hdd/bfaq/jlyu3/lcm/hf_cache
export HF_DATASETS_CACHE=/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets
export TMPDIR=/work/hdd/bfaq/jlyu3/lcm/tmp
# ========================================================

cd "$PROJECT_ROOT"

# 设置库路径
export LD_LIBRARY_PATH=$PWD/.venv/lib:$LD_LIBRARY_PATH
export PYTHONBREAKPOINT=0

echo "=========================================="
echo "检查数据..."
echo "=========================================="

if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/validation" ]; then
    echo "✓ 数据目录: $DATA_DIR"
    echo "  - train: $(ls $DATA_DIR/train/*.parquet 2>/dev/null | wc -l) parquet files"
    echo "  - validation: $(ls $DATA_DIR/validation/*.parquet 2>/dev/null | wc -l) parquet files"
else
    echo "❌ 数据目录不存在！"
    exit 1
fi

echo ""
echo "=========================================="
echo "训练 780M 模型 (2 GPU)"
echo "=========================================="

mkdir -p logs checkpoints

echo "使用 2 GPU 训练..."
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    -m lcm.train launcher=standalone \
    +post_training=tom_tracking_4GPU \
    ++trainer.model_arch=base_lcm_780M \
    ++trainer.output_dir="$CHECKPOINT_DIR" \
    ++trainer.experiment_name="$EXPERIMENT_NAME" \
    +trainer.use_submitit=false \
    ++trainer.max_steps=500 \
    ++trainer.validate_every_n_steps=100 \
    ++trainer.checkpoint_every_n_steps=100 \
    ++trainer.data_loading_config.max_tokens=1500

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
echo "检查点目录: $CHECKPOINT_DIR"





