#!/bin/bash
set -e

# ========================================================
# 准备数据并训练模型
# ========================================================
# 注意：使用 /work/nvme/bfaq/xlin5/ 路径，该存储空间较大
export PROJECT_ROOT="/work/nvme/bfaq/xlin5/large_concept_model"
export DATA_OUTPUT_DIR="/work/nvme/bfaq/xlin5/large_concept_model/processed_data/tom_tracking_test"
export CACHE_DIR="/work/nvme/bfaq/xlin5/large_concept_model/preprocessed_data"
export CHECKPOINT_DIR="/work/nvme/bfaq/xlin5/large_concept_model/checkpoints/tom_tracking_780M_2gpu"
export EXPERIMENT_NAME="tom_tracking_780M_2gpu"
# ========================================================

cd "$PROJECT_ROOT"

# 检查虚拟环境是否存在
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ 错误：虚拟环境不存在！"
    echo "请先运行: bash reinstall_venv.sh"
    exit 1
fi

# 设置库路径（包括 venv 和 conda 环境的库）
export LD_LIBRARY_PATH=$PWD/.venv/lib:$LD_LIBRARY_PATH

# 如果存在 conda 环境，也添加到 LD_LIBRARY_PATH
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
elif [ -f "$HOME/miniconda3/envs/lcm-helper/lib/libsndfile.so" ] || [ -f "$HOME/miniconda/envs/lcm-helper/lib/libsndfile.so" ]; then
    # 尝试常见 conda 路径
    for conda_lib in "$HOME/miniconda3/envs/lcm-helper/lib" "$HOME/miniconda/envs/lcm-helper/lib"; do
        if [ -d "$conda_lib" ]; then
            export LD_LIBRARY_PATH="$conda_lib:$LD_LIBRARY_PATH"
            break
        fi
    done
fi

# 禁用 Python 调试断点
export PYTHONBREAKPOINT=0

echo "=========================================="
echo "Step 1: 准备 Tom Tracking 数据 (Local 模式)"
echo "=========================================="

# 准备数据（使用 local 模式，不提交到 SLURM）
# 直接使用 .venv/bin/python 避免 uv 的环境变量警告
# 在运行 Python 时取消 VIRTUAL_ENV，避免某些库检测到错误的虚拟环境路径
env -u VIRTUAL_ENV .venv/bin/python scripts/prepare_tom_tracking.py \
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

echo "使用 2 GPU 训练 780M 模型..."
# 取消 VIRTUAL_ENV 避免警告
env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
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
echo "✓ 数据准备和训练完成！"
echo "=========================================="
echo "数据目录: $DATA_OUTPUT_DIR"
echo "检查点目录: $CHECKPOINT_DIR"
echo ""
