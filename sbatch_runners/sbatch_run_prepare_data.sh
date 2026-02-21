#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --account=bfaq-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/prepare_data-%j.out
#SBATCH --error=logs/prepare_data-%j.err

# 设置环境变量 - 使用大空间目录
export UV_CACHE_DIR=/work/hdd/bfaq/jlyu3/lcm/uv_cache
export HF_HOME=/work/hdd/bfaq/jlyu3/lcm/hf_cache
export HF_DATASETS_CACHE=/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets
export TMPDIR=/work/hdd/bfaq/jlyu3/lcm/tmp

# Hugging Face Token (如果需要访问私有数据集)
# 如果环境变量中已有，则使用环境变量；否则使用默认值
# TODO: set your huggingface token

# 设置 Python 无缓冲输出
export PYTHONUNBUFFERED=1

# 切换到项目目录
# 在 SLURM 环境中，${BASH_SOURCE[0]} 可能指向临时脚本位置
# 优先使用 SLURM_SUBMIT_DIR（如果可用），否则使用硬编码路径
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # SLURM_SUBMIT_DIR 是提交作业时的目录
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -n "${BASH_SOURCE[0]}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    # 尝试从脚本位置计算（适用于非 SLURM 环境）
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    # 使用硬编码路径作为后备方案
    PROJECT_ROOT="/projects/bfaq/jlyu3/large_concept_model"
fi

cd "$PROJECT_ROOT" || {
    echo "❌ 错误：无法切换到项目目录: $PROJECT_ROOT"
    exit 1
}

# 设置库路径
export LD_LIBRARY_PATH=$PWD/.venv/lib:$LD_LIBRARY_PATH

# 如果存在 conda 环境，也添加到 LD_LIBRARY_PATH
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

echo "=========================================="
echo "🚀 数据准备任务 (SLURM)"
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU数量: 8"
echo "项目目录: $PROJECT_ROOT"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 运行数据准备脚本
# 使用绝对路径确保能找到脚本
PREPARE_DATA_SCRIPT="$PROJECT_ROOT/prepare_data.sh"
if [ ! -f "$PREPARE_DATA_SCRIPT" ]; then
    echo "❌ 错误：找不到脚本文件: $PREPARE_DATA_SCRIPT"
    echo "   当前目录: $(pwd)"
    echo "   PROJECT_ROOT: $PROJECT_ROOT"
    exit 1
fi

bash "$PREPARE_DATA_SCRIPT" \
    --num_gpus=8 \
    --num_samples=9670000 \
    --output_dir=/work/hdd/bfaq/jlyu3/lcm/preprocessed_data

echo ""
echo "=========================================="
echo "✅ 数据准备任务完成"
echo "时间: $(date)"
echo "=========================================="
