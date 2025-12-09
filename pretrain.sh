#!/bin/bash
# LCM 780M 预训练 - 支持可配置GPU数量
# 使用方法: bash pretrain.sh --num_gpus=2

set -e
# 自动检测项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
NUM_GPUS=2
# DATA_DIR="output/c4_10b"
DATA_DIR="output/fine_web"
OUTPUT_DIR="checkpoints/mse_lcm_780m"
EXPERIMENT_NAME="mse_lcm_780m_10b"
MAX_TOKENS=6000
MAX_STEPS=100000
CHECKPOINT_EVERY=5000
# 数据集名称（datacard 名称，会自动填充 training 和 validation）
DATA_NAME="pretraining_data"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus=*)
            NUM_GPUS="${1#*=}"
            shift
            ;;
        --data_dir=*)
            DATA_DIR="${1#*=}"
            shift
            ;;
        --output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --experiment_name=*)
            EXPERIMENT_NAME="${1#*=}"
            shift
            ;;
        --max_tokens=*)
            MAX_TOKENS="${1#*=}"
            shift
            ;;
        --max_steps=*)
            MAX_STEPS="${1#*=}"
            shift
            ;;
        --data_name=*)
            DATA_NAME="${1#*=}"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --num_gpus=N [--data_dir=PATH] [--output_dir=PATH] [--max_tokens=N] [--max_steps=N]"
            echo "     [--data_name=NAME]  # 例如: --data_name=fine_web_edu (会自动使用 fine_web_edu=train 和 fine_web_edu=validation)"
            exit 1
            ;;
    esac
done

# 自动填充训练和验证数据集名称
TRAINING_DATA_NAME="${DATA_NAME}=train"
VALIDATION_DATA_NAME="${DATA_NAME}=validation"

# 构建GPU列表
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

echo "======================================"
echo "🚀 LCM 780M预训练"
echo "======================================"
echo "GPU数量: $NUM_GPUS"
echo "GPU列表: $GPU_LIST"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "实验名称: $EXPERIMENT_NAME"
echo "Max tokens: $MAX_TOKENS"
echo "Max steps: $MAX_STEPS"
echo "数据集名称: $DATA_NAME"
echo "  - 训练: $TRAINING_DATA_NAME"
echo "  - 验证: $VALIDATION_DATA_NAME"
echo "======================================"
echo ""

# 检查数据
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误: 数据目录不存在: $DATA_DIR"
    echo "请先运行: bash prepare_data.sh --num_gpus=$NUM_GPUS"
    exit 1
fi

echo "✓ 数据目录存在"
echo ""

# 设置环境变量 - 使用大空间目录
export UV_CACHE_DIR=/work/hdd/bfaq/jlyu3/lcm/uv_cache
export TMPDIR=/work/hdd/bfaq/jlyu3/lcm/tmp

mkdir -p $TMPDIR $OUTPUT_DIR

# 启动训练
echo "🚀 启动训练..."
echo ""
# TODO: added
CUDA_VISIBLE_DEVICES=$GPU_LIST .venv/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    -m lcm.train \
    launcher=standalone \
    +pretrain=mse_780M \
    ++trainer.output_dir=$OUTPUT_DIR \
    ++trainer.experiment_name=$EXPERIMENT_NAME \
    ++trainer.data_loading_config.max_tokens=$MAX_TOKENS \
    ++trainer.use_fsdp=true \
    ++trainer.max_steps=$MAX_STEPS \
    ++trainer.checkpoint_every_n_steps=$CHECKPOINT_EVERY \
    ++trainer.save_model_every_n_steps=$CHECKPOINT_EVERY \
    ++trainer.publish_metrics_every_n_steps=100 \
    '++trainer.training_data.0.name="'${TRAINING_DATA_NAME}'"' \
    '++trainer.validation_data.0.name="'${VALIDATION_DATA_NAME}'"'

# TODO: pdb是submitit在用--debug模式运行指令时自带的。

echo ""
echo "✅ 训练已启动！"
echo ""
echo "📊 监控命令:"
echo "  查看日志: tail -f $OUTPUT_DIR/logs/*.log"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo "  查看checkpoint: ls -lh $OUTPUT_DIR/checkpoints/"
echo ""
