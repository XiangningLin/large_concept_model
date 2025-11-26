#!/bin/bash
# 准备C4数据 - 支持多GPU并行
# 使用方法: bash prepare_data.sh --num_gpus=8 --num_samples=2000000

set -e
cd /work/nvme/bfaq/xlin5/large_concept_model

# 默认参数
NUM_GPUS=2
NUM_SAMPLES=2000000  # 约10B tokens
OUTPUT_DIR="output/c4_10b"
BATCH_SIZE=20

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus=*)
            NUM_GPUS="${1#*=}"
            shift
            ;;
        --num_samples=*)
            NUM_SAMPLES="${1#*=}"
            shift
            ;;
        --output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --num_gpus=N [--num_samples=M] [--output_dir=PATH] [--batch_size=B]"
            exit 1
            ;;
    esac
done

# 计算每个GPU处理的样本数
SAMPLES_PER_GPU=$((NUM_SAMPLES / NUM_GPUS))

echo "======================================"
echo "🚀 C4数据准备（多GPU并行）"
echo "======================================"
echo "GPU数量: $NUM_GPUS"
echo "总样本数: $NUM_SAMPLES"
echo "每GPU样本: $SAMPLES_PER_GPU"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "======================================"
echo ""

# 设置环境变量
export UV_CACHE_DIR=/work/nvme/bfaq/xlin5/uv_cache
export HF_HOME=/work/nvme/bfaq/xlin5/hf_cache
export HF_DATASETS_CACHE=/work/nvme/bfaq/xlin5/hf_cache/datasets
export TMPDIR=/work/nvme/bfaq/xlin5/tmp
mkdir -p $TMPDIR logs $OUTPUT_DIR

# 启动所有GPU任务
echo "🚀 启动 $NUM_GPUS 个GPU任务..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    START_IDX=$((i * SAMPLES_PER_GPU))
    
    echo "  GPU $i: 样本 $START_IDX - $((START_IDX + SAMPLES_PER_GPU))"
    
    CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python scripts/prepare_c4.py \
        --output_dir=$OUTPUT_DIR/shard_$i \
        --num_samples=$SAMPLES_PER_GPU \
        --start_index=$START_IDX \
        --batch_size=$BATCH_SIZE \
        > logs/prepare_gpu${i}.log 2>&1 &
    
    echo "  PID: $!"
done

echo ""
echo "✅ 所有任务已启动！"
echo ""
echo "📊 监控命令:"
echo "  查看日志: tail -f logs/prepare_gpu*.log"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo "  查看进度: du -sh $OUTPUT_DIR/shard_*/"
echo ""
echo "⏱️  预计时间（$NUM_GPUS个GPU）:"
HOURS=$((SAMPLES_PER_GPU / 10000))
echo "  约 $HOURS 小时"
echo ""
