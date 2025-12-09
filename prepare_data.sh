#!/bin/bash
# 准备C4数据 - 支持多GPU并行
# 使用方法: bash prepare_data.sh --num_gpus=8 --num_samples=2000000

# TODO: add HF_TOKEN

set -e
# cd /work/nvme/bfaq/xlin5/large_concept_model

# 默认参数
NUM_GPUS=2
NUM_SAMPLES=2000  # 约10B tokens，设置为 "all" 或 "None" 表示处理整个数据集
OUTPUT_DIR="output/fine_web"
BATCH_SIZE=20

# prepare_fine_web 参数的默认值
MAX_SENTENCE_LENGTH=256
ADD_SPLIT_COLUMN=True
TRAIN_RATIO=0.8
SEED=42

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus=*)
            NUM_GPUS="${1#*=}"
            shift
            ;;
        --num_samples=*)
            NUM_SAMPLES="${1#*=}"
            # 支持 "all" 或 "None" 表示处理整个数据集
            if [ "$NUM_SAMPLES" = "all" ] || [ "$NUM_SAMPLES" = "None" ] || [ "$NUM_SAMPLES" = "none" ]; then
                NUM_SAMPLES="all"
            fi
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
        --max_sentence_length=*)
            MAX_SENTENCE_LENGTH="${1#*=}"
            shift
            ;;
        --add_split_column=*)
            ADD_SPLIT_COLUMN="${1#*=}"
            shift
            ;;
        --train_ratio=*)
            TRAIN_RATIO="${1#*=}"
            shift
            ;;
        --seed=*)
            SEED="${1#*=}"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --num_gpus=N [--num_samples=M] [--output_dir=PATH] [--batch_size=B]"
            echo "     [--max_sentence_length=LEN] [--add_split_column=BOOL] [--train_ratio=RATIO] [--seed=SEED]"
            exit 1
            ;;
    esac
done

# 计算每个GPU处理的样本数
if [ "$NUM_SAMPLES" = "all" ]; then
    # 处理整个数据集时，每个GPU都处理全部数据（通过 start_index 区分）
    SAMPLES_PER_GPU="None"
    echo "======================================"
    echo "🚀 C4数据准备（多GPU并行）"
    echo "======================================"
    echo "GPU数量: $NUM_GPUS"
    echo "总样本数: 全部（处理整个数据集）"
    echo "⚠️  注意：多GPU并行处理整个数据集时，每个GPU会处理不同的数据段（通过 start_index 区分）"
else
    SAMPLES_PER_GPU=$((NUM_SAMPLES / NUM_GPUS))
    echo "======================================"
    echo "🚀 C4数据准备（多GPU并行）"
    echo "======================================"
    echo "GPU数量: $NUM_GPUS"
    echo "总样本数: $NUM_SAMPLES"
    echo "每GPU样本: $SAMPLES_PER_GPU"
fi
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "最大句子长度: $MAX_SENTENCE_LENGTH"
echo "添加split列: $ADD_SPLIT_COLUMN"
echo "训练集比例: $TRAIN_RATIO"
echo "随机种子: $SEED"
echo "======================================"
echo ""

# 设置环境变量 - 使用大空间目录
export UV_CACHE_DIR=/work/hdd/bfaq/jlyu3/lcm/uv_cache
export HF_HOME=/work/hdd/bfaq/jlyu3/lcm/hf_cache
export HF_DATASETS_CACHE=/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets
export TMPDIR=/work/hdd/bfaq/jlyu3/lcm/tmp

mkdir -p $TMPDIR logs $OUTPUT_DIR

# 清理旧的日志文件（避免混淆）
echo "🧹 清理旧的日志文件..."
rm -f logs/prepare_gpu*.log

# 启动所有GPU任务
echo "🚀 启动 $NUM_GPUS 个GPU任务..."
echo "🔍 [调试] NUM_GPUS=$NUM_GPUS, 循环范围: seq 0 $((NUM_GPUS - 1))"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "🔍 [调试] 循环迭代: i=$i"
    
    # 计算 start_index
    if [ "$NUM_SAMPLES" = "all" ]; then
        # 处理整个数据集时，每个GPU处理不同的数据段
        # 使用一个大的步长来避免重叠（假设数据集很大，每个GPU处理不同的部分）
        # 这里使用一个固定的大步长，实际使用时可能需要根据数据集大小调整
        START_IDX=$((i * 1000000))  # 每个GPU间隔100万条数据
        echo "  GPU $i: 从索引 $START_IDX 开始处理（处理整个数据集）"
    else
        START_IDX=$((i * SAMPLES_PER_GPU))
        echo "  GPU $i: 样本 $START_IDX - $((START_IDX + SAMPLES_PER_GPU))"
    fi
    
    # CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python scripts/prepare_c4.py \
    #     --output_dir=$OUTPUT_DIR/shard_$i \
    #     --num_samples=$SAMPLES_PER_GPU \
    #     --start_index=$START_IDX \
    #     --batch_size=$BATCH_SIZE \
    #     > logs/prepare_gpu${i}.log 2>&1 &
    
    # TODO: added -u to force flush output
    # 构建 num_samples 参数
    if [ "$NUM_SAMPLES" = "all" ]; then
        NUM_SAMPLES_ARG="None"
    else
        NUM_SAMPLES_ARG="$SAMPLES_PER_GPU"
    fi
    
    MAX_SENTENCE_LENGTH=$MAX_SENTENCE_LENGTH \
    ADD_SPLIT_COLUMN=$ADD_SPLIT_COLUMN \
    TRAIN_RATIO=$TRAIN_RATIO \
    SEED=$SEED \
    CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python -u scripts/prepare_fine_web.py \
        --output_dir=$OUTPUT_DIR/shard_$i \
        --num_samples=$NUM_SAMPLES_ARG \
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
if [ "$NUM_SAMPLES" != "all" ]; then
    echo "⏱️  预计时间（$NUM_GPUS个GPU）:"
    HOURS=$((SAMPLES_PER_GPU / 10000))
    echo "  约 $HOURS 小时"
    echo ""
else
    echo "⏱️  预计时间: 未知（处理整个数据集）"
    echo ""
fi

# 等待所有任务完成
echo "⏳ 等待所有 GPU 任务完成..."
wait

echo ""
echo "✅ 所有 GPU 任务已完成！"
echo ""

# 统一更新 datacard（使用统一的路径，不包含 shard）
echo "📋 更新 datacard..."
.venv/bin/python scripts/update_datacards.py \
    --output_dir="$OUTPUT_DIR" \
    --dataset_name=fine_web_edu \
    --cluster_name=s3 \
    --add_split_column=$ADD_SPLIT_COLUMN \
    --train_ratio=$TRAIN_RATIO \
    --seed=$SEED

echo ""
echo "🎉 数据准备完成！"
echo "📁 数据目录: $OUTPUT_DIR"
echo "📋 Datacard 已更新到: lcm/datacards/datacards.yaml"
echo ""
