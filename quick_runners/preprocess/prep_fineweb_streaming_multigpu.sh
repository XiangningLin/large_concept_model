#!/bin/bash
# Multi-GPU FineWeb preprocessing (streaming + islice, no accelerate)
# 用 bash 循环启动多进程，每个进程通过 rank/num_shards 做 islice 交错分片
#
# Usage: ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh
#   Or: NUM_GPUS=4 START_INDEX=0 NUM_SAMPLES=100000 ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh
#
# Env: HF_TOKEN for private datasets (LGVamper/fineweb-edu-20B)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/hdd/bfaq/jlyu3/lcm/uv_cache}"
export HF_HOME="${HF_HOME:-/work/hdd/bfaq/jlyu3/lcm/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets}"
export TMPDIR="${TMPDIR:-/work/hdd/bfaq/jlyu3/lcm/tmp}"

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bfaq/jlyu3/lcm/preprocessed_data}"
NUM_GPUS="${NUM_GPUS:-8}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-1000000}"

echo "======================================"
echo "FineWeb 流式多卡预处理 (islice)"
echo "======================================"
echo "GPU 数: $NUM_GPUS"
echo "start_index: $START_INDEX"
echo "num_samples: $NUM_SAMPLES"
echo "输出目录: $OUTPUT_DIR"
echo "======================================"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i uv run python scripts/prepare_fine_web.py \
    --rank="$i" \
    --num_shards="$NUM_GPUS" \
    --start_index="$START_INDEX" \
    --num_samples="$NUM_SAMPLES" \
    --output_dir="$OUTPUT_DIR" \
    --checkpoint_interval=5000 \
    "$@" &
done
wait

echo ""
echo "所有 rank 已完成。输出: $OUTPUT_DIR/rank_0/ .. rank_$((NUM_GPUS - 1))/"
echo ""
