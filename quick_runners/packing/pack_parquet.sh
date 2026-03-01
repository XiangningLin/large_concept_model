#!/bin/bash
# Pack preprocessed parquet (merge short docs into fixed-length rows)
# Uses 1 GPU.
#
# Usage: ./quick_runners/packing/pack_parquet.sh
#   Or via sbatch: sbatch --gres=gpu:1 sbatch_runners/sbatch_bash_runner.sh ./quick_runners/packing/pack_parquet.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_DIR="${SOURCE_DIR:-/work/hdd/bfaq/jlyu3/lcm/preprocessed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bfaq/jlyu3/lcm/preprocessed_data_packed}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"

CUDA_VISIBLE_DEVICES=0 uv run python scripts/pack_parquet.py \
  --source_dir="$SOURCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --max_seq_len="$MAX_SEQ_LEN" \
  "$@"
