#!/bin/bash
#SBATCH --job-name=lcm_runner
#SBATCH --account=bfaq-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ======== Universal SBATCH Bash Runner for LCM ========
#
# Usage: sbatch sbatch_bash_runner.sh <bash_script_path> [script_arguments...]
#
# Examples:
#   sbatch sbatch_bash_runner.sh ./quick_runners/train/pretrain.sh
#   sbatch sbatch_bash_runner.sh ./quick_runners/preprocess/prep_fineweb_parallel_8gpu.sh
#   sbatch sbatch_bash_runner.sh ./quick_runners/train/lr_sweep_pretrain.sh
#
# For 1-GPU tasks (packing, single-GPU prep), override resources:
#   sbatch --gres=gpu:1 --mem=128G sbatch_bash_runner.sh ./quick_runners/packing/pack_parquet.sh
#
# =====================================================

export HF_TOKEN="${HF_TOKEN:-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}"
echo "HF_TOKEN has been set for this session"

if [ $# -eq 0 ]; then
    echo "Error: No bash script path provided!"
    echo "Usage: sbatch $0 <bash_script_path> [script_arguments...]"
    exit 1
fi

SCRIPT_PATH=$1
shift

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "========================================"
echo "SBATCH Runner Starting (LCM)"
echo "========================================"
echo "Script: $SCRIPT_PATH"
echo "Arguments: $*"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"
echo ""

export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/hdd/bfaq/jlyu3/lcm/uv_cache}"
export HF_HOME="${HF_HOME:-/work/hdd/bfaq/jlyu3/lcm/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets}"
export TMPDIR="${TMPDIR:-/work/hdd/bfaq/jlyu3/lcm/tmp}"
export PYTHONUNBUFFERED=1

PROJECT_ROOT="/projects/bfaq/jlyu3/large_concept_model"
cd "$PROJECT_ROOT" || { echo "Error: cannot cd to $PROJECT_ROOT"; exit 1; }

mkdir -p logs
chmod +x "$SCRIPT_PATH" 2>/dev/null || true

echo "Executing: bash $SCRIPT_PATH $*"
echo ""
bash "$SCRIPT_PATH" "$@"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "SBATCH Runner Finished"
echo "Exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "========================================"

exit $EXIT_CODE
