#!/bin/bash
# LR Sweep Pretraining Script for LCM
# Runs multiple short pretrain runs with different peak_lr values,
# records tail validation loss (avg + std) for plotting.
#
# Usage:
#   ./quick_runners/train/lr_sweep_pretrain.sh [extra hydra overrides...]
#
# Examples:
#   # Default LR range with mse_60M recipe:
#   ./quick_runners/train/lr_sweep_pretrain.sh
#
#   # Custom LR range:
#   LR_VALUES="1e-5 1e-4 1e-3" ./quick_runners/train/lr_sweep_pretrain.sh
#
#   # Override recipe or data:
#   RECIPE=mse ./quick_runners/train/lr_sweep_pretrain.sh
#
#   # Use fp32 for high LR (avoids slow fp16 overflow retries):
#   LR_VALUES="1e-4 1e-3 1e-2" FP32_LR_VALUES="1e-2" ./quick_runners/train/lr_sweep_pretrain.sh
#
# Environment variables:
#   LR_VALUES       Space-separated LR values (default: 1e-5 ... 1e-1)
#   RECIPE          Hydra recipe name (default: mse_60M)
#   SWEEP_STEPS     Total training steps per LR (default: 500)
#   WARMUP_STEPS    Warmup steps (default: 20)
#   EVAL_STEPS      Validate every N steps (default: 10)
#   TAIL_RATIO      Fraction of steps for tail metrics (default: 0.2)
#   RESULTS_DIR     Output directory for CSV (default: ./results/lr_sweep/)
#   NPROC           GPUs per node (default: 1)
#   CUDA_DEVICES    CUDA_VISIBLE_DEVICES override (default: 0)
#   USE_FP32        If "1" or "true", use fp32 for all runs (avoids fp16 overflow
#                   with high LR like 1e-2; slower but stable). Default: unset.
#   FP32_LR_VALUES  Space-separated LRs to run in fp32 (e.g. "1e-2 1e-1").
#                   Use when only high LRs overflow; overrides recipe dtype.
#
# Output: results/lr_sweep/lr_sweep_results.csv  (peak_lr,avg_val_loss,tail_std)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configurable defaults ─────────────────────────────────────────
LR_VALUES="${LR_VALUES:-1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 5e-3 1e-2 1e-1}"
RECIPE="${RECIPE:-mse_60M}"
SWEEP_STEPS="${SWEEP_STEPS:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
EVAL_STEPS="${EVAL_STEPS:-10}"
TAIL_RATIO="${TAIL_RATIO:-0.2}"
NPROC="${NPROC:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

RESULTS_DIR="${RESULTS_DIR:-./results/lr_sweep}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/lr_sweep_results.csv}"
mkdir -p "$RESULTS_DIR"

OUTPUT_BASE="checkpoints/lr_sweep_tmp"

echo "peak_lr,avg_val_loss,tail_std" > "$RESULTS_FILE"

# ── Fixed sweep overrides ─────────────────────────────────────────
# Use lr_schedule=noop so the LR stays constant after warmup (no WSD phases)
SWEEP_ARGS=(
    "++trainer.lr_sweep_mode=true"
    "++trainer.tail_ratio=$TAIL_RATIO"
    "++trainer.max_steps=$SWEEP_STEPS"
    "++trainer.num_lr_warmup_steps=$WARMUP_STEPS"
    "++trainer.lr_schedule=noop"
    "++trainer.validate_every_n_steps=$EVAL_STEPS"
    "++trainer.checkpoint_every_n_steps=999999"
    "++trainer.save_model_every_n_steps=999999"
    "++trainer.publish_metrics_every_n_steps=$EVAL_STEPS"
    "++trainer.use_fsdp=false"
    "++trainer.checkpoint_milestones=[]"
)

echo "=== LCM LR Sweep ==="
echo "  Recipe:       $RECIPE"
echo "  Steps:        $SWEEP_STEPS  (warmup: $WARMUP_STEPS)"
echo "  Eval every:   $EVAL_STEPS steps"
echo "  Tail ratio:   $TAIL_RATIO"
echo "  LR values:    $LR_VALUES"
echo "  Results:      $RESULTS_FILE"
echo ""

for lr in $LR_VALUES; do
    echo ">>> Running with peak_lr=$lr"
    run_dir="${OUTPUT_BASE}/lr_${lr}"

    # Use fp32 for high LR to avoid fp16 overflow (avoids slow overflow retries)
    DTYPE_ARGS=()
    if [ -n "$USE_FP32" ] && { [ "$USE_FP32" = "1" ] || [ "$USE_FP32" = "true" ]; }; then
        DTYPE_ARGS=("++trainer.dtype=torch.float32")
        echo "    (using fp32 for stability)"
    elif [ -n "$FP32_LR_VALUES" ]; then
        for fp32_lr in $FP32_LR_VALUES; do
            if [ "$lr" = "$fp32_lr" ]; then
                DTYPE_ARGS=("++trainer.dtype=torch.float32")
                echo "    (using fp32 for lr=$lr to avoid overflow)"
                break
            fi
        done
    fi

    set +e
    output=$(CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run torchrun \
        --standalone --nnodes=1 --nproc-per-node="$NPROC" \
        -m lcm.train launcher=standalone \
        "+pretrain=$RECIPE" \
        "++trainer.output_dir=$run_dir" \
        "++trainer.experiment_name=lr_sweep_${lr}" \
        "++trainer.lr=$lr" \
        "${SWEEP_ARGS[@]}" \
        "${DTYPE_ARGS[@]}" \
        "$@" 2>&1)
    exit_code=$?
    set -e

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "!!! LR Sweep FAILED at peak_lr=$lr (exit code $exit_code)"
        echo "=== Error output ==="
        echo "$output"
        exit $exit_code
    fi

    result=$(echo "$output" | grep "FINAL LOSS: tail_avg_val_loss=" | tail -1)
    if [ -n "$result" ]; then
        tail_avg=$(echo "$result" | sed -n 's/.*tail_avg_val_loss=\([^,]*\).*/\1/p')
        tail_std=$(echo "$result" | sed -n 's/.*tail_std=\([^,]*\).*/\1/p')
        echo "$lr,$tail_avg,$tail_std" >> "$RESULTS_FILE"
        echo "    peak_lr=$lr  avg_val_loss=$tail_avg  tail_std=$tail_std"
    else
        echo "!!! No FINAL LOSS found in output (run may have failed silently)"
        echo "=== Output (last 50 lines) ==="
        echo "$output" | tail -50
        exit 1
    fi
    echo ""
done

# Clean up temporary checkpoints
rm -rf "$OUTPUT_BASE"

echo "=== LR Sweep complete. Results in $RESULTS_FILE ==="
cat "$RESULTS_FILE"
