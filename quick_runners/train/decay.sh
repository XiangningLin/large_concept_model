#!/bin/bash
# LCM Decay phase (decay_from_pretrain, 8-GPU)
# From pretrain milestone, WSD scheduler with decay_ratio=0.1.
#
# Usage: ./quick_runners/train/decay.sh
#   Or: RESUME_FROM=/path/to/milestone sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/decay.sh
#
# Env: RESUME_FROM (required), WANDB_RUN_ID (optional)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RESUME_FROM="${RESUME_FROM:-/work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_pretrain/milestones/milestone_tokens_14400000000}"

RECIPE="${RECIPE:-mse_370M}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_decay}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lcm_370m_decay}"
WANDB_PROJECT="${WANDB_PROJECT:-lcm_pretrain}"
MAX_STEPS="${MAX_STEPS:-4752}"
DECAY_RATIO="${DECAY_RATIO:-0.1}"
NPROC="${NPROC:-8}"
MILESTONES="[14500000000,15000000000,15500000000,16000000000]"

WANDB_ARGS=()
[ -n "$WANDB_RUN_ID" ] && WANDB_ARGS+=("++trainer.wandb_run_id=$WANDB_RUN_ID")

uv run torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m lcm.train \
  launcher=standalone \
  "+pretrain=$RECIPE" \
  "++trainer.output_dir=$OUTPUT_DIR" \
  "++trainer.experiment_name=$EXPERIMENT_NAME" \
  "++trainer.wandb_project=$WANDB_PROJECT" \
  "++trainer.training_mode=decay_from_pretrain" \
  "++trainer.resume_from=$RESUME_FROM" \
  "++trainer.max_steps=$MAX_STEPS" \
  "++trainer.decay_ratio=$DECAY_RATIO" \
  "++trainer.checkpoint_milestones=$MILESTONES" \
  "++trainer.validate_every_n_steps=50" \
  "++trainer.publish_metrics_every_n_steps=10" \
  "${WANDB_ARGS[@]}" \
  "$@"
