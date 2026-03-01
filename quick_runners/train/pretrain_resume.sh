#!/bin/bash
# LCM Pretrain resume from milestone checkpoint
#
# Usage: ./quick_runners/train/pretrain_resume.sh
#   Or: RESUME_FROM=/path/to/milestone sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain_resume.sh
#
# Env: RESUME_FROM (required), WANDB_RUN_ID (optional, to inherit WandB run)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "$RESUME_FROM" ]; then
  echo "Error: RESUME_FROM is required (e.g. /work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_pretrain/milestones/milestone_tokens_3600000000)"
  exit 1
fi

RECIPE="${RECIPE:-mse_370M}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_pretrain}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lcm_370m_pretrain}"
WANDB_PROJECT="${WANDB_PROJECT:-lcm_pretrain}"
MAX_STEPS="${MAX_STEPS:-4277}"
NPROC="${NPROC:-8}"
MILESTONES="[225000000,318600000,450000000,636300000,900000000,1272600000,1800000000,2545200000,3600000000,5091300000,7200000000,10182600000,14400000000]"

WANDB_ARGS=()
[ -n "$WANDB_RUN_ID" ] && WANDB_ARGS+=("++trainer.wandb_run_id=$WANDB_RUN_ID")

uv run torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m lcm.train \
  launcher=standalone \
  "+pretrain=$RECIPE" \
  "++trainer.output_dir=$OUTPUT_DIR" \
  "++trainer.experiment_name=$EXPERIMENT_NAME" \
  "++trainer.wandb_project=$WANDB_PROJECT" \
  "++trainer.resume_from=$RESUME_FROM" \
  "++trainer.max_steps=$MAX_STEPS" \
  '++trainer.lr_stage_ratios=[0.1,0.9,0.0]' \
  "++trainer.checkpoint_milestones=$MILESTONES" \
  "++trainer.validate_every_n_steps=100" \
  "++trainer.publish_metrics_every_n_steps=10" \
  "${WANDB_ARGS[@]}" \
  "$@"
