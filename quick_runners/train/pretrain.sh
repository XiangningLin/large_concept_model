#!/bin/bash
# LCM Pretrain (8-GPU standard)
# 370M example: S=4752, pretrain 0.9S=4277 steps.
#
# Usage: ./quick_runners/train/pretrain.sh [hydra overrides...]
#   Or via sbatch: sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain.sh
#
# Env: RECIPE (default mse_370M), OUTPUT_DIR, EXPERIMENT_NAME, MAX_STEPS, WANDB_PROJECT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RECIPE="${RECIPE:-mse_370M}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_pretrain}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lcm_370m_pretrain}"
WANDB_PROJECT="${WANDB_PROJECT:-lcm_pretrain}"
MAX_STEPS="${MAX_STEPS:-4277}"
NPROC="${NPROC:-8}"

# Checkpoint milestones (SentenceSSM scaling law points)
MILESTONES="[225000000,318600000,450000000,636300000,900000000,1272600000,1800000000,2545200000,3600000000,5091300000,7200000000,10182600000,14400000000]"

uv run torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m lcm.train \
  launcher=standalone \
  "+pretrain=$RECIPE" \
  "++trainer.output_dir=$OUTPUT_DIR" \
  "++trainer.experiment_name=$EXPERIMENT_NAME" \
  "++trainer.wandb_project=$WANDB_PROJECT" \
  "++trainer.max_steps=$MAX_STEPS" \
  '++trainer.lr_stage_ratios=[0.1,0.9,0.0]' \
  "++trainer.checkpoint_milestones=$MILESTONES" \
  "++trainer.validate_every_n_steps=100" \
  "++trainer.publish_metrics_every_n_steps=10" \
  "$@"
