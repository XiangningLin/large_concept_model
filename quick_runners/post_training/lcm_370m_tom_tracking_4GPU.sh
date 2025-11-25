#!/bin/bash
#SBATCH --job-name=lcm_370m_tom_tracking_4GPU_submit
#SBATCH --account=p32721
#SBATCH --partition=gengpu  # 使用 CPU 节点（如果 Quest 有 normal partition，否则用 gengpu 但不申请 GPU）
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4 
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00  # CPU 节点只需要很短时间提交任务
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# Activate conda environment
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate lcm-helper

# Set LD_LIBRARY_PATH for libsndfile and MKL
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set Python unbuffered output
export PYTHONUNBUFFERED=1

# Change to project directory
cd /projects/p32721/large_concept_model

# Create logs directory
mkdir -p logs

# Run training with submitit launcher
# 工作流程：
# 1. 这个脚本先用 sbatch 申请一个 CPU 节点（normal partition）
# 2. 在 CPU 节点上运行 uv run python -m lcm.train launcher=submitit
# 3. submitit 会检测到当前不在 GPU 节点上，因此会提交一个新的 SLURM 任务到 gengpu partition
# 4. GPU 任务会在 GPU 节点上执行实际的训练
uv run python -m lcm.train \
  launcher=submitit \
  +post_training=tom_tracking_4GPU \
  ++trainer.output_dir="checkpoints/lcm_370m_tom_tracking_4GPU" \
  ++trainer.experiment_name=lcm_370m_tom_tracking_4GPU \
  ++launcher.partition=gengpu \
  ++launcher.account=p32721 \
  ++launcher.update_parameters.slurm_gres="gpu:a100:4" \
  ++launcher.update_parameters.slurm_srun_args='["--ntasks-per-node=4","--gpus-per-task=1","--gpu-bind=single:1"]'
