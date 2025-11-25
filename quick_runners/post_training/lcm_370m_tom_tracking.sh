#!/bin/bash
#SBATCH --job-name=lcm_370m_tom_tracking
#SBATCH --account=p32721
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load CUDA module (required for PyTorch to detect CUDA)
# module load cuda/cuda-12.1.0-openmpi-4.1.4

# Activate conda environment
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate lcm-helper

# Set LD_LIBRARY_PATH for libsndfile and MKL
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# Set Python unbuffered output
export PYTHONUNBUFFERED=1

# Change to project directory
cd /projects/p32721/large_concept_model

# Create logs directory
mkdir -p logs

# Run training with submitit launcher
# Note: Since we're already in a SLURM job, submitit will detect this and run directly
# without submitting nested jobs (it uses the existing SLURM allocation)
uv run python -m lcm.train \
    launcher=submitit \
    +post_training=tom_tracking \
    ++trainer.output_dir="checkpoints/lcm_370m_tom_tracking" \
    ++trainer.experiment_name=lcm_370m_tom_tracking \
    ++launcher.partition=gengpu \
    ++launcher.account=p32721 \
    ++launcher.update_parameters.slurm_gres="gpu:a100:1" \
    ++launcher.update_parameters.slurm_srun_args='["--gres=gpu:a100:1"]' 
