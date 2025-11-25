#!/bin/bash
#SBATCH --job-name=lcm_370m_tom_tracking_4GPU_submit
#SBATCH --account=p32721
#SBATCH --partition=gengpu  
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4 
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00  
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# Activate conda environment
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate lcm-helper

# Set LD_LIBRARY_PATH for libsndfile and MKL
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
# Set Python unbuffered output
export PYTHONUNBUFFERED=1

# Change to project directory
cd /projects/p32721/large_concept_model

CUDA_VISIBLE_DEVICES=0,1,2,3 

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    -m lcm.train launcher=standalone \
    +post_training=tom_tracking_4GPU \
    ++trainer.output_dir="checkpoints/lcm_370m_tom_tracking_4GPU" \
    ++trainer.experiment_name=lcm_370m_tom_tracking_4GPU \
    +trainer.use_submitit=false \
    ++trainer.model_config_or_name=base_lcm_370M