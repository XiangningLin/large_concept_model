#!/bin/bash
#
# Script to run prepare_wikipedia.py on a GPU node
# 
# This script handles the SLURM-stopes incompatibility by manually allocating
# a GPU node and then running the script in local mode.
#

set -e

echo "================================================================"
echo "Wikipedia Preparation Pipeline"
echo "================================================================"
echo ""
echo "This script will:"
echo "  1. Request a GPU node from SLURM"
echo "  2. Set up the environment"
echo "  3. Run the data preparation pipeline"
echo ""
echo "Note: You will enter an interactive session on the GPU node."
echo "      The script will run automatically once allocated."
echo "================================================================"
echo ""

# Request GPU node
salloc --account=p32721 --partition=gengpu --nodes=1 --mem=256G \
       --cpus-per-task=8 --gres=gpu:a100:1 --time=6:00:00 \
       bash -c '
cd /projects/p32721/large_concept_model
export LD_LIBRARY_PATH="/gpfs/projects/p32721/large_concept_model/.venv/lib:/home/tgx0519/.conda/envs/lcm-helper/lib:$LD_LIBRARY_PATH"

echo ""
echo "GPU node allocated. Starting pipeline..."
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Could not detect GPU")"
echo ""

uv run --extra data python scripts/prepare_wikipedia.py output/dir_test

echo ""
echo "Pipeline completed!"
'

