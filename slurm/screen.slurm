#!/bin/bash
#SBATCH --job-name=aya-screen
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/screen-%j.out
#SBATCH --error=logs/screen-%j.err

set -euo pipefail
: "${NLP_PROJECT_ROOT:?Set the copied project root}"
cd "$NLP_PROJECT_ROOT"
source "$NLP_PROJECT_ROOT/venv/bin/activate"
export HF_HOME="$NLP_PROJECT_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
python ff.py screen "$@"
