#!/bin/bash
#SBATCH --job-name=aya-e6-pos
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e6-pos-%j.out
#SBATCH --error=logs/e6-pos-%j.err

set -euo pipefail
: "${NLP_PROJECT_ROOT:?Set NLP_PROJECT_ROOT to the copied project root}"
cd "$NLP_PROJECT_ROOT"
source "$NLP_PROJECT_ROOT/venv/bin/activate"
export HF_HOME="$NLP_PROJECT_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$NLP_PROJECT_ROOT/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

python project_plan/feasibility/e6_position_decomposition_run_20260928.py "$@"
