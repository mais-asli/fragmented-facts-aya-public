#!/bin/bash
#SBATCH --job-name=aya-smoke
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

set -euo pipefail
: "${NLP_PROJECT_ROOT:?Set the copied project root}"
cd "$NLP_PROJECT_ROOT"
source "$NLP_PROJECT_ROOT/venv/bin/activate"
export HF_HOME="$NLP_PROJECT_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python project_plan/feasibility/aya_smoke.py \
  --snapshot-manifest model_snapshot.json \
  --precision "${AYA_PRECISION:-nf4}" \
  --output "results/aya-smoke-${AYA_PRECISION:-nf4}-${SLURM_JOB_ID}.json"
