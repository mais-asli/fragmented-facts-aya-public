#!/bin/bash
#SBATCH --job-name=aya-production-check
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/production-check-%j.out
#SBATCH --error=logs/production-check-%j.err
set -euo pipefail
cd "${NLP_PROJECT_ROOT:?Project root required}"
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
python project_plan/feasibility/aya_production_check.py --output "results/aya-production-check-${SLURM_JOB_ID}.json"
