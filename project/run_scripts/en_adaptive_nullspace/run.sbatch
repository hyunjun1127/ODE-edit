#!/bin/bash
#SBATCH --job-name=odeedit_en_adaptive_nullspace_s3
#SBATCH --partition=gpu
#SBATCH --nodelist=ubuntu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=121856M
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
attempt="$1"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1 SAVE_CHECKPOINTS=false
export PYTHONPATH="$attempt/source"
export SLURM_EXPORT_ENV=ALL
cd "$attempt/source"
exec /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python -u -m project.run_scripts.en_adaptive_nullspace.runner --output "$attempt/output"
