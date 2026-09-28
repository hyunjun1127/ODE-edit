#!/bin/bash
#SBATCH --job-name=odeedit_memit_history_10k_s3
#SBATCH --partition=gpu
#SBATCH --nodelist=ubuntu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=121856M
#SBATCH --time=7-00:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$1"
exec /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python -m project.run_scripts.memit_history_lifelong.runner --lock "$2"
