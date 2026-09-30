#!/bin/bash
#SBATCH --job-name=odeedit_memit_hj_v2_s3
#SBATCH --partition=gpu
#SBATCH --nodelist=ubuntu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=121856M
#SBATCH --time=30-00:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
TASK_SOURCE="$1"
TASK_LOCK="$2"
TASK_GROUP="$3"
export PYTHONPATH="$TASK_SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
cd "$TASK_SOURCE"
exec /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python -u -m project.run_scripts.memit_hj.runner --lock "$TASK_LOCK" --group "$TASK_GROUP"
