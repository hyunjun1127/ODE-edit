#!/usr/bin/env bash
#SBATCH --job-name=odeedit_cake_native_lifelong_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
umask 077
task_root="$1"
task_helper=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/source-tech-r2
task_deps=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_deps:$task_root/source:$task_helper"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
cd "$task_root/execution/CAKE"
exec /data/janghj/EasyEdit/.venv/bin/python -B -u -m cake_native_lifelong.runtime --lock "$task_root/execution.lock.json"
