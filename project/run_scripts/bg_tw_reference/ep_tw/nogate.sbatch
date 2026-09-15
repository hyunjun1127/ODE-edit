#!/usr/bin/env bash
#SBATCH --job-name=odeedit_ep_tw1_nogate_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
umask 077
task_source="$1"
task_lock="$2"
task_output="$3"
task_deps=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_deps:$task_source"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
cd "$task_source"
exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.bg_tw_reference.ep_tw.runner --lock "$task_lock" --output "$task_output"
