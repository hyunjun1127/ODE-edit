#!/usr/bin/env bash
#SBATCH --job-name=odeedit_blue_seq_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --export=NONE
set -euo pipefail
umask 077
task_repo="$1"
task_campaign="$2"
task_mode="$3"
task_output="$4"
task_dependencies="$5"
cd "$task_repo"
bash scripts/check-session-boundary.sh 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_dependencies:$task_repo"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.blue_alphaedit_sequential_comparison.runtime \
  --lock "$task_campaign/execution.lock.json" --output "$task_output" --mode "$task_mode"
