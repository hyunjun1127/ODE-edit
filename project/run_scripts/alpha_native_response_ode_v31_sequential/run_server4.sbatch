#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server4
#SBATCH --job-name=odeedit_alpha_l8_s4_takeover
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --array=4-5%2
#SBATCH --export=NONE
set -euo pipefail
umask 077
task_repo="$1"
task_root="$2"
cd "$task_repo"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONPATH="$task_repo"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
bash scripts/check-session-boundary.sh 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.alpha_native_response_ode_v31_sequential.server4_takeover \
  run --repo "$task_repo" --root "$task_root" --index "${SLURM_ARRAY_TASK_ID:?}"
