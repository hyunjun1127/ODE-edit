#!/usr/bin/env bash
#SBATCH --job-name=odeedit_alpha_jv_seq_s2
#SBATCH --partition=gpu
#SBATCH --nodelist=server2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --export=NONE
set -euo pipefail
umask 077
task_repo="$1"
task_root="$2"
task_mode="$3"
cd "$task_repo"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONPATH="$task_repo"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
bash scripts/check-session-boundary.sh 01a0493a-074c-7f91-9a13-769116326fef
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m \
  project.run_scripts.alpha_native_response_ode_v31_sequential.runtime \
  --repo "$task_repo" --root "$task_root" --mode "$task_mode" --index "$SLURM_ARRAY_TASK_ID"
