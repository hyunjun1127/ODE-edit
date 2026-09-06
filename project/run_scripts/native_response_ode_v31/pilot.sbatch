#!/usr/bin/env bash
#SBATCH --job-name=odeedit_native_v31_pilot
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=182272M
#SBATCH --time=02:00:00
#SBATCH --array=0-3%1
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
pilot_repo="$1"
pilot_root="$2"
cd "$pilot_repo"
export PYTHONPATH="$pilot_repo"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.native_response_ode_v31.runtime \
    --repo "$pilot_repo" --run-root "$pilot_root" --cell "$SLURM_ARRAY_TASK_ID"
