#!/usr/bin/env bash
#SBATCH --job-name=odeedit_alpha_jv_ds_sweep_s1
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=182272M
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
ds_repo="$1"
ds_run_root="$2"
cd "$ds_repo"
export PYTHONPATH="$ds_repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.alpha_jv_llama_diagnosis.launch premodel \
    --repo "$ds_repo" --run-root "$ds_run_root" --cell "$SLURM_ARRAY_TASK_ID"
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.alpha_jv_llama_diagnosis.gpu_runtime \
    --repo "$ds_repo" --run-root "$ds_run_root" --cell "$SLURM_ARRAY_TASK_ID"
