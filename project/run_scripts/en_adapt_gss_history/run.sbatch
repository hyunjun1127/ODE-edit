#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=ubuntu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=121856M
#SBATCH --time=7-00:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
attempt="$1"
source_root="$2"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1 SAVE_CHECKPOINTS=false
export EN_ADAPT_READY_MANIFEST="$attempt/execution-inputs/manifest.json"
export EN_ADAPT_EVALUATOR_MAP="$attempt/execution-inputs/evaluator-map.json"
export PYTHONPATH="$source_root"
export SLURM_EXPORT_ENV=ALL
cd "$source_root"
exec /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python -u -m project.run_scripts.en_adapt_gss_history.launch --attempt "$attempt"
