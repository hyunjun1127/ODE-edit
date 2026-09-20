#!/bin/bash
#SBATCH --job-name=odeedit_en_adapt_B300_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
attempt="$1"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1 SAVE_CHECKPOINTS=false
export EN_ADAPT_READY_MANIFEST="$attempt/execution-inputs/manifest.json"
export EN_ADAPT_EVALUATOR_MAP="$attempt/execution-inputs/evaluator-map.json"
export PYTHONPATH="$attempt/source:/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2"
export SLURM_EXPORT_ENV=ALL
cd "$attempt/source"
exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.en_adaptive_nullspace.runner --output "$attempt/output"
