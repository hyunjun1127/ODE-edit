#!/bin/bash
#SBATCH --job-name=odeedit_server3_readiness
#SBATCH --partition=gpu
#SBATCH --nodelist=ubuntu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=121856M
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
source /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/runtime.env
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SH3_REPO"
for model in llama3-8b-inst qwen2.5-7b-inst; do
  "$SH3_PYTHON" "$SH3_SMOKE_SOURCE" --manifest "$SH3_RUNTIME_MANIFEST" --model "$model" --output "$SH3_SMOKE_OUTPUT/$SLURM_JOB_ID-$model.json"
done
