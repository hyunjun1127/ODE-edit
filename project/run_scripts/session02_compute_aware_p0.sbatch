#!/usr/bin/env bash
#SBATCH --job-name=odeedit-s02-p0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65000M
#SBATCH --time=04:00:00
#SBATCH --output=local/logs/session02-p0-%x-%j.out
#SBATCH --error=local/logs/session02-p0-%x-%j.err

set -euo pipefail

: "${MODEL_ALIAS:?set MODEL_ALIAS to llama3-8b-inst or qwen2.5-7b-inst}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT under local/results/session02-*}"
: "${SLURM_SUBMIT_DIR:?Slurm must preserve the approved submission directory}"

cd "${SLURM_SUBMIT_DIR}"
test -f project/run_scripts/session02_compute_aware_p0.py

export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:/mnt/raid5/janghj/EasyEdit

/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  project/run_scripts/session02_compute_aware_p0.py \
  --model-alias "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --execute
