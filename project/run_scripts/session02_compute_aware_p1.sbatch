#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server1
#SBATCH --job-name=odeedit-s02-p1-four-case
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65000M
#SBATCH --time=16:00:00
#SBATCH --output=local/logs/session02-p1-%x-%j.out
#SBATCH --error=local/logs/session02-p1-%x-%j.err

set -euo pipefail

# This tracked template is intentionally inert.  A separate GH P1 execution
# envelope must provide the exact token; implementation prep grants no submit.
: "${ODEEDIT_P1_SUBMISSION_AUTHORIZED:?separate GH P1 execution envelope required}"
test "${ODEEDIT_P1_SUBMISSION_AUTHORIZED}" = "p1-four-case-mechanism-v1"
: "${MODEL_ALIAS:?set MODEL_ALIAS to llama3-8b-inst or qwen2.5-7b-inst}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT under local/results/session02-*}"
: "${SLURM_SUBMIT_DIR:?Slurm must preserve the approved submission directory}"

cd "${SLURM_SUBMIT_DIR}"
test -f project/run_scripts/session02_compute_aware_p1.py

export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:/mnt/raid5/janghj/EasyEdit

/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  project/run_scripts/session02_compute_aware_p1.py \
  --model-alias "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --execute
