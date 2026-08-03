#!/usr/bin/env bash
#SBATCH --job-name=odeedit-s02-oracle-mean-p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65000M
#SBATCH --time=08:00:00
#SBATCH --output=local/logs/session02-oracle-mean-event-p1-r1-%x-%j.out
#SBATCH --error=local/logs/session02-oracle-mean-event-p1-r1-%x-%j.err

set -euo pipefail

: "${ODEEDIT_ORACLE_MEAN_P1_SUBMISSION_AUTHORIZED:?conditional GH P1 authority required}"
test "${ODEEDIT_ORACLE_MEAN_P1_SUBMISSION_AUTHORIZED}" = "oracle-mean-event-v2-r1-p1-after-p0-pass"
: "${MODEL_ALIAS:?set one canonical model alias}"
: "${OUTPUT_ROOT:?set the exact dry-plan output root}"
: "${SLURM_SUBMIT_DIR:?Slurm submission directory is required}"

cd "${SLURM_SUBMIT_DIR}"
test -f project/run_scripts/session02_oracle_mean_event_p1.py

export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:/mnt/raid5/janghj/EasyEdit

/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  project/run_scripts/session02_oracle_mean_event_p1.py \
  --model-alias "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --execute
