#!/usr/bin/env bash
#SBATCH --job-name=odeedit-s02-bf16-context-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65000M
#SBATCH --time=00:30:00
#SBATCH --output=local/logs/session02-bf16-context-probe-%x-%j.out
#SBATCH --error=local/logs/session02-bf16-context-probe-%x-%j.err

set -euo pipefail

# This template is deliberately non-executable without a future GH envelope.
: "${ODEEDIT_CONTEXT_PROBE_SUBMISSION_AUTHORIZED:?separate paired calibration envelope required}"
test "${ODEEDIT_CONTEXT_PROBE_SUBMISSION_AUTHORIZED}" = "bf16-context-lock-probe-v1"
: "${MODEL_ALIAS:?set one fixed model alias}"
: "${OUTPUT_ROOT:?set one new local context-probe output root}"
: "${SLURM_SUBMIT_DIR:?Slurm must preserve the approved submission directory}"

cd "${SLURM_SUBMIT_DIR}"
test -f project/run_scripts/session02_bf16_context_lock_probe.py

export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:/mnt/raid5/janghj/EasyEdit

/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  project/run_scripts/session02_bf16_context_lock_probe.py \
  --model-alias "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --execute
