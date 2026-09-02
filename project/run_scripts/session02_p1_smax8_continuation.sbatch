#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server1
#SBATCH --job-name=odeedit-s02-p1-smax8-cont
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65000M
#SBATCH --time=04:00:00
#SBATCH --output=local/logs/session02-p1-smax8-%x-%j.out
#SBATCH --error=local/logs/session02-p1-smax8-%x-%j.err

set -euo pipefail

# This template is intentionally inert until a separate GH execution envelope
# provides the exact token.  Source-derived zero eligibility means no Qwen job
# is part of the future submission batch.
: "${ODEEDIT_P1_SMAX8_SUBMISSION_AUTHORIZED:?separate GH continuation envelope required}"
test "${ODEEDIT_P1_SMAX8_SUBMISSION_AUTHORIZED}" = "p1-smax8-affected-cont-v1"
: "${MODEL_ALIAS:?MODEL_ALIAS must be llama3-8b-inst}"
test "${MODEL_ALIAS}" = "llama3-8b-inst"
: "${OUTPUT_ROOT:?set the exact predeclared continuation output root}"
: "${SLURM_SUBMIT_DIR:?Slurm must preserve the approved submission directory}"

cd "${SLURM_SUBMIT_DIR}"
test -f project/run_scripts/session02_p1_smax8_continuation.py

export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:/mnt/raid5/janghj/EasyEdit
export ODEEDIT_P1_SMAX8_EXECUTION_AUTHORIZED=p1-smax8-affected-cont-v1

/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  project/run_scripts/session02_p1_smax8_continuation.py \
  --model-alias "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --execute
