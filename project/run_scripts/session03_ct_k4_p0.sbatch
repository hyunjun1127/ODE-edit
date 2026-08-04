#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65000M
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodelist=devbox
#SBATCH --output=local/logs/session03-ct-k4-p0-%x-%j.out
#SBATCH --error=local/logs/session03-ct-k4-p0-%x-%j.err
set -euo pipefail
: "${MODEL_ALIAS:?MODEL_ALIAS is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${ODEEDIT_SESSION03_CT_K4_P0_AUTHORIZED:?execution token is required}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export PYTHONPATH=/mnt/raid5/janghj/EasyEdit
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python \
  project/run_scripts/session03_ct_k4_p0.py \
  --model-alias "${MODEL_ALIAS}" --output-root "${OUTPUT_ROOT}" --execute
