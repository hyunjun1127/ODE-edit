#!/usr/bin/env bash
# W0-only fresh ctrl/gate anchor calibration. Edited endpoint access is zero.
#SBATCH --job-name=barrier_f2_anchor_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=70000M
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/data/janghj/ODE-edit/local/logs/barrier-usefulness-f2-v1/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/barrier-usefulness-f2-v1/%x-%A_%a.err

set -euo pipefail
umask 077
: "${BARRIER_F2_SOURCE_ROOT:?}"
: "${BARRIER_F2_EXPECTED_HEAD:?}"
: "${BARRIER_F2_ANCHOR_ROOT:?}"
readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CELL="${BARRIER_F2_ANCHOR_ROOT}/${MODEL}"
[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${BARRIER_F2_SOURCE_ROOT}" rev-parse HEAD)" == "${BARRIER_F2_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${BARRIER_F2_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ ! -e "${CELL}" && ! -L "${CELL}" ]]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/janghj/.cache/huggingface HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${BARRIER_F2_SOURCE_ROOT}:/data/janghj/EasyEdit-stock-14cea824"
exec /data/janghj/.local/bin/uv run --offline --project /data/janghj/EasyEdit --frozen --no-sync python \
  -m project.run_scripts.barrier_usefulness.anchors --model "${MODEL}" --source-root "${BARRIER_F2_SOURCE_ROOT}" \
  --output-root "${CELL}" --output "${CELL}/anchor-seal.json"
