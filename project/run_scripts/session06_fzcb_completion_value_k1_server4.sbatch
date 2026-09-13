#!/usr/bin/env bash
# FzCB completion-value K1 Llama/Qwen independent B1-8 screen.
#SBATCH --job-name=fzcb_k1_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=90000M
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/data/janghj/ODE-edit/local/logs/fzcb-completion-value-fast-kill-v1/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/fzcb-completion-value-fast-kill-v1/%x-%A_%a.err

set -euo pipefail
umask 077

: "${FZCB_SOURCE_ROOT:?}"
: "${FZCB_EXPECTED_HEAD:?}"
: "${FZCB_RESULT_ROOT:?}"
: "${FZCB_PREFLIGHT_RECEIPT:?}"
: "${FZCB_EXPECTED_PREFLIGHT_SHA:?}"
: "${FZCB_K0_RESULT:?}"
: "${FZCB_EXPECTED_K0_SHA:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly ENV_PROJECT=/data/janghj/EasyEdit
readonly UV_BIN=/data/janghj/.local/bin/uv
readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CELL_ROOT="${FZCB_RESULT_ROOT}/${MODEL}-memit-k1"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${FZCB_SOURCE_ROOT}" rev-parse HEAD)" == "${FZCB_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${FZCB_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(sha256sum "${FZCB_PREFLIGHT_RECEIPT}" | awk '{print $1}')" == "${FZCB_EXPECTED_PREFLIGHT_SHA}" ]]
[[ "$(sha256sum "${FZCB_K0_RESULT}" | awk '{print $1}')" == "${FZCB_EXPECTED_K0_SHA}" ]]
[[ "$(jq -r '.status' "${FZCB_K0_RESULT}")" == TERMINAL_VALID ]]
[[ "$(jq -r '.source.head' "${FZCB_PREFLIGHT_RECEIPT}")" == "${FZCB_EXPECTED_HEAD}" ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${FZCB_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python \
  -m project.run_scripts.fzcb_completion_value.runtime \
  --stage k1 --model "${MODEL}" --source-root "${FZCB_SOURCE_ROOT}" \
  --output "${CELL_ROOT}/result.json"
