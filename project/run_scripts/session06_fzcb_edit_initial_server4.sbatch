#!/usr/bin/env bash
# FzCB-Edit MEMIT initial B1/B10 cell launcher (one model/batch per job).
#SBATCH --job-name=fzcb_edit_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=90000M
#SBATCH --time=48:00:00
#SBATCH --output=/data/janghj/ODE-edit/local/logs/fzcb-edit-main-method-initial-v1/%x-%j.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/fzcb-edit-main-method-initial-v1/%x-%j.err

set -euo pipefail
umask 077

: "${FZCB_EDIT_SOURCE_ROOT:?}"
: "${FZCB_EDIT_EXPECTED_HEAD:?}"
: "${FZCB_EDIT_PREFLIGHT_RECEIPT:?}"
: "${FZCB_EDIT_EXPECTED_PREFLIGHT_SHA:?}"
: "${FZCB_EDIT_RESULT_ROOT:?}"
: "${FZCB_EDIT_MODEL:?}"
: "${FZCB_EDIT_BATCH_SIZE:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly ENV_PROJECT=/data/janghj/EasyEdit
readonly UV_BIN=/data/janghj/.local/bin/uv
readonly CELL_ROOT="${FZCB_EDIT_RESULT_ROOT}/${FZCB_EDIT_MODEL}-B${FZCB_EDIT_BATCH_SIZE}"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${FZCB_EDIT_SOURCE_ROOT}" rev-parse HEAD)" == "${FZCB_EDIT_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${FZCB_EDIT_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ -f "${FZCB_EDIT_PREFLIGHT_RECEIPT}" && ! -L "${FZCB_EDIT_PREFLIGHT_RECEIPT}" ]]
[[ "$(sha256sum "${FZCB_EDIT_PREFLIGHT_RECEIPT}" | awk '{print $1}')" == "${FZCB_EDIT_EXPECTED_PREFLIGHT_SHA}" ]]
[[ "$(jq -r '.source.head' "${FZCB_EDIT_PREFLIGHT_RECEIPT}")" == "${FZCB_EDIT_EXPECTED_HEAD}" ]]
[[ "${FZCB_EDIT_BATCH_SIZE}" == 1 || "${FZCB_EDIT_BATCH_SIZE}" == 10 ]]
[[ "${FZCB_EDIT_MODEL}" == llama3-8b-inst || "${FZCB_EDIT_MODEL}" == qwen2.5-7b-inst ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${FZCB_EDIT_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python \
  -m project.run_scripts.fzcb_edit.runtime \
  --model "${FZCB_EDIT_MODEL}" --batch-size "${FZCB_EDIT_BATCH_SIZE}" \
  --source-root "${FZCB_EDIT_SOURCE_ROOT}" --output "${CELL_ROOT}/result.json"

