#!/usr/bin/env bash
# Atomic-B10 production pilot: array0=Llama, array1=Qwen.
#SBATCH --job-name=fzcb_b10_prod_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=180000M
#SBATCH --time=48:00:00

set -euo pipefail
umask 077

: "${FZCB_B10_SOURCE_ROOT:?}"
: "${FZCB_B10_EXPECTED_HEAD:?}"
: "${FZCB_B10_PREFLIGHT:?}"
: "${FZCB_B10_PREFLIGHT_SHA:?}"
: "${FZCB_B10_RESULT_ROOT:?}"
: "${FZCB_B10_CAMPAIGN_ID:?}"
: "${FZCB_B10_STAGE:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly ENV_PROJECT=/data/janghj/EasyEdit
readonly UV_BIN=/data/janghj/.local/bin/uv

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly MODEL=llama3-8b-inst ;;
  1) readonly MODEL=qwen2.5-7b-inst ;;
  *) exit 64 ;;
esac
[[ "${FZCB_B10_STAGE}" == canary || "${FZCB_B10_STAGE}" == b10 ]]
readonly CELL_ROOT="${FZCB_B10_RESULT_ROOT}/${MODEL}-${FZCB_B10_STAGE}"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${FZCB_B10_SOURCE_ROOT}" rev-parse HEAD)" == "${FZCB_B10_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${FZCB_B10_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ -f "${FZCB_B10_PREFLIGHT}" && ! -L "${FZCB_B10_PREFLIGHT}" ]]
[[ "$(sha256sum "${FZCB_B10_PREFLIGHT}" | awk '{print $1}')" == "${FZCB_B10_PREFLIGHT_SHA}" ]]
[[ "$(jq -r '.source.head' "${FZCB_B10_PREFLIGHT}")" == "${FZCB_B10_EXPECTED_HEAD}" ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${FZCB_B10_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python \
  -m project.run_scripts.fzcb_edit.production_runtime \
  --stage "${FZCB_B10_STAGE}" --model "${MODEL}" \
  --source-root "${FZCB_B10_SOURCE_ROOT}" --cell-root "${CELL_ROOT}" \
  --campaign-id "${FZCB_B10_CAMPAIGN_ID}"
