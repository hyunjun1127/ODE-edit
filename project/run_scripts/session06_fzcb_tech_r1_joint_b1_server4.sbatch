#!/usr/bin/env bash
# TECH-R1 joint-B1 controller-validity matrix: array0=Llama, array1=Qwen.
#SBATCH --job-name=fzcb_tech_r1_b1_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=90000M
#SBATCH --time=48:00:00

set -euo pipefail
umask 077

: "${FZCB_TECH_R1_SOURCE_ROOT:?}"
: "${FZCB_TECH_R1_EXPECTED_HEAD:?}"
: "${FZCB_TECH_R1_PREFLIGHT:?}"
: "${FZCB_TECH_R1_PREFLIGHT_SHA:?}"
: "${FZCB_TECH_R1_RESULT_ROOT:?}"
: "${FZCB_TECH_R1_CAMPAIGN_ID:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly ENV_PROJECT=/data/janghj/EasyEdit
readonly UV_BIN=/data/janghj/.local/bin/uv

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly MODEL=llama3-8b-inst ;;
  1) readonly MODEL=qwen2.5-7b-inst ;;
  *) exit 64 ;;
esac
readonly CELL_ROOT="${FZCB_TECH_R1_RESULT_ROOT}/${MODEL}-B1"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${FZCB_TECH_R1_SOURCE_ROOT}" rev-parse HEAD)" == "${FZCB_TECH_R1_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${FZCB_TECH_R1_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ -f "${FZCB_TECH_R1_PREFLIGHT}" && ! -L "${FZCB_TECH_R1_PREFLIGHT}" ]]
[[ "$(sha256sum "${FZCB_TECH_R1_PREFLIGHT}" | awk '{print $1}')" == "${FZCB_TECH_R1_PREFLIGHT_SHA}" ]]
[[ "$(jq -r '.source.head' "${FZCB_TECH_R1_PREFLIGHT}")" == "${FZCB_TECH_R1_EXPECTED_HEAD}" ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${FZCB_TECH_R1_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python \
  -m project.run_scripts.fzcb_edit.tech_r1_runtime \
  --model "${MODEL}" --source-root "${FZCB_TECH_R1_SOURCE_ROOT}" \
  --cell-root "${CELL_ROOT}" --campaign-id "${FZCB_TECH_R1_CAMPAIGN_ID}"
