#!/usr/bin/env bash
# Read-only observer over stock Official MEMIT/AlphaEdit layer writes.
#SBATCH --job-name=official_layer_debt_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=64000M
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/janghj/ODE-edit/local/logs/official-layer-realization-debt-b10x10-v1/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/official-layer-realization-debt-b10x10-v1/%x-%A_%a.err

set -euo pipefail
umask 077

: "${ODE_LRD_SOURCE_ROOT:?}"
: "${ODE_LRD_EXPECTED_HEAD:?}"
: "${ODE_LRD_STAGE:?}"
: "${ODE_LRD_RESULT_ROOT:?}"
: "${ODE_LRD_PREFLIGHT_RECEIPT:?}"
: "${ODE_LRD_EXPECTED_PREFLIGHT_SHA:?}"
: "${ODE_LRD_CAMPAIGN_ID:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly UV_BIN=/data/janghj/.local/bin/uv
readonly ENV_PROJECT=/data/janghj/EasyEdit
readonly MODELS=(llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst)
readonly METHODS=(memit alphaedit memit alphaedit)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"
readonly CELL_ROOT="${ODE_LRD_RESULT_ROOT}/${MODEL}-${METHOD}-${ODE_LRD_STAGE}"
readonly RUNNER="${ODE_LRD_SOURCE_ROOT}/project/run_scripts/official_layer_realization_debt/runtime.py"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${ODE_LRD_SOURCE_ROOT}" rev-parse HEAD)" == "${ODE_LRD_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${ODE_LRD_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ -f "${ODE_LRD_PREFLIGHT_RECEIPT}" && ! -L "${ODE_LRD_PREFLIGHT_RECEIPT}" ]]
[[ "$(sha256sum "${ODE_LRD_PREFLIGHT_RECEIPT}" | awk '{print $1}')" == "${ODE_LRD_EXPECTED_PREFLIGHT_SHA}" ]]
[[ -f "${RUNNER}" && ! -L "${RUNNER}" ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

EXTRA_ARGS=()
if [[ "${ODE_LRD_STAGE}" == b10x10 ]]; then
  : "${ODE_LRD_B1_ROOT:?}"
  readonly B1_RESULT="${ODE_LRD_B1_ROOT}/${MODEL}-${METHOD}-b1/result.json"
  [[ -f "${B1_RESULT}" && ! -L "${B1_RESULT}" ]]
  EXTRA_ARGS+=(--b1-result "${B1_RESULT}")
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${ODE_LRD_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python \
  -m project.run_scripts.official_layer_realization_debt.runtime \
  --stage "${ODE_LRD_STAGE}" \
  --model "${MODEL}" \
  --method "${METHOD}" \
  --source-root "${ODE_LRD_SOURCE_ROOT}" \
  --expected-head "${ODE_LRD_EXPECTED_HEAD}" \
  --cell-root "${CELL_ROOT}" \
  --campaign-id "${ODE_LRD_CAMPAIGN_ID}" \
  "${EXTRA_ARGS[@]}"
