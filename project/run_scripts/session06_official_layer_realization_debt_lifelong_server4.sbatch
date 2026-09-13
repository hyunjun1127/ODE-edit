#!/usr/bin/env bash
# Official B100 lifelong layer action-realization observation, cap2 wave launcher.
#SBATCH --job-name=off_layer_debt_life_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=120000M
#SBATCH --time=7-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/data/janghj/ODE-edit/local/logs/official-layer-realization-debt-lifelong-b100-v1/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/official-layer-realization-debt-lifelong-b100-v1/%x-%A_%a.err

set -euo pipefail
umask 077

: "${ODE_LRD_LIFE_SOURCE_ROOT:?}"
: "${ODE_LRD_LIFE_EXPECTED_HEAD:?}"
: "${ODE_LRD_LIFE_RESULT_ROOT:?}"
: "${ODE_LRD_LIFE_PREFLIGHT_RECEIPT:?}"
: "${ODE_LRD_LIFE_EXPECTED_PREFLIGHT_SHA:?}"
: "${ODE_LRD_LIFE_STREAM_SEAL:?}"
: "${ODE_LRD_LIFE_EXPECTED_STREAM_SHA:?}"
: "${ODE_LRD_LIFE_CAMPAIGN_ID:?}"
: "${ODE_LRD_LIFE_WAVE:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly UV_BIN=/data/janghj/.local/bin/uv
readonly ENV_PROJECT=/data/janghj/EasyEdit

case "${ODE_LRD_LIFE_WAVE}" in
  1)
    readonly MODELS=(qwen2.5-7b-inst qwen2.5-7b-inst)
    readonly METHODS=(alphaedit memit)
    ;;
  2)
    readonly MODELS=(llama3-8b-inst llama3-8b-inst)
    readonly METHODS=(alphaedit memit)
    ;;
  *)
    echo "invalid ODE_LRD_LIFE_WAVE=${ODE_LRD_LIFE_WAVE}" >&2
    exit 2
    ;;
esac

readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"
readonly CELL_ROOT="${ODE_LRD_LIFE_RESULT_ROOT}/${MODEL}-${METHOD}-lifelong-b100x100"
readonly RUNNER="${ODE_LRD_LIFE_SOURCE_ROOT}/project/run_scripts/official_layer_realization_debt/lifelong_runtime.py"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${ODE_LRD_LIFE_SOURCE_ROOT}" rev-parse HEAD)" == "${ODE_LRD_LIFE_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${ODE_LRD_LIFE_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ -f "${ODE_LRD_LIFE_PREFLIGHT_RECEIPT}" && ! -L "${ODE_LRD_LIFE_PREFLIGHT_RECEIPT}" ]]
[[ "$(sha256sum "${ODE_LRD_LIFE_PREFLIGHT_RECEIPT}" | awk '{print $1}')" == "${ODE_LRD_LIFE_EXPECTED_PREFLIGHT_SHA}" ]]
[[ -f "${ODE_LRD_LIFE_STREAM_SEAL}" && ! -L "${ODE_LRD_LIFE_STREAM_SEAL}" ]]
[[ "$(sha256sum "${ODE_LRD_LIFE_STREAM_SEAL}" | awk '{print $1}')" == "${ODE_LRD_LIFE_EXPECTED_STREAM_SHA}" ]]
[[ -f "${RUNNER}" && ! -L "${RUNNER}" ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${ODE_LRD_LIFE_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python \
  -m project.run_scripts.official_layer_realization_debt.lifelong_runtime \
  --model "${MODEL}" \
  --method "${METHOD}" \
  --source-root "${ODE_LRD_LIFE_SOURCE_ROOT}" \
  --expected-head "${ODE_LRD_LIFE_EXPECTED_HEAD}" \
  --cell-root "${CELL_ROOT}" \
  --campaign-id "${ODE_LRD_LIFE_CAMPAIGN_ID}" \
  --stream-seal "${ODE_LRD_LIFE_STREAM_SEAL}"
