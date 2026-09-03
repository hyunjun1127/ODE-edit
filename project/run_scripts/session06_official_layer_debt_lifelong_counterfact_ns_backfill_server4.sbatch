#!/usr/bin/env bash
# Evaluation-only CounterFact locality target-new backfill; no edit replay.
#SBATCH --job-name=lrd_cfns_v6_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=60416M
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/data/janghj/ODE-edit/local/logs/official-layer-debt-lifelong-counterfact-v6/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/official-layer-debt-lifelong-counterfact-v6/%x-%A_%a.err

set -euo pipefail
umask 077

: "${ODE_LRD_CF_SOURCE_ROOT:?}"
: "${ODE_LRD_CF_EXPECTED_HEAD:?}"
: "${ODE_LRD_CF_AVAILABILITY_AUDIT:?}"
: "${ODE_LRD_CF_EXPECTED_AUDIT_SHA:?}"
: "${ODE_LRD_CF_RESULT_ROOT:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly UV_BIN=/data/janghj/.local/bin/uv
readonly ENV_PROJECT=/data/janghj/EasyEdit
readonly CELL_ROOT="${ODE_LRD_CF_RESULT_ROOT}/cell-${SLURM_ARRAY_TASK_ID}"
readonly RUNNER="${ODE_LRD_CF_SOURCE_ROOT}/project/run_scripts/official_layer_realization_debt/lifelong_counterfact_runtime.py"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${ODE_LRD_CF_SOURCE_ROOT}" rev-parse HEAD)" == "${ODE_LRD_CF_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${ODE_LRD_CF_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ -f "${ODE_LRD_CF_AVAILABILITY_AUDIT}" && ! -L "${ODE_LRD_CF_AVAILABILITY_AUDIT}" ]]
[[ "$(sha256sum "${ODE_LRD_CF_AVAILABILITY_AUDIT}" | awk '{print $1}')" == "${ODE_LRD_CF_EXPECTED_AUDIT_SHA}" ]]
[[ -f "${RUNNER}" && ! -L "${RUNNER}" ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

python3 "${ODE_LRD_CF_SOURCE_ROOT}/scripts/slurm_memory_policy.py" request \
  --server server4 --gpus 1 --mem 60416M --local-limit-mib-per-gpu 60416

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${ODE_LRD_CF_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${ODE_LRD_CF_SOURCE_ROOT}"
exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python \
  -m project.run_scripts.official_layer_realization_debt.lifelong_counterfact_runtime \
  --source-root "${ODE_LRD_CF_SOURCE_ROOT}" \
  --expected-head "${ODE_LRD_CF_EXPECTED_HEAD}" \
  --availability-audit "${ODE_LRD_CF_AVAILABILITY_AUDIT}" \
  --expected-audit-sha "${ODE_LRD_CF_EXPECTED_AUDIT_SHA}" \
  --cell-index "${SLURM_ARRAY_TASK_ID}" \
  --cell-root "${CELL_ROOT}"
