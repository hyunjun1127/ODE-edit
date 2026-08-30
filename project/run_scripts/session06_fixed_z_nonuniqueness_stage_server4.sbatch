#!/usr/bin/env bash
# Fresh fixed-z non-uniqueness smoke/screen launcher for server4.
#SBATCH --job-name=fznu_s06_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=180000M
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/data/janghj/ODE-edit/local/logs/fixed-z-nonuniqueness-screen-v1/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/fixed-z-nonuniqueness-screen-v1/%x-%A_%a.err

set -euo pipefail
umask 077

: "${FZNU_SOURCE_ROOT:?}"
: "${FZNU_EXPECTED_HEAD:?}"
: "${FZNU_STAGE:?}"
: "${FZNU_METHOD:?}"
: "${FZNU_RESULT_ROOT:?}"
: "${FZNU_PREFLIGHT_RECEIPT:?}"

readonly EASYEDIT_ROOT="/data/janghj/EasyEdit-stock-14cea824"
readonly EXPECTED_EASYEDIT_HEAD="14cea8245f06715684592ab55184939b99d70784"
readonly ENV_PROJECT="/data/janghj/EasyEdit"
readonly UV_BIN="/data/janghj/.local/bin/uv"
readonly DATASET="/data/janghj/EasyEdit/data/counterfact/counterfact.json"
readonly CASE_MANIFEST="${FZNU_SOURCE_ROOT}/project/run_scripts/fixed_z_nonuniqueness/config/case-manifest-v1.json"
readonly RUNNER="${FZNU_SOURCE_ROOT}/project/run_scripts/fixed_z_nonuniqueness/experiment.py"
readonly MODEL_ALIASES=("llama3-8b-inst" "qwen2.5-7b-inst")
readonly MODEL_ALIAS="${MODEL_ALIASES[${SLURM_ARRAY_TASK_ID}]}"
readonly CELL_DIR="${FZNU_RESULT_ROOT}/${MODEL_ALIAS}-${FZNU_METHOD}-${FZNU_STAGE}"

[[ "$(hostname -s)" == "server4" ]]
[[ "$(git -C "${FZNU_SOURCE_ROOT}" rev-parse HEAD)" == "${FZNU_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${FZNU_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
for path in "${UV_BIN}" "${DATASET}" "${CASE_MANIFEST}" "${RUNNER}" "${FZNU_PREFLIGHT_RECEIPT}"; do
  [[ -e "${path}" && ! -L "${path}" ]]
done
[[ ! -e "${CELL_DIR}" && ! -L "${CELL_DIR}" ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${FZNU_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python -m project.run_scripts.fixed_z_nonuniqueness.experiment \
  --stage "${FZNU_STAGE}" \
  --model "${MODEL_ALIAS}" \
  --method "${FZNU_METHOD}" \
  --source-root "${FZNU_SOURCE_ROOT}" \
  --dataset "${DATASET}" \
  --case-manifest "${CASE_MANIFEST}" \
  --output "${CELL_DIR}/result.json"
