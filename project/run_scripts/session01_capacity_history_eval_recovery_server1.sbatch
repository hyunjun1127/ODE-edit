#!/usr/bin/env bash
# One model/family evaluator-only worker bound to the completed v3 controllers.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly LOCKED_ROOT="${REPO_ROOT}/local/scratch/caphist-eval-9900a51"
readonly LOCKED_COMMIT="9900a51f16669447bf502e7df0d5d5e0d61ef360"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly OUTPUT_ROOT="${LOCKED_ROOT}/local/results/raw/session01_motivation"
readonly EXPECTED_SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly SESSION_CHECK="${REPO_ROOT}/scripts/check-session-boundary.sh"
readonly UV_BIN="/usr/local/bin/uv"
readonly GIT_BIN="/usr/bin/git"

[[ "$#" -eq 2 ]] || exit 2
readonly FAMILY="$1"
readonly MODEL_ALIAS="$2"
[[ "${FAMILY}" == "memit" || "${FAMILY}" == "alpha_history" ]] || exit 2
[[ "${MODEL_ALIAS}" == "llama3-8b-inst" || "${MODEL_ALIAS}" == "qwen2.5-7b-inst" ]] || exit 2
[[ "${SLURM_JOB_NAME:-}" == "odeedit_capacity_history_pair_v3" ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || exit 2
[[ -n "${SLURM_STEP_ID:-}" && "${SLURM_STEP_ID}" != "batch" ]] || exit 2
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || exit 2
[[ -x "${UV_BIN}" && -x "${GIT_BIN}" && -x "${SESSION_CHECK}" ]] || exit 2
[[ -d "${LOCKED_ROOT}" && ! -L "${LOCKED_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" -C "${LOCKED_ROOT}" rev-parse HEAD)" == "${LOCKED_COMMIT}" ]] || exit 2
[[ -z "$("${GIT_BIN}" -C "${LOCKED_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 2
[[ -z "$("${GIT_BIN}" -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] || exit 2
[[ "$("${GIT_BIN}" -C "${REPO_ROOT}" rev-parse HEAD)" == "$("${GIT_BIN}" -C "${REPO_ROOT}" rev-parse origin/main)" ]] || exit 2
"${SESSION_CHECK}" "${EXPECTED_SESSION_ID}"

case "${MODEL_ALIAS}" in
  llama3-8b-inst)
    readonly MODEL_TAG="llama"
    ;;
  qwen2.5-7b-inst)
    readonly MODEL_TAG="qwen"
    ;;
esac

if [[ "${FAMILY}" == "memit" ]]; then
  readonly BRANCH_ONE="memit_native"
  readonly BRANCH_TWO="memit_capacity_qp"
  readonly EVALUATOR_ONE="caphist_eval_memit_native_${MODEL_TAG}_c0_v3"
  readonly EVALUATOR_TWO="caphist_eval_memit_qp_${MODEL_TAG}_c0_v3"
else
  readonly BRANCH_ONE="alpha_native_history"
  readonly BRANCH_TWO="alpha_capacity_qp_history"
  readonly EVALUATOR_ONE="caphist_eval_alpha_native_${MODEL_TAG}_c0_v3"
  readonly EVALUATOR_TWO="caphist_eval_alpha_qp_${MODEL_TAG}_c0_v3"
fi

readonly CONTROLLER_BARRIER="caphist_memit_native_${MODEL_TAG}_c0_v3+caphist_memit_qp_${MODEL_TAG}_c0_v3+caphist_alpha_native_${MODEL_TAG}_c0_v3+caphist_alpha_qp_${MODEL_TAG}_c0_v3"
for run_name in "${EVALUATOR_ONE}" "${EVALUATOR_TWO}"; do
  [[ ! -e "${OUTPUT_ROOT}/${run_name}" && ! -L "${OUTPUT_ROOT}/${run_name}" ]] || exit 2
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export UV_OFFLINE=1
export UV_NO_SYNC=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${LOCKED_ROOT}:${EASYEDIT_ROOT}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY OPENAI_API_KEY

run_evaluator() {
  local branch="$1"
  local run_id="$2"
  (
    cd "${LOCKED_ROOT}"
    ODEEDIT_CAPACITY_PROCESS_ROLE=evaluator \
    ODEEDIT_CAPACITY_CONTROLLER_BARRIER="${CONTROLLER_BARRIER}" \
      "${UV_BIN}" run \
      --offline \
      --project "${EASYEDIT_ROOT}" \
      --frozen \
      --no-sync \
      python -m project.run_scripts.ode_edit_motivation.capacity_history_evaluator run \
      --easyedit-root "${EASYEDIT_ROOT}" \
      --branch "${branch}" \
      --model "${MODEL_ALIAS}" \
      --run-id "${run_id}" \
      --output-root "${OUTPUT_ROOT}"
  )
}

run_evaluator "${BRANCH_ONE}" "${EVALUATOR_ONE}"
run_evaluator "${BRANCH_TWO}" "${EVALUATOR_TWO}"
