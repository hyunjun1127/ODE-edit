#!/usr/bin/env bash
# One model/family worker for the capacity/history controller or evaluator phase.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly EXPECTED_SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly RUNTIME_ENV="${REPO_ROOT}/servers/local/method-runtime.env"
readonly SESSION_ENV="${REPO_ROOT}/servers/local/session-boundary.env"
readonly SESSION_CHECK="${REPO_ROOT}/scripts/check-session-boundary.sh"
readonly CHILD_WRAPPER="${REPO_ROOT}/project/run_scripts/session01_capacity_history_server1.sbatch"
readonly CONTROLLER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/capacity_history_controller.py"
readonly EVALUATOR="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/capacity_history_evaluator.py"
readonly ANALYZER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/capacity_history_analysis.py"
readonly CAPACITY_QP="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/capacity_qp.py"
readonly CAPACITY_GEOMETRY="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/capacity_geometry.py"
readonly TARGET_LINEAGE="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/frozen_target_lineage.py"
readonly ALPHA_HISTORY="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/alphaedit_history.py"
readonly ALPHA_FACTORS="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/alphaedit_factors.py"
readonly ALPHA_ADAPTER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/alphaedit_proposal_adapter.py"
readonly SPEC="${REPO_ROOT}/plans/global/2026-08-02-session01-bf-share-magnitude-control-c3-spec.md"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-08-02-session01-bf-share-magnitude-control-c3-preflight.md"
readonly GIT_BIN="/usr/bin/git"
readonly UV_BIN="/usr/local/bin/uv"

load_allowlisted_env() {
  local source_path="$1"
  local allowed_keys="$2"
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9_./:@+-]+)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
    elif [[ "${line}" == "ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE='Sol Ultra'" ]]; then
      key="ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE"
      value="Sol Ultra"
    elif [[ "${line}" == "ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE='Sol Ultra'" ]]; then
      key="ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE"
      value="Sol Ultra"
    else
      echo "environment file contains an unsafe assignment" >&2
      exit 2
    fi
    [[ ",${allowed_keys}," == *",${key},"* ]] || exit 2
    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done <"${source_path}"
}

require_tracked_clean_file() {
  local absolute_path="$1"
  local relative_path actual_blob tracked_blob
  [[ "${absolute_path}" == "${REPO_ROOT}/"* ]] || exit 2
  [[ -f "${absolute_path}" && ! -L "${absolute_path}" && -r "${absolute_path}" ]] || exit 2
  relative_path="${absolute_path#"${REPO_ROOT}/"}"
  "${GIT_BIN}" ls-files --error-unmatch -- "${relative_path}" >/dev/null || exit 2
  actual_blob="$("${GIT_BIN}" hash-object -- "${absolute_path}")"
  tracked_blob="$("${GIT_BIN}" rev-parse "HEAD:${relative_path}")"
  [[ "${actual_blob}" == "${tracked_blob}" ]] || exit 2
  "${GIT_BIN}" diff --quiet -- "${relative_path}" || exit 2
  "${GIT_BIN}" diff --cached --quiet -- "${relative_path}" || exit 2
}

[[ "$#" -eq 3 ]] || {
  echo "capacity/history worker requires phase, family, and model" >&2
  exit 2
}
readonly PHASE="$1"
readonly FAMILY="$2"
readonly MODEL_ALIAS="$3"
[[ "${PHASE}" == "controller" || "${PHASE}" == "evaluator" ]] || exit 2
[[ "${FAMILY}" == "memit" || "${FAMILY}" == "alpha_history" ]] || exit 2
[[ "${SLURM_JOB_NAME:-}" == "odeedit_capacity_history_pair_c3_v1" ]] || exit 2

case "${MODEL_ALIAS}" in
  llama3-8b-inst)
    readonly MEMIT_NATIVE_CONTROLLER="caphist_memit_native_llama_c3_v1"
    readonly MEMIT_QP_CONTROLLER="caphist_memit_qp_llama_c3_v1"
    readonly ALPHA_NATIVE_CONTROLLER="caphist_alpha_native_llama_c3_v1"
    readonly ALPHA_QP_CONTROLLER="caphist_alpha_qp_llama_c3_v1"
    readonly MEMIT_NATIVE_EVALUATOR="caphist_eval_memit_native_llama_c3_v1"
    readonly MEMIT_QP_EVALUATOR="caphist_eval_memit_qp_llama_c3_v1"
    readonly ALPHA_NATIVE_EVALUATOR="caphist_eval_alpha_native_llama_c3_v1"
    readonly ALPHA_QP_EVALUATOR="caphist_eval_alpha_qp_llama_c3_v1"
    ;;
  qwen2.5-7b-inst)
    readonly MEMIT_NATIVE_CONTROLLER="caphist_memit_native_qwen_c3_v1"
    readonly MEMIT_QP_CONTROLLER="caphist_memit_qp_qwen_c3_v1"
    readonly ALPHA_NATIVE_CONTROLLER="caphist_alpha_native_qwen_c3_v1"
    readonly ALPHA_QP_CONTROLLER="caphist_alpha_qp_qwen_c3_v1"
    readonly MEMIT_NATIVE_EVALUATOR="caphist_eval_memit_native_qwen_c3_v1"
    readonly MEMIT_QP_EVALUATOR="caphist_eval_memit_qp_qwen_c3_v1"
    readonly ALPHA_NATIVE_EVALUATOR="caphist_eval_alpha_native_qwen_c3_v1"
    readonly ALPHA_QP_EVALUATOR="caphist_eval_alpha_qp_qwen_c3_v1"
    ;;
  *) exit 2 ;;
esac

case "${FAMILY}" in
  memit)
    readonly BRANCH_ONE="memit_native"
    readonly BRANCH_TWO="memit_capacity_qp"
    readonly CONTROLLER_ONE="${MEMIT_NATIVE_CONTROLLER}"
    readonly CONTROLLER_TWO="${MEMIT_QP_CONTROLLER}"
    readonly EVALUATOR_ONE="${MEMIT_NATIVE_EVALUATOR}"
    readonly EVALUATOR_TWO="${MEMIT_QP_EVALUATOR}"
    ;;
  alpha_history)
    readonly BRANCH_ONE="alpha_native_history"
    readonly BRANCH_TWO="alpha_capacity_qp_history"
    readonly CONTROLLER_ONE="${ALPHA_NATIVE_CONTROLLER}"
    readonly CONTROLLER_TWO="${ALPHA_QP_CONTROLLER}"
    readonly EVALUATOR_ONE="${ALPHA_NATIVE_EVALUATOR}"
    readonly EVALUATOR_TWO="${ALPHA_QP_EVALUATOR}"
    ;;
esac
readonly CONTROLLER_BARRIER="${MEMIT_NATIVE_CONTROLLER}+${MEMIT_QP_CONTROLLER}+${ALPHA_NATIVE_CONTROLLER}+${ALPHA_QP_CONTROLLER}"

[[ -r "${RUNTIME_ENV}" && ! -L "${RUNTIME_ENV}" ]] || exit 2
[[ -r "${SESSION_ENV}" && ! -L "${SESSION_ENV}" ]] || exit 2
[[ -x "${SESSION_CHECK}" && ! -L "${SESSION_CHECK}" ]] || exit 2
[[ -x "${GIT_BIN}" && -x "${UV_BIN}" ]] || exit 2
readonly SLURM_ASSIGNED_CUDA="${CUDA_VISIBLE_DEVICES:-}"

cd "${REPO_ROOT}"
load_allowlisted_env \
  "${RUNTIME_ENV}" \
  "EASYEDIT_ROOT,KR_UV_BIN,HF_HOME,HF_HUB_CACHE,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,PYTHONPATH"
load_allowlisted_env \
  "${SESSION_ENV}" \
  "ODEEDIT_REPOSITORY_ID,ODEEDIT_REPOSITORY_CWD,ODEEDIT_CODEX_SESSION_ID,ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE,ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE,ODEEDIT_CURRENT_SERVER"

[[ "${EASYEDIT_ROOT:-}" == "/mnt/raid5/janghj/EasyEdit" ]] || exit 2
[[ "${KR_UV_BIN:-}" == "uv" ]] || exit 2
[[ "${HF_HOME:-}" == "/mnt/raid5/janghj/.cache/huggingface" ]] || exit 2
[[ "${HF_HUB_CACHE:-}" == "/mnt/raid5/janghj/.cache/huggingface/hub" ]] || exit 2
[[ "${HF_HUB_OFFLINE:-}" == "1" && "${TRANSFORMERS_OFFLINE:-}" == "1" ]] || exit 2
[[ "${PYTHONPATH:-}" == "${EASYEDIT_ROOT}" ]] || exit 2
[[ "${ODEEDIT_REPOSITORY_CWD:-}" == "${REPO_ROOT}" ]] || exit 2
[[ "${ODEEDIT_REPOSITORY_ID:-}" == "hyunjun1127/ODE-edit" ]] || exit 2
[[ "${ODEEDIT_CODEX_SESSION_ID:-}" == "${EXPECTED_SESSION_ID}" ]] || exit 2
[[ "${ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE:-}" == "Sol Ultra" ]] || exit 2
[[ "${ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE:-}" == "Sol Ultra" ]] || exit 2
[[ "${ODEEDIT_CURRENT_SERVER:-}" == "server1" ]] || exit 2
[[ "$("${GIT_BIN}" rev-parse --show-toplevel)" == "${REPO_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || exit 2
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || exit 2
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || exit 2
[[ -n "${SLURM_ASSIGNED_CUDA}" && "${SLURM_ASSIGNED_CUDA}" != *,* ]] || exit 2
[[ -n "${SLURM_STEP_ID:-}" && "${SLURM_STEP_ID}" != "batch" ]] || exit 2

"${SESSION_CHECK}" "${EXPECTED_SESSION_ID}"
for tracked_path in \
  "${CHILD_WRAPPER}" "${CONTROLLER}" "${EVALUATOR}" "${ANALYZER}" \
  "${CAPACITY_QP}" "${CAPACITY_GEOMETRY}" "${TARGET_LINEAGE}" "${ALPHA_HISTORY}" \
  "${ALPHA_FACTORS}" "${ALPHA_ADAPTER}" "${SPEC}"; do
  require_tracked_clean_file "${tracked_path}"
done
require_tracked_clean_file "${RED_GATE}"

readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
mkdir -p "${OUTPUT_ROOT}"
[[ -d "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || exit 2
if [[ "${PHASE}" == "controller" ]]; then
  for run_name in "${CONTROLLER_ONE}" "${CONTROLLER_TWO}"; do
    [[ ! -e "${OUTPUT_ROOT}/${run_name}" && ! -L "${OUTPUT_ROOT}/${run_name}" ]] || exit 2
  done
else
  for run_name in \
    "${MEMIT_NATIVE_CONTROLLER}" "${MEMIT_QP_CONTROLLER}" \
    "${ALPHA_NATIVE_CONTROLLER}" "${ALPHA_QP_CONTROLLER}"; do
    [[ -f "${OUTPUT_ROOT}/${run_name}/summary.json" \
      && ! -L "${OUTPUT_ROOT}/${run_name}/summary.json" ]] || exit 2
  done
  for run_name in "${EVALUATOR_ONE}" "${EVALUATOR_TWO}"; do
    [[ ! -e "${OUTPUT_ROOT}/${run_name}" && ! -L "${OUTPUT_ROOT}/${run_name}" ]] || exit 2
  done
fi

export CUDA_VISIBLE_DEVICES="${SLURM_ASSIGNED_CUDA}"
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
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY OPENAI_API_KEY

run_python() {
  "${UV_BIN}" run \
    --offline \
    --project "${EASYEDIT_ROOT}" \
    --frozen \
    --no-sync \
    python "$@"
}

if [[ "${PHASE}" == "controller" ]]; then
  ODEEDIT_CAPACITY_PROCESS_ROLE=controller run_python \
    -m project.run_scripts.ode_edit_motivation.capacity_history_controller \
    --easyedit-root "${EASYEDIT_ROOT}" \
    --branch "${BRANCH_ONE}" \
    --model "${MODEL_ALIAS}" \
    --run-id "${CONTROLLER_ONE}" \
    --output-root "${OUTPUT_ROOT}"
  ODEEDIT_CAPACITY_PROCESS_ROLE=controller run_python \
    -m project.run_scripts.ode_edit_motivation.capacity_history_controller \
    --easyedit-root "${EASYEDIT_ROOT}" \
    --branch "${BRANCH_TWO}" \
    --model "${MODEL_ALIAS}" \
    --run-id "${CONTROLLER_TWO}" \
    --output-root "${OUTPUT_ROOT}"
else
  ODEEDIT_CAPACITY_PROCESS_ROLE=evaluator \
  ODEEDIT_CAPACITY_CONTROLLER_BARRIER="${CONTROLLER_BARRIER}" \
  run_python \
    -m project.run_scripts.ode_edit_motivation.capacity_history_evaluator run \
    --easyedit-root "${EASYEDIT_ROOT}" \
    --branch "${BRANCH_ONE}" \
    --model "${MODEL_ALIAS}" \
    --run-id "${EVALUATOR_ONE}" \
    --output-root "${OUTPUT_ROOT}"
  ODEEDIT_CAPACITY_PROCESS_ROLE=evaluator \
  ODEEDIT_CAPACITY_CONTROLLER_BARRIER="${CONTROLLER_BARRIER}" \
  run_python \
    -m project.run_scripts.ode_edit_motivation.capacity_history_evaluator run \
    --easyedit-root "${EASYEDIT_ROOT}" \
    --branch "${BRANCH_TWO}" \
    --model "${MODEL_ALIAS}" \
    --run-id "${EVALUATOR_TWO}" \
    --output-root "${OUTPUT_ROOT}"
fi
