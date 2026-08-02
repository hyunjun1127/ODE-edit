#!/usr/bin/env bash
# One-shot GH submission helper for the locked capacity/history pair.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly GIT_BIN="/usr/bin/git"
readonly GREP_BIN="/usr/bin/grep"
readonly INSTALL_BIN="/usr/bin/install"
readonly MKDIR_BIN="/usr/bin/mkdir"
readonly CHMOD_BIN="/usr/bin/chmod"
readonly SBATCH_BIN="/usr/bin/sbatch"
readonly SQUEUE_BIN="/usr/bin/squeue"
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_capacity_history_pair_server1.sbatch"
readonly CHILD_FILE="${REPO_ROOT}/project/run_scripts/session01_capacity_history_server1.sbatch"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-08-02-session01-capacity-history-matched-quarter-c1-preflight.md"
readonly RED_GATE_VERDICT='- 최종 판정: `PASS` — clean pushed main에서 c1 4-GPU pair 1회 제출에만 유효'
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly PAIR_MARKER="${STATE_ROOT}/caphist_pair_c1_v1.submitted"
readonly JOB_NAME="odeedit_capacity_history_pair_c1_v1"

[[ "$#" -eq 0 ]] || exit 2
[[ -x "${GIT_BIN}" && -x "${GREP_BIN}" && -x "${INSTALL_BIN}" \
  && -x "${MKDIR_BIN}" && -x "${CHMOD_BIN}" && -x "${SBATCH_BIN}" \
  && -x "${SQUEUE_BIN}" ]] || exit 2
[[ -f "${SBATCH_FILE}" && ! -L "${SBATCH_FILE}" && -r "${SBATCH_FILE}" ]] || exit 2
[[ -f "${CHILD_FILE}" && ! -L "${CHILD_FILE}" && -x "${CHILD_FILE}" ]] || exit 2
[[ -f "${RED_GATE}" && ! -L "${RED_GATE}" && -r "${RED_GATE}" ]] || exit 2

cd "${REPO_ROOT}"
[[ "$("${GIT_BIN}" rev-parse --show-toplevel)" == "${REPO_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.id)" == "head-server1-gh" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.role)" == "global-head" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.hostname)" == "server1" ]] || exit 2
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || exit 2

"${GIT_BIN}" ls-files --error-unmatch \
  project/run_scripts/session01_capacity_history_server1.sbatch \
  project/run_scripts/session01_capacity_history_pair_server1.sbatch \
  project/run_scripts/submit_session01_capacity_history_pair_server1.sh \
  project/run_scripts/ode_edit_motivation/capacity_history_controller.py \
  project/run_scripts/ode_edit_motivation/capacity_history_evaluator.py \
  project/run_scripts/ode_edit_motivation/capacity_history_analysis.py \
  project/run_scripts/ode_edit_motivation/capacity_qp.py \
  project/run_scripts/ode_edit_motivation/capacity_geometry.py \
  project/run_scripts/ode_edit_motivation/alphaedit_history.py \
  project/run_scripts/ode_edit_motivation/alphaedit_factors.py \
  project/run_scripts/ode_edit_motivation/alphaedit_proposal_adapter.py \
  plans/global/2026-08-02-session01-capacity-history-matched-quarter-c1-spec.md \
  audits/global/2026-08-02-session01-capacity-history-matched-quarter-c1-preflight.md \
  messages/inbox/server1.md \
  >/dev/null
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "capacity/history submission requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "capacity/history submission requires main == origin/main" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Fxc -- "${RED_GATE_VERDICT}" "${RED_GATE}" || true)" == "1" ]] || exit 2
[[ "$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${RED_GATE}" || true)" == "1" ]] || exit 2

for run_name in \
  caphist_memit_native_llama_c1_v1 caphist_memit_native_qwen_c1_v1 \
  caphist_memit_qp_llama_c1_v1 caphist_memit_qp_qwen_c1_v1 \
  caphist_alpha_native_llama_c1_v1 caphist_alpha_native_qwen_c1_v1 \
  caphist_alpha_qp_llama_c1_v1 caphist_alpha_qp_qwen_c1_v1 \
  caphist_eval_memit_native_llama_c1_v1 caphist_eval_memit_native_qwen_c1_v1 \
  caphist_eval_memit_qp_llama_c1_v1 caphist_eval_memit_qp_qwen_c1_v1 \
  caphist_eval_alpha_native_llama_c1_v1 caphist_eval_alpha_native_qwen_c1_v1 \
  caphist_eval_alpha_qp_llama_c1_v1 caphist_eval_alpha_qp_qwen_c1_v1 \
  caphist_combined_llama_c1_v1 caphist_combined_qwen_c1_v1 caphist_pair_c1_v1; do
  [[ ! -e "${OUTPUT_ROOT}/${run_name}" && ! -L "${OUTPUT_ROOT}/${run_name}" ]] || exit 2
done
[[ ! -e "${PAIR_MARKER}" && ! -L "${PAIR_MARKER}" ]] || exit 2
[[ -z "$("${SQUEUE_BIN}" -h --name="${JOB_NAME}")" ]] || exit 2

"${REPO_ROOT}/scripts/check-session-boundary.sh" "${SESSION_ID}"
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 4 260000
"${INSTALL_BIN}" -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
"${MKDIR_BIN}" -m 700 "${PAIR_MARKER}" || exit 2

job_id="$("${SBATCH_BIN}" --parsable --export=NONE "${SBATCH_FILE}")"
printf '%s\n' "${job_id}" >"${PAIR_MARKER}/job-id"
"${CHMOD_BIN}" 600 "${PAIR_MARKER}/job-id"
printf '%s\n' "${job_id}"
