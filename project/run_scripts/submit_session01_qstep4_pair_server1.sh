#!/usr/bin/env bash
# Fail-closed helper for the single approved pre-run env-parser repair retry.

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
readonly SACCT_BIN="/usr/bin/sacct"
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_qstep4_pair_server1.sbatch"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-07-31-session01-quarter-step-execution-preflight.md"
readonly REPAIR_GATE="${REPO_ROOT}/audits/global/2026-07-31-session01-quarter-step-env-parser-repair-preflight.md"
readonly RED_GATE_VERDICT='- 최종 판정: `PASS` — locked quarter-step Llama/Qwen 동시 pair 한 건에만 유효'
readonly REPAIR_GATE_VERDICT='- 최종 판정: `PASS` — quarter-step pre-run env-parser repair retry 한 건에만 유효'
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly ORIGINAL_PAIR_MARKER="${STATE_ROOT}/qstep4_pair_v1.submitted"
readonly PAIR_MARKER="${STATE_ROOT}/qstep4_pair_v1_retry1.submitted"
readonly JOB_NAME="odeedit_qstep_pair_v1"

[[ "$#" -eq 0 ]] || {
  echo "quarter-step pair helper takes no arguments" >&2
  exit 2
}
[[ -x "${GIT_BIN}" \
  && -x "${GREP_BIN}" \
  && -x "${INSTALL_BIN}" \
  && -x "${MKDIR_BIN}" \
  && -x "${CHMOD_BIN}" \
  && -x "${SBATCH_BIN}" \
  && -x "${SQUEUE_BIN}" \
  && -x "${SACCT_BIN}" ]] || exit 2
[[ -f "${SBATCH_FILE}" && ! -L "${SBATCH_FILE}" && -r "${SBATCH_FILE}" ]] || exit 2
[[ -f "${RED_GATE}" && ! -L "${RED_GATE}" && -r "${RED_GATE}" ]] || exit 2
[[ -f "${REPAIR_GATE}" && ! -L "${REPAIR_GATE}" && -r "${REPAIR_GATE}" ]] || exit 2

cd "${REPO_ROOT}"
[[ "$("${GIT_BIN}" rev-parse --show-toplevel)" == "${REPO_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.id)" == "head-server1-gh" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.role)" == "global-head" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.hostname)" == "server1" ]] || exit 2
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || exit 2

"${GIT_BIN}" ls-files --error-unmatch \
  project/run_scripts/session01_qstep4_server1.sbatch \
  project/run_scripts/session01_qstep4_pair_server1.sbatch \
  project/run_scripts/submit_session01_qstep4_pair_server1.sh \
  project/run_scripts/ode_edit_motivation/easyedit_bridge.py \
  project/run_scripts/ode_edit_motivation/hooks.py \
  project/run_scripts/ode_edit_motivation/frozen_target_lineage.py \
  project/run_scripts/ode_edit_motivation/trajectory.py \
  project/run_scripts/ode_edit_motivation/quarter_step_refresh.py \
  project/run_scripts/ode_edit_motivation/quarter_step_analysis.py \
  project/run_scripts/ode_edit_motivation/tests/test_frozen_target_lineage.py \
  project/run_scripts/ode_edit_motivation/tests/test_trajectory.py \
  project/run_scripts/ode_edit_motivation/tests/test_quarter_step_refresh.py \
  project/run_scripts/ode_edit_motivation/tests/test_quarter_step_analysis.py \
  plans/global/2026-07-31-session01-quarter-step-implementation-spec.md \
  audits/global/2026-07-31-session01-quarter-step-execution-preflight.md \
  audits/global/2026-07-31-session01-quarter-step-env-parser-repair-preflight.md \
  messages/inbox/server1.md \
  >/dev/null
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "quarter-step submission requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "quarter-step submission requires main == origin/main" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Fxc -- "${RED_GATE_VERDICT}" "${RED_GATE}" || true)" == "1" ]] || {
  echo "quarter-step exact red-gate verdict is unavailable" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${RED_GATE}" || true)" == "1" ]] || {
  echo "quarter-step red-gate verdict is ambiguous" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Fxc -- "${REPAIR_GATE_VERDICT}" "${REPAIR_GATE}" || true)" == "1" ]] || {
  echo "quarter-step repair exact verdict is unavailable" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${REPAIR_GATE}" || true)" == "1" ]] || {
  echo "quarter-step repair verdict is ambiguous" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/qstep4_llama_f0_v1" \
  && ! -L "${OUTPUT_ROOT}/qstep4_llama_f0_v1" ]] || {
  echo "Llama quarter-step output already exists" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/qstep4_qwen_f0_v1" \
  && ! -L "${OUTPUT_ROOT}/qstep4_qwen_f0_v1" ]] || {
  echo "Qwen quarter-step output already exists" >&2
  exit 2
}
[[ -f "${ORIGINAL_PAIR_MARKER}/job-id" \
  && ! -L "${ORIGINAL_PAIR_MARKER}" \
  && ! -L "${ORIGINAL_PAIR_MARKER}/job-id" ]] || {
  echo "original quarter-step marker is unavailable" >&2
  exit 2
}
original_job_id="$(<"${ORIGINAL_PAIR_MARKER}/job-id")"
[[ "${original_job_id}" =~ ^[0-9]+$ ]] || {
  echo "original quarter-step job ID is malformed" >&2
  exit 2
}
original_state=""
read -r original_state < <(
  "${SACCT_BIN}" -n -X -j "${original_job_id}" --format=State --noheader
)
[[ "${original_state}" == "FAILED" ]] || {
  echo "original quarter-step job is not exact FAILED state" >&2
  exit 2
}
[[ ! -e "${PAIR_MARKER}" && ! -L "${PAIR_MARKER}" ]] || {
  echo "quarter-step repair retry marker already exists" >&2
  exit 2
}
if [[ -n "$("${SQUEUE_BIN}" -h --name="${JOB_NAME}")" ]]; then
  echo "quarter-step pair job is already active" >&2
  exit 2
fi

"${REPO_ROOT}/scripts/check-session-boundary.sh" "${SESSION_ID}"
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 2 130000
"${INSTALL_BIN}" -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
"${MKDIR_BIN}" -m 700 "${PAIR_MARKER}" || exit 2

job_id="$("${SBATCH_BIN}" --parsable --export=NONE "${SBATCH_FILE}")"
printf '%s\n' "${job_id}" >"${PAIR_MARKER}/job-id"
"${CHMOD_BIN}" 600 "${PAIR_MARKER}/job-id"
printf '%s\n' "${job_id}"
