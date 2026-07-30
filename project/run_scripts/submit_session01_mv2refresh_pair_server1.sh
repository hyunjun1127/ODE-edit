#!/usr/bin/env bash
# Fail-closed helper for the single approved technical retry of paired MV-2.

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
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_mv2refresh_pair_server1.sbatch"
readonly MV1_POSTRUN="${REPO_ROOT}/audits/global/2026-07-31-mv1mix-untouched-pair-v1.postrun.md"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-07-31-session01-mv2-execution-preflight.md"
readonly RETRY_GATE="${REPO_ROOT}/audits/global/2026-07-31-session01-mv2-technical-retry-preflight.md"
readonly MV1_POSTRUN_VERDICT='- 최종 판정: `MV2 PREPARE` — MV-1 untouched pair reproduced; locked MV-2 pair 한 건 허용'
readonly RED_GATE_VERDICT='- 최종 판정: `PASS` — locked MV-2 Llama/Qwen 동시 pair 한 건에만 유효'
readonly RETRY_GATE_VERDICT='- 최종 판정: `PASS` — locked MV-2 pair의 technical retry 1회에만 유효'
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly ORIGINAL_PAIR_MARKER="${STATE_ROOT}/mv2refresh_pair_v1.submitted"
readonly PAIR_MARKER="${STATE_ROOT}/mv2refresh_pair_v1_retry1.submitted"
readonly JOB_NAME="odeedit_mv2refresh_pair_v1"

require_exact_final_verdict() {
  local audit_path="$1"
  local expected_verdict="$2"
  local label="$3"
  local expected_count final_count
  [[ -f "${audit_path}" && ! -L "${audit_path}" && -r "${audit_path}" ]] || {
    echo "${label} audit is unavailable" >&2
    exit 2
  }
  expected_count="$("${GREP_BIN}" -Fxc -- "${expected_verdict}" "${audit_path}" || true)"
  final_count="$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${audit_path}" || true)"
  [[ "${expected_count}" == "1" && "${final_count}" == "1" ]] || {
    echo "${label} exact final verdict is unavailable or ambiguous" >&2
    exit 2
  }
}

[[ "$#" -eq 0 ]] || {
  echo "this paired MV-2 helper takes no arguments" >&2
  exit 2
}
[[ -x "${GIT_BIN}" \
  && -x "${GREP_BIN}" \
  && -x "${INSTALL_BIN}" \
  && -x "${MKDIR_BIN}" \
  && -x "${CHMOD_BIN}" \
  && -x "${SBATCH_BIN}" \
  && -x "${SQUEUE_BIN}" ]] || {
  echo "required fixed system command is unavailable" >&2
  exit 2
}
[[ -f "${SBATCH_FILE}" && ! -L "${SBATCH_FILE}" && -r "${SBATCH_FILE}" ]] || {
  echo "locked MV-2 pair wrapper is unavailable" >&2
  exit 2
}

cd "${REPO_ROOT}"
[[ "$("${GIT_BIN}" rev-parse --show-toplevel)" == "${REPO_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.id)" == "head-server1-gh" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.role)" == "global-head" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.hostname)" == "server1" ]] || exit 2
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || exit 2

"${GIT_BIN}" ls-files --error-unmatch \
  project/run_scripts/session01_mv2refresh_server1.sbatch \
  project/run_scripts/session01_mv2refresh_pair_server1.sbatch \
  project/run_scripts/submit_session01_mv2refresh_pair_server1.sh \
  project/run_scripts/ode_edit_motivation/easyedit_bridge.py \
  project/run_scripts/ode_edit_motivation/frozen_target_lineage.py \
  project/run_scripts/ode_edit_motivation/trajectory.py \
  project/run_scripts/ode_edit_motivation/mv2_refresh.py \
  project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py \
  project/run_scripts/ode_edit_motivation/tests/test_frozen_target_lineage.py \
  project/run_scripts/ode_edit_motivation/tests/test_trajectory.py \
  project/run_scripts/ode_edit_motivation/tests/test_mv2_refresh.py \
  project/run_scripts/ode_edit_motivation/tests/test_mv2_refresh_analysis.py \
  plans/global/2026-07-31-session01-mv2-implementation-spec.md \
  audits/global/2026-07-31-mv1mix-untouched-pair-v1.postrun.md \
  audits/global/2026-07-31-session01-mv2-execution-preflight.md \
  audits/global/2026-07-31-session01-mv2-technical-retry-preflight.md \
  >/dev/null
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "paired MV-2 submission requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "paired MV-2 submission requires main == origin/main" >&2
  exit 2
}

require_exact_final_verdict \
  "${MV1_POSTRUN}" \
  "${MV1_POSTRUN_VERDICT}" \
  "MV-1 untouched postrun"
require_exact_final_verdict \
  "${RED_GATE}" \
  "${RED_GATE_VERDICT}" \
  "MV-2 execution preflight"
require_exact_final_verdict \
  "${RETRY_GATE}" \
  "${RETRY_GATE_VERDICT}" \
  "MV-2 technical retry preflight"

[[ ! -e "${OUTPUT_ROOT}/mv2refresh_llama_e0_v1" \
  && ! -L "${OUTPUT_ROOT}/mv2refresh_llama_e0_v1" ]] || {
  echo "Llama MV-2 output already exists" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/mv2refresh_qwen_e0_v1" \
  && ! -L "${OUTPUT_ROOT}/mv2refresh_qwen_e0_v1" ]] || {
  echo "Qwen MV-2 output already exists" >&2
  exit 2
}
[[ -f "${ORIGINAL_PAIR_MARKER}/job-id" \
  && ! -L "${ORIGINAL_PAIR_MARKER}" \
  && ! -L "${ORIGINAL_PAIR_MARKER}/job-id" ]] || {
  echo "original MV-2 submission marker is unavailable" >&2
  exit 2
}
[[ "$(<"${ORIGINAL_PAIR_MARKER}/job-id")" =~ ^[0-9]+$ ]] || {
  echo "original MV-2 job ID is malformed" >&2
  exit 2
}
[[ ! -e "${PAIR_MARKER}" && ! -L "${PAIR_MARKER}" ]] || {
  echo "MV-2 technical-retry marker already exists" >&2
  exit 2
}
if [[ -n "$("${SQUEUE_BIN}" -h --name="${JOB_NAME}")" ]]; then
  echo "paired MV-2 job is already active" >&2
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
