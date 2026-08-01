#!/usr/bin/env bash
# One-shot fail-closed submission helper for the direct-z possibility pair.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly SESSION_ID="019fbb9a-e810-7330-ac55-5b72b7c24337"
readonly GIT_BIN="/usr/bin/git"
readonly GREP_BIN="/usr/bin/grep"
readonly INSTALL_BIN="/usr/bin/install"
readonly MKDIR_BIN="/usr/bin/mkdir"
readonly CHMOD_BIN="/usr/bin/chmod"
readonly SBATCH_BIN="/usr/bin/sbatch"
readonly SQUEUE_BIN="/usr/bin/squeue"
readonly WC_BIN="/usr/bin/wc"
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_direct_z_possibility_pair_server1.sbatch"
readonly CHILD_FILE="${REPO_ROOT}/project/run_scripts/session01_direct_z_possibility_server1.sbatch"
readonly SUBMIT_FILE="${REPO_ROOT}/project/run_scripts/submit_session01_direct_z_possibility_pair_server1.sh"
readonly RUNNER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/direct_z_possibility.py"
readonly SPEC="${REPO_ROOT}/plans/global/2026-08-01-session01-direct-z-possibility-spec.md"
readonly PREFLIGHT="${REPO_ROOT}/audits/global/2026-08-01-session01-direct-z-possibility-preflight.md"
readonly SESSION_ENV="${REPO_ROOT}/servers/local/session-boundaries/direct-z-019fbb9a.env"
readonly PREFLIGHT_VERDICT='- 최종 판정: `PASS` — user-authorized direct-z possibility pair 1회에만 유효'
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly LLAMA_RUN="dzf_llama_p0_v1"
readonly QWEN_RUN="dzf_qwen_p0_v1"
readonly PAIR_MARKER="${STATE_ROOT}/dzf_pair_v1.submitted"
readonly JOB_NAME="odeedit_dzf_pair_v1"

[[ "$#" -eq 0 ]] || {
  echo "direct-z possibility pair helper takes no arguments" >&2
  exit 2
}
[[ -x "${GIT_BIN}" \
  && -x "${GREP_BIN}" \
  && -x "${INSTALL_BIN}" \
  && -x "${MKDIR_BIN}" \
  && -x "${CHMOD_BIN}" \
  && -x "${SBATCH_BIN}" \
  && -x "${SQUEUE_BIN}" \
  && -x "${WC_BIN}" ]] || exit 2
[[ -f "${SBATCH_FILE}" && ! -L "${SBATCH_FILE}" && -r "${SBATCH_FILE}" ]] || exit 2
[[ -f "${CHILD_FILE}" && ! -L "${CHILD_FILE}" && -r "${CHILD_FILE}" ]] || exit 2
[[ -f "${SUBMIT_FILE}" && ! -L "${SUBMIT_FILE}" && -r "${SUBMIT_FILE}" ]] || exit 2
[[ -f "${SPEC}" && ! -L "${SPEC}" && -r "${SPEC}" ]] || exit 2
[[ -f "${PREFLIGHT}" && ! -L "${PREFLIGHT}" && -r "${PREFLIGHT}" ]] || exit 2
[[ -f "${SESSION_ENV}" && ! -L "${SESSION_ENV}" && -r "${SESSION_ENV}" ]] || exit 2

cd "${REPO_ROOT}"
[[ "$("${GIT_BIN}" rev-parse --show-toplevel)" == "${REPO_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.id)" == "head-server1-gh" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.role)" == "global-head" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.hostname)" == "server1" ]] || exit 2
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || exit 2

"${GIT_BIN}" ls-files --error-unmatch \
  project/run_scripts/session01_direct_z_possibility_server1.sbatch \
  project/run_scripts/session01_direct_z_possibility_pair_server1.sbatch \
  project/run_scripts/submit_session01_direct_z_possibility_pair_server1.sh \
  project/run_scripts/ode_edit_motivation/direct_z_possibility.py \
  plans/global/2026-08-01-session01-direct-z-possibility-spec.md \
  audits/global/2026-08-01-session01-direct-z-possibility-preflight.md \
  >/dev/null
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "direct-z possibility submission requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "direct-z possibility submission requires main == origin/main" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Fxc -- "${PREFLIGHT_VERDICT}" "${PREFLIGHT}" || true)" == "1" ]] || {
  echo "direct-z possibility exact preflight verdict is unavailable" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${PREFLIGHT}" || true)" == "1" ]] || {
  echo "direct-z possibility preflight verdict is ambiguous" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/${LLAMA_RUN}" && ! -L "${OUTPUT_ROOT}/${LLAMA_RUN}" ]] || {
  echo "Llama direct-z possibility output already exists" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/${QWEN_RUN}" && ! -L "${OUTPUT_ROOT}/${QWEN_RUN}" ]] || {
  echo "Qwen direct-z possibility output already exists" >&2
  exit 2
}
[[ ! -e "${PAIR_MARKER}" && ! -L "${PAIR_MARKER}" ]] || {
  echo "direct-z possibility one-shot marker already exists" >&2
  exit 2
}
[[ -z "$("${SQUEUE_BIN}" -h --name="${JOB_NAME}")" ]] || {
  echo "direct-z possibility pair job is already active" >&2
  exit 2
}

[[ "$("${GREP_BIN}" -Fxc -- "ODEEDIT_REPOSITORY_ID=hyunjun1127/ODE-edit" "${SESSION_ENV}" || true)" == "1" ]] || exit 2
[[ "$("${GREP_BIN}" -Fxc -- "ODEEDIT_REPOSITORY_CWD=${REPO_ROOT}" "${SESSION_ENV}" || true)" == "1" ]] || exit 2
[[ "$("${GREP_BIN}" -Fxc -- "ODEEDIT_CODEX_SESSION_ID=${SESSION_ID}" "${SESSION_ENV}" || true)" == "1" ]] || exit 2
[[ "$("${GREP_BIN}" -Fxc -- "ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE='Sol Ultra'" "${SESSION_ENV}" || true)" == "1" ]] || exit 2
[[ "$("${GREP_BIN}" -Fxc -- "ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE='Sol Ultra'" "${SESSION_ENV}" || true)" == "1" ]] || exit 2
[[ "$("${GREP_BIN}" -Fxc -- "ODEEDIT_CURRENT_SERVER=server1" "${SESSION_ENV}" || true)" == "1" ]] || exit 2
[[ "$("${WC_BIN}" -l <"${SESSION_ENV}")" == "6" ]] || exit 2
# This pair consumes two slots under server1's local aggregate GPU cap of four.
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 2 130000
"${INSTALL_BIN}" -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
"${MKDIR_BIN}" -m 700 "${PAIR_MARKER}" || exit 2

job_id="$("${SBATCH_BIN}" --parsable --export=NONE --job-name="${JOB_NAME}" \
  "${SBATCH_FILE}")"
printf '%s\n' "${job_id}" >"${PAIR_MARKER}/job-id"
"${CHMOD_BIN}" 600 "${PAIR_MARKER}/job-id"
printf '%s\n' "${job_id}"
