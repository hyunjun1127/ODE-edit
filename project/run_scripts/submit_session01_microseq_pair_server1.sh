#!/usr/bin/env bash
# One-shot GH submission helper for the locked four-edit Motivation pair.

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
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_microseq_pair_server1.sbatch"
readonly CHILD_FILE="${REPO_ROOT}/project/run_scripts/session01_microseq_server1.sbatch"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-08-01-session01-microseq-execution-preflight.md"
readonly RED_GATE_VERDICT='- 최종 판정: `PASS` — Alpha technical-valid 및 pre-action contract repair 검증 후 locked four-edit pair 1회 재제출에만 유효'
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly PAIR_MARKER="${STATE_ROOT}/microseq_pair_m0_v1.submitted"
readonly JOB_NAME="odeedit_microseq_pair_v1"
readonly ALPHA_MARKER="${STATE_ROOT}/agate_alpha_pair_v3.submitted"

[[ "$#" -eq 0 ]] || exit 2
[[ -x "${GIT_BIN}" && -x "${GREP_BIN}" && -x "${INSTALL_BIN}" \
  && -x "${MKDIR_BIN}" && -x "${CHMOD_BIN}" && -x "${SBATCH_BIN}" \
  && -x "${SQUEUE_BIN}" && -x "${SACCT_BIN}" ]] || exit 2
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
  project/run_scripts/session01_microseq_server1.sbatch \
  project/run_scripts/session01_microseq_pair_server1.sbatch \
  project/run_scripts/submit_session01_microseq_pair_server1.sh \
  project/run_scripts/ode_edit_motivation/microseq_controller.py \
  project/run_scripts/ode_edit_motivation/microseq_artifacts.py \
  project/run_scripts/ode_edit_motivation/microseq_evaluator.py \
  project/run_scripts/ode_edit_motivation/microseq_analysis.py \
  plans/global/2026-08-01-session01-micro-sequential-specialization-spec.md \
  audits/global/2026-08-01-session01-microseq-execution-preflight.md \
  messages/inbox/server1.md \
  >/dev/null
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "microseq submission requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "microseq submission requires main == origin/main" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Fxc -- "${RED_GATE_VERDICT}" "${RED_GATE}" || true)" == "1" ]] || exit 2
[[ "$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${RED_GATE}" || true)" == "1" ]] || exit 2

[[ -f "${ALPHA_MARKER}/job-id" && ! -L "${ALPHA_MARKER}" \
  && ! -L "${ALPHA_MARKER}/job-id" ]] || {
  echo "microseq requires the Alpha pair submission marker" >&2
  exit 2
}
alpha_job_id="$(<"${ALPHA_MARKER}/job-id")"
[[ "${alpha_job_id}" =~ ^[0-9]+$ ]] || exit 2
alpha_state=""
read -r alpha_state < <(
  "${SACCT_BIN}" -n -X -j "${alpha_job_id}" --format=State --noheader
)
[[ "${alpha_state}" == "COMPLETED" ]] || {
  echo "microseq waits for exact Alpha COMPLETED state" >&2
  exit 4
}
for summary in \
  "${OUTPUT_ROOT}/agate_alpha_llama_g0_v3/summary.json" \
  "${OUTPUT_ROOT}/agate_alpha_qwen_g0_v3/summary.json"; do
  [[ -f "${summary}" && ! -L "${summary}" ]] || exit 2
  [[ "$("${GREP_BIN}" -o -- '"all_pass":true' "${summary}" | wc -l)" == "1" ]] || {
    echo "microseq requires technical-valid Alpha summaries" >&2
    exit 2
  }
done

for run_name in \
  microseq_native_llama_m0_v1 microseq_native_qwen_m0_v1 \
  microseq_ode_llama_m0_v1 microseq_ode_qwen_m0_v1 \
  microseq_eval_native_llama_m0_v1 microseq_eval_native_qwen_m0_v1 \
  microseq_eval_ode_llama_m0_v1 microseq_eval_ode_qwen_m0_v1 \
  microseq_combined_llama_m0_v1 microseq_combined_qwen_m0_v1 \
  microseq_pair_m0_v1; do
  [[ ! -e "${OUTPUT_ROOT}/${run_name}" && ! -L "${OUTPUT_ROOT}/${run_name}" ]] || exit 2
done
[[ ! -e "${PAIR_MARKER}" && ! -L "${PAIR_MARKER}" ]] || exit 2
[[ -z "$("${SQUEUE_BIN}" -h --name="${JOB_NAME}")" ]] || exit 2

"${REPO_ROOT}/scripts/check-session-boundary.sh" "${SESSION_ID}"
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 2 130000
"${INSTALL_BIN}" -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
"${MKDIR_BIN}" -m 700 "${PAIR_MARKER}" || exit 2

job_id="$("${SBATCH_BIN}" --parsable --export=NONE "${SBATCH_FILE}")"
printf '%s\n' "${job_id}" >"${PAIR_MARKER}/job-id"
"${CHMOD_BIN}" 600 "${PAIR_MARKER}/job-id"
printf '%s\n' "${job_id}"
