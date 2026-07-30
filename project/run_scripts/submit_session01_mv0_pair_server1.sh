#!/usr/bin/env bash
# Fail-closed paired submit helper for the two exact MV-0 c3 runs.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_mv0_pair_server1.sbatch"
readonly RUNTIME_FAILURE_ANALYSIS="${REPO_ROOT}/experiment-reports/global/2026-07-30-mv0-pair-c3-failure-analysis.md"
readonly CONCURRENCY_ANALYSIS="${REPO_ROOT}/experiment-reports/global/2026-07-30-mv0-pair-c3-v2-concurrency-analysis.md"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-07-30-session01-mv0-paired-c3-v3-execution-preflight.md"
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly FAILED_PAIR_MARKER_V1="${STATE_ROOT}/mv0_pair_c3_v1.submitted"
readonly CANCELLED_PAIR_MARKER_V2="${STATE_ROOT}/mv0_pair_c3_v2.submitted"
readonly PAIR_MARKER="${STATE_ROOT}/mv0_pair_c3_v3.submitted"
readonly JOB_NAME="odeedit_mv0_pair_c3v3"

[[ "$#" -eq 0 ]] || {
  echo "this paired helper takes no arguments" >&2
  exit 2
}

cd "${REPO_ROOT}"
[[ "$(git config --local agent.id)" == "head-server1-gh" ]] || exit 2
[[ "$(git config --local agent.role)" == "global-head" ]] || exit 2
[[ "$(git config --local agent.hostname)" == "server1" ]] || exit 2
[[ "$(git branch --show-current)" == "main" ]] || exit 2

git ls-files --error-unmatch \
  project/run_scripts/session01_mv0_pair_server1.sbatch \
  project/run_scripts/session01_mv0_server1.sbatch \
  project/run_scripts/submit_session01_mv0_pair_server1.sh \
  experiment-reports/global/2026-07-30-mv0-pair-c3-failure-analysis.md \
  experiment-reports/global/2026-07-30-mv0-pair-c3-v2-concurrency-analysis.md \
  audits/global/2026-07-30-session01-mv0-paired-c3-v3-execution-preflight.md \
  experiment-reports/global/2026-07-30-mv0-llama-smoke-analysis.md \
  experiment-reports/global/2026-07-30-mv0-qwen-smoke-analysis.md \
  audits/global/2026-07-30-mv0-llama-smoke.postrun.md \
  audits/global/2026-07-30-mv0-qwen-smoke.postrun.md \
  project/run_scripts/ode_edit_motivation/mv0_fidelity.py \
  >/dev/null
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "paired MV-0 submission requires a clean repository" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || exit 2
grep -Fxq -- '- 최종 판정: `PASS` — exact paired MV-0 c3 v3 한 건에만 유효' \
  "${RED_GATE}" || {
  echo "paired execution preflight has not passed" >&2
  exit 2
}
grep -Fxq -- '- 결론: **실험 실패가 아니라 launcher/runtime 환경의 공통 조기 실패**' \
  "${RUNTIME_FAILURE_ANALYSIS}" || {
  echo "first paired failure has not been independently analyzed" >&2
  exit 2
}
grep -Fxq -- '- 결론: **동시 실행 요건을 충족하지 못한 Slurm step-memory envelope 결함**' \
  "${CONCURRENCY_ANALYSIS}" || {
  echo "second paired concurrency failure has not been independently analyzed" >&2
  exit 2
}
[[ -f "${FAILED_PAIR_MARKER_V1}/job-id" ]] || exit 2
[[ "$(<"${FAILED_PAIR_MARKER_V1}/job-id")" == "15508" ]] || exit 2
[[ -s "${LOG_ROOT}/odeedit_mv0_pair_c3-15508.out" ]] || exit 2
[[ -s "${LOG_ROOT}/odeedit_mv0_pair_c3-15508.err" ]] || exit 2
[[ -f "${CANCELLED_PAIR_MARKER_V2}/job-id" ]] || exit 2
[[ "$(<"${CANCELLED_PAIR_MARKER_V2}/job-id")" == "15512" ]] || exit 2
[[ -e "${LOG_ROOT}/odeedit_mv0_pair_c3v2-15512.out" ]] || exit 2
[[ -s "${LOG_ROOT}/odeedit_mv0_pair_c3v2-15512.err" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/mv0_llama_c3_v1" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/mv0_qwen_c3_v1" ]] || exit 2
[[ -s "${OUTPUT_ROOT}/mv0_llama_c3_v2/manifest.json" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/mv0_qwen_c3_v2" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/mv0_llama_c3_v3" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/mv0_qwen_c3_v3" ]] || exit 2
[[ ! -e "${PAIR_MARKER}" ]] || exit 2
if [[ -n "$(squeue -h --name="${JOB_NAME}")" ]]; then
  echo "paired MV-0 job is already active" >&2
  exit 2
fi

"${REPO_ROOT}/scripts/check-session-boundary.sh" "${SESSION_ID}"
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 2 130000
install -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
mkdir -m 700 "${PAIR_MARKER}" || exit 2

job_id="$(sbatch --parsable --export=NONE "${SBATCH_FILE}")"
printf '%s\n' "${job_id}" >"${PAIR_MARKER}/job-id"
chmod 600 "${PAIR_MARKER}/job-id"
printf '%s\n' "${job_id}"
