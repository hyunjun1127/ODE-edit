#!/usr/bin/env bash
# Fail-closed paired submit helper for the two exact MV-0 c3 runs.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_mv0_pair_server1.sbatch"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-07-30-session01-mv0-paired-c3-execution-preflight.md"
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly PAIR_MARKER="${STATE_ROOT}/mv0_pair_c3_v1.submitted"
readonly JOB_NAME="odeedit_mv0_pair_c3"

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
  audits/global/2026-07-30-session01-mv0-paired-c3-execution-preflight.md \
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
grep -Fxq -- '- 최종 판정: `PASS` — exact paired MV-0 c3 한 건에만 유효' \
  "${RED_GATE}" || {
  echo "paired execution preflight has not passed" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/mv0_llama_c3_v1" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/mv0_qwen_c3_v1" ]] || exit 2
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
