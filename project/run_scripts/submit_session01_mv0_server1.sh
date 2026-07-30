#!/usr/bin/env bash
# Fail-closed GH submit helper for one server1 MV-0 job.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_mv0_server1.sbatch"
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"

[[ "$#" -eq 3 ]] || {
  echo "usage: $0 MODEL_ALIAS AUDITED_RUN_ID 1" >&2
  exit 2
}
readonly MODEL_ALIAS="$1"
readonly RUN_ID="$2"
readonly CASES="$3"

case "${MODEL_ALIAS}|${RUN_ID}|${CASES}" in
  "llama3-8b-inst|mv0_llama_smoke_v1|1")
    JOB_NAME="odeedit_mv0_llama_smoke"
    RED_GATE="${REPO_ROOT}/audits/global/2026-07-30-session01-mv0-execution-preflight.md"
    ;;
  "qwen2.5-7b-inst|mv0_qwen_smoke_v1|1")
    JOB_NAME="odeedit_mv0_qwen_smoke"
    RED_GATE="${REPO_ROOT}/audits/global/2026-07-30-session01-mv0-qwen-execution-preflight.md"
    ;;
  *)
    echo "arguments do not match an audited exact one-shot envelope" >&2
    exit 2
    ;;
esac
readonly JOB_NAME RED_GATE

cd "${REPO_ROOT}"
[[ "$(git config --local agent.id)" == "head-server1-gh" ]] || {
  echo "Git agent ID is not the registered global-head" >&2
  exit 2
}
[[ "$(git config --local agent.role)" == "global-head" ]] || {
  echo "only the registered global-head may use this exception helper" >&2
  exit 2
}
[[ "$(git config --local agent.hostname)" == "server1" ]] || {
  echo "Git agent hostname is not server1" >&2
  exit 2
}
[[ "$(git branch --show-current)" == "main" ]] || {
  echo "MV-0 submission requires main" >&2
  exit 2
}
git ls-files --error-unmatch \
  project/run_scripts/session01_mv0_server1.sbatch \
  project/run_scripts/submit_session01_mv0_server1.sh \
  "${RED_GATE#${REPO_ROOT}/}" \
  project/run_scripts/ode_edit_motivation/__init__.py \
  project/run_scripts/ode_edit_motivation/artifacts.py \
  project/run_scripts/ode_edit_motivation/contracts.py \
  project/run_scripts/ode_edit_motivation/diagnostic_math.py \
  project/run_scripts/ode_edit_motivation/direct_z.py \
  project/run_scripts/ode_edit_motivation/easyedit_bridge.py \
  project/run_scripts/ode_edit_motivation/gpu_runtime.py \
  project/run_scripts/ode_edit_motivation/hooks.py \
  project/run_scripts/ode_edit_motivation/manifests.py \
  project/run_scripts/ode_edit_motivation/mv0_fidelity.py \
  >/dev/null
if [[ "${MODEL_ALIAS}" == "qwen2.5-7b-inst" ]]; then
  git ls-files --error-unmatch \
    experiment-reports/global/2026-07-30-mv0-llama-smoke-analysis.md \
    audits/global/2026-07-30-mv0-llama-smoke.postrun.md \
    >/dev/null
fi
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "MV-0 submission requires a clean repository with no untracked files" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || {
  echo "local HEAD is not the pushed origin/main commit" >&2
  exit 2
}
[[ -x "${SBATCH_FILE}" || -r "${SBATCH_FILE}" ]] || {
  echo "tracked sbatch file is unavailable" >&2
  exit 2
}
[[ -r "${RED_GATE}" ]] || {
  echo "red-team execution preflight is missing" >&2
  exit 2
}
grep -Fxq -- '- 최종 판정: `PASS` — 아래 exact MV-0 smoke 한 건에만 유효' \
  "${RED_GATE}" || {
  echo "red-team execution preflight has not passed" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/${RUN_ID}" ]] || {
  echo "exclusive run directory already exists" >&2
  exit 2
}
readonly SUBMISSION_MARKER="${STATE_ROOT}/${RUN_ID}.submitted"
[[ ! -e "${SUBMISSION_MARKER}" ]] || {
  echo "the one-shot submission marker already exists; retry requires a new audit" >&2
  exit 2
}
if [[ -n "$(squeue -h --name="${JOB_NAME}")" ]]; then
  echo "a same-name ODE-Edit MV-0 job is already active" >&2
  exit 2
fi

"${REPO_ROOT}/scripts/check-session-boundary.sh" "${SESSION_ID}"
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 1 65000
install -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
mkdir -m 700 "${SUBMISSION_MARKER}" || {
  echo "failed to acquire the durable one-shot submission marker" >&2
  exit 2
}

job_id="$(sbatch \
  --parsable \
  --export=NONE \
  --job-name="${JOB_NAME}" \
  "${SBATCH_FILE}" \
  "${MODEL_ALIAS}" \
  "${RUN_ID}" \
  "${CASES}")"
printf '%s\n' "${job_id}" >"${SUBMISSION_MARKER}/job-id"
chmod 600 "${SUBMISSION_MARKER}/job-id"
printf '%s\n' "${job_id}"
