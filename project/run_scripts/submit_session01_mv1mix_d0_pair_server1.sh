#!/usr/bin/env bash
# Fail-closed one-shot submit helper for the exact paired MV-1 score-mix D0.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly GIT_BIN="/usr/bin/git"
readonly SBATCH_BIN="/usr/bin/sbatch"
readonly SQUEUE_BIN="/usr/bin/squeue"
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_mv1mix_d0_pair_server1.sbatch"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-07-30-session01-mv1mix-d0-execution-preflight.md"
readonly IDENTIFIABILITY_GATE="${REPO_ROOT}/audits/global/2026-07-30-mv1-finite-action-identifiability-postpilot.md"
readonly MV0_CLOSURE="${REPO_ROOT}/audits/global/2026-07-30-mv0-pair-c3-v3.postrun.md"
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly PAIR_MARKER="${STATE_ROOT}/mv1mix_d0_pair_v1.submitted"
readonly JOB_NAME="odeedit_mv1mix_d0_pair_v1"

[[ "$#" -eq 0 ]] || {
  echo "this paired MV-1 score-mix D0 helper takes no arguments" >&2
  exit 2
}
[[ -x "${GIT_BIN}" && -x "${SBATCH_BIN}" && -x "${SQUEUE_BIN}" ]] || {
  echo "required fixed system command is unavailable" >&2
  exit 2
}

cd "${REPO_ROOT}"
[[ "$("${GIT_BIN}" config --local agent.id)" == "head-server1-gh" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.role)" == "global-head" ]] || exit 2
[[ "$("${GIT_BIN}" config --local agent.hostname)" == "server1" ]] || exit 2
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || exit 2

"${GIT_BIN}" ls-files --error-unmatch \
  project/run_scripts/session01_mv1mix_server1.sbatch \
  project/run_scripts/session01_mv1mix_d0_pair_server1.sbatch \
  project/run_scripts/submit_session01_mv1mix_d0_pair_server1.sh \
  project/run_scripts/ode_edit_motivation/mv1_score_mix.py \
  project/run_scripts/ode_edit_motivation/mv1_score_mix_analysis.py \
  project/run_scripts/ode_edit_motivation/tests/test_mv1_score_mix.py \
  project/run_scripts/ode_edit_motivation/tests/test_mv1_score_mix_analysis.py \
  plans/global/2026-07-30-session01-mv1-implementation-spec.md \
  audits/global/2026-07-30-mv1-finite-action-identifiability-postpilot.md \
  audits/global/2026-07-30-session01-mv1mix-d0-execution-preflight.md \
  >/dev/null
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "paired MV-1 score-mix D0 submission requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "paired MV-1 score-mix D0 submission requires main == origin/main" >&2
  exit 2
}
grep -Fxq -- \
  '- 최종 판정: `PASS` — MV-1 score-mix D0 동시 pair 한 건에만 유효' \
  "${RED_GATE}" || {
  echo "MV-1 score-mix D0 execution preflight has not passed" >&2
  exit 2
}
grep -Fxq -- \
  '- 실행 허용: v2 `D0 calibration[3:8]`, model별 5 case 동시 pair만' \
  "${IDENTIFIABILITY_GATE}" || {
  echo "MV-1 score-mix D0 identifiability gate is unavailable" >&2
  exit 2
}
grep -Fxq -- \
  '- 최종 판정: **`PASS` — 두 고정 모델의 MV-0 implementation fidelity closure에만 유효**' \
  "${MV0_CLOSURE}" || {
  echo "MV-0 two-model implementation closure is unavailable" >&2
  exit 2
}

[[ ! -e "${OUTPUT_ROOT}/mv1mix_llama_d0_v1" ]] || {
  echo "Llama score-mix D0 output already exists" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/mv1mix_qwen_d0_v1" ]] || {
  echo "Qwen score-mix D0 output already exists" >&2
  exit 2
}
[[ ! -e "${PAIR_MARKER}" ]] || {
  echo "MV-1 score-mix D0 one-shot marker already exists" >&2
  exit 2
}
if [[ -n "$("${SQUEUE_BIN}" -h --name="${JOB_NAME}")" ]]; then
  echo "paired MV-1 score-mix D0 job is already active" >&2
  exit 2
fi

"${REPO_ROOT}/scripts/check-session-boundary.sh" "${SESSION_ID}"
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 2 130000
install -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
mkdir -m 700 "${PAIR_MARKER}" || exit 2

job_id="$("${SBATCH_BIN}" --parsable --export=NONE "${SBATCH_FILE}")"
printf '%s\n' "${job_id}" >"${PAIR_MARKER}/job-id"
chmod 600 "${PAIR_MARKER}/job-id"
printf '%s\n' "${job_id}"
