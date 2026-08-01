#!/usr/bin/env bash
# One-shot GH helper for the locked MEMIT or AlphaEdit-projected pair.

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
readonly SBATCH_FILE="${REPO_ROOT}/project/run_scripts/session01_adaptive_gate_pair_server1.sbatch"
readonly RED_GATE="${REPO_ROOT}/audits/global/2026-08-01-session01-adaptive-gate-execution-preflight.md"
readonly RED_GATE_VERDICT='- 최종 판정: `PASS` — locked adaptive-gate MEMIT/Alpha pair 각각 1회에만 유효'
readonly REPAIR_GATE="${REPO_ROOT}/audits/global/2026-08-01-session01-adaptive-gate-lineage-label-repair-preflight.md"
readonly REPAIR_GATE_VERDICT='- 최종 판정: `PASS` — zero-outcome lineage-label repair v2 pair에만 유효'
readonly LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"

[[ "$#" -eq 1 ]] || {
  echo "usage: $0 memit|alphaedit_projected" >&2
  exit 2
}
readonly TRACK="$1"
case "${TRACK}" in
  memit)
    readonly JOB_NAME="odeedit_agate_memit_pair_v2"
    readonly LLAMA_RUN="agate_memit_llama_g0_v2"
    readonly QWEN_RUN="agate_memit_qwen_g0_v2"
    readonly PAIR_MARKER="${STATE_ROOT}/agate_memit_pair_v2.submitted"
    ;;
  alphaedit_projected)
    readonly JOB_NAME="odeedit_agate_alpha_pair_v2"
    readonly LLAMA_RUN="agate_alpha_llama_g0_v2"
    readonly QWEN_RUN="agate_alpha_qwen_g0_v2"
    readonly PAIR_MARKER="${STATE_ROOT}/agate_alpha_pair_v2.submitted"
    ;;
  *)
    echo "unknown adaptive-gate track" >&2
    exit 2
    ;;
esac

[[ -x "${GIT_BIN}" && -x "${GREP_BIN}" && -x "${INSTALL_BIN}" \
  && -x "${MKDIR_BIN}" && -x "${CHMOD_BIN}" && -x "${SBATCH_BIN}" \
  && -x "${SQUEUE_BIN}" && -x "${SACCT_BIN}" ]] || exit 2
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
  project/run_scripts/session01_adaptive_gate_server1.sbatch \
  project/run_scripts/session01_adaptive_gate_pair_server1.sbatch \
  project/run_scripts/submit_session01_adaptive_gate_pair_server1.sh \
  project/run_scripts/ode_edit_motivation/adaptive_gate_refresh.py \
  project/run_scripts/ode_edit_motivation/adaptive_path.py \
  project/run_scripts/ode_edit_motivation/projector_adapter.py \
  project/run_scripts/ode_edit_motivation/quarter_step_refresh.py \
  plans/global/2026-08-01-session01-motivation-closure-spec.md \
  audits/global/2026-08-01-session01-adaptive-gate-execution-preflight.md \
  audits/global/2026-08-01-session01-adaptive-gate-lineage-label-repair-preflight.md \
  messages/inbox/server1.md \
  >/dev/null
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "adaptive-gate submission requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "adaptive-gate submission requires main == origin/main" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Fxc -- "${RED_GATE_VERDICT}" "${RED_GATE}" || true)" == "1" ]] || {
  echo "adaptive-gate exact red-gate verdict is unavailable" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${RED_GATE}" || true)" == "1" ]] || {
  echo "adaptive-gate red-gate verdict is ambiguous" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Fxc -- "${REPAIR_GATE_VERDICT}" "${REPAIR_GATE}" || true)" == "1" ]] || {
  echo "adaptive-gate exact repair verdict is unavailable" >&2
  exit 2
}
[[ "$("${GREP_BIN}" -Ec -- '^- 최종 판정:' "${REPAIR_GATE}" || true)" == "1" ]] || {
  echo "adaptive-gate repair verdict is ambiguous" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/${LLAMA_RUN}" && ! -L "${OUTPUT_ROOT}/${LLAMA_RUN}" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/${QWEN_RUN}" && ! -L "${OUTPUT_ROOT}/${QWEN_RUN}" ]] || exit 2
[[ ! -e "${PAIR_MARKER}" && ! -L "${PAIR_MARKER}" ]] || {
  echo "adaptive-gate one-shot marker already exists" >&2
  exit 2
}
[[ -z "$("${SQUEUE_BIN}" -h --name="${JOB_NAME}")" ]] || {
  echo "adaptive-gate track is already active" >&2
  exit 2
}

if [[ "${TRACK}" == "memit" ]]; then
  readonly ORIGINAL_MARKER="${STATE_ROOT}/agate_memit_pair_v1.submitted"
  [[ -f "${ORIGINAL_MARKER}/job-id" && ! -L "${ORIGINAL_MARKER}" \
    && ! -L "${ORIGINAL_MARKER}/job-id" ]] || exit 2
  original_job_id="$(<"${ORIGINAL_MARKER}/job-id")"
  [[ "${original_job_id}" =~ ^[0-9]+$ ]] || exit 2
  original_state=""
  read -r original_state < <(
    "${SACCT_BIN}" -n -X -j "${original_job_id}" --format=State --noheader
  )
  [[ "${original_state}" == "FAILED" ]] || {
    echo "v2 repair requires exact original FAILED state" >&2
    exit 2
  }
  for outcome in \
    "${OUTPUT_ROOT}/agate_memit_llama_g0_v1/outcomes.jsonl" \
    "${OUTPUT_ROOT}/agate_memit_qwen_g0_v1/outcomes.jsonl"; do
    [[ -f "${outcome}" && ! -L "${outcome}" && ! -s "${outcome}" ]] || {
      echo "v2 repair requires zero original scientific outcomes" >&2
      exit 2
    }
  done
elif [[ "${TRACK}" == "alphaedit_projected" ]]; then
  readonly MEMIT_MARKER="${STATE_ROOT}/agate_memit_pair_v2.submitted"
  [[ -f "${MEMIT_MARKER}/job-id" && ! -L "${MEMIT_MARKER}" \
    && ! -L "${MEMIT_MARKER}/job-id" ]] || {
    echo "Alpha pair requires completed MEMIT pair marker" >&2
    exit 2
  }
  memit_job_id="$(<"${MEMIT_MARKER}/job-id")"
  [[ "${memit_job_id}" =~ ^[0-9]+$ ]] || exit 2
  memit_state=""
  read -r memit_state < <(
    "${SACCT_BIN}" -n -X -j "${memit_job_id}" --format=State --noheader
  )
  [[ "${memit_state}" == "COMPLETED" ]] || {
    echo "Alpha pair waits for exact MEMIT COMPLETED state" >&2
    exit 4
  }
  for summary in \
    "${OUTPUT_ROOT}/agate_memit_llama_g0_v2/summary.json" \
    "${OUTPUT_ROOT}/agate_memit_qwen_g0_v2/summary.json"; do
    [[ -f "${summary}" && ! -L "${summary}" ]] || exit 2
    [[ "$("${GREP_BIN}" -o -- '"all_pass":true' "${summary}" | wc -l)" == "1" ]] || {
      echo "Alpha pair requires technical-valid MEMIT summaries" >&2
      exit 2
    }
  done
fi

"${REPO_ROOT}/scripts/check-session-boundary.sh" "${SESSION_ID}"
"${REPO_ROOT}/scripts/check-slurm-resource-cap.sh" server1 2 130000
"${INSTALL_BIN}" -d -m 700 "${LOG_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}"
"${MKDIR_BIN}" -m 700 "${PAIR_MARKER}" || exit 2

job_id="$("${SBATCH_BIN}" --parsable --export=NONE --job-name="${JOB_NAME}" \
  "${SBATCH_FILE}" "${TRACK}")"
printf '%s\n' "${job_id}" >"${PAIR_MARKER}/job-id"
"${CHMOD_BIN}" 600 "${PAIR_MARKER}/job-id"
printf '%s\n' "${job_id}"
