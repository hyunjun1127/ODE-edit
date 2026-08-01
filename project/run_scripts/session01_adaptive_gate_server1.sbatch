#!/usr/bin/env bash
# Motivation adaptive-gate child; valid only as one step of a locked pair.
#SBATCH --job-name=odeedit_agate_child_server1
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=65000M
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.err

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly EXPECTED_SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly RUNTIME_ENV="${REPO_ROOT}/servers/local/method-runtime.env"
readonly SESSION_ENV="${REPO_ROOT}/servers/local/session-boundary.env"
readonly SESSION_CHECK="${REPO_ROOT}/scripts/check-session-boundary.sh"
readonly CHILD_WRAPPER="${REPO_ROOT}/project/run_scripts/session01_adaptive_gate_server1.sbatch"
readonly RUNNER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/adaptive_gate_refresh.py"
readonly GATE="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/adaptive_path.py"
readonly PROJECTOR="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/projector_adapter.py"
readonly QSTEP="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/quarter_step_refresh.py"
readonly SPEC="${REPO_ROOT}/plans/global/2026-08-01-session01-motivation-closure-spec.md"
readonly GIT_BIN="/usr/bin/git"
readonly UV_BIN="/usr/local/bin/uv"

load_allowlisted_env() {
  local source_path="$1"
  local allowed_keys="$2"
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9_./:@+-]+)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
    elif [[ "${line}" == "ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE='Sol Ultra'" ]]; then
      key="ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE"
      value="Sol Ultra"
    elif [[ "${line}" == "ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE='Sol Ultra'" ]]; then
      key="ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE"
      value="Sol Ultra"
    else
      echo "environment file contains an unsafe assignment" >&2
      exit 2
    fi
    [[ ",${allowed_keys}," == *",${key},"* ]] || {
      echo "environment file contains a non-allowlisted key" >&2
      exit 2
    }
    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done <"${source_path}"
}

require_tracked_clean_file() {
  local absolute_path="$1"
  local relative_path actual_blob tracked_blob
  [[ "${absolute_path}" == "${REPO_ROOT}/"* ]] || exit 2
  [[ -f "${absolute_path}" && ! -L "${absolute_path}" && -r "${absolute_path}" ]] || exit 2
  relative_path="${absolute_path#"${REPO_ROOT}/"}"
  "${GIT_BIN}" ls-files --error-unmatch -- "${relative_path}" >/dev/null || exit 2
  actual_blob="$("${GIT_BIN}" hash-object -- "${absolute_path}")"
  tracked_blob="$("${GIT_BIN}" rev-parse "HEAD:${relative_path}")"
  [[ "${actual_blob}" == "${tracked_blob}" ]] || exit 2
  "${GIT_BIN}" diff --quiet -- "${relative_path}" || exit 2
  "${GIT_BIN}" diff --cached --quiet -- "${relative_path}" || exit 2
}

[[ "$#" -eq 3 ]] || {
  echo "adaptive-gate child requires track, model, and run ID" >&2
  exit 2
}
readonly TRACK="$1"
readonly MODEL_ALIAS="$2"
readonly RUN_ID="$3"
case "${TRACK}|${MODEL_ALIAS}|${RUN_ID}|${SLURM_JOB_NAME:-}" in
  "memit|llama3-8b-inst|agate_memit_llama_g0_v2|odeedit_agate_memit_pair_v2") ;;
  "memit|qwen2.5-7b-inst|agate_memit_qwen_g0_v2|odeedit_agate_memit_pair_v2") ;;
  "alphaedit_projected|llama3-8b-inst|agate_alpha_llama_g0_v2|odeedit_agate_alpha_pair_v2") ;;
  "alphaedit_projected|qwen2.5-7b-inst|agate_alpha_qwen_g0_v2|odeedit_agate_alpha_pair_v2") ;;
  *)
    echo "track/model/run/job identity is outside the locked pair" >&2
    exit 2
    ;;
esac

[[ -r "${RUNTIME_ENV}" && ! -L "${RUNTIME_ENV}" ]] || exit 2
[[ -r "${SESSION_ENV}" && ! -L "${SESSION_ENV}" ]] || exit 2
[[ -x "${SESSION_CHECK}" && ! -L "${SESSION_CHECK}" ]] || exit 2
[[ -x "${GIT_BIN}" && -x "${UV_BIN}" ]] || exit 2

readonly SLURM_ASSIGNED_CUDA="${CUDA_VISIBLE_DEVICES:-}"
cd "${REPO_ROOT}"
load_allowlisted_env \
  "${RUNTIME_ENV}" \
  "EASYEDIT_ROOT,KR_UV_BIN,HF_HOME,HF_HUB_CACHE,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,PYTHONPATH"
load_allowlisted_env \
  "${SESSION_ENV}" \
  "ODEEDIT_REPOSITORY_ID,ODEEDIT_REPOSITORY_CWD,ODEEDIT_CODEX_SESSION_ID,ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE,ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE,ODEEDIT_CURRENT_SERVER"

[[ "${EASYEDIT_ROOT:-}" == "/mnt/raid5/janghj/EasyEdit" ]] || exit 2
[[ "${KR_UV_BIN:-}" == "uv" ]] || exit 2
[[ "${HF_HOME:-}" == "/mnt/raid5/janghj/.cache/huggingface" ]] || exit 2
[[ "${HF_HUB_CACHE:-}" == "/mnt/raid5/janghj/.cache/huggingface/hub" ]] || exit 2
[[ "${HF_HUB_OFFLINE:-}" == "1" && "${TRANSFORMERS_OFFLINE:-}" == "1" ]] || exit 2
[[ "${PYTHONPATH:-}" == "${EASYEDIT_ROOT}" ]] || exit 2
[[ "${ODEEDIT_REPOSITORY_CWD:-}" == "${REPO_ROOT}" ]] || exit 2
[[ "${ODEEDIT_REPOSITORY_ID:-}" == "hyunjun1127/ODE-edit" ]] || exit 2
[[ "${ODEEDIT_CODEX_SESSION_ID:-}" == "${EXPECTED_SESSION_ID}" ]] || exit 2
[[ "${ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE:-}" == "Sol Ultra" ]] || exit 2
[[ "${ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE:-}" == "Sol Ultra" ]] || exit 2
[[ "${ODEEDIT_CURRENT_SERVER:-}" == "server1" ]] || exit 2
[[ "$("${GIT_BIN}" rev-parse --show-toplevel)" == "${REPO_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || exit 2
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || exit 2
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || exit 2
[[ -n "${SLURM_ASSIGNED_CUDA}" && "${SLURM_ASSIGNED_CUDA}" != *,* ]] || exit 2
[[ -n "${SLURM_STEP_ID:-}" && "${SLURM_STEP_ID}" != "batch" ]] || {
  echo "adaptive-gate child must be an srun step" >&2
  exit 2
}

"${SESSION_CHECK}" "${EXPECTED_SESSION_ID}"
for tracked_path in \
  "${CHILD_WRAPPER}" \
  "${RUNNER}" \
  "${GATE}" \
  "${PROJECTOR}" \
  "${QSTEP}" \
  "${SPEC}"; do
  require_tracked_clean_file "${tracked_path}"
done

readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly RUN_DIRECTORY="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIRECTORY}" && ! -L "${RUN_DIRECTORY}" ]] || {
  echo "exclusive adaptive-gate run directory already exists" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}"
[[ -d "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || exit 2

export CUDA_VISIBLE_DEVICES="${SLURM_ASSIGNED_CUDA}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export UV_OFFLINE=1
export UV_NO_SYNC=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY OPENAI_API_KEY

exec "${UV_BIN}" run \
  --offline \
  --project "${EASYEDIT_ROOT}" \
  --frozen \
  --no-sync \
  python -m project.run_scripts.ode_edit_motivation.adaptive_gate_refresh \
  --easyedit-root "${EASYEDIT_ROOT}" \
  --track "${TRACK}" \
  --model "${MODEL_ALIAS}" \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}"
