#!/usr/bin/env bash
# Direct-z possibility diagnostic; exact one-model child on server1.
# Direct submission is rejected: valid only inside the locked simultaneous pair.
#SBATCH --job-name=odeedit_dzf_child_server1
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
readonly EXPECTED_SESSION_ID="019fbb9a-e810-7330-ac55-5b72b7c24337"
readonly RUNTIME_ENV="${REPO_ROOT}/servers/local/method-runtime.env"
readonly SESSION_ENV="${REPO_ROOT}/servers/local/session-boundaries/direct-z-019fbb9a.env"
readonly CHILD_WRAPPER="${REPO_ROOT}/project/run_scripts/session01_direct_z_possibility_server1.sbatch"
readonly PAIR_WRAPPER="${REPO_ROOT}/project/run_scripts/session01_direct_z_possibility_pair_server1.sbatch"
readonly SUBMIT_HELPER="${REPO_ROOT}/project/run_scripts/submit_session01_direct_z_possibility_pair_server1.sh"
readonly RUNNER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/direct_z_possibility.py"
readonly SPEC="${REPO_ROOT}/plans/global/2026-08-01-session01-direct-z-possibility-spec.md"
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
  [[ -f "${absolute_path}" && ! -L "${absolute_path}" && -r "${absolute_path}" ]] || {
    echo "required tracked input is unavailable" >&2
    exit 2
  }
  relative_path="${absolute_path#"${REPO_ROOT}/"}"
  "${GIT_BIN}" ls-files --error-unmatch -- "${relative_path}" >/dev/null || exit 2
  actual_blob="$("${GIT_BIN}" hash-object -- "${absolute_path}")"
  tracked_blob="$("${GIT_BIN}" rev-parse "HEAD:${relative_path}")"
  [[ "${actual_blob}" == "${tracked_blob}" ]] || {
    echo "required tracked input differs from HEAD" >&2
    exit 2
  }
  "${GIT_BIN}" diff --quiet -- "${relative_path}" || exit 2
  "${GIT_BIN}" diff --cached --quiet -- "${relative_path}" || exit 2
}

[[ "$#" -eq 2 ]] || {
  echo "direct-z possibility child requires exact model and run ID" >&2
  exit 2
}
readonly MODEL_ALIAS="$1"
readonly RUN_ID="$2"
case "${MODEL_ALIAS}|${RUN_ID}|${SLURM_JOB_NAME:-}" in
  "llama3-8b-inst|dzf_llama_p0_v2|odeedit_dzf_pair_v2") ;;
  "qwen2.5-7b-inst|dzf_qwen_p0_v2|odeedit_dzf_pair_v2") ;;
  *)
    echo "model/run/job identity is outside the locked direct-z possibility pair" >&2
    exit 2
    ;;
esac

[[ -r "${RUNTIME_ENV}" && ! -L "${RUNTIME_ENV}" ]] || exit 2
[[ -r "${SESSION_ENV}" && ! -L "${SESSION_ENV}" ]] || exit 2
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
  echo "direct-z possibility child must be an srun step" >&2
  exit 2
}

for tracked_path in \
  "${CHILD_WRAPPER}" \
  "${PAIR_WRAPPER}" \
  "${SUBMIT_HELPER}" \
  "${RUNNER}" \
  "${SPEC}"; do
  require_tracked_clean_file "${tracked_path}"
done

readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly RUN_DIRECTORY="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIRECTORY}" && ! -L "${RUN_DIRECTORY}" ]] || {
  echo "exclusive direct-z possibility run directory already exists" >&2
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
  python -m project.run_scripts.ode_edit_motivation.direct_z_possibility \
  --easyedit-root "${EASYEDIT_ROOT}" \
  --model "${MODEL_ALIAS}" \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}"
