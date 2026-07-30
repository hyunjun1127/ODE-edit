#!/usr/bin/env bash
# Session 01 Motivation MV-2 refresh; exact one-model child on server1.
# Direct submission is rejected: this child is valid only inside the locked
# simultaneous Llama/Qwen parent allocation.
#SBATCH --job-name=odeedit_mv2refresh_server1
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=65000M
#SBATCH --time=08:00:00
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
readonly CHILD_WRAPPER="${REPO_ROOT}/project/run_scripts/session01_mv2refresh_server1.sbatch"
readonly RUNNER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/mv2_refresh.py"
readonly BRIDGE="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/easyedit_bridge.py"
readonly LINEAGE="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/frozen_target_lineage.py"
readonly TRAJECTORY="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/trajectory.py"
readonly SPEC="${REPO_ROOT}/plans/global/2026-07-31-session01-mv2-implementation-spec.md"
readonly GIT_BIN="/usr/bin/git"
readonly UV_BIN="/usr/local/bin/uv"

load_allowlisted_env() {
  local source_path="$1"
  local allowed_keys="$2"
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    [[ "${line}" =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9_./:@+-]+)$ ]] || {
      echo "environment file contains an unsafe assignment" >&2
      exit 2
    }
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
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
  [[ "${absolute_path}" == "${REPO_ROOT}/"* ]] || {
    echo "tracked-file boundary mismatch" >&2
    exit 2
  }
  [[ -f "${absolute_path}" && ! -L "${absolute_path}" && -r "${absolute_path}" ]] || {
    echo "required tracked input is unavailable" >&2
    exit 2
  }
  relative_path="${absolute_path#"${REPO_ROOT}/"}"
  "${GIT_BIN}" ls-files --error-unmatch -- "${relative_path}" >/dev/null || {
    echo "required input is not tracked" >&2
    exit 2
  }
  actual_blob="$("${GIT_BIN}" hash-object -- "${absolute_path}")"
  tracked_blob="$("${GIT_BIN}" rev-parse "HEAD:${relative_path}")"
  [[ "${actual_blob}" == "${tracked_blob}" ]] || {
    echo "required tracked input differs from HEAD" >&2
    exit 2
  }
  "${GIT_BIN}" diff --quiet -- "${relative_path}" || {
    echo "required tracked input has an unstaged change" >&2
    exit 2
  }
  "${GIT_BIN}" diff --cached --quiet -- "${relative_path}" || {
    echo "required tracked input has a staged change" >&2
    exit 2
  }
}

[[ "$#" -eq 2 ]] || {
  echo "this MV-2 child requires its exact model and run ID" >&2
  exit 2
}
readonly MODEL_ALIAS="$1"
readonly RUN_ID="$2"

case "${MODEL_ALIAS}|${RUN_ID}|${SLURM_JOB_NAME:-}" in
  "llama3-8b-inst|mv2refresh_llama_e0_v1|odeedit_mv2refresh_pair_v1") ;;
  "qwen2.5-7b-inst|mv2refresh_qwen_e0_v1|odeedit_mv2refresh_pair_v1") ;;
  *)
    echo "model/run/job identity is outside the locked MV-2 pair" >&2
    exit 2
    ;;
esac

[[ -r "${RUNTIME_ENV}" && ! -L "${RUNTIME_ENV}" ]] || {
  echo "missing runtime environment" >&2
  exit 2
}
[[ -r "${SESSION_ENV}" && ! -L "${SESSION_ENV}" ]] || {
  echo "missing session boundary" >&2
  exit 2
}
[[ -x "${SESSION_CHECK}" && ! -L "${SESSION_CHECK}" ]] || {
  echo "session-boundary checker is unavailable" >&2
  exit 2
}
[[ -x "${GIT_BIN}" && -x "${UV_BIN}" ]] || {
  echo "fixed Git/uv runtime is unavailable" >&2
  exit 2
}

readonly SLURM_ASSIGNED_CUDA="${CUDA_VISIBLE_DEVICES:-}"
cd "${REPO_ROOT}"

load_allowlisted_env \
  "${RUNTIME_ENV}" \
  "EASYEDIT_ROOT,KR_UV_BIN,HF_HOME,HF_HUB_CACHE,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,PYTHONPATH"
load_allowlisted_env \
  "${SESSION_ENV}" \
  "ODEEDIT_REPOSITORY_ID,ODEEDIT_REPOSITORY_CWD,ODEEDIT_CODEX_SESSION_ID,ODEEDIT_CURRENT_SERVER"

[[ "${EASYEDIT_ROOT:-}" == "/mnt/raid5/janghj/EasyEdit" ]] || {
  echo "EasyEdit root boundary mismatch" >&2
  exit 2
}
[[ "${KR_UV_BIN:-}" == "uv" ]] || {
  echo "method runtime must select uv" >&2
  exit 2
}
[[ "${HF_HOME:-}" == "/mnt/raid5/janghj/.cache/huggingface" ]] || {
  echo "Hugging Face home boundary mismatch" >&2
  exit 2
}
[[ "${HF_HUB_CACHE:-}" == "/mnt/raid5/janghj/.cache/huggingface/hub" ]] || {
  echo "Hugging Face hub cache boundary mismatch" >&2
  exit 2
}
[[ "${HF_HUB_OFFLINE:-}" == "1" && "${TRANSFORMERS_OFFLINE:-}" == "1" ]] || {
  echo "offline model runtime is required" >&2
  exit 2
}
[[ "${PYTHONPATH:-}" == "${EASYEDIT_ROOT}" ]] || {
  echo "method PYTHONPATH boundary mismatch" >&2
  exit 2
}
[[ "${ODEEDIT_REPOSITORY_CWD:-}" == "${REPO_ROOT}" ]] || {
  echo "repository boundary mismatch" >&2
  exit 2
}
[[ "${ODEEDIT_REPOSITORY_ID:-}" == "hyunjun1127/ODE-edit" ]] || {
  echo "repository identity mismatch" >&2
  exit 2
}
[[ "${ODEEDIT_CODEX_SESSION_ID:-}" == "${EXPECTED_SESSION_ID}" ]] || {
  echo "Codex session boundary mismatch" >&2
  exit 2
}
[[ "${ODEEDIT_CURRENT_SERVER:-}" == "server1" ]] || {
  echo "server boundary mismatch" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse --show-toplevel)" == "${REPO_ROOT}" ]] || {
  echo "Git repository root boundary mismatch" >&2
  exit 2
}
[[ "$("${GIT_BIN}" branch --show-current)" == "main" ]] || {
  echo "MV-2 child requires branch main" >&2
  exit 2
}
[[ -z "$("${GIT_BIN}" status --porcelain --untracked-files=normal)" ]] || {
  echo "MV-2 child requires a clean repository" >&2
  exit 2
}
[[ "$("${GIT_BIN}" rev-parse HEAD)" == "$("${GIT_BIN}" rev-parse origin/main)" ]] || {
  echo "MV-2 child requires main == origin/main" >&2
  exit 2
}
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || {
  echo "Slurm node boundary mismatch" >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "MV-2 child requires a numeric Slurm job ID" >&2
  exit 2
}
[[ -n "${SLURM_ASSIGNED_CUDA}" && "${SLURM_ASSIGNED_CUDA}" != *,* ]] || {
  echo "MV-2 child requires exactly one Slurm-visible GPU" >&2
  exit 2
}
[[ -n "${SLURM_STEP_ID:-}" && "${SLURM_STEP_ID}" != "batch" ]] || {
  echo "MV-2 child must run as an srun step in its paired allocation" >&2
  exit 2
}

"${SESSION_CHECK}" "${EXPECTED_SESSION_ID}"
for tracked_path in \
  "${CHILD_WRAPPER}" \
  "${RUNNER}" \
  "${BRIDGE}" \
  "${LINEAGE}" \
  "${TRAJECTORY}" \
  "${SPEC}"; do
  require_tracked_clean_file "${tracked_path}"
done

readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly RUN_DIRECTORY="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIRECTORY}" && ! -L "${RUN_DIRECTORY}" ]] || {
  echo "exclusive MV-2 run directory already exists" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}"
[[ -d "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || {
  echo "MV-2 output root boundary mismatch" >&2
  exit 2
}

# This envelope authorizes only the tracked MV-2 diagnostic. EasyEdit, model
# cache, dataset, covariance moments, and projectors remain input-only.
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
  python -m project.run_scripts.ode_edit_motivation.mv2_refresh \
  --easyedit-root "${EASYEDIT_ROOT}" \
  --model "${MODEL_ALIAS}" \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}"
