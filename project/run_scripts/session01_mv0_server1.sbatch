#!/usr/bin/env bash
# Session 01 MV-0; target_server=server1; GH emergency/time-critical smoke only.
#SBATCH --job-name=odeedit_mv0_server1
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=65000M
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.err

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly EXPECTED_SESSION_ID="019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2"
readonly RUNTIME_ENV="${REPO_ROOT}/servers/local/method-runtime.env"
readonly SESSION_ENV="${REPO_ROOT}/servers/local/session-boundary.env"
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
  done < "${source_path}"
}

[[ -r "${RUNTIME_ENV}" ]] || {
  echo "missing runtime environment" >&2
  exit 2
}
[[ -r "${SESSION_ENV}" ]] || {
  echo "missing session boundary" >&2
  exit 2
}
[[ -x "${UV_BIN}" ]] || {
  echo "fixed uv runtime is unavailable" >&2
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
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || {
  echo "Slurm node boundary mismatch" >&2
  exit 2
}
[[ -n "${SLURM_ASSIGNED_CUDA}" ]] || {
  echo "Slurm did not assign CUDA visibility" >&2
  exit 2
}
export CUDA_VISIBLE_DEVICES="${SLURM_ASSIGNED_CUDA}"

[[ "$#" -eq 3 ]] || {
  echo "this Slurm envelope requires its exact three one-shot arguments" >&2
  exit 2
}
readonly MODEL_ALIAS="$1"
readonly RUN_ID="$2"
readonly CASES="$3"
case "${MODEL_ALIAS}|${RUN_ID}|${CASES}|${SLURM_JOB_NAME:-}" in
  "llama3-8b-inst|mv0_llama_smoke_v1|1|odeedit_mv0_llama_smoke") ;;
  "qwen2.5-7b-inst|mv0_qwen_smoke_v1|1|odeedit_mv0_qwen_smoke") ;;
  "llama3-8b-inst|mv0_llama_c3_v2|3|odeedit_mv0_pair_c3v2") ;;
  "qwen2.5-7b-inst|mv0_qwen_c3_v2|3|odeedit_mv0_pair_c3v2") ;;
  *)
    echo "Slurm arguments/job-name do not match an audited one-shot envelope" >&2
    exit 2
    ;;
esac

readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
[[ ! -e "${OUTPUT_ROOT}/${RUN_ID}" ]] || {
  echo "exclusive run directory already exists" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export UV_OFFLINE=1
export UV_NO_SYNC=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY OPENAI_API_KEY

exec "${UV_BIN}" run \
  --offline \
  --project "${EASYEDIT_ROOT}" \
  --frozen \
  --no-sync \
  python -m project.run_scripts.ode_edit_motivation.mv0_fidelity run \
  --easyedit-root "${EASYEDIT_ROOT}" \
  --model "${MODEL_ALIAS}" \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --cases "${CASES}" \
  --seed 17 \
  --selection-seed ode-edit-motivation-counterfact-v1
