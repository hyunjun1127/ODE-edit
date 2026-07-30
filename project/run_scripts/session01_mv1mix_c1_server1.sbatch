#!/usr/bin/env bash
# Session 01 Motivation MV-1 C1 fold 0; exact one-model child on server1.
# This wrapper is intentionally C1-only and binds each model to its locked
# calibration-only static/forecast policies.
#SBATCH --job-name=odeedit_mv1mix_c1_server1
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
readonly RUNNER="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/mv1_score_mix_confirmatory.py"
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
[[ -r "${RUNNER}" ]] || {
  echo "tracked MV-1 C1 runner is unavailable" >&2
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
  "EASYEDIT_ROOT,KR_UV_BIN,HF_HOME,HF_HUB_CACHE,HF_DATASETS_CACHE,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,PYTHONPATH"
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
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || {
  echo "Slurm node boundary mismatch" >&2
  exit 2
}
[[ -n "${SLURM_ASSIGNED_CUDA}" ]] || {
  echo "Slurm did not assign CUDA visibility" >&2
  exit 2
}
export CUDA_VISIBLE_DEVICES="${SLURM_ASSIGNED_CUDA}"

[[ "$#" -eq 2 ]] || {
  echo "this MV-1 C1 envelope requires its exact model and run ID" >&2
  exit 2
}
readonly MODEL_ALIAS="$1"
readonly RUN_ID="$2"
case "${MODEL_ALIAS}|${RUN_ID}|${SLURM_JOB_NAME:-}" in
  "llama3-8b-inst|mv1mix_llama_c1_v1|odeedit_mv1mix_c1_pair_v1")
    static_policy_path="${REPO_ROOT}/experiment-reports/global/2026-07-31-mv1mix-llama-d1-v1-static-policy.json"
    forecast_policy_path="${REPO_ROOT}/experiment-reports/global/2026-07-31-mv1mix-llama-d0-d1-forecast-v1.policy.json"
    ;;
  "qwen2.5-7b-inst|mv1mix_qwen_c1_v1|odeedit_mv1mix_c1_pair_v1")
    static_policy_path="${REPO_ROOT}/experiment-reports/global/2026-07-31-mv1mix-qwen-d1-v1-static-policy.json"
    forecast_policy_path="${REPO_ROOT}/experiment-reports/global/2026-07-31-mv1mix-qwen-d0-d1-forecast-v1.policy.json"
    ;;
  *)
    echo "Slurm arguments/job-name do not match the audited MV-1 C1 fold 0" >&2
    exit 2
    ;;
esac
readonly STATIC_POLICY="${static_policy_path}"
readonly FORECAST_POLICY="${forecast_policy_path}"
unset static_policy_path forecast_policy_path

for policy_path in "${STATIC_POLICY}" "${FORECAST_POLICY}"; do
  [[ -f "${policy_path}" && ! -L "${policy_path}" && -r "${policy_path}" ]] || {
    echo "locked C1 policy file is unavailable" >&2
    exit 2
  }
done

readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
[[ ! -e "${OUTPUT_ROOT}/${RUN_ID}" ]] || {
  echo "exclusive run directory already exists" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}"

# The C1 runner uses the existing read-only EasyEdit bridge and pinned
# covariance bundle. Projectors are provenance-checked but never deserialized;
# no covariance, null-space, Wikidata, model, or dependency download/recompute
# is authorized through this envelope.
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
  python -m project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory \
  --easyedit-root "${EASYEDIT_ROOT}" \
  --model "${MODEL_ALIAS}" \
  --run-id "${RUN_ID}" \
  --static-policy "${STATIC_POLICY}" \
  --forecast-policy "${FORECAST_POLICY}" \
  --output-root "${OUTPUT_ROOT}" \
  --seed 17
