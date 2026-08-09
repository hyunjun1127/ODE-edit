#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_PARENT="314ab399b0bbba25a4475d54473b22acc89a6167"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly EXPECTED_BRANCH="codex/odeeditsh1-s05-integrated-physical-writer-p1r14-v1"
readonly MODEL_ALIAS="${1:-}"
readonly OUTPUT_ROOT="${2:-}"
readonly SOURCE_HEAD="${3:-}"
readonly RUN_TOKEN="${4:-}"
readonly RUN_ATTEMPT_ID="${5:-}"
readonly LAUNCH_MODE="${6:-RUN}"

if [[ -z "${SLURM_SUBMIT_DIR:-}" || "${SLURM_SUBMIT_DIR}" != /* \
  || ! -d "${SLURM_SUBMIT_DIR}" || -L "${SLURM_SUBMIT_DIR}" ]]; then
  echo "P1R14 SLURM_SUBMIT_DIR differs" >&2
  exit 2
fi
readonly REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"

if [[ "$#" -eq 5 ]]; then
  [[ "${LAUNCH_MODE}" == "RUN" ]] || { echo "P1R14 launch mode differs" >&2; exit 2; }
elif [[ "$#" -eq 6 ]]; then
  [[ "${LAUNCH_MODE}" == "P1R14_LAUNCHER_PROBE_NO_MODEL" ]] || {
    echo "P1R14 launcher probe mode differs" >&2
    exit 2
  }
else
  echo "locked P1R14 invocation requires five run arguments or one probe marker" >&2
  exit 2
fi
[[ "${MODEL_ALIAS}" == "llama3-8b-inst" || "${MODEL_ALIAS}" == "qwen2.5-7b-inst" ]] || {
  echo "server1 P1R14 launcher alias differs" >&2
  exit 2
}
[[ -n "${OUTPUT_ROOT}" && -n "${RUN_ATTEMPT_ID}" \
  && "${RUN_TOKEN}" == "integrated-physical-writer-p1r14-v1" ]] || {
  echo "missing locked P1R14 invocation" >&2
  exit 2
}
[[ "${SOURCE_HEAD}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "P1R14 source head must be a lowercase SHA-1" >&2
  exit 2
}
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || { echo "P1R14 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R14 source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^)" == "${EXPECTED_PARENT}" ]] || { echo "P1R14 parent differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R14 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R14 tracked source dirty" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R14 output root already exists" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
if [[ "${LAUNCH_MODE}" == "P1R14_LAUNCHER_PROBE_NO_MODEL" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || {
    echo "P1R14 launcher probe allocation differs" >&2
    exit 2
  }
  echo "P1R14_LAUNCHER_PROVENANCE_PASS_PRE_UV"
fi
[[ -x /usr/local/bin/uv && -x "${EASYEDIT_ROOT}/.venv/bin/python" ]] || {
  echo "P1R14 offline runtime unavailable" >&2
  exit 2
}
if [[ "${LAUNCH_MODE}" == "RUN" ]]; then
  [[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
    echo "P1R14 requires exactly one visible GPU" >&2
    exit 2
  }
fi

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-s05-p1r14"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

NAMESPACE_PROBE_ARGS=()
if [[ "${LAUNCH_MODE}" == "P1R14_LAUNCHER_PROBE_NO_MODEL" ]]; then
  NAMESPACE_PROBE_ARGS=(--namespace-probe)
fi
readonly -a NAMESPACE_PROBE_ARGS

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_integrated_physical_writer \
  --model "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}" \
  --run-attempt "${RUN_ATTEMPT_ID}" \
  "${NAMESPACE_PROBE_ARGS[@]}"
