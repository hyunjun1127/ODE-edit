#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server2
#SBATCH --mem=65000M
#SBATCH --time=12:00:00
#SBATCH --export=NONE

set -euo pipefail

readonly REPO_ROOT="/mnt/raid5/janghj/.codex/worktrees/odeeditsh2/ODE-edit"
readonly EXPECTED_SESSION="019fc5ec-f85b-7770-a73a-1d19be1cd491"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly MODEL_ALIAS="${1:-}"
readonly OUTPUT_ROOT="${2:-}"
readonly SOURCE_HEAD="${3:-}"
readonly RUN_TOKEN="${4:-}"
readonly EXECUTION_TOKEN="${5:-}"

case "${MODEL_ALIAS}" in
  llama3-8b-inst|qwen2.5-7b-inst) ;;
  *) echo "invalid locked P1 model alias" >&2; exit 2 ;;
esac
[[ -n "${OUTPUT_ROOT}" && -n "${SOURCE_HEAD}" && "${RUN_TOKEN}" == "p1v1" && "${EXECUTION_TOKEN}" == "p1r1v1" ]] || {
  echo "missing locked P1 invocation" >&2
  exit 2
}
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || {
  echo "P1 repository CWD boundary differs" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || {
  echo "P1 source freeze differs" >&2
  exit 2
}
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ -x /usr/local/bin/uv && -x /usr/bin/srun && -x "${EASYEDIT_ROOT}/.venv/bin/python" ]] || {
  echo "fixed offline runtime is unavailable" >&2
  exit 2
}
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
  echo "P1 allocation must expose exactly one GPU" >&2
  exit 2
}

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/bin/srun --unbuffered --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  /usr/local/bin/uv run --offline \
  --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session04_ode_alloc_p1 \
  --model "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}" \
  --execution-token "${EXECUTION_TOKEN}"
