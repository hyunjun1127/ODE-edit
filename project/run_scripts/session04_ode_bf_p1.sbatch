#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server2
#SBATCH --mem=65000M
#SBATCH --time=24:00:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/.codex/worktrees/odeeditsh2/ODE-edit"
readonly EXPECTED_SESSION="019fc5ec-f85b-7770-a73a-1d19be1cd491"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly MODEL_ALIAS="${1:-}"
readonly OUTPUT_ROOT="${2:-}"
readonly SOURCE_HEAD="${3:-}"
readonly RUN_TOKEN="${4:-}"

case "${MODEL_ALIAS}" in
  llama3-8b-inst|qwen2.5-7b-inst) ;;
  *) echo "invalid locked ODE-BF P1 alias" >&2; exit 2 ;;
esac
[[ -n "${OUTPUT_ROOT}" && -n "${SOURCE_HEAD}" && "${RUN_TOKEN}" == "seqb10-native-floor-p1r2-v2" ]] || {
  echo "missing locked ODE-BF P1 invocation" >&2
  exit 2
}
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || {
  echo "ODE-BF P1 repository CWD boundary differs" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || {
  echo "ODE-BF P1 source freeze differs" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "ODE-BF P1 tracked source is not clean" >&2
  exit 2
}
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ -x /usr/local/bin/uv && -x "${EASYEDIT_ROOT}/.venv/bin/python" ]] || {
  echo "fixed offline runtime is unavailable" >&2
  exit 2
}
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
  echo "ODE-BF P1 allocation must expose exactly one GPU" >&2
  exit 2
}

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline \
  --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session04_ode_bf_p1 \
  --model "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
