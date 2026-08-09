#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=24:00:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-canonical-alpha-posfield-p1r11-v1/ODE-edit"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly MODEL_ALIAS="${1:-}"
readonly ACTUATOR="${2:-}"
readonly OUTPUT_ROOT="${3:-}"
readonly SOURCE_HEAD="${4:-}"
readonly RUN_TOKEN="${5:-}"

case "${MODEL_ALIAS}" in
  llama3-8b-inst|qwen2.5-7b-inst) ;;
  *) echo "invalid locked S05 R11 alias" >&2; exit 2 ;;
esac
case "${ACTUATOR}" in
  CURRENT-SHARED-AE|CANONICAL-ORDERED-AE) ;;
  *) echo "invalid locked S05 R11 actuator" >&2; exit 2 ;;
esac
[[ -n "${OUTPUT_ROOT}" && -n "${SOURCE_HEAD}" && "${RUN_TOKEN}" == "canonical-alpha-posfield-p1r11-v1" ]] || {
  echo "missing locked S05 R11 invocation" >&2
  exit 2
}
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || { echo "S05 R11 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "S05 R11 source differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "S05 R11 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ -x /usr/local/bin/uv && -x "${EASYEDIT_ROOT}/.venv/bin/python" ]] || { echo "offline runtime unavailable" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "exactly one GPU required" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-s05-r11"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_canonical_alpha_posfield \
  --model "${MODEL_ALIAS}" \
  --actuator "${ACTUATOR}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
