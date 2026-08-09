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

readonly REPO_ROOT="/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-universal-obs-rs-bg-targethold-p1r13-v1/ODE-edit"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BASE_RUNTIME="bfd77ddbf046333c59a8e589429d426fd5eee4c4"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly EXPECTED_EXECUTION_PARENT="bfd77ddbf046333c59a8e589429d426fd5eee4c4"
readonly EXECUTION_BRANCH="codex/odeeditsh1-s05-universal-obs-rs-bg-targethold-p1r13-v1"
readonly MODEL_ALIAS="${1:-}"
readonly OUTPUT_ROOT="${2:-}"
readonly SOURCE_HEAD="${3:-}"
readonly RUN_TOKEN="${4:-}"
readonly CELL_ID="${5:-}"

[[ "$#" -eq 5 ]] || { echo "locked R13 invocation requires five arguments" >&2; exit 2; }
[[ "${MODEL_ALIAS}" == "llama3-8b-inst" ]] || {
  echo "R13 SH1 is Llama-only; Qwen submission is forbidden" >&2
  exit 2
}
case "${CELL_ID}" in
  RS-NEUTRAL|RS-SOFT|BG-NEUTRAL) ;;
  *) echo "invalid locked R13 universal-observability cell" >&2; exit 2 ;;
esac
[[ -n "${OUTPUT_ROOT}" && "${RUN_TOKEN}" == "universal-obs-rs-bg-targethold-p1r13-a1-v1" ]] || {
  echo "missing locked R13 invocation" >&2
  exit 2
}
[[ "${SOURCE_HEAD}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R13 source head must be a lowercase SHA-1" >&2
  exit 2
}
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || { echo "R13 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "R13 source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^)" == "${EXPECTED_EXECUTION_PARENT}" ]] || {
  echo "R13 execution parent differs" >&2
  exit 2
}
[[ "$(git branch --show-current)" == "${EXECUTION_BRANCH}" ]] || {
  echo "R13 execution branch differs" >&2
  exit 2
}
git merge-base --is-ancestor "${EXPECTED_BASE_RUNTIME}" "${SOURCE_HEAD}" || {
  echo "R13 source is not rooted at the locked runtime" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "R13 tracked source dirty" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "R13 output root already exists" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ -x /usr/local/bin/uv && -x "${EASYEDIT_ROOT}/.venv/bin/python" ]] || {
  echo "offline runtime unavailable" >&2
  exit 2
}
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
  echo "exactly one GPU required" >&2
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
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-s05-r13-universal-obs-a2-sh1"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_universal_observability \
  --model "${MODEL_ALIAS}" \
  --cell "${CELL_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
