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

readonly REPO_ROOT="/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-bg-soft-missing-cell-p1r12-v1/ODE-edit"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly EXPECTED_SCIENTIFIC_PARENT="38592cd14eddbfd40cd063b1891c75588ddcb7fa"
readonly EXPECTED_IMPLEMENTATION_PARENT="8d0e8c80e3a100fccd4c295c3c3dbab5153602be"
readonly EXPECTED_EXECUTION_REPAIR_PARENT="1f58ea7b6732a67cbced132ee56056cd1f79cad3"
readonly EXECUTION_BRANCH="codex/odeeditsh1-s05-bg-soft-missing-cell-p1r12-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly MODEL_ALIAS="${1:-}"
readonly OUTPUT_ROOT="${2:-}"
readonly SOURCE_HEAD="${3:-}"
readonly RUN_TOKEN="${4:-}"

[[ "$#" -eq 4 ]] || { echo "locked R12 invocation requires four arguments" >&2; exit 2; }
[[ "${MODEL_ALIAS}" == "llama3-8b-inst" ]] || {
  echo "R12 BG-SOFT is Llama-only; Qwen submission is forbidden" >&2
  exit 2
}
[[ -n "${OUTPUT_ROOT}" && "${RUN_TOKEN}" == "bg-soft-missing-cell-p1r12-a1-v1" ]] || {
  echo "missing locked R12 invocation" >&2
  exit 2
}
[[ "${SOURCE_HEAD}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R12 source head must be a lowercase SHA-1" >&2
  exit 2
}
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || { echo "R12 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "R12 source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^)" == "${EXPECTED_EXECUTION_REPAIR_PARENT}" ]] || {
  echo "R12 execution parent differs" >&2
  exit 2
}
[[ "$(git rev-parse HEAD^^)" == "${EXPECTED_IMPLEMENTATION_PARENT}" ]] || {
  echo "R12 implementation parent differs" >&2
  exit 2
}
[[ "$(git rev-parse HEAD^^^)" == "${EXPECTED_SCIENTIFIC_PARENT}" ]] || {
  echo "R12 scientific parent differs" >&2
  exit 2
}
[[ "$(git branch --show-current)" == "${EXECUTION_BRANCH}" ]] || {
  echo "R12 execution branch differs" >&2
  exit 2
}
git merge-base --is-ancestor "${EXPECTED_SCIENTIFIC_PARENT}" "${SOURCE_HEAD}" || {
  echo "R12 source is not rooted at the locked parent" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "R12 tracked source dirty" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "R12 output root already exists" >&2; exit 2; }
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
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-s05-r12-bg-soft"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_bg_soft_missing_cell \
  --model "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
