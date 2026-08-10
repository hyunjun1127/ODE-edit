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

readonly REPO_ROOT="/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-full-drive-soft-dynamic-p1r17-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_PARENT="e6facd2d5dfae12d3c094b51981ad99951174109"
readonly EXPECTED_BRANCH="codex/odeeditsh1-s05-full-drive-soft-dynamic-p1r17-v1"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly MODEL_ALIAS="${1:-}"
readonly CELL_ID="${2:-}"
readonly OUTPUT_ROOT="${3:-}"
readonly SOURCE_HEAD="${4:-}"
readonly RUN_TOKEN="${5:-}"

[[ "$#" -eq 5 ]] || exit 2
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || { echo "P1R17 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R17 source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^)" == "${EXPECTED_PARENT}" ]] || { echo "P1R17 parent differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R17 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R17 tracked source dirty" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R17 output collision" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || exit 2

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r17"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_full_drive_dynamic \
  --model "${MODEL_ALIAS}" --cell "${CELL_ID}" \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
