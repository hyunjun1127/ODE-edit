#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-7%4
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly RUN_TOKEN="historical-h0-bg-sequential-p1r20-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 2 ]] || { echo "P1R20 requires source and result parent" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R20 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R20 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "codex/p1r20-historical-h0-bg-sequential-v1" ]] || { echo "P1R20 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R20 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R20 exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst llama3-8b-inst llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst qwen2.5-7b-inst qwen2.5-7b-inst)
readonly METHODS=(BG-NEUTRAL BG-SOFT-H MEMIT ALPHAEDIT BG-NEUTRAL BG-SOFT-H MEMIT ALPHAEDIT)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"
readonly METHOD_SLUG="${METHOD,,}"
readonly RESULT_NAME="s05-p1r20-historical-h0-bg-sequential-${METHOD_SLUG//-/_}-${MODEL}-v1"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R20 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r20"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_historical_h0_sequential \
  --model "${MODEL}" --method "${METHOD}" --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
