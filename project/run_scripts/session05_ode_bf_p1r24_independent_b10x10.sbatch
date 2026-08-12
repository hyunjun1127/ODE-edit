#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly EXPECTED_SESSION="019fe491-16f4-7bd3-adf5-4b1eb4a57d1f"
readonly RUN_TOKEN="p1r24-rs-soft-independent-b10x10-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="codex/p1r24-rs-soft-independent-b10x10-v1"

[[ "$#" -eq 2 ]] || { echo "P1R24 independent requires source and result parent" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R24 independent CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R24 independent source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R24 independent branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R24 independent tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R24 independent exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_NAME="s05-p1r24-rs-soft-independent-b10x10-${MODEL}-v1"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R24 independent output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r24-independent-b10x10"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r24_independent_b10x10 \
  --model "${MODEL}" --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
