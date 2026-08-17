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
readonly PHASE="${3:-}"
readonly ATTEMPT_SUFFIX="${4:-}"
readonly EXPECTED_SESSION="019fe491-16f4-7bd3-adf5-4b1eb4a57d1f"
readonly RUN_TOKEN="p1r51-p1r43-rsa-a1-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="main"

[[ "$#" -eq 3 || "$#" -eq 4 ]] || { echo "P1R51 arguments differ" >&2; exit 2; }
[[ "${PHASE}" =~ ^(b1|pilot-neutral|neutral-full|soft-full)$ ]] || { echo "P1R51 phase differs" >&2; exit 2; }
[[ -z "${ATTEMPT_SUFFIX}" || "${ATTEMPT_SUFFIX}" =~ ^[a-zA-Z0-9-]+$ ]] || { echo "P1R51 attempt suffix differs" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R51 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R51 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R51 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R51 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R51 exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly SUFFIX_TOKEN="$([[ -n "${ATTEMPT_SUFFIX}" ]] && printf -- '-%s' "${ATTEMPT_SUFFIX}" || true)"
case "${PHASE}" in
  b1) readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p1r51-rsa-a1-b1-${MODEL}-neutral${SUFFIX_TOKEN}-v1" ;;
  pilot-neutral) readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p1r51-rsa-a1-pilot-${MODEL}-neutral${SUFFIX_TOKEN}-v1" ;;
  neutral-full) readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p1r51-rsa-a1-independent-b10x10-${MODEL}-neutral${SUFFIX_TOKEN}-v1" ;;
  soft-full) readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p1r51-rsa-a1-independent-b10x10-${MODEL}-soft${SUFFIX_TOKEN}-v1" ;;
esac
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R51 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=51 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r51-${PHASE}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

RUN_ARGS=(
  /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync
  python -m project.run_scripts.session05_ode_bf_p1r51_rsa_a1
  --model "${MODEL}" --output-root "${OUTPUT_ROOT}"
  --source-head "${SOURCE_HEAD}" --phase "${PHASE}" --run-token "${RUN_TOKEN}"
)
if [[ -n "${ATTEMPT_SUFFIX}" ]]; then
  RUN_ARGS+=(--attempt-suffix "${ATTEMPT_SUFFIX}")
fi
exec "${RUN_ARGS[@]}"
