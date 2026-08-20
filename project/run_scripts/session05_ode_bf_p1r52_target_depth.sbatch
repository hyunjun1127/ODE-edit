#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server2
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-3%4
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly CASE_COUNT="${3:-}"
readonly ATTEMPT_SUFFIX="${4:-}"
readonly EXPECTED_SESSION="01a00e5c-f7ae-72a2-98b2-b8b0907168b4"
readonly RUN_TOKEN="p1r52-target-depth-il1-il3full-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="codex/p1r52-target-depth-il1-il3full-v1"

[[ "$#" -eq 3 || "$#" -eq 4 ]] || { echo "P1R52 target-depth arguments differ" >&2; exit 2; }
[[ "${CASE_COUNT}" == 1 || "${CASE_COUNT}" == 10 ]] || { echo "P1R52 target-depth case count differs" >&2; exit 2; }
[[ -z "${ATTEMPT_SUFFIX}" || "${ATTEMPT_SUFFIX}" =~ ^[a-zA-Z0-9-]+$ ]] || { echo "P1R52 target-depth attempt suffix differs" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R52 target-depth CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R52 target-depth source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R52 target-depth branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R52 target-depth tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R52 target-depth exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst)
readonly DEPTHS=(IL1 IL3-FULL IL1 IL3-FULL)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly DEPTH="${DEPTHS[${SLURM_ARRAY_TASK_ID}]}"
readonly PHASE="$([[ "${CASE_COUNT}" == 1 ]] && printf 'b1' || printf 'b10x10')"
readonly DEPTH_TOKEN="${DEPTH,,}"
readonly DEPTH_TOKEN_CLEAN="${DEPTH_TOKEN//-/}"
readonly SUFFIX_TOKEN="$([[ -n "${ATTEMPT_SUFFIX}" ]] && printf -- '-%s' "${ATTEMPT_SUFFIX}" || true)"
readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p1r52-target-depth-target-only-${PHASE}-${MODEL}-${DEPTH_TOKEN_CLEAN}${SUFFIX_TOKEN}-v1"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R52 target-depth output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-target-depth-${MODEL}-${DEPTH_TOKEN_CLEAN}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

RUN_ARGS=(
  /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync
  python -m project.run_scripts.session05_ode_bf_p1r52_target_depth
  --model "${MODEL}" --depth "${DEPTH}" --case-count "${CASE_COUNT}"
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
)
if [[ -n "${ATTEMPT_SUFFIX}" ]]; then
  RUN_ARGS+=(--attempt-suffix "${ATTEMPT_SUFFIX}")
fi
exec "${RUN_ARGS[@]}"
