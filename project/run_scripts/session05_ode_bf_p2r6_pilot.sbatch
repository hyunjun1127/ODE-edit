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

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly PHASE="${3:-}"
readonly SELECTED_CONTROLLER="${4:-}"
readonly ATTEMPT_SUFFIX="${5:-}"
readonly EXPECTED_SESSION="019fe491-16f4-7bd3-adf5-4b1eb4a57d1f"
readonly RUN_TOKEN="p2r6-llama-qp-repair-phase1-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="codex/p2r6-llama-qp-repair-phase1-v1"

[[ "$#" -eq 4 || "$#" -eq 5 ]] || { echo "P2R6 arguments differ" >&2; exit 2; }
[[ "${PHASE}" == "phase1" ]] || { echo "P2R6 repair launcher authorizes Phase1 only" >&2; exit 2; }
[[ -z "${SELECTED_CONTROLLER}" ]] || { echo "P2R6 Phase1 selection differs" >&2; exit 2; }
[[ -z "${ATTEMPT_SUFFIX}" || "${ATTEMPT_SUFFIX}" =~ ^[a-zA-Z0-9-]+$ ]] || { echo "P2R6 attempt suffix differs" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P2R6 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P2R6 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P2R6 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P2R6 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P2R6 exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
readonly CASES=(5 1)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CASE_INDEX="${CASES[${SLURM_ARRAY_TASK_ID}]}"
readonly SUFFIX_TOKEN="$([[ -n "${ATTEMPT_SUFFIX}" ]] && printf -- '-%s' "${ATTEMPT_SUFFIX}" || true)"
readonly SELECT_TOKEN="$([[ -n "${SELECTED_CONTROLLER}" ]] && printf -- '-%s' "${SELECTED_CONTROLLER,,}" || true)"
readonly CASE_TOKEN="$(printf '%02d' "${CASE_INDEX}")"
readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p2r6-red-r2-final-${PHASE}${SELECT_TOKEN}-${MODEL}-case-${CASE_TOKEN}-paired${SUFFIX_TOKEN}-v1"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P2R6 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=48 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p2r6-red-r2-final"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

RUN_ARGS=(
  /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync
  python -m project.run_scripts.session05_ode_bf_p2r6_pilot
  --model "${MODEL}" --phase "${PHASE}" --case-index "${CASE_INDEX}"
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
)
if [[ -n "${SELECTED_CONTROLLER}" ]]; then
  RUN_ARGS+=(--selected-controller "${SELECTED_CONTROLLER}")
fi
if [[ -n "${ATTEMPT_SUFFIX}" ]]; then
  RUN_ARGS+=(--attempt-suffix "${ATTEMPT_SUFFIX}")
fi
exec "${RUN_ARGS[@]}"
