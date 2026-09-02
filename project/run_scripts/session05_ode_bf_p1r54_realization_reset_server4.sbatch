#!/usr/bin/env bash
#SBATCH --job-name=p1r54_reset_s4
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server4
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly FINAL_PRE_GPU_RECEIPT="${3:-}"
readonly CELL="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly EXPECTED_SESSION="codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d"
readonly EXPECTED_BRANCH="codex/server4-p1r54-realization-reset-four-runs-v1"
readonly EASYEDIT_ROOT="/data/janghj/EasyEdit"
readonly RUN_TOKEN="p1r54-realization-reset-four-runs-v1"
[[ "$#" -eq 3 ]] || { echo "P1R54 reset arguments differ" >&2; exit 2; }
[[ "$CELL" =~ ^[0-3]$ ]] || { echo "P1R54 reset array cell differs" >&2; exit 2; }

case "$CELL" in
  0) readonly RESULT_NAME="s05-p1r54-realization-reset-independent-b100-fz-tech-r3-v1" ;;
  1) readonly RESULT_NAME="s05-p1r54-realization-reset-independent-b100-pdz-t1-tech-r3-v1" ;;
  2) readonly RESULT_NAME="s05-p1r54-realization-reset-sequential-10xb100-fz-tech-r3-v1" ;;
  3) readonly RESULT_NAME="s05-p1r54-realization-reset-sequential-10xb100-pdz-t1-tech-r3-v1" ;;
esac

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R54 reset CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R54 reset source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^{tree})" == "${SOURCE_TREE}" ]] || { echo "P1R54 reset tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R54 reset branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R54 reset tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "server4" ]] || { echo "P1R54 reset node differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R54 reset exactly one GPU required" >&2; exit 2; }
[[ -f "${FINAL_PRE_GPU_RECEIPT}" && ! -L "${FINAL_PRE_GPU_RECEIPT}" ]] || { echo "P1R54 reset final receipt differs" >&2; exit 2; }

readonly OUTPUT_ROOT="${REPO_ROOT}/local/odebf/results/${RESULT_NAME}"
readonly STATE_ROOT="${REPO_ROOT}/local/odebf/state/${RESULT_NAME}"
readonly LOG_ROOT="${REPO_ROOT}/local/odebf/logs/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || { echo "P1R54 reset output exists" >&2; exit 2; }
[[ ! -e "${STATE_ROOT}" && ! -L "${STATE_ROOT}" ]] || { echo "P1R54 reset state exists" >&2; exit 2; }
[[ ! -e "${LOG_ROOT}" && ! -L "${LOG_ROOT}" ]] || { echo "P1R54 reset log exists" >&2; exit 2; }
mkdir -m 0700 -- "${STATE_ROOT}"
mkdir -m 0700 -- "${LOG_ROOT}"

export PROJECT_GPU_CAP=4
export HF_HOME="/data/janghj/.cache/huggingface"
export HF_HUB_CACHE="/data/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r54-reset-cell${CELL}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec "${EASYEDIT_ROOT}/.venv/bin/python" -m \
  project.run_scripts.session05_ode_bf_p1r54_realization_reset \
  --cell "${CELL}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --output-root "${OUTPUT_ROOT}" --state-root "${STATE_ROOT}" \
  --log-root "${LOG_ROOT}" --final-pre-gpu-receipt "${FINAL_PRE_GPU_RECEIPT}" \
  --run-token "${RUN_TOKEN}" \
  >"${LOG_ROOT}/stdout.jsonl" 2>"${LOG_ROOT}/stderr.log"
