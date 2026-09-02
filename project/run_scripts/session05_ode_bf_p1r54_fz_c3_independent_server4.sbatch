#!/usr/bin/env bash
#SBATCH --job-name=p1r54_fz_ind_s4
#SBATCH --array=0-9%3
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
readonly EXPECTED_BRANCH="codex/server4-p1r54-fz-c3-independent-10xb100-v1"
readonly EASYEDIT_ROOT="/data/janghj/EasyEdit"
readonly RUN_TOKEN="p1r54-fz-c3-independent-10xb100-v1"
[[ "$#" -eq 3 ]] || { echo "P1R54 FZ independent arguments differ" >&2; exit 2; }
[[ "$CELL" =~ ^[0-9]$ ]] || { echo "P1R54 FZ independent array cell differs" >&2; exit 2; }
printf -v BATCH_TAG 'b%02d' "$((CELL + 1))"
readonly RESULT_NAME="s05-p1r54-fz-c3-independent-${BATCH_TAG}-b100-tech-r2-v1"

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R54 FZ independent CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R54 FZ independent source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^{tree})" == "${SOURCE_TREE}" ]] || { echo "P1R54 FZ independent tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R54 FZ independent branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R54 FZ independent tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "server4" ]] || { echo "P1R54 FZ independent node differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R54 FZ independent exactly one GPU required" >&2; exit 2; }
[[ -f "${FINAL_PRE_GPU_RECEIPT}" && ! -L "${FINAL_PRE_GPU_RECEIPT}" ]] || { echo "P1R54 FZ independent final receipt differs" >&2; exit 2; }

readonly OUTPUT_ROOT="${REPO_ROOT}/local/odebf/results/${RESULT_NAME}"
readonly STATE_ROOT="${REPO_ROOT}/local/odebf/state/${RESULT_NAME}"
readonly LOG_ROOT="${REPO_ROOT}/local/odebf/logs/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || { echo "P1R54 FZ independent output exists" >&2; exit 2; }
[[ ! -e "${STATE_ROOT}" && ! -L "${STATE_ROOT}" ]] || { echo "P1R54 FZ independent state exists" >&2; exit 2; }
[[ ! -e "${LOG_ROOT}" && ! -L "${LOG_ROOT}" ]] || { echo "P1R54 FZ independent log exists" >&2; exit 2; }
mkdir -m 0700 -- "${STATE_ROOT}"
mkdir -m 0700 -- "${LOG_ROOT}"

export PROJECT_GPU_CAP=4
export HF_HOME="/data/janghj/.cache/huggingface"
export HF_HUB_CACHE="/data/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r54-fz-independent-${BATCH_TAG}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec "${EASYEDIT_ROOT}/.venv/bin/python" -m \
  project.run_scripts.session05_ode_bf_p1r54_fz_c3_independent \
  --cell "${CELL}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --output-root "${OUTPUT_ROOT}" --state-root "${STATE_ROOT}" \
  --log-root "${LOG_ROOT}" --final-pre-gpu-receipt "${FINAL_PRE_GPU_RECEIPT}" \
  --run-token "${RUN_TOKEN}" \
  >"${LOG_ROOT}/stdout.jsonl" 2>"${LOG_ROOT}/stderr.log"
