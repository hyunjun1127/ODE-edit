#!/usr/bin/env bash
#SBATCH --job-name=p1r52_tscale_b100_s4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server4
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly FINAL_PRE_GPU_RECEIPT="${3:-}"
readonly EXPECTED_SESSION="codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d"
readonly EXPECTED_BRANCH="codex/server4-p1r52-target-timescale-b100-v1"
readonly EASYEDIT_ROOT="/data/janghj/EasyEdit"
readonly CELL="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly RUN_TOKEN="p1r52-target-timescale-b100-v1"
readonly RESULT_NAMES=(
  "s05-p1r52-target-timescale-b100-z0-coarse-tech-r4-v1"
  "s05-p1r52-target-timescale-b100-z1-refine-tech-r4-v1"
  "s05-p1r52-target-timescale-b100-z15-tech-r4-v1"
  "s05-p1r52-target-timescale-b100-z20-tech-r4-v1"
  "s05-p1r52-target-timescale-b100-z30-tech-r4-v1"
)

[[ "$#" -eq 3 ]] || { echo "target-timescale arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "target-timescale CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "target-timescale source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^{tree})" == "${SOURCE_TREE}" ]] || { echo "target-timescale tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "target-timescale branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "target-timescale tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "server4" ]] || { echo "target-timescale node differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "target-timescale exactly one GPU required" >&2; exit 2; }
[[ "${CELL}" -ge 0 && "${CELL}" -le 4 ]] || { echo "target-timescale cell differs" >&2; exit 2; }
[[ -f "${FINAL_PRE_GPU_RECEIPT}" && ! -L "${FINAL_PRE_GPU_RECEIPT}" ]] || { echo "target-timescale final receipt differs" >&2; exit 2; }
readonly OUTPUT_ROOT="${REPO_ROOT}/local/odebf/results/${RESULT_NAMES[${CELL}]}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "target-timescale output exists" >&2; exit 2; }

export PROJECT_GPU_CAP=2
export HF_HOME="/data/janghj/.cache/huggingface"
export HF_HUB_CACHE="/data/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-target-timescale-cell${CELL}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec "${EASYEDIT_ROOT}/.venv/bin/python" -m \
  project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100 \
  --cell "${CELL}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --output-root "${OUTPUT_ROOT}" --final-pre-gpu-receipt "${FINAL_PRE_GPU_RECEIPT}" \
  --run-token "${RUN_TOKEN}"
