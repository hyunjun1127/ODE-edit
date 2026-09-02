#!/usr/bin/env bash
#SBATCH --job-name=p4_euler_za_h1_llama_s4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server4
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --array=2-10%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly FINAL_PRE_GPU_RECEIPT="${3:-}"
readonly STREAM_ROOT="${4:-}"
readonly EXPECTED_SESSION="codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d"
readonly EXPECTED_BRANCH="codex/server4-p4-euler-calibration-r1"
readonly EASYEDIT_ROOT="/data/janghj/EasyEdit"
readonly MODEL="llama3-8b-inst"
readonly CASE_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly OUTPUT_ROOT="${RESULT_PARENT}/B${CASE_INDEX}-llama3-8b-inst"

[[ "$#" -eq 4 ]] || { echo "P4 Euler ZA h1 arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P4 Euler ZA h1 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P4 Euler ZA h1 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P4 Euler ZA h1 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P4 Euler ZA h1 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "server4" ]] || { echo "P4 Euler ZA h1 node differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P4 Euler ZA h1 exactly one GPU required" >&2; exit 2; }
[[ "${CASE_INDEX}" -ge 2 && "${CASE_INDEX}" -le 10 ]] || { echo "P4 Euler ZA h1 B2-B10 differs" >&2; exit 2; }
[[ -f "${FINAL_PRE_GPU_RECEIPT}" && ! -L "${FINAL_PRE_GPU_RECEIPT}" ]] || { echo "P4 Euler ZA h1 final receipt differs" >&2; exit 2; }
[[ -d "${STREAM_ROOT}" && ! -L "${STREAM_ROOT}" ]] || { echo "P4 Euler ZA h1 stream root differs" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P4 Euler ZA h1 output exists" >&2; exit 2; }

export PROJECT_GPU_CAP=2
export HF_HOME="/data/janghj/.cache/huggingface"
export HF_HUB_CACHE="/data/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p4-euler-za-h1-B${CASE_INDEX}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec "${EASYEDIT_ROOT}/.venv/bin/python" -m \
  project.run_scripts.session05_ode_bf_p4_euler_za_llama_h1 \
  --model "${MODEL}" --case-index "${CASE_INDEX}" \
  --output-root "${OUTPUT_ROOT}" --stream-root "${STREAM_ROOT}" \
  --final-pre-gpu-receipt "${FINAL_PRE_GPU_RECEIPT}" \
  --source-head "${SOURCE_HEAD}"
