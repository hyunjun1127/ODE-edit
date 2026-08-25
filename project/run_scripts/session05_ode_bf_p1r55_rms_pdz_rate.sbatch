#!/usr/bin/env bash
#SBATCH --job-name=p1r55_rms_pdz_rate_r1
#SBATCH --array=0-11%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=47:59:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly FINAL_PRE_GPU_RECEIPT="${3:-}"
readonly CELL="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly EXPECTED_SESSION="01a0278d-456b-7d20-820f-63fc80a57b8d"
readonly EXPECTED_BRANCH="codex/p1r55-rms-pdz-floored-rate-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly RUN_TOKEN="p1r55-rms-pdz-floored-rate-phase1-warm-t1-tech-r1-v1"

[[ "$#" -eq 3 ]] || { echo "P1R55 Phase1 arguments differ" >&2; exit 2; }
[[ "${CELL}" =~ ^([0-9]|1[01])$ ]] || { echo "P1R55 Phase1 cell differs" >&2; exit 2; }

readonly SNAPSHOTS=(b01 b01 b01 b01 b06 b06 b06 b06 b10 b10 b10 b10)
readonly ARMS=(a0 a1 a2 a3 a0 a1 a2 a3 a0 a1 a2 a3)
readonly RESULT_NAME="s05-p1r55-rms-pdz-rate-phase1-warm-${SNAPSHOTS[${CELL}]}-${ARMS[${CELL}]}-t1-tech-r1-v1"

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R55 Phase1 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R55 Phase1 source differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "P1R55 Phase1 tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R55 Phase1 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R55 Phase1 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || { echo "P1R55 Phase1 node differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R55 Phase1 exactly one GPU required" >&2; exit 2; }
[[ -f "${FINAL_PRE_GPU_RECEIPT}" && ! -L "${FINAL_PRE_GPU_RECEIPT}" ]] || { echo "P1R55 Phase1 receipt differs" >&2; exit 2; }

readonly OUTPUT_ROOT="${REPO_ROOT}/local/odebf/results/${RESULT_NAME}"
readonly STATE_ROOT="${REPO_ROOT}/local/odebf/state/${RESULT_NAME}"
readonly LOG_ROOT="${REPO_ROOT}/local/odebf/logs/${RESULT_NAME}"
mkdir -p -m 0700 -- "${REPO_ROOT}/local/odebf/results" "${REPO_ROOT}/local/odebf/state" "${REPO_ROOT}/local/odebf/logs"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || { echo "P1R55 Phase1 output exists" >&2; exit 2; }
[[ ! -e "${STATE_ROOT}" && ! -L "${STATE_ROOT}" ]] || { echo "P1R55 Phase1 state exists" >&2; exit 2; }
[[ ! -e "${LOG_ROOT}" && ! -L "${LOG_ROOT}" ]] || { echo "P1R55 Phase1 log exists" >&2; exit 2; }
mkdir -m 0700 -- "${STATE_ROOT}"
mkdir -m 0700 -- "${LOG_ROOT}"

export PROJECT_GPU_CAP=3
export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONHASHSEED=55 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r55-rms-pdz-rate-${CELL}-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec "${EASYEDIT_ROOT}/.venv/bin/python" -m \
  project.run_scripts.session05_ode_bf_p1r55_rms_pdz_rate \
  --cell "${CELL}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --output-root "${OUTPUT_ROOT}" --state-root "${STATE_ROOT}" \
  --final-pre-gpu-receipt "${FINAL_PRE_GPU_RECEIPT}" \
  --run-token "${RUN_TOKEN}" \
  >"${LOG_ROOT}/stdout.jsonl" 2>"${LOG_ROOT}/stderr.log"
