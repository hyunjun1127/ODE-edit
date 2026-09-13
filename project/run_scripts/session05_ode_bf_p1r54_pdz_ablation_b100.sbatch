#!/usr/bin/env bash
#SBATCH --job-name=p1r54_pdz_ablation_b100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=48:00:00
#SBATCH --array=0-3%3
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly FINAL_PRE_GPU_RECEIPT="${3:-}"
readonly EXPECTED_SESSION="01a0278d-456b-7d20-820f-63fc80a57b8d"
readonly EXPECTED_BRANCH="codex/p1r54-pdz-ablation-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly CELL="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly RUN_TOKEN="p1r54-pdz-ablation-llama-b100-v1"
readonly RESULT_NAMES=(
  "s05-p1r54-pdz-ablation-llama-b100-t2-prc-tech-r1-v1"
  "s05-p1r54-pdz-ablation-llama-b100-t3-prc-tech-r1-v1"
  "s05-p1r54-pdz-ablation-llama-b100-t5-prc-tech-r1-v1"
  "s05-p1r54-pdz-ablation-llama-b100-t1-direct-tech-r1-v1"
)

[[ "$#" -eq 3 ]] || { echo "P1R54 ablation arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R54 ablation CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" && "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "P1R54 ablation source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" && -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R54 ablation branch differs" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || { echo "P1R54 ablation node differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R54 ablation exactly one GPU required" >&2; exit 2; }
[[ "${CELL}" =~ ^[0-3]$ ]] || { echo "P1R54 ablation cell differs" >&2; exit 2; }
[[ -f "${FINAL_PRE_GPU_RECEIPT}" && ! -L "${FINAL_PRE_GPU_RECEIPT}" ]] || { echo "P1R54 ablation pre-GPU receipt differs" >&2; exit 2; }
readonly OUTPUT_ROOT="${REPO_ROOT}/local/odebf/results/${RESULT_NAMES[${CELL}]}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R54 ablation output exists" >&2; exit 2; }

export PROJECT_GPU_CAP=3
export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r54-pdz-ablation-${CELL}-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync python -m \
  project.run_scripts.session05_ode_bf_p1r54_pdz_ablation_b100 \
  --cell "${CELL}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --output-root "${OUTPUT_ROOT}" --final-pre-gpu-receipt "${FINAL_PRE_GPU_RECEIPT}" \
  --run-token "${RUN_TOKEN}"
