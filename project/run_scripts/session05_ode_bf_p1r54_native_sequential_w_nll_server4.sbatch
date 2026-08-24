#!/usr/bin/env bash
#SBATCH --job-name=p1r54_native_wnll_s4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server4
#SBATCH --mem=65000M
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly FINAL_PRE_GPU_RECEIPT="${3:-}"
readonly RESULT_PARENT="${4:-}"
readonly EXPECTED_SESSION="codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d"
readonly EXPECTED_BRANCH="codex/server4-p1r54-native-seq-w-nll-v1"
readonly EASYEDIT_ROOT="/data/janghj/EasyEdit"
readonly RUN_TOKEN="p1r54-native-sequential-w-nll-backfill-v1"
readonly RESULTS=(
  "s05-p1r52-c-writer-phase1-full-fp32-sequential-official-alphaedit-10xb100-tech-r1-v1"
  "s05-p1r52-c-writer-phase1-full-fp32-sequential-official-memit-10xb100-tech-r1-v1"
)

[[ "$#" -eq 4 ]] || { echo "native W-NLL arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "native W-NLL CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "native W-NLL source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^{tree})" == "${SOURCE_TREE}" ]] || { echo "native W-NLL tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "native W-NLL branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "native W-NLL tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "server4" ]] || { echo "native W-NLL node differs" >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[01]$ ]] || { echo "native W-NLL cell differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "native W-NLL exactly one GPU required" >&2; exit 2; }
[[ -f "${FINAL_PRE_GPU_RECEIPT}" && ! -L "${FINAL_PRE_GPU_RECEIPT}" ]] || { echo "native W-NLL pre-GPU receipt differs" >&2; exit 2; }
[[ -d "${RESULT_PARENT}" && ! -L "${RESULT_PARENT}" ]] || { echo "native W-NLL result parent differs" >&2; exit 2; }
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULTS[${SLURM_ARRAY_TASK_ID}]}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "native W-NLL output exists" >&2; exit 2; }

export PROJECT_GPU_CAP=4
export HF_HOME="/data/janghj/.cache/huggingface"
export HF_HUB_CACHE="/data/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r54-native-wnll-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec "${EASYEDIT_ROOT}/.venv/bin/python" -m \
  project.run_scripts.session05_ode_bf_p1r54_native_sequential_w_nll \
  --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --output-root "${OUTPUT_ROOT}" --result-parent "${RESULT_PARENT}" \
  --final-pre-gpu-receipt "${FINAL_PRE_GPU_RECEIPT}" \
  --cell "${SLURM_ARRAY_TASK_ID}" --run-token "${RUN_TOKEN}"
