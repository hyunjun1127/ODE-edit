#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=100000M
#SBATCH --time=47:59:00
#SBATCH --array=0-5%3
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly OUTPUT_PARENT="${3:-}"
readonly TASK_OUTPUT_PARENT="${OUTPUT_PARENT}/task-${SLURM_ARRAY_TASK_ID:?array task id required}"
readonly EXPECTED_SESSION="01a0278d-456b-7d20-820f-63fc80a57b8d"
readonly EXPECTED_BRANCH="codex/bgode-r2-rho-free-prefix-event-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 3 ]] || { echo "BGODE-R2 Stage-C arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "BGODE-R2 Stage-C cwd differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "BGODE-R2 Stage-C source HEAD differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "BGODE-R2 Stage-C source tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "BGODE-R2 Stage-C branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "BGODE-R2 Stage-C tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "BGODE-R2 Stage-C exactly one GPU required" >&2; exit 2; }
[[ -d "${OUTPUT_PARENT}" && ! -L "${OUTPUT_PARENT}" ]] || { echo "BGODE-R2 Stage-C output parent is not presealed" >&2; exit 2; }
[[ ! -e "${TASK_OUTPUT_PARENT}" ]] || { echo "BGODE-R2 Stage-C task output exists" >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?array task id required}" in
  0) MODEL_ALIAS="llama3-8b-inst"; TOPOLOGY="both-multi-unequal-nonprefix" ;;
  1) MODEL_ALIAS="llama3-8b-inst"; TOPOLOGY="source-prefix-target" ;;
  2) MODEL_ALIAS="llama3-8b-inst"; TOPOLOGY="target-prefix-source" ;;
  3) MODEL_ALIAS="qwen2.5-7b-inst"; TOPOLOGY="both-multi-unequal-nonprefix" ;;
  4) MODEL_ALIAS="qwen2.5-7b-inst"; TOPOLOGY="source-prefix-target" ;;
  5) MODEL_ALIAS="qwen2.5-7b-inst"; TOPOLOGY="target-prefix-source" ;;
  *) echo "BGODE-R2 Stage-C array mapping differs" >&2; exit 2 ;;
esac
readonly MODEL_ALIAS TOPOLOGY
readonly RUN_TOKEN="s05-bgode-r2-stage-c-${MODEL_ALIAS}-${TOPOLOGY}-fp64-two-equality-v1"

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-bgode-r2-stage-c-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_bgode_r2_stage_c \
  --easyedit-root "${EASYEDIT_ROOT}" --output-root "${TASK_OUTPUT_PARENT}" \
  --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --model-alias "${MODEL_ALIAS}" --topology "${TOPOLOGY}" --run-token "${RUN_TOKEN}"
