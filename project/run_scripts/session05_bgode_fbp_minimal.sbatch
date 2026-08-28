#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=100000M
#SBATCH --time=47:59:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly OUTPUT_PARENT="${3:-}"
readonly TASK_OUTPUT_PARENT="${OUTPUT_PARENT}/task-${SLURM_ARRAY_TASK_ID:?array task id required}"
readonly EXPECTED_SESSION="01a04664-c753-7461-b914-b4ebf24a581a"
readonly EXPECTED_BRANCH="codex/bgode-fbp-minimal-barrier-ode-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 3 ]] || { echo "fixed-basis experiment arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "fixed-basis cwd differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "fixed-basis source HEAD differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "fixed-basis source tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "fixed-basis branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "fixed-basis tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "fixed-basis exactly one GPU required" >&2; exit 2; }
[[ -d "${OUTPUT_PARENT}" && ! -L "${OUTPUT_PARENT}" ]] || { echo "fixed-basis output parent is not presealed" >&2; exit 2; }
[[ ! -e "${TASK_OUTPUT_PARENT}" ]] || { echo "fixed-basis task output exists" >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?array task id required}" in
  0)
    readonly MODEL_ALIAS="llama3-8b-inst"
    readonly RUN_ID="s05-bgode-fbp-minimal-llama3-8b-inst-8case-four-arm-v1"
    ;;
  1)
    readonly MODEL_ALIAS="qwen2.5-7b-inst"
    readonly RUN_ID="s05-bgode-fbp-minimal-qwen2.5-7b-inst-8case-four-arm-v1"
    ;;
  *) echo "fixed-basis array mapping differs" >&2; exit 2 ;;
esac

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-bgode-fbp-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_bgode_fbp_minimal \
  --easyedit-root "${EASYEDIT_ROOT}" --output-root "${TASK_OUTPUT_PARENT}" \
  --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --model-alias "${MODEL_ALIAS}" --run-id "${RUN_ID}"
