#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server1
#SBATCH --job-name=odeedit_orbode_pre_ns
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=81920M
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE
#SBATCH --output=local/logs/ordered-response-barrier-ode/preedit-ns-%A_%a.out
#SBATCH --error=local/logs/ordered-response-barrier-ode/preedit-ns-%A_%a.err

set -euo pipefail
umask 077

[[ "$#" -eq 3 ]] || { echo "ORBODE pre-edit NS arguments differ" >&2; exit 2; }
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="$1"
readonly SOURCE_TREE="$2"
readonly OUTPUT_PARENT="$3"
readonly CELL_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
readonly TASK_OUTPUT_ROOT="${OUTPUT_PARENT}/task-${CELL_ID}"
readonly EXPECTED_SESSION="01a04939-f93a-7b50-bca0-65438eab2062"
readonly EASYEDIT_ARTIFACT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
readonly PYTHON_BIN="/mnt/raid5/janghj/EasyEdit/.venv/bin/python"

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "ORBODE pre-edit NS cwd differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "ORBODE pre-edit NS source HEAD differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "ORBODE pre-edit NS source tree differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "ORBODE pre-edit NS tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "ORBODE pre-edit NS exactly one GPU required" >&2; exit 2; }
[[ -d "${OUTPUT_PARENT}" && ! -L "${OUTPUT_PARENT}" ]] || { echo "ORBODE pre-edit NS output parent differs" >&2; exit 2; }
[[ ! -e "${TASK_OUTPUT_ROOT}" && ! -L "${TASK_OUTPUT_ROOT}" ]] || { echo "ORBODE pre-edit NS task root exists" >&2; exit 2; }

case "${CELL_ID}" in
  0) readonly RUN_TOKEN="s06-orbode-round0-preedit-ns-llama3-8b-inst-v1" ;;
  1) readonly RUN_TOKEN="s06-orbode-round0-preedit-ns-qwen2.5-7b-inst-v1" ;;
  *) echo "ORBODE pre-edit NS array mapping differs" >&2; exit 2 ;;
esac

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}"

exec "${PYTHON_BIN}" -m project.run_scripts.ordered_response_barrier_ode.preedit_ns_backfill run-cell \
  --repo-root "${REPO_ROOT}" --easyedit-artifact-root "${EASYEDIT_ARTIFACT_ROOT}" \
  --hf-hub-cache "${HF_HUB_CACHE}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --output-root "${TASK_OUTPUT_ROOT}" --cell-id "${CELL_ID}" --run-token "${RUN_TOKEN}"
