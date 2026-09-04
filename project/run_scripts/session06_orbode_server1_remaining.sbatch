#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server1
#SBATCH --job-name=odeedit_orbode_r1_9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=182272M
#SBATCH --time=14-00:00:00
#SBATCH --array=0-3%4
#SBATCH --export=NONE
#SBATCH --output=local/logs/ordered-response-barrier-ode/remaining-%A_%a.out
#SBATCH --error=local/logs/ordered-response-barrier-ode/remaining-%A_%a.err

set -euo pipefail
umask 077

[[ "$#" -eq 6 ]] || { echo "ORBODE remaining arguments differ" >&2; exit 2; }
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="$1"
readonly SOURCE_TREE="$2"
readonly OUTPUT_PARENT="$3"
readonly PREFLIGHT_RECEIPT="$4"
readonly AUTHORITATIVE_ROOT="$5"
readonly ROUND0_ROOT="$6"
readonly TASK_OUTPUT_ROOT="${OUTPUT_PARENT}/task-${SLURM_ARRAY_TASK_ID:?array task id required}"
readonly EXPECTED_SESSION="01a04939-f93a-7b50-bca0-65438eab2062"
readonly EASYEDIT_SOURCE_ROOT="/mnt/raid5/janghj/.codex/worktrees/easyeditsh1-official-readonly-v1"
readonly EASYEDIT_ARTIFACT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
readonly PYTHON_BIN="/mnt/raid5/janghj/EasyEdit/.venv/bin/python"

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "ORBODE remaining cwd differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "ORBODE remaining source HEAD differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "ORBODE remaining source tree differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "ORBODE remaining tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "ORBODE exactly one GPU required" >&2; exit 2; }
[[ -d "${OUTPUT_PARENT}" && ! -L "${OUTPUT_PARENT}" ]] || { echo "ORBODE remaining output parent is not presealed" >&2; exit 2; }
[[ -d "${ROUND0_ROOT}" && ! -L "${ROUND0_ROOT}" ]] || { echo "ORBODE round0 gate root differs" >&2; exit 2; }
[[ ! -e "${TASK_OUTPUT_ROOT}" && ! -L "${TASK_OUTPUT_ROOT}" ]] || { echo "ORBODE remaining task output exists" >&2; exit 2; }
[[ -f "${PREFLIGHT_RECEIPT}" && ! -L "${PREFLIGHT_RECEIPT}" ]] || { echo "ORBODE preflight receipt differs" >&2; exit 2; }
[[ "$(stat -c '%a' "${PREFLIGHT_RECEIPT}")" == "600" ]] || { echo "ORBODE preflight receipt mode differs" >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly RUN_TOKEN="s06-orbode-server1-remaining-cell0-llama3-8b-inst-memit-r1" ;;
  1) readonly RUN_TOKEN="s06-orbode-server1-remaining-cell1-llama3-8b-inst-alphaedit-r1" ;;
  2) readonly RUN_TOKEN="s06-orbode-server1-remaining-cell2-qwen2.5-7b-inst-memit-r1" ;;
  3) readonly RUN_TOKEN="s06-orbode-server1-remaining-cell3-qwen2.5-7b-inst-alphaedit-r1" ;;
  *) echo "ORBODE remaining array mapping differs" >&2; exit 2 ;;
esac

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_SOURCE_ROOT}"
export MPLCONFIGDIR="${REPO_ROOT}/local/ordered-response-barrier-ode/matplotlib-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

readonly IMPORTED_EASYEDIT="$(${PYTHON_BIN} -c 'import pathlib,easyeditor; print(pathlib.Path(easyeditor.__file__).resolve())')"
[[ "${IMPORTED_EASYEDIT}" == "${EASYEDIT_SOURCE_ROOT}"/* ]] || { echo "ORBODE imported EasyEdit source differs: ${IMPORTED_EASYEDIT}" >&2; exit 2; }

exec "${PYTHON_BIN}" -m project.run_scripts.session06_orbode_server1 run-cell \
  --repo-root "${REPO_ROOT}" --authoritative-root "${AUTHORITATIVE_ROOT}" \
  --easyedit-source-root "${EASYEDIT_SOURCE_ROOT}" --easyedit-artifact-root "${EASYEDIT_ARTIFACT_ROOT}" \
  --hf-hub-cache "${HF_HUB_CACHE}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --wave remaining --round0-root "${ROUND0_ROOT}" --preflight-receipt "${PREFLIGHT_RECEIPT}" \
  --output-root "${TASK_OUTPUT_ROOT}" --cell-id "${SLURM_ARRAY_TASK_ID}" --run-token "${RUN_TOKEN}"
