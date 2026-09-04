#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server4
#SBATCH --job-name=odeedit_orbode_seq_s4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodelist=server4
#SBATCH --mem=60416M
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%2
#SBATCH --export=NONE
#SBATCH --output=local/logs/orbode-sequential-b100x10/%A_%a.out
#SBATCH --error=local/logs/orbode-sequential-b100x10/%A_%a.err

set -euo pipefail
umask 077

[[ "$#" -eq 6 ]] || { echo "ORBODE sequential arguments differ" >&2; exit 2; }
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="$1"
readonly SOURCE_TREE="$2"
readonly OUTPUT_PARENT="$3"
readonly LOG_ROOT="$4"
readonly PREFLIGHT_RECEIPT="$5"
readonly LOCAL_CAPS="$6"
readonly EXPECTED_SESSION="01a04939-b5c7-7a03-ba2d-ef3343d62cfd"
readonly EASYEDIT_SOURCE_ROOT="/data/janghj/EasyEdit-stock-14cea824"
readonly EASYEDIT_ARTIFACT_ROOT="/data/janghj/EasyEdit"
readonly HF_HUB_CACHE="/data/janghj/.cache/huggingface/hub"
readonly PYTHON_BIN="/data/janghj/EasyEdit/.venv/bin/python"

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "ORBODE sequential cwd differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "ORBODE sequential source HEAD differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "ORBODE sequential source tree differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "ORBODE sequential tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "ORBODE exactly one GPU required" >&2; exit 2; }
[[ -d "${OUTPUT_PARENT}" && ! -L "${OUTPUT_PARENT}" ]] || { echo "ORBODE output parent differs" >&2; exit 2; }
[[ -d "${LOG_ROOT}" && ! -L "${LOG_ROOT}" ]] || { echo "ORBODE log root differs" >&2; exit 2; }
[[ -f "${PREFLIGHT_RECEIPT}" && ! -L "${PREFLIGHT_RECEIPT}" ]] || { echo "ORBODE preflight receipt differs" >&2; exit 2; }
[[ "$(stat -c '%a' "${PREFLIGHT_RECEIPT}")" == "600" ]] || { echo "ORBODE preflight receipt mode differs" >&2; exit 2; }
[[ -f "${LOCAL_CAPS}" && ! -L "${LOCAL_CAPS}" ]] || { echo "ORBODE cap registry differs" >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?array task id required}" in
  0) readonly RUN_TOKEN="s06-orbode-sequential-server4-cell0-llama3-8b-inst-memit-r1" ;;
  1) readonly RUN_TOKEN="s06-orbode-sequential-server4-cell1-llama3-8b-inst-alphaedit-r1" ;;
  2) readonly RUN_TOKEN="s06-orbode-sequential-server4-cell2-qwen2.5-7b-inst-memit-r1" ;;
  3) readonly RUN_TOKEN="s06-orbode-sequential-server4-cell3-qwen2.5-7b-inst-alphaedit-r1" ;;
  *) echo "ORBODE sequential array mapping differs" >&2; exit 2 ;;
esac

export HF_HOME="/data/janghj/.cache/huggingface"
export HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_SOURCE_ROOT}"
export MPLCONFIGDIR="${REPO_ROOT}/local/orbode-sequential/matplotlib-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

readonly IMPORTED_EASYEDIT="$(${PYTHON_BIN} -c 'import pathlib,easyeditor; print(pathlib.Path(easyeditor.__file__).resolve())')"
[[ "${IMPORTED_EASYEDIT}" == "${EASYEDIT_SOURCE_ROOT}"/* ]] || { echo "ORBODE imported EasyEdit source differs" >&2; exit 2; }

exec "${PYTHON_BIN}" -m project.run_scripts.session06_orbode_sequential_server4 run-cell \
  --repo-root "${REPO_ROOT}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" \
  --easyedit-source-root "${EASYEDIT_SOURCE_ROOT}" --easyedit-artifact-root "${EASYEDIT_ARTIFACT_ROOT}" \
  --hf-hub-cache "${HF_HUB_CACHE}" --local-caps "${LOCAL_CAPS}" \
  --output-parent "${OUTPUT_PARENT}" --log-root "${LOG_ROOT}" \
  --preflight-receipt "${PREFLIGHT_RECEIPT}" --cell-id "${SLURM_ARRAY_TASK_ID}" \
  --run-token "${RUN_TOKEN}"

