#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server2
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly EXPECTED_SESSION="019fe491-954b-70a0-8ba8-0588e9f8d741"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly RUN_TOKEN="atomic-runtime-optimization-p1r22-v1"

[[ "$#" -eq 2 ]] || { echo "P1R22 C0/C1 requires source and result parent" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R22 C0/C1 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R22 C0/C1 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "codex/odeeditsh2-s05-atomic-runtime-optimization-p1r22-v1" ]] || { echo "P1R22 C0/C1 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R22 C0/C1 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R22 C0/C1 exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-atomic-runtime-optimization-p1r22-c0c1-${MODEL}-v1"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R22 C0/C1 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r22-c0c1"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_atomic_runtime_optimization \
  --model "${MODEL}" --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}" --conformance-only
