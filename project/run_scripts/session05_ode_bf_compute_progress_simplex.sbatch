#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly EXPECTED_BRANCH="codex/p1r23-progress-simplex-compute-a1"
readonly RUN_TOKEN="p1r23-compute-progress-simplex-a1-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 2 ]] || { echo "Compute-A1 requires source and result parent" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "Compute-A1 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "Compute-A1 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "Compute-A1 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Compute-A1 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || { echo "Compute-A1 node differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "Compute-A1 exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst)
readonly ROLES=(COMPUTE_PROGRESS_SIMPLEX_BG_PAIR COMPUTE_PROGRESS_SIMPLEX_RS_PAIR COMPUTE_PROGRESS_SIMPLEX_BG_PAIR COMPUTE_PROGRESS_SIMPLEX_RS_PAIR)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly ROLE="${ROLES[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_NAME="$(${EASYEDIT_ROOT}/.venv/bin/python - <<PY
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import expected_p1r23_result_name
print(expected_p1r23_result_name("${MODEL}", batch_size=10, role="${ROLE}"))
PY
)"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Compute-A1 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=41 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r23-compute-a1"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_compute_progress_simplex \
  --model "${MODEL}" --role "${ROLE}" --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
