#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server2
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly STAGE="${3:-}"
readonly EXPECTED_SESSION="019fe491-954b-70a0-8ba8-0588e9f8d741"
readonly RUN_TOKEN="scalable-batched-runtime-p1r23-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 3 ]] || { echo "P1R23 requires source, result parent, and stage" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R23 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R23 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "codex/odeeditsh2-s05-scalable-batched-runtime-p1r23-a5-rs-v1" ]] || { echo "P1R23 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R23 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R23 exactly one GPU required" >&2; exit 2; }

case "${STAGE}" in
  calibration)
    readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
    readonly ROLES=(CALIBRATION CALIBRATION)
    readonly BATCH_SIZE=10
    ;;
  b10)
    readonly MODELS=(
      llama3-8b-inst llama3-8b-inst llama3-8b-inst
      qwen2.5-7b-inst qwen2.5-7b-inst qwen2.5-7b-inst
    )
    readonly ROLES=(
      ODE_BF_K8_PAIR OPTIMIZED_NATIVE_K1 OFFICIAL_NATIVE
      ODE_BF_K8_PAIR OPTIMIZED_NATIVE_K1 OFFICIAL_NATIVE
    )
    readonly BATCH_SIZE=10
    ;;
  rs-b10)
    readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
    readonly ROLES=(ODE_BF_K8_RS_PAIR ODE_BF_K8_RS_PAIR)
    readonly BATCH_SIZE=10
    ;;
  b100)
    readonly MODELS=(
      llama3-8b-inst llama3-8b-inst llama3-8b-inst llama3-8b-inst
      qwen2.5-7b-inst qwen2.5-7b-inst qwen2.5-7b-inst qwen2.5-7b-inst
    )
    readonly ROLES=(
      ODE_BF_K8_PAIR ODE_BF_K8_RS_PAIR OPTIMIZED_NATIVE_K1 OFFICIAL_NATIVE
      ODE_BF_K8_PAIR ODE_BF_K8_RS_PAIR OPTIMIZED_NATIVE_K1 OFFICIAL_NATIVE
    )
    readonly BATCH_SIZE=100
    ;;
  *) echo "P1R23 stage differs" >&2; exit 2 ;;
esac

readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly ROLE="${ROLES[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_NAME="$(${EASYEDIT_ROOT}/.venv/bin/python - <<PY
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import expected_p1r23_result_name
print(expected_p1r23_result_name("${MODEL}", batch_size=${BATCH_SIZE}, role="${ROLE}"))
PY
)"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R23 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r23"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_scalable_batched_runtime \
  --model "${MODEL}" --batch-size "${BATCH_SIZE}" --role "${ROLE}" \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
