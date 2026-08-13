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
readonly PHASE="${3:-}"
readonly EXPECTED_SESSION="019fe491-954b-70a0-8ba8-0588e9f8d741"
readonly RUN_TOKEN="p1r30-debt-priority-a0-relative-barrier-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="codex/odeeditsh2-s05-p1r30-debt-priority-a0-barrier-v1"

[[ "$#" -eq 3 ]] || { echo "P1R30 requires source, result parent and phase" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R30 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R30 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R30 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R30 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R30 exactly one GPU required" >&2; exit 2; }

case "${PHASE}" in
  b1)
    readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
    readonly ROLES=(P1R30_B1_RS_REFERENCE_SOFT P1R30_B1_RS_REFERENCE_SOFT)
    ;;
  b10-neutral)
    readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
    readonly ROLES=(P1R30_B10_RS_DEBT_NEUTRAL P1R30_B10_RS_DEBT_NEUTRAL)
    ;;
  full-matrix-completion)
    readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst llama3-8b-inst qwen2.5-7b-inst)
    readonly ROLES=(P1R30_B10_RS_DEBT_SOFT P1R30_B10_RS_DEBT_SOFT P1R30_B10_BG_PAIR P1R30_B10_BG_PAIR)
    ;;
  *) echo "P1R30 phase differs" >&2; exit 2 ;;
esac
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly ROLE="${ROLES[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_NAME="$(${EASYEDIT_ROOT}/.venv/bin/python - <<PY
from project.run_scripts.ode_bf.p1r30_debt_priority_panel import expected_p1r30_result_name
print(expected_p1r30_result_name("${MODEL}", "${ROLE}"))
PY
)"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R30 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r30"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r30_debt_priority \
  --model "${MODEL}" --role "${ROLE}" --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
