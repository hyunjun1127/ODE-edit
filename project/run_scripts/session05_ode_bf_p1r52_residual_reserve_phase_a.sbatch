#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-3%4
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:?source HEAD required}"
readonly RESULT_PARENT="${2:?result parent required}"
readonly ALIAS="${3:?model alias required}"
readonly ATTEMPT_SUFFIX="${4:-}"
readonly EXPECTED_SESSION="01a00e5d-29e8-7a01-822b-7acf43226035"
readonly RUN_TOKEN="p1r52-residual-reserve-phase-a-fp32-b10-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="codex/p1r52-residual-reserve-pcsoft-writer-v1"

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || exit 2
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || exit 2
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || exit 2
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || exit 2

readonly ARMS=(j0 rr-uniform rr-pcsoft official-alphaedit)
readonly ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
if [[ -n "${ATTEMPT_SUFFIX}" && "${ATTEMPT_SUFFIX}" != "tech-r1" && "${ATTEMPT_SUFFIX}" != "tech-r2" && "${ATTEMPT_SUFFIX}" != "tech-r3" && "${ATTEMPT_SUFFIX}" != "tech-r4" ]]; then
  exit 2
fi
readonly ATTEMPT_PART="${ATTEMPT_SUFFIX:+-${ATTEMPT_SUFFIX}}"
readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p1r52-residual-reserve-phase-a-fp32-${ALIAS}-${ARM}${ATTEMPT_PART}-v1"
[[ ! -e "${OUTPUT_ROOT}" ]] || exit 2

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-rr-${ALIAS}-${ARM}"

RUN_ARGS=(
  --alias "${ALIAS}" --arm "${ARM}" --output-root "${OUTPUT_ROOT}"
  --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
)
if [[ -n "${ATTEMPT_SUFFIX}" ]]; then
  RUN_ARGS+=(--attempt-suffix "${ATTEMPT_SUFFIX}")
fi
exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_residual_reserve_phase_a \
  "${RUN_ARGS[@]}"
