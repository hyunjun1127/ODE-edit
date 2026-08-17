#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=47:59:00
#SBATCH --array=0-3%4
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly ATTEMPT_SUFFIX="${3:-}"
readonly EXPECTED_SESSION="01a00e5d-29e8-7a01-822b-7acf43226035"
readonly EXPECTED_BRANCH="codex/p1r52-llama-seq-10xb100-fourarm-r1-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 3 ]] || { echo "P1R52 B100x10 arguments differ" >&2; exit 2; }
[[ "${ATTEMPT_SUFFIX}" == tech-r1 || "${ATTEMPT_SUFFIX}" == tech-r2 || "${ATTEMPT_SUFFIX}" == tech-r3 || "${ATTEMPT_SUFFIX}" == tech-r3-release-r1 ]] || { echo "P1R52 B100x10 attempt differs" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R52 B100x10 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R52 B100x10 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R52 B100x10 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R52 B100x10 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R52 B100x10 exactly one GPU required" >&2; exit 2; }

if [[ "${ATTEMPT_SUFFIX}" == tech-r1 ]]; then
  readonly RUN_TOKEN="p1r52-llama-sequential-10xb100-fourcell-tech-r1-v1"
  readonly ROLES=(
    official-memit-sequential
    native-alphaedit-sequential-cache-on-corrected
    r52-soft-sequential-h
    r52-soft-sequential-alphacache-on-structuralh-off
  )
  readonly RESULTS=(
    s05-p1r52-official-memit-sequential-10xb100-tech-r1-v1
    s05-p1r52-official-alphaedit-sequential-cache-on-10xb100-tech-r1-v1
    s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-tech-r1-v1
    s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-tech-r1-v1
  )
elif [[ "${ATTEMPT_SUFFIX}" == tech-r2 ]]; then
  readonly RUN_TOKEN="p1r52-llama-sequential-10xb100-r52-tech-r2-v1"
  readonly ROLES=(
    r52-soft-sequential-h
    r52-soft-sequential-alphacache-on-structuralh-off
  )
  readonly RESULTS=(
    s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-tech-r2-v1
    s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-tech-r2-v1
  )
elif [[ "${ATTEMPT_SUFFIX}" == tech-r3 ]]; then
  readonly RUN_TOKEN="p1r52-llama-sequential-10xb100-r52-tech-r3-v1"
  readonly ROLES=(
    r52-soft-sequential-h
    r52-soft-sequential-alphacache-on-structuralh-off
  )
  readonly RESULTS=(
    s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-tech-r3-v1
    s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-tech-r3-v1
  )
else
  readonly RUN_TOKEN="p1r52-llama-sequential-10xb100-r52-tech-r3-release-r1-v1"
  readonly ROLES=(
    r52-soft-sequential-h
    r52-soft-sequential-alphacache-on-structuralh-off
  )
  readonly RESULTS=(
    s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-tech-r3-release-r1-v1
    s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-tech-r3-release-r1-v1
  )
fi
readonly ROLE="${ROLES[${SLURM_ARRAY_TASK_ID}]}"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULTS[${SLURM_ARRAY_TASK_ID}]}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R52 B100x10 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-b100x10-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_sequential_b100x10_fourcell \
  --role "${ROLE}" --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" --attempt-suffix "${ATTEMPT_SUFFIX}" \
  --run-token "${RUN_TOKEN}"
