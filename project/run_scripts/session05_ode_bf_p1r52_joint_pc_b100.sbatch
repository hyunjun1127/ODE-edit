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
readonly STAGE="${3:-}"
readonly EXPECTED_SESSION="01a00e5c-f7ae-72a2-98b2-b8b0907168b4"
readonly EXPECTED_BRANCH="codex/p1r52-joint-pc-c1-c2-independent-b100-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly RUN_TOKEN="p1r52-joint-pc-c1-c2-independent-b100-v1"

[[ "$#" -eq 3 ]] || { echo "joint P/C arguments differ" >&2; exit 2; }
[[ "${STAGE}" == pilot || "${STAGE}" == pilot-tech-r1 || "${STAGE}" == production ]] || { echo "joint P/C stage differs" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "joint P/C CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "joint P/C source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "joint P/C branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "joint P/C tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "joint P/C exactly one GPU required" >&2; exit 2; }

if [[ "${STAGE}" == pilot ]]; then
  readonly RESULT_NAME="s05-p1r52-joint-pc-c1-c2-pilot-b100-v1"
  readonly ROLE_ARGS=(--pilot)
elif [[ "${STAGE}" == pilot-tech-r1 ]]; then
  readonly RESULT_NAME="s05-p1r52-joint-pc-c1-c2-pilot-b100-tech-r1-v1"
  readonly ROLE_ARGS=(--pilot-tech-r1)
else
  readonly CASE_INDEX="$((SLURM_ARRAY_TASK_ID + 1))"
  printf -v RESULT_NAME 's05-p1r52-joint-pc-c1-c2-independent-b100-case-%02d-v1' "${CASE_INDEX}"
  readonly RESULT_NAME
  readonly ROLE_ARGS=(--case-index "${CASE_INDEX}")
fi
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAME}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "joint P/C output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-joint-pc-${STAGE}-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_joint_pc_b100 \
  "${ROLE_ARGS[@]}" --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
