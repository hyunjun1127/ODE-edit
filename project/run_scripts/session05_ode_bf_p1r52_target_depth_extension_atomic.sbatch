#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server2
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-2%3
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly EXPECTED_SESSION="01a00e5c-f7ae-72a2-98b2-b8b0907168b4"
readonly RUN_TOKEN="p1r52-target-depth-atomic-extension-il8-il10-il15-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="codex/p1r52-target-depth-il8-il10-il15-v1"

[[ "$#" -eq 2 ]] || { echo "P1R52 target-depth extension arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P1R52 target-depth extension CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R52 target-depth extension source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R52 target-depth extension branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R52 target-depth extension tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P1R52 target-depth extension exactly one GPU required" >&2; exit 2; }

readonly DEPTHS=(IL8-FULL IL10-FULL IL15-FULL)
readonly DEPTH="${DEPTHS[${SLURM_ARRAY_TASK_ID}]}"
readonly DEPTH_LOWER="${DEPTH,,}"
readonly DEPTH_SLUG="${DEPTH_LOWER//-/}"
readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p1r52-target-depth-atomic-extension-b10x10-llama3-8b-inst-soft-${DEPTH_SLUG}-v1"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R52 target-depth extension output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-target-depth-extension-${DEPTH_SLUG}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_target_depth_extension_atomic \
  --model llama3-8b-inst --arm soft --depth "${DEPTH}" \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" --run-token "${RUN_TOKEN}"
