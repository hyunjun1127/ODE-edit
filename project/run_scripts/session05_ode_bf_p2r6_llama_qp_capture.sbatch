#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=02:00:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly ATTEMPT_SUFFIX="${3:-}"
readonly EXPECTED_SESSION="019fe491-16f4-7bd3-adf5-4b1eb4a57d1f"
readonly EXPECTED_BRANCH="codex/p2r6-llama-outer1-qp-capture-v1"
readonly RUN_TOKEN="p2r6-llama-outer1-qp-capture-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 3 ]] || { echo "P2R6 capture arguments differ" >&2; exit 2; }
[[ "${ATTEMPT_SUFFIX}" == "a1" ]] || { echo "P2R6 capture attempt differs" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P2R6 capture CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P2R6 capture source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P2R6 capture branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P2R6 capture tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P2R6 capture exactly one GPU required" >&2; exit 2; }

readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p2r6-llama-outer1-qp-capture-llama3-8b-inst-case-05-ar-cap-${ATTEMPT_SUFFIX}-v1"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P2R6 capture output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=48 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p2r6-llama-capture"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p2r6_llama_qp_capture \
  --model llama3-8b-inst --case-index 5 \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}" --attempt-suffix "${ATTEMPT_SUFFIX}"
