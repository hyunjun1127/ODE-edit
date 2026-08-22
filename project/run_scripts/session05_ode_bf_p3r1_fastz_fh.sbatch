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
readonly ALIAS="${3:-}"
readonly CASE_INDEX="${4:-}"
readonly EXPECTED_SESSION="01a00e5c-f7ae-72a2-98b2-b8b0907168b4"
readonly EXPECTED_BRANCH="codex/p3r1-two-timescale-fastz-fh-c013-fp32-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 4 ]] || { echo "P3R1 arguments differ" >&2; exit 2; }
[[ "${ALIAS}" == "llama3-8b-inst" || "${ALIAS}" == "qwen2.5-7b-inst" ]] || { echo "P3R1 alias differs" >&2; exit 2; }
[[ "${CASE_INDEX}" =~ ^(0?[1-9]|10)$ ]] || { echo "P3R1 case differs" >&2; exit 2; }
readonly CASE_PADDED="$(printf '%02d' "$((10#${CASE_INDEX}))")"
readonly RESULT_NAME="s05-p3r1-two-timescale-fastz-fh-c013-fp32-${ALIAS}-case-${CASE_PADDED}-v1"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAME}"

[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P3R1 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P3R1 source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P3R1 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P3R1 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P3R1 exactly one GPU required" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P3R1 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p3r1-${ALIAS}-${CASE_PADDED}-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p3r1_fastz_fh \
  --alias "${ALIAS}" --case-index "$((10#${CASE_INDEX}))" \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}"
