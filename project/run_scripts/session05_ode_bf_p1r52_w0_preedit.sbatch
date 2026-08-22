#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=06:00:00
#SBATCH --export=NONE

set -euo pipefail
umask 077
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly OUTPUT_ROOT="${3:-}"
readonly EXPECTED_SESSION="01a0278d-456b-7d20-820f-63fc80a57b8d"
readonly EXPECTED_BRANCH="codex/p1r52-c-writer-phase1-preedit-r1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly RUN_TOKEN="p1r52-phase1-llama3-w0-preedit-1000-tech-r1-v1"
[[ "$#" -eq 3 ]] || { echo "W0 pre-edit arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "W0 pre-edit CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "W0 pre-edit source HEAD differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "W0 pre-edit source tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "W0 pre-edit branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "W0 pre-edit tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "W0 pre-edit exactly one GPU required" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "W0 pre-edit output exists" >&2; exit 2; }
export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-w0-preedit-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"
exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_w0_preedit \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --source-tree "${SOURCE_TREE}" --run-token "${RUN_TOKEN}"
