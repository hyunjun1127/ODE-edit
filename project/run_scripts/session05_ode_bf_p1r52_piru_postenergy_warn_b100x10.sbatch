#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=47:59:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly OUTPUT_ROOT="${2:-}"
readonly EXPECTED_SESSION="01a00e5d-29e8-7a01-822b-7acf43226035"
readonly EXPECTED_BRANCH="codex/p1r52-piru-seq-postenergy-warn-r1-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"

[[ "$#" -eq 2 ]] || { echo "PIR-U post-energy-WARN arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "PIR-U post-energy-WARN CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "PIR-U post-energy-WARN source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "PIR-U post-energy-WARN branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "PIR-U post-energy-WARN tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "PIR-U post-energy-WARN exactly one GPU required" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "PIR-U post-energy-WARN output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-piru-postenergy-warn"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_piru_postenergy_warn_b100x10 \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --run-token p1r52-pir-u-sequential-postenergy-warn-r1-v1
