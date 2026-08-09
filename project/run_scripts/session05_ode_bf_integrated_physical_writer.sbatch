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

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd -P)"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_PARENT="f0f48f964750dbd11483ea327ae679c24c710fde"
readonly EXPECTED_SESSION="019fe489-c968-75f3-9965-7cfbc26c0a99"
readonly EXPECTED_BRANCH="codex/odeeditsh1-s05-integrated-physical-writer-p1r14-v1"
readonly MODEL_ALIAS="${1:-}"
readonly OUTPUT_ROOT="${2:-}"
readonly SOURCE_HEAD="${3:-}"
readonly RUN_TOKEN="${4:-}"

[[ "$#" -eq 4 ]] || { echo "locked P1R14 invocation requires four arguments" >&2; exit 2; }
[[ "${MODEL_ALIAS}" == "llama3-8b-inst" ]] || {
  echo "server1 P1R14 launcher is Llama-only" >&2
  exit 2
}
[[ -n "${OUTPUT_ROOT}" && "${RUN_TOKEN}" == "integrated-physical-writer-p1r14-v1" ]] || {
  echo "missing locked P1R14 invocation" >&2
  exit 2
}
[[ "${SOURCE_HEAD}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "P1R14 source head must be a lowercase SHA-1" >&2
  exit 2
}
[[ "$(pwd -P)" == "${REPO_ROOT}" ]] || { echo "P1R14 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P1R14 source differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD^)" == "${EXPECTED_PARENT}" ]] || { echo "P1R14 parent differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P1R14 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P1R14 tracked source dirty" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P1R14 output root already exists" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ -x /usr/local/bin/uv && -x "${EASYEDIT_ROOT}/.venv/bin/python" ]] || {
  echo "P1R14 offline runtime unavailable" >&2
  exit 2
}
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
  echo "P1R14 requires exactly one visible GPU" >&2
  exit 2
}

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=41
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-s05-p1r14"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_integrated_physical_writer \
  --model "${MODEL_ALIAS}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
