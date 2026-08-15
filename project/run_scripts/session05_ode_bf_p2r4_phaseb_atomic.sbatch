#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-3%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly CASE_COUNT="${3:-}"
readonly ATTEMPT_SUFFIX="${4:-}"
readonly EXPECTED_SESSION="019fe491-16f4-7bd3-adf5-4b1eb4a57d1f"
readonly RUN_TOKEN="p2r4-phaseb-clamp-on-off-p2r2-v2-writer-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly EXPECTED_BRANCH="codex/odeeditsh2-s05-p2r4-phaseb-clamp-on-off-writer-v1"

[[ "$#" -eq 3 || "$#" -eq 4 ]] || { echo "P2R4 Phase-B arguments differ" >&2; exit 2; }
[[ "${CASE_COUNT}" == 1 || "${CASE_COUNT}" == 10 ]] || { echo "P2R4 Phase-B case count differs" >&2; exit 2; }
[[ -z "${ATTEMPT_SUFFIX}" || "${ATTEMPT_SUFFIX}" =~ ^[a-zA-Z0-9-]+$ ]] || { echo "P2R4 Phase-B attempt suffix differs" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "P2R4 Phase-B CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "P2R4 Phase-B source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "P2R4 Phase-B branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "P2R4 Phase-B tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "P2R4 Phase-B exactly one GPU required" >&2; exit 2; }

readonly MODELS=(llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst)
readonly CLAMP_POLICIES=(ON OFF ON OFF)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CLAMP_POLICY="${CLAMP_POLICIES[${SLURM_ARRAY_TASK_ID}]}"
readonly PHASE="$([[ "${CASE_COUNT}" == 1 ]] && printf 'sealed-b10-smoke' || printf 'b10x10')"
readonly SUFFIX_TOKEN="$([[ -n "${ATTEMPT_SUFFIX}" ]] && printf -- '-%s' "${ATTEMPT_SUFFIX}" || true)"
readonly CLAMP_TOKEN="${CLAMP_POLICY,,}"
readonly OUTPUT_ROOT="${RESULT_PARENT}/s05-p2r4-phaseb-${PHASE}-${MODEL}-clamp-${CLAMP_TOKEN}-p2r2-v2-paired${SUFFIX_TOKEN}-v1"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "P2R4 Phase-B output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=48 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p2r4-phaseb"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

RUN_ARGS=(
  /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync
  python -m project.run_scripts.session05_ode_bf_p2r4_phaseb_atomic
  --model "${MODEL}" --output-root "${OUTPUT_ROOT}"
  --source-head "${SOURCE_HEAD}" --case-count "${CASE_COUNT}"
  --clamp-policy "${CLAMP_POLICY}" --run-token "${RUN_TOKEN}"
)
if [[ -n "${ATTEMPT_SUFFIX}" ]]; then
  RUN_ARGS+=(--attempt-suffix "${ATTEMPT_SUFFIX}")
fi
exec "${RUN_ARGS[@]}"
