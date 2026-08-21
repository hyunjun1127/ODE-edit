#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-4%4
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly SOURCE_TREE="${2:-}"
readonly RESULT_PARENT="${3:-}"
readonly EXPECTED_SESSION="01a00e5d-29e8-7a01-822b-7acf43226035"
readonly EXPECTED_BRANCH="codex/p1r52-joint-pc-fullfp32-independent-10xb100-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly RUN_TOKEN="p1r52-joint-pc-full-fp32-independent-b100x10-v1"
readonly RESULTS=(
  "s05-p1r52-joint-pc-full-fp32-independent-b100x10-baseline-group-v1"
  "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c0-v1"
  "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c1-v1"
  "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c2-v1"
  "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c3-v1"
)

[[ "$#" -eq 3 ]] || { echo "independent FP32 arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "independent FP32 CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "independent FP32 source HEAD differs" >&2; exit 2; }
[[ "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || { echo "independent FP32 source tree differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "independent FP32 branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "independent FP32 tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-4]$ ]] || { echo "independent FP32 array cell differs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "independent FP32 exactly one GPU required" >&2; exit 2; }
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULTS[${SLURM_ARRAY_TASK_ID}]}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "independent FP32 output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-joint-pc-independent-fp32-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_joint_pc_independent_fp32 \
  --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --source-tree "${SOURCE_TREE}" --cell "${SLURM_ARRAY_TASK_ID}" \
  --run-token "${RUN_TOKEN}"
