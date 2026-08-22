#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=devbox
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-2%3
#SBATCH --export=NONE
set -euo pipefail
umask 077
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}" SOURCE_TREE="${2:-}" RESULT_PARENT="${3:-}"
readonly EXPECTED_SESSION="01a0278d-456b-7d20-820f-63fc80a57b8d"
readonly EXPECTED_BRANCH="codex/p1r52-c-writer-phase2-tech-r1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly RUN_TOKEN="p1r52-c-writer-phase2-independent-kstep-full-fp32-v1"
readonly RESULTS=("s05-p1r52-c-writer-phase2-independent-full-fp32-c0-kstep-10xb100-v1" "s05-p1r52-c-writer-phase2-independent-full-fp32-c1-kstep-10xb100-v1" "s05-p1r52-c-writer-phase2-independent-full-fp32-c3-kstep-10xb100-v1")
[[ "$#" -eq 3 && "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || exit 2
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" && "$(git rev-parse 'HEAD^{tree}')" == "${SOURCE_TREE}" ]] || exit 2
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-2]$ && "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || exit 2
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULTS[${SLURM_ARRAY_TASK_ID}]}"
[[ ! -e "${OUTPUT_ROOT}" ]] || exit 2
export HF_HOME="/mnt/raid5/janghj/.cache/huggingface" HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-c-writer-phase2-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"
exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync python -m project.run_scripts.session05_ode_bf_p1r52_c_writer_phase2 --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" --source-tree "${SOURCE_TREE}" --cell "${SLURM_ARRAY_TASK_ID}" --run-token "${RUN_TOKEN}"
