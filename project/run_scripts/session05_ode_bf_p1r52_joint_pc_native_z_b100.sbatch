#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=server2
#SBATCH --mem=65000M
#SBATCH --time=23:59:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE

set -euo pipefail
umask 077

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
readonly SOURCE_HEAD="${1:-}"
readonly RESULT_PARENT="${2:-}"
readonly EXPECTED_SESSION="01a00e5c-f7ae-72a2-98b2-b8b0907168b4"
readonly EXPECTED_BRANCH="codex/p1r52-joint-pc-native-z-b100-v1"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly RUN_TOKEN="p1r52-joint-pc-native-z-b100-v1"

[[ "$#" -eq 2 ]] || { echo "Joint P+C native-z arguments differ" >&2; exit 2; }
[[ "$(pwd -P)" == "$(readlink -f "${REPO_ROOT}")" ]] || { echo "Joint P+C native-z CWD differs" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${SOURCE_HEAD}" ]] || { echo "Joint P+C native-z source differs" >&2; exit 2; }
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "Joint P+C native-z branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Joint P+C native-z tracked source dirty" >&2; exit 2; }
scripts/check-session-boundary.sh "${EXPECTED_SESSION}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "Joint P+C native-z exactly one GPU required" >&2; exit 2; }

readonly ROLES=(
  joint-pc-official-alphaedit-native-z-b100
  joint-pc-official-memit-native-z-b100
)
readonly RESULT_NAMES=(
  s05-p1r52-joint-pc-official-alphaedit-native-z-b100-v1
  s05-p1r52-joint-pc-official-memit-native-z-b100-v1
)
readonly ROLE="${ROLES[${SLURM_ARRAY_TASK_ID}]}"
readonly OUTPUT_ROOT="${RESULT_PARENT}/${RESULT_NAMES[${SLURM_ARRAY_TASK_ID}]}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Joint P+C native-z output exists" >&2; exit 2; }

export HF_HOME="/mnt/raid5/janghj/.cache/huggingface"
export HF_HUB_CACHE="/mnt/raid5/janghj/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=52 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${REPO_ROOT}/local/odebf/matplotlib-p1r52-joint-pc-native-z-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"

exec /usr/local/bin/uv run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync \
  python -m project.run_scripts.session05_ode_bf_p1r52_joint_pc_native_z_b100 \
  --role "${ROLE}" --output-root "${OUTPUT_ROOT}" --source-head "${SOURCE_HEAD}" \
  --run-token "${RUN_TOKEN}"
