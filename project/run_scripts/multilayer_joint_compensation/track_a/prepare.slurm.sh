#!/usr/bin/env bash
#SBATCH --job-name=odeedit_multilayer_common_s1
#SBATCH --nodelist=devbox
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=182272M
#SBATCH --time=12:00:00
#SBATCH --export=NONE
set -euo pipefail
umask 077

# Positional bindings supplied in a create-once submission receipt. Never load
# from the mutable development checkout while its source is being extended.
execution_root="${1:?execution_root}"
source_head="${2:?source_head}"
entry_name="${3:?entry}"
cpu_input="${4:?cpu_input}"
result_path="${5:?result_path}"
cd "${execution_root}"
test "$(git rev-parse HEAD)" = "${source_head}"
test -z "$(git status --porcelain --untracked-files=no)"
export CUMRISK_ASSET_ROOT=/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1
export PYTHONPATH="${CUMRISK_ASSET_ROOT}/deps-py312:${execution_root}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.multilayer_joint_compensation.common_reference.prepare \
  --source-head "${source_head}" --entry "${entry_name}" --cpu-input "${cpu_input}" --output "${result_path}"
