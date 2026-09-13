#!/usr/bin/env bash
#SBATCH --job-name=odeedit_e01_cold_l4_s1
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=182272M
#SBATCH --time=04:00:00
#SBATCH --export=NONE
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: launch_cold_l4.sh EXACT_WORKTREE SOURCE_HEAD INPUT_LOCK NEW_OUTPUT' >&2
  exit 2
fi
e01_worktree="$1"
e01_source_head="$2"
e01_input_lock="$3"
e01_output="$4"
cd "$e01_worktree"
[[ "$(git rev-parse HEAD)" == "$e01_source_head" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ ! -e "$e01_output" ]]
export PYTHONPATH="/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/deps-py312:$e01_worktree"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.baseline_mechanism_first.runner \
  --lock "$e01_input_lock" --output "$e01_output"
