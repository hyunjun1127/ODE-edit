#!/usr/bin/env bash
#SBATCH --job-name=odeedit_enfc_m_b001_s1
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=182272M
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
[[ $# == 3 ]]
enfc_worktree="$1"; enfc_head="$2"; enfc_lock="$3"
cd "$enfc_worktree"
[[ "$(git rev-parse HEAD)" == "$enfc_head" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
export PYTHONPATH="/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/deps-py312:$enfc_worktree"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.single_layer_edit_preserving_correction.server1_single_batch.runner --lock "$enfc_lock" --episode 0
