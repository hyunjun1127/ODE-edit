#!/usr/bin/env bash
#SBATCH --job-name=odeedit_l4_two_memory_v2
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=182272M
#SBATCH --time=08:00:00
#SBATCH --export=NONE
set -euo pipefail
task_repo=/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-routing-v2
cd "$task_repo"
export CUMRISK_ASSET_ROOT=/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1
export PYTHONPATH="$CUMRISK_ASSET_ROOT/deps-py312:$task_repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export L4_ROUTING_SOURCE_HEAD
L4_ROUTING_SOURCE_HEAD=$(git rev-parse HEAD)
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.l4_two_memory_conflict_routing.runtime "$@"
