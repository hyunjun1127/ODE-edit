#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server2
#SBATCH --job-name=odeedit_multilayer_b_s2
#SBATCH --nodelist=server2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --export=NONE
set -euo pipefail
source_dir=$1
source_head=$2
shift 2
cd "$source_dir"
[[ "$(git rev-parse HEAD)" == "$source_head" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
export CUMRISK_ASSET_ROOT=/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/imports
export PYTHONPATH=/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2:"$source_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export SLURM_EXPORT_ENV=ALL
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.multilayer_joint_compensation.track_b.runtime "$@"
