#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server2
#SBATCH --job-name=odeedit_l4ep_s2
#SBATCH --nodelist=server2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=12:00:00
#SBATCH --export=NONE
set -euo pipefail
source_dir=$1
source_head=$2
shift 2
cd "$source_dir"
[[ "$(git rev-parse HEAD)" == "$source_head" ]]
export L4_EP_SOURCE_HEAD="$source_head"
export CUMRISK_ASSET_ROOT=/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/imports
export PYTHONPATH=/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2:"$source_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.blue_l4_progress_barrier.runtime "$@"
