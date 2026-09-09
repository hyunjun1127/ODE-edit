#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server1
#SBATCH --job-name=odeedit_cumrisk_A_gen_s1
#SBATCH --nodelist=devbox
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=182272M
#SBATCH --time=00:30:00
#SBATCH --export=NONE
set -euo pipefail
source_dir=$1
asset_dir=$2
source_head=$3
shift 3
cd "$source_dir"
[[ "$(git rev-parse HEAD)" == "$source_head" ]]
export CUMRISK_ASSET_ROOT="$asset_dir"
export CUMRISK_SOURCE_HEAD="$source_head"
export PYTHONPATH="$asset_dir/deps-py312:$source_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.single_layer_cumulative_risk.saved_generation "$@"
