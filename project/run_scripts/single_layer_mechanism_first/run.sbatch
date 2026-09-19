#!/bin/bash
#SBATCH --job-name=odeedit_slmf_T0_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
task_source=$1
task_lock=$2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2:$task_source"
cd "$task_source"
exec /data/janghj/EasyEdit/.venv/bin/python -u -B -m project.run_scripts.single_layer_mechanism_first.entry --lock "$task_lock"
