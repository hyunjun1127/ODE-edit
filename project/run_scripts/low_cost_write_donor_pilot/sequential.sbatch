#!/usr/bin/env bash
#SBATCH --job-name=odeedit_lowcost_seq10_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --export=NONE
set -euo pipefail
umask 077
task_attempt="$1"
task_source="$task_attempt/source"
task_deps=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_deps:$task_source"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
task_arms=(N4 RES8 S875 S75 FULL8 REFIT4)
task_cell="${SLURM_ARRAY_TASK_ID:?array cell required}"
[[ "$task_cell" =~ ^[0-5]$ ]]
cd "$task_source"
exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.low_cost_write_donor_pilot.sequential_runtime --lock "$task_attempt/execution.lock.json" --output "$task_attempt/output/cell-$task_cell" --arm "${task_arms[$task_cell]}"
