#!/usr/bin/env bash
#SBATCH --job-name=odeedit_refresh_seq1000_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --export=NONE
set -euo pipefail
umask 077
task_attempt="$1"
task_mode="${2:-main}"
task_source="$task_attempt/source"
task_deps=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_deps:$task_source"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
task_cell="${SLURM_ARRAY_TASK_ID:?array required}"
cd "$task_source"
if [[ "$task_mode" == technical ]]; then
  task_modes=(native I1)
  [[ "$task_cell" =~ ^[0-1]$ ]]
  exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.low_cost_write_donor_pilot.refresh_technical --lock "$task_attempt/execution.lock.json" --output "$task_attempt/technical/${task_modes[$task_cell]}" --mode "${task_modes[$task_cell]}"
fi
if [[ "$task_mode" == repair-I1 ]]; then
  [[ "$task_cell" == 0 ]]
  exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.low_cost_write_donor_pilot.refresh_technical --lock "$task_attempt/execution.lock.json" --output "$task_attempt/technical/I1" --mode I1
fi
[[ "$task_mode" == main && "$task_cell" =~ ^[0-3]$ ]]
task_policies=(FROZEN2 I2 FROZEN4 I4)
exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.low_cost_write_donor_pilot.refresh_runtime --lock "$task_attempt/execution.lock.json" --output "$task_attempt/output/cell-$task_cell" --policy "${task_policies[$task_cell]}"
