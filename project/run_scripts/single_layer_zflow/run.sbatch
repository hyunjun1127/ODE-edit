#!/usr/bin/env bash
#SBATCH --job-name=odeedit_sl_zflow_seq1000_s2
#SBATCH --partition=gpu
#SBATCH --nodelist=server2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
umask 077
task_source="$1"
task_lock="$2"
task_mode="$3"
task_output="$4"
task_technical="${5:-}"
cd "$task_source"
bash scripts/check-session-boundary.sh 01a0493a-074c-7f91-9a13-769116326fef
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH=/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2
task_python=/mnt/raid5/janghj/EasyEdit/.venv/bin/python
"$task_python" -B -m project.run_scripts.single_layer_zflow.provenance verify --lock "$task_lock"
if [[ "$task_mode" == technical ]]; then
  "$task_python" -B -u -m project.run_scripts.single_layer_zflow.technical --phase initial --lock "$task_lock" --output "$task_output"
  "$task_python" -B -u -m project.run_scripts.single_layer_zflow.technical --phase resume --lock "$task_lock" --output "$task_output"
elif [[ "$task_mode" == main ]]; then
  test -n "$task_technical"
  "$task_python" -B -u -m project.run_scripts.single_layer_zflow.runner --lock "$task_lock" --technical "$task_technical" --output "$task_output"
else
  echo 'Unknown task mode' >&2
  exit 2
fi
