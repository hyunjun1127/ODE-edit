#!/usr/bin/env bash
set -euo pipefail
umask 077
task_source="$1"
task_mode="$2"
task_root="$3"
cd "$task_source"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_source"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
if [[ "$task_mode" == science ]]; then
  exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.native_delayed_write_e3.stages --lock "$task_root/execution.lock.json" --output "$task_root/output"
elif [[ "$task_mode" == collector ]]; then
  task_parent="$4"
  exec /data/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.native_delayed_write_e3.reduce --output "$task_root/output" --destination "$task_root/report" --job-id "$task_parent"
else
  exit 64
fi
