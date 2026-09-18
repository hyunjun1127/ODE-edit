#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=1-00:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
umask 077
task_source="$1"
task_lock="$2"
task_stage="$3"
task_deps=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_deps:$task_source"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
cd "$task_source"
case "$task_stage" in
  GENERATED_REFERENCE_PREPARATION) task_module=project.run_scripts.en_execution_reuse.prepare_runner ;;
  MATCHED_B1) task_module=project.run_scripts.en_execution_reuse.runner ;;
  *) exit 64 ;;
esac
exec /data/janghj/EasyEdit/.venv/bin/python -u -m "$task_module" --lock "$task_lock" --max-batches 1
