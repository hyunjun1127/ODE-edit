#!/bin/bash
#SBATCH --job-name=odeedit_memit_hj_collect_s3
#SBATCH --partition=gpu
#SBATCH --nodelist=ubuntu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768M
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
TASK_SOURCE="$1"
TASK_LOCK="$2"
TASK_MAPPING="$3"
export PYTHONPATH="$TASK_SOURCE" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$TASK_SOURCE"
exec /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python -u -m project.run_scripts.memit_hj.collector --lock "$TASK_LOCK" --mapping "$TASK_MAPPING"
