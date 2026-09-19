#!/usr/bin/env bash
#SBATCH --job-name=odeedit_checkpoint_mechanism_s2_analysis
#SBATCH --partition=gpu
#SBATCH --nodelist=server2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
umask 077
task_source="$1"
task_lock="$2"
cd "$task_source"
bash /mnt/raid5/janghj/ODE-edit/scripts/check-session-boundary.sh 01a0493a-074c-7f91-9a13-769116326fef
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH=/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -u -m project.run_scripts.checkpoint_mechanism_audit.control execute --lock "$task_lock"
