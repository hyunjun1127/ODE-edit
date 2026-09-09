#!/usr/bin/env bash
#SBATCH --job-name=odeedit_fixed10k_preedit_s2
#SBATCH --partition=gpu
#SBATCH --nodelist=server2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60416M
#SBATCH --time=48:00:00
#SBATCH --export=NONE
set -euo pipefail
umask 077
task_root="$1"
task_repo="$2"
cd "$task_repo"
bash scripts/check-session-boundary.sh 01a0493a-074c-7f91-9a13-769116326fef
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$task_root/deps-transformers-4.44.2:$task_root/native-source:$task_root/helper-source"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u \
  "$task_repo/project/run_scripts/fixed10k_preedit_eval/observe.py" \
  --lock "$task_root/execution.lock.json" --output "$task_root/output"
