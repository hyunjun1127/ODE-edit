#!/usr/bin/env bash
#SBATCH --job-name=odeedit_jlz_native_pilot_v2
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/raid5/janghj/ODE-edit

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec /mnt/raid5/janghj/EasyEdit/.venv/bin/python -u -m project.run_scripts.jlz_pilot.run \
  --output-dir "${JLZ_PILOT_OUTPUT:?required}" --cap 120 --tol 0.0001 \
  --microbatch 2 --batch-seconds 1200 --total-seconds 3300
