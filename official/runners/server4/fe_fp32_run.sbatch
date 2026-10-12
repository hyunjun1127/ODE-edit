#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=48:00:00
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
ATTEMPT=$1
RUN_PYTHON=/data/janghj/ODE-edit/local/fe-zsre-lane-b-s4-20261011/runtime/fe-native-venv/bin/python
export PYTHONPATH="$ATTEMPT/source"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$ATTEMPT/tmp"
mkdir -p "$TMPDIR"
cd "$ATTEMPT/source"
"$RUN_PYTHON" -u -m official.diagnostics.fe_qwen_current.author_basic_eot_fp32 run "$ATTEMPT/bindings.json" "$ATTEMPT/output"
"$RUN_PYTHON" -u -m official.diagnostics.fe_qwen_current.author_basic_eot_fp32 collect "$ATTEMPT/output/benchmark-result.json" "$ATTEMPT/final-cpu-audit.json"
