#!/usr/bin/env bash
# Session 01 Motivation MV-1 score-mix D0; fixed Llama/Qwen runs start together.
#SBATCH --job-name=odeedit_mv1mix_d0_pair_v1
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=130000M
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.err

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly RUNTIME="${REPO_ROOT}/project/run_scripts/session01_mv1mix_server1.sbatch"

[[ "${SLURM_JOB_NAME:-}" == "odeedit_mv1mix_d0_pair_v1" ]] || {
  echo "paired Slurm job-name boundary mismatch" >&2
  exit 2
}
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || {
  echo "paired Slurm node boundary mismatch" >&2
  exit 2
}
[[ -x "${RUNTIME}" ]] || {
  echo "tracked MV-1 score-mix runtime wrapper is unavailable" >&2
  exit 2
}

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --mem=65000M \
  "${RUNTIME}" llama3-8b-inst mv1mix_llama_d0_v1 3 5 &
llama_pid="$!"
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --mem=65000M \
  "${RUNTIME}" qwen2.5-7b-inst mv1mix_qwen_d0_v1 3 5 &
qwen_pid="$!"

set +e
wait "${llama_pid}"
llama_rc="$?"
wait "${qwen_pid}"
qwen_rc="$?"
set -e

if [[ "${llama_rc}" -ne 0 || "${qwen_rc}" -ne 0 ]]; then
  echo "paired MV-1 score-mix D0 child failed" >&2
  exit 1
fi
