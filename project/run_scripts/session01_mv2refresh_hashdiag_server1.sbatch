#!/usr/bin/env bash
# Technical-only Llama first-case CUDA RCA; not a scientific MV-2 run.
#SBATCH --job-name=odeedit_mv2refresh_pair_v1
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=65000M
#SBATCH --time=00:12:00
#SBATCH --export=NONE
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-hashdiag-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-hashdiag-%j.err

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly RUNTIME="${REPO_ROOT}/project/run_scripts/session01_mv2refresh_server1.sbatch"
readonly SRUN_BIN="/usr/bin/srun"
readonly DIAG_EXPORT="ALL,CUDA_LAUNCH_BLOCKING=1,ODEEDIT_MV2_TECHNICAL_DIAGNOSTIC=hashdiag-v1,ODEEDIT_MV2_TECHNICAL_MAX_CASES=1"

[[ "${SLURM_JOB_NAME:-}" == "odeedit_mv2refresh_pair_v1" ]] || {
  echo "MV-2 hash diagnostic job-name boundary mismatch" >&2
  exit 2
}
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || {
  echo "MV-2 hash diagnostic node boundary mismatch" >&2
  exit 2
}
[[ "${SLURM_NTASKS:-}" == "1" && "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || {
  echo "MV-2 hash diagnostic CPU/task boundary mismatch" >&2
  exit 2
}
[[ -x "${RUNTIME}" && ! -L "${RUNTIME}" && -x "${SRUN_BIN}" ]] || {
  echo "MV-2 hash diagnostic runtime boundary mismatch" >&2
  exit 2
}

# This diagnostic intentionally synchronizes CUDA kernels so an asynchronous
# runtime fault is attributed to its originating repo-local operation.  It is
# not eligible for scientific analysis and is capped at twelve minutes.
exec "${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --mem=65000M \
  --export="${DIAG_EXPORT}" \
  "${RUNTIME}" llama3-8b-inst mv2refresh_llama_e0_v1
