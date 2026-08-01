#!/usr/bin/env bash
# AlphaEdit-geometry direct-z diagnostic; Llama/Qwen start together and fail fast.
#SBATCH --job-name=odeedit_dzf_alpha_pair_v1
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=130000M
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.err

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly RUNTIME="${REPO_ROOT}/project/run_scripts/session01_direct_z_alpha_possibility_server1.sbatch"
readonly BASH_BIN="/usr/bin/bash"
readonly SRUN_BIN="/usr/bin/srun"

[[ "$#" -eq 0 ]] || {
  echo "Alpha direct-z pair wrapper takes no arguments" >&2
  exit 2
}
[[ "${SLURM_JOB_NAME:-}" == "odeedit_dzf_alpha_pair_v1" ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || exit 2
[[ "${SLURM_NTASKS:-}" == "2" && "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || exit 2
[[ -f "${RUNTIME}" && ! -L "${RUNTIME}" && -r "${RUNTIME}" ]] || exit 2
[[ -x "${BASH_BIN}" && -x "${SRUN_BIN}" ]] || exit 2

"${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:a6000:1 --mem=65000M \
  "${BASH_BIN}" "${RUNTIME}" llama3-8b-inst dzf_alpha_llama_p0_v1 &
llama_pid="$!"
"${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:a6000:1 --mem=65000M \
  "${BASH_BIN}" "${RUNTIME}" qwen2.5-7b-inst dzf_alpha_qwen_p0_v1 &
qwen_pid="$!"

terminate_and_reap() {
  local pid alive deadline
  for pid in "$@"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  deadline="$((SECONDS + 10))"
  while true; do
    alive=0
    for pid in "$@"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    [[ "${alive}" -eq 0 ]] && break
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      for pid in "$@"; do
        kill -KILL "${pid}" 2>/dev/null || true
      done
      break
    fi
    sleep 1
  done
  for pid in "$@"; do
    wait "${pid}" 2>/dev/null || true
  done
}

child_pids=("${llama_pid}" "${qwen_pid}")
first_failure_rc=0
set +e
while [[ "${#child_pids[@]}" -gt 0 ]]; do
  finished_pid=""
  wait -n -p finished_pid "${child_pids[@]}"
  child_rc="$?"
  if [[ "${child_rc}" -ne 0 ]]; then
    first_failure_rc="${child_rc}"
    terminate_and_reap "${child_pids[@]}"
    break
  fi
  remaining_pids=()
  for child_pid in "${child_pids[@]}"; do
    if [[ "${child_pid}" != "${finished_pid}" ]]; then
      remaining_pids+=("${child_pid}")
    fi
  done
  child_pids=("${remaining_pids[@]}")
done
set -e

if [[ "${first_failure_rc}" -ne 0 ]]; then
  echo "paired Alpha direct-z child failed; sibling terminated fail-fast" >&2
  exit 1
fi
