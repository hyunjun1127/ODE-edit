#!/usr/bin/env bash
# Llama/Qwen four-edit Motivation pair; both model children start together.
#SBATCH --job-name=odeedit_microseq_pair_v1
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
readonly RUNTIME="${REPO_ROOT}/project/run_scripts/session01_microseq_server1.sbatch"
readonly SRUN_BIN="/usr/bin/srun"
readonly UV_BIN="/usr/local/bin/uv"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly PAIR_DIRECTORY="${OUTPUT_ROOT}/microseq_pair_m0_v1"

[[ "$#" -eq 0 ]] || exit 2
[[ "${SLURM_JOB_NAME:-}" == "odeedit_microseq_pair_v1" ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || exit 2
[[ "${SLURM_NTASKS:-}" == "2" && "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || exit 2
[[ -x "${RUNTIME}" && ! -L "${RUNTIME}" ]] || exit 2
[[ -x "${SRUN_BIN}" && -x "${UV_BIN}" ]] || exit 2
[[ ! -e "${PAIR_DIRECTORY}" && ! -L "${PAIR_DIRECTORY}" ]] || exit 2

"${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --mem=65000M \
  "${RUNTIME}" llama3-8b-inst &
llama_pid="$!"
"${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --mem=65000M \
  "${RUNTIME}" qwen2.5-7b-inst &
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
  echo "paired microseq child failed; sibling terminated fail-fast" >&2
  exit 1
fi

mkdir "${PAIR_DIRECTORY}"
chmod 700 "${PAIR_DIRECTORY}"
env \
  PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  UV_OFFLINE=1 \
  UV_NO_SYNC=1 \
  UV_PYTHON_DOWNLOADS=never \
  PYTHONDONTWRITEBYTECODE=1 \
  "${UV_BIN}" run \
    --offline \
    --project "${EASYEDIT_ROOT}" \
    --frozen \
    --no-sync \
    python -m project.run_scripts.ode_edit_motivation.microseq_analysis pair \
    --llama-analysis "${OUTPUT_ROOT}/microseq_combined_llama_m0_v1/analysis.json" \
    --qwen-analysis "${OUTPUT_ROOT}/microseq_combined_qwen_m0_v1/analysis.json" \
    --output-json "${PAIR_DIRECTORY}/analysis.json"
