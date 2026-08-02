#!/usr/bin/env bash
# Four-GPU pair: Llama/Qwen and MEMIT/Alpha-history workers start together.
#SBATCH --job-name=odeedit_capacity_history_pair_v1
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=260000M
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-%j.err

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly RUNTIME="${REPO_ROOT}/project/run_scripts/session01_capacity_history_server1.sbatch"
readonly SRUN_BIN="/usr/bin/srun"
readonly UV_BIN="/usr/local/bin/uv"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly LLAMA_COMBINED="${OUTPUT_ROOT}/caphist_combined_llama_c0_v1"
readonly QWEN_COMBINED="${OUTPUT_ROOT}/caphist_combined_qwen_c0_v1"
readonly PAIR_DIRECTORY="${OUTPUT_ROOT}/caphist_pair_c0_v1"

[[ "$#" -eq 0 ]] || exit 2
[[ "${SLURM_JOB_NAME:-}" == "odeedit_capacity_history_pair_v1" ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || exit 2
[[ "${SLURM_NTASKS:-}" == "4" && "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || exit 2
[[ -x "${RUNTIME}" && ! -L "${RUNTIME}" ]] || exit 2
[[ -x "${SRUN_BIN}" && -x "${UV_BIN}" ]] || exit 2
for path in "${LLAMA_COMBINED}" "${QWEN_COMBINED}" "${PAIR_DIRECTORY}"; do
  [[ ! -e "${path}" && ! -L "${path}" ]] || exit 2
done

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

run_phase() {
  local phase="$1"
  local finished_pid child_rc first_failure_rc
  local -a child_pids remaining_pids
  "${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --mem=65000M \
    "${RUNTIME}" "${phase}" memit llama3-8b-inst &
  child_pids+=("$!")
  "${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --mem=65000M \
    "${RUNTIME}" "${phase}" alpha_history llama3-8b-inst &
  child_pids+=("$!")
  "${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --mem=65000M \
    "${RUNTIME}" "${phase}" memit qwen2.5-7b-inst &
  child_pids+=("$!")
  "${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --mem=65000M \
    "${RUNTIME}" "${phase}" alpha_history qwen2.5-7b-inst &
  child_pids+=("$!")

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
    for pid in "${child_pids[@]}"; do
      [[ "${pid}" == "${finished_pid}" ]] || remaining_pids+=("${pid}")
    done
    child_pids=("${remaining_pids[@]}")
  done
  set -e
  if [[ "${first_failure_rc}" -ne 0 ]]; then
    echo "capacity/history ${phase} worker failed; siblings terminated" >&2
    exit 1
  fi
}

run_phase controller
run_phase evaluator

export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_NO_SYNC=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONDONTWRITEBYTECODE=1

run_python() {
  "${UV_BIN}" run \
    --offline \
    --project "${EASYEDIT_ROOT}" \
    --frozen \
    --no-sync \
    python "$@"
}

mkdir "${LLAMA_COMBINED}" "${QWEN_COMBINED}" "${PAIR_DIRECTORY}"
chmod 700 "${LLAMA_COMBINED}" "${QWEN_COMBINED}" "${PAIR_DIRECTORY}"

run_python -m project.run_scripts.ode_edit_motivation.capacity_history_evaluator merge \
  --memit-native "${OUTPUT_ROOT}/caphist_eval_memit_native_llama_c0_v1/checkpoints.jsonl" \
  --memit-qp "${OUTPUT_ROOT}/caphist_eval_memit_qp_llama_c0_v1/checkpoints.jsonl" \
  --alpha-native "${OUTPUT_ROOT}/caphist_eval_alpha_native_llama_c0_v1/checkpoints.jsonl" \
  --alpha-qp "${OUTPUT_ROOT}/caphist_eval_alpha_qp_llama_c0_v1/checkpoints.jsonl" \
  --output "${LLAMA_COMBINED}/checkpoints.jsonl" \
  --model llama3-8b-inst \
  --run-id caphist_combined_llama_c0_v1
run_python -m project.run_scripts.ode_edit_motivation.capacity_history_analysis model \
  --checkpoints "${LLAMA_COMBINED}/checkpoints.jsonl" \
  --model llama3-8b-inst \
  --output-json "${LLAMA_COMBINED}/analysis.json"

run_python -m project.run_scripts.ode_edit_motivation.capacity_history_evaluator merge \
  --memit-native "${OUTPUT_ROOT}/caphist_eval_memit_native_qwen_c0_v1/checkpoints.jsonl" \
  --memit-qp "${OUTPUT_ROOT}/caphist_eval_memit_qp_qwen_c0_v1/checkpoints.jsonl" \
  --alpha-native "${OUTPUT_ROOT}/caphist_eval_alpha_native_qwen_c0_v1/checkpoints.jsonl" \
  --alpha-qp "${OUTPUT_ROOT}/caphist_eval_alpha_qp_qwen_c0_v1/checkpoints.jsonl" \
  --output "${QWEN_COMBINED}/checkpoints.jsonl" \
  --model qwen2.5-7b-inst \
  --run-id caphist_combined_qwen_c0_v1
run_python -m project.run_scripts.ode_edit_motivation.capacity_history_analysis model \
  --checkpoints "${QWEN_COMBINED}/checkpoints.jsonl" \
  --model qwen2.5-7b-inst \
  --output-json "${QWEN_COMBINED}/analysis.json"

run_python -m project.run_scripts.ode_edit_motivation.capacity_history_analysis pair \
  --llama-analysis "${LLAMA_COMBINED}/analysis.json" \
  --qwen-analysis "${QWEN_COMBINED}/analysis.json" \
  --output-json "${PAIR_DIRECTORY}/analysis.json"
