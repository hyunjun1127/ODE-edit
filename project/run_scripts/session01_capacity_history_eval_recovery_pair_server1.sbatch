#!/usr/bin/env bash
# Evaluation-only four-GPU recovery for terminal v3 controller artifacts.
#SBATCH --job-name=odeedit_capacity_history_pair_v3
#SBATCH --partition=gpu
#SBATCH --nodelist=devbox
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=260000M
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-eval-recovery-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/slurm/session01_motivation/%x-eval-recovery-%j.err

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly LOCKED_ROOT="${REPO_ROOT}/local/scratch/caphist-eval-9900a51"
readonly LOCKED_COMMIT="9900a51f16669447bf502e7df0d5d5e0d61ef360"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly SOURCE_OUTPUT_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly LOCKED_OUTPUT_ROOT="${LOCKED_ROOT}/local/results/raw/session01_motivation"
readonly CHILD="${REPO_ROOT}/project/run_scripts/session01_capacity_history_eval_recovery_server1.sbatch"
readonly RECOVERY_ENTRYPOINT="${REPO_ROOT}/project/run_scripts/ode_edit_motivation/capacity_history_evaluator_recovery.py"
readonly LLAMA_COMBINED="${SOURCE_OUTPUT_ROOT}/caphist_combined_llama_c0_v3"
readonly QWEN_COMBINED="${SOURCE_OUTPUT_ROOT}/caphist_combined_qwen_c0_v3"
readonly PAIR_DIRECTORY="${SOURCE_OUTPUT_ROOT}/caphist_pair_c0_v3"
readonly WORKER_LOG_ROOT="${REPO_ROOT}/local/logs/slurm/session01_motivation/workers-eval-${SLURM_JOB_ID}"
readonly SRUN_BIN="/usr/bin/srun"
readonly UV_BIN="/usr/local/bin/uv"
readonly GIT_BIN="/usr/bin/git"

readonly -a SCIENTIFIC_PATHS=(
  project/run_scripts/ode_edit_motivation/capacity_history_controller.py
  project/run_scripts/ode_edit_motivation/capacity_history_evaluator.py
  project/run_scripts/ode_edit_motivation/capacity_history_analysis.py
  project/run_scripts/ode_edit_motivation/capacity_qp.py
  project/run_scripts/ode_edit_motivation/capacity_geometry.py
  project/run_scripts/ode_edit_motivation/alphaedit_history.py
  project/run_scripts/ode_edit_motivation/alphaedit_factors.py
  project/run_scripts/ode_edit_motivation/alphaedit_proposal_adapter.py
)

readonly -a CONTROLLER_RUNS=(
  caphist_memit_native_llama_c0_v3
  caphist_memit_qp_llama_c0_v3
  caphist_alpha_native_llama_c0_v3
  caphist_alpha_qp_llama_c0_v3
  caphist_memit_native_qwen_c0_v3
  caphist_memit_qp_qwen_c0_v3
  caphist_alpha_native_qwen_c0_v3
  caphist_alpha_qp_qwen_c0_v3
)

readonly -a EVALUATOR_RUNS=(
  caphist_eval_memit_native_llama_c0_v3
  caphist_eval_memit_qp_llama_c0_v3
  caphist_eval_alpha_native_llama_c0_v3
  caphist_eval_alpha_qp_llama_c0_v3
  caphist_eval_memit_native_qwen_c0_v3
  caphist_eval_memit_qp_qwen_c0_v3
  caphist_eval_alpha_native_qwen_c0_v3
  caphist_eval_alpha_qp_qwen_c0_v3
)

[[ "$#" -eq 0 ]] || exit 2
[[ "${SLURM_JOB_NAME:-}" == "odeedit_capacity_history_pair_v3" ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == "devbox" ]] || exit 2
[[ "${SLURM_NTASKS:-}" == "4" && "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || exit 2
[[ -x "${CHILD}" && ! -L "${CHILD}" && -x "${SRUN_BIN}" && -x "${UV_BIN}" && -x "${GIT_BIN}" ]] || exit 2
[[ -f "${RECOVERY_ENTRYPOINT}" && ! -L "${RECOVERY_ENTRYPOINT}" ]] || exit 2
[[ -d "${LOCKED_ROOT}" && ! -L "${LOCKED_ROOT}" ]] || exit 2
[[ "$("${GIT_BIN}" -C "${LOCKED_ROOT}" rev-parse HEAD)" == "${LOCKED_COMMIT}" ]] || exit 2
[[ -z "$("${GIT_BIN}" -C "${LOCKED_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 2
[[ -z "$("${GIT_BIN}" -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] || exit 2
[[ "$("${GIT_BIN}" -C "${REPO_ROOT}" rev-parse HEAD)" == "$("${GIT_BIN}" -C "${REPO_ROOT}" rev-parse origin/main)" ]] || exit 2
"${GIT_BIN}" -C "${REPO_ROOT}" diff --quiet "${LOCKED_COMMIT}"..HEAD -- "${SCIENTIFIC_PATHS[@]}" || exit 2
[[ -d "${SOURCE_OUTPUT_ROOT}" && ! -L "${SOURCE_OUTPUT_ROOT}" ]] || exit 2
[[ -d "${LOCKED_OUTPUT_ROOT}" && ! -L "${LOCKED_OUTPUT_ROOT}" ]] || exit 2
for run_name in "${CONTROLLER_RUNS[@]}"; do
  source_directory="${SOURCE_OUTPUT_ROOT}/${run_name}"
  locked_directory="${LOCKED_OUTPUT_ROOT}/${run_name}"
  [[ -d "${source_directory}" && ! -L "${source_directory}" ]] || exit 2
  [[ -d "${locked_directory}" && ! -L "${locked_directory}" ]] || exit 2
  [[ -z "$(find "${source_directory}" "${locked_directory}" -type l -print -quit)" ]] || exit 2
  diff -qr --no-dereference "${source_directory}" "${locked_directory}" >/dev/null || exit 2
done
for run_name in "${EVALUATOR_RUNS[@]}"; do
  [[ ! -e "${SOURCE_OUTPUT_ROOT}/${run_name}" && ! -L "${SOURCE_OUTPUT_ROOT}/${run_name}" ]] || exit 2
  [[ ! -e "${LOCKED_OUTPUT_ROOT}/${run_name}" && ! -L "${LOCKED_OUTPUT_ROOT}/${run_name}" ]] || exit 2
done
for path in "${LLAMA_COMBINED}" "${QWEN_COMBINED}" "${PAIR_DIRECTORY}"; do
  [[ ! -e "${path}" && ! -L "${path}" ]] || exit 2
done
mkdir -p "${WORKER_LOG_ROOT}"
[[ -d "${WORKER_LOG_ROOT}" && ! -L "${WORKER_LOG_ROOT}" ]] || exit 2

terminate_and_reap() {
  local pid alive deadline
  for pid in "$@"; do kill -TERM "${pid}" 2>/dev/null || true; done
  deadline="$((SECONDS + 10))"
  while true; do
    alive=0
    for pid in "$@"; do kill -0 "${pid}" 2>/dev/null && alive=1 || true; done
    [[ "${alive}" -eq 0 ]] && break
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      for pid in "$@"; do kill -KILL "${pid}" 2>/dev/null || true; done
      break
    fi
    sleep 1
  done
  for pid in "$@"; do wait "${pid}" 2>/dev/null || true; done
}

run_evaluators() {
  local finished_pid child_rc first_failure_rc pid
  local -a child_pids=() remaining_pids=()
  for family_model in \
    "memit llama3-8b-inst" \
    "alpha_history llama3-8b-inst" \
    "memit qwen2.5-7b-inst" \
    "alpha_history qwen2.5-7b-inst"; do
    read -r family model <<<"${family_model}"
    "${SRUN_BIN}" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
      --gres=gpu:1 --mem=65000M \
      "${CHILD}" "${family}" "${model}" \
      >"${WORKER_LOG_ROOT}/${family}-${model}.out" \
      2>"${WORKER_LOG_ROOT}/${family}-${model}.err" &
    child_pids+=("$!")
  done

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
  [[ "${first_failure_rc}" -eq 0 ]] || exit 1
}

export PYTHONPATH="${LOCKED_ROOT}:${EASYEDIT_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_NO_SYNC=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONDONTWRITEBYTECODE=1

run_python() {
  (
    cd "${LOCKED_ROOT}"
    "${UV_BIN}" run --offline --project "${EASYEDIT_ROOT}" --frozen --no-sync python "$@"
  )
}

run_evaluators

copy_evaluator_exact() {
  local run_name="$1"
  local source_directory="${LOCKED_OUTPUT_ROOT}/${run_name}"
  local destination_directory="${SOURCE_OUTPUT_ROOT}/${run_name}"
  [[ -d "${source_directory}" && ! -L "${source_directory}" ]] || exit 2
  [[ ! -e "${destination_directory}" && ! -L "${destination_directory}" ]] || exit 2
  [[ -z "$(find "${source_directory}" -type l -print -quit)" ]] || exit 2
  cp -a -- "${source_directory}" "${destination_directory}"
  diff -qr --no-dereference "${source_directory}" "${destination_directory}" >/dev/null
}

for run_name in "${EVALUATOR_RUNS[@]}"; do
  copy_evaluator_exact "${run_name}"
done

mkdir "${LLAMA_COMBINED}" "${QWEN_COMBINED}" "${PAIR_DIRECTORY}"
chmod 700 "${LLAMA_COMBINED}" "${QWEN_COMBINED}" "${PAIR_DIRECTORY}"

run_python -m project.run_scripts.ode_edit_motivation.capacity_history_evaluator merge \
  --memit-native "${SOURCE_OUTPUT_ROOT}/caphist_eval_memit_native_llama_c0_v3/checkpoints.jsonl" \
  --memit-qp "${SOURCE_OUTPUT_ROOT}/caphist_eval_memit_qp_llama_c0_v3/checkpoints.jsonl" \
  --alpha-native "${SOURCE_OUTPUT_ROOT}/caphist_eval_alpha_native_llama_c0_v3/checkpoints.jsonl" \
  --alpha-qp "${SOURCE_OUTPUT_ROOT}/caphist_eval_alpha_qp_llama_c0_v3/checkpoints.jsonl" \
  --output "${LLAMA_COMBINED}/checkpoints.jsonl" --model llama3-8b-inst \
  --run-id caphist_combined_llama_c0_v3
run_python -m project.run_scripts.ode_edit_motivation.capacity_history_analysis model \
  --checkpoints "${LLAMA_COMBINED}/checkpoints.jsonl" --model llama3-8b-inst \
  --output-json "${LLAMA_COMBINED}/analysis.json"

run_python -m project.run_scripts.ode_edit_motivation.capacity_history_evaluator merge \
  --memit-native "${SOURCE_OUTPUT_ROOT}/caphist_eval_memit_native_qwen_c0_v3/checkpoints.jsonl" \
  --memit-qp "${SOURCE_OUTPUT_ROOT}/caphist_eval_memit_qp_qwen_c0_v3/checkpoints.jsonl" \
  --alpha-native "${SOURCE_OUTPUT_ROOT}/caphist_eval_alpha_native_qwen_c0_v3/checkpoints.jsonl" \
  --alpha-qp "${SOURCE_OUTPUT_ROOT}/caphist_eval_alpha_qp_qwen_c0_v3/checkpoints.jsonl" \
  --output "${QWEN_COMBINED}/checkpoints.jsonl" --model qwen2.5-7b-inst \
  --run-id caphist_combined_qwen_c0_v3
run_python -m project.run_scripts.ode_edit_motivation.capacity_history_analysis model \
  --checkpoints "${QWEN_COMBINED}/checkpoints.jsonl" --model qwen2.5-7b-inst \
  --output-json "${QWEN_COMBINED}/analysis.json"

run_python -m project.run_scripts.ode_edit_motivation.capacity_history_analysis pair \
  --llama-analysis "${LLAMA_COMBINED}/analysis.json" \
  --qwen-analysis "${QWEN_COMBINED}/analysis.json" \
  --output-json "${PAIR_DIRECTORY}/analysis.json"
