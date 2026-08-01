#!/usr/bin/env bash
# Compact terminal analysis for the locked Alpha projected pair.

set -euo pipefail
umask 077

readonly REPO_ROOT="/mnt/raid5/janghj/ODE-edit"
readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
readonly UV_BIN="/usr/local/bin/uv"
readonly SACCT_BIN="/usr/bin/sacct"
readonly GREP_BIN="/usr/bin/grep"
readonly RAW_ROOT="${REPO_ROOT}/local/results/raw/session01_motivation"
readonly ANALYSIS_ROOT="${REPO_ROOT}/local/results/analysis/session01_motivation"
readonly STATE_ROOT="${REPO_ROOT}/local/state/slurm-submissions/session01_motivation"
readonly MARKER="${STATE_ROOT}/agate_alpha_pair_v3.submitted"

[[ "$#" -eq 0 ]] || exit 2
[[ -x "${UV_BIN}" && -x "${SACCT_BIN}" && -x "${GREP_BIN}" ]] || exit 2
[[ -f "${MARKER}/job-id" && ! -L "${MARKER}" && ! -L "${MARKER}/job-id" ]] || exit 2
job_id="$(<"${MARKER}/job-id")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || exit 2
job_state=""
read -r job_state < <(
  "${SACCT_BIN}" -n -X -j "${job_id}" --format=State --noheader
)
[[ "${job_state}" == "COMPLETED" ]] || {
  echo "Alpha analysis requires exact COMPLETED state" >&2
  exit 4
}

readonly LLAMA_RAW="${RAW_ROOT}/agate_alpha_llama_g0_v3"
readonly QWEN_RAW="${RAW_ROOT}/agate_alpha_qwen_g0_v3"
readonly LLAMA_OUT="${ANALYSIS_ROOT}/agate_alpha_llama_g0_v3"
readonly QWEN_OUT="${ANALYSIS_ROOT}/agate_alpha_qwen_g0_v3"
readonly PAIR_OUT="${ANALYSIS_ROOT}/agate_alpha_pair_g0_v3"
readonly POSTPIVOT_OUT="${ANALYSIS_ROOT}/agate_alpha_postpivot_pair_g0_v3"

for raw_directory in "${LLAMA_RAW}" "${QWEN_RAW}"; do
  for filename in summary.json analysis_cases.jsonl features.jsonl; do
    [[ -f "${raw_directory}/${filename}" && ! -L "${raw_directory}/${filename}" ]] || exit 2
  done
  [[ "$("${GREP_BIN}" -o -- '"all_pass":true' "${raw_directory}/summary.json" | wc -l)" == "1" ]] || {
    echo "Alpha analysis requires technical-valid summaries" >&2
    exit 2
  }
done
for output_directory in "${LLAMA_OUT}" "${QWEN_OUT}" "${PAIR_OUT}" "${POSTPIVOT_OUT}"; do
  [[ ! -e "${output_directory}" && ! -L "${output_directory}" ]] || {
    echo "Alpha compact analysis outputs are exclusive" >&2
    exit 2
  }
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${EASYEDIT_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_NO_SYNC=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY OPENAI_API_KEY

run_python() {
  "${UV_BIN}" run \
    --offline \
    --project "${EASYEDIT_ROOT}" \
    --frozen \
    --no-sync \
    python "$@"
}

run_python -m project.run_scripts.ode_edit_motivation.adaptive_gate_analysis \
  --analysis-cases "${LLAMA_RAW}/analysis_cases.jsonl" \
  --features "${LLAMA_RAW}/features.jsonl" \
  --model llama3-8b-inst \
  --track alphaedit_projected \
  --output-json "${LLAMA_OUT}/analysis.json" \
  --output-markdown "${LLAMA_OUT}/analysis.md"
run_python -m project.run_scripts.ode_edit_motivation.adaptive_gate_analysis \
  --analysis-cases "${QWEN_RAW}/analysis_cases.jsonl" \
  --features "${QWEN_RAW}/features.jsonl" \
  --model qwen2.5-7b-inst \
  --track alphaedit_projected \
  --output-json "${QWEN_OUT}/analysis.json" \
  --output-markdown "${QWEN_OUT}/analysis.md"
run_python -m project.run_scripts.ode_edit_motivation.adaptive_gate_pair_analysis \
  --llama-analysis "${LLAMA_OUT}/analysis.json" \
  --qwen-analysis "${QWEN_OUT}/analysis.json" \
  --track alphaedit_projected \
  --output-json "${PAIR_OUT}/analysis.json" \
  --output-markdown "${PAIR_OUT}/analysis.md"
run_python -m project.run_scripts.ode_edit_motivation.alpha_postpivot_analysis \
  --llama-analysis "${LLAMA_OUT}/analysis.json" \
  --qwen-analysis "${QWEN_OUT}/analysis.json" \
  --output-json "${POSTPIVOT_OUT}/analysis.json" \
  --output-markdown "${POSTPIVOT_OUT}/analysis.md"
