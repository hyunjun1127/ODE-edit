#!/usr/bin/env bash
# Physically separate Q_gate stage, requiring immutable selector lock.
#SBATCH --job-name=barrier_f2_gate_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=48000M
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/janghj/ODE-edit/local/logs/barrier-usefulness-f2-v1/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/barrier-usefulness-f2-v1/%x-%A_%a.err

set -euo pipefail
umask 077
: "${BARRIER_F2_SOURCE_ROOT:?}"
: "${BARRIER_F2_EXPECTED_HEAD:?}"
: "${BARRIER_F2_ANCHOR_ROOT:?}"
: "${BARRIER_F2_CTRL_ROOT:?}"
: "${BARRIER_F2_GATE_ROOT:?}"
: "${BARRIER_F2_SELECTOR_LOCK:?}"
: "${BARRIER_F2_SELECTOR_IDENTITY:?}"
readonly MODELS=(llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst)
readonly METHODS=(alphaedit memit alphaedit memit)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"
readonly CTRL_CELL="${BARRIER_F2_CTRL_ROOT}/${MODEL}-${METHOD}"
readonly GATE_CELL="${BARRIER_F2_GATE_ROOT}/${MODEL}-${METHOD}"
readonly ANCHOR="${BARRIER_F2_ANCHOR_ROOT}/${MODEL}/anchor-seal.json"
[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${BARRIER_F2_SOURCE_ROOT}" rev-parse HEAD)" == "${BARRIER_F2_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${BARRIER_F2_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
for path in "${ANCHOR}" "${CTRL_CELL}/ctrl-result.json" "${BARRIER_F2_SELECTOR_LOCK}"; do [[ -f "${path}" && ! -L "${path}" ]]; done
[[ ! -e "${GATE_CELL}" && ! -L "${GATE_CELL}" ]]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/janghj/.cache/huggingface HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${BARRIER_F2_SOURCE_ROOT}:/data/janghj/EasyEdit-stock-14cea824"
exec /data/janghj/.local/bin/uv run --offline --project /data/janghj/EasyEdit --frozen --no-sync python \
  -m project.run_scripts.barrier_usefulness.phase_b_gate --model "${MODEL}" --method "${METHOD}" \
  --source-root "${BARRIER_F2_SOURCE_ROOT}" --dataset /data/janghj/EasyEdit/data/counterfact/counterfact.json \
  --anchor-seal "${ANCHOR}" --ctrl-result "${CTRL_CELL}/ctrl-result.json" --ctrl-root "${CTRL_CELL}" \
  --gate-root "${GATE_CELL}" --selector-lock "${BARRIER_F2_SELECTOR_LOCK}" \
  --selector-identity "${BARRIER_F2_SELECTOR_IDENTITY}" --output "${GATE_CELL}/gate-result.json"
