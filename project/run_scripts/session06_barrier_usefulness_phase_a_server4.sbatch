#!/usr/bin/env bash
# Corrected F1b Phase-A launcher; array cells are model/method.
#SBATCH --job-name=barrier_f2a_s4
#SBATCH --partition=gpu
#SBATCH --nodelist=server4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=90000M
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/janghj/ODE-edit/local/logs/barrier-usefulness-f2-v1/%x-%A_%a.out
#SBATCH --error=/data/janghj/ODE-edit/local/logs/barrier-usefulness-f2-v1/%x-%A_%a.err

set -euo pipefail
umask 077

: "${BARRIER_F2_SOURCE_ROOT:?}"
: "${BARRIER_F2_EXPECTED_HEAD:?}"
: "${BARRIER_F2_STAGE:?}"
: "${BARRIER_F2_RESULT_ROOT:?}"
: "${BARRIER_F2_G0_RECEIPT:?}"

readonly EASYEDIT_ROOT=/data/janghj/EasyEdit-stock-14cea824
readonly EXPECTED_EASYEDIT_HEAD=14cea8245f06715684592ab55184939b99d70784
readonly UV_BIN=/data/janghj/.local/bin/uv
readonly ENV_PROJECT=/data/janghj/EasyEdit
readonly DATASET=/data/janghj/EasyEdit/data/counterfact/counterfact.json
readonly CASE_MANIFEST="${BARRIER_F2_SOURCE_ROOT}/project/run_scripts/fixed_z_nonuniqueness/config/case-manifest-v1.json"
readonly MODELS=(llama3-8b-inst llama3-8b-inst qwen2.5-7b-inst qwen2.5-7b-inst)
readonly METHODS=(alphaedit memit alphaedit memit)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"
readonly CELL_ROOT="${BARRIER_F2_RESULT_ROOT}/${MODEL}-${METHOD}-${BARRIER_F2_STAGE}"

[[ "$(hostname -s)" == server4 ]]
[[ "$(git -C "${BARRIER_F2_SOURCE_ROOT}" rev-parse HEAD)" == "${BARRIER_F2_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${BARRIER_F2_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${EASYEDIT_ROOT}" rev-parse HEAD)" == "${EXPECTED_EASYEDIT_HEAD}" ]]
[[ -z "$(git -C "${EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ -f "${BARRIER_F2_G0_RECEIPT}" && ! -L "${BARRIER_F2_G0_RECEIPT}" ]]
[[ ! -e "${CELL_ROOT}" && ! -L "${CELL_ROOT}" ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/data/janghj/.cache/huggingface
export HF_HUB_CACHE=/data/janghj/.cache/huggingface/hub
export PYTHONPATH="${BARRIER_F2_SOURCE_ROOT}:${EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python -m project.run_scripts.barrier_usefulness.phase_a \
  --stage "${BARRIER_F2_STAGE}" --model "${MODEL}" --method "${METHOD}" \
  --source-root "${BARRIER_F2_SOURCE_ROOT}" --dataset "${DATASET}" --case-manifest "${CASE_MANIFEST}" \
  --output "${CELL_ROOT}/result.json"
