#!/usr/bin/env bash
# HA1 Llama/Qwen Official dense Alpha replay.
#SBATCH --job-name=fzcb_ha1_s2
#SBATCH --partition=gpu
#SBATCH --nodelist=server2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65000M
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/fzcb-hard-alpha-densec-ha0-b10-v1/ha1-%A_%a.scheduler.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/fzcb-hard-alpha-densec-ha0-b10-v1/ha1-%A_%a.scheduler.err

set -euo pipefail
umask 077
: "${FZCB_SOURCE_ROOT:?}"
: "${FZCB_EXPECTED_HEAD:?}"
: "${FZCB_OFFICIAL_ROOT:?}"
: "${FZCB_RESULT_ROOT:?}"
: "${FZCB_LOG_ROOT:?}"
: "${FZCB_PREFLIGHT:?}"
: "${FZCB_PREFLIGHT_SHA256:?}"
readonly MODELS=(llama3-8b-inst qwen2.5-7b-inst)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CELL="${FZCB_RESULT_ROOT}/${MODEL}"
readonly OUTPUT="${CELL}/ha1-result.json"
readonly DATASET=/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json
readonly CASE_MANIFEST="${FZCB_SOURCE_ROOT}/project/run_scripts/fzcb_hard_alpha_densec/config/case-manifest-v1.json"
[[ "$(hostname -s)" == server2 ]]
[[ "$(git -C "${FZCB_SOURCE_ROOT}" rev-parse HEAD)" == "${FZCB_EXPECTED_HEAD}" ]]
[[ -z "$(git -C "${FZCB_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "${FZCB_OFFICIAL_ROOT}" rev-parse HEAD)" == 3488a66ee988d83ee7891a8abbbe6bcb24a77daf ]]
[[ -z "$(git -C "${FZCB_OFFICIAL_ROOT}" status --porcelain)" ]]
[[ "$(sha256sum "${FZCB_PREFLIGHT}" | awk '{print $1}')" == "${FZCB_PREFLIGHT_SHA256}" ]]
[[ ! -e "${CELL}" && ! -L "${CELL}" ]]
mkdir -p "${CELL}" "${FZCB_LOG_ROOT}"
exec >"${FZCB_LOG_ROOT}/ha1-${MODEL}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>"${FZCB_LOG_ROOT}/ha1-${MODEL}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/mnt/raid5/janghj/.cache/huggingface HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export PYTHONPATH="${FZCB_SOURCE_ROOT}:${FZCB_OFFICIAL_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /usr/local/bin/uv run --offline --project /mnt/raid5/janghj/EasyEdit --frozen --no-sync python \
  -m project.run_scripts.fzcb_hard_alpha_densec.ha1_replay \
  --model "${MODEL}" --source-root "${FZCB_SOURCE_ROOT}" --expected-head "${FZCB_EXPECTED_HEAD}" \
  --official-root "${FZCB_OFFICIAL_ROOT}" --dataset "${DATASET}" --case-manifest "${CASE_MANIFEST}" \
  --preflight "${FZCB_PREFLIGHT}" --preflight-sha256 "${FZCB_PREFLIGHT_SHA256}" --output "${OUTPUT}"
