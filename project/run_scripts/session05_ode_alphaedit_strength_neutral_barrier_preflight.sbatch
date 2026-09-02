#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server1
# ODE-edit hook preflight against an immutable stock EasyEdit checkout.
#SBATCH --job-name=ode_snb_preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=180000M
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/raid5/janghj/ODE-edit/local/logs/strength-neutral-barrier/%x-%j.out
#SBATCH --error=/mnt/raid5/janghj/ODE-edit/local/logs/strength-neutral-barrier/%x-%j.err

set -euo pipefail
umask 077

: "${ODE_SNB_SOURCE_ROOT:?}"
: "${ODE_SNB_EXPECTED_HEAD:?}"
: "${ODE_SNB_EASYEDIT_ROOT:?}"
: "${ODE_SNB_EXPECTED_EASYEDIT_HEAD:?}"
: "${ODE_SNB_PREFLIGHT_ROOT:?}"

readonly UV_BIN="/usr/local/bin/uv"
readonly ENV_PROJECT="${ODE_SNB_ENV_PROJECT:-/mnt/raid5/janghj/EasyEdit}"
readonly MODEL_PATH="${ODE_SNB_MODEL_PATH:-/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2}"
readonly DATASET="${ODE_SNB_DATASET:-/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json}"
readonly PROJECTOR="${ODE_SNB_PROJECTOR:-/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt}"
readonly HPARAMS="${ODE_SNB_HPARAMS:-${ODE_SNB_EASYEDIT_ROOT}/hparams/AlphaEdit/llama3-8b.yaml}"
readonly RUNNER="${ODE_SNB_SOURCE_ROOT}/project/run_scripts/alphaedit_strength_neutral_barrier/run_preflight.py"

if [[ -n "${ODE_SNB_EXPECTED_HOST:-}" ]]; then
  [[ "$(hostname -s)" == "${ODE_SNB_EXPECTED_HOST}" ]] || exit 2
fi
[[ "$(git -C "${ODE_SNB_SOURCE_ROOT}" rev-parse HEAD)" == "${ODE_SNB_EXPECTED_HEAD}" ]] || exit 2
[[ -z "$(git -C "${ODE_SNB_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 2
[[ "$(git -C "${ODE_SNB_EASYEDIT_ROOT}" rev-parse HEAD)" == "${ODE_SNB_EXPECTED_EASYEDIT_HEAD}" ]] || exit 2
[[ -z "$(git -C "${ODE_SNB_EASYEDIT_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 2
for path in "${MODEL_PATH}" "${DATASET}" "${PROJECTOR}" "${HPARAMS}" "${RUNNER}"; do
  [[ -e "${path}" && ! -L "${path}" ]] || exit 2
done
[[ ! -e "${ODE_SNB_PREFLIGHT_ROOT}" && ! -L "${ODE_SNB_PREFLIGHT_ROOT}" ]] || exit 2

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export HF_HOME=/mnt/raid5/janghj/.cache/huggingface
export HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub
export PYTHONPATH="${ODE_SNB_SOURCE_ROOT}:${ODE_SNB_EASYEDIT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${UV_BIN}" run --offline --project "${ENV_PROJECT}" --frozen --no-sync python "${RUNNER}" \
  --result-dir "${ODE_SNB_PREFLIGHT_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --dataset "${DATASET}" \
  --projector "${PROJECTOR}" \
  --hparams "${HPARAMS}" \
  --expected-head "${ODE_SNB_EXPECTED_HEAD}" \
  --easyedit-root "${ODE_SNB_EASYEDIT_ROOT}" \
  --expected-easyedit-head "${ODE_SNB_EXPECTED_EASYEDIT_HEAD}"
