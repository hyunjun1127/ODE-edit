#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=65000M
#SBATCH --time=23:59:00

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "P1R16 missing SLURM_SUBMIT_DIR" >&2
  exit 2
fi
REPO_ROOT="$(pwd -P)"
if [[ "$(realpath -e -- "$SLURM_SUBMIT_DIR")" != "$REPO_ROOT" ]]; then
  echo "P1R16 submit directory differs" >&2
  exit 2
fi
if [[ -z "${ODEBF_P1R16_MODEL_ALIAS:-}" || -z "${ODEBF_P1R16_OUTPUT_ROOT:-}" || -z "${ODEBF_P1R16_SOURCE_HEAD:-}" ]]; then
  echo "P1R16 launcher identity is incomplete" >&2
  exit 2
fi
if [[ "${HF_HUB_OFFLINE:-}" != "1" || "${TRANSFORMERS_OFFLINE:-}" != "1" ]]; then
  echo "P1R16 offline environment differs" >&2
  exit 2
fi
EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"
if [[ ! -d "$EASYEDIT_ROOT/easyeditor" || -L "$EASYEDIT_ROOT/easyeditor" ]]; then
  echo "P1R16 EasyEdit source path differs" >&2
  exit 2
fi
export PYTHONPATH="$EASYEDIT_ROOT"
observed_head="$(git rev-parse HEAD)"
observed_parent="$(git rev-parse HEAD^)"
observed_branch="$(git branch --show-current)"
if [[ "$observed_head" != "$ODEBF_P1R16_SOURCE_HEAD" || "$observed_parent" != "ce0d7c0a9ccb91a7d67209bf93cf6b6ac34cff48" || "$observed_branch" != "codex/odeeditsh1-s05-coupled-demand-soft-transport-p1r16-v1" ]]; then
  echo "P1R16 source chain differs" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "P1R16 tracked worktree differs" >&2
  exit 2
fi
bash scripts/check-session-boundary.sh 019fe489-c968-75f3-9965-7cfbc26c0a99 >/dev/null
exec uv run --offline --project /mnt/raid5/janghj/EasyEdit --frozen --no-sync \
  python project/run_scripts/session05_ode_bf_coupled_demand_soft_transport.py \
  --model "$ODEBF_P1R16_MODEL_ALIAS" \
  --output-root "$ODEBF_P1R16_OUTPUT_ROOT" \
  --source-head "$ODEBF_P1R16_SOURCE_HEAD" \
  --run-token coupled-demand-soft-transport-p1r16-a1-stage-a-v1
