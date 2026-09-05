#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server4
#SBATCH --job-name=odeedit_orbode_cum_s4_r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodelist=server4
#SBATCH --mem=60416M
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%2
#SBATCH --export=NONE
#SBATCH --output=local/logs/orbode-cumulative-rerun-v1/%A_%a.out
#SBATCH --error=local/logs/orbode-cumulative-rerun-v1/%A_%a.err
set -euo pipefail
umask 077
# Reuse the sealed session/source/offline/model/resource launch binding.
# Nested #SBATCH comments are not processed by bash; this is one allocation.
export ODE_ORBODE_CUMULATIVE=1
exec bash "${SLURM_SUBMIT_DIR:?}/project/run_scripts/session06_orbode_sequential_server4.sbatch" "$@"
