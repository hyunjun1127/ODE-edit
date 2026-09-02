#!/usr/bin/env bash
# ODEEDIT_SLURM_SERVER=server2
#SBATCH --job-name=odealloc-s04-p0-template
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60416M
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=local/odealloc/logs/p0-%A-%a.out
#SBATCH --error=local/odealloc/logs/p0-%A-%a.err

set -euo pipefail

printf '%s\n' 'NO-SUBMIT TEMPLATE: a later GH execution envelope must replace this guard.' >&2
exit 64
