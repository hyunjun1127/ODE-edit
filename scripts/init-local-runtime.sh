#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
local_root="${AGENT_CONTROL_LOCAL_ROOT:-${repo_root}/local}"

mkdir -p \
  "${local_root}/results/raw" \
  "${local_root}/datasets" \
  "${local_root}/checkpoints" \
  "${local_root}/logs/slurm" \
  "${local_root}/logs/run" \
  "${local_root}/transfers" \
  "${local_root}/secrets"

chmod 700 "${local_root}/secrets" 2>/dev/null || true

printf 'initialized local runtime layout\n'
printf 'local_root=%s\n' "${local_root}"
