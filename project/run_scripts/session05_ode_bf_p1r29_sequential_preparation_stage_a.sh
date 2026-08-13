#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"
easyedit_root="/mnt/raid5/janghj/EasyEdit"

cd "${repo_root}"
export PYTHONPATH="${easyedit_root}:${repo_root}"
uv run --offline --project "${easyedit_root}" python -W error -m unittest \
  project.run_scripts.ode_bf.tests.test_p1r29_sequential_preparation
uv run --offline --project "${easyedit_root}" python \
  project/run_scripts/session05_ode_bf_p1r29_sequential_preparation_stage_a.py

