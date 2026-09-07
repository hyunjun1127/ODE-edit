#!/usr/bin/env bash
# Pure CPU admission tests: no real scheduler calls or existing-job mutations.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENT_GPU_CAPS_FILE=/dev/null
export AGENT_GPU_CAP_SERVER1=4 AGENT_GPU_CAP_SERVER2=3
export AGENT_GPU_CAP_SERVER3=2 AGENT_GPU_CAP_SERVER4=4
export AGENT_PROJECT_JOB_PATTERNS='odeedit_*'
squeue() {
  [[ " $* " == *" RUNNING,COMPLETING,CONFIGURING "* ]] || return 97
  [[ "${MOCK_SQUEUE_FAIL:-0}" == 0 ]] || return 98
  printf '%s' "${MOCK_QUEUE:-}"
}
export -f squeue
scontrol() {
  [[ "${MOCK_SCONTROL_FAIL:-0}" == 0 ]] || return 98
  printf '%s' "${MOCK_JOB_RECORD:-JobId=10 AllocTRES=cpu=1,gres/gpu=2}"
}
export -f scontrol
check() {
  local expected="$1" server="$2" requested="$3" output rc
  set +e
  output="$(bash "${repo_root}/scripts/check-slurm-gpu-cap.sh" "$server" "$requested" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" == "$expected" ]] || { printf '%s\n' "$output"; return 1; }
}
export MOCK_QUEUE=''
for server in server1 server2 server3 server4; do
  check 0 "$server" 2
  check 4 "$server" 3
done
export MOCK_QUEUE=$'10|odeedit_existing|gpu:1\n'
check 0 server1 1
check 4 server1 2
export MOCK_QUEUE=$'10|odeedit_existing|gpu:2\n'
check 4 server1 1
# Grandfathered over-cap runs remain untouched; only new admission is refused.
export MOCK_QUEUE=$'10|odeedit_existing|gpu:4\n'
check 4 server4 1
export MOCK_QUEUE='' AGENT_GPU_CAP_SERVER1=1
check 4 server1 2
export AGENT_GPU_CAP_SERVER3=0
check 2 server3 1
export AGENT_GPU_CAP_SERVER1=4 MOCK_SQUEUE_FAIL=1
check 2 server1 1
export MOCK_SQUEUE_FAIL=0 MOCK_QUEUE=$'10|odeedit_existing|N/A\n'
check 4 server1 1
export MOCK_SCONTROL_FAIL=1
check 2 server1 1
export MOCK_SCONTROL_FAIL=0 MOCK_JOB_RECORD='JobId=10'
check 2 server1 1
export MOCK_JOB_RECORD='JobId=10 AllocTRES=cpu=1,mem=1G'
check 0 server1 1
printf '%s\n' 'PASS 19 prospective cap / unresolved-allocation / stricter-local / disabled-server checks'
