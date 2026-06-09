#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/check-slurm-gpu-cap.sh SERVER REQUESTED_GPUS [JOB_PATTERNS]

Check whether submitting another Slurm job would exceed this repo's per-server
running GPU cap. The script counts RUNNING jobs on the configured Slurm node
whose job name matches one of the comma-separated shell patterns.

Cap configuration:
  preferred: servers/local/gpu-caps.tsv
  override:  AGENT_GPU_CAPS_FILE=/path/to/gpu-caps.tsv
  env:       AGENT_GPU_CAP_<SERVER>, AGENT_GPU_NODE_<SERVER>,
             AGENT_GPU_JOB_PATTERNS_<SERVER>, AGENT_DEFAULT_GPU_CAP

TSV format:
  server<TAB>slurm_node<TAB>max_project_gpus<TAB>job_patterns

Default job patterns: project_*,*_stage0,stage0_*.

Exit codes:
  0: submit is allowed under the cap
  2: usage, missing config, or unsupported server/request
  4: requested job should stay pending_resource_cap
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

server="${1:-}"
requested_gpus="${2:-}"
job_patterns_arg="${3:-}"
caps_file="${AGENT_GPU_CAPS_FILE:-${repo_root}/servers/local/gpu-caps.tsv}"

if [[ -z "${server}" || -z "${requested_gpus}" ]]; then
  usage >&2
  exit 2
fi

if ! [[ "${requested_gpus}" =~ ^[0-9]+$ ]] || [[ "${requested_gpus}" -lt 1 ]]; then
  echo "ERROR: REQUESTED_GPUS must be a positive integer" >&2
  exit 2
fi

server_env="$(printf '%s' "${server}" | tr '[:lower:]-' '[:upper:]_')"
cap_var="AGENT_GPU_CAP_${server_env}"
node_var="AGENT_GPU_NODE_${server_env}"
patterns_var="AGENT_GPU_JOB_PATTERNS_${server_env}"

cap="${!cap_var-}"
node="${!node_var-}"
job_patterns="${!patterns_var-}"

if [[ -z "${cap}" ]]; then
  cap="${AGENT_DEFAULT_GPU_CAP:-}"
fi
if [[ -z "${node}" ]]; then
  node="${server}"
fi
if [[ -z "${job_patterns}" ]]; then
  job_patterns="${AGENT_PROJECT_JOB_PATTERNS:-project_*,*_stage0,stage0_*}"
fi

if [[ -f "${caps_file}" ]]; then
  while IFS=$'\t' read -r cfg_server cfg_node cfg_cap cfg_patterns rest; do
    [[ -n "${cfg_server}" ]] || continue
    [[ "${cfg_server}" != \#* ]] || continue
    [[ "${cfg_server}" == "${server}" ]] || continue
    [[ -n "${cfg_node}" ]] && node="${cfg_node}"
    [[ -n "${cfg_cap}" ]] && cap="${cfg_cap}"
    [[ -n "${cfg_patterns}" ]] && job_patterns="${cfg_patterns}"
    break
  done < "${caps_file}"
fi

if [[ -z "${cap}" ]]; then
  echo "ERROR: missing GPU cap for server=${server}" >&2
  echo "Create servers/local/gpu-caps.tsv from servers/templates/gpu-caps.tsv or set ${cap_var}." >&2
  exit 2
fi

if ! [[ "${cap}" =~ ^[0-9]+$ ]] || [[ "${cap}" -lt 1 ]]; then
  echo "ERROR: cap for server=${server} must be a positive integer; got ${cap}" >&2
  exit 2
fi

if [[ -n "${job_patterns_arg}" ]]; then
  job_patterns="${job_patterns_arg}"
fi

if [[ "${requested_gpus}" -gt "${cap}" ]]; then
  echo "DENY server=${server} requested_gpus=${requested_gpus} cap=${cap} reason=single_job_exceeds_cap"
  exit 4
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "ERROR: squeue is required for GPU cap checks" >&2
  exit 2
fi

active_gpus=0
while IFS='|' read -r job_id job_name tres_per_node; do
  [[ -n "${job_id}" ]] || continue

  matched=0
  IFS=',' read -r -a patterns <<< "${job_patterns}"
  for pattern in "${patterns[@]}"; do
    [[ -n "${pattern}" ]] || continue
    if [[ "${job_name}" == ${pattern} ]]; then
      matched=1
      break
    fi
  done
  [[ "${matched}" == "1" ]] || continue

  gpu_count=0
  if [[ "${tres_per_node}" =~ gpu(:[^,=]+)?:([0-9]+) ]]; then
    gpu_count="${BASH_REMATCH[2]}"
  elif [[ "${tres_per_node}" =~ gpu[^,=]*=([0-9]+) ]]; then
    gpu_count="${BASH_REMATCH[1]}"
  fi
  active_gpus=$((active_gpus + gpu_count))
done < <(squeue -h -t RUNNING -w "${node}" -o '%i|%j|%b')

total=$((active_gpus + requested_gpus))
if [[ "${total}" -gt "${cap}" ]]; then
  echo "DENY server=${server} node=${node} job_patterns=${job_patterns} active_project_gpus=${active_gpus} requested_gpus=${requested_gpus} cap=${cap} decision=pending_resource_cap"
  exit 4
fi

echo "ALLOW server=${server} node=${node} job_patterns=${job_patterns} active_project_gpus=${active_gpus} requested_gpus=${requested_gpus} cap=${cap} decision=submit_now"
