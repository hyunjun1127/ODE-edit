#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/check-slurm-gpu-cap.sh SERVER REQUESTED_GPUS [JOB_PATTERNS]

Check whether submitting another Slurm job would exceed this repo's per-server
running GPU cap. The script counts RUNNING/COMPLETING/CONFIGURING jobs on the configured Slurm node
whose job name matches one of the comma-separated shell patterns.

Cap configuration (bounded by control/gpu-concurrency-policy.tsv):
  preferred: servers/local/gpu-caps.tsv
  override:  AGENT_GPU_CAPS_FILE=/path/to/gpu-caps.tsv
  env:       AGENT_GPU_CAP_<SERVER>, AGENT_GPU_NODE_<SERVER>,
             AGENT_GPU_JOB_PATTERNS_<SERVER>, AGENT_DEFAULT_GPU_CAP

TSV format:
  server<TAB>slurm_node<TAB>max_project_gpus<TAB>mem_mb_per_gpu<TAB>job_patterns

Default job patterns: project_*,motivation_*,session01_*.

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
  job_patterns="${AGENT_PROJECT_JOB_PATTERNS:-project_*,motivation_*,session01_*}"
fi

if [[ -f "${caps_file}" ]]; then
  while IFS=$'\t' read -r cfg_server cfg_node cfg_cap _cfg_mem_mb_per_gpu cfg_patterns rest; do
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

# A stale task/worktree cap must not enlarge the current tracked ceiling.
# This is admission control only: it never modifies existing Slurm jobs.
policy_file="${repo_root}/control/gpu-concurrency-policy.tsv"
if [[ ! -f "${policy_file}" ]]; then
  echo "ERROR: missing tracked GPU concurrency policy" >&2
  exit 2
fi
policy_cap=""
while IFS=$'\t' read -r policy_server policy_value rest; do
  [[ "${policy_server}" == "${server}" ]] || continue
  policy_cap="${policy_value}"
  break
done < "${policy_file}"
if ! [[ "${policy_cap}" =~ ^[0-9]+$ ]] || [[ "${policy_cap}" -lt 1 ]]; then
  echo "ERROR: missing/invalid tracked GPU ceiling for server=${server}" >&2
  exit 2
fi
if [[ "${cap}" -gt "${policy_cap}" ]]; then
  cap="${policy_cap}"
fi

if [[ "${requested_gpus}" -gt "${cap}" ]]; then
  echo "DENY server=${server} requested_gpus=${requested_gpus} cap=${cap} reason=single_job_exceeds_cap"
  exit 4
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "ERROR: squeue is required for GPU cap checks" >&2
  exit 2
fi

if ! queue_snapshot="$(squeue -h -t RUNNING,COMPLETING,CONFIGURING -w "${node}" -o '%i|%j|%b')"; then
  echo "ERROR: scheduler query failed; resource admission is unresolved" >&2
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

  # Some Slurm versions emit N/A for %b when jobs request GPUs via --gpus.
  # The running job record still carries the allocated TRES in scontrol.
  if [[ "${tres_per_node}" == "N/A" ]]; then
    if ! command -v scontrol >/dev/null 2>&1 ||
       ! job_record="$(scontrol show job "${job_id}" --oneliner 2>/dev/null)"; then
      echo "ERROR: GPU allocation unresolved for job=${job_id}" >&2
      exit 2
    fi
    if [[ "${job_record}" =~ AllocTRES=([^[:space:]]+) ]]; then
      tres_per_node="${BASH_REMATCH[1]}"
    elif [[ "${job_record}" =~ ReqTRES=([^[:space:]]+) ]]; then
      # Conservative reservation for a configuring job without AllocTRES yet.
      tres_per_node="${BASH_REMATCH[1]}"
    else
      echo "ERROR: missing allocation/reservation TRES for job=${job_id}" >&2
      exit 2
    fi
  fi

  gpu_count=0
  if [[ "${tres_per_node}" =~ gpu(:[^,=]+)?:([0-9]+) ]]; then
    gpu_count="${BASH_REMATCH[2]}"
  elif [[ "${tres_per_node}" =~ gpu[^,=]*=([0-9]+) ]]; then
    gpu_count="${BASH_REMATCH[1]}"
  fi
  active_gpus=$((active_gpus + gpu_count))
done <<< "${queue_snapshot}"

total=$((active_gpus + requested_gpus))
if [[ "${total}" -gt "${cap}" ]]; then
  echo "DENY server=${server} node=${node} job_patterns=${job_patterns} active_project_gpus=${active_gpus} requested_gpus=${requested_gpus} cap=${cap} decision=pending_resource_cap"
  exit 4
fi

echo "ALLOW server=${server} node=${node} job_patterns=${job_patterns} active_project_gpus=${active_gpus} requested_gpus=${requested_gpus} cap=${cap} decision=submit_now"
