#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/check-slurm-resource-cap.sh SERVER REQUESTED_GPUS REQUESTED_MEM_MB [JOB_PATTERNS]

Checks the project GPU concurrency cap and the configured maximum total Slurm
memory request. REQUESTED_MEM_MB is the job-total value that will be passed to
Slurm (for example, --mem=64000M), not GPU VRAM.

The local cap TSV has this format:
  server<TAB>slurm_node<TAB>max_project_gpus<TAB>mem_mb_per_gpu<TAB>job_patterns

Exit codes: 0 allowed; 2 invalid/missing configuration; 4 keep pending.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server="${1:-}"
requested_gpus="${2:-}"
requested_mem_mb="${3:-}"
job_patterns="${4:-}"
caps_file="${AGENT_GPU_CAPS_FILE:-${repo_root}/servers/local/gpu-caps.tsv}"

if [[ -z "${server}" || -z "${requested_gpus}" || -z "${requested_mem_mb}" ]]; then
  usage >&2
  exit 2
fi
if ! [[ "${requested_gpus}" =~ ^[0-9]+$ ]] || [[ "${requested_gpus}" -lt 1 ]] || ! [[ "${requested_mem_mb}" =~ ^[0-9]+$ ]] || [[ "${requested_mem_mb}" -lt 1 ]]; then
  echo "ERROR: GPU and memory requests must be positive integers" >&2
  exit 2
fi
if [[ ! -f "${caps_file}" ]]; then
  echo "ERROR: missing resource cap config: ${caps_file}" >&2
  exit 2
fi

mem_mb_per_gpu=""
while IFS=$'\t' read -r cfg_server _cfg_node _cfg_cap cfg_mem_mb_per_gpu _cfg_patterns rest; do
  [[ -n "${cfg_server}" ]] || continue
  [[ "${cfg_server}" != \#* ]] || continue
  [[ "${cfg_server}" == "${server}" ]] || continue
  mem_mb_per_gpu="${cfg_mem_mb_per_gpu}"
  break
done < "${caps_file}"

if ! [[ "${mem_mb_per_gpu}" =~ ^[0-9]+$ ]] || [[ "${mem_mb_per_gpu}" -lt 1 ]]; then
  echo "ERROR: missing or invalid mem_mb_per_gpu for server=${server}" >&2
  exit 2
fi

max_mem_mb=$((requested_gpus * mem_mb_per_gpu))
if [[ "${requested_mem_mb}" -gt "${max_mem_mb}" ]]; then
  echo "DENY server=${server} requested_mem_mb=${requested_mem_mb} max_mem_mb=${max_mem_mb} requested_gpus=${requested_gpus} decision=pending_resource_cap"
  exit 4
fi

if [[ -n "${job_patterns}" ]]; then
  "${repo_root}/scripts/check-slurm-gpu-cap.sh" "${server}" "${requested_gpus}" "${job_patterns}"
else
  "${repo_root}/scripts/check-slurm-gpu-cap.sh" "${server}" "${requested_gpus}"
fi

echo "ALLOW_MEMORY server=${server} requested_mem_mb=${requested_mem_mb} max_mem_mb=${max_mem_mb} decision=submit_now"
