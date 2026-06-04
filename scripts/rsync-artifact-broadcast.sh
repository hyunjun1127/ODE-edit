#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/rsync-artifact-broadcast.sh [--dry-run] [--target SERVER ...] LOCAL_PATH

Broadcast one Git-excluded local artifact path to configured active servers.
LOCAL_PATH must be local/ or under local/.

Targets are read from servers/local/rsync-targets.tsv unless --target is used.
Format:
  server<TAB>ssh_alias<TAB>repo_path

Example:
  server1	rke-server1	/mnt/shared/project-repo

Private/runtime-only trees are refused: local/secrets, local/scratch,
local/run_scripts, local/conflicts, local/tmp, local/.cache.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ssh_config="${AGENT_SSH_CONFIG:-${repo_root}/servers/local/ssh_config}"
targets_file="${AGENT_RSYNC_TARGETS_FILE:-${repo_root}/servers/local/rsync-targets.tsv}"
current_server="${AGENT_CURRENT_SERVER:-$(git config --get agent.hostname 2>/dev/null || hostname -s)}"
dry_run=0
targets=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --target)
      [[ $# -ge 2 ]] || { echo "ERROR: --target requires a server name" >&2; exit 2; }
      targets+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -eq 1 ]] || { usage >&2; exit 2; }

source_arg="$1"
if [[ "${source_arg}" = /* ]]; then
  source_abs="${source_arg%/}"
else
  source_abs="${repo_root}/${source_arg%/}"
fi

if [[ ! -e "${source_abs}" ]]; then
  echo "ERROR: source does not exist: ${source_abs}" >&2
  exit 2
fi

case "${source_abs}" in
  "${repo_root}/local"|\
  "${repo_root}/local/"*) ;;
  *)
    echo "ERROR: source must be under repo local/: ${source_abs}" >&2
    exit 2
    ;;
esac

case "${source_abs}" in
  "${repo_root}/local/secrets"|\
  "${repo_root}/local/secrets/"*|\
  "${repo_root}/local/scratch"|\
  "${repo_root}/local/scratch/"*|\
  "${repo_root}/local/run_scripts"|\
  "${repo_root}/local/run_scripts/"*|\
  "${repo_root}/local/conflicts"|\
  "${repo_root}/local/conflicts/"*|\
  "${repo_root}/local/tmp"|\
  "${repo_root}/local/tmp/"*|\
  "${repo_root}/local/.cache"|\
  "${repo_root}/local/.cache/"*)
    echo "ERROR: refusing private/runtime-only local tree: ${source_abs}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${targets_file}" ]]; then
  echo "ERROR: missing target config: ${targets_file}" >&2
  echo "Create it as local-only TSV: server<TAB>ssh_alias<TAB>repo_path" >&2
  exit 3
fi

if [[ ! -f "${ssh_config}" ]]; then
  ssh_config="${HOME}/.ssh/config"
fi

source_rel="${source_abs#"${repo_root}/"}"
rsync_args=(
  -avz
  --info=progress2
  --partial
  "--exclude=/secrets/***"
  "--exclude=/scratch/***"
  "--exclude=/run_scripts/***"
  "--exclude=/conflicts/***"
  "--exclude=/tmp/***"
  "--exclude=/.cache/***"
)
if [[ "${dry_run}" == "1" ]]; then
  rsync_args+=(--dry-run)
fi

target_filter=" ${targets[*]} "
while IFS=$'\t' read -r server alias repo_path rest; do
  [[ -n "${server}" ]] || continue
  [[ "${server}" != \#* ]] || continue
  [[ -n "${alias}" && -n "${repo_path}" ]] || {
    echo "WARN skipping malformed target row for server=${server}" >&2
    continue
  }
  if [[ "${server}" == "${current_server}" ]]; then
    echo "SKIP ${server}: current server"
    continue
  fi
  if [[ ${#targets[@]} -gt 0 && "${target_filter}" != *" ${server} "* ]]; then
    continue
  fi

  dest="${repo_path}/${source_rel}"
  dest_parent="$(dirname "${dest}")"
  echo "Checking SSH alias ${alias} for ${server}"
  ssh -F "${ssh_config}" -o BatchMode=yes -o ConnectTimeout=10 "${alias}" "mkdir -p '${dest_parent}'"
  echo "Broadcast ${source_rel} -> ${server}:${dest}"
  rsync "${rsync_args[@]}" -e "ssh -F ${ssh_config}" "${source_abs}/" "${alias}:${dest}/"
done < "${targets_file}"
