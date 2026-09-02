#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit-artifact-broadcast-dependency.sh --job-id JOB_ID [--job-id JOB_ID ...] [--target SERVER ...] [--name NAME] LOCAL_PATH [...]

Submit a Slurm afterany dependency job that broadcasts completed local artifacts.
LOCAL_PATH may be a future path, but it must be local/ or under local/.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_ids=()
targets=()
paths=()
broadcast_name="artifact_broadcast"
time_limit="${SLURM_BROADCAST_TIME:-04:00:00}"
mem="${SLURM_BROADCAST_MEM:-2G}"
cpus="${SLURM_BROADCAST_CPUS:-1}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      [[ $# -ge 2 ]] || { echo "ERROR: --job-id requires a value" >&2; exit 2; }
      job_ids+=("$2")
      shift 2
      ;;
    --target)
      [[ $# -ge 2 ]] || { echo "ERROR: --target requires a value" >&2; exit 2; }
      targets+=("$2")
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "ERROR: --name requires a value" >&2; exit 2; }
      broadcast_name="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
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
      paths+=("${1%/}")
      shift
      ;;
  esac
done

if [[ ${#job_ids[@]} -eq 0 || ${#paths[@]} -eq 0 ]]; then
  usage >&2
  exit 2
fi

for path in "${paths[@]}"; do
  case "${path}" in
    local|local/*) ;;
    *)
      echo "ERROR: broadcast path must be repo-relative under local/: ${path}" >&2
      exit 2
      ;;
  esac
  case "${path}" in
    local/secrets|local/secrets/*|\
    local/scratch|local/scratch/*|\
    local/run_scripts|local/run_scripts/*|\
    local/conflicts|local/conflicts/*|\
    local/tmp|local/tmp/*|\
    local/.cache|local/.cache/*)
      echo "ERROR: refusing private/runtime-only local tree: ${path}" >&2
      exit 2
      ;;
  esac
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is required to submit a broadcast dependency job." >&2
  exit 3
fi

# This helper is portable across all four servers and requests no GPU. Apply
# the smallest tracked per-GPU ceiling as a conservative job-total ceiling so
# an environment override cannot create an auto-cancelled allocation.
python3 "${repo_root}/scripts/slurm_memory_policy.py" request \
  --server server4 \
  --gpus 1 \
  --mem "${mem}" >/dev/null

if [[ ${#job_ids[@]} -eq 1 ]]; then
  canonical_broadcast_name="bc_${job_ids[0]}"
else
  canonical_broadcast_name="bc_$(IFS=_; echo "${job_ids[*]}")"
fi

if [[ "${broadcast_name}" != "artifact_broadcast" && "${broadcast_name}" != "${canonical_broadcast_name}" ]]; then
  echo "WARN ignoring custom broadcast name '${broadcast_name}'; using '${canonical_broadcast_name}'" >&2
fi
broadcast_name="${canonical_broadcast_name}"

safe_name="$(printf '%s' "${broadcast_name}" | tr -c '[:alnum:]_-' '_' | cut -c1-80)"
timestamp="$(date +%Y%m%dT%H%M%S)"
script_dir="${repo_root}/local/run_scripts"
log_dir="${repo_root}/local/logs/slurm/artifact-broadcast"
mkdir -p "${script_dir}" "${log_dir}"
script_path="${script_dir}/${timestamp}_${safe_name}.sh"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'repo_root=%q\n' "${repo_root}"
  echo 'cd "${repo_root}"'
  echo 'broadcast_args=(scripts/rsync-artifact-broadcast.sh)'
  for target in "${targets[@]}"; do
    printf 'broadcast_args+=(--target %q)\n' "${target}"
  done
  echo 'paths=('
  for path in "${paths[@]}"; do
    printf '  %q\n' "${path}"
  done
  echo ')'
  echo 'echo "artifact broadcast started at $(date --iso-8601=seconds)"'
  echo 'for path in "${paths[@]}"; do'
  echo '  if [[ ! -e "${path}" ]]; then'
  echo '    echo "WARN missing path, skip: ${path}" >&2'
  echo '    continue'
  echo '  fi'
  echo '  "${broadcast_args[@]}" "${path}"'
  echo 'done'
  echo 'echo "artifact broadcast finished at $(date --iso-8601=seconds)"'
} > "${script_path}"
chmod +x "${script_path}"

dependency="afterany:$(IFS=:; echo "${job_ids[*]}")"
sbatch_args=(
  --parsable
  --job-name="${safe_name}"
  --dependency="${dependency}"
  --cpus-per-task="${cpus}"
  --mem="${mem}"
  --time="${time_limit}"
  --chdir="${repo_root}"
  --output="${log_dir}/%j_%x.out"
  --error="${log_dir}/%j_%x.err"
)

if [[ "${dry_run}" == "1" ]]; then
  printf 'DRY RUN sbatch'
  printf ' %q' "${sbatch_args[@]}" "${script_path}"
  printf '\n'
  exit 0
fi

sbatch "${sbatch_args[@]}" "${script_path}"
