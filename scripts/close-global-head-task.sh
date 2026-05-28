#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/agent-policy.sh"

task_arg="${1:-}"
state="${2:-done}"
reason="${3:-}"
branch="${AGENT_BRANCH:-main}"

if [ -z "${task_arg}" ]; then
  echo "usage: scripts/close-global-head-task.sh <task_id|tasks/pending/file.yaml> [done|failed] [reason]" >&2
  exit 2
fi

case "${state}" in
  done|failed)
    ;;
  *)
    echo "state must be done or failed" >&2
    exit 2
    ;;
esac

agent_id="$(agent_policy_get_id)"
agent_role="$(agent_policy_get_role)"

agent_policy_require_config_match "" ""
agent_policy_require_global_head_role "${agent_role}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; commit or stash local changes first" >&2
  exit 3
fi

parse_field() {
  field="${1}"
  path="${2}"
  awk -F: -v field="${field}" '
    $1 == field {
      value = $2
      sub(/^[ \t"]+/, "", value)
      sub(/[ \t"]+$/, "", value)
      print value
      exit
    }
  ' "${path}"
}

parse_targets() {
  path="${1}"
  awk '
    $1 == "target_servers:" {
      in_targets = 1
      next
    }
    in_targets && /^[^[:space:]-]/ {
      exit
    }
    in_targets && /^[[:space:]]*-/ {
      sub(/^[[:space:]]*-[[:space:]]*/, "")
      gsub(/"/, "")
      print
    }
  ' "${path}"
}

read_status() {
  path="${1}"
  if [ ! -f "${path}" ]; then
    echo "missing"
    return
  fi
  sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${path}" | head -n 1
}

git pull --rebase origin "${branch}"

if [ -f "${task_arg}" ]; then
  task_path="${task_arg}"
else
  task_path="$(
    find tasks/pending -maxdepth 1 -type f \
      \( -name "${task_arg}.yaml" -o -name "${task_arg}.yml" -o -name "${task_arg}.json" \) \
      | sort \
      | head -n 1
  )"
fi

if [ -z "${task_path}" ]; then
  while IFS= read -r candidate_path; do
    candidate_id="$(parse_field id "${candidate_path}")"
    if [ "${candidate_id}" = "${task_arg}" ]; then
      task_path="${candidate_path}"
      break
    fi
  done < <(
    find tasks/pending -maxdepth 1 -type f \
      \( -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) \
      | sort
  )
fi

if [ -z "${task_path}" ] || [ ! -f "${task_path}" ]; then
  echo "pending task not found: ${task_arg}" >&2
  exit 4
fi

task_file="$(basename "${task_path}")"
task_id="$(parse_field id "${task_path}")"
task_id="${task_id:-${task_file%.*}}"
dest_dir="tasks/${state}"
dest_path="${dest_dir}/${task_file}"
closure_path="${dest_dir}/${task_id}.closure.md"
closed_at="$(date --iso-8601=seconds)"

if [ "${state}" = "done" ]; then
  mapfile -t target_servers < <(parse_targets "${task_path}")
  blocked=0
  for server in "${target_servers[@]}"; do
    status_path="tasks/status/${task_id}/${server}.json"
    server_status="$(read_status "${status_path}")"
    case "${server_status}" in
      done|waived)
        ;;
      *)
        echo "cannot close ${task_id}: ${server} status is ${server_status}; expected done or waived" >&2
        blocked=1
        ;;
    esac
  done
  if [ "${blocked}" -ne 0 ]; then
    echo "run scripts/audit-pending-tasks.sh and collect missing server status before closing" >&2
    exit 5
  fi
fi

mkdir -p "${dest_dir}"
git mv "${task_path}" "${dest_path}"

{
  echo "# Task Closure: ${task_id}"
  echo
  echo "- closed_at: ${closed_at}"
  echo "- closed_by: ${agent_id}"
  echo "- closure_state: ${state}"
  echo "- source_task: ${dest_path}"
  echo "- status_dir: tasks/status/${task_id}/"
  echo "- reason: ${reason:-not specified}"
  echo
  echo "## Server Status"
  echo
  if compgen -G "tasks/status/${task_id}/*.json" >/dev/null; then
    for status_file in tasks/status/"${task_id}"/*.json; do
      server="$(basename "${status_file}" .json)"
      status="$(sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${status_file}" | head -n 1)"
      updated_at="$(sed -n 's/.*"updated_at"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${status_file}" | head -n 1)"
      echo "- ${server}: ${status:-unknown} (${updated_at:-unknown time})"
    done
  else
    echo "- no status files found"
  fi
} > "${closure_path}"

git add "${closure_path}"
git commit -m "close command ${task_id} by ${agent_id}"
git push origin "${branch}"
