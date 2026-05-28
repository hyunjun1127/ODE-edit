#!/usr/bin/env bash
set -euo pipefail

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

latest_ack() {
  server="${1}"
  if [ ! -d "messages/acks/${server}" ]; then
    return 0
  fi
  find "messages/acks/${server}" -maxdepth 1 -type f -name '*.md' 2>/dev/null | sort | tail -n 1
}

matching_report() {
  server="${1}"
  task_id="${2}"
  if [ ! -d "messages/server-heads/${server}" ]; then
    return 0
  fi
  find "messages/server-heads/${server}" -maxdepth 1 -type f -name "*${task_id}*.md" 2>/dev/null | sort | tail -n 1
}

shopt -s nullglob
pending_tasks=(tasks/pending/*.yaml tasks/pending/*.yml tasks/pending/*.json)

if [ "${#pending_tasks[@]}" -eq 0 ]; then
  echo "No pending tasks."
  exit 0
fi

echo "| Task | Type | Server | Status | Status File | Latest Ack | Matching Report |"
echo "| --- | --- | --- | --- | --- | --- | --- |"

for task_path in "${pending_tasks[@]}"; do
  task_file="$(basename "${task_path}")"
  fallback_id="${task_file%.*}"
  task_id="$(parse_field id "${task_path}")"
  task_type="$(parse_field type "${task_path}")"
  task_id="${task_id:-${fallback_id}}"
  task_type="${task_type:-unknown}"

  mapfile -t targets < <(parse_targets "${task_path}")
  if [ "${#targets[@]}" -eq 0 ]; then
    targets=("-")
  fi

  for server in "${targets[@]}"; do
    status_path="tasks/status/${task_id}/${server}.json"
    status="$(read_status "${status_path}")"
    ack_path="$(latest_ack "${server}")"
    report_path="$(matching_report "${server}" "${task_id}")"
    echo "| \`${task_id}\` | \`${task_type}\` | \`${server}\` | \`${status:-missing}\` | ${status_path} | ${ack_path:-missing} | ${report_path:-missing} |"
  done
done
