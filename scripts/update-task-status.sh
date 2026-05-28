#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/agent-policy.sh"

task_id="${1:-}"
status="${2:-}"
note="${3:-}"
branch="${AGENT_BRANCH:-main}"

if [ -z "${task_id}" ] || [ -z "${status}" ]; then
  echo "usage: scripts/update-task-status.sh <task_id> <acknowledged|running|done|failed|blocked|manual_required|waived> [note]" >&2
  exit 2
fi

case "${status}" in
  acknowledged|running|done|failed|blocked|manual_required|waived)
    ;;
  *)
    echo "unknown task status: ${status}" >&2
    exit 2
    ;;
esac

agent_id="$(agent_policy_get_id)"
agent_role="$(agent_policy_get_role)"
agent_hostname="$(git config --get agent.hostname || hostname)"

agent_policy_require_config_match "" ""
agent_policy_require_task_status_writer_role "${agent_role}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; commit or stash local changes first" >&2
  exit 3
fi

json_escape() {
  printf '%s' "${1}" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

git pull --rebase origin "${branch}"

today="$(date +%F)"
updated_at="$(date --iso-8601=seconds)"
status_dir="tasks/status/${task_id}"
status_path="${status_dir}/${agent_hostname}.json"
mkdir -p "${status_dir}"

cat > "${status_path}" <<EOF
{
  "task_id": "$(json_escape "${task_id}")",
  "server": "$(json_escape "${agent_hostname}")",
  "agent_id": "$(json_escape "${agent_id}")",
  "agent_role": "$(json_escape "${agent_role}")",
  "status": "$(json_escape "${status}")",
  "updated_at": "$(json_escape "${updated_at}")",
  "ack_path": "messages/acks/$(json_escape "${agent_hostname}")/${today}.md",
  "report_dir": "messages/server-heads/$(json_escape "${agent_hostname}")/",
  "note": "$(json_escape "${note}")"
}
EOF

git add "${status_path}"
git commit -m "update task status ${task_id} ${agent_hostname} ${status} by ${agent_id}"
git push origin "${branch}"
