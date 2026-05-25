#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/agent-policy.sh"

default_agent_id="$(agent_policy_get_id)"
default_role="$(agent_policy_get_role)"
requested_agent_id="${1:-}"
requested_role="${2:-}"
agent_id="${requested_agent_id:-${default_agent_id}}"
role="${requested_role:-${default_role}}"
branch="${AGENT_BRANCH:-main}"
agent_hostname="$(git config --get agent.hostname || hostname)"

agent_policy_require_config_match "${requested_agent_id}" "${requested_role}"
agent_policy_require_git_writer_role "${role}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; commit or stash local changes first" >&2
  exit 3
fi

git pull --rebase origin "${branch}"

status_dir="agents/${agent_hostname}"
status_file="${status_dir}/${agent_id}.json"
mkdir -p "${status_dir}"

cat > "${status_file}" <<EOF
{
  "agent_id": "${agent_id}",
  "role": "${role}",
  "hostname": "${agent_hostname}",
  "status": "online",
  "last_seen": "$(date --iso-8601=seconds)",
  "repo_path": "$(pwd)"
}
EOF

git add "${status_file}"

if git diff --cached --quiet; then
  echo "heartbeat unchanged"
  exit 0
fi

git commit -m "heartbeat ${agent_id}"
git push origin "${branch}"
