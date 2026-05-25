#!/usr/bin/env bash
set -euo pipefail

default_agent_id="$(git config --get agent.id || git config user.name || true)"
default_role="$(git config --get agent.role || true)"
agent_id="${1:-${default_agent_id}}"
role="${2:-${default_role:-worker}}"
branch="${AGENT_BRANCH:-main}"
agent_hostname="$(git config --get agent.hostname || hostname)"

if [ -z "${agent_id}" ]; then
  echo "usage: scripts/heartbeat.sh <agent_id> [role]" >&2
  exit 2
fi

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
