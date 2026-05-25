#!/usr/bin/env bash
set -euo pipefail

task_id="${1:-}"
state="${2:-}"
exit_code="${3:-}"
agent_id="${4:-$(git config user.name || true)}"
branch="${AGENT_BRANCH:-main}"

if [ -z "${task_id}" ] || [ -z "${state}" ] || [ -z "${exit_code}" ] || [ -z "${agent_id}" ]; then
  echo "usage: scripts/finish-task.sh <task_id> <done|failed> <exit_code> [agent_id]" >&2
  exit 2
fi

if [ "${state}" != "done" ] && [ "${state}" != "failed" ]; then
  echo "state must be done or failed" >&2
  exit 2
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; commit or stash local changes first" >&2
  exit 3
fi

git pull --rebase origin "${branch}"

running_path="$(
  find tasks/running -maxdepth 1 -type f \
    \( -name "${task_id}.${agent_id}.yaml" -o -name "${task_id}.${agent_id}.yml" -o -name "${task_id}.${agent_id}.json" \) \
    | sort \
    | head -n 1
)"

if [ -z "${running_path}" ]; then
  echo "no running task for ${task_id} owned by ${agent_id}" >&2
  exit 4
fi

mkdir -p "runs/${task_id}"
dest_path="tasks/${state}/$(basename "${running_path}")"

git mv "${running_path}" "${dest_path}"

cat > "runs/${task_id}/status.${agent_id}.json" <<EOF
{
  "task_id": "${task_id}",
  "agent_id": "${agent_id}",
  "state": "${state}",
  "exit_code": ${exit_code},
  "finished_at": "$(date --iso-8601=seconds)",
  "artifact_paths": "runs/${task_id}/artifact_paths.${agent_id}.json"
}
EOF

git add "runs/${task_id}/status.${agent_id}.json"
git commit -m "${state/failed/fail} ${task_id} by ${agent_id} exit=${exit_code}"
git push origin "${branch}"
