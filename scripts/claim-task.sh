#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/agent-policy.sh"

default_agent_id="$(agent_policy_get_id)"
requested_agent_id="${1:-}"
agent_id="${requested_agent_id:-${default_agent_id}}"
agent_role="$(agent_policy_get_role)"
branch="${AGENT_BRANCH:-main}"

agent_policy_require_config_match "${requested_agent_id}" ""
agent_policy_require_worker_role "${agent_role}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; commit or stash local changes first" >&2
  exit 3
fi

while true; do
  git pull --rebase origin "${branch}"

  task_path="$(
    find tasks/pending -maxdepth 1 -type f \
      \( -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) \
      | sort \
      | head -n 1
  )"

  if [ -z "${task_path}" ]; then
    echo "no pending task"
    exit 0
  fi

  task_file="$(basename "${task_path}")"
  task_ext="${task_file##*.}"
  task_id="${task_file%.*}"
  dest_path="tasks/running/${task_id}.${agent_id}.${task_ext}"

  git mv "${task_path}" "${dest_path}"
  git commit -m "claim ${task_id} by ${agent_id}"

  if git push origin "${branch}"; then
    echo "${dest_path}"
    exit 0
  fi

  echo "push failed; stop and inspect instead of rewriting local task state" >&2
  echo "report this claim failure to the server-head/global-head before retrying" >&2
  exit 20
done
