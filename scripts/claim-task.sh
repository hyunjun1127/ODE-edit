#!/usr/bin/env bash
set -euo pipefail

default_agent_id="$(git config --get agent.id || git config user.name || true)"
agent_id="${1:-${default_agent_id}}"
branch="${AGENT_BRANCH:-main}"

if [ -z "${agent_id}" ]; then
  echo "usage: scripts/claim-task.sh <agent_id>" >&2
  exit 2
fi

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

  echo "push failed; another agent may have claimed first, resyncing" >&2
  git rebase --abort >/dev/null 2>&1 || true
  git fetch origin "${branch}"
  git reset --hard "origin/${branch}"
done
