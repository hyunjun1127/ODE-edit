#!/usr/bin/env bash
set -euo pipefail

branch="${AGENT_BRANCH:-main}"
remote="${AGENT_REMOTE:-origin}"
agent_id="$(git config --get agent.id || git config user.name || true)"

if [ -z "${agent_id}" ]; then
  echo "agent identity is not configured; set git config agent.id" >&2
  exit 2
fi

lock_file=".git/agent-sync.lock"
exec 9>"${lock_file}"

if ! flock -n 9; then
  echo "sync already running for this clone"
  exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; skip scheduled sync for ${agent_id}" >&2
  exit 0
fi

git fetch "${remote}" "${branch}"
git rebase "${remote}/${branch}"

ahead_count="$(git rev-list --count "${remote}/${branch}..HEAD")"

if [ "${ahead_count}" = "0" ]; then
  echo "up to date; no push needed for ${agent_id}"
  exit 0
fi

git fetch "${remote}" "${branch}"
git rebase "${remote}/${branch}"
git push "${remote}" "${branch}"
