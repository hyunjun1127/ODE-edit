#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/agent-policy.sh"

branch="${AGENT_BRANCH:-main}"
remote="${AGENT_REMOTE:-origin}"
agent_id="$(agent_policy_get_id)"
agent_role="$(agent_policy_get_role)"
agent_hostname="$(git config --get agent.hostname || hostname)"
pause_file=".git/agent-sync.paused"
repo_pause_file="control/sync-paused"

agent_policy_require_identity "${agent_id}" "${agent_role}"
agent_policy_require_git_writer_role "${agent_role}"

write_conflict_report() {
  reason="${1:-unknown sync failure}"
  timestamp="$(date --iso-8601=seconds)"
  safe_timestamp="$(date +%Y%m%dT%H%M%S%z)"
  report_dir="local/conflicts"
  report_file="${report_dir}/${safe_timestamp}.${agent_id}.md"

  mkdir -p "${report_dir}"
  {
    echo "# Git Sync 충돌 보고"
    echo
    echo "- 시간: ${timestamp}"
    echo "- 보고 agent: ${agent_id}"
    echo "- 서버: ${agent_hostname}"
    echo "- branch: ${branch}"
    echo "- remote: ${remote}"
    echo "- 원인: ${reason}"
    echo "- HEAD: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "- Upstream: $(git rev-parse "${remote}/${branch}" 2>/dev/null || echo unknown)"
    echo
    echo "## Git 상태"
    echo
    git status --short --branch 2>&1 || true
    echo
    echo "## 다음 행동"
    echo
    echo "이 clone의 자동 sync를 멈추고 이 파일을 server-head/global-head에게 보고한다."
  } > "${report_file}"

  printf '%s\n' "${reason}" > "${pause_file}"
  echo "sync paused for ${agent_id}; conflict report written to ${report_file}" >&2
}

run_rebase_or_pause() {
  if ! git rebase "${remote}/${branch}"; then
    write_conflict_report "rebase failed against ${remote}/${branch}"
    exit 20
  fi
}

lock_file=".git/agent-sync.lock"
exec 9>"${lock_file}"

if ! flock -n 9; then
  echo "sync already running for this clone"
  exit 0
fi

if [ -f "${pause_file}" ]; then
  echo "sync is locally paused for ${agent_id}; inspect ${pause_file}" >&2
  exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; skip scheduled sync for ${agent_id}" >&2
  exit 0
fi

git fetch "${remote}" "${branch}"
run_rebase_or_pause

if [ -f "${repo_pause_file}" ]; then
  echo "repository-wide sync pause marker exists at ${repo_pause_file}; skip sync"
  exit 0
fi

ahead_count="$(git rev-list --count "${remote}/${branch}..HEAD")"

if [ "${ahead_count}" = "0" ]; then
  echo "up to date; no push needed for ${agent_id}"
  exit 0
fi

git fetch "${remote}" "${branch}"
run_rebase_or_pause

if ! git push "${remote}" "${branch}"; then
  write_conflict_report "push failed after rebase"
  exit 21
fi
