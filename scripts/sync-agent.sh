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
sync_started_at="$(date --iso-8601=seconds)"

agent_policy_require_identity "${agent_id}" "${agent_role}"
agent_policy_require_git_writer_role "${agent_role}"

json_bool() {
  case "${1:-false}" in
    true|1|yes)
      printf 'true'
      ;;
    *)
      printf 'false'
      ;;
  esac
}

update_readme_server_progress() {
  readme_file="README.md"
  timestamp_kst="$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M')"

  if [ ! -f "${readme_file}" ]; then
    return 0
  fi

  tmp_file="$(mktemp)"
  if ! awk -v server="${agent_hostname}" -v timestamp="${timestamp_kst}" -v agent="${agent_id}" '
    BEGIN {
      target = "| `" server "` |"
    }
    index($0, target) == 1 {
      field_count = split($0, fields, "|")
      if (field_count >= 8) {
        fields[3] = " " timestamp " "
        fields[4] = " `" agent "` "
        line = fields[1]
        for (i = 2; i <= field_count; i++) {
          line = line "|" fields[i]
        }
        print line
        next
      }
    }
    { print }
  ' "${readme_file}" > "${tmp_file}"; then
    rm -f "${tmp_file}"
    return 1
  fi

  if cmp -s "${readme_file}" "${tmp_file}"; then
    rm -f "${tmp_file}"
  else
    mv "${tmp_file}" "${readme_file}"
  fi
}

write_sync_status() {
  status="${1:-success}"
  push_plan="${2:-pending}"
  pulled_updates="${3:-false}"
  unpushed_commits="${4:-0}"
  head_before_fetch="${5:-unknown}"
  head_after_rebase="${6:-unknown}"
  upstream_before_fetch="${7:-unknown}"
  upstream_after_fetch="${8:-unknown}"
  timestamp="$(date --iso-8601=seconds)"
  status_dir="agents/${agent_hostname}"
  status_file="${status_dir}/${agent_id}.sync.json"

  mkdir -p "${status_dir}"
  cat > "${status_file}" <<EOF
{
  "agent_id": "${agent_id}",
  "role": "${agent_role}",
  "hostname": "${agent_hostname}",
  "status": "${status}",
  "sync_started_at": "${sync_started_at}",
  "sync_report_written_at": "${timestamp}",
  "remote": "${remote}",
  "branch": "${branch}",
  "head_before_fetch": "${head_before_fetch}",
  "head_after_rebase": "${head_after_rebase}",
  "upstream_before_fetch": "${upstream_before_fetch}",
  "upstream_after_fetch": "${upstream_after_fetch}",
  "pulled_updates": $(json_bool "${pulled_updates}"),
  "unpushed_commits_before_report": ${unpushed_commits},
  "push_plan": "${push_plan}",
  "repo_path": "$(pwd)",
  "remote_visibility_note": "If this file is visible on origin/main, this agent completed fetch/rebase and pushed this sync report."
}
EOF

  update_readme_server_progress

  git add "${status_file}"
  git add README.md
  if ! git diff --cached --quiet -- "${status_file}"; then
    git commit -m "sync status ${agent_id}"
  elif ! git diff --cached --quiet -- README.md; then
    git commit -m "sync progress ${agent_id}"
  fi
}

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

head_before_fetch="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
upstream_before_fetch="$(git rev-parse "${remote}/${branch}" 2>/dev/null || echo unknown)"

git fetch "${remote}" "${branch}"
run_rebase_or_pause
head_after_rebase="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
upstream_after_fetch="$(git rev-parse "${remote}/${branch}" 2>/dev/null || echo unknown)"
pulled_updates=false
if [ "${head_before_fetch}" != "${head_after_rebase}" ]; then
  pulled_updates=true
fi

if [ -f "${repo_pause_file}" ]; then
  echo "repository-wide sync pause marker exists at ${repo_pause_file}; skip sync"
  exit 0
fi

ahead_count="$(git rev-list --count "${remote}/${branch}..HEAD")"
write_sync_status "success" "push_report_to_remote" "${pulled_updates}" "${ahead_count}" "${head_before_fetch}" "${head_after_rebase}" "${upstream_before_fetch}" "${upstream_after_fetch}"
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
