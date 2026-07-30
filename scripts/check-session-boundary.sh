#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/check-session-boundary.sh SESSION_ID

Verifies that the caller is in this repository's configured clone and presents
the Codex session ID assigned to that clone. The local configuration is
servers/local/session-boundary.env and is never committed.
USAGE
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
presented_session_id="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
config_file="${repo_root}/servers/local/session-boundary.env"

[[ -f "${config_file}" ]] || { echo "BLOCK missing local session boundary config: ${config_file}" >&2; exit 4; }
# shellcheck disable=SC1090
source "${config_file}"

[[ -n "${ODEEDIT_REPOSITORY_ID:-}" && -n "${ODEEDIT_REPOSITORY_CWD:-}" && -n "${ODEEDIT_CODEX_SESSION_ID:-}" && -n "${ODEEDIT_CURRENT_SERVER:-}" ]] || {
  echo "BLOCK incomplete local session boundary configuration" >&2
  exit 4
}

expected_cwd="$(realpath -m "${ODEEDIT_REPOSITORY_CWD}")"
[[ "${repo_root}" == "${expected_cwd}" ]] || {
  echo "BLOCK repository CWD mismatch: expected=${expected_cwd} actual=${repo_root}" >&2
  exit 4
}
[[ "${presented_session_id}" == "${ODEEDIT_CODEX_SESSION_ID}" ]] || {
  echo "BLOCK Codex session ID mismatch for server=${ODEEDIT_CURRENT_SERVER}" >&2
  exit 4
}

origin_url="$(git -C "${repo_root}" remote get-url origin 2>/dev/null || true)"
expected_https="https://github.com/${ODEEDIT_REPOSITORY_ID}.git"
expected_ssh="git@github.com:${ODEEDIT_REPOSITORY_ID}.git"
[[ "${origin_url}" == "${expected_https}" || "${origin_url}" == "${expected_ssh}" ]] || {
  echo "BLOCK origin mismatch for repository=${ODEEDIT_REPOSITORY_ID}" >&2
  exit 4
}

echo "PASS repository=${ODEEDIT_REPOSITORY_ID} server=${ODEEDIT_CURRENT_SERVER} session=${presented_session_id}"
