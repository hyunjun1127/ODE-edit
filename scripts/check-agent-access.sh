#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/agent-policy.sh"

mode="${1:---staged}"
agent_id="$(agent_policy_get_id)"
agent_role="$(agent_policy_get_role)"
agent_hostname="$(git config --get agent.hostname || hostname)"

agent_policy_require_identity "${agent_id}" "${agent_role}"
agent_policy_require_known_role "${agent_role}"

case "${mode}" in
  --staged)
    mapfile -t paths < <(git diff --cached --name-only)
    ;;
  --all-changed)
    mapfile -t paths < <(
      git diff --name-only
      git diff --cached --name-only
      git ls-files --others --exclude-standard
    )
    ;;
  *)
    echo "usage: scripts/check-agent-access.sh [--staged|--all-changed]" >&2
    exit 2
    ;;
esac

if [ "${#paths[@]}" -eq 0 ]; then
  echo "no paths to check"
  exit 0
fi

is_global_head_allowed() {
  path="${1}"

  case "${path}" in
    PROTOCOL.md|README.md|.gitignore|local/README.md)
      return 0
      ;;
    scripts/*|project/*|subagents/*|docs/*|local/templates/*|messages/templates/*|tasks/templates/*|runs/templates/*|audits/templates/*|experiment-reports/templates/*|transfers/templates/*|servers/templates/*|run-scripts/*)
      return 0
      ;;
    plans/global/*|tasks/pending/*|tasks/done/*|tasks/failed/*|tasks/status/.gitkeep|tasks/status/*/*|messages/README.md|messages/head/*|messages/inbox/.gitkeep|messages/inbox/*|messages/server-heads/*|messages/acks/.gitkeep|messages/acks/"${agent_hostname}"/*|transfers/approvals/*|transfers/verifications/*|servers/connection-inventory.md|servers/active/*|servers/retired/*|control/*|experiment-reports/global/*|audits/*)
      return 0
      ;;
    agents/"${agent_hostname}"/*.json)
      return 0
      ;;
  esac

  return 1
}

is_server_head_allowed() {
  path="${1}"

  case "${path}" in
    README.md)
      return 0
      ;;
    agents/"${agent_hostname}"/*|plans/updates/"${agent_hostname}"/*|tasks/proposed/"${agent_hostname}"/*|tasks/status/*/"${agent_hostname}".json|messages/server-heads/"${agent_hostname}"/*|messages/acks/"${agent_hostname}"/*|audits/servers/"${agent_hostname}"/*|experiment-reports/servers/"${agent_hostname}"/*)
      return 0
      ;;
    transfers/requests/*|transfers/verifications/*)
      return 0
      ;;
    servers/active/"${agent_hostname}".md)
      return 0
      ;;
  esac

  return 1
}

is_worker_allowed() {
  path="${1}"

  case "${path}" in
    agents/"${agent_hostname}"/"${agent_id}".json)
      return 0
      ;;
    tasks/running/*."${agent_id}".yaml|tasks/running/*."${agent_id}".yml|tasks/running/*."${agent_id}".json)
      return 0
      ;;
    tasks/done/*."${agent_id}".yaml|tasks/done/*."${agent_id}".yml|tasks/done/*."${agent_id}".json)
      return 0
      ;;
    tasks/failed/*."${agent_id}".yaml|tasks/failed/*."${agent_id}".yml|tasks/failed/*."${agent_id}".json)
      return 0
      ;;
    runs/*/*."${agent_id}".json|runs/*/*."${agent_id}".txt|runs/*/*."${agent_id}".md)
      return 0
      ;;
    audits/servers/"${agent_hostname}"/*|experiment-reports/servers/"${agent_hostname}"/*|transfers/verifications/*)
      return 0
      ;;
  esac

  return 1
}

is_allowed() {
  path="${1}"

  case "${agent_role}" in
    global-head)
      is_global_head_allowed "${path}"
      ;;
    server-head)
      is_server_head_allowed "${path}"
      ;;
    worker)
      is_worker_allowed "${path}"
      ;;
    *)
      return 1
      ;;
  esac
}

blocked=0

for path in "${paths[@]}"; do
  if ! is_allowed "${path}"; then
    echo "blocked by agent access policy: role=${agent_role} agent=${agent_id} host=${agent_hostname} path=${path}" >&2
    blocked=1
  fi
done

if [ "${blocked}" -ne 0 ]; then
  echo "access check failed; ask the parent agent or global-head to own these paths" >&2
  exit 7
fi

echo "agent access check passed for ${agent_role}/${agent_id}"
