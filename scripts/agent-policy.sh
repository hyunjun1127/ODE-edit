#!/usr/bin/env bash

agent_policy_get_id() {
  git config --get agent.id || true
}

agent_policy_get_role() {
  git config --get agent.role || true
}

agent_policy_require_identity() {
  agent_id="${1:-}"
  agent_role="${2:-}"

  if [ -z "${agent_id}" ]; then
    echo "agent identity is not configured; set git config agent.id" >&2
    exit 2
  fi

  if [ -z "${agent_role}" ]; then
    echo "agent role is not configured; set git config agent.role" >&2
    exit 2
  fi
}

agent_policy_require_config_match() {
  requested_id="${1:-}"
  requested_role="${2:-}"
  configured_id="$(agent_policy_get_id)"
  configured_role="$(agent_policy_get_role)"

  agent_policy_require_identity "${configured_id}" "${configured_role}"

  if [ -n "${requested_id}" ] && [ "${requested_id}" != "${configured_id}" ]; then
    echo "requested agent_id ${requested_id} does not match configured agent.id ${configured_id}" >&2
    exit 2
  fi

  if [ -n "${requested_role}" ] && [ "${requested_role}" != "${configured_role}" ]; then
    echo "requested role ${requested_role} does not match configured agent.role ${configured_role}" >&2
    exit 2
  fi
}

agent_policy_require_known_role() {
  agent_role="${1:-}"

  case "${agent_role}" in
    global-head|server-head|worker|subagent|blue-team|red-team|blue-plan-runner|blue-experiment-runner|blue-result-analyst|red-data-eval-auditor|red-logic-evidence-auditor|red-git-protocol-auditor)
      ;;
    *)
      echo "unknown agent role: ${agent_role}" >&2
      exit 2
      ;;
  esac
}

agent_policy_require_git_writer_role() {
  agent_role="${1:-}"
  agent_policy_require_known_role "${agent_role}"

  case "${agent_role}" in
    global-head|server-head|worker)
      ;;
    *)
      echo "role ${agent_role} must not run Git-writing helper scripts directly; report to its parent agent" >&2
      exit 6
      ;;
  esac
}

agent_policy_require_worker_role() {
  agent_role="${1:-}"
  agent_policy_require_known_role "${agent_role}"

  if [ "${agent_role}" != "worker" ]; then
    echo "task claim/finish helper scripts are worker-only; current role is ${agent_role}" >&2
    exit 6
  fi
}
