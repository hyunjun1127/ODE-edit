#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
. "${script_dir}/agent-policy.sh"

unit_name="${SYNC_TIMER_UNIT:-agent-control-sync}"
default_startup_delay() {
  hostname_hint="$(agent_policy_get_hostname)"

  case "${hostname_hint}" in
    server1)
      printf '1min'
      ;;
    server2)
      printf '3min'
      ;;
    server3)
      printf '7min'
      ;;
    *)
      printf '9min'
      ;;
  esac
}

interval="${SYNC_TIMER_INTERVAL:-10min}"
startup_delay="${SYNC_TIMER_STARTUP_DELAY:-$(default_startup_delay)}"
accuracy="${SYNC_TIMER_ACCURACY:-30s}"
systemd_user_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
service_file="${systemd_user_dir}/${unit_name}.service"
timer_file="${systemd_user_dir}/${unit_name}.timer"
agent_id="$(agent_policy_get_id)"
agent_role="$(agent_policy_get_role)"

agent_policy_require_identity "${agent_id}" "${agent_role}"
agent_policy_require_known_role "${agent_role}"

case "${agent_role}" in
  global-head|server-head)
    ;;
  *)
    echo "sync timer installation is only for global-head or server-head; current role is ${agent_role}" >&2
    exit 6
    ;;
esac

mkdir -p "${systemd_user_dir}"

{
  echo "[Unit]"
  echo "Description=Agent control Git sync for ${agent_id}"
  echo "Documentation=file://${repo_root}/PROTOCOL.md"
  echo
  echo "[Service]"
  echo "Type=oneshot"
  echo "WorkingDirectory=${repo_root}"
  echo "ExecStart=/usr/bin/env bash ${repo_root}/scripts/sync-agent.sh"
} > "${service_file}"

{
  echo "[Unit]"
  echo "Description=Run Agent control Git sync for ${agent_id}"
  echo
  echo "[Timer]"
  echo "OnStartupSec=${startup_delay}"
  echo "OnUnitActiveSec=${interval}"
  echo "AccuracySec=${accuracy}"
  echo "Persistent=true"
  echo
  echo "[Install]"
  echo "WantedBy=timers.target"
} > "${timer_file}"

systemctl --user daemon-reload
systemctl --user enable --now "${unit_name}.timer"

echo "installed ${service_file}"
echo "installed ${timer_file}"
echo "enabled ${unit_name}.timer with interval ${interval}, startup delay ${startup_delay}, accuracy ${accuracy}"
