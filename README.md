# Agent Control Template

Reusable Git-backed coordination template for multi-server agents, task
handoffs, experiment plans, and run summaries.

## Layout

Detailed structure and cleanup policy are in
`docs/repo-structure-and-dispatch.md`. The short version is:

- `project/run_scripts/`: repository-managed experiment execution scripts.
  Root `run-scripts/` is deprecated.
- `local/`: ignored server-local raw artifacts, Slurm logs, run logs,
  checkpoints, datasets, transfers, and secrets. Share ordinary project
  artifacts by rsync broadcast, not Git.
- `messages/`: human-readable Korean coordination. `messages/inbox/<server>.md`
  is the global-head-owned durable instruction inbox for a target server;
  `messages/README.md` is the cross-server index; all active server-heads read
  it and write their own ack.
- `tasks/`: durable state only for approved work that needs per-server status,
  worker ownership, or global-head closure. Do not create a task for every
  message. A task file or inbox message is not executed unless a live agent or
  explicit automation reads and acts on it.
- `runs/`: small machine-readable run status, log tails, metrics, and artifact
  manifests. Full logs stay under `local/`.
- `experiment-reports/`: Korean experiment summaries and interpretation.
- `audits/`: Korean red-team/protocol/research audits.
- `agents/`: agent heartbeat, sync, and subagent status files.
- `servers/`: active/retired server records and redacted SSH/rsync inventory.
  Raw host/IP/user/port remains in ignored `servers/local/`.
- `scripts/`: repo coordination helpers, including heartbeat, sync, task
  audit/status/closure, worker claim/finish, and artifact broadcast.
- `subagents/`: blue/red role specifications.
- `transfers/`: manual-exception transfer records. Ordinary experiment
  artifact broadcast uses `scripts/rsync-artifact-broadcast.sh` and does not
  need per-transfer user approval.

Agent coordination rules are defined in [PROTOCOL.md](PROTOCOL.md).

## Research Handoff

When this template is used to start the superseded-prior reflective KE project,
the initial proposal for the new global-head is in
`project/proposals/superseded-prior-reflective-ke-proposal.md`. Treat it as the
starting research handoff, not as a finalized paper plan.

## Artifact Rule

Keep large logs, checkpoints, model weights, datasets, and credentials outside
Git. Track only small metadata and paths to external artifacts.
