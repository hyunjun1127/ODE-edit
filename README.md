# Agent Control Template

Reusable Git-backed coordination template for multi-server agents, task
handoffs, experiment plans, and run summaries.

## Layout

- `plans/global/`: canonical plans owned by the configured global head.
- `plans/updates/<server>/`: plan updates from each server head.
- `tasks/pending/`: task specs waiting to be claimed.
- `tasks/proposed/<server>/`: task proposals from server heads.
- `tasks/running/`: claimed tasks, named with the claiming agent.
- `tasks/done/`: completed task specs.
- `tasks/failed/`: failed task specs.
- `run-scripts/`: repository-managed experiment execution scripts.
- `subagents/blue/`: required blue-team subagent role specs.
- `subagents/red/`: required red-team subagent role specs.
- `audits/servers/<server>/`: Korean red-team audit reports.
- `audits/templates/`: Korean audit templates.
- `servers/`: server onboarding/offboarding records and templates.
- `transfers/`: user-approved large file transfer requests, approvals, and
  verification records.
- `runs/`: machine-readable run status, metrics, log tails, and artifact path
  manifests.
- `experiment-reports/global/`: Korean final experiment summaries.
- `experiment-reports/servers/<server>/`: Korean server-specific experiment
  summaries.
- `experiment-reports/templates/`: Korean experiment report templates.
- `messages/head/`: global-head communication messages.
- `messages/server-heads/<server>/`: cross-server updates and requests from
  each server head.
- `messages/templates/`: message templates.
- `agents/<server>/<agent>.json`: per-agent status files.
- `control/`: repository-wide control markers such as sync pause signals.
- `scripts/`: helper scripts for heartbeat, task claim, and task finish.

Agent coordination rules are defined in [PROTOCOL.md](PROTOCOL.md).

## Artifact Rule

Keep large logs, checkpoints, model weights, datasets, and credentials outside
Git. Track only small metadata and paths to external artifacts.
