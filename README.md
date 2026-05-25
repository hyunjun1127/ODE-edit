# Reflection-based-KE

Reflection-based knowledge editing experiments and coordination records.

## Layout

- `plans/global/`: canonical plans owned by the final head on `server2`.
- `plans/updates/<server>/`: plan updates from each server head.
- `tasks/pending/`: task specs waiting to be claimed.
- `tasks/proposed/<server>/`: task proposals from server heads.
- `tasks/running/`: claimed tasks, named with the claiming agent.
- `tasks/done/`: completed task specs.
- `tasks/failed/`: failed task specs.
- `runs/`: summaries, metrics, log tails, and artifact path manifests.
- `messages/head/`: global-head announcements.
- `messages/server-heads/<server>/`: cross-server updates and requests from
  each server head.
- `messages/templates/`: message templates.
- `agents/<server>/<agent>.json`: per-agent status files.
- `scripts/`: helper scripts for heartbeat, task claim, and task finish.

Agent coordination rules are defined in [PROTOCOL.md](PROTOCOL.md).

## Artifact Rule

Keep large logs, checkpoints, model weights, datasets, and credentials outside
Git. Track only small metadata and paths to external artifacts.
