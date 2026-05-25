# Reflection-based-KE

Reflection-based knowledge editing experiments and coordination records.

## Layout

- `plans/`: experiment plans and decisions.
- `tasks/pending/`: task specs waiting to be claimed.
- `tasks/running/`: claimed tasks, named with the claiming agent.
- `tasks/done/`: completed task specs.
- `tasks/failed/`: failed task specs.
- `runs/`: summaries, metrics, log tails, and artifact path manifests.
- `messages/`: dated head/worker messages.
- `workers/`: per-agent status files.
- `scripts/`: helper scripts for heartbeat, task claim, and task finish.

Agent coordination rules are defined in [PROTOCOL.md](PROTOCOL.md).

## Artifact Rule

Keep large logs, checkpoints, model weights, datasets, and credentials outside
Git. Track only small metadata and paths to external artifacts.
