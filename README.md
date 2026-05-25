# Reflection-based-KE

Git-backed control plane for Reflection-based KE experiments.

This repository is used by the head agent and worker agents to exchange
plans, task metadata, status summaries, and small experiment records. Large
logs, checkpoints, datasets, and secrets stay outside Git and are referenced by
path.

## Roles

- `head-server2`: head agent on `server2`; talks to the user, writes plans,
  creates tasks, and reviews worker results.
- worker agents: pull tasks from Git, claim work under their own Git identity,
  execute jobs, and push status summaries.

## Layout

- `plans/`: human-readable experiment plans.
- `messages/`: dated messages from the head and workers.
- `tasks/pending/`: task specs waiting to be claimed.
- `tasks/running/`: claimed tasks, named with the claiming agent.
- `tasks/done/`: completed task specs.
- `tasks/failed/`: failed task specs.
- `runs/`: small summaries, metrics, log tails, and artifact path manifests.
- `workers/`: per-agent heartbeat/status files.
- `scripts/`: small helper scripts for heartbeat and task claiming.
- `docs/agent-protocol.md`: operating protocol for all agents.

## Sync Rule

Every agent must run `git pull --rebase` before editing and must push after each
state transition. A failed push means another agent changed the control plane
first; pull again, inspect the current task state, and retry only if the task is
still available.

## Worker Bootstrap

On each worker clone, configure a unique local identity:

```bash
git config user.name "agent-server3"
git config user.email "agent-server3@lab.local"
```

Then workers can publish heartbeats and claim one task:

```bash
scripts/heartbeat.sh agent-server3 worker
scripts/claim-task.sh agent-server3
```
