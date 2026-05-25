# Agent Protocol

## Current Head

- Server hostname: `server2`
- Head agent ID: `head-server2`
- Repository path: `/mnt/raid5/janghj/agent-control/Reflection-based-KE`

## Purpose

Git is the durable control plane for experiment coordination. It is not a
message queue for high-frequency events and it is not storage for large
artifacts.

## Git Identity

Each agent must configure a repository-local identity before writing commits.

```bash
git config user.name "agent-serverN"
git config user.email "agent-serverN@lab.local"
```

The head agent on this server uses:

```bash
git config user.name "head-server2"
git config user.email "head-server2@lab.local"
```

Worker examples:

```bash
git config user.name "agent-server3"
git config user.email "agent-server3@lab.local"
```

## Task Lifecycle

1. Head creates a task file in `tasks/pending/`.
2. Worker runs `git pull --rebase`.
3. Worker claims the task by moving it to `tasks/running/<task_id>.<agent>.yaml`.
4. Worker commits with `claim <task_id> by <agent>`.
5. Worker pushes.
6. If push fails, worker pulls/rebases and checks whether the task is still in
   `tasks/pending/`.
7. Worker executes the task.
8. Worker moves the task to `tasks/done/` or `tasks/failed/`, writes a small
   run summary, commits, and pushes.

The helper script `scripts/claim-task.sh <agent_id>` implements steps 2-6 for
the simple first-pending-task case.

## Commit Messages

Use short, machine-readable commit messages:

```text
plan exp_27011 by head-server2
create task exp_27011 by head-server2
claim exp_27011 by agent-server3
heartbeat agent-server3
finish exp_27011 by agent-server3 exit=0
fail exp_27011 by agent-server3 exit=1
```

## File Ownership

Avoid shared hot files. Prefer one file per worker or one file per task.

Good:

```text
workers/server3/status.json
runs/exp_27011/summary.agent-server3.md
```

Avoid:

```text
workers/status.json
runs/all_status.json
```

## Artifact Rule

Git may store:

- plans
- task YAML
- small JSON metrics
- short log tails
- artifact path manifests

Git must not store:

- SSH keys
- API tokens
- full stdout/stderr logs
- checkpoints
- model weights
- datasets

Store large files on shared storage or local server storage, then reference the
path from `runs/<task_id>/artifact_paths.json`.

## Head Workflow

The head agent should keep user-facing reasoning in `plans/` and executable
instructions in `tasks/pending/`. A message in `messages/head/` can describe
intent, but workers should execute only validated task files.

## Worker Bootstrap

On each worker clone:

```bash
git clone https://github.com/hyunjun1127/Reflection-based-KE.git ~/agent-control/Reflection-based-KE
cd ~/agent-control/Reflection-based-KE
git config user.name "agent-server3"
git config user.email "agent-server3@lab.local"
scripts/heartbeat.sh agent-server3 worker
scripts/claim-task.sh agent-server3
```
