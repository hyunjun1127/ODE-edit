# Agent Protocol

## Current Global Head

- Final head server: `server2`
- Global head agent ID: `head-server2`
- Repository path: `/mnt/raid5/janghj/agent-control/Reflection-based-KE`

## Purpose

Git is the durable control plane for experiment coordination. It stores plans,
task metadata, status summaries, and small run records. It is not a real-time
message queue and it is not storage for large artifacts.

## Roles

- `global-head`: final coordinator. `head-server2` owns canonical plans,
  resolves conflicts between server heads, creates approved tasks, and talks to
  the user.
- `server-head`: per-server coordinator. A `head-serverN` agent can update its
  server plan notes, propose tasks, inspect local resources, and coordinate
  local workers/subagents.
- `worker`: execution agent. Claims approved tasks, runs jobs, monitors logs,
  and reports results.
- `subagent`: local helper owned by a server head or worker. Subagents should
  not push to Git directly; their parent agent summarizes and commits results.

## Git Identity

Each agent must configure repository-local identity before writing commits.
The `user.*` values are used by Git history, and the `agent.*` values are used
by helper scripts when no explicit agent ID is passed.

Global head on `server2`:

```bash
git config user.name "head-server2"
git config user.email "head-server2@lab.local"
git config agent.id "head-server2"
git config agent.role "global-head"
git config agent.hostname "server2"
```

Server head example:

```bash
git config user.name "head-server3"
git config user.email "head-server3@lab.local"
git config agent.id "head-server3"
git config agent.role "server-head"
git config agent.hostname "server3"
```

Worker example:

```bash
git config user.name "agent-server3"
git config user.email "agent-server3@lab.local"
git config agent.id "agent-server3"
git config agent.role "worker"
git config agent.hostname "server3"
```

## Plan Ownership

Plan updates are allowed from every server head, but avoid shared hot files.

- `plans/global/`: canonical plans. Owned by `head-server2`.
- `plans/updates/<server>/`: server-specific plan updates. Owned by that
  server's `server-head`.
- `tasks/proposed/<server>/`: task proposals from server heads.
- `tasks/pending/`: approved executable tasks. Owned by the global head unless
  explicitly delegated.

Server heads should write updates as append-only notes or separate files, then
the global head can promote accepted changes into `plans/global/` and
`tasks/pending/`.

## Task Lifecycle

1. A server head writes plan updates under `plans/updates/<server>/` or task
   proposals under `tasks/proposed/<server>/`.
2. The global head reviews updates and creates approved task files in
   `tasks/pending/`.
3. Worker runs `git pull --rebase`.
4. Worker claims the task by moving it to
   `tasks/running/<task_id>.<agent>.yaml`.
5. Worker commits with `claim <task_id> by <agent>`.
6. Worker pushes.
7. If push fails, worker pulls/rebases and checks whether the task is still in
   `tasks/pending/`.
8. Worker executes the task.
9. Worker moves the task to `tasks/done/` or `tasks/failed/`, writes a small
   run summary, commits, and pushes.

The helper script `scripts/claim-task.sh <agent_id>` implements the claim/push
part for the simple first-pending-task case.

## Periodic Sync

Every server head should sync every 10 minutes. The sync rule is:

1. do nothing if the working tree has uncommitted changes
2. fetch and rebase onto `origin/main`
3. push only if local commits are ahead of `origin/main`
4. immediately before push, fetch/rebase again

Use `scripts/sync-agent.sh` for this. It takes a local lock under `.git/` so
two scheduled syncs on the same clone do not overlap.

Recommended cron entry:

```cron
*/10 * * * * cd /path/to/Reflection-based-KE && scripts/sync-agent.sh >> local/sync-agent.log 2>&1
```

Do not use the periodic sync as a substitute for task state transitions. When
an agent claims, finishes, fails, or publishes an important shared message, it
should commit and push that state transition immediately. The 10-minute sync is
a safety net for ordinary server-head updates and cross-server coordination.

Do not enable automatic `git stash` or `git pull --autostash` for agents. A
dirty working tree means the agent is in the middle of writing something; the
scheduled sync should skip rather than hide a partial edit in a stash.

## Commit Messages

Use short, machine-readable commit messages:

```text
plan exp_27011 by head-server2
update plan exp_27011 by head-server3
propose task exp_27011 by head-server3
create task exp_27011 by head-server2
claim exp_27011 by agent-server3
heartbeat head-server3
heartbeat agent-server3
finish exp_27011 by agent-server3 exit=0
fail exp_27011 by agent-server3 exit=1
```

## File Ownership

Avoid shared hot files. Prefer one file per server, one file per agent, or one
file per task.

Good:

```text
agents/server3/head-server3.json
agents/server3/agent-server3.json
plans/updates/server3/exp_27011.md
runs/exp_27011/summary.agent-server3.md
```

Avoid:

```text
agents/status.json
plans/current.md
runs/all_status.json
```

Conflict risk is low if ownership boundaries are respected. The main conflict
cases are:

- two agents editing the same plan or message file
- two clones using the same `agent.id` and writing the same status file
- a scheduled sync running while an agent has half-written local changes
- direct edits to global files outside the owning role

Avoid these by using per-server/per-agent paths, committing complete changes,
and letting scheduled sync skip dirty working trees.

## Message Policy

`messages/` is for shared coordination, not raw logs. Agents should write
detailed but curated messages that are useful to another agent or to the user.

Use these paths:

- `messages/head/YYYY-MM-DD.md`: global-head announcements and decisions.
- `messages/server-heads/<server>/YYYY-MM-DD.md`: shared updates written by
  that server's `server-head`.
- `messages/templates/`: reusable message templates.

Put these in `messages/`:

- decisions and their rationale
- plan changes and rejected alternatives
- task handoffs, blockers, and requested review
- failure summaries with enough context to debug
- important resource or environment changes
- cross-server observations, requests, and handoffs
- links or paths to local logs and artifacts

Do not put these in `messages/`:

- full shell transcripts
- complete stdout/stderr logs
- repeated progress ticks with no new information
- private scratch reasoning
- credentials, tokens, or SSH material

Raw local logs should stay outside Git under `local/`, server-local storage, or
shared storage. A shared message may reference them by path. The right level of
detail is enough for a future agent to understand what changed, why it changed,
what evidence supports it, and what should happen next without reading the full
local transcript.

## Server Head Shared Messages

Server heads must leave a shared message whenever their work affects another
server, global scheduling, or the user's next decision. These messages should
be detailed enough for another server head to act without private context.

Write a server-head message for:

- remote log inspection or partial verification on another server
- requests for a missing path, permission, environment detail, or account setup
- planned file transfers such as `rsync` of trained `.pt` files, checkpoints,
  datasets, or generated artifacts
- discovery that a task should move to another server
- blockers that require the global head or another server head
- completion of a cross-server handoff

Every cross-server request should include:

- `from`: requesting agent and server
- `to`: target agent/server or role
- related `plan`, `task`, or `run`
- status: `info`, `request`, `blocked`, `handoff`, or `done`
- exact source and destination paths when file movement is involved
- log coverage, for example "checked server2 log through line 1842" or "read
  until timestamp 2026-05-25T12:20:00+09:00"
- evidence: key metric, short error excerpt, checksum, file size, or command
  result when useful
- requested action and owner
- deadline or priority if relevant

Example:

```text
2026-05-25T12:40:00+09:00 head-server1 -> head-server2 [request]
Plan: plan_004, Task: exp_27011
Observed: head-server1 checked server2 local log
/mnt/raid5/janghj/local/runs/exp_27011/train.log through timestamp
2026-05-25T12:20:00+09:00. Training appears complete and produced
best.pt on server1.
Request: Need the destination path on server2 before rsyncing the trained
file. Proposed source is /mnt/raid5/janghj/artifacts/exp_27011/best.pt.
Please provide the server2 destination directory and whether existing files
may be overwritten.
Next owner: head-server2.
```

## Artifact Rule

Git may store:

- plans
- task YAML
- small JSON metrics
- short log tails
- artifact path manifests
- detailed shared messages

Git must not store:

- SSH keys
- API tokens
- full stdout/stderr logs
- checkpoints
- model weights
- datasets

Store large files on shared storage or local server storage, then reference the
path from `runs/<task_id>/artifact_paths.json`.

## Server Head Bootstrap

On each server head clone:

```bash
git clone https://github.com/hyunjun1127/Reflection-based-KE.git ~/agent-control/Reflection-based-KE
cd ~/agent-control/Reflection-based-KE
git config user.name "head-server3"
git config user.email "head-server3@lab.local"
git config agent.id "head-server3"
git config agent.role "server-head"
git config agent.hostname "server3"
scripts/heartbeat.sh
```

## Worker Bootstrap

On each worker clone:

```bash
git clone https://github.com/hyunjun1127/Reflection-based-KE.git ~/agent-control/Reflection-based-KE
cd ~/agent-control/Reflection-based-KE
git config user.name "agent-server3"
git config user.email "agent-server3@lab.local"
git config agent.id "agent-server3"
git config agent.role "worker"
git config agent.hostname "server3"
scripts/heartbeat.sh
scripts/claim-task.sh
```
