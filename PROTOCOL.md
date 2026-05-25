# Agent Protocol

## Deployment Configuration

Choose one final head server per deployment and configure it as
`global-head`. All server names, repository URLs, and local paths in this file
are examples and should be replaced for each deployment.

## Purpose

Git is the durable control plane for experiment coordination. It stores plans,
task metadata, repository-managed scripts, status summaries, Korean experiment
reports, and small run records. It is not a real-time message queue and it is
not storage for large artifacts.

## Language Policy

모든 agent 간 통신은 사용자가 바로 읽을 수 있도록 한글로 작성한다.
다음 항목은 반드시 한글로 쓴다.

- `messages/`
- `plans/`
- `experiment-reports/`
- global-head 또는 사용자에게 전달되는 conflict report

단, 기술적 정확성을 위해 command, path, metric 이름, filename, error
snippet은 원문 그대로 남길 수 있다.

## Roles

- `global-head`: final coordinator. Owns canonical plans,
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

Global head example:

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

- `plans/global/`: canonical plans. Owned by the configured `global-head`.
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
   machine-readable run summary under `runs/`, writes or updates a Korean
   experiment summary under `experiment-reports/`, commits, and pushes.

The helper script `scripts/claim-task.sh <agent_id>` implements the claim/push
part for the simple first-pending-task case.

## Project Repository Sharing

For each deployment, every server should keep a clone of the same project
repository. The remote repository is the source of truth, and server-local
clones are replaceable working copies.

Use this model:

- every server has a local clone of the project/control repository
- server heads sync their clone periodically and before important writes
- if one server fails, another server can clone the same remote and continue
  from the latest pushed state
- large artifacts remain outside Git and are referenced by path
- server heads record cross-server backup, restore, and transfer requests in
  `messages/server-heads/<server>/`

Do not rely on unpushed local commits as the only copy of important state.
Important plan updates, task state transitions, and handoff messages should be
committed and pushed promptly.

## Repository-Managed And Local-Managed Files

Keep scripts and coordination state in Git. Keep datasets and heavy experiment
outputs local.

Repository-managed:

- `scripts/`: agent helper scripts such as sync, heartbeat, claim, and finish
- `run-scripts/`: experiment execution scripts and wrappers
- `plans/`: Korean plans and plan updates
- `tasks/`: task specs and task lifecycle files
- `messages/`: Korean server-to-server communication only
- `experiment-reports/`: Korean experiment result summaries
- `runs/`: small machine-readable status, metrics, log tails, and artifact
  path manifests

Local-managed:

- datasets and preprocessed data
- raw model outputs and generated artifacts
- checkpoints, model weights, and trained `.pt` files
- full stdout/stderr logs
- temporary scratch files

Local-managed files should be referenced by path from `runs/` and
`experiment-reports/`, not copied into Git.

## Conflict Stop Policy

If any server detects a Git conflict or unrecoverable sync failure, it must
stop automated Git activity for that clone and report upward instead of trying
to fix the conflict independently.

Fail-stop sequence:

1. the detecting agent stops scheduled sync for its local clone
2. it writes a local report under `local/conflicts/`
3. it notifies its `server-head`
4. the `server-head` reports to the `global-head`
5. the `global-head` tells the user and waits for a resolution decision
6. no server should continue automated pull/rebase/push while the conflict is
   being triaged

The `global-head` may publish a repository-wide pause marker at
`control/sync-paused` when all servers should stop scheduled sync. Clean clones
that receive this marker must skip sync until the marker is removed by the
`global-head`.

The local pause marker is `.git/agent-sync.paused`. It is not committed. Remove
it only after the conflict is resolved and the local clone is known clean.

## Periodic Sync

Every server head should sync once per hour. The sync rule is:

1. do nothing if the working tree has uncommitted changes
2. fetch and rebase onto `origin/main`
3. push only if local commits are ahead of `origin/main`
4. immediately before push, fetch/rebase again
5. on conflict, write a local conflict report and pause the clone

Use `scripts/sync-agent.sh` for this. It takes a local lock under `.git/` so
two scheduled syncs on the same clone do not overlap.

Agents do not wake themselves up after a session exits. Use `cron`,
`systemd timer`, or another scheduler to call `scripts/sync-agent.sh`.

Recommended cron entry:

```cron
0 * * * * cd /path/to/agent-control && scripts/sync-agent.sh >> local/sync-agent.log 2>&1
```

Do not use the periodic sync as a substitute for task state transitions. When
an agent claims, finishes, fails, or publishes an important shared message, it
should commit and push that state transition immediately. The hourly sync is
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
experiment-reports/servers/server3/exp_27011.md
runs/exp_27011/status.agent-server3.json
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

`messages/` is for Korean server-to-server communication only. It is not for
experiment result reports and it is not for raw logs. Agents should write
detailed but curated Korean messages that are useful to another agent or to the
user.

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
- conflict summaries after the global head has been notified
- important resource or environment changes
- cross-server observations, requests, and handoffs
- links or paths to local logs and artifacts

Do not put these in `messages/`:

- experiment result reports; use `experiment-reports/`
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

## Experiment Reports

Experiment result summaries must be written in Korean and stored separately
from communication messages.

Use these paths:

- `experiment-reports/global/<experiment_id>.md`: final integrated summary
  curated by the `global-head`
- `experiment-reports/servers/<server>/<experiment_id>.md`: server-specific
  observations written by that server's `server-head` or worker
- `runs/<run_id>/`: machine-readable metadata such as status JSON, metrics JSON,
  short log tails, and artifact path manifests

Experiment reports should include:

- experiment purpose and related plan/task/run IDs
- server, agent, command, environment, and resource summary
- dataset path and output/artifact paths without copying the data into Git
- key metrics and comparison against expectations
- notable failure modes, warnings, or caveats
- interpretation of the result and recommended next action

Do not use `messages/` as a result report archive. Use `messages/` only to tell
other servers what changed, what is requested, and where the report/artifacts
are located.

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
/path/to/local/runs/exp_27011/train.log through timestamp
2026-05-25T12:20:00+09:00. Training appears complete and produced
best.pt on server1.
Request: Need the destination path on server2 before rsyncing the trained
file. Proposed source is /path/to/artifacts/exp_27011/best.pt.
Please provide the server2 destination directory and whether existing files
may be overwritten.
Next owner: head-server2.
```

## Artifact Rule

Git may store:

- plans
- task YAML
- experiment run scripts under `run-scripts/`
- small JSON metrics
- short log tails
- artifact path manifests
- detailed shared messages
- Korean experiment result summaries under `experiment-reports/`

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
git clone https://github.com/<owner>/<repo>.git ~/agent-control/<repo>
cd ~/agent-control/<repo>
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
git clone https://github.com/<owner>/<repo>.git ~/agent-control/<repo>
cd ~/agent-control/<repo>
git config user.name "agent-server3"
git config user.email "agent-server3@lab.local"
git config agent.id "agent-server3"
git config agent.role "worker"
git config agent.hostname "server3"
scripts/heartbeat.sh
scripts/claim-task.sh
```
