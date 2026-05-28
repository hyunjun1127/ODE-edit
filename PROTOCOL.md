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
SSH, Slurm, and rsync are the execution plane. Once SSH mesh is available,
global-head may submit remote jobs after preflight, but the target server-head
still owns the experiment lifecycle and evidence review.

## Language Policy

모든 agent 간 통신은 사용자가 바로 읽을 수 있도록 한글로 작성한다.
다음 항목은 반드시 한글로 쓴다.

- `messages/`
- `plans/`
- `experiment-reports/`
- `audits/`
- `transfers/`
- `servers/`
- global-head 또는 사용자에게 전달되는 conflict report

단, 기술적 정확성을 위해 command, path, metric 이름, filename, error
snippet은 원문 그대로 남길 수 있다.

## Roles

- `global-head`: final coordinator. Owns canonical plans,
  resolves conflicts between server heads, creates approved tasks, and talks to
  the user.
- `server-head`: per-server coordinator. A `head-serverN` agent can update its
  server plan notes, propose tasks, inspect local resources, and coordinate
  local workers/subagents. The server-head is responsible for integrating
  subagent outputs and deciding what gets committed.
- `worker`: execution agent. Claims approved tasks, runs jobs, monitors logs,
  and reports results.
- `subagent`: local helper owned by a server head or worker. Subagents should
  not push to Git directly; their parent agent reviews, summarizes, and commits
  results.
- `blue-team subagent`: 연구 파이프라인을 실행하는 subagent. Plan 실행,
  실험 실행, 결과 정리, 해석을 담당한다.
- `red-team subagent`: blue team 작업을 감사하는 subagent. 데이터 분리,
  실험 논리, 근거, Git/protocol 준수 여부를 검사한다.

## Universal Agent Rules

All agents must follow these rules.

- keep repository-local `user.*` and `agent.*` identity accurate
- write shared communication, plans, reports, audits, transfer records, and
  server lifecycle records in Korean
- pull/rebase before important writes and before push
- keep one file per server, agent, task, run, request, or verification when
  possible
- never commit credentials, SSH material, datasets, checkpoints, raw outputs,
  generated outputs at scale, or full logs
- never run destructive `rsync --delete`, sensitive file transfer, or repo
  external large transfer without explicit user/global-head approval. Ordinary
  project artifact fan-out under `local/` uses the approved helper flow.
- stop and report upward on Git conflict, uncertain destructive action, or
  red-team `block`
- write enough evidence for another agent to reproduce the decision without
  reading private local scratch
- do not impersonate another `agent.id`, `agent.role`, or server hostname

## Parent Chain And Access Control

Parent agents own final responsibility for their child agents' outputs.
Subagents can draft, inspect, summarize, and audit, but they do not directly
push Git state.

Parent chain:

```text
user
  -> global-head
       -> server-head
            -> worker
            -> blue-team subagents
            -> red-team subagents
```

Role access matrix:

| Role | Primary parent | May own/write | Must not directly write |
| --- | --- | --- | --- |
| `global-head` | user | `plans/global/`, `tasks/pending/`, `messages/head/`, `transfers/approvals/`, `servers/active/`, `servers/retired/`, `control/`, `experiment-reports/global/`, protocol/template updates | server-local raw output, another role's unreviewed execution results, unapproved transfer execution |
| `server-head` | global-head | `plans/updates/<server>/`, `tasks/proposed/<server>/`, `messages/server-heads/<server>/`, `agents/<server>/`, `audits/servers/<server>/`, `experiment-reports/servers/<server>/`, `transfers/requests/`, approved `transfers/verifications/`, own `servers/active/<server>.md` updates | `plans/global/`, `tasks/pending/`, `messages/head/`, `transfers/approvals/`, other server-owned files |
| `worker` | server-head | its claimed `tasks/running/<task>.<agent>.*`, matching `tasks/done/` or `tasks/failed/`, `runs/<run_id>/...<agent>.*`, `agents/<server>/<agent>.json`, assigned report/audit evidence under its server | creating tasks, editing canonical plans, approving transfers, changing server lifecycle, editing other agents' task/run files |
| `blue-team subagent` | server-head or worker | local scratch, draft plan/run/report material for parent review | direct Git push, final report promotion, task approval, transfer approval, red-team waiver |
| `red-team subagent` | server-head or worker | local scratch and audit drafts for parent review | direct Git push, modifying blue-team artifacts instead of reporting issues, waiving its own blocker |

Actions that require escalation:

- promote `tasks/proposed/<server>/` to `tasks/pending/`: `global-head`
- edit `plans/global/`: `global-head`
- record `transfers/approvals/`: `global-head` after user decision
- execute a large transfer: designated agent only after recorded user approval
- execute ordinary project artifact fan-out under `local/`: submitting or
  source-server agent through `scripts/rsync-artifact-fanout.sh` or a Slurm
  `afterany` fan-out dependency
- waive red-team `warn` or `block`: `global-head`, with reason in `audits/`
- register or retire a server: `global-head` with user/server-owner context
- modify `control/sync-paused`: `global-head`

Helper scripts are intentionally narrow. `claim-task.sh` and `finish-task.sh`
are for `worker` agents. `heartbeat.sh` and `sync-agent.sh` are for
`global-head`, `server-head`, or `worker` clones. Subagents should report to
their parent instead of running Git-writing helper scripts directly.

Before committing sensitive changes, run:

```bash
scripts/check-agent-access.sh --staged
```

This checks staged paths against the current clone's `agent.role`,
`agent.id`, and `agent.hostname`. It is a guardrail, not a replacement for
red-team review or global-head/user approval.

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

Do not create a task for every message. Most policy announcements, research
decisions, and one-off status updates belong only in `messages/`. Use
`tasks/` when work needs durable per-server state, execution ownership, or a
global-head closure decision.

Worker execution lifecycle:

1. A server head writes plan updates under `plans/updates/<server>/` or task
   proposals under `tasks/proposed/<server>/`.
2. Red team performs a pre-flight audit before the proposal is promoted.
3. The global head reviews updates and creates approved task files in
   `tasks/pending/`.
4. Worker runs `git pull --rebase`.
5. Worker claims the task by moving it to
   `tasks/running/<task_id>.<agent>.yaml`.
6. Worker commits with `claim <task_id> by <agent>`.
7. Worker pushes.
8. If push fails, worker pulls/rebases and checks whether the task is still in
   `tasks/pending/`.
9. Worker executes the task.
10. Red team performs a post-run audit before results are finalized.
11. Worker moves the task to `tasks/done/` or `tasks/failed/`, writes a small
   machine-readable run summary under `runs/`, writes or updates a Korean
   experiment summary under `experiment-reports/`, commits, and pushes.

The helper script `scripts/claim-task.sh <agent_id>` implements the claim/push
part for the simple first-pending-task case.

Broadcast/global-head command tasks are not worker-claimable. When a command
targets multiple servers, the global-head writes a readable message under
`messages/head/`, optionally creates a structured task under `tasks/pending/`,
and each target server writes status under `tasks/status/<task_id>/<server>.json`.

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
  `messages/server-heads/<server>/` and `transfers/`

Do not rely on unpushed local commits as the only copy of important state.
Important plan updates, task state transitions, and handoff messages should be
committed and pushed promptly.

## SSH Mesh Execution Model

When server-to-server SSH aliases are configured, use Git for durable
instruction and evidence, but use SSH/Slurm/rsync for execution.

Rules:

1. Global-head may run a remote submission such as
   `ssh <alias> 'cd <repo> && sbatch ...'` only after checking the target
   clone, required local artifacts, current Git commit, Slurm access, and
   server-specific resource caps.
2. A remotely submitted job is target-server owned. The target server-head
   acknowledges it after sync and performs or delegates post-submit red-team,
   post-run blue-team, and post-run red-team review when the deployment uses
   those gates.
3. Ordinary artifacts under `local/` are shared by automatic rsync fan-out and,
   for Slurm jobs, an `afterany:<job_id>` fan-out dependency.
4. `transfers/` is reserved for manual exceptions: external paths, destructive
   mirror behavior, sensitive material, or unusual overwrite risk.
5. A Git message or task does not execute LLM analysis by itself. Analysis
   happens only when an agent session or explicit automation is alive.

## Repository-Managed And Local-Managed Files

Keep scripts and coordination state in Git. Keep datasets and heavy experiment
outputs local.

Repository-managed:

- `scripts/`: agent helper scripts such as sync, heartbeat, claim, and finish
- `project/run_scripts/`: experiment execution scripts and wrappers
- `subagents/`: required blue/red team role specs
- `audits/`: Korean red-team audit reports
- `servers/`: server onboarding/offboarding records and templates
- `transfers/`: large file transfer requests, approvals, and verification
  summaries
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

## Large Output And Transfer Approval Policy

Raw output, datasets, checkpoints, generated artifacts, and full logs can be
very large and must stay local or on shared storage. Git stores only metadata,
summaries, paths, and verification records.

Store in Git:

- short log tail
- metrics summary
- experiment report
- artifact path manifest
- file size/checksum when useful
- transfer request, user approval record, and verification summary

Do not store in Git:

- raw output directories
- generated samples or model outputs at scale
- checkpoints, trained `.pt` files, model weights
- full stdout/stderr logs
- datasets or preprocessed datasets

Ordinary project artifact sharing under repo-local `local/` is automatic once
SSH mesh is configured. It uses `scripts/rsync-artifact-fanout.sh` and, for
Slurm jobs, the mandatory `afterany` dependency helper. Per-transfer user
approval is not required for normal experiment raw results, run logs, Slurm
logs, datasets, and checkpoints that are already part of this project.

Manual transfer approval is still required for exceptions:

- `rsync --delete` or destructive mirror behavior
- repo-external source or destination paths
- sensitive/private paths such as `local/secrets/`, SSH material, private
  inventories, credentials, tokens, or passphrases
- unusual overwrite risk or unclear ownership
- one-time import/export outside normal project fan-out

Manual-exception transfer workflow:

1. requesting server-head writes a Korean request under `transfers/requests/`
   and links it from `messages/server-heads/<server>/`
2. request includes source server/path, destination server/path, expected size,
   checksum if available, overwrite policy, reason, priority, and one concrete
   transfer command or transfer plan
3. global-head reviews the request and reports it to the user
4. user approves, rejects, or asks for changes
5. if approved, global-head records approval under `transfers/approvals/`
6. only the approved one-shot transfer may be executed by the designated agent
7. after transfer, agent verifies destination existence, file size, checksum,
   and expected path
8. verification result is written under `transfers/verifications/` and linked
   from `runs/` or `experiment-reports/`

If SSH/rsync authentication is not already configured, the user must handle the
credential setup. Passwords, SSH private keys, tokens, and private connection
details must never be written to Git, messages, reports, or local conflict
reports.

Approval scope is narrow. A user approval applies only to the source,
destination, overwrite policy, and command described in that request. Any
different path, retry strategy, recursive directory, or overwrite behavior
requires a new approval.

### Ordinary Artifact Fan-Out

Server-local `local/` folders are not shared through Git. When ordinary project
artifacts need to be copied between active servers, use:

```bash
scripts/rsync-artifact-fanout.sh local/results/raw/<run_id>
scripts/rsync-artifact-fanout.sh local/logs/slurm/<family>
```

For Slurm experiment submissions, submit a second dependency job:

```bash
scripts/submit-artifact-fanout-dependency.sh --job-id <experiment_job_id> \
  local/results/raw/<run_id> \
  local/logs/slurm/<family> \
  local/logs/run/<family>
```

Required constraints:

- no `--delete`
- exclude `local/secrets/`, SSH material, credentials, private inventory,
  `local/scratch/`, `local/run_scripts/`, `local/conflicts/`, `local/tmp/`,
  and `local/.cache/`
- record compact evidence in `runs/`, `messages/`, reports, or fan-out Slurm
  logs
- use `transfers/` only for manual exceptions

## Dynamic Server Lifecycle

Deployments may add temporary rental servers and later remove them. Server
lifecycle changes must be explicit and auditable.

Server onboarding workflow:

1. user or global-head registers the server identity and expected use in
   `servers/active/<server>.md`
2. user handles account access, SSH trust, storage mount, Slurm access, and any
   rsync authentication setup when needed
3. server-head clones the project repo and configures `agent.id`,
   `agent.role=server-head`, and `agent.hostname`
4. server-head writes a heartbeat under `agents/<server>/`
5. blue/red team subagents are initialized and documented
6. red team performs onboarding audit: Git identity, Slurm command access,
   local paths, dataset/output policy, and message/report separation
7. global-head assigns tasks only after onboarding audit passes or is explicitly
   waived

Server offboarding workflow:

1. server-head stops claiming new tasks and writes an offboarding notice
2. running tasks are completed, failed, or reassigned by global-head
3. needed large outputs/checkpoints/log archives are requested through
   `transfers/requests/` and approved by the user through global-head
4. approved transfers are executed and verified
5. server-head writes final server summary under
   `experiment-reports/servers/<server>/` if needed
6. global-head moves server record from `servers/active/` to
   `servers/retired/`
7. credentials, scheduled sync, Slurm submissions, and local agent processes are
   disabled by the user or server owner

Rental servers should not be treated as durable storage. Important state must
be pushed to Git, and important large artifacts must be transferred with user
approval or explicitly marked disposable before the rental period ends.

## Required Red/Blue Team Subagents

Every server head must maintain both blue-team and red-team subagents. Blue
team executes the research pipeline. Red team audits blue-team work before it
can affect canonical plans, approved tasks, or finalized reports.

Minimum blue-team subagents per server:

- `blue-plan-runner`: plan/task를 실행 가능한 형태로 해석하고, 필요한
  `project/run_scripts/`, config, command, resource 요구사항을 정리한다. 이전
  실험 대비 config 차이도 함께 정리한다.
- `blue-experiment-runner`: 승인된 task를 실행하고, job 상태, log tail,
  artifact path, file size/checksum, exit code를 `runs/`에 기계가 읽을 수
  있게 정리한다. 실패 시 CUDA OOM, import error, data path error, logic
  error 등으로 1차 분류한다.
- `blue-result-analyst`: 실험 결과를 한글로 정리하고 해석하여
  `experiment-reports/servers/<server>/`에 작성한다. 재현성 정보, baseline
  비교표, 핵심 metric, caveat, 다음 실험 제안을 포함한다.

Minimum red-team subagents per server:

- `red-data-eval-auditor`: train/test/validation 분리, data leakage,
  evaluation set 오염, metric 계산 조건, 재현 가능한 data path를 검사한다.
- `red-logic-evidence-auditor`: 실험 가정, 비교 기준, ablation 논리,
  결과 해석, hallucination 가능성, 근거 없는 주장 여부를 검사한다. report
  claim이 실제 metric/artifact/log 근거와 연결되는지 확인한다.
- `red-git-protocol-auditor`: Git file ownership, message/report 분리,
  local-managed output 미추적, secret/checkpoint/full-log 유입 여부,
  artifact manifest 존재 여부, sync/conflict protocol 준수를 검사한다.

Red team 결과는 반드시 한글로 `audits/servers/<server>/`에 남긴다. Red
team이 `block`으로 판정한 경우 server-head는 해당 task 승격, 결과 확정,
또는 관련 push를 진행하지 않고 `messages/server-heads/<server>/`에
blocker를 공유해야 한다. Global-head가 예외를 허용할 때는
`audits/`에 waiver 사유를 남긴다.

Required gates:

- `pre-flight`: `tasks/proposed/<server>/`가 `tasks/pending/`으로 승격되기
  전에 data split, 논리, 실행 경로, Git 경로 정책을 검사한다.
- `post-run`: `tasks/done/` 또는 `tasks/failed/`로 확정하기 전에 metric,
  report 해석, artifact path, log 근거를 검사한다.
- `pre-push-sensitive`: canonical plan, approved task, experiment report,
  audit 결과처럼 여러 서버에 영향을 주는 변경은 red-team check 후 push한다.
- `server-onboarding`: 새 서버에 task를 배정하기 전에 Slurm, Git identity,
  local path, output policy, subagent readiness를 검사한다.
- `server-offboarding`: 서버 종료 전에 task handoff, transfer approval,
  transfer verification, scheduler/credential 정리 여부를 검사한다.

Red-team 판정은 다음 네 단계 중 하나를 사용한다.

- `pass`: 진행 가능
- `warn`: 진행 가능하지만 report와 message에 caveat를 남겨야 함
- `block`: 진행 중단. 수정 또는 사용자/global-head 결정 필요
- `waived`: 원래는 block 또는 warn이지만 global-head가 사유를 남기고 예외 허용

## Server Head And Subagent Policy

Subagent에게 결과 정리, 실행 준비, 감사, 분석을 맡기는 것은 맞다. 다만
subagent는 repo의 최종 작성자가 아니다. Subagent는 초안, 검사 결과,
근거 목록을 만들고, server-head 또는 명시된 parent agent가 이를 검토한 뒤
commit/push한다.

Server-head 책임:

- blue/red subagent를 호출하고 작업 범위를 나눈다
- subagent 산출물을 모아 서로 모순되는 부분을 정리한다
- Red team `block` 또는 `warn`을 확인하고 필요한 수정/waiver를 결정한다
- repo에 반영할 파일을 선별하고 file ownership을 지킨다
- 중요한 변경 전 `git pull --rebase`를 수행하고, push 후 공유 메시지를 남긴다
- global-head/user에게 필요한 결정을 한글로 보고한다

Subagent 제한:

- 직접 `git push`하지 않는다
- `tasks/pending/`, `plans/global/` 같은 global-head 소유 경로를 직접
  바꾸지 않는다
- full log, checkpoint, dataset, raw output을 Git에 추가하지 않는다
- 확정되지 않은 분석을 최종 report처럼 작성하지 않는다
- 의심 사항은 수정하지 말고 audit 또는 message 초안으로 parent에게 보고한다

권장 local 작업 흐름:

1. subagent는 local scratch 또는 parent agent의 작업 메모리에 초안을 만든다
2. server-head가 초안을 검토하고 필요한 repo path로 옮긴다
3. red-team이 gate를 통과시키거나 `warn/block`을 남긴다
4. server-head가 최종 파일을 commit/push한다

Worker가 task lifecycle을 직접 처리하는 배포에서는 worker가 parent agent가
될 수 있다. 이 경우에도 worker 내부 subagent는 직접 push하지 않고, worker가
server-head 정책을 따른다.

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
audits/servers/server3/exp_27011.preflight.md
audits/servers/server3/exp_27011.postrun.md
servers/active/server3.md
transfers/requests/exp_27011.server1-to-server2.md
transfers/approvals/exp_27011.server1-to-server2.md
transfers/verifications/exp_27011.server1-to-server2.md
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
  datasets, or generated artifacts. This is a transfer request only; execution
  waits for user approval recorded by the global-head.
- server onboarding/offboarding status that affects scheduling
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
Transfer request: transfers/requests/exp_27011.server1-to-server2.md
Request: Need the destination path on server2 before asking global-head for
user approval. Proposed source is /path/to/artifacts/exp_27011/best.pt.
Please provide the server2 destination directory and whether existing files
may be overwritten. No rsync will run until user approval is recorded under
transfers/approvals/.
Next owner: head-server2.
```

## Artifact Rule

Git may store:

- plans
- task YAML
- subagent role specs under `subagents/`
- red-team audit reports under `audits/`
- server lifecycle records under `servers/`
- user-approved transfer requests, approvals, and verification summaries under
  `transfers/`
- experiment run scripts under `project/run_scripts/`
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
- raw output directories
- generated samples or model outputs at scale

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
