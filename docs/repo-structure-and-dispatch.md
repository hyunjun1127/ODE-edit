# Repo Structure And Agent Dispatch

- Purpose: define the reusable post-SSH-mesh layout for this agent-control template.

## Core Split

This repository has two layers.

| Layer | Responsibility | Storage |
|---|---|---|
| Control plane | instructions, status, reports, audits, compact manifests | Git tracked paths |
| Execution plane | SSH, Slurm, rsync, raw artifacts, checkpoints, full logs | each server's ignored `local/` |

Git messages and tasks are durable instructions. They are not schedulers. Work
runs through active agents, SSH, Slurm, and one-shot rsync jobs.

## Canonical Paths

| Purpose | Canonical path | Notes |
|---|---|---|
| Operating rules | `PROTOCOL.md` | Final policy source |
| Human overview | `README.md` | Layout and artifact boundary |
| Research proposals | `project/proposals/` | New GH handoff and project proposal drafts |
| Redacted SSH/rsync inventory | `servers/connection-inventory.md` | Raw host/IP/user/port stays in `servers/local/` |
| Server/agent state | `agents/<server>/` | Heartbeat, sync, and subagent status |
| Global-head messages | `messages/head/` | Human-readable broadcasts and decisions |
| Server inbox | `messages/inbox/<server>.md` | Global-head-owned durable instructions for one target server |
| Server-head messages | `messages/server-heads/<server>/` | Source server reports and blockers |
| Message index | `messages/README.md` | All server-heads read this and ack |
| Message ack | `messages/acks/<server>/` | Each server writes only its own ack |
| Durable task state | `tasks/` | Only when per-server status and closure are needed |
| Run status | `runs/<run_id>/` | Compact status, log tails, artifact manifests |
| Experiment reports | `experiment-reports/` | Human interpretation and result summaries |
| Red-team audits | `audits/` | Logic, protocol, data, and result audits |
| Run scripts | `project/run_scripts/` | Root `run-scripts/` is deprecated |
| Raw artifacts | `local/` | Git ignored; share by rsync broadcast |
| Artifact broadcast helper | `scripts/rsync-artifact-broadcast.sh` | Source server executes |
| Slurm broadcast dependency | `scripts/submit-artifact-broadcast-dependency.sh` | Optional immediate post-job broadcast |
| GPU cap helper | `scripts/check-slurm-gpu-cap.sh` | Uses ignored `servers/local/gpu-caps.tsv` |
| GPU + host-memory cap helper | `scripts/check-slurm-resource-cap.sh` | Enforces the lower of local limits and the tracked server ceiling |
| Slurm memory policy/auditor | `scripts/slurm_memory_policy.py` | Requires explicit `--mem` and checks all tracked `.sbatch` files |
| Tracked server memory ceilings | `servers/slurm-memory-policy.tsv` | Scheduler maxima plus 1-GiB-headroom repo maxima |
| Codex session boundary helper | `scripts/check-session-boundary.sh` | Refuses a session/CWD/origin mismatch before repo operations |
| GPU cap config template | `servers/templates/gpu-caps.tsv` | Copy to `servers/local/` and edit |

## Deprecated Or Legacy Paths

| Path | Status | Replacement |
|---|---|---|
| `run-scripts/` | deprecated | `project/run_scripts/` |
| `workers/` | deprecated/unused | `agents/<server>/`, `tasks/running/`, `runs/<run_id>/` |
| `transfers/` for ordinary artifacts | legacy | automatic `local/` broadcast |
| old policy-adoption tasks in `tasks/pending/` | cleanup backlog | close or waive once policy is merged |

## SSH Mesh Dispatch

When passwordless SSH aliases are configured, global-head may submit remote
jobs after preflight.

```text
user
  -> global-head
       -> ssh <target-server> preflight
       -> ssh <target-server> sbatch
       -> server-head completion/audit
       -> artifact broadcast
       -> target server-head blue/red review
```

Rules:

1. A remotely submitted job is owned by the target server.
2. The target server-head acknowledges and reviews the job after sync.
3. Slurm work must declare explicit `--mem` and respect the target server's
   project GPU and host-memory cap. If either cap is unknown, `--mem` is
   absent, or a cap is exceeded, leave the work pending instead of submitting.
4. Ordinary artifacts stay under `local/` and are shared by artifact broadcast.
5. Git receives compact status, reports, summaries, and manifests only.
6. LLM analysis requires an active agent process or explicit automation.

Inbox/task instructions do not execute by themselves. A target server-head must
sync, read the inbox/task, run preflight, submit or delegate the job, monitor
completion, audit, report, and then broadcast artifacts when appropriate.
Actionable global-head instructions must include allowed write paths, Slurm
permission, GPU cap, red-team gate, artifact broadcast duty, and completion
report paths.
They must also include the target Codex session ID, repository CWD, and Git
repository identity; mismatch with any value is a hard stop.

## Task Compression

Use `messages/` for most policy announcements, research decisions, handoffs,
and blockers.

Create `tasks/pending/` entries only when:

- multiple servers need tracked status,
- closure or waiver is required,
- a Slurm job lifecycle or user-visible blocker must be tracked,
- a worker needs to claim execution ownership.

Do not turn every message into a task.

## Artifact Transfer Policy

Ordinary project artifacts are automatically shared when useful:

- `local/results/raw/...`
- `local/logs/slurm/...`
- `local/logs/run/...`
- required dataset/checkpoint subsets

Manual approval is still required for:

- `rsync --delete` or destructive mirrors,
- repo-external paths,
- `local/secrets/`, SSH material, credentials, private inventories,
- unclear ownership or high overwrite risk,
- non-project or sensitive data.
