# server1 active record

## 현재 authority

- server: `server1`
- physical hostname: `devbox`
- host ID: `remote-ssh-codex-managed:lab120`
- repository: `hyunjun1127/ODE-edit`
- 갱신 시각: 2026-08-17

| 역할 | Codex session ID / deeplink | Required / confirmed model | 실제 CWD | 상태 |
| --- | --- | --- | --- | --- |
| global-head (GH) | `01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` / `codex://threads/01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` | user-managed; observed `gpt-5.6-sol/xhigh` | `/mnt/raid5/janghj/ODE-edit` | active |
| server-head (SH1) | `01a00e5d-29e8-7a01-822b-7acf43226035` / `codex://threads/01a00e5d-29e8-7a01-822b-7acf43226035` | user-managed; observed `gpt-5.6-sol/xhigh` | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | direct inbox ACK PASS |

새 primary session은 이전 active session authority를 supersede한다. 과거 task,
report, audit, receipt와 completed launcher의 old session ID는 historical
provenance로 보존하며 현재 command authority로 해석하지 않는다.

## Repository 상태

- canonical remote: `https://github.com/hyunjun1127/ODE-edit.git`
- GH root clone: `/mnt/raid5/janghj/ODE-edit`; detached/dirty user state이므로
  reset, revert, cleanup 또는 main 통합에 사용하지 않는다.
- canonical main integration worktree:
  `/mnt/raid5/janghj/.codex/worktrees/odeedit-p2r7-main-publish-v1`
- SH1 worktree:
  `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit`
- SH1 ACK snapshot: detached
  `cdb80bc70032c203334531edb7020ff654f2938d`, tree
  `cc8285949540f70a4afcc0278968c3978068b465`; tracked diff 0; untracked
  `project/run_scripts/ode_bf/`와 `runs/session02-p0-tech-v1/` 보존.
- SH1은 detached HEAD를 그대로 유지한 채
  `origin/main=1caed88867db3087fa5db26995c7e3719c064216`을 확인했다. 해당
  worktree에서는 pull/merge하지 않는다.

## Session boundary

SH1은 local `servers/local/session-boundary.env`의 SH1/GH ID를 새 값으로
갱신했고 detached HEAD와 user-owned untracked paths를 보존했다. Runtime
model/profile은 사용자 관리 관측값이며 checker의 hard boundary가 아니다.

1. CWD, repository identity와 session ID를 실제 값으로 기록한다.
2. `scripts/check-session-boundary.sh
   01a00e5d-29e8-7a01-822b-7acf43226035`를 통과한다.

Hard boundary checker가 실패하면 Git write, Slurm, rsync, model/GPU execution은
HOLD다.

## Slurm 및 resource

- 2026-08-17 ACK 시 ODE-edit active Slurm job: 0
- local cap file의 현재 server1 project GPU cap: 4
- 실제 제출 전에는 `scripts/check-slurm-resource-cap.sh`로 point-in-time GPU와
  host-memory를 다시 확인한다.

## 판정

- session routing/direct inbox: PASS
- tracked repository update owner: GH canonical main integration worktree
- SH1 detached worktree update: HOLD, user-owned untracked paths 보존
- SH1 execution: hard session/repository boundary PASS 필요
- `origin/main` registry update 확인: PASS
- 다음 담당자: SH1은 최신 tracked checker로 hard boundary를 재확인한다.
