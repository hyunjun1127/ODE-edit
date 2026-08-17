# server1 active record

## 현재 authority

- server: `server1`
- physical hostname: `devbox`
- host ID: `remote-ssh-codex-managed:lab120`
- repository: `hyunjun1127/ODE-edit`
- 갱신 시각: 2026-08-17

| 역할 | Codex session ID / deeplink | Required / confirmed model | 실제 CWD | 상태 |
| --- | --- | --- | --- | --- |
| global-head (GH) | `01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` / `codex://threads/01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` | `Sol Ultra` / current-session runtime confirmation pending | `/mnt/raid5/janghj/ODE-edit` | active |
| server-head (SH1) | `01a00e5d-29e8-7a01-822b-7acf43226035` / `codex://threads/01a00e5d-29e8-7a01-822b-7acf43226035` | `Sol Ultra` / current-session runtime confirmation pending | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | direct inbox ACK PASS; execution boundary pending |

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
- SH1 cached `origin/main` 대비 277 behind였으므로 해당 detached worktree에서
  pull/merge하지 않는다.

## Session boundary

SH1 direct ACK 시 local `servers/local/session-boundary.env`는 inactive
`019fe489-c968-75f3-9965-7cfbc26c0a99`를 기록하고 있었다. 새 session에서
actionable work를 시작하기 전에 다음을 모두 만족해야 한다.

1. displayed/runtime primary model이 required `Sol Ultra`인지 직접 확인한다.
2. local-only boundary의 session ID를
   `01a00e5d-29e8-7a01-822b-7acf43226035`로 갱신한다.
3. CWD, repository identity, model confirmation을 실제 값으로 기록한다.
4. `scripts/check-session-boundary.sh
   01a00e5d-29e8-7a01-822b-7acf43226035`를 통과한다.

Confirmation이 없거나 checker가 실패하면 Git write, Slurm, rsync, model/GPU
execution은 HOLD다.

## Slurm 및 resource

- 2026-08-17 ACK 시 ODE-edit active Slurm job: 0
- local cap file의 현재 server1 project GPU cap: 4
- 실제 제출 전에는 `scripts/check-slurm-resource-cap.sh`로 point-in-time GPU와
  host-memory를 다시 확인한다.

## 판정

- session routing/direct inbox: PASS
- tracked repository update owner: GH canonical main integration worktree
- SH1 detached worktree update: HOLD, user-owned untracked paths 보존
- SH1 execution: local session/model boundary PASS 전 HOLD
- 다음 담당자: GH가 main을 push한 뒤 SH1은 tracked 파일을 직접 변경하지 않고
  최신 registry를 확인하고 local boundary 결과를 ACK한다.
