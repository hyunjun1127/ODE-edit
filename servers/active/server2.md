# server2 active record

## 현재 authority

- server: `server2`
- physical hostname: `server2`
- host ID: `remote-ssh-codex-managed:lab121`
- repository: `hyunjun1127/ODE-edit`
- 갱신 시각: 2026-08-17
- server-head (SH2):
  `01a00e5c-f7ae-72a2-98b2-b8b0907168b4`
  (`codex://threads/01a00e5c-f7ae-72a2-98b2-b8b0907168b4`)
- model/profile: user-managed; observed `gpt-5.6-sol/max`
- repository CWD: `/mnt/raid5/janghj/ODE-edit`

사용자 메시지에 SH2 session ID가 SH1과 동일하게 중복 기재됐으나, app host
metadata와 SH2 direct ACK로 위 ID를 확인했다. 이 값이 현재 canonical SH2
authority다.

## Repository 상태

- remote: `https://github.com/hyunjun1127/ODE-edit.git`
- ACK snapshot branch: `main`
- ACK snapshot HEAD/tree:
  `6145406ae4b11e05b683c46aa604c972eb727f5a` /
  `8284a6d0230c275056aa38840a03e6a336c2032c`
- worktree: clean
- cached `origin/main`:
  `858f562cc8282b9ae164d41b85dfad070a8273f2`
- ACK 시 local main은 cached `origin/main` 대비 ahead 0 / behind 30

GH는 canonical main
`1caed88867db3087fa5db26995c7e3719c064216`을 push했다. 최초 sync에서는
model gate로 fetch/ff-only가 보류됐지만, 사용자가 model/profile을 직접 관리한다고
명시해 해당 gate를 해제했다.

## Session boundary

Initial sync ACK 시 ignored `servers/local/session-boundary.env`는 inactive
`019fe491-954b-70a0-8ba8-0588e9f8d741`를 기록해 session ID gate가 rc4로
실패하는 상태였다. Runtime `gpt-5.6-sol/max`는 사용자 관리 관측값이며 더 이상
hard boundary가 아니다.

1. local-only session ID를
   `01a00e5c-f7ae-72a2-98b2-b8b0907168b4`로 갱신한다.
2. 실제 CWD와 repository identity를 기록한다.
3. `scripts/check-session-boundary.sh
   01a00e5c-f7ae-72a2-98b2-b8b0907168b4`를 통과한다.

Boundary PASS 전 Git write, fetch/merge, Slurm, rsync, model/GPU execution은
HOLD다.

## Slurm 및 resource

- 2026-08-17 ACK 시 ODE-edit active Slurm job: 0
- local cap file의 현재 server2 project GPU cap은 clone-local authoritative
  record로 재확인한다.
- 실제 제출 전에는 `scripts/check-slurm-resource-cap.sh`로 point-in-time GPU와
  host-memory를 다시 확인한다.

## 판정

- session routing/direct inbox: PASS
- repository worktree: clean; local fast-forward 미실행
- current blocker: stale local session ID only
- next: boundary ID update/check, fetch, ff-only sync, final SH2 ACK
