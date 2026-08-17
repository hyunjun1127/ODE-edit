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
- initial ACK branch: `main`
- initial ACK HEAD/tree:
  `6145406ae4b11e05b683c46aa604c972eb727f5a` /
  `8284a6d0230c275056aa38840a03e6a336c2032c`
- worktree: clean
- cached `origin/main`:
  `858f562cc8282b9ae164d41b85dfad070a8273f2`
- final ff-only sync HEAD/tree:
  `d4206536d0c9629e1061b6575bc967e0cf1f742b` /
  `7f8b563c89361e48c126301eb906a0dd6484a9c9`
- final sync worktree: clean, ahead/behind 0/0

사용자 model-management 정정 뒤 SH2는 session ID/CWD/repository hard boundary를
PASS했고 clean `main`을 `git merge --ff-only origin/main`으로 동기화했다.

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
- repository worktree: clean; ff-only sync PASS
- hard session/repository boundary: PASS
- registry GH/SH1/SH2 ID verification: PASS
- current blocker: 없음
- next: 새 actionable task가 있을 때 point-in-time repository/resource boundary 재확인
