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
- required model: `Sol Ultra`
- confirmed model: current-session runtime confirmation pending
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

새 GH가 canonical main update를 push한 뒤 SH2는 clean 상태를 다시 확인하고
`git fetch origin`, `git merge --ff-only origin/main`만 수행한다. Dirty 또는
diverged 상태이면 자동 정리하지 않고 HOLD/ACK한다.

## Session boundary

ACK 시 ignored `servers/local/session-boundary.env`는 inactive
`019fe491-954b-70a0-8ba8-0588e9f8d741`를 기록해 새 session checker가 rc4로
실패했다. Actionable work 전에 다음을 수행한다.

1. displayed/runtime primary model이 required `Sol Ultra`인지 직접 확인한다.
2. local-only session ID를
   `01a00e5c-f7ae-72a2-98b2-b8b0907168b4`로 갱신한다.
3. 실제 CWD, repository identity, confirmed model을 기록한다.
4. `scripts/check-session-boundary.sh
   01a00e5c-f7ae-72a2-98b2-b8b0907168b4`를 통과한다.

Runtime model metadata를 조회하지 못했으므로 이전 session의 confirmation을
재사용하지 않는다. Boundary PASS 전 Git write, Slurm, rsync, model/GPU
execution은 HOLD다.

## Slurm 및 resource

- 2026-08-17 ACK 시 ODE-edit active Slurm job: 0
- local cap file의 현재 server2 project GPU cap은 clone-local authoritative
  record로 재확인한다.
- 실제 제출 전에는 `scripts/check-slurm-resource-cap.sh`로 point-in-time GPU와
  host-memory를 다시 확인한다.

## 판정

- session routing/direct inbox: PASS
- repository worktree: clean, fast-forward eligible after boundary PASS
- current blocker: stale local boundary와 current-session model confirmation pending
- next: GH main push 후 boundary 확인, fetch/ff-only sync, final SH2 ACK
