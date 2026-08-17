# 2026-08-17 GH/SH 새 세션 registry reconciliation

## 목적

GH, SH1, SH2의 새 Codex session을 실제 app host metadata와 direct inbox ACK로
확인하고 repository의 current authority record를 갱신한다. 과거 experiment
provenance는 변경하지 않는다.

## 확인된 현재 session

| 역할 | server/host | session ID | direct inbox |
| --- | --- | --- | --- |
| GH | server1 / `remote-ssh-codex-managed:lab120` | `01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` | active caller |
| SH1 | server1 / `remote-ssh-codex-managed:lab120` | `01a00e5d-29e8-7a01-822b-7acf43226035` | ACK PASS |
| SH2 | server2 / `remote-ssh-codex-managed:lab121` | `01a00e5c-f7ae-72a2-98b2-b8b0907168b4` | ACK PASS |

사용자가 전달한 SH2 ID
`01a00e5d-29e8-7a01-822b-7acf43226035`는 SH1 ID의 중복이었다. SH2의 실제
ID는 app host metadata와 server2 session direct ACK가 일치한
`01a00e5c-f7ae-72a2-98b2-b8b0907168b4`다.

## ACK repository snapshot

- SH1: detached `cdb80bc70032c203334531edb7020ff654f2938d`; tracked
  clean/untracked-only dirty; cached `origin/main` 대비 277 behind; ODE-edit
  Slurm job 0. Detached worktree는 보존하고 업데이트하지 않는다.
- SH2: clean `main` `6145406ae4b11e05b683c46aa604c972eb727f5a`;
  cached `origin/main` 대비 30 behind; ODE-edit Slurm job 0. Boundary PASS 뒤
  fetch/ff-only sync 대상이다.
- canonical `main` integration worktree:
  `/mnt/raid5/janghj/.codex/worktrees/odeedit-p2r7-main-publish-v1`.

## Boundary 판정

세 clone의 local-only boundary에는 이전 session generation ID가 남아 있었다.
또한 새 primary session의 displayed/runtime model confirmation은 ACK에서 독립
확정되지 않았다. 따라서 tracked registry는 required `Sol Ultra`와
confirmation pending을 사실대로 기록한다. 각 session은 model을 직접 확인한 뒤
자신의 ignored boundary만 갱신하고 checker를 통과해야 한다.

## 변경 범위

이번 reconciliation이 갱신하는 current control-plane 파일:

- `servers/connection-inventory.md`
- `servers/active/server1.md`
- `servers/active/server2.md`
- 이 메시지

과거 plans, audits, reports, receipts, result manifests와 completed experiment
launchers의 old session ID는 당시 실행 provenance 및 source lock이므로 치환하지
않는다. 새 instruction과 future launcher는 이 registry의 현재 session boundary를
사용해야 한다.
