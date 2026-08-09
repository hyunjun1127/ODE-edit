# 2026-08-09 ODE-Edit session registry 전환

- 기록 시각: 2026-08-09 12:36:25 KST
- SH2 revised ACK 반영: 2026-08-09 12:41:38 KST
- authority: 사용자 직접 지시
- repository: `hyunjun1127/ODE-edit`
- 작성 role: `global-head`

## Canonical session 전환

| Role | 이전 session | 새 canonical session | 상태 |
| --- | --- | --- | --- |
| GH / server1 | `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2` | `019fe491-16f4-7bd3-adf5-4b1eb4a57d1f` | root local boundary PASS; prior context inherited |
| SH1 / server1 | `019fc63e-5217-7250-9c22-c5b2ec4248f0` | `019fe489-c968-75f3-9965-7cfbc26c0a99` | direct ACK; local boundary PASS; ready for GH instruction |
| SH2 / server2 | `019fc5ec-f85b-7770-a73a-1d19be1cd491` | `019fe491-954b-70a0-8ba8-0588e9f8d741` | revised direct ACK; local boundary PASS; ready for GH instruction |

이전 session은 모두 superseded됐고 새 command authority가 없다. 이전 session ID가
기록된 plan, report, audit, message, source lock과 run artifact는 당시 실행 provenance로
유효하며 수정하지 않는다.

## Direct ACK 요약

### SH1

- runtime: `gpt-5.6-sol / Sol Ultra`
- CWD: `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit`
- effective identity: `head-server1-sh1 / server-head / server1`
- branch/HEAD: `codex/odeeditsh1-ct-k4-v1` /
  `cdb80bc70032c203334531edb7020ff654f2938d`
- tracked/index: clean; 기존 untracked run path는 사용자/이전 작업 소유로 보존
- R10-R4 evidence: readable and hash-verified
- action: `GPU0/Slurm0/model0/source0/commit0/push0/rsync0`

### SH2

- runtime: `gpt-5.6-sol / Sol Ultra`
- CWD: `/mnt/raid5/janghj/ODE-edit`
- branch/HEAD: `main` / `6145406ae4b11e05b683c46aa604c972eb727f5a`
- tracked/index: clean
- prior context/artifact: inherited/readable
- effective identity: `head-server2-sh2 / server-head / server2`
- local session boundary: PASS
- status: `READY_FOR_GH_INSTRUCTION`; blockers 없음
- authority: 새 GH instruction envelope 대기; 현재 source/model/GPU/Slurm/push/merge/rsync
  action 없음

## Consistency boundary

현재 권위값은 `servers/connection-inventory.md`, `servers/active/server1.md`,
`servers/active/server2.md`와 각 clone의 ignored local session boundary다. GH root clone은
새 GH session으로 boundary checker PASS를 확인했다. SH1은 자기 worktree에서 PASS를
확인했다. SH2도 server2 root clone의 local Git identity와 ignored boundary를 갱신하고
checker PASS를 확인했다.
