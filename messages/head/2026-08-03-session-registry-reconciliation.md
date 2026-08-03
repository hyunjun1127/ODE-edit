# 2026-08-03 Codex session registry 정합화

- 작성자: head-server1-gh
- coordination key: `ODE-EDIT-SH-HANDOFF-2026-08-03`
- 상태: user correction applied / SH1 active / SH2 onboarding HOLD

## 사용자 지정 canonical assignment

| 역할 | Codex session ID | 상태 |
| --- | --- | --- |
| GH | `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2` | active |
| SH1 / server1 | `019fc63e-5217-7250-9c22-c5b2ec4248f0` | active canonical SH1; GPU/Slurm/push HOLD |
| SH2 / server2 | `019fc5ec-f85b-7770-a73a-1d19be1cd491` | assignment ACK / onboarding HOLD |

Tracked common registry와 `servers/active/server1.md`, `servers/active/server2.md`를 위
assignment로 맞췄다. SH1의 dedicated worktree session/role boundary는 PASS이며 현재
numerical-lock revision을 수행한다. SH2 assignment는 즉시 Slurm/Git 실행 권한을 뜻하지
않으며 local boundary와 onboarding audit 전까지 HOLD다.

## 사용자 정정

최초 등록된 `019fc5e0-eb7e-78a3-9436-93885621b8dc` SH1 assignment는 사용자 정정으로
superseded되었다. `019fc63e-5217-7250-9c22-c5b2ec4248f0`이 server1의 canonical SH1이며
`ODEEDIT-S02-NUMLOCK-REVISION-V1` 소유권을 유지한다. Numerical lock 승인 전까지 GPU,
Slurm과 push는 HOLD다.

## 금지 및 다음 조치

- 다른 repo session ID로 대체하지 않는다.
- SH1은 dedicated worktree boundary를 사용하며 GH root clone의 local boundary를 덮어쓰지 않는다.
- SH2는 local method runtime/session boundary를 완료하기 전 실행하지 않는다.
- raw SSH/IP/user/credential은 tracked registry에 기록하지 않는다.
- GH는 onboarding gate가 닫히기 전 새로운 submit envelope를 발행하지 않는다.
