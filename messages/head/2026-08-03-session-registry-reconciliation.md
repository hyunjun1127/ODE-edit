# 2026-08-03 Codex session registry 정합화

- 작성자: head-server1-gh
- coordination key: `ODE-EDIT-SH-HANDOFF-2026-08-03`
- 상태: registry assignment complete / onboarding HOLD

## 사용자 지정 canonical assignment

| 역할 | Codex session ID | 상태 |
| --- | --- | --- |
| GH | `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2` | active |
| SH1 / server1 | `019fc5e0-eb7e-78a3-9436-93885621b8dc` | assignment ACK / onboarding HOLD |
| SH2 / server2 | `019fc5ec-f85b-7770-a73a-1d19be1cd491` | assignment ACK / onboarding HOLD |

Tracked common registry와 `servers/active/server1.md`, `servers/active/server2.md`를 위
assignment로 맞췄다. Assignment는 즉시 Slurm/Git 실행 권한을 뜻하지 않는다. 각 SH는 자기
clone의 local session boundary, Git role identity, heartbeat와 onboarding audit를 통과해야 한다.
두 canonical SH session 모두 direct assignment ACK를 회신했다.

## Temporary implementation session 경계

`019fc63e-5217-7250-9c22-c5b2ec4248f0`은 현재 진행 중인
`ODEEDIT-S02-NUMLOCK-REVISION-V1`을 clean checkpoint와 completion report까지 닫는
task-local delegated implementation session이다. Canonical SH1이 아니며 새 task, GPU,
Slurm, push 권한이 없다. 완료 후 HOLD한다.

## 금지 및 다음 조치

- 다른 repo session ID로 대체하지 않는다.
- SH1은 GH root clone의 local session boundary를 덮어쓰지 않는다.
- SH2는 local method runtime/session boundary를 완료하기 전 실행하지 않는다.
- raw SSH/IP/user/credential은 tracked registry에 기록하지 않는다.
- GH는 onboarding gate가 닫히기 전 새로운 submit envelope를 발행하지 않는다.
