# server4 onboarding record

## 요약

- 서버 이름: server4
- 서버 유형: registered remote host
- 사용 목적: 향후 ODE-Edit Session 01/재현 실행 후보
- 예상 사용 기간: 사용자 확인 필요
- 담당 global-head: head-server1-gh
- 담당 server-head: 미지정
- global-head 승인: `registered-pending-clone`

| 역할 | Codex session ID | Required/confirmed Codex model | Codex session CWD | 상태 |
| --- | --- | --- | --- | --- |
| server-head | 미지정 | `Sol Ultra` / 미지정 | `/data/janghj/ODE-edit` | pending clone and assignment |

## Git/Agent 설정

- 예상 repo clone 경로: local-only `servers/local/rsync-targets.tsv`의
  `server4` row와 일치해야 함
- repository identity: `hyunjun1127/ODE-edit`
- `agent.id`, `agent.role`, `agent.hostname`: clone과 SH session 생성 후 설정
- heartbeat 경로: `agents/server4/<server-head-agent-id>.json`

## Codex Session Boundary

server4에서 이 repo 작업은 registry의 전용 Codex session ID, confirmed
`Sol Ultra` primary model profile, 해당 `ODE-edit` clone CWD, Git identity
`hyunjun1127/ODE-edit`가 `servers/active/server4.md`와
`servers/local/session-boundary.env`에 모두 기록된 뒤에만 시작한다. 다른
repo session 또는 CWD로 SSH/Slurm/rsync/Git 작업을 수행하지 않는다.

## Slurm/Resource

- GPU cap: 동시 최대 `3` GPU (`servers/local/gpu-caps.tsv`)
- GPU memory request cap: GPU 1개당 최대 `65984 MiB` (server3와 동일)
- partition/CPU/time limit: server4 SH가 local-only Slurm query로 확인 필요
- submit policy: repo clone, session boundary, red-team onboarding audit,
  `scripts/check-slurm-resource-cap.sh` pass 전까지 금지

## 판정

- 상태: `registered-pending-clone`
- 남은 작업: repo clone, local SSH/rsync config 확인, Codex SH session ID/CWD
  등록, heartbeat, red-team onboarding audit
- 다음 담당자: global-head 및 향후 server4 server-head
