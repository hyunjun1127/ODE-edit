# server2 onboarding record

## 요약

- 서버 이름: server2
- 서버 유형: registered remote lab host
- 사용 목적: ODE-Edit locked-source reproduction 및 향후 SH2 실행
- 예상 사용 기간: 사용자 확인 필요
- 담당 global-head: head-server1-gh
- 담당 server-head: head-server2-sh2 (canonical SH2 session assigned)
- global-head 승인: `session-registered-ready-for-instruction`

| 역할 | Codex session ID | Required/confirmed Codex model | Codex session CWD | 상태 |
| --- | --- | --- | --- | --- |
| server-head (SH2) | `019fe491-954b-70a0-8ba8-0588e9f8d741` | `Sol Ultra` / `Sol Ultra` (`gpt-5.6-sol`, runtime metadata) | `/mnt/raid5/janghj/ODE-edit` | active canonical SH2; prior context inherited; boundary PASS; ready for GH instruction |

2026-08-09 사용자 직접 session 교체에 따라 이전 SH2
`019fc5ec-f85b-7770-a73a-1d19be1cd491`은 superseded됐으며 새 command authority가 없다.
새 SH2는 이전 대화와 artifact context를 인계받았고, local-only Git identity/session
boundary provisioning 및 checker 검증을 완료했다.

## 접근과 권한

- 사용자 계정과 raw SSH 정보: Git 기록 금지
- repository identity: `hyunjun1127/ODE-edit` 확인
- repo clone: `/mnt/raid5/janghj/ODE-edit` 확인
- initial ACK 당시 branch/head/worktree: `main` /
  `6145406ae4b11e05b683c46aa604c972eb727f5a` / clean
- Slurm command와 server2 node: available로 확인
- method runtime: prior context inherited; 실제 task별 envelope preflight 필요
- session boundary: `servers/local/session-boundary.env` 갱신 및 checker PASS
- rsync/SSH artifact path: local-only config와 dry-run 검증 전까지 사용 금지

비밀번호, private key, token, raw HostName/IP, username, port, private dataset
secret은 이 record에 기록하지 않는다.

## Git/Agent 설정

- expected `agent.id`: `head-server2-sh2`
- expected `agent.role`: `server-head`
- expected `agent.hostname`: `server2`
- heartbeat 경로: `agents/server2/head-server2-sh2.json` (SH2 작성 전까지 pending)
- effective identity: `agent.id=head-server2-sh2`, `agent.role=server-head`,
  `agent.hostname=server2`
- sync: 별도 GH envelope와 local peer 검증 뒤에만 실행

## Codex Session Boundary

server2의 ODE-Edit command는 canonical session
`019fe491-954b-70a0-8ba8-0588e9f8d741`, confirmed `Sol Ultra`, CWD
`/mnt/raid5/janghj/ODE-edit`, repository identity와 local Git identity가 모두 일치한
뒤에만 실행한다. 다른 repo 또는 다른 server session은 대체할 수 없다.

## Local 경로

- dataset 경로: `local/datasets/` 또는 local-only runtime config
- output 경로: `local/results/raw/`
- checkpoint 경로: `local/checkpoints/`
- log 경로: `local/logs/`
- scratch 경로: `local/`

## Slurm/Resource

- GPU cap: 동시 최대 `3` GPU (`servers/local/gpu-caps.tsv`)
- GPU memory request cap: GPU 1개당 최대 `66017 MiB`
- partition/CPU/time limit: SH2 local-only Slurm preflight에서 확인
- submit policy: local session boundary, method runtime, heartbeat와 red-team onboarding
  audit가 닫히고 GH full envelope가 발행되기 전까지 금지

## Subagent 준비

- Blue team 준비 상태: onboarding 뒤 필요
- Red team 준비 상태: onboarding 뒤 필요
- red-team onboarding audit: 미작성
- subagent runtime: 사용 시 `Terra Ultra`만 허용

## 판정

- 상태: `ready-for-gh-instruction`
- 완료: 새 canonical SH2 session 등록, repo/context 확인, local Git identity와 session
  boundary checker PASS
- blocker: registration 기준 없음; task별 method/runtime/resource gate는 향후 envelope에서 확인
- 현재 권한: 새 GH instruction envelope 대기; 별도 지시 없는 Git write, Slurm, rsync,
  artifact 작업 금지
- 다음 담당자: canonical SH2와 global-head
