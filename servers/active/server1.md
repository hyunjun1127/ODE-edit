# server1 onboarding record

## 요약

- 서버 이름: server1
- 서버 유형: local GH host
- 사용 목적: ODE-Edit repository bootstrap 및 Session 01 Motivation 실행 host
- 예상 사용 기간: 사용자 확인 필요
- 담당 global-head: head-server1-gh
- 담당 server-head: head-server1-sh1 (canonical SH1 session assigned)
- global-head 승인: `active-local-gh-exception`; SH1은 dedicated worktree에서 active

| 역할 | Codex session ID | Required/confirmed Codex model | Codex session CWD | 상태 |
| --- | --- | --- | --- | --- |
| global-head | `019fe491-16f4-7bd3-adf5-4b1eb4a57d1f` | `Sol Ultra` / `Sol Ultra` (`gpt-5.6-sol`, runtime metadata) | `/mnt/raid5/janghj/ODE-edit` | active; prior context inherited |
| server-head (SH1) | `019fe489-c968-75f3-9965-7cfbc26c0a99` | `Sol Ultra` / `Sol Ultra` (`gpt-5.6-sol`, runtime metadata) | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | active canonical SH1; prior context inherited; boundary PASS; ready for GH instruction |

사용자 정정에 따라 이전 assignment `019fc5e0-eb7e-78a3-9436-93885621b8dc`는
superseded되었으며 server1 SH authority가 없다. 현재 SH1은 instruction
`ODEEDIT-S02-P1-SMAX8-AFFECTED-CONT-LLAMA-V1`까지 완료하고 다음 GH envelope를 기다린다.

2026-08-09 사용자 직접 session 교체에 따라 이전 GH
`019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`와 이전 SH1
`019fc63e-5217-7250-9c22-c5b2ec4248f0`은 superseded됐으며 새 command authority가
없다. 새 SH1은 이전 대화와 R10-R4 evidence context를 인계받아 hash 확인을 완료했다.

## 접근과 권한

- 사용자 계정: Git 기록 금지
- SSH 접속 확인: local-only `servers/local/ssh_config`를 통한 read-only check 필요
- host key 확인: 사용자 또는 향후 server-head 확인 필요
- Slurm 접근 확인: 사용자 time-critical 지시와 protocol 예외 아래 GH helper
  submit/monitor/terminal artifact 검증 완료
- storage mount 확인: `/mnt/raid5/janghj/ODE-edit` local clone 존재 확인
- rsync 인증 상태: local-only config 이식 후 dry-run으로 확인 필요
- method runtime: `servers/local/method-runtime.env`에 기존 EasyEdit/Hugging
  Face cache compatibility 설정을 local-only로 적용

비밀번호, private key, token, raw HostName/IP, username, port, private dataset
secret은 절대 기록하지 않는다.

## Git/Agent 설정

- repo clone 경로: `/mnt/raid5/janghj/ODE-edit`
- repository identity: `hyunjun1127/ODE-edit` (remote 등록 후 검증)
- root clone의 현재 `agent.id`: `head-server1-gh`
- root clone의 현재 `agent.role`: `global-head`
- root clone의 현재 `agent.hostname`: `server1`
- GH heartbeat 경로: `agents/server1/head-server1-gh.json`
- SH1 worktree: `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit`
- SH1 branch: `codex/odeeditsh1-ct-k4-v1`
- SH1 effective identity: `agent.id=head-server1-sh1`, `agent.role=server-head`,
  `agent.hostname=server1`
- SH1 session/role boundary: PASS (`019fe489-c968-75f3-9965-7cfbc26c0a99`)
- SH1 heartbeat 경로: `agents/server1/head-server1-sh1.json` (tracked heartbeat는 pending)
- SH1은 GH root clone의 local session boundary를 덮어쓰지 않는다.
- sync 설정: GitHub remote와 main bootstrap push 완료; GH가 설치 여부를 결정

## Codex Session Boundary

이 record의 command/message는 역할별 registry의 Codex session ID,
confirmed `Sol Ultra` primary model profile, CWD를 확인한 session만 수행한다.
`knowledge-revision` 등 다른 repository CWD 또는 그 session ID를 대상으로
message, shell command, artifact transfer, task claim을 실행하지 않는다. 새
GH/SH session은 이 record를 갱신하고, 기존 session ID를 재사용하지 않는다.

## Local 경로

- dataset 경로: `local/datasets/` 또는 local-only runtime config
- output 경로: `local/results/raw/`
- checkpoint 경로: `local/checkpoints/`
- log 경로: `local/logs/`
- scratch 경로: `local/`

## Slurm/Resource

- partition: local-only Slurm query로 확인 필요
- GPU cap: 동시 최대 `4` GPU (`servers/local/gpu-caps.tsv`); 2026-08-09
  사용자 직접 지시로 AlphaEdit realization 실험의 병렬 실행을 위해 `4` GPU로
  증액했다. 제출 직전에는 기존 project job을 포함한 실제 합계가 `4` 이하인지 다시
  확인한다.
- GPU memory request cap: GPU 1개당 최대 `198117 MiB`
- CPU: task별 명시 필요
- time limit: task별 명시 필요
- 주의할 quota/사용 정책: `scripts/check-slurm-gpu-cap.sh` 및
  `scripts/check-slurm-resource-cap.sh`가 통과하고 red pre-flight가 `pass`
  또는 기록된 `waived`인 경우에만 submit 가능

## Subagent 준비

- Blue team 준비 상태: SH1 onboarding 뒤 필요
- Red team 준비 상태: SH1 onboarding 뒤 필요
- red-team onboarding audit 경로:
  `audits/servers/server1/session01-motivation-onboarding.preflight.md`

## 판정

- 상태: `active-local-gh`; canonical SH1 active on dedicated worktree; Session 02 P1 R2와
  prelocked `S_max=8` continuation 완료, GPU/Slurm/push HOLD
- 완료: repository/remote/session/resource boundary, EasyEdit runtime,
  pinned dataset/cache, Slurm pair 실행과 Session 01 Motivation closure
- 남은 작업: GH final integration, common-controller redesign, heartbeat, local SSH/rsync
  dry-run, peer artifact broadcast 검증, red-team execution preflight
- 다음 담당자: canonical SH1과 global-head
