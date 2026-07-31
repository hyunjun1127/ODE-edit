# server1 onboarding record

## 요약

- 서버 이름: server1
- 서버 유형: local GH host
- 사용 목적: ODE-Edit repository bootstrap 및 Session 01 preflight 후보
- 예상 사용 기간: 사용자 확인 필요
- 담당 global-head: head-server1-gh
- 담당 server-head: 미지정
- global-head 승인: `pending-onboarding`

| 역할 | Codex session ID | Required/confirmed Codex model | Codex session CWD | 상태 |
| --- | --- | --- | --- | --- |
| global-head | `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2` | `Terra Ultra` / `Terra Ultra` (`gpt-5.6-terra`, runtime metadata) | `/mnt/raid5/janghj/ODE-edit` | active |
| server-head | 미지정 | `Terra Ultra` / 미지정 | `/mnt/raid5/janghj/ODE-edit` | pending assignment |

## 접근과 권한

- 사용자 계정: Git 기록 금지
- SSH 접속 확인: local-only `servers/local/ssh_config`를 통한 read-only check 필요
- host key 확인: 사용자 또는 향후 server-head 확인 필요
- Slurm 접근 확인: controller read-only visibility만 확인됨; job submit은 미승인
- storage mount 확인: `/mnt/raid5/janghj/ODE-edit` local clone 존재 확인
- rsync 인증 상태: local-only config 이식 후 dry-run으로 확인 필요
- method runtime: `servers/local/method-runtime.env`에 기존 EasyEdit/Hugging
  Face cache compatibility 설정을 local-only로 적용

비밀번호, private key, token, raw HostName/IP, username, port, private dataset
secret은 절대 기록하지 않는다.

## Git/Agent 설정

- repo clone 경로: `/mnt/raid5/janghj/ODE-edit`
- repository identity: `hyunjun1127/ODE-edit` (remote 등록 후 검증)
- `agent.id`: `head-server1-gh`
- `agent.role`: `global-head`
- `agent.hostname`: `server1`
- heartbeat 경로: `agents/server1/head-server1-gh.json`
- sync 설정: GitHub remote와 main bootstrap push 완료; GH가 설치 여부를 결정

## Codex Session Boundary

이 record의 command/message는 역할별 registry의 Codex session ID,
confirmed `Terra Ultra` model profile, CWD를 확인한 session만 수행한다.
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
- GPU cap: 동시 최대 `3` GPU (`servers/local/gpu-caps.tsv`)
- GPU memory request cap: GPU 1개당 최대 `198117 MiB`
- CPU: task별 명시 필요
- time limit: task별 명시 필요
- 주의할 quota/사용 정책: `scripts/check-slurm-gpu-cap.sh` 및
  `scripts/check-slurm-resource-cap.sh`가 통과하고 red pre-flight가 `pass`
  또는 기록된 `waived`인 경우에만 submit 가능

## Subagent 준비

- Blue team 준비 상태: server-head 배정 후 필요
- Red team 준비 상태: server-head 배정 후 필요
- red-team onboarding audit 경로:
  `audits/servers/server1/session01-motivation-onboarding.preflight.md`

## 판정

- 상태: `pending-onboarding`
- 남은 작업: local SSH/rsync dry-run, runtime/dataset
  path 확인, server-head 배정 여부, red-team onboarding audit
- 다음 담당자: global-head 및 사용자
