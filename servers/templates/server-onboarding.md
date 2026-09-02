# 서버 온보딩 템플릿

## 요약

- 서버 이름:
- 서버 유형: lab/rental/cloud/other
- 사용 목적:
- 예상 사용 기간:
- 담당 server-head:
- Codex session registry (role별 별도 행):

| 역할 | Codex session ID | Observed model (optional) | Codex session CWD | 상태 |
| --- | --- | --- | --- | --- |
| server-head |  | user-managed /  |  | pending/active/retired |

Model family와 reasoning effort는 사용자가 관리한다. 관측값은 진단용으로 기록할
수 있지만, user task contract가 별도 고정하지 않는 한 session authority를
결정하지 않는다. Subagent도 parent task의 경계를 따르며 server-head session
authority를 대체할 수 없다.
- global-head 승인:

## 접근과 권한

- 사용자 계정:
- SSH 접속 확인:
- host key 확인:
- Slurm 접근 확인:
- storage mount 확인:
- rsync 인증 상태: 사용자가 직접 확인

비밀번호, private key, token은 절대 기록하지 않는다.

## Git/Agent 설정

- repo clone 경로:
- `agent.id`:
- `agent.role`:
- `agent.hostname`:
- repository identity: GitHub `owner/repo` + expected `origin` URL
- heartbeat 경로:
- hourly sync 설정:

## Local 경로

- dataset 경로:
- output 경로:
- checkpoint 경로:
- log 경로:
- scratch 경로:

## Slurm/Resource

- partition:
- GPU:
- CPU:
- memory: 모든 batch allocation에 `--mem`을 명시한다. GPU 1개당 요청량은
  `servers/slurm-memory-policy.tsv`와 `servers/local/gpu-caps.tsv` 중 더 낮은
  한도를 적용한다. 현재 repo 안전 한도는 server1/2/3/4 각각
  179/59/119/59 GiB이다.
- time limit:
- 주의할 quota/사용 정책: `--mem` 누락, GPU cap 또는 memory cap unknown,
  tracked ceiling 초과 중 하나라도 있으면 submit 금지

## Subagent 준비

- Blue team 준비 상태:
- Red team 준비 상태:
- red-team onboarding audit 경로:

## 판정

- 상태: pending/pass/warn/block/waived
- 남은 작업:
- 다음 담당자:
