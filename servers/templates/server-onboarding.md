# 서버 온보딩 템플릿

## 요약

- 서버 이름:
- 서버 유형: lab/rental/cloud/other
- 사용 목적:
- 예상 사용 기간:
- 담당 server-head:
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
- memory:
- time limit:
- 주의할 quota/사용 정책:

## Subagent 준비

- Blue team 준비 상태:
- Red team 준비 상태:
- red-team onboarding audit 경로:

## 판정

- 상태: pending/pass/warn/block/waived
- 남은 작업:
- 다음 담당자:
