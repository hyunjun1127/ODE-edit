# GH Bootstrap Pre-push Waiver

- 감사 유형: pre-push-sensitive / bootstrap
- 판정: waived
- 감사 시간: 2026-07-30
- 작성 agent: head-server1-gh (global-head)
- 대상: BF-ODE-Edit 첫 GitHub push

## Waiver 사유

현재 server1에는 global-head Codex session만 있고 독립 server-head 및 red-team
session이 아직 없다. 따라서 red-team `pass`를 가장해 기록하지 않는다. 대신
첫 bootstrap commit 범위에 한해 GH가 protocol상 waiver를 기록한다. 이 waiver는
Stage 0 실행, Slurm 제출, experiment report claim, task 승격에는 적용되지 않는다.

## 확인 범위

- `scripts/check-agent-access.sh --staged` path ownership pass
- `git diff --cached --check` whitespace pass
- tracked paths에 `servers/local/`, `local/`, SSH key, token, credential,
  dataset, checkpoint, raw log가 없는지 staged filename/content scan
- session boundary helper가 server1의 registered session ID/CWD/origin을 pass
- resource helper가 server1의 1 GPU / 198117 MiB request를 cap 이내로 판정

## 제한과 다음 gate

- 이 감사는 코드 실행 결과나 연구 claim을 검증하지 않는다.
- server1 또는 server4에 SH session이 생기면 onboarding pre-flight red audit을
  새로 수행한다.
- 실행 instruction, Slurm submission, artifact broadcast는 해당 audit과
  session boundary pass 전까지 `block`이다.
