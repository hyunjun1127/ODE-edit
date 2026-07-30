# GH Naming and Session Pre-push Waiver

- 감사 유형: pre-push-sensitive / canonical naming update
- 판정: waived
- 감사 시간: 2026-07-30
- 작성 agent: head-server1-gh (global-head)
- 대상: ODE-Edit 명명과 Session 01 — Motivation Validation 전환

## Waiver 사유

독립 red-team session이 아직 미배정이다. GH는 이 변경을 repository control
plane의 이름·경로·instruction label 정합성으로 한정하며, red-team `pass`로
표현하지 않는다. 이 waiver는 Session 01 실험 실행, Slurm 제출, task 승인,
연구 claim 확정에는 적용되지 않는다.

## 확인 범위

- canonical 신규 문서, report, plan, script prefix에서 legacy stage label과
  이전 방법론 표기가 제거됐는지 확인했다.
- `project/proposals/00.proposal`은 수령 원문으로 보존하며, proposal README에
  current canonical name `ODE-Edit`과 원문 표기의 경계를 명시했다.
- 기존 tracked stage 경로는 Session 01 Motivation Validation 경로로 rename했고,
  inbound link가 새 경로를 가리키는지 확인한다.

## 다음 gate

Session 01의 실제 task/run/report 전에는 역할별 SH/red-team session이
onboarding audit을 수행해야 한다.
