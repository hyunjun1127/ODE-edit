# Session 01 — Motivation Validation: Global Evidence Index

- 작성 시각: 2026-07-30
- 작성 agent: head-server1-gh (global-head)
- 상태: exact Llama 1-case MV-0 smoke 실행 전
- proposal link: `project/proposals/sections/01-motivation-validation.md`
- canonical plan: `plans/global/2026-07-30-session-01-motivation-validation.md`

## 질문

same-snapshot layer utility heterogeneity, partial-update 뒤 ranking
non-stationarity, sequential capacity concentration이 실제로 존재하는가?

## 실행 범위와 현재 결과

실험 result는 아직 없다. 다만 server1의 고정 runtime/resource/session
boundary, 두 model·CounterFact·기존 MEMIT moments·기존 AlphaEdit projector의
byte identity, read-only EasyEdit bridge, MV-0 fidelity runner, one-shot Slurm
envelope까지 준비했다.

첫 허용 실행은 `llama3-8b-inst`, case 1개,
`mv0_llama_smoke_v1` 한 건뿐이다. 이는 research signal이 아니라 native
EasyEdit MEMIT 대비 implementation fidelity와 trace neutrality smoke다. raw
evidence는 `local/`에만 남기고, 실행 agent와 다른 analysis/red agent가 compact
artifact만 검토한 뒤 이 문서에 승격한다.

## Claim boundary

현재 claim은 **hypothesis only**다. 향후 Session 01이 통과해도
`motivation diagnostic only`이며,
long-horizon superiority, novelty, causal mechanism, paper-ready claim을
뜻하지 않는다.

## Curated reports

- proposal/EasyEdit baseline audit:
  `audits/global/2026-07-30-proposal-easyedit-baseline-audit.md`
- exact one-shot execution preflight:
  `audits/global/2026-07-30-session01-mv0-execution-preflight.md`
- related work 및 novelty boundary:
  `project/proposals/sections/02-related-work-and-novelty-boundary.md`
- 실행 후 예정:
  `experiment-reports/global/2026-07-30-mv0-llama-smoke-analysis.md`
- 실행 후 예정:
  `audits/global/2026-07-30-mv0-llama-smoke.postrun.md`

## 현재 결정

- Proposal에서 온 내용: same-snapshot layer utility, partial-update
  non-stationarity, sequential load concentration은 검증할 가설이다.
- Repo/protocol에서 확인한 사실: 현재 별도 SH/peer clone은 없고,
  `PROTOCOL.md:506-509`의 명시적 사용자 지시 예외 아래 GH one-shot 제출만
  audit됐다.
- GH 추정: 첫 1-case smoke는 Motivation을 지지하거나 kill할 연구
  denominator가 아니다.
- 사용자 확인 필요: 없음. Qwen smoke, case 확장, MV-1은 현재 승인 범위 밖이며
  앞 run의 독립 보고 뒤 새 audit가 필요하다.

## Protocol ownership note

`PROTOCOL.md`의 experiment-section layout은
`experiment-reports/experiments/<section>/`을 권장하지만, GH role access
matrix와 `scripts/check-agent-access.sh`는 GH의 write를
`experiment-reports/global/`로 제한한다. 이 초기화에서는 access matrix를
우선했다. experiment-section index의 GH 소유권을 허용할지는 사용자 확인 후
protocol/template 변경으로 결정한다.
