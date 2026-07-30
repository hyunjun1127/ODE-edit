# GH 초기화 공지 — BF-ODE-Edit Stage 0

- 작성 시각: 2026-07-30
- 작성 agent: head-server1-gh (global-head)
- 상태: `info / execution blocked pending onboarding`
- 관련 plan: `plans/global/2026-07-30-stage0-diagnostic.md`
- 관련 proposal: `project/proposals/00.proposal`

## 결정

이 repo는 기존 실험의 후속 patch가 아니라 BF-ODE-Edit 독립 연구 방향의
초기화 상태로 판정한다. proposal은 canonical research input으로 보존하되,
현재 claim은 `hypothesis only`다. Stage 0는 method 성능이 아니라
same-snapshot utility heterogeneity, partial-update ranking non-stationarity,
sequential capacity concentration의 최소 신호를 검사한다.

## 확인된 운영 상태

- server1에는 GH clone이 있고, 별도 server-head inbox, task, run, audit는 없다.
- tracked `project/run_scripts/`에는 실행 script가 없다.
- raw connection 값과 GPU cap은 tracked inventory에 없으며 local-only
  inventory가 필요하다.
- GH clone에서 Slurm controller 응답은 확인했으나, project GPU cap과 target
  server ownership이 없어 제출 권한은 생기지 않는다.
- server1의 Codex session ID/CWD/repository identity를 기록하고, 다른 repo
  session의 command·artifact·message target 사용을 금지한다.
- 따라서 이 공지는 실행 명령이 아니고, 등록 전 SH envelope 초안은 global
  plan에만 있다.

## 다음 행동

private research remote `hyunjun1127/ODE-edit`를 등록했고, 첫 bootstrap push를
준비 중이다. server1/4 onboarding, red-team gate, server별 actionable inbox는
SH session이 등록된 뒤에 순서대로 발행한다. 그 전에는 Slurm/SSH/rsync 실행을
하지 않는다.
