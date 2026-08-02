# Project Proposals

이 디렉터리는 user proposal과 GH research handoff를 보관한다. proposal은
canonical research input이지만 final paper plan이나 검증된 claim이 아니다.

## Naming note

`00.proposal`은 수령 당시의 원문을 보존하므로 `BF-ODE-Edit` 표기를 포함한다.
사용자 결정에 따라 이 repo의 현재 canonical 방법론 이름은 **ODE-Edit**이며,
첫 진행 단위는 **Session 01 — Motivation Validation**이다. 원문 proposal의
표기를 과거 handoff로 취급하고, 신규 plan, task, report, run script, session
명명에서는 `ODE-Edit`만 사용한다.

GH는 proposal을 읽은 뒤 목적이 명확한 research session plan, kill criterion,
next-session criterion, server-head instruction envelope를 `PROTOCOL.md`에 따라
작성한다.

## 현재 rationale section

- [`sections/01-motivation-validation.md`](sections/01-motivation-validation.md):
  Session 01의 mechanism chain, 2026-07-31 판정과 2026-08-02 c1 supersession
- [`sections/02-related-work-and-novelty-boundary.md`](sections/02-related-work-and-novelty-boundary.md):
  layer allocation, sequential regularization, ODE prior와 baseline/novelty 경계

Session 01은 c1에서 `CAPACITY_HISTORY_HARM_SIGNAL`, passing family `[]`로 최종
종료됐다. Atomic/local possibility는 보존하지만 현재 sequential controller는
cross-model negative다. 후속 fixed-direction coefficient, exact-length control,
static allocation 또는 architecture-specific track은 기존 ODE-Edit 성공의 자동
다음 단계가 아니라 새 proposal로 시작해야 한다.
