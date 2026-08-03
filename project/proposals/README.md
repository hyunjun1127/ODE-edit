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
  Session 01 mechanism chain과 C3 최종 Motivation closure
- [`sections/02-related-work-and-novelty-boundary.md`](sections/02-related-work-and-novelty-boundary.md):
  layer allocation, sequential regularization, ODE prior와 baseline/novelty 경계
- [`sections/03-direct-z-review-and-motivation-closure-design.md`](sections/03-direct-z-review-and-motivation-closure-design.md):
  direct-z atomic/de-bundling의 historical design과 별도 scope
- [`sections/04-method-design.md`](sections/04-method-design.md):
  constrained layer-synchronous ODE-Edit controller와 common strong pilot 진입 계약

Session 01의 최종 판정은
**`CLOSED_DIRECTIONAL_POSITIVE; STRONG_METHOD_GATE_FAIL`**이다. C1의
`CAPACITY_HISTORY_HARM_SIGNAL`과 C2 bundled-share verdict는 C3에 의해 supersede됐다.
C3는 BF relative share와 global magnitude의 분리가 필요함을 Llama/Qwen과
MEMIT/Alpha에 걸쳐 확인했지만, deployable method superiority를 확립하지 않았다.

Motivation 내부 추가 rescue/retune은 닫혔다. Method Session의 primary track은
model-common monotone-load controller, adaptive trust/rollback, first-hit, Scalar/Static/
Ordered baseline과 ODE necessity 비교에 한해 열린다. Accepted round 수는 사전 평균값이
아니라 trajectory statistic이며, performance와 `N_field`/GPU time frontier를 co-primary로
판정한다. Canonical 실행 spec은
[`Session 02 compute-aware main-table spec`](../../plans/global/2026-08-03-session02-compute-aware-main-table-spec.md)이다.
Signed-capacity hard barrier와 AlphaEdit은 첫 MEMIT main table 뒤 ablation/extension으로
미룬다. Strong pilot 전 large/lifelong execution, model-specific rescue,
preservation/capability guarantee는 허용하지 않는다. 최종 Motivation decision report는
[`2026-08-03 Session 01 closure`](../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)다.
