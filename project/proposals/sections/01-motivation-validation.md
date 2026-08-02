# Session 01 — Motivation Validation: 최종 연구 판정

- 최종 갱신: 2026-08-02
- 상태: **closed — atomic possibility only; c1 `CAPACITY_HISTORY_HARM_SIGNAL`; no model-specific rescue**
- 현재 방법명: `ODE-Edit` (원 handoff의 `BF-ODE-Edit`은 원문 보존)

## 2026-08-02 c1 supersession

사용자 지시로 기존 MV-2 이후 atomic/direct-z와 constrained capacity/history
diagnostic을 추가 수행했다. c0의 fixed fractional progress와 weak terminal은 구현
오류로 판정해 폐기했고, c1 `capacity-qp-history-k4-native-progress-v2`로 다시
구현·실행했다.

- controller/evaluator 16개가 technical pass했다.
- QP-minus-native current utility는 Llama MEMIT `-4.032392`, Llama Alpha-history
  `-2.495123`, Qwen MEMIT `-0.397837`, Qwen Alpha-history `+0.029707`이다.
- pair passing family는 `[]`, verdict는 `CAPACITY_HISTORY_HARM_SIGNAL`이다.
- atomic/local relinearization possibility는 보존하지만 현재 fixed-`K=4`
  sequential controller와 method-superiority claim은 kill한다.
- 모델별 rescue, threshold/K/case retune, lifelong 확대는 열지 않는다.

Canonical c1 근거는
[`matched-quarter c1 spec`](../../../plans/global/2026-08-02-session01-capacity-history-matched-quarter-c1-spec.md)과
[`c1 pair synthesis`](../../../experiment-reports/global/2026-08-02-session01-caphist-pair-c1-v3-synthesis.md)다.
아래 2026-07-31 evidence chain은 초기 판정의 역사적 기록이며, 현재 최종 상태는 이
supersession을 따른다.

## 출발 가설과 proposal의 지위

### Proposal에서 온 내용

동일 direct-z에는 여러 parameter realization이 있을 수 있고, 여러
MEMIT/AlphaEdit layer proposal을 같은 model snapshot에서 비교·재배분한 뒤
partial write마다 proposal direction까지 다시 계산하면 one-shot 또는 static
allocation보다 나을 수 있다는 가설이었다.

`project/proposals/00.proposal`은 이 가설을 시작하기 위한 handoff다. 최종
paper plan, 확정된 방법, 검증된 성능 claim이 아니므로 Session 01은 작은
diagnostic에서 구현 fidelity → allocation signal → refreshed-direction
advantage 순서로 반증 가능하게 분해했다.

## Motivation evidence chain

| 질문 | 관측 | 판정 |
| --- | --- | --- |
| EasyEdit MEMIT hook이 native 동작을 보존하는가? | Llama/Qwen 각각 `3/3/3`, primary tensor/logit error `0`, rollback exact | MV-0 fidelity `PASS` |
| same-snapshot allocation signal이 held-out에서도 살아남는가? | MV-1 untouched mean: Llama `+0.0102628` (`20/20`), Qwen `+0.0483206` (`20/20`) | two-model allocation signal 재현 |
| partial update 뒤 direction refresh가 fixed direction보다 나은가? | Llama mean `-0.0044081`, sign `0/12`; Qwen mean `+0.0068874`, sign `7/12` | architecture-dependent sign reversal |
| refreshed direction+coefficient의 총 proxy가 fixed/fixed보다 나은가? | Llama mean `-0.0043044`, sign `0/12`; Qwen mean `+0.0085523`, sign `7/12` | cross-model total gain 불성립 |
| MV-3 matched-progress frontier를 열 수 있는가? | 사전등록 pair rule 1–6 불일치 후 rule 7 | `NO MV3` |

작은 보조 metric 하나 때문에 중단한 것이 아니다. Llama에서 primary
direction과 total이 mean, trimmed mean, median, sign, bootstrap CI 전체에서
일관되게 음수였고, `e_m=0.0001`보다 충분히 낮았다. Qwen의 양의 단일-model
신호가 이 반대 방향을 상쇄하지 않는다.

## 기대효과 판정

### Repo/protocol에서 확인한 사실

- MV-1 forecast는 양의 방향을 두 모델 모두 맞혔다. Pearson은 Llama
  `0.9271`, Qwen `0.8164`였다.
- aggregate magnitude prior는 Llama에서 약 `6.39%` 과대였지만 Qwen에서 약
  `56.86%` 과대였다. 따라서 Qwen prior `0.1120`은 후속 개선폭으로 재사용할
  수 없다.
- 현재 숫자로 허용되는 기대효과는 고정된 diagnostic의
  **teacher-forced absolute rewrite-utility proxy**뿐이다.
- full ODE-Edit의 accuracy, efficacy, locality, retention, long-horizon
  robustness 또는 EasyEdit baseline 대비 개선률은 산출되지 않았으며 현재
  예측할 수 없다.

안전한 요약은 다음과 같다.

> allocation controller 자체의 local signal은 두 모델에서 관측됐지만,
> proposal direction을 state마다 refresh하는 ODE/relinearization의
> cross-model 기대효과는 0보다 안정적으로 크다고 예측할 수 없다. 현재
> evidence에서는 Llama에 해롭고 Qwen에만 유리하다.

## Kill 범위와 남는 가설

### GH 추정

- **Kill:** 현재 설계의 cross-model refreshed-direction ODE/relinearization
  mechanism, 이를 전제로 한 MV-3/MV-4, ODE 일반성, capacity/retention
  improvement narrative.
- **보존:** MV-1이 보인 same-snapshot allocation signal.
- **새 proposal로만 검토 가능:** fixed-direction dynamic coefficient,
  Qwen-specific refreshed-direction controller, 또는 static capacity-aware
  routing. 이들은 현재 ODE-Edit 방법의 성공이나 자동 후속 단계가 아니다.
- 추가 fold, threshold 변경, retuning 또는 post-hoc model pooling으로
  MV-2를 구제하지 않는다.

## 사용자 확인 필요

- Session 01 종결과 `NO MV3`에는 추가 확인이 필요 없다.
- 이후 연구를 재개하려면 다음 중 하나를 **새 독립 방향**으로 선택해야 한다:
  fixed-direction coefficient controller, static routing, 또는
  architecture-specific Qwen track.
- paper method name은 `ODEdit`, `ODESteer`, `ODE-M`과 충돌하므로 새 방향이
  생존한 뒤 명칭 유지 여부를 다시 확인해야 한다. Repo의 현재 이름
  `ODE-Edit`은 사용자 지시대로 유지한다.

## Canonical evidence

- 실행 계획:
  [`plans/global/2026-07-30-session-01-motivation-validation.md`](../../../plans/global/2026-07-30-session-01-motivation-validation.md)
- 최종 GH 보고서:
  [`experiment-reports/global/2026-07-31-session-01-motivation-final.md`](../../../experiment-reports/global/2026-07-31-session-01-motivation-final.md)
- MV-0 pair red:
  [`audits/global/2026-07-30-mv0-pair-c3-v3.postrun.md`](../../../audits/global/2026-07-30-mv0-pair-c3-v3.postrun.md)
- MV-1 untouched pair:
  [`experiment-reports/global/2026-07-31-mv1mix-untouched-pair-v1-analysis.md`](../../../experiment-reports/global/2026-07-31-mv1mix-untouched-pair-v1-analysis.md)
- MV-2 Llama/Qwen:
  [`Llama`](../../../experiment-reports/global/2026-07-31-mv2refresh-llama-e0-v2-analysis.md),
  [`Qwen`](../../../experiment-reports/global/2026-07-31-mv2refresh-qwen-e0-v2-analysis.md)
- MV-2 pair red:
  [`audits/global/2026-07-31-mv2refresh-pair-v2.postrun.md`](../../../audits/global/2026-07-31-mv2refresh-pair-v2.postrun.md)
- 관련 연구:
  [`project/proposals/sections/02-related-work-and-novelty-boundary.md`](02-related-work-and-novelty-boundary.md)
