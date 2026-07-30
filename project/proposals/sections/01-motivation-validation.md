# Session 01 — Motivation Validation: ODE-Edit 진단 근거와 claim boundary

## 연구 입력과 현재 상태

- **proposal에서 온 내용:** 동일 direct-z에 여러 parameter realization이
  있을 수 있고, layer별 MEMIT/AlphaEdit proposal을 same snapshot에서 비교해
  capacity-aware하게 재배분하면 long-horizon editing의 trade-off가 개선될 수
  있다는 가설이다.
- **repo/protocol에서 확인한 사실:** 현재 repository에는 실행 script, active
  server, run record, server report, audit result가 없다. `PROTOCOL.md`는
  Session 01 실행 전에 server onboarding, GPU cap, pre-flight red-team audit,
  artifact broadcast와 Korean evidence record를 요구한다.
- **GH 추정:** ODE라는 표현을 유지하려면 단순 update split이나 static
  layer-wise scaling으로 설명되지 않는 state-dependent signal이 먼저 있어야
  한다.
- **사용자 확인 필요:** 실제 remote, 사용할 server/model/dataset mount,
  baseline implementation revision, GPU cap과 실행 예산은 아직 제공되지 않았다.

## Motivation Validation 질문

Session 01은 방법의 성능을 보이는 단계가 아니다. proposal의 motivation이
정당한지 다음 세 mechanism signal이 MEMIT instrumentation에서 관측되는지만
판정한다.

1. 동일 model snapshot에서 layer utility가 의미 있게 서로 다른가?
2. 허용된 rewrite context만 사용한 작은 joint partial update 뒤 utility
   ranking 또는 proposal direction이 변하는가?
3. sequential baseline의 cumulative capacity load가 layer에 집중되는가?

이 세 질문의 답이 모두 약하면 ODE-Edit의 ODE/relinearization claim은
유지하지 않으며, static capacity routing 또는 연구 중단으로 전환한다.

## 증거 연결

- 실행 계획과 사전 판정: [`plans/global/2026-07-30-session-01-motivation-validation.md`](../../../plans/global/2026-07-30-session-01-motivation-validation.md)
- Session 01 global evidence index: [`experiment-reports/global/2026-07-30-session-01-motivation-validation.md`](../../../experiment-reports/global/2026-07-30-session-01-motivation-validation.md)
- 원 handoff: [`project/proposals/00.proposal`](../00.proposal)

## Claim 상태

**hypothesis only.** `motivation supported`는 Session 01의 사전 기준을 통과한
것만 뜻하며, long-horizon superiority, novelty, causal mechanism 또는
paper-ready claim을 뜻하지 않는다.
