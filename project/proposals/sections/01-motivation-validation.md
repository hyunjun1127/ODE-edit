# Session 01 — Motivation Validation: ODE-Edit 진단 근거와 claim boundary

## 연구 입력과 현재 상태

- **proposal에서 온 내용:** 동일 direct-z에 여러 parameter realization이
  있을 수 있고, layer별 MEMIT/AlphaEdit proposal을 same snapshot에서 비교해
  capacity-aware하게 재배분하면 long-horizon editing의 trade-off가 개선될 수
  있다는 가설이다.
- **repo/protocol에서 확인한 사실:** repository remote와 server1/server4
  resource policy는 등록돼 있으나 별도 SH와 completed run은 없다.
  `PROTOCOL.md`는 Session 01 실행 전에 resource-cap check, pre-flight
  red-team audit, local artifact boundary와 Korean evidence record를 요구한다.
  proposal/EasyEdit 재감사는
  [`audits/global/2026-07-30-proposal-easyedit-baseline-audit.md`](../../../audits/global/2026-07-30-proposal-easyedit-baseline-audit.md)에
  기록했다.
- **GH 추정:** ODE라는 표현을 유지하려면 단순 update split이나 static
  layer-wise scaling으로 설명되지 않는 state-dependent signal이 먼저 있어야
  한다.
- **repo/protocol에서 확인한 사실:** 고정 모델은
  `Meta-Llama-3-8B-Instruct`와 `Qwen2.5-7B-Instruct`이고 local snapshot,
  CounterFact, 두 모델의 precomputed Wikipedia covariance와 AlphaEdit
  projector가 존재한다. EasyEdit는 commit `3488a66`의 dirty worktree이므로
  imported-file hash를 run마다 별도로 고정해야 한다.
- **사용자 확인 필요:** 별도 server-head Codex session은 아직 등록되지 않았다.

## Motivation Validation 질문

Session 01은 방법의 성능을 보이는 단계가 아니다. proposal의 motivation이
정당한지 다음 mechanism chain이 MEMIT instrumentation에서 관측되는지
판정한다.

1. custom hook이 canonical EasyEdit MEMIT을 보존하는가?
2. 동일 snapshot의 analytic layer utility가 actual finite-step progress
   차이를 예측하는가?
3. joint partial update 뒤 refreshed decision이 stale decision보다 실제로
   나은가?
4. layer 간 allocation이 matched rewrite progress에서 global/static
   scaling보다 낮은 displacement frontier를 만드는가?
5. 짧은 sequential intervention에서 그 displacement 차이가 retention/locality
   proxy와 연결되는가?

단순 utility CV, rank turnover, load Gini는 descriptive metric일 뿐 pass
criterion이 아니다. 2가 실패하면 routing motivation을 kill하고, 2는
통과하지만 3이 실패하면 static allocation으로 pivot한다. 3은 통과하지만
4가 실패하면 capacity-free dynamic scheduler로 claim을 낮추며, 4는
통과하지만 5가 실패하면 displacement를 mechanism이 아닌 descriptor로만
남긴다.

## 증거 연결

- 실행 계획과 사전 판정: [`plans/global/2026-07-30-session-01-motivation-validation.md`](../../../plans/global/2026-07-30-session-01-motivation-validation.md)
- 관련 연구와 novelty 경계: [`project/proposals/sections/02-related-work-and-novelty-boundary.md`](02-related-work-and-novelty-boundary.md)
- Session 01 global evidence index: [`experiment-reports/global/2026-07-30-session-01-motivation-validation.md`](../../../experiment-reports/global/2026-07-30-session-01-motivation-validation.md)
- 원 handoff: [`project/proposals/00.proposal`](../00.proposal)

## Claim 상태

**hypothesis only.** `motivation supported`는 Session 01의 사전 기준을 통과한
것만 뜻하며, long-horizon superiority, novelty, causal mechanism 또는
paper-ready claim을 뜻하지 않는다.

현재 세부 판정은 **preflight block**이다. 이는 가설 kill이 아니라, read-only
EasyEdit hook과 1–3 edit native-fidelity gate가 통과하기 전에는 GPU 결과를
evidence로 승격하지 않는다는 뜻이다.
