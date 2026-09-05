# Project Proposals

이 디렉터리는 ODE-Edit의 현재 method proposal, 대안 mechanism, 역사적 원문과
research rationale를 보관한다. Proposal은 검증할 연구 계약이지 완료된 paper claim이
아니다.

## 현재 문서

- [2026-09-03 Ordered Response-Barrier ODE-Edit](2026-09-03-ordered-response-barrier-ode-edit-proposal.md):
  current primary research proposal. Official-form full-residual actuator, current terminal-response
  JVP로 만든 \(h\)-independent velocity multiplier, shallow-to-deep ordered state rebuild,
  multi-sweep revisit와 semantic first-hit을 정의한다. Realization potential은 local/soft
  barrier이며 finite-step safety를 보장하지 않는다. Residual excursion과 baseline-native
  parameter action은 controller constraint가 아닌 mechanism telemetry로만 사용한다.
- [2026-08-31 FzCB-Edit method pivot](2026-08-31-fzcb-edit-method-pivot-proposal.md):
  historical predecessor. Stock direct target의 정확한 context semantics,
  MEMIT/AlphaEdit native writer basis, full-model fixed-\(z\) homotopy equality,
  spent-plus-suffix completion budget, whitened equality-null rectification과
  predictor--corrector pipeline을 정의한다. Output KL과 reference fact는 controller에서
  제거하고 held-out 평가로만 사용한다.
- [`2026-08-30-fixed-z-functional-safe-write-proposal.md`](2026-08-30-fixed-z-functional-safe-write-proposal.md):
  2026-08-30 research-reset proposal. Functional anchor/KL contract와 FCW branch를
  정의했던 직전 proposal이며, 2026-08-31 method pivot 이후 historical 문서로 보존한다.
- [`ODE_BF_Dynamic_Layer_Proposal.md`](ODE_BF_Dynamic_Layer_Proposal.md):
  cold target, fixed-E8, full-residual dynamic routing, soft H/P audit와
  W64/BF16 transaction을 결합한 pre-reset proposal. 현재 primary method로 해석하지 않는다.
- [`00.ODE_Alloc_Proposal_Report.md`](00.ODE_Alloc_Proposal_Report.md):
  model trajectory 대신 layer coefficient를 탐색한 pre-reset alternative track. 새 proposal의
  coefficient-space MVP와 동일한 scientific contract로 간주하지 않는다.
- [`00.proposal.md`](00.proposal.md):
  최초 수령한 BF-ODE-Edit proposal 원문. 현재 method와 동일한 문서로 해석하지 않는다.

## 현재 proposal의 상태

현재 primary research direction은 **Ordered Response-Barrier ODE-Edit**이다.

- Entry state에서 baseline이 계산한 \(z^\star\)와 baseline-native artifacts를 고정한다.
- Prescribed remaining-layer \(R/n\) quota를 제거하고 full residual로 official-form writer
  direction을 구성한다.
- Current terminal-response JVP가 target-realization potential을 줄이는 범위에서
  \(h\)-independent velocity multiplier를 계산한다.
- L4→L8 architectural order로 actual virtual state를 전달하고 downstream field를 재구성한다.
- \(T_{\max}=Nh=1\), primary \(N=4,h=0.25\) 아래 multi-sweep를 수행하며 numerical
  convergence와 first-hit을 분리한다.
- Target-bearing compute-\(z\) contexts의 모든 target-new token이 strict teacher-forced top-1일
  때만 first-hit으로 종료하며, horizon miss와 zero-flow stall도 finite endpoint를 commit한다.
- Residual excursion, path/net action과 locality/retention은 controller 밖 telemetry/evaluation이다.
- Inner reject/retry/rollback 없이 advance-and-observe하고 terminal에서 한 번만 commit한다.
- Barrier는 continuous/local shaping이며 finite Euler invariance를 보장하지 않는다.
- PS/NS, locality와 prior-edit outcomes는 controller가 보지 않는다.
- Barrier/ODE contribution은 no-quota, frozen/Jacobi와 fixed-horizon ablation을 통과한 뒤에만
  주장한다.

FzCB, FCW와 이전 ODE-BF 결과/구현은 historical evidence와 transaction 자산으로 보존하지만,
새 proposal의 hypothesis support로 자동 승계하지 않는다.

## Historical/companion fast falsification plan

- [`2026-08-30-fixed-z-fast-falsification-plan.md`](../../plans/global/2026-08-30-fixed-z-fast-falsification-plan.md):
  직전 FCW branch의 F1/F2 execution-direction lock. 결과와 historical decision을 보존하는
  companion plan이며 2026-08-31 FzCB method 수식을 정의하지 않는다. 새 execution 순서와
  promotion 기준은 별도 date-stamped plan으로 관리한다.

## Rationale sections

- [`sections/01-motivation-validation.md`](sections/01-motivation-validation.md):
  초기 BF-share/magnitude motivation chain
- [`sections/02-related-work-and-novelty-boundary.md`](sections/02-related-work-and-novelty-boundary.md):
  MetaKE, CAKE, EvoEdit, ODE prior와 novelty boundary
- [`sections/03-direct-z-review-and-motivation-closure-design.md`](sections/03-direct-z-review-and-motivation-closure-design.md):
  direct-z de-bundling과 역사적 설계
- [`sections/04-method-design.md`](sections/04-method-design.md):
  이전 constrained controller 설계. 해당 section과 FzCB 수식은 historical이며 current
  scientific contract는 `2026-09-03-ordered-response-barrier-ode-edit-proposal.md`를 따른다.
- [`sections/05-bgode-r2-rho-free-prefix-event-fp64-two-equality.md`](sections/05-bgode-r2-rho-free-prefix-event-fp64-two-equality.md):
  FP64 reduction, singular-aware boundary와 ordered-dictionary fidelity의 engineering precedent.
  Prefix-event와 two-equality controller 수식은 current proposal로 승계하지 않는다.

## Evidence navigation

아래 보고서는 모두 pre-reset evidence다. Current proposal 또는 다른 successor method의
hypothesis support로 자동 재사용하지 않는다.

- [전체 ODE-BF 실험 파이프라인](../../experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
- [SH1/SH2 실험 리뷰](../../experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
- [Session 01 closure](../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)

새 experiment는 `PROTOCOL.md`의 firewall, common-policy, immutable receipt와 resource
contract를 따라 preregister한다. Model-specific rescue, post-outcome case selection,
held-out controller access와 raw artifact의 Git 반입은 허용하지 않는다.
