# Project Proposals

이 디렉터리는 ODE-Edit의 현재 method proposal, 대안 mechanism, 역사적 원문과
research rationale를 보관한다. Proposal은 검증할 연구 계약이지 완료된 paper claim이
아니다.

## 현재 문서

- [2026-08-31 FzCB-Edit method pivot](2026-08-31-fzcb-edit-method-pivot-proposal.md):
  current primary method proposal. Stock direct target의 정확한 context semantics,
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

현재 primary research direction은 **FzCB-Edit: Fixed-z Conditional-Completion Barrier
Editing**이다.

- direct-\(z\)와 shared \(\delta^\star\)는 기존 editor가 한 번 계산한 값을 고정한다.
- 여러 context에 동일 absolute \(z^\star\)를 반복하지 않고
  \(a_{i,p}^0+\delta_i^\star\)를 context target으로 사용한다.
- ODE step 수 자체를 contribution으로 보지 않는다.
- Fixed-\(z\) progress는 full-model hard homotopy equality로 강제한다.
- Barrier는 하나이며
  \(E_{\mathrm{spent}}+\widehat V_{\mathrm{suf}}\le A_0\)만 제어한다.
- Output KL, locality와 prior-edit facts는 controller가 보지 않는다.
- MEMIT/AlphaEdit closed form은 endpoint가 아니라 native \(T,H\) geometry로 사용한다.
- Whitened equality-null freedom에서 scalar barrier rectification을 계산한다.
- Actual nonlinear corrector와 completion-budget 검증을 finite-step authority로 둔다.
- static constrained solver보다 multi-step feedback이 우월할 때만 ODE 명칭을 유지한다.

이전 ODE-BF Cold-FR-E8 결과와 구현은 역사적 evidence와 transaction 자산으로 보존하지만,
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
  이전 constrained controller 설계. Current scientific contract는
  `2026-08-31-fzcb-edit-method-pivot-proposal.md`를 따른다.

## Evidence navigation

아래 보고서는 모두 pre-reset evidence다. FzCB method를 지지하는 근거로 자동 재사용하지
않는다.

- [전체 ODE-BF 실험 파이프라인](../../experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
- [SH1/SH2 실험 리뷰](../../experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
- [Session 01 closure](../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)

새 experiment는 `PROTOCOL.md`의 firewall, common-policy, immutable receipt와 resource
contract를 따라 preregister한다. Model-specific rescue, post-outcome case selection,
held-out controller access와 raw artifact의 Git 반입은 허용하지 않는다.
