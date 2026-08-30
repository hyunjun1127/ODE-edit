# Project Proposals

이 디렉터리는 ODE-Edit의 현재 method proposal, 대안 mechanism, 역사적 원문과
research rationale를 보관한다. Proposal은 검증할 연구 계약이지 완료된 paper claim이
아니다.

## 현재 문서

- [`2026-08-30-fixed-z-functional-safe-write-proposal.md`](2026-08-30-fixed-z-functional-safe-write-proposal.md):
  2026-08-30 연구 리셋 이후의 current primary proposal. Frozen direct-z,
  functional preservation contract, coefficient-space constrained writer와
  falsification-first Gate 0--5를 정의한다. ODE는 static constrained solve 대비
  추가 가치가 확인된 뒤에만 조건부로 승격한다.
- [`ODE_BF_Dynamic_Layer_Proposal.md`](ODE_BF_Dynamic_Layer_Proposal.md):
  cold target, fixed-E8, full-residual dynamic routing, soft H/P audit와
  W64/BF16 transaction을 결합한 pre-reset proposal. 현재 primary method로 해석하지 않는다.
- [`00.ODE_Alloc_Proposal_Report.md`](00.ODE_Alloc_Proposal_Report.md):
  model trajectory 대신 layer coefficient를 탐색한 pre-reset alternative track. 새 proposal의
  coefficient-space MVP와 동일한 scientific contract로 간주하지 않는다.
- [`00.proposal.md`](00.proposal.md):
  최초 수령한 BF-ODE-Edit proposal 원문. 현재 method와 동일한 문서로 해석하지 않는다.

## 현재 proposal의 상태

현재 primary research direction은 **Fixed-z Functional Safe Write**다.

- direct-z는 기존 editor가 계산한 값을 고정한다.
- ODE step 수 자체를 contribution으로 보지 않는다.
- same-z weight multiplicity를 Gate 1에서 먼저 실증한다.
- functional signal이 key/null-space risk보다 추가 가치가 있는지 Gate 2에서 검증한다.
- 초기 구현은 full W-space가 아니라 native low-rank coefficient space에서 수행한다.
- actual finite functional check와 authoritative-dtype commit identity를 safety authority로 둔다.
- static constrained solver보다 multi-step feedback이 우월할 때만 ODE 명칭을 유지한다.
- Gate 0--4 통과 전에는 long-horizon, formal CBF, lifelong superiority를 주장하지 않는다.

이전 ODE-BF Cold-FR-E8 결과와 구현은 역사적 evidence와 transaction 자산으로 보존하지만,
새 proposal의 hypothesis support로 자동 승계하지 않는다.

## Rationale sections

- [`sections/01-motivation-validation.md`](sections/01-motivation-validation.md):
  초기 BF-share/magnitude motivation chain
- [`sections/02-related-work-and-novelty-boundary.md`](sections/02-related-work-and-novelty-boundary.md):
  MetaKE, CAKE, EvoEdit, ODE prior와 novelty boundary
- [`sections/03-direct-z-review-and-motivation-closure-design.md`](sections/03-direct-z-review-and-motivation-closure-design.md):
  direct-z de-bundling과 역사적 설계
- [`sections/04-method-design.md`](sections/04-method-design.md):
  이전 constrained controller 설계. 2026-08-30 이후 current scientific contract는
  `2026-08-30-fixed-z-functional-safe-write-proposal.md`를 따른다.

## Evidence navigation

아래 보고서는 모두 pre-reset evidence다. 새 proposal의 Gate를 통과한 근거로 자동 재사용하지
않는다.

- [전체 ODE-BF 실험 파이프라인](../../experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
- [SH1/SH2 실험 리뷰](../../experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
- [Session 01 closure](../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)

새 experiment는 `PROTOCOL.md`의 firewall, common-policy, immutable receipt와 resource
contract를 따라 preregister한다. Model-specific rescue, post-outcome case selection,
held-out controller access와 raw artifact의 Git 반입은 허용하지 않는다.
