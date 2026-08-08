# Project Proposals

이 디렉터리는 ODE-Edit의 현재 method proposal, 대안 mechanism, 역사적 원문과
research rationale를 보관한다. Proposal은 검증할 연구 계약이지 완료된 paper claim이
아니다.

## 현재 문서

- [`ODE_BF_Dynamic_Layer_Proposal.md`](ODE_BF_Dynamic_Layer_Proposal.md):
  cold target, fixed-E8, full-residual dynamic routing, soft H/P audit와
  W64/BF16 transaction을 결합한 현재 primary proposal
- [`00.ODE_Alloc_Proposal_Report.md`](00.ODE_Alloc_Proposal_Report.md):
  model trajectory 대신 layer coefficient만 탐색하는 저비용 component/alternative track
- [`00.proposal.md`](00.proposal.md):
  최초 수령한 BF-ODE-Edit proposal 원문. 현재 method와 동일한 문서로 해석하지 않는다.

## 현재 proposal의 상태

현재 primary design은 **ODE-BF Cold-FR-E8**이다.

- `z_base` cold start와 target-new-NLL field
- target-only bootstrap과 joint Euler clock의 분리
- request별 shared terminal full residual
- `K=8`, `h=1/8`, `tau=1` fixed rollout
- Neutral/Soft routing과 observation-only first hit
- H/P soft signal 및 raw audit; 임의 hard budget 미사용
- W64 reduced solve, BF16-authoritative virtual transition과 terminal atomic commit

이 설계 중 common cold-coordinate 수정은 아직 실행 검증 중이다. Warm ALLOFF의 강한
완료 결과를 cold method의 결과로 재사용하지 않으며, same-seal Native/cold rerun과
ordered sequential gate가 끝나기 전에는 lifelong 또는 formal barrier claim을 하지 않는다.

## Rationale sections

- [`sections/01-motivation-validation.md`](sections/01-motivation-validation.md):
  초기 BF-share/magnitude motivation chain
- [`sections/02-related-work-and-novelty-boundary.md`](sections/02-related-work-and-novelty-boundary.md):
  MetaKE, CAKE, EvoEdit, ODE prior와 novelty boundary
- [`sections/03-direct-z-review-and-motivation-closure-design.md`](sections/03-direct-z-review-and-motivation-closure-design.md):
  direct-z de-bundling과 역사적 설계
- [`sections/04-method-design.md`](sections/04-method-design.md):
  이전 constrained controller 설계. 최신 primary contract는 위 ODE-BF proposal과
  실험 파이프라인 보고서를 함께 따른다.

## Evidence navigation

- [전체 ODE-BF 실험 파이프라인](../../experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
- [SH1/SH2 실험 리뷰](../../experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
- [Session 01 closure](../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)

새 experiment는 `PROTOCOL.md`의 firewall, common-policy, immutable receipt와 resource
contract를 따라 preregister한다. Model-specific rescue, post-outcome case selection,
held-out controller access와 raw artifact의 Git 반입은 허용하지 않는다.
