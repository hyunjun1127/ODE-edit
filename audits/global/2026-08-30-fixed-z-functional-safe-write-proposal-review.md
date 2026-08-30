# Fixed-z Functional Safe Write proposal 과학 검토

## 검토 범위

- 대상: `project/proposals/2026-08-30-fixed-z-functional-safe-write-proposal.md`
- 검토일: 2026-08-30
- 검토 역할: global-head
- 산출물 성격: proposal pre-execution scientific review

## 결론

제안서의 연구 리셋과 falsification-first 구조는 타당하다. 특히 direct-\(z\)를 고정하고
same-z multiplicity, functional signal의 conditional value, static solver 대비 feedback의
필요성을 순차적으로 검증하는 구조는 이전 ODE 가설을 자동 승계하지 않는다.

다만 원문 상태로는 실행 계약으로 사용할 수 없는 다섯 개의 중요 문제가 있었다. 본 review와
같은 commit에서 이를 proposal 본문에 교정했다.

## 교정한 중요 문제

1. **Gate 0 finite-time equivalence 오류**
   - Quadratic gradient flow는 일반적으로 유한 terminal time에 closed-form minimizer와 같지
     않다.
   - Exact algebraic split parity와 analytic finite-time flow convergence를 분리했다.
2. **Per-anchor Jacobian 비용 과소계상**
   - Aggregate scalar ordinary backward는 microbatch gradient의 합만 제공한다.
   - Batched VJP/`vmap(grad)` 또는 anchor/group별 backward를 요구하고 실제 호출 수를
     compute receipt에 포함했다.
3. **Gate selection과 final audit 누수**
   - Go/No-go에 사용한 bank는 더 이상 final held-out audit일 수 없다.
   - `Q_ctrl`, `Q_cal`, `Q_gate`, `Q_audit`의 네 bank로 분리하고 audit은 모든 선택 뒤 한 번만
     열도록 고쳤다.
4. **실행 dtype authority 불일치**
   - BF16을 기본 safety authority로 고정하지 않는다.
   - Gate 0--2는 FULL_FP32로 고정하고 저정밀 deployment는 별도 study로 분리했다.
5. **Local equality와 full-model fixed-z의 혼동**
   - \(\Delta K_E=R_E\)는 layer-local 조건이다.
   - Virtual/commit 뒤 full-model activation, target event와 logits realization gate를 추가했다.

## 함께 강화한 경계

- Coefficient Gram이 singular할 수 있으므로 preregistered numerical range solve와 rank receipt를
  요구했다. 결과 후 ridge 또는 rank threshold 변경은 금지한다.
- Same-z candidate direction과 noise floor는 calibration-only로 고정하고 gate score 기반
  candidate 재생성을 금지했다.
- Compute matching은 outcome-independent budget으로 사전 등록하며, baseline search budget을
  결과 후 늘리는 것을 금지했다.
- Exact Alpha projector 식은 orthonormal basis 조건에서만 사용하도록 명시했다.

## 실행 전 필수 조건

이 review는 방법의 성공 판정이 아니다. 다음 조건을 충족하기 전에는 Gate 3 이후 구현이나
scientific promotion을 진행하지 않는다.

1. 네 bank의 source-family-disjoint manifest와 one-shot audit policy
2. Gate 0 split parity와 finite-time flow reference의 독립 test
3. Gate 1 same-z consistency, local equality, full-model realization의 분리 receipt
4. Per-anchor VJP/backward 호출 수와 memory accounting
5. Singular Gram typed handling과 outcome-independent tolerance lock

Gate 1 premise 또는 Gate 2 conditional incremental value가 실패하면 method를 중단한다.
Gate 4에서 static constrained solver가 지배하면 ODE 명칭을 사용하지 않는다.
