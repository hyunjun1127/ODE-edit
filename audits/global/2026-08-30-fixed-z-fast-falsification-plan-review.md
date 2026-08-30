# Fixed-z Fast Falsification Plan 검토

## 범위

- 대상: `plans/global/2026-08-30-fixed-z-fast-falsification-plan.md`
- 상위 계약: `project/proposals/2026-08-30-fixed-z-functional-safe-write-proposal.md`
- 검토일: 2026-08-30
- 상태: execution-direction review; scientific execution 미승인

## 판정

`PASS_WITH_PREEXECUTION_LOCKS`다. Plan은 F0 algebra, F1 same-z multiplicity, F2 conditional
functional value를 먼저 반증하고, 통과한 뒤에만 static FCW와 ODE necessity를 여는 구조다.
상위 proposal의 falsification order, four-bank firewall, FULL_FP32 authority와 one-shot final
audit 경계를 보존한다.

## 본 commit에서 교정한 항목

1. FULL_FP32를 exact arithmetic으로 표현하지 않고 algebraic identity와 numerical tolerance
   receipt를 분리했다.
2. \(C^{-1}\)와 Gram inverse를 dense materialization하지 않고 fixed factorization/solve
   backend만 사용하도록 고정했다.
3. CBF/CLF 식의 penalty와 혼동되지 않도록 matched tangent energy를
   \(\rho_{\mathrm{tan}}\)으로 이름 붙였다.

## 실행 전 필수 lock

- `Q_edit/Q_hist/Q_cal/Q_ctrl/Q_gate/Q_audit` source-family-disjoint manifest
- Gate 결과를 보기 전에 candidate sign, seed, scale, rank tolerance와 solve backend 봉인
- F0 mandatory identity 6개 전부 PASS
- F1-T는 technical-only이며 fresh scientific denominator 영향 0
- F1/F2 screen 전 GPU/model/runtime root와 compute cap 별도 승인
- `Q_audit`은 F4와 모든 선택이 끝날 때까지 open 0

## Claim boundary

8-case/model F1/F2 screen은 engineering premise screen이다. Canonical Gate 1/2나 method
효능 근거가 아니다. F1 또는 F2가 실패하면 F3, ODE, AlphaEdit, multi-layer와 sequential을
열지 않는다. Static comparator가 refreshed trajectory를 지배하면 ODE 명칭을 사용하지 않는다.
