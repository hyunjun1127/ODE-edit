# F2 fixed-z equal-action barrier usefulness 사실 보고서

상태: **PHASE_B_ENGINEERING_HOLD**

핵심 중단 사유는 사전 등록된 W0 factual anchor admissibility 부족이다. Q_gate edited-candidate 평가는 한 번도 열리지 않았으므로 barrier usefulness 결과는 없다.

## FACT — Phase A corrected F1b

| model | method | valid cases | spread > 3×noise | median spread | epsilon nuisance | gate |
|---|---|---:|---:|---:|---:|---|
| llama3-8b-inst | alphaedit | 8/8 | 8/8 | 7.2054565e-05 | 3.13390046e-07 | PASS |
| llama3-8b-inst | memit | 8/8 | 8/8 | 4.65899211e-06 | 1.06520019e-07 | PASS |
| qwen2.5-7b-inst | alphaedit | 8/8 | 8/8 | 9.24296328e-05 | 4.52622771e-07 | PASS |
| qwen2.5-7b-inst | memit | 8/8 | 8/8 | 6.78609358e-05 | 2.31375452e-07 | PASS |

네 cell 모두 direct-z 1회/case, candidate 8/8, exact reset/action/all-context realized-z, left-padding, FULL-FP32 경계를 통과했다.

## FACT — Phase B entry anchor gate

| model | failed case | split | admissible | required | deficit |
|---|---:|---|---:|---:|---:|
| llama3-8b-inst | 5685 | ctrl | 26 | 32 | 6 |
| qwen2.5-7b-inst | 5685 | ctrl | 20 | 32 | 12 |

사전 proposal은 각 fresh case/split에서 W0가 target_true의 모든 token을 strict하게 지지하고 locked numerical floor보다 큰 anchor를 정확히 32개 요구한다. 부족하면 typed HOLD이며 pool 확장, case 교체, margin floor 변경은 0이다.

## INFERENCE

이번 중단은 Phase A의 fixed-z non-uniqueness 소실이 아니라 Phase B reference-anchor 구성 실패다. Barrier-safe/adverse selector, KL comparator, candidate bank 및 held-out Q_gate가 생성·평가되지 않아 barrier 유용성의 방향이나 크기는 판단할 수 없다.

## DECISION

`PHASE_B_ENGINEERING_HOLD`. candidate generation=0, selector lock=0, Q_gate edited access=0, case replacement=0, Phase C/ODE submit=0, scientific_promotion=false.

요구된 gate 기반 표와 6개 figure는 결과가 없는 상태를 수치처럼 보이지 않도록 `NOT_GENERATED_Q_GATE_NEVER_OPENED` 상태 표/placeholder로만 남겼다.
