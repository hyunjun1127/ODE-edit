# FzCB-Edit main method initial factual report — B1 scientific hold

상태: **HOLD_AFTER_QWEN_B1_LOCAL_INFEASIBILITY**. Llama B1은 terminal-valid였지만
Qwen B1에서 계약상 typed local barrier infeasibility가 발생했으므로 tolerance, ridge,
barrier, target 또는 sample을 완화하지 않았고 B10은 제출하지 않았다. promotion=false.

## 1. 수식과 코드 경로

수식→자산 상세 mapping은 `plans/updates/server4/fzcb-edit-main-method-equation-mapping.md`에 있다.
stock MEMIT `compute_z`는 request당 정확히 한 번 호출되어 다섯 arm이 같은 z*를 공유했다.
새 package는 target, basis/metric, matrix-free control map, equality/suffix KKT, scalar barrier,
predictor-corrector, transaction, telemetry를 분리했다. F2 q-KL/reference controller와 K0 promotion
logic은 import하지 않았다.

## 2. 구현 경계

- Exact: stock target 1회, MEMIT factor orientation, frozen M0 covariance metric, gauge removal,
  matrix-free JVP/VJP equality map, equality-only minimum-H control, one-barrier scalar rectification,
  net corrected-action accounting, s=1 terminal-only atomic commit/rollback.
- Engineering approximation: suffix-value sensitivity는 reduced equality-null 2-axis finite difference
  correctness prototype다. exact production HVP로 해석하지 않는다.
- Endpoint output NLL/locality는 observation-only이며 controller influence count=0이다.

## 3. Focused gates

G0 CPU/toy 7/7 PASS: factorized/dense parity, gauge/H, C/C^T adjoint, equality/suffix KKT,
frozen identity, scalar rectifier/local infeasibility, torch.func operator, rollback/atomic commit,
forbidden-import. py_compile, bash syntax, access gate도 PASS했다.

Llama runtime: padding safety PASS, FULL-FP32, direct-z compute/recompute=1/0, five-arm shared target,
Official bypass parity=true, cache entry/terminal identity exact, W0 pointer/bytes restore=true/true.

## 4. Llama B1 initial arm comparison

분모는 sealed B1의 deterministic first request 1개다. rewrite prompt=1, rephrase prompts=2,
locality prompts=10이므로 일반 성능 결론이 아니라 technical/engineering smoke다.

| arm | rewrite new NLL mean | rewrite true NLL mean | rephrase new NLL mean | rephrase true NLL mean | locality true NLL mean | rewrite strict num | den |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_EDIT | 13.5086 | 15.2707 | 8.11219 | 11.0309 | 6.7169 | 0 | 1 |
| OFFICIAL_MEMIT | 0.0130157 | 11.6493 | 7.15251 | 11.1798 | 6.63539 | 1 | 1 |
| FROZEN_SPLIT | 0.000458731 | 16.2591 | 6.97792 | 11.1637 | 6.65153 | 1 | 1 |
| REFRESHED_EQUALITY_ONLY | 0.000463498 | 16.2413 | 7.03635 | 11.1499 | 6.65892 | 1 | 1 |
| FZCB | 0.000463498 | 16.2413 | 7.03635 | 11.1499 | 6.65892 | 1 | 1 |
| STATIC_PATH | 0.000456229 | 16.2312 | 6.79606 | 11.1729 | 6.64811 | 1 | 1 |

## 5. Range, suffix/barrier, corrector/action

| arm | waypoints | barrier active | terminal closure | budget | spent action | budget error |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_SPLIT | 4 | 0 | 0.00119498 | 0.000176959 | 0.000219515 | 4.25565e-05 |
| REFRESHED_EQUALITY_ONLY | 4 | 0 | 0.001351 | 0.000176959 | 0.000170774 | -6.18445e-06 |
| FZCB | 4 | 0 | 0.001351 | 0.000176959 | 0.000170774 | -6.18445e-06 |
| STATIC_PATH | 2 | 0 | 0.000155862 | 0.000176959 | 0.000233555 | 5.65966e-05 |

Llama FZCB barrier active count는 0/4이다.
FZCB와 REFRESHED_EQUALITY_ONLY endpoint identity/핵심 NLL exact equality=true.
따라서 이 B1에서는 barrier contribution claim을 제거한다. STATIC_PATH가 rephrase-new NLL에서
가장 낮았지만 request 1개 관측이므로 우월성이나 ODE necessity를 주장하지 않는다.

## 6. Qwen typed boundary

job 30715은 `typed local barrier infeasibility psi0=0.0904913303 g2=0.000154264271 psi_min=0.0904141982`로 종료됐다. 이는 equality-null subspace에서
현재 상태의 barrier가 불가능하다는 typed scientific/mathematical boundary다. 공식 target/writer
계산 후 검출됐으며 fallback, backtracking에 의한 재분류, tolerance/ridge/kappa 변경은 0이다.
Qwen endpoint arm denominator=0이며 성능 수치는 산출하지 않는다.

## 7. 실행과 다음 권고

- Llama job 30670: COMPLETED, terminal-valid, wall=3936.02s,
  peak allocated=48348670464 bytes.
- Qwen job 30715: FAILED exit2 by intentional typed boundary; immutable evidence, denominator0.
- B10 jobs: 0. Qwen confirmation이 통과하지 않아 자동 흐름을 중단했다.
- 다음 실행은 barrier feasibility/authority semantics에 대한 사용자/GH 판단 이후에만 권고한다.
  현재 numerical lock을 사후 완화한 재실행은 권고하지 않는다.
