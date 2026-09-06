# JV sweep 후보를 위한 실제 runtime knob 감사

2026-09-07. Read-only source 조사와 후보 설계만 수행. Source 수정/실험 제출/GPU/기존 AlphaEdit hparams 변경 없음.

Source root: `/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-review-20260907`.
실행 source: `77358b1546d1baf83b3e251afcce663b08d7bfd7`.

## 1. 실제 적용되는 ours 설정

| 항목 | 실제 code binding | 현행값 | sweep 시 주의 |
|---|---|---|---|
| lambda | `alpha_native_response_ode_v31_sequential/contracts.py:10`; `trajectory.py:44` | .1 | NNLS regularization. AlphaEdit의 `hparams.L2=1`과 다른 값이다. |
| Euler node count | `trajectory.py:24`, `range(N)` | 4 | native dictionary/JVP/step을 실제 이 횟수만큼 수행한다. |
| Euler h | `trajectory.py:57` | .5 | `h*c/sqrt(q)`로 physical update에 한 번 적용한다. |
| Total T | `contracts.py:10,15` | 2 | 실행기에서는 T를 import/use하지 않고 실제 T는 N*h다. |
| Energy integration | `trajectory.py:67` | `E += h*qN` | h를 coefficient objective에 넣지 않는다. |
| Time telemetry | `trajectory.py:70` | `(node+1)*h` | source label T와 실제 N*h 일치 검사가 필요하다. |
| N0 | `runtime.py:125`, `native_response_ode_v31/algebra.py:134`–146 | FP32 entry residual norm 후 FP64 weighting | node마다 normalization을 갱신하지 않는다. Positive tiny norm은 clip하지 않는다. |
| Native qref | `runtime.py:126`–129, `native_binding.py:100`–125 | batch entry 전체5 direction action 합 | 매 batch 새로 생성, batch 안에서는 동결. L8 support comparator도 전체5 entry reference다. |
| Native residual divisor | `native_binding.py:22,32`–33 | 1 | objective weighting을 writer RHS에 넣지 않는다. |
| Native metric | `native_binding.py:65`–90 | M_entry + L2 I | L2/P/M lifecycle은 이번 ours sweep에서 고정한다. |
| Readout/target layer | `ordered_response_barrier_ode/runtime.py:635,673` | L8 block output/subject token | final logits가 아니며 편집layer inventory와 별도로 고정한다. |
| Editable inventory | `contracts.py:6`, `runtime.py:82` | L4–L8 down_proj | sweep에서 바꾸지 않는다. |
| First hit | `contracts.py:17`; `trajectory.py:24` | false | trajectory에 semantic-based break가 없다. |
| Backtracking/gain/retry | 동일 계약과 trajectory | 없음 | 낮은 성능을 이유로 해당 run 도중 추가하지 않는다. |

`test_sequential.py:44`는 기존 상수에 대해 N*H=T를 시험하지만 production loop 자체는 T를 참조하지 않는다. **T만 바꾸고 N/h를 그대로 두면 실제 실험은 바뀌지 않고 lock 표기만 달라질 수 있다.** 기존 run은 4*.5=2로 정확하므로 기존 결과 오류는 아니다. Sweep 설정 binding에서 막아야 할 위험이다.

## 2. H/h/lambda 혼동과 설정 전달 위험

1. Controller Gram은 H, Euler timestep은 코드 상수 H로 표기되는 파일도 있다. 문서/CSV에서는 반드시 `response_gram_H`와 `euler_h`를 분리해 명명한다.
2. `trajectory.py:10`은 `from contracts import H,N,LAMBDA`를 사용한다. 나중에 contracts 모듈 속성만 monkeypatch해도 import된 값이 자동 갱신되지 않는다.
3. `telemetry.py:4`는 LAMBDA를 별도로 import한다. `restricted_l8`, best-single-layer shadow, initial-history-cost NNLS가 이를 사용한다. trajectory lambda만 바꾸면 main과 shadow의 목적함수가 달라진다.
4. `SCIENCE`는 import 시 만들어지고 runtime `validate_inputs:25`에서 science.lock과 exact equality를 검사한다. JSON만 바꾸는 방식은 reject되거나 actual code를 바꾸지 않는다.
5. `algebra.nnls_response`의 default lambda=.1은 primary caller가 explicit override한다. 새로운 sweep caller도 모든 relevant NNLS/identities/shadow/ray 경로에 동일 lambda를 명시해야 한다.

권고: kernel/controller math를 다시 쓰지 말고, 별도 sweep runner에 immutable runtime config 하나를 두고 trajectory·telemetry·science lock에 같은 config를 전달한다. 최소 검증은 `T>0`, integer N>0, `h=T/N`, actual time ledger=Nh, explicit lambda>0, 모든 shadow 같은 lambda, h multiplication exactly once, baseline hparams identity unchanged다. 여기에 기존 state persistence gate를 재사용한다. 이 문서에서 구현 수정은 하지 않았다.

## 3. 원본 AlphaEdit compute-z는 그대로 고정

실제 YAML:

- Llama: `project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-llama3-8b.yaml`, SHA `63239e48ea8faf78dfcc40d7d4f5aff3ff0832eb4208384d11f4a812819765c3`.
- Qwen: `project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-qwen2.5-7b.yaml`, SHA `e980d9b5e4ee68b2dc6497672f9a6803f7a9ec5350888b23eb05b9c90f05e815`.

| Fixed baseline hparam | Llama | Qwen |
|---|---:|---:|
| v_num_grad_steps | 25 | 25 |
| v_lr | .1 | .5 |
| v_loss_layer | 31 | 27 |
| v_weight_decay | .5 | .001 |
| clamp_norm_factor | .75 | 4 |
| kl_factor | .0625 | .0625 |
| fact_token | subject_last | subject_last |
| L2 | 1 | 1 |
| nullspace_threshold | .02 | .02 |

각 YAML 6–27행에 layers/module templates, stats/projector policy가 고정되어 있다. Portability용 device/path 변경은 asset bytes identity를 유지하는 범위이며 과학 hparams 변경과 구분한다.

Pinned Official source root:
`/mnt/raid5/janghj/.codex/worktrees/easyeditsh1-official-readonly-v1`.
`easyeditor/models/alphaedit/compute_z.py`:

- 43–76행: target tokenization, context prompts, KL prompt, lookup-position semantics.
- 79행: loss_layer=max(v_loss_layer,target_layer).
- 119행: Adam v_lr.
- 123행: v_num_grad_steps.
- 166–176행: teacher-forced target NLL + KL factor + delta weight decay.
- 182–186행: stock **loss<.05 early break** 또는 마지막 iteration에서 break. 이것은 JV first-hit dynamics가 아니라 baseline compute-z의 기존 optimizer 종료 조건이다.
- 193–196행: clamp_norm_factor에 따른 delta norm 제한.

이 설정 및 stock early break를 이번 sweep에서 변경하지 않는다. Llama의 zero-delta/tiny-positive entry residual 현상을 조사하더라도 v_lr/clamp/stop threshold를 조용히 바꾸면 JV controller sweep이 아니라 target-optimizer 변경 실험이 된다.

또한 P asset, covariance/stats, L2=1, tokenization/context/readout, sample order, precision, history append/commit lifecycle을 동결한다. **JV lambda=.1와 AlphaEdit L2=1는 서로 다른 knob**이므로 lambda sweep에서 L2를 바꾸지 않는다.

## 4. 촘촘하지만 원인을 분리하는 단계형 후보

다음은 후보 설계이며 제출 승인/자동 확대가 아니다. 최종 candidate CSV/예산은 GH가 별도 봉인한다. Primary source-exact N0는 전 구간 그대로 둔다.

### A. 이미 기록된 same-state lambda shadow 우선

`lambda_k=10^(-3+k/4), k=0..12`:

`.001, .00177828, .00316228, .00562341, .01, .01778279, .03162278, .05623413, .1, .17782794, .31622777, .56234133, 1`.

이 13-point log grid는 existing80node의 g/H/G로 forward 없이 active support, physical coefficients, KKT, objective, predicted progress, same-state field angle/norm을 비교할 수 있다. 전체 실제 nonlinear sequential trajectory나 efficacy를 예측했다고 부르지 않는다. 특히 Llama B10 H가 매우 크고 alignment가 거의 0인 현상에서 .001–1 변화만으로 개선되지 않을 가능성도 결과로 보존한다.

실제 lambda trajectory를 선정할 때는 predeclared development sample에서 진행하고 최종 audit의 PS/NS를 selector로 사용하지 않는다. 광범위한 T/N 교차곱을 먼저 실행하지 않는다.

### B. Euler discretization과 horizon을 구분

| 목적 | lambda | T | N | h | 해석 |
|---|---:|---:|---|---|---|
| fixed-T refinement | .1 | 2 | 2/4/8/16 | 1/.5/.25/.125 | 동일 ODE horizon의 discretization 영향 |
| fixed-h exposure | .1 | 1/1.5/2/2.5/3/4 | 2/3/4/5/6/8 | .5 | fixed-grid 노출시간 영향; refinement와 다름 |

공통 (T2,N4,h.5)은 중복 실행하지 않는다. 초기 소규모 grid는 N2/4/8와 T1/2/3로 줄여도 되지만, 결과를 보고 임의 rescue하는 것이 아니라 실행 전 budget/priority를 봉인해야 한다. Fine grid 또는 N16은 비용 우선순위상 후속이다.

동일 T에서 N을 늘리면 one-step 크기는 줄고 JVP/solve 수는 늘어난다. JV의 main JVP는 active5layer를 전제로 5N개: N2/4/8/16 → 10/20/40/80. compute-z와 endpoint evaluation 비용은 별도이므로 총 GPU 시간이 정확히 이 배율인 것은 아니다.

Fixed N에서 T만 조정하여 h=T/N를 키우면 exposure와 numerical error가 동시에 바뀐다. 해당 설계도 가능하지만 “horizon만의 효과”라고 부르면 안 된다. T/h가 일치하도록 runtime assertion이 필수다.

### C. 제한적 interaction 후 audit

λ shadow와 fixed-T/fixed-h development에서 정한 작은 교차 조합만, 동일 두 모델/사전봉인표본에서 비교한다. 개선된 모델만 확대하는 선택은 피한다. 오래된 실패 B10 checkpoint를 analysis anchor로 쓸 수는 있으나 exact W/M/z가 없으면 재구성했다고 주장하지 않는다.

Final 1000/더긴lifelong 실행은 이 후보산출과 별개의 readiness/비용 판단이다. Current finite near-stall endpoint를 제외하거나 Official rescue로 대체하지 않는다.

## 5. Normalization은 지금 lambda/T/N sweep과 별도 science ablation

기록된 global g/H만으로 N0→NRMS/NNUM의 새 response Gram을 정확하게 재구성할 수 없다. Request별 scales는 있어도 request별 raw JVP/Gram 기여가 있어야 새 weighting을 계산할 수 있다. 이번 publication에는 per-request raw response tensor/Gram 분해가 없어 global H로 역산하면 underdetermined다.

NNUM은 반복 activation noise 측정도 필요하므로 현재 raw-free aggregate만으로 정확한 noise floor를 발명할 수 없다. NRMS/NNUM 진단을 승인할 경우 별도의 observer/data acquisition과 science version으로 명시하고, baseline compute-z는 그대로 두어야 한다. 현 source에서 tiny-positive N0를 floor/clip하는 것은 technical fix가 아니라 objective weighting 변경이다.

## 6. 최소 correctness와 해석 규칙

- 현재 run의 actual N*h=2, h 단일적용, λ=.1 일관성을 확인했다. 기존 실행이 knob-binding bug로 잘못 돌았다는 증거는 없다.
- 변경할 부분은 runtime config 전달/lock validation이지 native solve/JVP/NNLS 방정식의 재구현이 아니다.
- T/N/h sensitivity는 finite-step defect, actual V progress, physical materialization, rewrite NLL/coverage 및 old retention을 함께 봐야 한다.
- Near-stall이 tiny N0와 관련된 극단적 weighting/response 정렬에서 왔다면 h를 줄이는 것만으로 current objective의 잘못된 우선순위가 해결되지는 않는다. λ와 Euler sweep에 “반드시 개선된다”는 전제를 두지 않는다.
- Single-layer concentration을 없애는 방향으로 candidate를 고르지 않는다. 궁극적 비교는 같은 신규 edit 품질에서 누적 retention/locality와 비용이다.
