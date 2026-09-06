# GH 독립 history/barrier/response 코드 감사

작성일: 2026-09-07. 분석 전용. 모델/GPU/실험 재평가/서버2 접근/SH4 live artifact 접근 없음.

## 0. 범위와 결론

실행 source `77358b1546d1baf83b3e251afcce663b08d7bfd7`에 대해, 로컬 봉인 publication worktree `0d0a0131e4a6a2a645dfa6530377d420a084d136`에서 아래 코드를 독립 검토했다. 두 commit의 관련 diff는 보고서/분석 코드에 한정되며 아래 runtime/controller/state 핵심 파일은 동일하다.

Source root: `/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-review-20260907`.

Publication root: source root 아래 `experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1`.

핵심 결론:

1. 과거 edit history를 barrier/controller에서 아예 누락하거나 batch마다 reset한 구현 증거는 없다. 실제 코드에는 W/M persistence, M을 사용하는 native direction, M+L2 native metric이 모두 있다.
2. 그러나 보호 대상은 **과거 key에서의 quadratic disturbance surrogate**다. 과거 edit의 정답/잔차/현재 output을 직접 barrier에 넣은 것은 아니다. 따라서 “M이 반영됐다”와 “historical retention을 보장했다”는 전혀 다르다.
3. `G=I`는 history를 버린 identity metric이 아니라 native whitening 결과다. M은 direction, q, whitened response에 남는다.
4. 관측된 L8 우세는 target readout이 L8 block output이라는 기하와 잘 맞는다. 아래 layer의 response는 상대적으로 덜 정렬되어 있다. 이는 현재 잔차 최적화의 합리적 해일 수 있으며 barrier 오작동을 뜻하지 않는다.
5. FP64 KKT/dissipation 대수는 80 node에서 맞는다. 실제 유한 Euler barrier increment는 Qwen 40/40 양수, Llama 38/40 양수다. Llama B10 두 node는 음수다. Algebraic identity만으로 finite-step safety/locality/retention을 주장하면 안 된다.
6. 현 자료에서 큰 history-cost-induced physical turning은 입증되지 않았다. Qwen B10에는 비용 효과가 실제 존재하지만 대부분 amplitude 변화이고 방향각은 약 0.274도다. Native direction 생성에 대한 M의 효과는 이 shadow로 분리되지 않는다.

## 1. W/M은 어떻게 이어지는가: 확정 코드 경로

`alpha_native_response_ode_v31_sequential/runtime.py`:

- 98–116행: 매 batch 새로운 `ObservedFamily`; 첫 batch만 `prepare_method_state`, 이후 `bind_existing_method_state`.
- 104–109행: cold M0=0 확인/원본 W0,M0 snapshot. 이 초기화는 chain당 한 번이다.
- 115–116행: 이후 batch에서 기존 M을 bind하고 이전 commit과 연결 확인.
- 123–134행: 자기 arm의 현재 entry에서 z를 한 번 계산하고 새 normalization/NativeDictionary/reference를 생성한다.
- 151–154행: 이미 materialize/finalize/capture한 endpoint를 transaction 복원 뒤 commit한다.
- 181–183행: 이전 dictionary/family를 해제하므로 오래된 metric operator를 다음 batch에 재사용하지 않는다.
- 184–191행의 cold W0/M0 복원은 chain 전체 종료 후다.

`alpha_native_response_ode_v31_sequential/state.py`:

- 10–19행: append_count=1, transaction의 entry 복원, captured M bytes 및 실제 committed M equality를 검사한다. commit의 writer/z/forward/evaluator 재계산 count=0을 요구한다.
- 22–25행: 다음 family의 W entry hash와 M content hash가 이전 committed endpoint와 같은지 검사한다.
- 28–42행: checkpoint는 실제 selected weights와 alpha_cache를 저장하고 reload하여 hash를 검증한다. hash-only checkpoint가 아니다.

`ordered_response_barrier_ode/runtime.py`:

- 523–556행: 최초 cold M 생성과 P binding.
- 558–590행: warm M을 clone하여 `_alpha_cache_entry`에 고정한다. 기존 cache_c를 0으로 만들지 않는다.
- 592–608행: inner/transaction entry 복원은 **해당 batch entry**의 W/M으로 한다. pretrained W0와 혼동하면 안 된다.
- 798–808행: endpoint W와 post-finalization cache를 함께 capture.
- 811–856행: capture한 W/M을 copy-only commit. `cache_c_new=True`는 stock 의미상 “이미 초기화됨”으로서 다음 Official 호출이 M을 0으로 다시 만들지 않게 한다.
- 879–890행: endpoint에서 전체 L4–L8 keys를 구해 각 M에 `K K^T`를 한 번 append. L8에만 실제 write했어도 history inventory는 전체 5 layer다.

이 경로는 `sequential_commit_checks.csv` / `state_continuity_summary.csv`의 publication 증거와 일치한다. CSV를 별도로 재계산했을 때 네 chain 각각 9/9 inter-batch W/M links, 10/10 history_append=1, 합계 36/36 links와 40/40 append가 맞았다. 서버2 raw tensor 자체의 독립 rehash는 이번 감사에서 불가능하며 기존 terminal/checkpoint receipt에 의존한다.

## 2. History는 수학의 어디에 들어가는가

AlphaEdit current native-form direction:

`ordered_response_barrier_ode/runtime.py:704`에서 current-state keys를 구하고, 712–716행에서 현재 batch fixed z minus current terminal을 divisor 1로 사용한다. 734–742행의 solve는

`right = solve(P (K K^T + M_entry) + L2 I, P K)`

이며 stored update orientation에서 `D = residual @ right^T`다. 즉 M은 **candidate direction 생성**부터 들어간다. 이 factorized native-form 경로는 stock dense RHS와 같은 실수 대수 family이나 FP32 bitwise Official equivalence를 새로 주장하는 것은 아니다.

`native_response_ode_v31/native_binding.py:65`–90:

`S = _alpha_cache_entry[layer] + L2 I`를 실제 operator로 적용한다. 비대칭 P-inside solve matrix를 quadratic metric으로 잘못 사용하지 않는다. 92–98행의 factor-space inner product는 `tr(A S B^T)`이고, 100–125행이 reference와 whitening이다.

따라서 M의 두 경로를 분리해야 한다.

- direction 경로: M이 native solve 자체를 바꾼다.
- cost 경로: 동일 physical candidate라도 `tr(F M F^T)`가 controller의 disturbance cost를 바꾼다.

현재 cache가 endpoint별 historical key Gram의 누적이라면, 실수 대수에서

`tr(F M F^T) = sum_past ||F K_past_at_its_commit||_F^2`.

이는 과거 key의 해당 linear module output을 바꾸는 비용 surrogate다. 다음은 포함되지 않는다.

- 과거 request의 현재 key를 다시 계산해 key drift를 반영하는 과정
- 과거 z residual/현재 logits/정답 margin/old-edit success에 대한 constraint
- 한 layer의 변화가 downstream old output으로 전달되는 전체 Jacobian
- 서로 다른 layer의 functional interference cross-term

과거 key는 자신의 commit 시점에서 동결되어 누적된다. 모델의 앞 layer가 나중에 달라져 current old keys가 바뀌면 이 surrogate와 실제 old output 변화가 벌어질 수 있다. 반대로 early-layer write가 작으면 downstream key drift가 덜할 가능성은 있으나, 본 결과만으로 인과적 입증은 아니다.

## 3. Batch마다 qref가 갱신되는 의미

`runtime.py:123`–134와 `native_binding.py:100`–125:

- 매 batch 자기 entry direction 전체 L4–L8의 raw native action 합을 qN_ref로 한 번 계산한다.
- inner node에서는 reference를 다시 계산하지 않는다.
- 다음 batch에서는 새 M, 새 W, 새 residual/dictionary로 qN_ref를 새로 계산한다.

이것은 사용자 sequential 계약대로이며 implementation bug가 아니다. 하지만 해석상 중요한 한계다.

Controller가 최소화하는 비용은 raw action이 아니라 `lambda * Q_raw(F)/qN_ref`다. raw history cost가 커져도 entry reference 또한 커지면 **normalized penalty의 절대 크기가 함께 증가하지 않을 수 있다**. 고정된 physical candidates에 대해 S와 qref를 공통 scalar로 같이 늘리면 normalized metric은 정확히 불변이다. 실제 batch에서는 M의 anisotropy, W, D도 바뀌므로 전 과정 불변을 뜻하지 않는다.

따라서 “편집 수가 늘어 M이 커지면 언제나 기존 edit 보호 압력이 더 강해져야 한다”는 주장은 현재 계약에서 나오지 않는다. M trace, raw action, normalized action, realized forgetting은 별도 축이다.

## 4. 이 barrier가 실제로 하는 일과 하지 않는 일

`alpha_native_response_ode_v31_sequential/trajectory.py:16`–21에서 `V0`와 `energy=0`가 batch마다 새로 시작된다. 26행의 e는 현재 batch만 사용한다. 44–48행의 controller는 nonnegative regularized response least squares이며, b 값/old retention을 입력으로 받아 별도 safe-set correction을 수행하는 구조는 아니다.

`native_response_ode_v31/algebra.py:28`–72의 FP64 active-set NNLS와 75–85행의 identity:

`g^T c = c^T H c + lambda c^T G c`

에서 `dV/dt = -g^T c`, `dE/dt=Q_N(F)`이면

`db/dt = d(V0-V-lambda E)/dt = ||Psi c||^2 >= 0`.

이는 current residual progress와 native quadratic work 사이의 **dissipation identity**다. 과거 model-state 전체를 원상보존하는 barrier나 output-side locality guarantee가 아니다. b를 batch 간 누적하여 global lifelong safety reserve로 사용하지도 않는다.

공개 80 node의 c,g,H,G로 독립 scalar 재계산:

- max active-coordinate KKT residual: `8.881784197001252e-16`.
- max absolute dissipation residual: `3.3133218391157016e-16`.
- native speed-bound violation: 0.

이러한 대수 합격은 objective를 제대로 풀었다는 근거다. 목적 자체가 원하는 locality/retention을 최적화한다는 근거와는 다르다.

## 5. 실제 finite Euler barrier 재검산

공개 `layer_allocation_nodes.csv`의 V_before/V_after/c를 사용해, h=.5, lambda=.1, G=I에서

`actual_delta_b = V_before - V_after - .05 * sum(c_l^2)`

를 재계산했다. 이것은 코드의 predicted defect 필드와 별개로 실제 관측 activation에 기반한다.

| 모델 | 양수 node | 최소 actual delta b | 최대 actual delta b | 특이점 |
|---|---:|---:|---:|---|
| Llama | 38/40 | -3.7809908579e-5 | .3588415492 | B10 node0/node3 음수 |
| Qwen | 40/40 | .005688102756 | .3638661613 | 모든 node 양수 |

Llama B10 actual delta b (node0..3):

`[-3.7809908579e-5, +6.1257490158e-5, +3.9401500914e-5, -3.6443364990e-5]`.

합계 `+2.6405717503e-5`이나 사실상 progress가 매우 작다. 반면 Llama B1–9의 batch barrier increment는 `.4818877`–`.4846805`, Qwen B1–10은 `.4693099`–`.4891447`다.

기존 code의 `finite_step_dissipation_defect` (`trajectory.py:78`)는 actual delta b와 affine-predicted `h(1-h/2)||Psi c||²`의 차이를 부호 반대로 기록한다. Positive defect 자체를 곧바로 actual barrier 감소라고 부르면 안 된다. Qwen의 positive defect 35/40도 실제 delta b는 모두 양수다.

Llama B10에서 normalized observation/finite precision 영향과 실제 nonlinear model error를 이 CSV만으로 완전히 분리할 수 없다. 두 negative increment를 숨기면 안 되지만 이를 수학 KKT failure라고도 부르면 안 된다.

## 6. L8가 선택되는 구조적 설명과 실증

`ordered_response_barrier_ode/runtime.py:624`–655는 z_layer=8의 fixed target을 계산한다. 658–682행의 Phi/JVP readout은 `model.layers.8`의 subject-token output이며 final logits가 아니다. Editable weights는 `model.layers.[4..8].mlp.down_proj.weight`다.

이 architecture의 residual block 구조에서 L8 down-projection update는 바로 readout되는 block output에 직접 가산되는 경로를 갖는다. 이전 layer update는 추가 nonlinear block/normalization/attention 경로를 거쳐 Phi8에 도달한다. 따라서 raw residual을 RHS로 하는 native-form dictionary에서 L8가 target residual에 가장 정렬되는 것은 구조적으로 예상 가능한 가설이다. Native geometry/keys/contexts 때문에 D8 response가 residual에 정확히 동일하다는 주장은 아니다.

공개 per-node response–residual cosine 독립 요약:

| 모델/범위 | L4 median | L5 median | L6 median | L7 median | L8 median |
|---|---:|---:|---:|---:|---:|
| Llama B1–9, 36 node | .251038 | .368100 | .508258 | .675544 | .997744 |
| Qwen B1–10, 40 node | .234970 | .264810 | .374788 | .520618 | .996040 |

Llama B10는 예외로 L8 cosine이 약 `[-.000363, .000395]`, L7도 `.000450`–`.002473` 수준이다. 그래서 “모든 batch에서 같은 정상 L8 집중”으로 묶을 수 없다. B10는 다른 layer로 유용하게 이전된 것이 아니라 response alignment와 physical action이 함께 붕괴한 특이점이다.

현재 controller는 layer-diversity 목적/균등배분 quota/각 layer 누적 사용량 cap을 갖고 있지 않다. M+L2 metric으로 평가해도 L8의 response per cost가 좋으면 계속 L8을 선택하는 것이 objective에 맞는 해다. 이는 forgetting이 없다는 뜻도, L8 집중이 해롭다는 뜻도 아니다.

## 7. 실제 history-cost shadow는 무엇을 보였는가

`alpha_native_response_ode_v31_sequential/telemetry.py:23`–29는 cold M0=0에 대해 `G_initial[l,l] = L2 ||D_l||_F² / (q_l qN_ref)`를 실제 current coordinates로 계산한다. G_actual=I를 초기 shadow에 그대로 재사용한 버그는 없다. 45–53행에서 e/Psi/dictionary/qref를 고정하고 history cost만 바꾸어 재해를 구한다.

공개 `history_cost_shadows.csv` 독립 계산:

| 모델/node0 | actual 대비 initial-cost shadow의 native angle | actual/shadow native norm | L8 candidate raw action 중 history 비율 |
|---|---:|---:|---:|
| Llama B1 | 0도 | 1 | 0% |
| Llama B5 | .027480도 | .99909923 | 2.118232% |
| Llama B10 | 0도 | .9999999985 | 5.640341% |
| Qwen B1 | 0도 | 1 | 0% |
| Qwen B5 | .000236도 | .99998865 | .050452% |
| Qwen B10 | .274436도 | .98330540 | 75.372473% |

Qwen B10에서 actual L8 physical coefficient `.9267687`, initial-history-cost shadow `.9425205`다. L7 coefficient `.00556454` 대 `.00111481`로 relative 차이는 있으나 L8에 비해 절대량은 작고 전체 field cosine `.99998853`다.

따라서 “M이 전혀 안 들어가서 L8에 몰렸다”는 설명은 맞지 않는다. 동시에 “history가 큰 physical re-routing을 만들어 Qwen NS를 올렸다”는 설명도 현재 shadow로 입증되지 않는다. 특히 Qwen B5의 cost-only effect는 거의 0이다. 이것은 **M이 dictionary 생성에 준 효과까지 0이라는 뜻이 아니다**. 해당 효과와 early-layer key-drift/operating-point의 영향은 지금 자료에서 분리되지 않았다.

## 8. 부가 구현 한계와 다음 분석에 필요한 구분

- P는 stock asset 그대로 native solve에 들어간다. `native_response_ode_v31/fidelity.py:67`–70은 `||P right-right||/||right||`를 observation으로 기록한다. 이 항목 자체에 별도 hard threshold를 거는 코드는 없다. 기존 warm G0의 값은 대략 Llama `4e-6`, Qwen `2.3e-5`–`1.32e-4` 범위였다. Exact projector feasibility 또는 zero pretrained disturbance를 주장하지 않는다. 이 사실은 이번 실패의 원인이라는 뜻이 아니다.
- Source preserves `cache_c_new` lifecycle. 변수 `_alpha_cache_entry_is_zero`는 이름과 달리 warm에서도 initialized flag를 보관하는 부분이 있으나 실제 stock semantics와 맞으며 reset-to-zero bug는 관측되지 않는다.
- L8-only 경로도 현재 `dictionary.build`는 먼저 5 layer direction을 만든 뒤 L8만 남기므로 extra native solves가 남는다. JVP는 L8만 수행한다. 이는 성능/비용 최적화 기회이지 main JV 결과 무효화 근거가 아니다.
- Qwen NS 상승은 CounterFact neighborhood NLL preference라는 평가 관측이다. 위 native barrier는 그 NS를 직접 미분/제약하지 않는다. Benefit의 원인에 대한 인과 판정은 actual L8-only comparator, 같은-state/cost decomposition 및 paired neighborhood transition과 함께 해야 한다.
- λ sweep은 objective sensitivity를 보여줄 수 있으나 M omission bug를 고치는 작업이 아니다. Llama B10의 매우 작은 N0와 큰 H를 λ만으로 해결할 수 있다는 보장도 없다. Primary normalization을 바꾸면 다른 science version으로 분리해야 한다.

## 9. L8 집중 JV와 “AlphaEdit을 L8에만 적용”은 같은가

일반적으로 같지 않다. 비교할 세 실행을 구분해야 한다.

| 실행 | 후보/support | integration | native scalar | history/reference |
|---|---|---|---|---|
| Official을 진짜 단일-layer inventory로 구성 | L8만 | one-pass 한 번, residual divisor 1 | stock full native write | 단일-layer hparams/P/cache 정책을 새로 정확히 binding해야 함 |
| JV_NATIVE가 결과적으로 L8에 집중 | L4–L8 후보 중 joint solve가 선택 | T2/N4/h.5, 매 node current response | nonnegative joint c와 q whitening | 전체 5-layer entry qref, 전체 5-layer history |
| 현재 승인 L8_ONLY_NATIVE comparator | admissible actual support만 L8 | T2/N4/h.5, current L8 direction/JVP | same-state `c8=[g8]+/(H88+.1)` | 전체 5-layer hparams/P indexing/qref/history 유지 |

현재 L8_ONLY_NATIVE가 가장 직접적인 controlled comparison이다. Official hparams.layers를 `[8]`로 바꾸면 stock `P[i]`, `cache_c[i]`, layer inventory와 finalization mapping도 달라진다. 원래 5-layer P의 첫 slice가 L4인데 `i=0`을 곧바로 L8로 해석하는 구현은 잘못된 projector를 쓸 수 있다. 따라서 단순 `[8]` 치환을 같은 comparator라고 부르면 안 된다.

JV coefficient가 **정확히** L8 support 한 개라면 해당 node에서 full NNLS와 support-restricted closed form은 같아진다. 그러나 현재 실제 trajectory는 작은 다른-layer coefficients가 남고, 각 arm이 이후 다른 state에 도달하므로 endpoint 동일성이 자동으로 따라오지는 않는다.

수식상 L8 raw native direction을 D, 그 weighted raw response를 `p = D Phi_tilde[D]`, q=`Q_N(D)`라 두면 L8-only physical velocity는

`F = a D`,

`a = [e^T p]+ / (||p||² + lambda q)`.

이것은 `c8/sqrt(q8)`를 raw coordinates로 다시 쓴 식이다. `h`는 objective 안에 없고 실제 update `h a D`에 한 번만 들어간다. a의 상한 1은 없다. response가 약하거나 달라지면 a>1도 가능하다. Stock one-pass는 `D`를 한 번 전량 적용한다. 따라서 같은 layer에만 쓴다는 사실만으로 같은 write가 아니다.

### 제한적인 동치/감쇠 예

다음은 관측값이 아니라 설명용 수학적 특수 조건이다.

1. 앞 layer들은 고정되고 L8 keys와 entry metric은 변하지 않는다.
2. Phi8 readout은 L8 down-projection에 대해 affine이다.
3. native direction이 목표 residual을 정확히 실현하여 `p=e`다.
4. lambda=0, e!=0이고 기타 layer write가 정확히 0이다.

그러면 `a=1`, 한 Euler step 후 residual은 `(1-h)R`이다. h=.5, N4이면

`R4=(1/2)^4 R0=R0/16`,

`Delta W4=(15/16) D_entry`.

이때 `V4/V0=1/256`이다.

Stock full one-pass `D_entry`와는 finite horizon에서 다르다. 이 제한적 예에서 h=1의 한 step은 stock write와 같고, 충분히 긴 residual flow의 limit도 같아질 수 있지만 현재 h=.5,T2,N4는 full native endpoint와 같지 않다.

lambda>0, p=e이면 `a=||e||²/(||e||²+lambda q)<1`다. residual shape가 단순 배율로만 변하면 e²와 q가 같이 줄어 a는 일정하고 `R_N=(1-h a)^N R0`; 더 감쇠된 finite write가 된다. 이것은 해당 이상적 조건에서의 설명이며 현재 실험 전반을 scalar slowdown으로 확정한 것은 아니다.

실제 native map은 P/regularization/history 및 여러 context 때문에 `p=e`가 아닐 수 있다. 고정된 L8 keys에서도 residual의 request/feature 방향이 반복적으로 바뀌면 `D_n = L(R_n)` 자체가 첫 D와 평행하지 않을 수 있다. 따라서 L8 support만 같더라도 iterative residual correction과 response gain이 다른 endpoint를 만든다. JV에서 early layers가 조금이라도 바뀌면 keys/terminal response도 추가로 달라진다.

이 실험에서 밝혀야 할 구분은 다음이다.

- JV ≈ L8_ONLY_NATIVE라면, 현재 이득이 adaptive multi-layer mixing 없이도 설명될 수 있다.
- 둘 다 Official보다 Qwen NS가 높다면, single-layer localization, iterative residual correction, response-based amplitude, horizon/strength 차이가 남는 경쟁 설명이다.
- JV가 L8_ONLY_NATIVE보다 같은 current efficacy에서 retention/locality 이득이면 작은 multi-layer mixture의 유용성 근거가 생긴다.
- Official single-layer one-pass까지 없으면 “반복 L8 JV가 stock L8 write 자체보다 낫다”는 주장은 미검증이다.

새 arm 실행이나 현 L8 실험 변경은 이 감사에서 승인/수행하지 않았다.

## 10. 증거 범위

코드와 publication scalar tables의 재계산은 독립 수행했다. Raw server2 W/M/checkpoint/prompt tables는 접근하지 않았다. 새 model/JVP/FD를 실행하지 않았다. 따라서 output/raw evaluator의 bitwise 재검증이나 현재 old-key drift 직접 측정, Qwen locality benefit의 causal attribution은 미실시다. SH4 L8 takeover/rerun은 읽거나 변경하지 않았다.
