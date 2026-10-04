# JLZ v12 구현 명세와 계측 계약

2026-10-04. 상태: **METHOD ONLY**. 이 파일은 구현 가능한 계산 순서와 관측 의미를 정한다. GPU 실행, qualification 완료, 작업 등록, dispatch를 뜻하지 않는다. NumPy 참조 코드는 수치 구성요소 검산용이며 production editor가 아니다. 본문은 [method-ko.md](method-ko.md), JSONL 이벤트 형식은 [telemetry-schema.json](telemetry-schema.json)을 따른다.

## 1. 실행 단위와 adapter

하나의 논리 batch는 entry 모델 `W_t`, entry history `H_t`, 실제 요청 수 `B`, 순서가 있는 전체 eligible write layer 집합 `L`을 갖는다. `B`, `m=|L|`, 층별 차원, native z layer, 문맥 수, target 길이, model/dataset 및 module 경로는 runtime 값이다. 마지막 부분 batch를 예정 batch 크기로 나누지 않는다. 향후 현 profile의 기본 규모는 2,000요청, BS100×20이지만 이 파일은 그 실행을 등록하지 않는다.

Adapter는 다음 내용을 값과 identity/hash로 제공한다.

- native rewrite/KL 문장과 context-group membership, target token IDs, subject lookup, NLL scoring 위치, KL readout 위치 및 full-vocabulary 방향 `current || entry`.
- full transformer-block output의 주입 위치, canonical subject 측정 위치, writer 입력 key 위치, weight orientation과 cast/add 순서. 주입 block output과 writer 출력 사이의 국소 가법성은 모델별로 검증한다. 후속 postnorm 등으로 `U k`가 단순히 해당 공간의 가법 변화가 아니면 같은 width라는 이유만으로 지원하지 않고 `UNSUPPORTED_ADAPTER`로 보고한다.
- native z layer의 anchor `a*_r`, 각 eligible layer의 canonical entry anchor `a_lr`, NLL의 target-token 평균 후 context 평균, KL의 native reduction, writer의 native nested context mean과 final history-key 추출 규칙. Loss context 가중치와 key 평균 가중치를 혼용하지 않는다.
- native coefficient, clamp, 학습률 및 Adam beta/epsilon, 모델·activation·moment·solve·history dtype, tokenizer padding/position/cache 의미. 현 profile 예시는 L4–L8, native z layer L8, loss layer31, `λ_K=.0625`, `λ_n=.5`, `c=.75`, `η_native=.1`, `ε_native=1e-8`이다.

모든 eligible layer는 첫 후보부터 변수와 gradient 대상이다. 초기 `u=0`이며 warm-up, layer pruning, 특정층 우선 순서, 최소 활성층 수, 강제 균등/다양성 조건은 없다. 사영 후 0이 된 층도 다음 후보에서 gradient를 받아 다시 진입할 수 있다. 구조나 native row 의미가 지원되지 않으면 명시적 `UNSUPPORTED_ADAPTER`로 끝낸다. 다른 모델에 현 profile 숫자를 자동 이식하지 않는다.

## 2. 계획의 상태와 계산

Fit 동안 모델 weight, entry teacher와 anchor를 고정한다. Anchor norm은 native activation/norm-reduction dtype(현 profile FP32)으로 entry에서 한 번 계산한다. Geometry/projection의 FP64 선택을 이유로 anchor norm을 FP64에서 다시 계산하지 않는다. `H_t`는 fit 목적/optimizer에 들어가지 않고 이후 native writer에 사용된다. 요청 `r`의 모든 native rewrite/KL context의 subject block output에 같은 `δ_lr=a_lr u_lr`를 주입한다. 한 요청의 모든 eligible layer 주입은 같은 forward에 존재한다. 아래층 주입에서 위층을 거쳐 loss로 가는 gradient를 끊지 않는다.

`F_r=NLL_r+λ_K KL_r`, `ρ_r=λ_n/a*_r`, `J_r=F_r+ρ_r Σ_l||u_lr||`로 둔다. 기본 production dtype는 model/u/moments FP32, geometry/projection FP64, history native CPU FP32다. 사영 후 FP32 cast한 저장 u의 feasibility를 `1e-6·max(1,c)` 허용오차로 검사한다. 위반 시 숨은 재사영·축소 없이 technical failure다. 공유 예산은 `Σ_l||u_lr||≤c`이며 **virtual 계획에만** 적용한다. 실제 weight, writer 잔차, 직접 실현량, 최종 hidden drift에는 이 부등식을 적용하거나 이를 상한이라고 주장하지 않는다.

요청별 상태는 `u_lr`, Adam `m_lr,v_lr,s_lr`, 실제 update counter `t_r`, 후보 번호, 종료 flag, 마지막 평가 후보의 loss와 post-injection canonical `z^v_lr`이다. `s_lr`는 layer gradient RMS의 제곱에 대한 beta2 EMA다. Moment는 사영 후에도 유지한다. 모델 parameter는 optimizer에 등록하지 않는다.

계속 진행하는 후보에서만 `g^F_lr=∂F_r/∂u_lr`를 한 번 backward로 계산한다. Norm subgradient를

```
n_lr = ρ_r * u_lr / ||u_lr||   if ||u_lr|| > 0
n_lr = 0                     otherwise
g_lr = g^F_lr + n_lr
```

로 더한다. 이 `g_lr`가 EfficiencyAdam 입력이다. 다른 norm smoothing/epsilon을 추가하지 않는다. Loss scalar와 이 gradient 구성의 일치는 CPU 및 adapter qualification에서 검사한다.

Optimizer는 첨부 EfficiencyAdam의 heuristic을 유지한다. 요청별 `η_u=η_native/a*_r`, `ε_u=a*_r ε_native`를 쓰고 `g_lr`로 좌표 moments와 layer RMS EMA를 갱신한다. Bias-corrected `s_hat_lr`의 **해당 요청 전체 eligible layer 평균**으로 `γ_lr=sqrt(s_hat_lr/mean_l s_hat_lr)`를 계산한다. 분모가 정확히 0이면 전 층 `γ=1`이다. 다른 요청의 gradient를 gamma 평균에 섞지 않는다. Adam step 뒤 group-L1 ball로 정확히 사영하고, `τ_proj`와 active set을 기록한다. 이 알고리즘이 원목적 KKT 점으로 수렴한다는 보장은 없다.

## 3. 요청별 평가·종료·microbatch

후보0은 `u=0`이다. 평가 순번 1..25와 후보 번호 0..24를 구별한다. 후보별로 native row 전체의 `J_r`를 완성한 뒤 다음 순서를 적용한다.

1. Loss, hook capture, 모델/optimizer 상태가 nonfinite이거나 계약이 깨지면 technical failure.
2. `J_r < ε_stop`이면 해당 요청을 종료한다. 현 profile의 `ε_stop=.05`다. 후보0이면 `ZERO_STEP`, 그 이후이면 `OBJECTIVE_THRESHOLD`다.
3. 후보24(25번째 평가)이면 `EVALUATION_BUDGET`으로 종료한다.
4. 나머지 요청에만 backward, Adam update, 사영을 수행한다.

요청당 최대 25 logical loss 평가와 24 backward/update다. 마지막으로 **평가된 유한 후보**를 채택하며 update만 한 미평가 후보는 채택하지 않는다. 학습 도중 `z^v_lr`는 기존 forward의 post-injection canonical hook에서 매번 capture/overwrite한다. 종료 요청의 값을 동결하므로 terminal z를 얻기 위한 추가 forward는 없다. KKT나 ACC/PS/NS를 이용한 종료, 과거 후보 best-of 선택, 추가 step은 없다.

Microbatch는 실행 편성일 뿐이다. 기본 편성은 각 요청의 native rewrite/KL row 전부를 같은 microbatch에 넣어 `J_r`를 완성하고 stop을 판단한다. 여러 요청을 묶을 수 있으며 native token/position 의미를 유지한 padding을 사용한다. 요청의 row를 여러 조각으로 나누어야 하는 adapter는 모든 조각의 loss와 그래프를 보존하여 전체 `J_r`가 나온 후에만 backward 여부를 결정한다. stop 판정을 위해 no-grad forward 후 같은 후보를 재평가하는 경로는 기본 구현에 없다.

Backward scalar는 그 microbatch의 **계속 진행하는 요청의 `F_r` 합**이다. 보고용 `mean_r J_r`를 그대로 backward하거나 microbatch 수, 전체 B, 남은 활성 요청 수로 다시 나누지 않는다. Request 내부 각 row에는 원 native token/context reduction weight를 정확히 한 번 적용한다. `g^F_r`가 독립 요청 계산과 같아야 한다. 같은 round에서 모든 필요한 microbatch backward가 끝난 뒤 요청별 moments/update를 진행하여 shared tensor의 in-place version 변경을 피한다.

종료된 요청은 이후 gradient, moment, step counter, projection에서 제외한다. 요청별 계획이 독립이라는 말은 동일한 고정 `W_t`에서의 fit에 한정한다. Writer는 full-batch key matrix를 사용하며 요청 간 coupling이 남는다. Zero-step 요청도 원 batch의 key/target column으로 유지한다. 그 요청의 계획이 0이어도 다른 요청의 weight write로 실제 hidden이 바뀔 수 있다.

```
entry = freeze_entry(W, H, ordered_requests, native_adapter)
state = zero_request_states(all_eligible_layers, entry.anchors)
for candidate in 0..24:
    active = requests_without_terminal
    for mb in pack_complete_native_requests(active):
        F, norm, J, post_injection_z = forward_native_rows(mb, state.u)
        validate_finite_and_capture(J, post_injection_z)
        terminal = (J < threshold) or (candidate == 24)
        freeze_last_evaluated_state_and_z(terminal)
        continuing = mb - terminal
        if continuing:
            gF = backward(sum(F[r] for r in continuing))
            g = gF + selected_norm_subgradient(state.u)
            retain_request_gradients_and_diagnostics(gF, g)
    for r in active_without_terminal:
        state[r] = efficiency_adam_step(state[r], g[r])
        state[r].u, tau_proj = project_shared_budget(state[r].u, c)
        record_update_and_projection(r, tau_proj)
    if all_requests_terminal: break
write_transaction(entry, all_frozen_terminal_z, all_requests)
```

## 4. KKT 진단은 optimizer와 분리한다

`τ_proj`는 해당 Adam proposal에 대한 Euclidean 사영의 승수다. 원목적 constraint multiplier `μ`와 다른 값이며 동일한 이름/열을 쓰지 않는다. `γ`도 효율의 실증값이나 KKT multiplier라고 부르지 않는다.

Backward를 실제 수행한 후보에서만 `g^F`, norm subgradient, total gradient를 분리 기록한다. `e_l=u_l/||u_l||`인 활성층에서 원문제 stationarity는 `g^F_l+(ρ+μ)e_l=0`, 비활성층에서 `||g^F_l||≤ρ+μ`다. Feasibility, `μ≥0`, `μ(Σ||u||−c)=0`도 별개다.

진단용 `μ_hat`은 내부점에서는 0이고, 경계점에서는 `max(0,-mean_active(<g^F_l,e_l>)-ρ)`로 둔다. 본 실행의 c는 양수다. 진단상 활성층은 raw `||u_l||>1e-10`, 경계는 `|c−Σ||u|||≤1e-6·max(1,c)`로 분류한다. 활성층이 없으면 0이다. 반경0의 CPU 수치검사에서는 `max(0,max_l||g^F_l||−ρ)`를 별도로 쓴다. 이 tolerance를 metadata에 기록하며 합격선으로 사용하지 않는다. 진단 분류로 실제 u를 바꾸거나 작은 층을 pruning하지 않는다. 다음 값을 기록한다.

- 활성층 vector stationarity norm과 radial/tangential 성분, 비활성층 `max(||g^F_l||−ρ−μ_hat,0)`.
- request 전체 stationarity residual, primal violation, complementarity residual 및 multiplier estimator 이름.
- task/norm/total gradient norm, 원본 plan support와 budget 사용률.

종료후보는 stop-before-backward이므로 gradient/KKT 필드를 `null`, 사유를 `NO_BACKWARD_TERMINAL`로 기록한다. 이전 후보 수치를 terminal 진단으로 다시 표시하지 않는다. 따라서 terminal KKT 수렴은 보장되지 않으며 기본 실행에서 terminal KKT residual 자체를 계산하지 않는다. 별도 고비용 probe/arm을 자동 추가하지 않는다. KKT residual이나 layer 다양성을 scientific gate로 쓰지 않는다.

## 5. Native 순차 write transaction

Fit 완료 후 낮은 층부터 모든 eligible layer를 순서대로 처리한다. 이때 terminal `z^v`는 고정한다.

1. 앞층의 실제 write가 반영된 현재 모델에서 native rewrite keys와 canonical `h^cur_lr`를 재측정한다. Key는 native nested mean으로 `K_l`을 만든다. Entry key를 재사용하지 않는다.
2. `R_l=[z^v_lr−h^cur_lr]_r`를 만든다. `R_l=D_l+E_l`에서 `D_l`은 자기층 계획, `E_l`은 virtual pre-injection hidden과 actual prewrite hidden의 차이다. E는 아래층 미실현 외의 경로 mismatch도 포함한다.
3. `A_l=λ_C C0_l+H_t,l`, `P_l=(A_l+K_l K_l^T)^−1 K_l`, `U_l=R_l P_l^T`를 native solve/weight orientation으로 계산한다. 명시적 inverse, inverse-M 보상, remaining-layer divisor, 추가 residual top-up, writer-cost objective는 없다.
4. Native dtype/cast/add 순서로 weight에 적용한다. Ideal `U_l`과 실제 FP32 등 weight 차이에 native orientation을 적용한 `U^eff_l=orient(W_after−W_before)`를 구분한다. `cast(U_l)`만으로 실효 update를 대신하지 않는다. 자기층 계획이 0이어도 R은 0이 아닐 수 있으므로 계획 support로 writer를 skip하지 않는다.
5. 다음층에서 keys와 hcur를 새 모델 상태로 측정한다. 모든 층 완료 후 최종 native keys로 history를 각층 정확히 한 번 append한다.

Full-batch ridge solve를 microbatch별 별도 write로 바꾸지 않는다. Native C0/H normalization, history dtype 및 순서도 유지한다. 임의 symmetrize/jitter나 fallback solver로 차이를 숨기지 않는다. 지원되는 native 수치 계약을 만족하지 못하면 technical failure다.

관측은 다음 네 종류를 합치지 않는다.

| 관측 | 정의 |
|---|---|
| 계획 | `δ_lr`, `||u_lr||`, plan support |
| writer 요구 | `r_lr`, 자기 몫 δ, inherited mismatch `r−δ` |
| 직접 실현 | ideal/effective `U_l K_l[:,r]`, canonical `U_l k_can`, 각 native rewrite context의 `U_l k_context` |
| 실제 net drift | 최종 canonical block output `h^final_lr−h^0_lr`; 아래층 효과 포함 |

실제 자기층 국소 변화 `h^final_lr−h^cur_lr`도 별도 기록한다. 이후 상위층 write가 하위층 output을 바꾸지 않는 구조인지 adapter가 검증하고, 그 조건 아래 final history-key extraction의 기존 forward hook에서 `h^final`을 함께 얻는다. canonical/native context mapping과 이 capture의 가용성을 qualification에서 확인한다. Missing canonical capture를 0으로 채우거나 새 probe를 몰래 실행하지 않는다.

직접 실현은 key mean과 개별 context에서 모두 기록한다. 평균한 key의 작용 norm은 context별 작용 norm 평균과 같지 않다. 계획 share는 `||u_l||/Σ||u||`이며 합이 0이면 0벡터와 `NO_PLANNED_EDIT`를 기록한다. 실제 실현 share는 canonical `||U^eff_l k_can||/a_l`를 층간 정규화하고, 합이 0이면 `null`과 `ZERO_REALIZED_TOTAL`을 기록한다. Mean-key share로 이를 대체하지 않는다. 그 밖의 ratio 분모가 0이면 `null`과 사유를 기록하며 epsilon을 넣어 유한 비율로 만들지 않는다. 개별 zero-plan 요청도 다른 column 때문에 직접 변화가 생길 수 있다.

## 6. Failure, rollback과 완료 경계

Transaction 시작 전에 selected weights, H, RNG/native mutable cache와 batch ledger의 entry identity를 보존한다. 큰 weight 사본은 dtype를 유지하여 CPU에 둘 수 있다. Fit 중 모델 weight/H가 바뀌면 실패다. Write 중 solve/state/hook 오류, nonfinite, history append 실패가 나면 weight와 H뿐 아니라 같은 transaction의 cache/ledger/RNG를 entry 상태로 복원하고 복원 검증 결과를 기록한다. 복원 여부가 불확실하면 `ROLLBACK_FAILED`로 중단하며 다음 batch로 가지 않는다.

모든 층 write와 final-key history append, state validation이 성공한 뒤에만 batch commit을 게시한다. 실패를 0점 성능으로 집계하지 않는다. 허용된 25회 안에 목적이 충분히 줄지 않거나 배분이 한층에 집중했다는 이유만으로 실패 처리하지 않는다. 추가 iteration, coefficient 변경, 다른 writer로의 자동 재시도는 없다.

## 7. 메모리와 계산 절감

- Fit에는 candidate별 ridge solve, actual writer materialization, replay/reference/pulse/controller가 없다. Native model weight는 frozen이고 backward 대상은 주입 변수뿐이다.
- Microbatch로 native row forward/backward를 편성하되 논리 batch writer와 독립 요청 gradient를 보존한다. Batch 확대의 GPU 이용률 이익과 FLOPs 감소를 구분해 기록한다.
- 필요한 scoring/KL 위치에서만 full-vocabulary head를 계산할 수 있다. Vocabulary를 줄이거나 evaluation/context 표본을 줄이지 않는다.
- Native token IDs와 position 의미가 같은 경우에만 padding trim 또는 entry-prefix cache를 사용한다. Gradient가 필요한 주입 이후 경로를 detach/cache하지 않는다.
- Entry anchor/teacher와 terminal z는 필요한 값만 저장한다. Teacher CPU offload는 dtype/값을 보존하며 양자화·축약을 하지 않는다. Fit graph는 해당 backward 뒤 해제한다.
- Writer는 한층씩 solve/update한다. Commit용 full-batch K와 rollback 상태를 유지하며 candidate graph를 writer까지 보관하지 않는다. Final canonical capture는 필수 history forward와 결합한다.

각 효율화는 native task loss, 주입 gradient, token/position, 후보/stop와 실제 commit 결과 parity를 통과한 경우에만 켠다. 지원되지 않는 최적화는 명시적으로 끄고 동일한 방법을 유지한다.

## 8. 검사 범위

필요한 CPU 검사 범위는 projection feasibility/해, 선택한 norm subgradient, heuristic Adam 수치, anchor-scaled lr/epsilon 변환, request별 stop/moment 동결, partial batch와 microbatch scaling, JSON event 관계, synthetic rollback이다. 이 목록 전체가 이미 구현·통과했다는 뜻은 아니다. 현재 reference의 실제 완료 검사 목록/개수는 reference README와 테스트 출력으로 한정한다. 그중 독립 요청 moments의 interleaving 검산은 model loop의 요청별 종료나 microbatch native parity를 검증한 것이 아니다. 실제 종료 loop, terminal capture, transaction rollback과 JSONL 전체 관계 validator는 production qualification 대상이며 이 문서에서 완료로 인증하지 않는다. 참조 검사 통과를 transformer native parity나 편집 성능의 증거로 부르지 않는다.

추후 실제 모델 qualification은 동일한 native row/token/lookup/readout, NLL/KL 방향·reduction, 모든층 주입 gradient, zero-step/early-stop/25번째 후보, frozen entry, terminal z hook, actual key/hcur 재측정, native solve/cast/history, rollback 및 final-key capture를 확인해야 한다. 작은 입력의 통과 범위를 전체 model/dataset/batch로 확대하지 않는다. 이 문서 작성 시점에 그 모델 검사를 실행했다고 주장하지 않는다.

성능 평가는 **ACC와 RS/PS/NS를 같은 수준으로** 보고한다. ACC는 teacher-forced target strict, token micro, prompt macro와 각각의 정확한 분자/분모를 기록하며 자유생성 정확도와 혼용하지 않는다. Current/at-write, seen-prefix, 고정 cohort retention, W0-correct neighborhood retention을 구별한다. NLL(new/true/desired)과 margin(true NLL−new NLL)에도 family와 모집단을 붙인다. Dataset에 없는 metric은 `NOT_APPLICABLE`과 근거를 쓰며 다른 metric으로 대체하지 않는다. 평가 문장/성능을 fit, 계수 선택, stop에 쓰지 않는다.

## 9. Telemetry 수록 및 검산

`telemetry-schema.json`은 한 JSONL event의 구조를 검증한다. 모든 event는 run identity, batch/candidate/request/layer identity와 payload를 가진다. Raw prompt/tensor 대신 ordered request/context hash와 필요 시 artifact identity를 사용한다. JSON의 NaN/Infinity는 허용하지 않는다. 미측정, 분모0, 해당없음은 명시된 사유와 `null`로 기록한다.

Schema 외 관계 검산은 필수다: 요청별 후보0..terminal 연속성, 평가≤25/update≤24, terminal 뒤 update 없음, backward/KKT availability 일치, terminal z의 candidate identity, all-layer/모든 native context coverage, planned-budget 범위, residue decomposition, history 1회, commit/rollback 정합, metric num/den와 실제 모집단·target-token 수 일치. 수치 tolerance는 adapter qualification에서 정하고 metadata에 기록한다. 이 검산은 과학 성능·KKT·배분 다양성 gate가 아니다.
