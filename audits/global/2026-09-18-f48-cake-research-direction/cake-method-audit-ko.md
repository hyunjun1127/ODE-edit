# CAKE의 causal score·분배·optimality 범위 감사

2026-09-18. 대상은 **CAKE: Causal-Guided Adaptive Knowledge Editing for LLMs**이며 Circuit-aware CaKE가 아니다. 원논문과 공식 저장소, server4의 봉인 실행 소스를 읽었다. 모델 실행·원자료 변경·GPU 제출은 없다.

**판정: 공개 CAKE는 고정된 causal prior를 쓰면서 현재 residual에 따라 순차 write를 바꾸는 방법이다.** score/softmax weight 자체는 online batch별 최적화가 아니다. 그렇다고 residual까지 고정된 단순 static method라고 부르면 틀린다. softmax와 quadratic allocation에는 조건부 최적화 근거가 있지만, 실제 efficacy/generalization/locality의 최적 분배를 증명한 것은 아니다.

## 1. 원논문이 정의한 score와 수학적 문제

[원논문](https://aclanthology.org/2026.acl-long.918.pdf): PDF22쪽/인쇄20061 Appendix E–E.1은 base model이 아는 fact1000개, noise10회, Llama noise0.02, 마지막 subject token의 post-MLP 복원을 기술한다. Llama tracing 범위는 기존 critical set4–8이다. Eq67–69는

\[
s_l=\frac1N\sum_i\left[\mathbb E_\xi p(o_i\mid corrupt_i,restore_l)-\mathbb E_\xi p(o_i\mid corrupt_i)\right].
\]

PDF3쪽 Eq2–3의 softmax는 주어진 score와 온도 아래 entropy-regularized objective의 해다. PDF4쪽 Eq4–5는 주어진 weight와 선형 sensitivity 아래 residual allocation 문제다.

\[
w_l=\operatorname{softmax}_l(s/\tau),\quad
w^*=\arg\max_{w\in\Delta}\{s^Tw+\tau H(w)\};
\]
\[
\min_{\delta}\frac12\sum_l\|\delta_l\|^2/w_l\quad
\text{s.t. }\sum_l M_l\delta_l=R,\qquad
\delta_l^*=w_lM_l^T\Big(\sum_jw_jM_jM_j^T\Big)^{-1}R.
\]

실용 구현은 sensitivity를 identity로 근사한다(PDF4쪽 §3.2.3). 초기 attribution을 유지한다는 비용 절충은 PDF20쪽/인쇄20059 Appendix C.4가 명시한다. 온도0.1은 PDF23쪽 Appendix F.1의 조사에서 선택했다. 오류 분석은 국소 선형화, residual 감소와 attribution–mismatch 정렬 가정 등에 의존한다(§4.1/A.5). [수식·가정 원문](https://aclanthology.org/2026.acl-long.918.pdf#page=4)

이 수식에는 우리 실험의 S64 W0-KL, Past64 성공 ID, actual paraphrase score 제약이 들어 있지 않다.

## 2. 공식 구현에서 실제로 변하는 것

공식 저장소의 확인 commit은 `c8243e1d7e43ca9cf64d552f96221fcb9561aac2`다. [Cake_main.py](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/Cake/Cake_main.py#L34)는 config의 score로 weight를 만든다. L68–114에서 마지막 선택층 L8의 target z를 만든 뒤, L116–166에서 L4→L8을 방문하며 현재 key와 L8 readout을 재계산한다. L132–147의 실제 규칙은

\[
R_l=z_8^*-z_8(W_{<l}),\qquad
\eta_l=\frac{w_l}{\sum_{k\ge l}w_k},\qquad
r_l=\eta_lR_l.
\]

이 r을 AlphaEdit형 solve에 전달한다. 같은 w라도 앞선 write가 달라지면 R과 다음 write가 달라진다. 반면 w와 η의 수치 일정은 동일 config라면 반복된다. active apply 경로는 전체 Jacobian을 계산하거나 actual quality로 w를 탐색하지 않는다. 별도 `compute_optimal_deltas` 함수(L284–303)는 apply에서 호출되지 않으며 전달받은 `M_list`도 계산에 사용하지 않는다.

### 기본 Llama 값의 CPU 재계산

[공식 Llama 설정](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/hparams/Cake/Llama3-8B.json)은 physical layers `[4,5,6,7,8]`, temperature0.1을 사용한다. `causal_scores`의 key0–4는 layer 배열의 위치다. 아래 w/η는 해당 숫자로 독립 계산했다.

| physical layer | config score | softmax w | 현재 remaining residual에 곱하는 η |
|---|---:|---:|---:|
| 4 | 0.4812439084 | 0.243951316 | 0.243951316 |
| 5 | 0.4743820429 | 0.227773116 | 0.301267789 |
| 6 | 0.4656370878 | 0.208700555 | 0.395060018 |
| 7 | 0.4440660179 | 0.168206060 | 0.526342966 |
| 8 | 0.4335190654 | 0.151368953 | 1.000000000 |

이 값들은 모두 양수다. 기본 설정은 exact sparse support selection이 아니다. η8=1은 **그 시점에 남은 residual 전부**라는 의미이며 최초 residual 전체를 L8이 혼자 담당한다는 의미가 아니다. w와 η를 우리 a4/a8 endpoint gate에 그대로 대응시킬 수 없다.

| 구분 | CAKE 공개 구현 | 현재 sequential local-z v2 |
|---|---|---|
| Target | entry에서 최종 선택층 L8 target | L4 target 후, 달라진 prefix에서 각층 local target fresh fit |
| 계수 의미 | 현재 global residual의 층별 비율 | 현재층 native endpoint 대비 write 강도 |
| 계수 결정 | 고정 causal score + temperature | 실제 completed endpoint의 E/H/ID guard와 S64 KL |
| Layer support | 기본 설정에서 L4–L8 모두 방문 | continuous boundary 및 exact-zero pruning |
| 현재-state 의존성 | key/readout/remaining residual | local target/key/solve, score와 최종 gate 선택 |

## 3. “Static인가, heuristic인가, optimal인가”에 대한 정확한 답

**Static은 어떤 변수를 말하는지 지정해야 한다.** CAKE의 score prior와 softmax weight는 고정이다. residual과 weight update는 현재 모델에 의존한다. 우리 F48도 gate(.75,.5)는 고정이지만 L8 target은 약화된 L4 이후 모델에서 계산하므로 write vector는 고정이 아니다. 고정 gate/weight와 고정 실제 update를 같은 것으로 취급하면 두 방법 모두 잘못 설명하게 된다.

**Heuristic이라는 단어도 나누어 써야 한다.** score는 측정한 causal attribution이므로 근거 없이 찍은 숫자는 아니다. softmax는 주어진 score와 entropy 목표를 최적화하는 원리 있는 변환이다. 그러나 score를 actual edit utility/cost의 대리값으로 택하는 것, critical set, temperature, identity sensitivity를 택하는 것은 모델링과 실용적 선택이다. 따라서 “측정 기반 prior를 쓰는 근사 allocator”가 정확하며, “실제 locality 제약하에서 최적 layer를 찾아낸 방법”이라고 확대하면 안 된다.

다음은 논문을 그대로 요약한 것이 아니라 **표시된 수식에 대한 독립 수학적 점검**이다.

1. Softmax 최적성은 objective가 `s·w+τH(w)`로 주어졌을 때의 이야기다. s가 실제 efficacy gain이나 preservation cost를 정확히 나타내는지는 이 최적성으로 증명되지 않는다.
2. Quadratic residual 문제는 w>0, 고정 M, 일차 모델, feasible R을 전제로 한다. 일반 inverse 표현에는 `ΣwMMᵀ`의 가역성도 필요하다. singular이면 R이 range 안에 있는지와 pseudoinverse 처리가 필요하며 strict convexity만으로 feasible 여부까지 보장되지는 않는다.
3. 여기서 최소화하는 것은 가중된 **activation residual norm**이다. 실제 parameter update norm은 key/solve operator에 좌우되고, output KL이나 paraphrase 성공은 별개의 함수다.
4. M≈I라면 δ_l=w_lR가 나오지만, 유한 write 뒤 key와 downstream response가 바뀌는 실제 모델에서는 그 등식으로 global optimum이 따라오지 않는다. 현재 residual 재계산은 이 불일치를 줄이기 위한 적응이다.
5. “높은 causal score이면 작은 write mismatch”라는 순서가 맞더라도 그 층이 “같은 efficacy에서 가장 낮은 W0 KL”을 준다는 결론은 추가 가정 없이는 나오지 않는다. 원래 object를 activation patch로 되살리는 개입과 새로운 object를 shared weight로 쓰는 개입은 서로 다르다.

## 4. 재현성 및 경계값에서 발견한 제한

### Score 원수치까지의 연결은 별도 검증이 필요하다

[공식 Llama tracer](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/experiments/causal_trace_llama3.py#L26)는 noise0.02, center layer4–8, `None/mlp/attn` 분기를 실행한다. 그러나 last-subject-only나 window1을 명시하지 않는다. [tracing helper](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/rome/causal_trace.py#L304)는 기본 `token_range=None`, `window=10`이고 MLP 분기에서 여러 인접층을 복원한다(L523–552).

따라서 공개 entry script를 그대로 실행한 산출물이 논문 E.1의 단일 last-subject/post-MLP AIE 및 config score에 곧바로 대응한다고 단정할 수 없다. 본 조사에서는 해당 다섯 config 숫자를 만든 raw trace→token/layer 선택→noise/sample 평균→export의 연결 receipt를 확인하지 못했다. 이는 **score 생성 provenance 미확인**이며, CAKE의 editing loop가 잘못 실행됐다는 판정은 아니다.

### τ=0의 sparsity 설명과 코드 동작은 다르다

`compute_layer_weights` L268–270은 τ=0에서 argmax one-hot을 반환한다. 그런데 apply L142–145는 remaining weight sum<1e−6이면 η=1을 사용하고 해당 layer를 skip하지 않는다. 기본 score의 argmax인 L4에 one-hot을 넣으면 CPU scalar 재현 결과 η는 `[1,1,1,1,1]`이다. 앞층 solve 후 residual이 남으면 뒤층도 그것을 쓰게 된다. 따라서 공개 코드에서 τ=0을 그대로 single-layer edit와 같다고 보면 안 된다. **본 실험의 τ=.1에서는 이 threshold가 작동하지 않으므로 이 경계 문제를 기존 결과의 원인으로 해석하지 않는다.**

## 5. Server4의 실제 CAKE 실행과 연결

읽은 경로는 `/data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1/`이다. `preparation-v1/preparation.json`의 upstream head는 위 공식 commit과 같다. 원본 `Cake/Cake_main.py`는9,774bytes/SHA256 `d1ae01c7f1d5f6f8af899a589b8e407171c1a5dbb69b8ad6ff08e39f02823e20`로 공식 commit에서 받은 파일과 일치했다. 실행본은 미사용 `notebooks.util` import 제거 및 최종 LF에 대한 compatibility patch가 기록되어 있다. 해당 실행의 score/temperature는 위 표와 같다.

하이퍼파라미터 차이도 분리해야 한다. CAKE native는 L2=10, clamp=.5, v_weight_decay=.4이고, 현재 v2 local-z는 L2=1, clamp=.75, decay=.5다. target 위치/횟수까지 다르므로 CAKE와 F48/C48의 차이를 causal weighting 하나의 효과로 읽을 수 없다. 실행 결과·chain provenance의 상세 비교는 별도 담당 감사와 통합한다.

## 6. F48 관측에서 이어지는 연구 방향에 주는 의미

다음은 새 실험 결과가 아니라 **방법 감사에서 도출한 후속 판단 기준**이다.

- F48이 작은 efficacy/generalization 손실 대비 큰 locality 이득을 준다면, 먼저 “비싼 online 탐색 전에 유용한 저차원 고정 강도 정책이 존재한다”는 가설을 강화한다. 고정(.75,.5)의 최적성이나 unseen/장기 우월성까지 증명하지는 않는다.
- CAKE도 static prior를 유용하게 사용하는 사례다. 따라서 novelty를 “static 대신 dynamic”으로 잡는 것보다 **같은 target/write family에서 무엇을 관측하여 어떤 성능–보존–비용 tradeoff를 개선하는가**로 잡는 것이 명확하다.
- 강도 효과를 분리하려면 같은 layer/local-z family의 calibrated-static과 online controller를 비교한다. CAKE weighting 효과를 분리하려면 CAKE의 target/solver를 그대로 고정하고 uniform, causal, score-shuffle 같은 weight 조건만 바꾸어야 한다. 두 비교를 혼합하지 않는다.
- Causal prior를 online으로 다시 계산하는 후속은 그 비용을 들이기 전에, score가 실제 finite candidate의 “같은 rewrite quality에서 얻는 preservation gain” 순위를 예측하는지 검증하는 편이 직접적이다. causal recovery가 그 순위와 약하게 연결된다면 tracing 재계산을 늘리는 것이 해법이라는 보장은 없다.
- H/P risk로 candidate write를 평가한다는 repo 설계는 CAKE와 구별할 축이지만 우월성은 미검증이다. [기존 proposal의 CAKE 구분](/mnt/raid5/janghj/ODE-edit/project/proposals/ODE_BF_Dynamic_Layer_Proposal.md:188)은 residual 재계산을 이미 인정하고 있어 이번 확인과 일치한다. [repair layer feasibility 문서](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-17-repair-layer-feasibility-audit-v1.md:15)의 “activation tracing과 shared-weight preservation repair는 다른 개입”이라는 구분도 유지해야 한다.

CAKE를 무시하거나 만능 비교 대상으로 삼을 이유는 없다. 직접 비교에 필요한 것은 동일 입력과 평가뿐 아니라 target family, 강도, solver, 과거 보호, 비용을 나눈 대조다. 이것이 F48의 저비용 tradeoff를 출발점으로 삼으면서도 causal/static/dynamic 효과를 구분할 수 있는 경로다.
