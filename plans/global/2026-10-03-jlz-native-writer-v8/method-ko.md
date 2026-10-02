# JLZ v8 replay 없는 native writer와 정규화 좌표 공동 최적화

2026-10-03. 이번 수정은 사용자 결정에 따라 **전체 ours 목적의 구성은 유지하되 과거 문장 replay를 제거하고, writer와 history 통계를 MEMIT-H에 맞추며, Adam 첫 갱신의 전면 clamp 포화를 교정**한다. 모든 eligible layer의 subject δ를 공동 최적화하고 한 층 집중을 허용한다. 이는 방법 설계 revision이며 실행 중인 v7을 변경하거나 새로운 GPU 실험을 제출하는 지시가 아니다.

## 변경의 범위

| 부분 | v8 결정 |
|---|---|
| Native 학습 문장·subject lookup·NLL/KL readout | 유지 |
| Native NLL·current‖entry KL·nonsquared norm | 유지 |
| Ours G/E 배분 및 current actual 정합·KL | 유지, G/E는 native writer에 맞게 정의 |
| 과거 문장·KL teacher·NLL anchor replay | 제거 |
| 과거 key의 history H | MEMIT-H 방식으로 유지 |
| Writer key | 모든 native context를 사용하되 요청별 native 그룹 평균으로 집계 |
| History update | 최종 actual 모델의 평균 key outer product를 native 방식으로 누적 |
| Adam 좌표 | 물리 δ 직접 최적화 → 크기·차원·층수로 정규화한 q |
| Warm-up·정지·후보 채택 | Warm-up 0, 25후보·24갱신, 최종 후보 채택 유지 |
| Arm | 기존 A/B 두 norm 형태 유지; 새 replay arm 없음 |

여기서 plain은 **replay가 없는 ours**라는 뜻이다. G/E나 current actual 항을 제거한 native-only 목적을 뜻하지 않는다. Prefix 표현 보존 loss, 새 reference 문장, shared final norm budget, radial gate, proximal optimizer는 이번 revision에 추가하지 않는다.

## 1 상태와 변수

Batch entry 모델 W_t, eligible layer 집합 L과 그 크기 m, 실제 요청 수 B, native canonical anchor a_lr=||h0_lr||>0, native KL teacher, covariance C0와 history H_t를 고정한다. d_l은 해당 subject δ의 실제 차원이다. a·d·m은 batch 안에서 바뀌지 않으며 m을 현재 비영 δ의 수로 재정의하지 않는다.

모든 층·요청에 q_lr를 두고 0에서 시작한다. 실제 intervention은

\[
s_{lr}=\frac{a_{lr}}{\sqrt{d_l m}},\qquad
\delta_{lr}=s_{lr}q_{lr},\qquad D_l=[\delta_{l1},\dots,\delta_{lB}].
\]

현재 수치 profile은 q·s·D·gradient·Adam moment FP32, geometry FP64다. 첫-step 상한의 구현 검사는 FP32 반올림 허용오차를 포함하며 clamp나 loss에는 새로운 ε를 추가하지 않는다.

q는 optimizer 좌표일 뿐이다. **모든 과학적 loss, writer target, regularizer, actual 모델과 δ telemetry는 D로 정의한다.** Model weight, C0, H, a, s는 Adam parameter가 아니다.

## 2 Native subject 주경로와 목적

Frozen W_t의 각 eligible full-block 출력에서 native subject lookup token에 해당 요청의 δ_lr를 더한다. 같은 요청의 모든 native rewrite/KL context에 같은 δ를 사용한다. 하층 δ가 상층 표현과 native key에 미치는 영향은 동일 forward에 포함된다. 각 층이 따로 전체 edit을 성공시키는 독립 loss를 만들지 않는다.

\[
L_N(D)=\frac1B\sum_r\left[
L^v_{NLL,r}(D)+\lambda_K KL(p^v_{K,r}(D)\Vert p^0_{K,r})
+\lambda_n\sum_l\frac{\|\delta_{lr}\|_2}{a_{lr}^2}\right].
\]

현재 profile은 native 6 rewrite contexts와 1 KL 문장, λ_K=.0625, λ_n=.5, NLL layer31이다. 이 수치는 model/benchmark adapter 설정이며 B와 m을 코드 상수로 만들지 않는다. Target token과 context의 native reduction을 유지한다. Prefix는 기존 context 문장이며 새 prefix 보존 loss가 아니다. 공식 paraphrase/neighborhood, 미래 요청, 새로운 reference set은 학습에 사용하지 않는다.

## 3 MEMIT H와 같은 평균 key writer

각 후보의 actual builder는 낮은 층부터 높은 층으로 진행한다. 자기 write 이전의 projection input key k_lrc를 **현재 후보의 하층 actual weight write가 이미 적용된 모델**에서 얻는다. Builder에는 subject δ hook이 없다. 모든 요청과 원래 native context를 forward하며, 층마다 전체 logical B의 key를 수집한 뒤 solve한다.

Context group 수를 J, group g의 크기를 n_g라 하면 native key는

\[
\kappa_{lr}=\frac1J\sum_g\frac1{n_g}\sum_{c\in g}k_{lrc},\qquad
\kappa_l\in\mathbb R^{d_{in,l}\times B}.
\]

실제 집계는 native `compute_ks()`와 같이 **FP32 group mean → FP32 stack/mean → 요청별 stack → FP64 cast** 순서다. Canonical .5·나머지 .1의 FP64 가중합으로 대체하지 않는다. Native task의 context 평균과 writer의 그룹 평균은 서로 다른 reduction이다. Context를 삭제하거나 학습 문장을 줄이는 변경이 아니다.

\[
A_l=\lambda_C\operatorname{float64}(C_{0,l}^{32})+
\operatorname{float64}(H_{t,l}^{32}),\quad
P_l=(A_l+\kappa_l\kappa_l^\top)^{-1}\kappa_l,
\]
\[
U_l=D_lP_l^\top,\qquad
W_l^a=W_{t,l}^{32}+\operatorname{cast}_{32}(D_l^{64}P_l^{64\top}).
\]

C0는 저장된 FP32 mom2/count를 native 순서로 정규화한 값이다. Writer는

\[
\arg\min_U\;\frac12\|U\kappa_l-D_l\|_F^2+
\frac12\operatorname{tr}(UA_lU^\top)
\]

의 ridge 해다. 동일 residual에 대한 key aggregation·ridge map·prior/history 정의를 MEMIT-H와 맞춘다. **Residual의 출처만 ours의 local D로 유지**한다. L8 z의 남은 오차를 재분배하거나 D를 남은 층 수로 다시 나누지 않는다. Virtual absolute z에서 actual hidden을 빼는 target 교체도 이번에는 하지 않는다.

최저 write layer의 κ/P는 batch 내 재사용 가능하다. 상층 κ/P는 모든 후보에서 다시 계산하며 D_lower→actual κ→P→U의 gradient를 유지한다. Native subject forward와 actual writer forward는 구분한다. Current key가 같다고 virtual/actual hidden 또는 모든 개별 context에서의 δ 실현이 정확히 같다는 뜻은 아니다.

## 4 Native writer에 맞춘 배분 비용

\[
G_l=P_l^\top A_lP_l,\qquad
E_l=(P_l^\top\kappa_l-I_B)(P_l^\top\kappa_l-I_B)^\top,
\]
\[
G_l+E_l=I_B-\kappa_l^\top P_l.
\]

이는 symmetric SPD 수학적 prior의 항등식이다. 실제 FP32 history에 생기는 symmetry roundoff는 dense native reference와 수치 비교하며, 식을 맞추기 위해 A/H를 임의 변경하지 않는다.

\[
g_l=\frac{\sqrt{\operatorname{tr}(D_lG_lD_l^\top)}}{\sqrt B\sigma_l},\quad
e_l=\frac{\sqrt{\operatorname{tr}(D_lE_lD_l^\top)}}{\sqrt B\sigma_l},\quad
\sigma_l^2=B^{-1}\sum_r a_{lr}^2.
\]

Arm A는 R_W=Σ_l g_l, R_E=Σ_l e_l이고, Arm B는 각각 √Σ_l g_l², √Σ_l e_l²다. λ_W=λ_E=.1을 유지한다. 같은 계수에서도 norm 모양에 따라 유효 규제 규모가 다르다는 기존 한계를 기록한다. 이를 자동 재보정하거나 공식 평가점수로 계수를 맞추지 않는다.

G/E의 off-diagonal과 K/P 전체 gradient를 유지한다. Root와 native nonsquared norm의 0에서 선택 subgradient는0이다. 새 ε smoothing, layer quota, entropy 목표, top-k, layer 품질 gate는 없다. Mean-key E로 바뀌므로 v7의 full-context E·Ω·Z를 그대로 재사용하지 않는다.

## 5 Replay 없이 유지하는 current actual 항

같은 후보의 native post-injection hidden·target-position 분포를 detached teacher로 사용한다. Current actual branch는 모든 B의 D로 구성된 weight를 사용한다.

\[
C_h(I)=\operatorname{mean}_{r\in I}\frac12\sum_l\sum_cw_{rc}
\frac{\|h^a_{lrc}[s]-\operatorname{sg}(h^v_{lrc}[s])\|^2}{a_{lr}^2},
\]
\[
C_d(I)=\operatorname{mean}_{r\in I}\sum_cw_{rc}\operatorname{mean}_{target}
KL(\operatorname{sg}(p^v_{rc})\Vert p^a_{rc}),
\]
\[
C(I)=.1C_h(I)+.1C_d(I)+\lambda_K\operatorname{mean}_{r\in I}
KL(p^a_{K,r}\Vert p^0_{K,r}).
\]

현재 요청을 seeded disjoint four partitions I_j로 나누고 후보5/10/15/20의 update 전에 각각 한 번 적용한다. 빈 partition은 건너뛴다.

\[
L_k=L_N+\lambda_WR_W+\lambda_ER_E+
\mathbf1_{k\in\{5,10,15,20\}}\frac{|I_{k/5}|}{B}C(I_{k/5}).
\]

기존 schedule과 계수는 유지한다. Partition 보정을 이유로 임의 4배/24배를 곱하지 않는다. 전체 평균 loss에 B를 곱한 request-SUM gradient를 쓴다. Actual current NLL을 새 primary loss로 추가하지 않는다.

**Past replay term은0**이다. 과거 문장 저장, reservoir, admission KL teacher, past NLL baseline, reference sampling, reference forward, replay loss, memory RNG·eviction 모두 없다. 과거 요청을 current partition에 섞지 않는다. Current native teacher와 covariance/history는 남는다. Replay 제거를 λ_past=0인 채 과거 모델 forward를 계속 수행하는 것으로 구현하지 않는다.

## 6 정규화 좌표 Adam과 포화 문제의 교정 범위

q_0=0에서 Adam(q), η=.1 고정, betas(.9,.999), eps=1e-8, weight_decay=0을 사용한다. Warm-up은0이다. η는 **정규화된 공동 상대 이동 단위의 learning rate**다. 숫자가 기존 native LR과 같아도 raw δ의 Adam과 동일한 optimizer라는 뜻은 아니다.

첫 Adam 갱신에서는 epsilon을 포함해 모든 좌표에서 |Δq_i|≤η이므로

\[
\frac{\|\delta^{(1)}_{lr}\|}{a_{lr}}\le\frac{\eta}{\sqrt m},\qquad
\sum_l\frac{\|\delta^{(1)}_{lr}\|^2}{a_{lr}^2}\le\eta^2.
\]

m=5이면 층별 첫 상대 norm은 최대 .04472이고, 기존 .75 cap의 약5.96%다. 따라서 현재 Llama profile에서 첫 갱신의 전면 clipping을 수학적으로 피한다. 다른 profile에서는 η/√m<c_native라는 조건을 확인한다. ε의 단위도 q-space이므로 단순히 δ의 LR만 줄인 것과 정확히 동일하다고 표현하지 않는다.

각 update 뒤 **원래와 동일한 물리 clamp** ||δ_lr||≤c_native a_lr를 적용한다. q 좌표에서는 ||q_lr||≤c_native√(d_l m)다. 구현 정본은 materialized FP32 D에서 계산한 clamp scale을 q에 곱하는 것이며, q cap 식과의 등가는 실수 연산 기준이다. Adam moment는 유지한다. q→D mapping은 고정 가역 scale이므로, replay 제거·writer 변경을 제외하고 이 좌표 변경 자체는 동일 D에서의 목적과 최종 feasible set을 바꾸지 않는다.

**이는 첫 자동 포화를 교정하며, 최종 배분 성공을 보장하지 않는다.** 일정 방향 gradient가 지속되면 m=5,η=.1,c=.75에서 약17번째 갱신에 다시 cap에 닿을 수 있다. Outward momentum도 남는다. 한 층만 유효해도 1/√m scale이 적용되므로 같은 24갱신 안에서 집중해에 도달하는 속도가 느려질 수 있다. 고정 예산에서 edit 강도가 부족해질 가능성은 실제 pilot 이전에 배제할 수 없다.

모든 층을 상한에서 떨어뜨리는 결과 gate, 강제 shrink, 최소 norm 차이 조건은 넣지 않는다. 최적화 결과로 한 층 집중·여러 층 사용·모든 층 큰 δ가 나오는 것은 허용한다. 현 목적이 경계 해를 선호하는지와 수치적 첫 step이 자동으로 경계를 채우는 현상은 다른 문제다.

## 7 후보별 실행과 정확한 gradient 연결

후보1–24에서 다음 순서를 따른다.

1. q를 고정하고 `D = (s*q).detach().requires_grad_(True)`라는 물리 leaf를 만든다.
2. 같은 D에서 전체 native loss와 gradient를 microbatch 합산한다.
3. Actual causal mean-key writer를 구성하고 전체-B G/E를 계산한다. 해당 후보가 pulse이면 current actual 보조 gradient도 계산한다.
4. Native·policy·direct D·P solve·direct κ의 **모든 D adjoint를 완성**한다.
5. `q.grad = s * D.grad`를 정확히 한 번 전달하고 Adam(q)을 한 번 실행한다. q.grad에 D.grad를 그대로 더하거나 branch마다 중복 전달하지 않는다.
6. q에 물리 clamp와 동등한 scale을 적용한다. 아직 checkpoint backward가 남아 있는 동안 q/D/weight를 바꾸지 않는다.

고정 선형 map에 대한 정확한 chain rule이며 writer를 detach하는 근사가 아니다. 공유 nonleaf D를 여러 microbatch backward에서 재사용해 graph가 해제되는 문제를 피한다. Actual P adjoint를 builder graph로 되돌리는 기존 경로도 유지한다.

후보25는24갱신 뒤 terminal forward만 수행한다. 기술적 finite·solve·key·commit 검사는 유지하되 낮은 점수, 층 집중, clipping, 고정 예산 미수렴은 후보 거절 기준이 아니다. Official P/N으로 후보나 계수를 고르지 않는다.

## 8 Commit과 native history

최종 D의 같은 causal writer를 평가하고 materialized FP32 weight를 그대로 commit한다. Final rescale, 다른 key로 post-evaluation 재매핑, absolute-z 보상은 없다.

모든 candidate write가 활성화된 terminal actual 모델에서 각 층의 raw context key를 포착하고 native FP32 nested mean으로 κ_terminal을 계산한다. 별도 native `compute_ks`와의 parity를 pilot에서 확인한다. Qualified causal adapter에서는 terminal과 builder의 pre-own-write key가 같아야 한다.

\[
H^{32}_{t+1,l}=H^{32}_{t,l}+
\kappa^{32,CPU}_{l,terminal}(\kappa^{32,CPU}_{l,terminal})^\top.
\]

Native처럼 CPU FP32 Gram을 한 번 더한다. B/context 수로 나누지 않고, fact deduplication도 하지 않는다. V7의 full-context weighted Gram이나 lower-triangle mirror는 사용하지 않는다. W/H/RNG의 atomic rollback은 유지하며 replay memory transaction은 제거한다.

History 의미가 달라졌으므로 **v7의 W/H를 v8의 이어달리기 초기값으로 재사용하지 않는다.** 후속 실험은 원래 W0/H0에서 시작해야 한다. Replay가 없어도 H는 이전 edit 방향에 대한 writer 보호 역할을 하므로 이름 그대로 MEMIT-H 계열의 순차 통계다.

## 9 계산과 검증의 범위

A가 symmetric SPD일 때 native dense solve와 동등한 B차원 dual을 사용할 수 있다. 전체 context C차원 solve에서 B차원으로 줄어 현재 B100·6contexts에서는600→100이 된다. 모든 context forward는 남으므로 전체 실행이6배 또는216배 빨라진다고 주장하지 않는다. A factor, first-layer geometry, prefix cache, selected-position head 등 기존의 의미 보존 최적화는 재사용할 수 있다.

A/H를 임의 symmetrize하거나 jitter를 더하지 않는다. SPD fast path가 native dense reference와 맞지 않으면 같은 A의 native dense solve로 기술적 fallback하거나 해당 adapter를 미지원으로 보고한다. Generic autograd를 우선 reference로 하여 G/E·κ/P 경로 누락을 방지한다.

실제 구현 함수와 검증 항목은 [구현 계약](implementation-ko.md), 고정 필드는 [contract.json](contract.json), 수학 검사 결과는 [math](math/README.md), 수식 정본은 [TeX](../../../docs/methods/jlz-native-writer-v8.tex)에 기록한다. CPU toy 통과는 실제 모델의 locality 개선, PS 유지, GPU parity, wall-time 검증을 대신하지 않는다. V8은 replay·writer·optimizer를 함께 변경하므로, 이후 v7과의 성능 차이만으로 어느 한 변경의 단독 효과를 주장하지 않는다.
