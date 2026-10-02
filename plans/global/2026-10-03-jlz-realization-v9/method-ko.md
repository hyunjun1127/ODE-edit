# JLZ v9 공동 local z와 실제 write 배분

2026-10-03. **Local target은 writer가 자기 층에 쓸 목표를 정의하지만, native ridge가 그 목표를 모두 실현한다는 보장은 없다.** 이번 방법은 MEMIT-H ridge를 유지하면서 그 비용과 실현량을 명확하게 반영한다. Exact writer는 동일 목표를 사용하는 pilot 대조이며 본 방법이 아니다. 모든 eligible layer를 처음부터 공동 최적화하고, 결과에 따른 한 층 집중을 허용한다.

사용자 확정 선택은 “Ridge 유지 + exact pilot 비교”다. [v8](../2026-10-03-jlz-native-writer-v8/method-ko.md)의 native 입력·subject 주입·손실·정규화 Adam·current 보조항·replay 제거·native history를 유지한다. 변경은 **배분 비용의 통합, 계획과 실현의 명시적 구분, 비교 실험 계약**이다. 성능을 보기 전에 이 명세를 고정한다.

## 1 보존하는 backbone

Batch entry 모델 W_t, 고정 eligible 집합 L, m=|L|, 실제 batch 크기 B를 사용한다. 각 층의 entry canonical subject 표현 norm a_lr>0, KL teacher, C0, H_t는 batch 동안 고정한다. 모든 요청의 native rewrite/KL 문장과 target·lookup·readout·reduction을 baseline과 동일하게 둔다. 새 reference나 학습용 paraphrase/neighborhood를 추가하지 않는다.

Frozen W_t의 각 eligible full-block 출력에서 subject 위치에 δ_lr를 더한다. 요청 r의 모든 native context에는 동일한 δ_lr를 사용한다. 모든 층을 함께 주입한 **한 모델 경로**의 손실을 계산하며, 층마다 전체 편집을 독립적으로 성공시키는 loss를 만들지 않는다.

\[
L_N(D)=\frac1B\sum_r\left[L^v_{NLL,r}+\lambda_K KL(p^v_{K,r}\Vert p^0_{K,r})+\lambda_n\sum_l\frac{\|\delta_{lr}\|_2}{a_{lr}^2}\right],\qquad D_l=[\delta_{l1},\ldots,\delta_{lB}].
\]

현재 profile은 λ_K=.0625, λ_n=.5, rewrite6개/KL1개, NLL layer31이다. 이는 adapter 설정이지 method 상수가 아니다. Native norm을 diagonal capacity-weighted norm으로 바꾸지 않는다. 같은 .5를 사용하더라도 그런 교체는 다른 regularizer다.

## 2 Current key를 쓰는 native ridge

각 후보에서 아래층부터 actual weight write를 구성한다. 자기 write 직전의 key는 **그 후보의 하위층 actual write가 적용된 모델**에서 구한다. Builder에는 virtual δ hook이 없다. 전체 logical B의 key를 모은 뒤 native 그룹 평균을 수행한다.

\[
\kappa_{lr}=\frac1J\sum_g\frac1{n_g}\sum_{c\in g}k_{lrc},\qquad
A_l=\lambda_C C_{0,l}+H_{t,l},\qquad
P_l=(A_l+\kappa_l\kappa_l^\top)^{-1}\kappa_l,
\]
\[
U_l=D_lP_l^\top,\qquad W_l^a=W_{t,l}^{32}+\operatorname{cast}_{32}(U_l^{64}).
\]

Native 순서인 FP32 group mean → FP32 group stack/mean → request stack → FP64 cast를 유지한다. C0의 native FP32 normalization과 H의 CPU FP32 저장·누적도 유지한다. 같은 residual에 대한 writer map을 MEMIT-H와 맞춘다. Residual은 공동 학습한 local D이며, L8 target 재분배나 remaining-layer division을 넣지 않는다.

최저 write layer의 geometry만 batch 내 재사용한다. 상층 κ/P는 매 후보 갱신하며 **D_lower→actual κ→P→U**의 전체 gradient와 요청 간 off-diagonal을 유지한다. 같은 후보의 모든 write를 활성화한 terminal key와 builder key의 정합을 검사한다. 검증된 adapter에서 downstream layer가 upstream key를 바꾸지 않는 인과성을 사용한다.

## 3 계획한 배분과 실현된 배분

수학적으로 A≻0인 경우 X=κᵀA⁻¹κ를 정의하면

\[
M_l=P_l^\top\kappa_l=X_l(I+X_l)^{-1},\qquad
Y_l=U_l\kappa_l=D_lM_l.
\]

D는 **virtual에서 계획한 local 증분**, Y는 **actual builder key에 대한 FP64 writer의 직접 증분**이다. Materialization과 실제 forward의 반올림을 포함한 실현량은 effective weight/동일 입력의 output difference로 별도 기록한다. 이 Y는 entry 모델 대비 총 hidden 변화나 최종 출력 변화를 뜻하지 않는다. M_rr은 자기 요청 계수이고, Y_r=δ_r M_rr+Σ_{s≠r}δ_s M_sr에는 다른 요청의 목표가 섞인다.

계획 배분과 실현 배분은 각각 norm 기반 layer share로 기록한다. 전체 분모가0이면 null로 기록한다. Norm share는 방향이나 인과적 기여도를 설명하지 않으므로 아래 투영·잔차·교차항을 함께 보고한다. 실제 사용 층을 선언하기 위한 임계값으로 layer를 제거하지 않는다.

G/E의 scalar 비율만으로 M_rr 또는 실제 Y_r/δ_r를 추정하지 않는다. \(g^2/e^2\)는 일반적으로 D 방향에 따라 가중된 X의 스펙트럼 평균이다. B=1 또는 단일 고유방향의 scalar 특수 경우를 BS100에 적용할 수 없다.

## 4 하나의 ridge 배분 비용

\[
G_l=P_l^\top A_lP_l,\qquad E_l=(M_l-I)(M_l-I)^\top,
\]
\[
\mathcal V_l(D)=\min_U\left\{\frac12\|U\kappa_l-D_l\|_F^2+\frac12\operatorname{tr}(U A_l U^\top)\right\}
=\frac12\operatorname{tr}\big[D_l(G_l+E_l)D_l^\top\big].
\]

G는 writer의 보존 에너지이고 E는 평균 key의 미실현 오차다. 서로 다른 항이므로 v8의 별도 root 합을 “중복 계산”이라고 부르지는 않는다. V9은 그 합성 목적에 맞춰 **root를 한 번만** 적용한다.

\[
c_l=\frac{\sqrt{\operatorname{tr}[D_l(G_l+E_l)D_l^\top]}}{\sqrt B\,\sigma_l},\qquad
\sigma_l^2=\frac1B\sum_r a_{lr}^2.
\]

\[
R_A=\sum_l c_l,\qquad R_B=\sqrt{\sum_l c_l^2},\qquad\lambda_{alloc}=.1.
\]

SPD의 실수 연산에서는 G+E=I−M=(I+X)⁻¹이다. 수치 평가는 G/E의 nonnegative norm 합 또는 SPD solve를 사용하고 I−M의 명시적 뺄셈으로 cancellation을 만들지 않는다. Fixed D에서 X의 용량이 Loewner 순서로 커질수록 이 비용은 감소한다. 이는 다른 층의 κ가 서로 순서화된다거나 total gradient가 δ 크기를 항상 줄인다는 뜻은 아니다. 하위층 변경이 상위층 geometry를 바꾸는 효과까지 미분한다.

A는 층별 합, B는 층별 제곱합의 root다. A는 상대적으로 집중을, B는 상대적으로 분산을 허용하는 norm 형태지만, B도 균등 분배를 강제하지 않는다. 둘 다 모든 층·한 층 집중을 허용한다. Adam이 정확한0을 보장하지 않으므로 sparsity 성공을 주장하지 않는다.

계수 효과를 숨기지 않는다. 각 arm에서 같은 D/geometry의 비용은 **v9≤v8≤√2 v9**다. A와 B 사이에도 R_B≤R_A≤√m R_B다. 따라서 동일 .1이 동일 유효 규제를 뜻하지 않는다. 이번 .1은 기존 두 계수와의 연속성을 위한 고정 선택이며, 최적 계수나 공정한 penalty-strength matching이 증명된 값은 아니다. Pilot 결과로 사후 보정하지 않는다.

σ는 batch RMS anchor를 유지한다. Native norm은 요청별 a²로 정규화하므로 요청 가중 의미가 다르다. 동일 상대 δ에서 단순 identity metric의 배분 비용에는 큰 a 요청이 더 기여하고, native norm에는 작은 a 요청이 더 기여한다. 이 차이는 telemetry에 남기며 이번 revision에서 normalization까지 동시에 바꾸지 않는다.

## 5 유지하는 current 보조항과 optimizer

V8의 current-only physical pulses를 유지한다. Seeded disjoint I_j를 후보5/10/15/20의 update 전에 한 번씩 사용한다. 전체 B의 D로 actual 모델을 만들며 보조 loss의 관측 요청만 partition한다.

\[
C(I)=.1 C_h(I)+.1 C_d(I)+\lambda_K\operatorname{mean}_{r\in I}KL(p^a_{K,r}\Vert p^0_{K,r}),
\]
\[
L_k=L_N+.1R_{A\text{ or }B}+\mathbf1_{k\in\{5,10,15,20\}}\frac{|I_{k/5}|}{B}C(I_{k/5}).
\]

C_h는 virtual post-injection subject hidden을 detached teacher로 하는 actual squared alignment이며 요청별 a², context의 native 가중치로 정규화한다. C_d는 native target 위치에서 detached virtual 분포에서 actual 분포로의 KL이다. 이와 별도로 actual essence KL은 current‖entry 방향이다. Virtual/actual essence KL은 서로 다른 모델 경로의 보존 항이며 같은 값을 두 번 더하는 계산은 아니다. 필요성 검증은 이번 pilot의 비교 축에 추가하지 않는다. Exact writer 비교에도 pulse fit을 다시 수행하지 않는다.

History replay, 과거 문장·teacher·loss·sampling·forward는 없다. Native key history H는 유지한다. 새 prefix 표현 보존 loss나 reference text는 추가하지 않는다.

\[
\delta_{lr}=\frac{a_{lr}}{\sqrt{d_lm}}q_{lr},\qquad q_0=0.
\]

V8의 q Adam을 유지한다: η=.1, betas(.9,.999), eps1e-8, weight_decay0, warm-up0, 후보25회·갱신24회, terminal 후보 채택. 전체 mean loss에 B를 곱한 request-SUM gradient를 누적하고 `q.grad=s*D.grad`를 한 번 전달한다. 물리 D의 native clamp를 적용하고 moment는 유지한다. 모델·q·D·moments FP32, geometry FP64다.

첫 갱신은 ||δ_lr||/a_lr≤η/√m, Σ_l||δ_lr||²/a_lr²≤η²다. 현재 m5에서는 층별 최대.04472 대 cap.75다. 이후 포화나 24회 내 수렴을 보장하지 않는다. 강도가 부족하다고 exact probe에만 더 큰 δ나 추가 fit budget을 주지 않는다. Native Adam과 같은 trajectory라는 주장도 하지 않는다.

## 6 남는 오차의 분해

같은 후보의 virtual post-injection 상태를 z^v_lrc, actual lower writes가 반영되고 자기 write만 아직 더하지 않은 같은 층 출력 상태를 h^{a,-l}_lrc라 두고 b_lrc=z^v_lrc−h^{a,-l}_lrc라 한다. Qualified linear-output adapter에서 actual own write는 U_l k_lrc다.

\[
z^v_{lrc}-h^{a,+l}_{lrc}
=\underbrace{b_{lrc}-\delta_{lr}}_{\text{상속된 경로 차이}}
+\underbrace{\delta_{lr}-(U_l\kappa_l)_r}_{\text{평균 key 실현 오차}}
+\underbrace{U_l(\kappa_{lr}-k_{lrc})}_{\text{context key 차이}}.
\]

세 벡터는 합치되 norm을 단순 합해서 원인 기여율로 해석하지 않는다. 서로 상쇄할 수 있다. Exact mean-key writer가0으로 만드는 것은 가운데 항뿐이다. 모든 context, subject 외 token, attention 상태와 최종 target 분포의 일치를 보장하지 않는다.

V9은 이 차이를 기록하지만 **absolute tracking, 부족분의 상층 자동 재배분, M 역행렬로 D 증폭**을 하지 않는다. 그런 변경은 계획된 층별 δ와 실제 추가 책임의 관계를 다시 정의해야 한다. 현재 pulse가 있는 구조를 “피드백이 전혀 없는 writer”라고 부르지도 않는다. 명시적인 absolute-state tracking이 없다는 것이 정확한 한계다.

## 7 Exact writer는 pilot 대조

A≻0, κ full-column-rank일 때

\[
U^{eq}=D X^{-1}\kappa^\top A^{-1}
=\arg\min_{U\kappa=D}\frac12\operatorname{tr}(UAU^\top),\qquad
U^{eq}\kappa=D,
\]
\[
\operatorname{tr}(U^{eq}A U^{eq\top})=\operatorname{tr}(DX^{-1}D^\top).
\]

등식 제약 batch writer는 [EMMET](https://arxiv.org/html/2403.14236v5#S5)의 선행이다. Native MEMIT-H ridge의 같은 해가 아니며, H0에서 A 전체의 λ_C scale은 exact 해에서 상쇄된다. 작은 고유값 방향의 큰 write와 locality 손실을 막는 보장은 없다. EMMET 논문의 안정화처럼 X에 εI를 추가하면 일반적으로 exact equality가 깨진다.

Rank가 부족하면 D(I−X†X)=0일 때에만 exact 해가 존재한다. 같은 subject/prefix로 동일 key가 생기고 서로 다른 D가 필요하면 불가능하다. 이번 pilot은 **full-rank 수치 qualification을 통과한 exact 비교만 실행**한다. Rank-deficient는 compatibility 진단을 남기고 exact endpoint는 unsupported로 기록한다. 조용한 pseudoinverse, jitter, ridge fallback, 요청 제거, 층 제거를 하지 않는다. Exact의 수학적·수치적 미지원은 정상 native ridge 본선의 실행을 막지 않지만, rollback이나 main state integrity 실패는 본선 기술 실패다.

비교는 두 단계다. 먼저 동일 D·동일 actual κ·동일 A에서 operator만 바꿔 Y와 에너지를 비교한다. 이후 동일 D를 고정하고 각 writer의 하위층 실제 write에 맞춰 상층 κ를 각각 새로 계산한 전체 actual 모델을 비교한다. 두 단계는 서로 다른 원인을 측정한다. V9의 학습된 D는 ridge 목적에 맞춰졌으므로 이 결과는 **고정 D에서의 writer 교체 효과**이며 independently optimized exact 방법의 성능이 아니다.

## 8 Commit과 적용 범위

Main은 terminal ridge model을 그대로 commit하고 native terminal mean key의 CPU FP32 Gram을 H에 한 번 더한다. W/H/RNG transaction을 보존한다. Shadow exact에는 main history admission이나 optimizer update가 없으며, W/H/RNG를 되돌린 뒤 ridge main 상태를 hash로 확인한다. 새 revision은 cold W0/H0에서 시작하고 v7/v8의 수정 W/H를 이어 쓰지 않는다.

B·문맥 수·target 길이·layer 수·차원·부분 batch는 runtime 입력에서 얻는다. 모든 configured eligible layer가 변수와 writer에 존재해야 한다. Adapter는 full-block 주입 위치와 linear write의 국소적 가법성, subject 위치, native KL/readout, key 집계 및 current-key 인과성을 검증해야 한다. 새로운 model 지원을 검증 없이 선언하지 않는다. Ridge는 singular κ에도 사용 가능하지만 exact probe는 별도의 적용 조건을 갖는다.

실행 범위는 [실험 설계](experiment-500/experiment-ko.md), 구현 및 진단은 [구현 계약](implementation-ko.md)에 정의한다. 이 방법은 “native objective를 따르는 공동 local-z와 writer 비용에 따른 동적 배분”을 시험한다. 평균 key 실현의 완전성, locality 개선, exact writer의 우월성 또는 최적 배분을 결과 전에 주장하지 않는다.
