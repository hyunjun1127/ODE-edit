# JLZ v6 subject δ 공동 학습과 writer 배분

2026-10-02. 이 문서는 구현할 method를 정의한다. 실제 모델 pilot이나 성능 개선은 아직 검증하지 않았다. 기존 v4/v5 소스와 실험 결과는 변경하지 않는다. [구현 계약](implementation-ko.md), [기계 판독 계약](contract.json), [TeX](../../../docs/methods/jlz-subject-adam-v6.tex)를 함께 사용한다.

**Native subject 위치의 δ 주입과 Adam을 유지하고, 모든 편집층 δ의 방향·크기를 공동 학습한다. 배분에는 실제 writer의 비용과 실현 가능성을 반영한다.** 주 task NLL을 모든 토큰에 적용된 수정 가중치의 NLL로 교체하지 않는다. 실제 writer는 저비용 geometry 항과 정해진 소수 시점의 보조 학습으로 δ에 피드백한다.

## 1 목표와 관측 근거

목표는 순차 편집 중 같은 사실을 어느 층에 어떤 방향·크기로 쓰는 것이 효율적이고 보존에 유리한지 학습하는 것이다. Eligible layer 전체를 처음부터 고려한다. 한 층 집중을 허용하며, 강제 균등화·최소 사용 층 수·사전 top-k는 없다.

v4 A는 subject 공동 δ와 Adam으로 W5 P98.7%, N62.24%였다. 보관된 2,500개 층·요청 group 중 97.56%가 native clamp 부근이었다. v5 A는 P73.2%, N87.46%였으며 P 실패 268개 중 266개가 해당 편집 직후부터 실패했다. v5는 보수적인 공통 step solver와 실제 모든 토큰 weight task graph를 사용했다. 이 비교는 graph·writer·solver 등 여러 변경이 섞여 있으므로 Adam 단독의 효과를 증명하지 않는다.

따라서 v6는 subject 개입을 유지하고, 추가 연구 대상을 **δ의 공동 배분과 실제 write의 연결**로 좁힌다. [원자료 리뷰](../../../experiment-reports/global/2026-10-02-jlz-v5-a-w5-review/report-ko.md)와 [Adam 점검](../../../experiment-reports/global/2026-10-02-jlz-v5-a-w5-review/adam-subject-injection-review-ko.md)을 근거로 삼는다.

## 2 전체 파이프라인

1. 실제 batch entry W_t와 history H_t에서 native 입력·anchor·teacher·문맥별 key를 수집한다.
2. 모든 eligible layer의 writer 기저 P와 작은 geometry 행렬을 한 번 계산한다.
3. 모델 W_t는 고정하고 모든 층의 subject δ를 0에서 시작한다.
4. 매 Adam update에서 전체 current batch의 native subject loss와 저비용 배분 비용을 함께 계산한다.
5. 후보 5·10·15·20에서는 해당 native forward를 teacher로 삼아 실제 writer의 정합·보존 보조 gradient를 추가한다.
6. 후보 25에서 최종 native/actual 모델을 관측하고, actual 후보의 가중치를 그대로 commit한다.
7. 실제 후보의 current key로 history를 한 번 갱신하고, 과거 native memory를 갱신한다.

이는 유한한 일정의 Adam 학습 알고리즘이다. 모든 step에서 하나의 고정 목적함수를 정확히 최소화한다거나 loss가 단조 감소한다고 주장하지 않는다. 추가 실제 writer 계산의 시점·표본·teacher 갱신을 아래에 고정한다.

## 3 Native subject 공동 변수

이번 logical batch의 실제 요청 수를 B, ordered eligible layer 집합을 L이라 한다. 요청별 native context 수와 target 길이는 가변이다. 층마다 출력 차원 d_out,l도 달라도 된다.

\[
D_l=[\delta_{l1},\ldots,\delta_{lB}],\qquad
a_{lr}=\|h^0_{lr}\|_2>0.
\]

h0는 canonical native 문장의 **clean entry block output**이다. 매 step 변경되는 hidden이나 down_proj 출력 norm으로 바꾸지 않는다. 같은 요청의 모든 native rewrite·KL 문장에 같은 δ_lr를 그 문장의 native subject lookup 위치에 더한다.

\[
h^v_{lrc}[s_{rc}]=h^{v,pre}_{lrc}[s_{rc}]+\delta_{lr},\qquad
\|\delta_{lr}\|_2\le c_{native}a_{lr}.
\]

상층의 h_v,pre에는 하층 δ의 전달이 이미 포함된다. 따라서 joint local target은 z_v=h_v,pre+δ이며 일반적으로 clean h0+δ만으로 표현되지 않는다. Subject 이외 위치에 직접 δ를 더하지 않지만 attention을 통한 자연스러운 downstream 변화는 그대로 미분한다.

Native 주목적은

\[
L_{native}^{subj}(D)=\frac1B\sum_r\left[
L_{NLL,r}^{v}+\lambda_{KL}KL(p_r^v\Vert p_r^{entry})
+\lambda_{norm}\sum_l\frac{\|\delta_{lr}\|_2}{a_{lr}^2}\right].
\]

NLL의 context·token 평균, target IDs, readout layer, KL subject lookup·최종 readout을 baseline profile과 같게 유지한다. KL은 full vocabulary current‖entry다. 배치가 작거나 마지막 partial batch여도 같은 정의를 사용한다.

## 4 실제 writer를 batch entry에서 고정한다

층별 K0는 전체 current 요청의 **native rewrite 문맥별 subject key**를 열로 갖는다. Context 개수를 C라 하고, Z∈R^(B×C)는 context 소유 요청의 one-hot 행렬, Omega=diag(α_c)라 한다. 요청별 α합은1이며 Z Omega Zᵀ=I다. Native task의 context 가중치 w와 writer α를 혼동하지 않는다. 현재 profile의 w는 6개 각각1/6, α는 canonical .5 및 나머지 각 .1이다.

\[
A_l=\lambda_C C_{0l}+H_{tl},\quad
\bar K_l=K_l^0\Omega Z^\top,\quad
M_l=A_l+K_l^0\Omega(K_l^0)^\top,\quad
P_l=M_l^{-1}\bar K_l.
\]

역행렬을 만들지 않고 FP64 SPD solve를 사용한다. 실제 공유 write는

\[
U_l(D)=D_lP_l^\top,\qquad
\widetilde W_l(D)=W_{tl}+\operatorname{cast}_{32}(D_l^{64}P_l^{64\top}).
\]

P는 해당 batch의 fit·보조 계산·최종 commit에서 불변이다. 다른 요청 열의 D도 같은 U에 들어간다. 실제 weight 모델에서는 U가 subject를 포함한 **모든 토큰**에 적용된다. Native 주경로에는 이 weight 모델을 설치하지 않는다.

이 선택은 반복 solve 비용을 줄이고 평가한 actual 후보와 commit을 일치시키기 위한 정의다. Entry geometry의 stale key 문제를 없애거나 δ의 완전 실현을 보장하지 않는다. Absolute z를 맞추기 위해 final residual을 새로 계산하거나, 마지막에 current key로 P를 재계산하지 않는다. 그러한 writer는 다른 method다.

## 5 저비용 물리 부담과 실현 비용

Entry에서 다음 B×B PSD 행렬을 구한다.

\[
G_l=P_l^\top A_lP_l,\quad
E_l=(P_l^\top K_l^0-Z)\Omega(P_l^\top K_l^0-Z)^\top.
\]

이는 각각

\[
\operatorname{tr}(D_lG_lD_l^\top)=\operatorname{tr}(U_lA_lU_l^\top),
\]
\[
\operatorname{tr}(D_lE_lD_l^\top)
=\sum_c\alpha_c\|U_lk^0_{lc}-\delta_{l,r(c)}\|_2^2
\]

다. G는 C0/history 방향에 가하는 물리 write 부담이고, E는 문맥별 δ 실현 실패다. 특히 write가0인데 virtual δ만 task를 개선하는 퇴화를 G만으로는 잡지 못하므로 E를 함께 사용한다.

\[
G_l+E_l=I-\bar K_l^\top P_l
\]

를 CPU 수학 검증에 포함한다. Off-diagonal 성분을 버리지 않는다. 요청 간 공유 write의 간섭이 여기에 들어간다.

Entry 고정 scale은 σ_l²=B^(-1)Σ_r a_lr²로 정한다. Candidate D norm으로 비용을 나누지 않는다.

\[
g_l=\frac{\sqrt{\operatorname{tr}(D_lG_lD_l^\top)}}{\sqrt B\,\sigma_l},\qquad
e_l=\frac{\sqrt{\operatorname{tr}(D_lE_lD_l^\top)}}{\sqrt B\,\sigma_l}.
\]

이 비용은 norm이므로 0에서 subgradient 0을 사용한다. Small PSD factor를 사용해 vector norm으로 계산하고, sqrt에 임의 ε를 넣어 원점을 바꾸지 않는다. PSD 수치오차 처리는 qualification에서 고정한다. 전체 D로 한 번 계산하므로 microbatch별 root의 합으로 바뀌지 않는다.

RMS 정규화는 요청 수에 따른 단위를 명시한다. B를 바꾸면 P·history 경계·모델 상태·σ도 달라지므로 logical batching 불변성을 주장하지 않는다. Microbatch만 바꿀 때는 목적이 같아야 한다.

## 6 두 arm은 배분 norm만 다르다

과거 사용자가 요청한 부담 분산 비교를 이 항에서만 분리한다. **두 arm 모두 native norm과 clamp가 같고, 실제 writer 보조 학습과 과거 보존을 동일하게 사용한다.** 이전 v5의 η=0/1 arm과 의미가 다르므로 이름을 이어 붙여 집계하지 않는다.

| 항목 | v6 A 자유 배분 | v6 B 분산 선호 |
|---|---|---|
| 물리 부담 R_W | Σ_l g_l | sqrt(Σ_l g_l²) |
| 실현 비용 R_E | Σ_l e_l | sqrt(Σ_l e_l²) |
| 나머지 method | 동일 | 동일 |

매 step의 기본 loss는

\[
L_{base}=L_{native}^{subj}+\lambda_W R_W+\lambda_E R_E.
\]

동일한 효과·G/E·σ를 갖는 L개 경로에 같은 방향의 총 변위를 1/L씩 나누면, A의 정책 비용은 같고 B의 비용은 1/sqrt(L)로 줄어든다. 따라서 A에는 이 특정 분할에 대한 자동 할인은 없고 B에는 분산 선호가 있다. 실제 Transformer에서는 경로·geometry가 다르므로 A의 전체 목적이 모든 배분에 중립이라는 주장은 아니다. 두 arm 모두 한 층 집중이 가능하며 B도 강제로 층을 사용시키지는 않는다.

공통 subject 정합 항 C_h도 제곱 오차를 사용하므로 조건에 따라 분산을 선호할 수 있다. A/B 비교는 **geometry norm의 추가적인 분할 할인 차이**를 분리한다. A에서 모든 종류의 분산 선호가 제거됐다는 의미는 아니다.

현재 시작 profile은 λ_W=λ_E=.1이다. 검증 전 시작값이며 optimal 값이나 성능 보장을 뜻하지 않는다. A는 이미 writer 기저에도 쓰이므로 외부 정책 비용에 다시 등장하는 것은 의도적인 추가 가격 부과다. 지나치게 큰 계수는 약한 편집을 선호할 수 있다. 계수와 norm shape를 실행 중 공식 P/N 결과에 따라 바꾸지 않는다.

## 7 실제 writer 보조 학습

Cheap G/E는 entry에서의 근사다. 실제 하층 write가 상층 key를 바꾸고 다른 토큰과 요청에 미치는 효과는 작은 actual branch에서 직접 관측한다. 이 branch의 목적은 native subject 해를 실제 write가 재현하고, 과거 native 입력을 보존하게 하는 것이다.

### 7.1 고정된 일정과 표본

Native 후보를1부터25까지 평가하고1–24에서 Adam update한다. Physical 보조 학습은 **후보5·10·15·20의 update 이전**에만 실행한다.

Current 요청의 seeded permutation을 네 균형 partition I_j로 나눈다. 서로 겹치지 않고 합집합은 전체 B다. B<4이면 빈 partition을 허용한다. 과거 native memory에서 이번 batch에 고정한 S_t도 네 partition J_j로 나눈다. |S_t|=min(16,B,유효 과거 fact 수)이며 현재 재편집 fact는 제외한다. 표본·시점은 모델의 loss·gradient·공식 평가와 독립이다. 선택된 fact에서는 native 문맥과 target 위치를 전부 사용한다.

이로써 보조 학습용 actual branch는 네 시점을 합쳐 current B개와 past |S_t|개 native bundle을 한 번씩 처리한다. Primary native task는 매번 전체 B를 처리한다. 학습 문장을 일부 제거하는 절감이 아니다.

### 7.2 Native teacher와 subject 정합

후보 k의 native joint forward에서 선택된 current 요청의 post-injection subject block output과 native NLL readout의 target 위치 분포를 detach해 teacher로 보관한다. Teacher와 actual branch는 **같은 update 전 D^(k)**를 사용한다. Teacher를 별도 모델 forward로 재생성하지 않는다.

Current subset I의 평균 정합 비용은

\[
C_h=\frac1{|I|}\sum_{r\in I}\frac12\sum_l\sum_{c\in RW_r}w_{rc}
\frac{\|h^a_{lrc}[s]-\operatorname{sg}(h^v_{lrc}[s])\|_2^2}{a_{lr}^2}.
\]

두 hidden 모두 block output이다. Down-projection 출력과 섞지 않는다. Subject 위치에서 각 층의 native 공동 목표가 실제로 실현되는지를 본다.

Native rewrite의 target 예측 위치에서는

\[
C_d=\frac1{|I|}\sum_{r\in I}\sum_c w_{rc}\frac1{T_r}\sum_j
KL\big(\operatorname{sg}(p^v_{rcj})\Vert p^a_{rcj}\big)
\]

를 쓴다. Full vocabulary teacher‖actual distillation이며, **native의 current‖entry KL과 다른 역할·방향**이다. Actual current-target NLL을 별도 task로 추가하지 않는다. Primary task는 계속 subject 주입 경로다.

Current native KL 입력에 대해서는 actual current‖이번 batch entry KL을 native λ_KL로 추가한다. Teacher 분포를 rewrite distillation teacher와 혼용하지 않는다.

\[
C_{current}=\beta_h C_h+\beta_d C_d+\lambda_{KL} C_{KL,current}^{a},
\qquad \beta_h=\beta_d=.1.
\]

### 7.3 과거 native 보존

외부 reference, 공식 paraphrase/neighborhood, 미래 edit는 쓰지 않는다. 이미 편집에 사용한 native KL/rewrite만 재사용한다. Memory capacity는 기본128 unique fact다.

- KL teacher는 fact admission 당시 실제 편집 전 모델의 native lookup 분포다. 동일 입력 identity는 기존 teacher를 공유하고 이후 batch entry로 덮어쓰지 않는다.
- Rewrite 기준 b_rc는 해당 fact의 최신 **실제 commit 직후** context별 target NLL이다.
- 동일 fact 재편집은 해당 batch replay에서 제외하고 commit 후 최신 target/b로 교체한다.
- Reservoir와 version/eviction 처리는 v5 계약을 재사용한다. 평가 성공 여부로 admission을 결정하지 않는다.

\[
R_{past}(J)=\lambda_{KL}\,\operatorname{mean}_{r\in J}KL(p_r^a\Vert\pi_r^{admission})
+\lambda_{past,E}\operatorname{mean}_{r\in J}\sum_cw_{rc}
\frac12[\ell_{rc}^a-b_{rc}]_+^2,\qquad\lambda_{past,E}=1.
\]

이 비용은 실제 writer에 대한 soft 비용이다. 과거 성공을 보장하는 hard constraint가 아니며 이전에 얻지 못한 편집을 복구하는 항도 아니다. 두 arm에 동일하게 적용한다.

### 7.4 Probe step의 정확한 update rule

Probe j에서는

\[
L_k=L_{base}+\frac{|I_j|}{B}C_{current}(I_j)
+\frac{|J_j|}{|S_t|}R_{past}(J_j)
\]

를 사용한다. 빈 집합의 항은0이고 0으로 나누지 않는다. 그 외 step은 L_k=L_base다. 네 current 가중치와 유효 past 가중치는 각각 합이1이다. 네 pulse를 full auxiliary epoch 한 번으로 가격 매긴 선택이며, 매 step full auxiliary objective를 평가한 것과 동등하지 않다. 임의의4배·6배 보정을 하지 않는다.

Teacher는 stop-gradient이고 probe마다 갱신된다. 따라서 고정된 단일 목적의 정확한 gradient나 수렴 보장을 주장하지 않는다. 이전 probe의 보조 gradient를 다음 step에 재사용하지 않는다. Native graph와 actual graph는 같은 절대 D에 gradient를 합산한다. Actual branch는 subset을 관측해도 **전체 B로 만든 공유 U**를 사용한다.

## 8 Adam과 고정 학습률

모든 D를 절대 δ 좌표의 Adam에 등록한다. betas=(.9,.999), eps=1e-8, weight_decay=0이다. Native norm은 explicit loss다. 실제 구현은 수식의 요청 평균 loss 전체에 B를 곱해 request SUM으로 backward한다. Norm·policy·current pulse·past pulse 모두 함께 곱한다.

Adam update 번호 n=1..24에 대해

\[
\mathrm{lr}_n=\mathrm{lr}_{native},\qquad n=1,\ldots,24.
\]

현재 profile은 첫 update부터 끝까지 lr_native=.1이다. 2026-10-02 사용자 지시에 따라 첫 시도에서 learning-rate warmup을 제외한다. 두 arm에 동일하며, 층별 gradient를 unit norm으로 만들거나 학습률을 층마다 재분배하지 않는다.

Clamp는 loss 항이 아닌 Adam update 뒤의 native 사후 투영으로 유지한다. Clamp에 맞춘 학습률 보정·별도 gate는 도입하지 않는다. 매 step 전체 logical batch gradient를 누적한 뒤 Adam step을 한 번 수행하고 각 (l,r)을 native ball로 clamp한다. Momentum을 임의로 reset하지 않는다. Clamp 비율은 관측값이며 배분 허용 조건이나 후보 선택 기준이 아니다.

Native `.05` early stop을 batch 평균에 그대로 적용하지 않는다. 이번 구현은 고정25후보/24 update다. 성능·층 몫·realization ratio에 의한 추가 gate나 후보 reject는 없다. Nonfinite·잘못된 shape·입력 identity·commit 불일치 같은 기술 오류는 결과로 감추지 않고 실패로 기록한다.

## 9 최종 평가와 commit

후보25에서 추가 update 없이 전체 current native subject forward와 전체 current actual weight forward를 실행한다. Native/actual NLL·KL, subject 정합, teacher distillation 값, entry/actual key 차이, group norm과 실제 ΔW norm을 기록한다. 이는 native 입력 관측이며 official P/N을 후보 선택에 사용하지 않는다.

Actual branch가 사용한 materialized FP32 가중치를 그대로 commit한다. Final current-key re-solve·후처리 strength 배율·절대-z residual 보상은 없다. 품질 수치가 약하더라도 별도의 명시적 실험 정책 없이 다른 후보를 고르거나 예산을 늘리지 않는다.

History에는 이 actual 후보 forward에서 포착한 각 층 **자기 down_proj 직전 current key**의 full-context second moment를 요청당 한 번 더한다. 상층 key는 하층 actual write를 이미 포함한다. δ-only branch key나 entry key를 실제 committed history로 표기하지 않는다. Current actual NLL을 memory의 b로 재사용한다.

Subject 목표와 actual 모델이 정확히 같아진다는 보장은 없다. Fixed P, ridge shrinkage, context 차이, 모든 토큰의 부수 작용이 남는다. 이 차이를 숨기지 않고 보조 학습과 최종 telemetry로 연결하는 것이 v6의 범위다.

## 10 어떤 cross effect가 들어가는가

| 효과 | 계산 경로 | 한계 |
|---|---|---|
| 같은 요청의 하층 δ→상층 상태·task | 매 step native joint forward | 다른 요청 δ는 이 가상 forward에 직접 작용하지 않음 |
| 요청 간 공유 key/write mixing | G/E의 off-diagonal | entry key geometry 수준 |
| 하층 실제 write→상층 actual key | actual probe 및 최종 actual forward | probe subset·정해진 시점에서 관측 |
| 다른 요청 D가 현재 문항에 미치는 작용 | 전체 U를 사용한 actual 보조 gradient | selected native 문장 범위 |
| 과거 편집과 일반 성질의 변화 | past native NLL/KL 보존 | bounded memory 밖의 지식·공식 N 전체 보장 불가 |

## 11 계산량과 재사용

Native branch는 첫 편집층 clean output과 subject 이전 causal prefix/KV를 재사용한다. Actual branch에서는 모든 토큰의 가중치 효과가 있으므로 첫 수정 down_proj 직전까지만 캐시한다. 두 cache를 섞지 않는다. Commit이 발생하면 다음 batch의 entry-dependent cache/P를 새로 만든다.

물리 branch의 backward는 전체 weight gradient 대신 direct D VJP를 사용한다. 입력 gradient도 유지해 하층으로 전달한다. Forward materialization·cast·덧셈 순서는 최종 commit과 같게 한다. Native teacher는 기존 forward에서 추출하고, 필요한 위치의 full-vocabulary head만 계산한다.

Native full context bundle을 요청1개의 계산 단위로 삼으면, M=|S_t|일 때:

- Native forward 25B, backward 24B.
- 네 actual 보조 pulse의 합: forward/backward 각각 B+M.
- 마지막 actual forward B, backward0.
- 총 forward **27B+M**, backward **25B+M** bundle.

B100,M16이면 native2500F/2400B에 actual216F/116B bundle이 추가된다. 이는 실제 GPU 호출 수나 시간 배율이 아니다. 두 branch의 suffix 길이와 cache 범위가 다르며, setup/geometry/history/공식 평가 비용도 별도다. Qualification에서 branch별 처리 token·layer·head 수, materialization, wall time, peak memory를 측정한다. 사전 목표는 native joint fit 대비 추가 시간이40% 이내이나 **검증 전 보장하지 않는다**. 초과하면 timing 결과를 보고하고 별도 profile revision으로 다루며 현재 설계의 문장이나 일정을 몰래 줄이지 않는다.

## 12 Portability와 검증

Core는 실제 B, 가변 context/target 길이, 층별 차원, adapter의 eligible set을 사용한다. BS100·L4–L8·CounterFact·Llama는 첫 profile일 뿐 수식의 고정 조건이 아니다. Model adapter는 native block/subject와 write module의 좌표 대응, post-intervention readout, causal prefix 재사용 범위를 검증해야 한다. 현재 Llama adapter만 근거가 있으며 다른 architecture 지원 완료를 주장하지 않는다.

CPU 수학 검증은 G/E trace identity, dense/dual writer 일치, gradient finite difference, off-diagonal coupling, A/B 동일 경로 분할 성질, 작은 B의 partition·budget을 확인한다. 실제 모델 qualification은 두 branch의 zero/nonzero candidate, gradient, microbatch invariance, 마지막 actual 후보와 commit identity를 확인한다. 이 문서 작성 중 GPU qualification은 실행하지 않는다.

관측할 주요 값은 primary NLL/KL/norm, G/E 비용·gradient, pulse별 보조 loss·gradient, layer/request clamp 비율, virtual/actual subject action, 실제 ΔW 및 past 손실이다. Nominal norm 몫을 causal layer contribution으로 보고하지 않는다. Official R/P/N과 자유 생성·teacher-forced 지표는 별도 평가로 유지한다.

## 13 Claim과 남은 제한

검증할 claim은 **native subject 개입을 유지한 전 층 local δ 공동 학습에서 writer의 상태·실현오차·보존 비용을 반영해 배분을 개선할 수 있는가**다. Adam의 신규성, 완전한 alignment, 이웃 지식 보존 보장, 무조건적인 층 분산을 claim하지 않는다.

λ_W/λ_E/β_h/β_d와 고정 native 학습률은 사전 고정한 시작 profile이다. 낮은 계수는 v4의 포화를 남길 수 있고 높은 계수는 v5처럼 약한 획득을 만들 수 있다. Pilot의 역할은 학습 진행과 branch 정합·비용의 확인이며, 아직 PS/NS 개선을 예측할 근거는 없다. 두 arm의 차이는 정책 norm 형태로 제한해 해석한다.
