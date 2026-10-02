# JLZ v4 method 구조 심층 점검

2026-10-02. 현재 v4의 수식, 실행 source, 기존 W5 검토를 바탕으로 학습 문제와 실제 write 및 순차 보존의 연결을 점검했다. **층별 local-z의 공동 학습은 구현됐지만, 실제 공유 가중치가 만드는 편집·보존 효과를 공동 학습하는 문제와는 차이가 있다.** 아래는 원인 후보를 나열하는 데 그치지 않고, 어떤 연결이 성립하고 어떤 정보가 빠지는지 구분한 결과다.

이 문서는 후속 method 검토이며 현재 실행 명세를 대체하지 않는다. 원래 method/source/job을 변경하거나 새 GPU 실험을 실행하지 않았다. 13개 CPU 수학 예제로 구조적 성질을 검산했다. 반례는 Llama W5에서 해당 원인이 차지하는 비중을 측정한 실험이 아니다.

## 현재 method가 푸는 문제

Entry 가중치를 W_t, 모든 eligible layer의 local payload를 D={D_l}라 하자. Fit은 subject 한 위치에 해당 요청의 δ를 넣는 가상 모델 F_virtual(W_t,D)를 사용한다.

\[
J_\eta(D)=L_{\mathrm{native}}\big(F_{\mathrm{virtual}}(W_t,D)\big)
+\eta\sum_l V_l^{\mathrm{entry}}(D_l)/s_l^2.
\]

여기서 native 항에는 같은 rewrite 문장의 NLL, current‖own-entry KL, 각 층의 nonsquared delta norm이 들어간다. Fit 종료 뒤 실제 writer Φ_t가 D를 공유 가중치 수정으로 바꾼다.

\[
W_{t+1}=\Phi_t(D),\qquad
U_l=D_lP_l(K_l(U_{<l}))^\top.
\]

최종 RS/PS/NS는 F(W_{t+1})에서 측정한다. 현재 최적화는 L_native(F_virtual)를 미분하며, L_native(F(Φ_t(D)))를 미분하지 않는다. B의 V는 이 차이를 완전히 연결하는 항이 아니다.

| 구조 | 구현 상태 | 해석의 경계 |
|---|---|---|
| 모든 eligible 층의 δ 공동 학습 | 구현 | 사전 layer subset 없음 |
| 한 요청 내부 층간 task cross effect | 구현 | 가상 subject intervention 경로에 대한 효과 |
| A의 요청 간 task coupling | 없음 | A의 fit은 요청별로 분리 가능 |
| B의 요청 간 geometry coupling | 구현 | full Q의 비대각 성분 유지 |
| 실제 shared writer의 요청 간 NLL/KL 간섭 | task graph에 없음 | V에서 일부 geometry만 반영 |
| 현재 key로 순차 write | 구현 | lower actual write 뒤 key 재측정 |
| 최적화에서 current-key writer 미분 | 없음 | entry geometry와 virtual graph 사용 |
| 보호 key에 대한 local write energy | 구현 | 의미적 NS와 동일하지 않음 |
| 보존의 층간·시간 누적 cross effect | 직접 포함하지 않음 | 일반 locality 보장 불가 |

A를 손상 인식 정책이라고 부르면 안 된다. A는 공동 local-z control이고, B가 history geometry를 z 단계에 넣는 policy arm이다.

## 정렬은 세 단계로 나누어야 한다

첫째는 좌표의 정렬이다. 해당 층의 block output에서 δ를 학습하고 같은 층 down_proj의 additive write로 연결한다. 이 부분은 v4에 반영됐다.

둘째는 개입 연산의 정렬이다. 가상 경로는 request ID로 해당 subject 위치에만 δ를 넣는다. 실제 writer는 같은 가중치를 모든 요청과 토큰에 적용한다. 이 부분은 일치하지 않는다.

셋째는 최적화 궤적과 commit의 정렬이다. Fit은 가상 하층 개입을 본 상층을 학습한다. Commit은 실제 하층 write를 본 상층 key를 사용한다. 현재는 이 두 경로의 차이를 다시 공동 최적화하지 않는다.

전체 hidden state의 가상/실제 경로를 H^v,H^a라 쓰면,

\[
H_l^v=f_l(H_{l-1}^v;W_l)+S_lD_l,\qquad
H_l^a=f_l(H_{l-1}^a;W_l)+U_lX_l(H_{l-1}^a).
\]

S_l은 요청별 subject 위치 선택이다. 이 차이는 정확히

\[
e_l=f_l(H_{l-1}^a)-f_l(H_{l-1}^v)
+U_lX_l(H_{l-1}^a)-S_lD_l
\]

로 분해된다. 하층의 실현 오차 전파와 해당 층에서 새로 발생한 개입 차이가 함께 있다.

현재 local-z는 문맥의 현재 표현에 더하는 공유 증분이다. 가상 경로의 절대 z_l을 실제 모델이 그대로 재현한다는 의미가 아니다. 절대 z에 맞추려고 z_virtual−h_actual을 upper residual로 다시 쓰면, 하층 오차를 상층이 보상하는 새 배분 방법이 된다. 이는 사소한 bug fix가 아니다. 무조건 Q의 역수를 곱하는 보정도 작은 고유값 방향의 write와 손상을 증폭할 수 있다.

## 요청과 문맥의 결합이 빠지는 지점

A의 목적은 요청별로 분리된다.

\[
J_A=\sum_r f_r(\delta_{1r},\ldots,\delta_{Lr})
+\sum_{l,r}\lambda_{\mathrm{norm}}\|\delta_{lr}\|/a_{lr}^2.
\]

하지만 실제 write에서 request r이 받는 변화는

\[
Uk_r=\sum_j\delta_j(p_j^\top k_r)
\]

이다. 다른 요청의 δ가 자신의 NLL/KL에 미치는 영향은 가상 task graph에 없다. 같은 층의 key가 동일한 두 요청은 하나의 linear write로 서로 다른 변위를 받을 수 없지만, 가상 request-ID injection은 이를 허용한다. CPU 예제에서 K=[1,1], D=[1,−1]의 ridge writer는 U=0이다. B는 이 미실현을 비용으로 부과하지만 task NLL은 여전히 가상 ±1을 본다.

문맥별 차이도 평균에서 사라진다. 현재 native NLL의 여섯 rewrite 가중치는 w_c=1/6이고, writer key의 native 가중치는 α=(.5,.1,.1,.1,.1,.1)이다. 이 차이는 의도된 native 규약으로, 구현 오류가 아니다.

μ_r^α=Σ_cα_ck_rc, μ_r^w=Σ_cw_ck_rc라 하면 실제 문맥별 실현 오차는

\[
\sum_cw_c\|Uk_{rc}-\delta_r\|^2
=\|U\mu_r^w-\delta_r\|^2+\operatorname{tr}(U\Sigma_r^wU^\top),
\]
\[
U\mu_r^w-\delta_r
=D(Q-I)e_r+U(\mu_r^w-\mu_r^\alpha).
\]

따라서 pooled DQ−D 외에 task/writer 평균의 차이와 문맥 분산의 오차가 있다. 좋은 pooled cosine만으로 여섯 context의 실제 개입이 같아졌다고 판단할 수 없다.

History도 평균 key의 Gram만 누적한다. 정확한 항등식

\[
\sum_c\alpha_c\|Uk_c\|^2
=\|U\bar k\|^2+\sum_c\alpha_c\|U(k_c-\bar k)\|^2
\]

에서 H는 첫 항을 담고 두 번째 문맥 분산은 따로 담지 않는다. 이것은 C0가 그 방향을 전혀 보호하지 않는다는 뜻이 아니라, H가 과거 문맥별 보호를 완전히 나타내지는 않는다는 뜻이다.

같은 기존 문맥으로 이를 보완하는 geometry를 정의할 수 있다. 가중치 합이 요청마다 1이고 모든 문맥에 같은 D_r를 요구하면,

\[
\mathrm{Scatter}=\sum_{r,c}\alpha_{rc}(k_{rc}-\bar k_r)(k_{rc}-\bar k_r)^\top,
\]
\[
M=A+\bar K\bar K^\top+\mathrm{Scatter},\qquad
U^*=D\bar K^\top M^{-1}
\]

가 context 전체의 weighted ridge 해다. 최적값은 .5 tr[D(I−KbarᵀM⁻¹Kbar)Dᵀ]이다. 이는 기존 writer를 그대로 둔 구현 최적화가 아니라 새로운 writer 목적이다. Context-aware writer와 context-aware H append도 각각 독립된 변경이다.

CPU 예제에서는 이 수정이 α-weighted ridge 목적을 .375→.3333으로 낮추지만, 균등 task 가중의 실현 제곱 오차는 .8333→.8519로 증가한다. 따라서 분산 보완만으로 NLL이나 NS 개선을 보장한다고 말할 수 없다.

## B는 어떤 배분을 선호하는가

V는 정확한 entry ridge 최적값이다.

\[
V_l=\tfrac12\|U_lK_l-D_l\|_F^2+
\tfrac12\operatorname{tr}(U_lA_lU_l^\top).
\]

보호 정보가 전혀 없는 항은 아니다. A≻0에서 고정 feature key k에 대해

\[
\|Uk\|^2\le (k^\top A^{-1}k)\operatorname{tr}(UAU^\top)
\le 2V(k^\top A^{-1}k)
\]

가 성립한다. 다만 key의 leverage, downstream 민감도, 층 결합, 답의 선호 margin을 알아야 이를 NS 제한으로 바꿀 수 있다.

V는 출력공간의 직교변환 O에 대해 V(OD)=V(D)이다. 같은 크기의 local 변위가 최종 답에 미치는 민감도가 다르면 V는 그 차이를 놓친다. CPU 예제에서는 같은 V를 가진 두 방향의 기능적 제곱 비용이 100배 다르다.

또한 **B에는 분산을 선호하는 quadratic 성질이 있다.** 동일한 효과 s=Σ_lδ_l를 만드는 동등한 양의 scalar 경로에서 anchor a와 ridge 계수 e가 같다고 하자. Native group norm은 λs/a²로 동일하지만 B의 추가항은

\[
\frac{\eta e}{2a^2}\sum_l\delta_l^2
\]

이므로 균등 분할에서 최소이며 비용은 1/L로 줄어든다. 실제 Transformer 층이 동등하다는 주장은 아니다. 하지만 목적 자체가 집중과 분산에 완전히 중립인 것은 아니다. 한 층 집중은 여전히 허용되며, 효율·geometry 차이가 이 분산 선호를 이길 때 선택될 수 있다.

사용자 목표가 손상이 작은 위치에 쓰는 것이라면, 분산 자체를 선호하는 성질과 실제 기능적 손상에 따른 배분을 분리해서 해석해야 한다.

## norm과 solver의 역할

현재 nonsquared native norm의 합은 weighted group penalty다. 정확한 최적해가 일부 layer block을 0으로 둘 수 있으므로, norm 자체가 집중을 선택할 수 없다는 해석은 틀리다.

Zero block에서의 최적성 조건은 smooth gradient를 g_lr라 할 때

\[
\|g_{lr}\|\le \lambda_{\mathrm{norm}}/a_{lr}^2
\]

이다. 현재는 norm=0에서 subgradient 0을 고른 일반 Adam을 사용한다. 이 threshold를 만족하는 작은 gradient도 최초 update에서 활성화될 수 있다. CPU의 convex 예제에서 정확한 최적점은 0인데 해당 Adam step이 약 .1로 이동하고 목적을 증가시킨다.

이 예제는 현재 Llama의 정확한 최적점이 sparse라는 증거가 아니다. W5 후보의 전체 gradient/KKT residual이 저장되지 않아, 상한 포화를 곧바로 미수렴이나 잘못된 최적해라고 단정할 수도 없다.

Native norm 목적을 유지하면서 group proximal solver를 사용해 이 구조를 다룰 수 있다. Scalar step의 Euclidean prox와 Adam의 좌표별 metric prox는 다르다. 단순 shrinkage를 Adam 뒤에 붙이고 정확한 proximal Adam이라고 부르면 안 된다. Solver 변경 후 같은 문장·목적·후보 예산의 의미를 다시 명시해야 한다.

공동 예산도 무엇을 일정하게 할지 먼저 정해야 한다. 동등한 scalar 경로에서:

| 제약 | 가능한 총 효과의 최대값 |
|---|---:|
| 층별 독립 native ball | Lca |
| 공동 상대 L2 ball | aρ√L |
| 공동 group-L1 ball | aρ |

따라서 앞선 W5 리뷰에서 예시로 제안한 공동 L2 예산은 전체 사용량을 줄일 수 있지만, 중복 경로 수에 따른 효과 증가를 완전히 제거하지 못한다. Group-L1 예산은 이 반례에서 한 층 집중과 분산을 모두 허용하면서 총량을 일정하게 한다. 실제 서로 다른 층의 의미적 효율까지 보정해 주는 것은 아니다. 최종 출력 반응에 대한 trust region은 더 직접적이지만 계산과 보호 문맥의 정의가 필요하다.

## 순차 보존의 시간 방향 문제

H는 저장 당시의 key에 새 U가 작용하는 크기를 제어한다. 같은 batch에서의 prewrite key 재사용은 올바르다. 하지만 이후 하층 편집이 과거 문맥의 상층 key를 바꾸는 것은 별개의 문제다.

현재 Llama profile의 L4 key는 수정되는 연산보다 앞에서 만들어지므로 하층 편집으로 변하지 않는다. L5~L8의 과거 key와 C0의 feature 기준은 이후 하층 write로 달라질 수 있다. H의 합계만으로 현재 key를 복원할 수 없다.

과거 입력의 선형 출력 변화도

\[
W_tk_t-W_{t-1}k_{t-1}
=U_tk_{t-1}+W_t(k_t-k_{t-1})
\]

이다. History가 근사하는 새 weight 작용 외에 key 자체의 변화가 있다.

시간 방향의 cross term도 누락된다.

\[
\left\|\sum_tU_tC_0^{1/2}\right\|_F^2
=\sum_t\|U_tC_0^{1/2}\|_F^2+
2\sum_{s<t}\langle U_sC_0^{1/2},U_tC_0^{1/2}\rangle.
\]

작은 증분이 같은 방향으로 누적되는 경우와 상쇄되는 경우는 증분 norm만으로 구분되지 않는다. 누적 C0 편차 항은 이 정보를 추가할 수 있지만, 의도한 edit까지 W0로 되돌리려는 힘을 만들 수 있다. 무조건 유리한 보존항은 아니다.

Native KL의 보호 범위도 좁다. 현재 subject-last에서 읽으므로 causal decoder에서 뒤의 “ is a”는 해당 KL에 영향을 주지 않는다. 보호 대상은 현재 subject prefix의 next-token 분포이며, neighborhood relation answer 전체가 아니다. Teacher는 batch마다 own entry로 교체된다.

같은 입력의 실제 분포에 KL을 적용하는 더 강한 가정에서도 작은 stepwise KL만으로 작은 전체 drift가 보장되지는 않는다. CPU Bernoulli 예제에서 각 step KL은 최대 .000235인데 20 step 뒤 초기 분포 대비 KL은 .0823이다. 현재 구현은 보호 입력도 batch마다 달라지고 가상 intervention을 사용하므로 보호 범위가 더 제한된다.

## 현재 key를 실제 목적에 연결할 때의 미분

실제 writer는 K_l=K_l(U_<l), P_l=(A_l+K_lK_lᵀ)⁻¹K_l의 삼각 구조다. M=A+KKᵀ라 하면,

\[
dP=M^{-1}[dK-(dKK^\top+KdK^\top)P],\qquad
dU=dDP^\top+D(dP)^\top.
\]

하층 D는 상층 hidden뿐 아니라 상층 writer의 gain과 요청 혼합을 바꾼다. 가상 task graph와 고정 entry V에는 이 writer 미분이 없다.

D=0에서는 D(dP)ᵀ=0이므로 올바른 all-token 물리 경로에서 P를 고정해도 원점의 1차 미분은 일치할 수 있다. 그러나 현재 subject-only 경로는 원점부터 토큰·요청의 Pᵀk 작용을 생략한다. 따라서 단순히 P 갱신 주기를 늘리는 것으로 개입 연산의 차이를 해결할 수 없다.

CPU의 비영점 예제에서 실제 current-key writer와 frozen-P writer는 같은 출력 −.2를 갖지만, 손실의 하층 δ 미분은 각각 +.168과 −.12다. 가상 경로 미분은 −2다. Frozen-P 근사가 방향까지 달라질 수 있음을 보이는 반례이며 실제 W5 gradient가 반대였다는 증거는 아니다.

## Batch와 model 범용성

현재 B의 s_l²=mean_r a_lr²는 무관한 요청의 anchor도 다른 요청의 policy gradient 크기를 바꾸게 한다. Q가 diagonal이고 두 번째 δ가 0이어도, 두 번째 anchor²가 첫 번째의 99배라면 첫 요청의 policy gradient는 단독 batch 대비 50배 작아진다. 이것은 정당한 Q 비대각 간섭과 별개의 batch 구성 의존성이다.

고정 layer별 profile scale이나 처음부터 유도한 요청별 weighted ridge가 대안이다. D column을 단순히 나누는 변경은 기존 ridge 최적값의 의미를 바꾸므로 조용히 적용할 수 없다.

Count 기반 ridge 자체도 batch에 의존한다. 같은 key와 payload를 m번 넣으면 V_m=m||d||²/[2(1+m kᵀA⁻¹k)]이다. 이는 implementation bug가 아니며 logical batch가 바뀌면 writer와 요청당 policy가 달라진다는 뜻이다. 임의 B/partial batch 지원과 결과의 batch 불변성은 구분해야 한다.

Model 좌표 scale도 중요하다. 기능적으로 동등한 재조정 h'=ch,δ'=cδ에서 상대 clamp는 같지만 native norm은 1/c배가 된다. 절대 Adam lr의 첫 상대 step도 대략 lr√d/||h0||에 의존한다. 모델별 native profile은 유지하되 같은 lr·η가 같은 개입량을 뜻한다고 가정하면 안 된다. 현재 구현은 Llama/CounterFact adapter만 제공하며, 일반 interface 설계와 타 모델 실행 검증은 별개다.

## 재설계 판단

현재 구조를 다음 세 수준으로 구분하는 것이 타당하다. 세 방향 모두 native rewrite 문장, 전체 eligible layer, local payload 공동 변수, 집중 허용을 유지할 수 있다.

| 방향 | 핵심 변경 | 계산상 의미 | 남는 한계 |
|---|---|---|---|
| 최소 변경 | norm의 proximal 처리, 명시적 공동 자원, batch scale 보정, context-aware H | 주로 optimizer/geometry 변경, 추가 Transformer pass 없이 가능한 부분 존재 | 가상 개입과 실제 write의 차이는 남음 |
| writer를 학습에 연결 | all-token U(D)로 같은 native NLL/KL 계산, 명시적 geometry refresh | 저랭크 gradient 가능, subject-prefix cache 대부분 무효, layer별 전체 B key 필요 | 고정 P이면 근사, native 문맥만으로 이웃 보존 보장 불가 |
| 기능적 순차 보존 | 일반 지식의 고정 기준과 과거 edit의 목표를 구분하고 층·시간 효과 평가 | 독립 reference forward/backward 또는 민감도 근사 비용 | Reference 범위·충돌·누출 규약 추가 필요 |

최소 변경은 위험을 줄이는 방향이며 구조 전체를 닫는 해결책은 아니다. 핵심 claim을 강하게 세우려면 학습할 D와 실제 배포할 writer의 연결을 명시해야 한다. Writer를 목적에 넣는 경우 native loss의 항과 문장은 유지할 수 있어도, 원래 compute_z의 가상 개입 문제와 수학적으로 같은 목적이라고 부르면 안 된다.

저랭크 적용은 XUᵀ=(XP)Dᵀ이고, output gradient G에 대해 ∇_D L=Gᵀ(XP), ∇_P L=Xᵀ(GD)이다. 전체 weight gradient 없이 구현할 수 있다. 그러나 actual FP32 materialization/cast와 저랭크 연산 재배치의 수치 의미를 검증해야 한다.

정확한 triangular writer는 각 층에서 전체 logical batch의 key를 모아야 한다. 현재처럼 microbatch별 전체 forward/backward를 독립적으로 끝내는 스케줄은 유지할 수 없다. 첫 write 연산 이전의 prefix만 안전하게 재사용하고, activation checkpoint/offload와 solve 미분을 함께 설계해야 한다. 동일 subject prefix의 teacher-forced 행에서 key를 얻으면 별도 target-free 모델 pass가 필수인 것은 아니다.

Frozen-P inner loop를 쓰면 cheaper approximation이라고 명시해야 한다. 마지막에 current P를 다시 계산해서 그대로 commit하면 다시 학습/commit 차이가 생긴다. 마지막 실제 writer 후보 평가 또는 고정 P commit 계약 중 하나를 정해야 한다.

Context-aware H는 이미 계산하는 문맥 key를 사용할 수 있지만 Gram 연산은 증가한다. Context-aware writer를 low-rank Woodbury로 처리하면 기존 A factor를 재사용할 수 있으나 RHS/dual 크기가 늘어난다. 추가 Transformer pass가 없다는 것과 총비용이 같다는 것은 다르다.

일반 지식의 기능적 보존을 추가할 때 공식 P/N, 평가 정답, 미래 edit를 학습하지 않는다. Native 학습 문장을 그대로 유지하면서 별도 보존 reference를 추가하는 것은 새 method 선택이다. 기존 C0/H만으로 보호 정보를 더 얻은 것처럼 설명해서는 안 된다. 원래 일반 지식과 의도한 새 지식이 충돌할 때 보호 우선순위도 정해야 한다.

사용자 의도와 맞는 순서는 “모든 후보 층에서 local-z를 공동 결정 → 실제 write의 편집 효과와 보존 비용으로 배분 → 필요한 만큼만 commit”이다. 현재 구현의 실제 연결은 “가상 subject 개입으로 성공 → pooled entry geometry로 비용 부과 → 다른 current-key 경로로 실제 write”다. 이 연결을 정합하게 만드는 것이 핵심 재구축 과제다. 단순 η 증가, 작은 layer subset, 강제 균등화, 후보의 공식 NS gate만으로 해결하려 해서는 안 된다.

## 근거와 검증 범위

- [이전 W5 원자료 검토](../2026-10-02-jlz-v4-w5-locality-review/report-ko.md)
- [13개 CPU 수학 예제](../../../local/jlz-v4-method-structure/counterexamples.py), [검산 결과](../../../local/jlz-v4-method-structure/counterexamples.json)
- [oracle](../../../project/run_scripts/jlz_native_joint/oracle.py): 요청별 native loss·KL readout·own-entry teacher
- [adapter](../../../project/run_scripts/jlz_native_joint/adapter.py): request-ID subject injection
- [optimizer](../../../project/run_scripts/jlz_native_joint/optimize.py): group norm·Adam·batch scale·entry V
- [geometry](../../../project/run_scripts/jlz_native_joint/allocation.py): ridge solve와 V
- [writer](../../../project/run_scripts/jlz_native_joint/writer.py): current-key triangular commit·history append
- [input pooling](../../../project/run_scripts/jlz_native_joint/inputs.py), [native prompts](../../../project/run_scripts/jlz_pilot/prompts.py): task와 writer 문맥 가중치
- [prefix cache](../../../project/run_scripts/jlz_native_joint/entry.py): subject-only intervention에서의 재사용 조건

13개 검산은 등가 경로의 분산 선호, norm zero block, batch scale, context pooling, 출력 방향, 층간 보존, history key drift, 시간 cross term, moving teacher, triangular gradient, 중복 batch geometry, 동일 key 충돌, 좌표 scale을 다룬다. 모두 stdlib CPU이며 model forward 0, GPU 호출 0이다. 반례의 수치와 실제 W5 관측을 혼동하지 않는다. 이번 결과는 method의 정의·한계와 수정 우선순위에 대한 점검이며 개선 성능이나 가속 검증이 아니다.
