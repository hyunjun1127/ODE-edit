**편집 진척을 보존하는 공간에서 손상 증가 성분을 제거하는 native-metric refinement 설계**

작성: 2026-09-10 KST. 상태: 연구 설계 초안. 모델/GPU 실험과 controller 구현은 아직 실행하지 않았다. 작은 합성 행렬로 projection identity와 bounded barrier의 closed-form 분기를 검산했다.

이 문서는 앞선 risk-only elastic-barrier 설계의 주 방법을 수정한다. 사용자 요구는 “편집 진척을 유지하는 공간 안에서 손상 증가 성분을 제거한다”이다. Native warm start, 동일 basis, actual-output feedback을 유지하되, correction이 edit 진척을 바꾸지 않도록 먼저 공간을 정한다. Scientific pass/fail gate는 추가하지 않는다.

**1. 주 방법은 편집 진척에 해당하는 성분을 유지하고, 자유 성분 중 risk를 높이는 성분만 완전히 제거한다.**

Nominal step을 d0, 현재 edit-loss observer를 J, cumulative-risk gradient를 a라고 하자. 주 방법의 목표는

\[
Jd=Jd_0
\]

를 유지하면서, nominal의 progress-null 성분 안에서 risk를 높이는 부분을 제거하는 것이다. 여기서 유지하는 것은 현재 loss의 일차 변화율이다. 모든 요청의 finite-step NLL 또는 PS가 정확히 같다는 뜻은 아니다.

주 방법 이름은 이 문서에서 `EP-Free`로 둔다. Edit progress를 보존하는 free component 안에서 선택적으로 제거한다는 뜻이다. Risk-only 제거는 비교군으로 남긴다.

ODE-M은 calibration-loss gradient에 정렬된 이동과 직교하는 이동을 나누고, loss 상승 성분만 선택적으로 감쇠한다. 별도의 editing-loss 변화율을 정확하게 유지하는 조건은 없다. 이번 방법은 edit observer와 risk observer를 따로 두는 확장이다. 논문 아이디어의 확장이지 ODE-M 전체 알고리즘이나 공개 코드의 재현은 아니다. [ODE-M 원문 §3.2](https://arxiv.org/html/2605.19409v1)

**2. Native geometry와 nominal objective는 앞선 비교를 유지한다.**

모델은 Llama3-8B-Instruct, 수정 weight는 `model.layers.4.mlp.down_proj.weight` 하나다. Shape는 [4096,14336]이고, native basis U의 shape는 [14336,100]이다.

\[
W(X)=W_N+XU^\top,\qquad X_0=0,\qquad X\in\mathbb R^{4096\times100}.
\]

- W0: 원래 pretrained weight. Cumulative risk의 reference다.
- We: 이번 batch 편집 직전 weight. KL teacher와 batch action의 reference다.
- WN: We에서 native writer를 한 번 적용한 실제 FP32 endpoint.
- DeltaN = WN − We.
- M: 현재 batch가 들어오기 전 history statistic.
- K: native가 실제 사용한 request별 key 100개.
- U: 기존 rank-100 native input basis의 orthonormal 표현.

WN을 U로 다시 재구성해서 시작하지 않는다. 정확한 저장 WN을 anchor로 사용한다. 수정 공간은 full-Q로 넓히지 않는다.

공통 loss는

\[
L(X)=L_E(W(X))+0.0625L_{\rm essence}(W(X);W_e)
+0.1\frac{J_N(\Delta_N+XU^\top)}{J_N(\Delta_N)},
\]

\[
J_N(\Delta)=\operatorname{tr}\{\Delta(M+I)\Delta^\top\}
\]

다. Native initialization을 포함한 전체 batch update에 penalty를 건다. Reference를 WN으로 reset하지 않는다. Current target NLL은 기존 6-context flat mean, request/token 평균을 유지하고, 실제 수정 weight가 모든 token에 적용된 output에서 계산한다.

\[
H_R=U^\top(M+I)U,\quad Z=U^\top K,\quad H_N=ZZ^\top+H_R.
\]

H_R는 loss의 native-action metric이고 H_N은 direction metric이다. Prepared MN은 이미 current KK.T를 포함하므로 M 대신 무심코 사용하지 않는다. Native의 비대칭 원 solve 행렬을 그대로 Hessian으로 쓰지 않는다.

Nominal step은

\[
d_0=-\eta G_{\rm total}H_N^{-1},\qquad G_{\rm total}=\nabla_XL.
\]

이후 수식은 coefficient를 vector로 펼친 표기를 쓴다. Metric 연산자 B는 matrix 표현에서 d를 d H_N / eta로 보내는 연산이다. 즉

\[
\|d\|_B^2=\operatorname{tr}(dH_Nd^\top)/\eta,
\qquad d_0=-B^{-1}g_{\rm total}.
\]

409600×409600 행렬을 실제로 만들지 않는다. 100×100 solve와 4×4 Gram으로 계산한다.

**3. 진척 observer는 total objective가 아니라 current target NLL로 만든다.**

Current100을 25 requests씩 네 group으로 고정한다. Case ID를 `EP-PROGRESS-GROUP-V1|case_id`의 SHA256 값으로 정렬하고 연속 25개씩 나눈다. 성능, loss, target 난이도를 보고 group을 바꾸지 않는다. Group은 해당 batch의 모든 method/step에서 같다.

\[
\ell_g(X)=\frac1{25}\sum_{i\in\mathcal G_g}
\frac1{6}\sum_{c\in C_i}\mathrm{NLL}_{i,c}(W(X)),
\qquad g=1,\ldots,4.
\]

\[
J_s=\begin{bmatrix}
\nabla_X\ell_1(X_s)^\top\\
\nabla_X\ell_2(X_s)^\top\\
\nabla_X\ell_3(X_s)^\top\\
\nabla_X\ell_4(X_s)^\top
\end{bmatrix}.
\]

보호 조건은

\[
J_s(d_s-d_{0,s})=0.
\]

네 group의 nominal 변화율을 유지하면 전체 평균 edit NLL의 변화율도 유지된다. Mean gradient 하나는 group 사이의 상쇄를 허용하므로 Middle의 보조 비교로 둔다.

이 equality의 우변은 임의의 target schedule이 아니라 J_s d0다. 따라서 d0 자체가 항상 feasible하다. Request별 loss ceiling 100개나 외부 terminal threshold를 요구하지 않는다.

J에는 KL과 native action을 넣지 않는다. Total objective의 gradient를 보호하면 target NLL 악화를 regularizer 개선으로 상쇄할 수 있다. 반대로 loss의 모든 구성항을 observer에 넣으면 d0 전체가 보호 span에 포함되어 free component가 사라질 수 있다.

Nominal에서 어떤 group의 J_g d0가 양수이면 그 group의 NLL은 일차적으로 증가하는 방향이다. EP-Free는 그것도 그대로 유지한다. 이 경우 “양의 편집 진척을 확보했다”가 아니라 “nominal의 변화율을 보존했다”고 보고한다. 진척 보존과 진척 생성은 다른 문제다.

네 scalar 조건은 409600차원 coefficient 공간에 걸린다. U의 input rank가 100이라고 해서 네 조건이 수정 공간을 거의 소진하는 것은 아니다. 다만 free 성분의 실제 크기는 별도로 측정한다.

**4. Native metric 안에서 편집 진척 성분과 자유 성분을 분해한다.**

\[
S=JB^{-1}J^\top,
\]

\[
d_{\rm prog}=B^{-1}J^\top S^\dagger Jd_0,
\qquad d_{\rm free}=d_0-d_{\rm prog}.
\]

그러면

\[
Jd_{\rm prog}=Jd_0,\qquad Jd_{\rm free}=0,
\]

\[
\langle d_{\rm prog},d_{\rm free}\rangle_B=0.
\]

d_prog는 지정한 nominal edit 변화율을 실현하는 최소 B-norm step이다. d_free는 현재 네 observer에 일차 변화를 주지 않는 성분이다. 후자를 일반화·과거 지식·finite-step 동작에 전혀 쓸모없는 성분이라고 부르지 않는다.

여기에는 중요한 구조적 관측이 있다. G_edit는 네 group gradient의 평균이므로 B^-1 G_edit는 progress span 안에 있다. 정확한 연산에서 d0의 free 성분은 KL/action 방향의 progress-null 성분에서만 나오며, 실제 수치 오차는 별도로 기록한다. Free 비중이 작게 나오는 것은 자연스러운 결과일 수 있다. 이를 숨기려고 group 수나 보호 조건을 자동으로 줄이지 않는다.

**5. 주 risk는 기존 C0를 사용한 cumulative mapping deformation으로 둔다.**

\[
D(X)=W_N-W_0+XU^\top,
\]

\[
R_C(X)=\tfrac12\operatorname{tr}\{D(X)C_0D(X)^\top\}.
\]

C0는 기존 native asset의 second moment를 재사용한다. 새 locality prompt나 held-out neighborhood label을 controller에 넣지 않는다. 기존 ABC에서 covariance 방향이 Frobenius보다 큰 NS 개선을 보였다는 개발 결과에 따라 첫 후보로 선택한다. 이는 독립 검증에서 확정된 최적 risk라는 뜻은 아니다.

\[
H_C=U^\top C_0U,\qquad C_C=(W_N-W_0)C_0U,
\]

\[
a_C(X)=\nabla_XR_C=C_C+XH_C.
\]

Risk 변화는 작은 행렬로 계산할 수 있다.

\[
R_C(X+d)-R_C(X)
=\langle a_C(X),d\rangle+
\tfrac12\operatorname{tr}(dH_Cd^\top).
\]

Frobenius는 동일한 식에서 C0를 I로 바꾼 risk다. Middle에서 RiskOnly/EP-Free를 모두 Frobenius로 반복해 risk 선택과 projection 선택을 분리한다.

Covariance risk는 실제 locality loss의 대리 지표다. 이 risk를 줄였다는 사실로 NS나 과거 edit 보존을 대체하지 않는다. Native projector/basis 때문에 이 risk가 거의 보이지 않는 경우도 가능하므로 projected risk gradient와 제거량을 기록한다.

EP-Free의 projection 비율은 R을 양의 상수배 해도 변하지 않는다. 작은 수를 큰 수로 보이게 하려는 임의 gradient normalization이나 risk scale tuning은 필요 없다.

**6. Risk gradient도 progress-null 공간으로 제한한 뒤 제거한다.**

\[
\mathcal K_B=B^{-1}-B^{-1}J^\top S^\dagger JB^{-1},
\]

\[
w=\mathcal K_Ba,\qquad q=a^\top w\ge0.
\]

Jw=0이므로 w 방향 correction은 nominal edit 변화율을 바꾸지 않는다. 여기서 a는 같은 현재 state의 risk gradient다.

Risk 기여를 두 부분으로 나눈다.

\[
p_R=a^\top d_{\rm prog},\qquad
k_R=a^\top d_{\rm free},\qquad
a^\top d_0=p_R+k_R.
\]

주 update는

\[
\boxed{d_{\rm EP}=d_0-\frac{[k_R]_+}{q}w.}
\]

q=0이면 수학적으로 k_R=0이므로 d_EP=d0다. 이때 임의 방향을 만들거나 작은 epsilon으로 나누어 방향을 증폭하지 않는다.

이 식은 다음 최소 수정 문제의 해다.

\[
\min_d\ \tfrac12\|d-d_0\|_B^2
\quad\text{s.t.}\quad
Jd=Jd_0,\quad a^\top d\le p_R.
\]

성립하는 성질은 세 가지다.

\[
\boxed{Jd_{\rm EP}=Jd_0}
\]

\[
\boxed{a^\top d_{\rm EP}=p_R+\min(k_R,0)}
\]

\[
\boxed{\|d_{\rm EP}\|_B^2
=\|d_0\|_B^2-\frac{[k_R]_+^2}{q}.}
\]

따라서 free 성분이 risk를 높이면 그 기여를 100% 제거하고, 낮추면 유지한다. 감쇠 계수를 별도로 sweep하지 않는다. 전체 B-norm도 늘지 않으며 correction norm은 원래 free component norm 이하이다.

이 norm은 native metric의 step 비용이다. Physical Frobenius norm, 전체 cumulative weight norm, actual NLL이 모두 감소한다는 뜻은 아니다.

“편집을 유지하면서 모든 locality 손상을 제거한다”는 보장은 하지 않는다. 여기서 제거하는 것은 **현재 네 edit observer에 일차 영향을 주지 않는 nominal 성분의, 선택한 cumulative risk에 대한 양의 일차 기여**다.

**7. Risk-only가 진척을 깎는 이유와 새 방법의 차이를 같은 수식으로 확인한다.**

RiskOnly 비교군은

\[
d_R=d_0-
\frac{[a^\top d_0]_+}{a^\top B^{-1}a}B^{-1}a
\]

로 둔다. 전체 risk 증가 성분을 완전히 제거하지만 J(d_R-d0)=0을 요구하지 않는다. 앞선 elastic controller의 수치 설정을 그대로 재현하는 arm은 아니며, 선택적 완전 제거를 비교하기 위한 risk-only 기준이다.

a.T B^-1 a=0이면 SPD metric에서 a=0이므로 d_R=d0로 둔다. 이 분모가 0인 경우 임의 방향이나 제거량을 만들지 않는다.

다음은 설명용 2차원 예시다. 실제 실험 결과가 아니다. B=I, J=(-1,0), risk gradient a=(1,1), d0=(1,1)로 둔다.

| 방법 | Step | Edit loss의 일차 변화 | Risk의 일차 변화 |
| --- | --- | ---: | ---: |
| Nominal | (1,1) | -1 | +2 |
| RiskOnly | (0,0) | 0 | 0 |
| EP-Free | (1,0) | -1 | +1 |
| 진척 보존 + 추가 risk 상쇄 | (1,-1) | -1 | 0 |

EP-Free는 편집 성분을 유지하고 원래 free 성분의 손상 기여를 없앤다. 마지막 행은 새로운 risk 감소 성분을 free 공간에 추가한 것이다. 기존 성분 제거와 보상 방향 생성은 다른 동작이다.

p_R를 “절대 피할 수 없는 총 손상”이라고 부르지 않는다. q>0이면 추가 free 방향으로 상쇄할 수 있다. 같은 nominal 진척과 같은 step 비용 안에서 가능한 최소 risk 변화율은

\[
\boxed{p_R-\sqrt q\,\|d_{\rm free}\|_B}
\]

다. 이 값을 진단에 남긴다.

**8. 고정 cap을 요구하는 barrier도 같은 진척 보존 공간에서 처리한다.**

EP-Free는 제거 가능한 성분의 선택적 삭제이며, 고정 global risk cap의 forward invariance를 보장하는 CBF라고 부르지 않는다. Cap까지 요구하는 경우에는 `EP-CBF`를 별도 companion으로 둔다.

\[
R_{\rm cap}=R_C(W_N),\qquad
h_s=R_{\rm cap}-R_C(W_s),\qquad b_s=0.25h_s.
\]

이때 d는 eta를 이미 곱한 한 iteration의 displacement다. Cap을 step마다 reset하지 않는다.

Progress equality만 넣은 CBF는

\[
d=d_0-
\frac{[a^\top d_0-b_s]_+}{q}w
\]

로 계산할 수 있다. 그러나 q가 작으면 새 보상 방향이 매우 커질 수 있다. 이를 방지하기 위해 companion은 **진척을 유지하며 nominal의 native-metric step 비용을 넘지 않는 범위**에서 cap을 맞춘다.

\[
Jd=Jd_0,\qquad \|d\|_B\le\|d_0\|_B.
\]

Risk 조건이 이 범위에서 불가능하면 risk slack을 남긴다. Edit equality를 풀거나 step을 폭증시켜 cap을 맞추지 않는다. 이 규칙은 실험의 통과 gate가 아니라 controller의 우선순위다.

이 companion도 작은 closed-form으로 계산할 수 있다. p=d_prog, f=d_free, r_f=||f||B로 두고 최종 d=p+z를 찾는다. q>0일 때

\[
n=w/\sqrt q,\quad
c_f=\langle n,f\rangle_B,\quad
t=(b_s-p_R)/\sqrt q.
\]

그러면 Jn=0, ||n||B=1이며 다음 문제로 줄어든다.

\[
\min_z\ \tfrac12\|z-f\|_B^2
\quad\text{s.t.}\quad
Jz=0,\ \langle n,z\rangle_B\le t,\ \|z\|_B\le r_f.
\]

해는 다음과 같다.

1. c_f<=t이면 z=f.
2. c_f>t이고 t>=-r_f이면 z0=f-(c_f-t)n을 구한다. ||z0||B<=r_f이면 z=z0.
3. 같은 feasible 구간에서 z0가 ball 밖이면 f_perp=f-c_f n에 대해

\[
z=tn+\sqrt{r_f^2-t^2}\frac{f_\perp}{\|f_\perp\|_B}.
\]

4. t<-r_f이면 cap이 이 비용 안에서 불가능하다. z=-r_f n을 택하고 최소 slack

\[
\xi=p_R-\sqrt q\,r_f-b_s>0
\]

를 기록한다. 이는 risk slack을 먼저 최소화하고 nominal과의 거리를 최소화하는 해다.

q=0이면 모든 progress-null 방향에서 risk 변화율이 같으므로 z=f를 유지하고 xi=[p_R-b_s]+를 기록한다. t=-r_f 경계는 z=-r_f n으로 직접 처리한다. Feasible한 두 제약 동시 활성 분기에서는 f_perp=0 문제가 생기지 않는다.

구현에서는 큰 t를 먼저 만들기보다 `a.T@d0<=b_s` 또는 `b_s<p_R-sqrt(q)*r_f`를 risk 단위에서 먼저 확인한다. 모든 분기는 수학적 controller 계산이며 성능 gate가 아니다.

결과적으로 EP-CBF도 nominal의 일차 진척을 유지하고 B-norm을 늘리지 않는다. 다만 기존 harmful 성분을 없애는 것에 더해 free 방향을 재배치할 수 있다. 실제 finite-step risk에는 quadratic remainder가 남는다.

**9. 실제 finite-step에서 edit 진척이 얼마나 유지되는지 직접 측정한다.**

J(d_EP-d0)=0이어도 nonlinear loss에서는

\[
\ell_g(X+d_{\rm EP})-\ell_g(X+d_0)
\]

가 0일 필요는 없다. 매 step 다음을 구분한다.

- Algebraic leakage: J_g(d-d0).
- 실제 적용 전후 group loss 변화: ell_g(X+d)-ell_g(X).
- Local remainder: 위 실제 변화 − J_g d.
- 같은 상태에서의 nominal 대비 실제 차이: ell_g(X+d)-ell_g(X+d0).

마지막 값은 별도 H trajectory의 현재 값으로 대신할 수 없다. Step1 이후 두 trajectory의 출발점이 다르기 때문이다.

Primary EP-Free의 각 entry에서 pre-state 0/3/7, 즉 첫째/넷째/여덟째 update를 정해 두고 실제 nominal shadow endpoint의 edit NLL을 측정한다. Pre0는 H의 첫 step 결과를 재사용하고, pre3/pre7은 같은 X에서 d0를 적용한 candidate를 forward-only로 평가한 뒤 원상 복원한다. 세 entry에서 총 여섯 번의 edit-only full-batch forward가 추가된다. Shadow를 채택할지 고르는 선택 과정은 없다.

Per-request NLL과 target-token 분포는 이 forward에서 함께 저장한다. Group 평균 유지가 소수 request 악화를 숨기는지 actual delta 분포로 본다. Per-request Jacobian100을 기본 controller에 추가하지 않는다.

Middle alpha=.003/.01/.03 비교로 finite-step 오차의 크기 의존성도 본다. NLL 차이가 커졌다고 자동 step rejection이나 무제한 line search를 넣지 않는다. 관측 후 nonlinear retraction이 필요한지 별도 질문으로 판단한다.

Continuous-time에서도 보장하는 것은 각 현재 상태에서 J(X)v(X)=J(X)u(X)이다. 서로 다른 상태를 거치는 두 방법의 전체 loss trajectory가 같다는 뜻은 아니다.

**10. 기본 실험은 15 trajectories와 작은 same-state probe 세 개다.**

기존 ABC의 Early W10→B11, Middle W50→B51, Late W90→B91을 사용한다. Current100, Fixed100, Past100과 native endpoints를 재사용한다.

| 구분 | 방법 / 조건 | Trajectories |
| --- | --- | ---: |
| 주 비교 | 세 entry × H, RiskOnly-COV, EP-Free-COV | 9 |
| Risk 종류 비교 | Middle × RiskOnly-Frob, EP-Free-Frob | 2 |
| Progress observer 비교 | Middle × EP-Free-COV, mean J1 | 1 |
| 고정 cap 비교 | Middle × bounded EP-CBF-COV, J4 | 1 |
| Step 크기 감도 | Middle × EP-Free-COV, alpha=.003/.03 | 2 |
| 합계 | N 세 endpoint는 기존 결과 재사용 | 15 |

각 trajectory는 8 steps다. 주 alpha=.01은 첫 nominal physical step을 ||DeltaN||F의 1%로 맞춘다는 뜻이다. G0에서 eta를 정한 후 고정한다. Momentum=0이며 모든 controller는 같은 eta를 사용한다. 보정 후 norm 재정규화는 하지 않는다.

G0=0이면 이동량 calibration을 강제로 만들지 않고 stationary branch로 기록한다. DeltaN=0이면 native 길이/action 정규화 단위가 퇴화하므로 no-action 상태로 구분한다.

J1은 이미 모은 네 group gradient의 평균으로 만든다. Nominal gradient나 데이터 순서를 바꾸지 않으므로 observer 변경만 비교할 수 있다. Covariance/Frobenius 변경도 nominal objective에는 영향을 주지 않는다.

이 round에서는 Euclidean/native metric을 다시 전체 factorial로 확장하지 않는다. Native metric을 공통 기반으로 고정하고 사용자가 지적한 progress 보존 기전에 집중한다. 별도 risk 계수·group 수·cap을 무제한 탐색하지 않는다.

**11. Risk 제거량 차이가 projection 효과로 오해되지 않도록 matched-removal probe를 둔다.**

RiskOnly는 전체 [a.T d0]+를 제거하고 EP-Free는 [a.T d_free]+만 제거한다. 둘의 차이는 보호 여부뿐 아니라 목표 제거량의 차이도 포함한다.

각 entry의 공통 WN에서 t_remove=[k_R]+를 사용한 unprotected 방향을 추가로 계산한다.

\[
d_{\rm match}=d_0-
\frac{t_{\rm remove}}{a^\top B^{-1}a}B^{-1}a.
\]

EP-Free와 이 방향은 같은 상태에서 **같은 일차 risk 감소량**을 만든다. 그러나 d_match에는 J correction=0 조건이 없다. 이 비교로 편집 진척을 보존하는 공간에서 보정한 효과를 더 직접적으로 볼 수 있다.

여기서도 a.T B^-1 a=0이면 a=0이고 t_remove=0이므로 d_match=d0로 둔다.

각 entry에서 alpha=.01의 단발 probe 하나만 실제 적용해 공통 train objective와 curve panel을 평가한다. 추가 backward는 필요 없다. 총 세 개의 추가 endpoint다. EP-Free의 첫 step은 기존 curve 결과를 재사용한다.

같은 일차 감소량이 actual finite-step risk까지 같다는 뜻은 아니다. 두 방향의 quadratic remainder, actual risk, 실제 edit NLL과 correction norm을 나란히 기록한다. 이후 각 method의 자체 trajectory에서는 출발 상태가 다르다는 점도 유지한다.

**12. 평가 시점과 기능적 결과는 기존 ABC 패널을 유지한다.**

| 시점 | 평가 |
| --- | --- |
| Step0 | 같은 WN의 Full 결과 재사용 |
| Step1/2/4 | Curve1100쌍 |
| Step8 | Full3900쌍; curve rows는 여기서 추출 |
| Pre-state0–7 | Group edit gradient, total gradient, 분해, controller 기록 |
| Terminal8 | Loss/risk/geometry forward-only; 추가 gradient 없음 |

Full은 각 Current/Fixed/Past에서 RS100, PS200, NS1000이다. Curve는 Current RS100/PS200, Fixed/Past RS 각100, 각 panel neighborhood200쌍이다. 중간 Fixed/Past PS는 측정하지 않으므로 보간해 채우지 않는다.

Middle의 N/H/RiskOnly-COV/EP-Free-COV에는 기존 20 rewrite + 40 rephrase 생성 panel을 사용한다. N 재사용 시 새 생성은 180 sequences다. 작은 literal-prefix 결과와 생성 원문을 보조로 남기며 semantic generation accuracy로 확대하지 않는다.

Controller는 held-out paraphrase/neighborhood의 label이나 metric을 읽지 않는다. C0는 기존 구조 통계다. 이번 결과는 이미 개발에 사용한 세 checkpoint/panel에 대한 진단이며 새 독립 benchmark가 아니다.

**13. 결과를 설명하는 핵심 진단값은 progress, free risk, 실제 기능의 세 묶음이다.**

Progress에는 다음을 저장한다.

- 네 group의 loss, Jd0, Jd, J(d-d0).
- 실제 group/request별 delta NLL과 p90.
- Same-state nominal 대비 finite-step 차이.
- J의 singular values, numerical rank, projection residual.

분해와 risk에는 다음을 저장한다.

- ||d0||B, ||d_prog||B, ||d_free||B와 free energy fraction.
- p_R, k_R, 제거량 [k_R]+, correction 전후 risk 변화율.
- q=a.T K_B a, q_all=a.T B^-1 a, q/q_all.
- 같은 progress와 같은 B-norm에서의 최소 risk rate p_R-sqrt(q)||d_free||B.
- Native metric norm identity와 실제 physical step norm.
- Covariance/Frobenius 실제 risk, 일차항, 이차항, FP32 materialization residual.
- EP-CBF의 cap headroom, 활성 분기, 최소 slack, free 방향 재배치량.

q/q_all은 risk gradient 중 보호 공간과 독립적으로 움직일 수 있는 비중을 설명한다. 분모가 0이면 undefined로 남긴다. 이 값이 작거나 free 성분이 작다고 해당 state를 제외하지 않는다.

기능적 결과는 Current PS/NS/NLL, Fixed/Past RS/PS/NS와 loss/recovery다. Risk 감소와 기존 edit retention은 같은 지표가 아니다. 특히 current edit 진척 보호는 past edit 진척 보호를 뜻하지 않으므로 Fixed/Past 변화를 반드시 함께 본다.

**14. 구현은 작은 projection과 공통 gradient accumulation으로 구성한다.**

기존 `group_edit_gradients()`는 edit NLL만 계산하므로 기반으로 사용할 수 있다. 다만 기존 `evaluate()` 뒤에 그대로 호출하면 rewrite forward/backward가 중복된다.

새 공통 accumulation에서는 한 state X를 고정한 채 다음처럼 계산한다.

1. 고정된 네 group을 순회해 각 group의 edit gradient와 request별 loss를 모은다.
2. 네 gradient의 평균으로 G_edit를 만든다.
3. 기존 KL과 전체 batch action gradient를 더해 G_total을 만든다.
4. 같은 G_total로 nominal d0를 계산하고 J4/J1/controller만 달리 적용한다.

모든 arm에 같은 group 순서·microbatch·reduction을 사용한다. Group 간에 X를 업데이트하지 않는다. Current group gradients는 KL이나 action과 섞기 전에 저장한다.

25개씩 네 group, microbatch2이면 group별 singleton 때문에 rewrite forward/backward가 312회, KL이 50회로 full logical gradient당 약 362회가 될 수 있다. 이전의 350회보다 조금 늘지만 group이 네 개라고 모델 계산이 네 배가 되지는 않는다.

J의 네 FP32 coefficient gradients는 약 6.25 MiB다. Dense weight gradient는 group별로 pullback한 뒤 재사용할 수 있다. H_N factorization은 batch 동안 고정하고, S의 4×4 system과 현재 risk projection만 갱신한다.

실제 projection은 B-whitened 좌표에서 구현하는 편이 안정적이다. Vector 표기로

\[
y=B^{1/2}d_0,\quad F=JB^{-1/2},\quad \tilde a=B^{-1/2}a.
\]

F의 row-space projector를 Pi라고 하면

\[
y_f=(I-\Pi)y,\quad a_f=(I-\Pi)\tilde a.
\]

a_f가 0이 아니면 n_f=a_f/||a_f||에 대해

\[
y_{\rm EP}=y-[n_f^\top y_f]_+n_f.
\]

이렇게 계산하면 큰 1/q multiplier를 직접 만들 필요가 없다. q도 큰 두 수의 차이 대신 ||a_f||²로 계산할 수 있다. Actual 100×100 factorization과 row-wise triangular solve로 whitening을 구현하며 거대한 square matrix를 만들지 않는다.

Rank-deficient J는 FP64의 작은 rank-revealing decomposition으로 처리한다. 수치 rank 기준은 구현 시작 때 고정하고 singular values와 residual을 남긴다. Loss 결과에 맞춰 ridge 계수를 조정하지 않는다. 기존 `soft_filter(g,J,gamma=.1)`은 exact null projection이 아니므로 그대로 사용하지 않는다.

Native K/U/P/M/C0와 risk reduced matrices는 inner loop 동안 고정한다. Current output gradient/J/risk gradient만 실제 state에서 갱신한다. Teacher는 We이며 actual forward anchor는 WN이다. Native wrapper를 매 step 재호출하지 않는다.

History는 현재 batch가 끝날 때 한 번만 반영한다. Prepared MN을 최종 state로 쓰면 이미 KK.T가 반영되어 있으므로 다시 append하지 않는다. 같은 native chain의 cached z를 미래의 서로 다른 continuation state에 그대로 쓰지 않는다.

**15. 계산 예산은 120 logical steps이며, group observer 때문에 중복 모델 pass를 만들지 않는다.**

| 항목 | 기본 설계 |
| --- | ---: |
| Trajectories | 15 |
| Logical optimizer steps | 120 |
| G0 공유 후 실제 full-gradient 계산 | 3+15×7=108 |
| Terminal objective, forward-only | 15 |
| Matched-removal probe objective, forward-only | 3 |
| Same-state nominal shadow, edit-only | 6 |
| Trajectory panel 평가 | 108,000 prompt pairs |
| Matched-removal curve probe | 3,300 prompt pairs |
| 총 panel 평가 | 111,300 pairs / 222,600 candidate sequences |
| N 재사용 후 새 생성 | 180 sequences |

G0 공유는 같은 objective와 계산 순서일 때 성립한다. J1은 J4의 평균이고 risk/controller는 nominal objective를 바꾸지 않으므로 초기 gradient를 공유할 수 있다.

위 362회 경로를 그대로 사용하면 nominal model backward는 39,096회다. Terminal/probe의 full-objective forward와 edit-only shadow까지 포함한 training forward 예상은 47,484회다. Teacher, panel/generation, small penalty backward, C0 준비, matrix solve는 따로 센다.

기존 ABC의 시간에 대응시킨 계획 예산은 준비 환경에 따라 **대략 3–5 GPU시간**이다. Group accumulation을 융합한 구현을 전제한다. 기존 함수를 중복 호출하는 구현을 택하면 실제 비용을 더 크게 보고해야 한다. Cold compute-z와 장기 continuation은 이 예산에 포함하지 않는다.

비용 축소가 필요하면 먼저 Middle의 alpha 감도 두 개와 J1 비교를 제외한다. 주 세 방법의 세 entry 비교와 same-state 검증은 유지한다. 성능이 나쁜 arm만 제거하지 않는다.

**16. 결과 해석은 gate 대신 관측 패턴으로 정리한다.**

| 관측 | 해석 |
| --- | --- |
| RiskOnly가 NLL 진척을 줄이고 EP-Free는 같은-state NLL을 더 잘 유지 | Progress-null 보정의 의도와 일치하는 기능적 신호 |
| Matched-removal에서 일차 risk 제거량은 같은데 EP-Free의 algebraic edit leakage가 작음 | 일차 보호 제약의 구현 확인; 기능적 효과는 실제 NLL/PS 차이로 별도 평가 |
| Algebraic leakage는 작지만 실제 NLL 차이가 큼 | Finite-step curvature 또는 observer 한계 |
| Group 평균은 유지되지만 특정 요청/PS가 악화 | Group-level progress가 충분한 functional observer인지 검토 필요 |
| d_free 또는 k_R가 거의 없음 | 현재 nominal에 제거할 harmful free 성분이 적음 |
| k_R를 제거했지만 total risk는 증가 | p_R와 quadratic remainder를 함께 확인; 제거 실패와 구분 |
| COV risk는 줄지만 NS가 개선되지 않음 | Structural surrogate와 실제 locality 사이의 간극 |
| EP-CBF는 더 개선되지만 EP-Free는 변화가 작음 | 단순 삭제보다 새 free 방향 재배치가 필요한 가능성 |
| EP-CBF의 slack이 큼 | 같은 진척과 같은 step 비용에서 cap 달성이 어려움 |
| Current는 유지되지만 Past가 악화 | Current 진척 보호와 historical retention의 차이 |

J residual, risk 증가, finite-step NLL 증가를 성능 통과 조건으로 삼지 않는다. 잘못된 input, update 미적용, NaN/Inf 등 계산 자체가 성립하지 않는 문제만 구현/실행 오류로 구분한다. Negative 결과를 지우거나 cap/group 수를 실행 중 자동 변경하지 않는다.

Current/Fixed/Past의 request 단위 paired 비교를 사용한다. 2000회 bootstrap에서 같은 request의 모든 방법을 함께 묶는다. 세 entry는 history와 현재 batch가 함께 다르므로 age-only 인과효과가 아니며, 반복된 Fixed100을 독립 300 requests로 합산하지 않는다.

Main alpha=.01을 주표로 유지한다. Middle 감도 중 NS가 가장 높은 점으로 주 결과를 바꾸지 않는다. NLL–PS–NS와 실제 step 비용을 함께 보고 동일 strength 여부를 해석한다.

**17. 산출물은 수학적 제거와 실제 기능적 효과를 연결해야 한다.**

- `run-index.csv`: entry/method/risk/observer/alpha/source/state/완료 상태.
- `trajectory.csv`: pre/post index, loss 구성, request/group actual NLL, actual step와 비용.
- `progress-decomposition.csv`: Jd0/Jd/leakage, progress/free energy, singular values와 projection residual.
- `damage-components.csv`: p_R/k_R/q/q_all, 제거량, actual risk와 일차/이차항.
- `same-state-probes.csv`: 공통 WN의 nominal/protected/unprotected 단발 endpoint는 train+curve 비교, pre3/pre7 nominal shadows는 edit-only 비교.
- `barrier-companion.csv`: cap, risk floor, 활성 분기, slack, free 방향 재배치.
- `request-metrics.csv` 및 `paired-summary.csv`: 기존 평가 identity, NLL/margin/RS/PS/NS, loss/recovery와 구간.
- `compute-summary.csv`: 공유 gradient, group accumulation, shadow/probe/evaluator/generation, small solve 비용.
- `diagnostic-discussion-ko.md`: 관측 사실, 일차 모델의 설명 범위, 다음에 분리할 질문 한 개.

결과가 유망하면 다음은 nonlinear retraction 또는 짧은 실제 sequential continuation 중 관측이 요구하는 하나를 선택한다. 이번에 보호한 네 group의 일차 진척을 정확한 request-level functional 보존이나 10k lifelong 안정성으로 확대하지 않는다.

**근거와 검산**

- [직전 native-metric risk-only barrier 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-native-metric-barrier-refinement-diagnostic-design.md)
- [ABC 최종 진단 보고서](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/final/final-diagnostic-report-ko.md)
- [사용자 draft의 progress-null 수식](/mnt/raid5/janghj/.codex/attachments/01f6c47b-2ca3-407a-a9c0-aadb34110ed2/pasted-text.txt:744)
- [기존 current edit-only group gradient 구현](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/objective.py:116)
- [기존 grouping/soft filter](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/directions.py:6)

합성 SPD metric의 검산에서 progress equality residual은 약 2.2e-16, 제거 후 free-risk contribution은 약 5.6e-17, norm identity residual은 수치상 0이었다. 같은 일차 risk 제거량의 unprotected 보정에서는 edit leakage가 남는 예도 확인했다. Bounded EP-CBF의 inactive/plane/ball/불가능한 cap 분기도 작은 예제로 확인했다. 이는 수식 검산이며 실제 LLM 성능 또는 실제 J의 수치 condition을 측정한 결과가 아니다.
