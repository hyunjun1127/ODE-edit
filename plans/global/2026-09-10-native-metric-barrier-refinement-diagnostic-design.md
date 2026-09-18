**Native metric과 cumulative-risk barrier를 결합한 closed-form refinement 진단 설계**

작성: 2026-09-10 KST. 상태: 실행 가능한 연구 설계 초안. 이번 작성 작업에서는 모델/GPU 실험을 실행하지 않았다. 작은 합성 행렬에서 아래 closed-form의 KKT 조건과 quadratic 전개만 확인했다.

후속 사용자 논의에 따라 주 방법은 [편집 진척 보존 공간에서의 손상 성분 제거 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-edit-progress-preserving-damage-removal-design.md)로 갱신했다. 이 문서는 risk-only correction의 비교 배경으로 남긴다.

사용자 요구는 native endpoint, 동일 basis, 동일 actual-output loss에서 native metric을 복원하는 효과를 확인하되, barrier 적용을 처음부터 고려하는 것이다. 성능 통과에 따라 다음 실험을 잠그는 gate는 만들지 않는다. 정해진 비교를 실행하고 작은 차이, 성능 하락, barrier 비활성, slack과 risk 증가도 결과로 남긴다.

**1. 이번 질문은 native metric과 barrier의 개별 효과 및 결합 효과다.**

한 layer에 native write를 적용한 같은 실제 모델에서 출발한다. Actual-output gradient를 매 step 다시 계산하고, 방향을 고르는 작은 quadratic 문제에 native metric과 누적위험 barrier를 넣는다.

| 방법 | 시작점 / 수정 공간 / 기본 loss | 방향 metric | 누적 risk barrier |
| --- | --- | --- | --- |
| N | 기존 native endpoint | 기존 native write | refinement 없음 |
| E | 같은 WN / 같은 Ub / 같은 L | Identity | 없음 |
| H | 같은 WN / 같은 Ub / 같은 L | Native quadratic metric | 없음 |
| E+B | E와 동일 | Identity | 단일 elastic barrier |
| H+B | H와 동일 | Native quadratic metric | 동일한 단일 elastic barrier |

H+B가 주 방법 후보다. N은 추가 refinement의 실용적 가치를 보는 기준이고, E/H/E+B/H+B가 기전을 분리하는 2×2 비교다. H+B만 E와 비교하면 metric과 barrier의 효과를 분리할 수 없다.

이번의 closed-form은 최종 nonlinear output loss의 해가 아니라, 매번 현재 gradient를 받아 계산하는 **local quadratic step의 해**다. Barrier를 포함한 단일 half-space 문제도 closed-form으로 계산한다. Metric과 barrier가 모두 들어가므로 H+B의 보정을 Euclidean projection으로 대체하지 않는다.

기존 ABC의 current-group J soft filter는 이번 barrier와 다른 연산이다. 이번 기본 실험에는 추가하지 않는다. Native target의 exact closure, request별 NLL ceiling, terminal progress schedule도 추가하지 않는다. Native가 이미 편집한 상태에서 시작하며, 현재 편집 품질과 과거 편집 보존은 공통 loss 및 실제 평가로 관측한다.

**2. 출발점과 데이터는 완료된 ABC 진단을 그대로 이어받는다.**

모델은 Llama3-8B-Instruct, 수정 대상은 `model.layers.4.mlp.down_proj.weight` 하나다. Shape는 output 4096 × input 14336이다.

| Entry | 편집 전 상태 | 현재 batch | Current ordinal, 0-based | 역할 |
| --- | --- | --- | --- | --- |
| Early | W10, M10 | B11, 100 requests | 1000–1099 | 초기 상태의 주 비교 |
| Middle | W50, M50 | B51, 100 requests | 5000–5099 | 주 비교와 amplitude 감도 |
| Late | W90, M90 | B91, 100 requests | 9000–9099 | 후기 상태의 주 비교 |

기호는 다음처럼 고정한다.

- W0: 편집 이전 원래 모델의 L4 weight. Cumulative risk의 reference다.
- We: 이번 batch가 들어오기 직전 history checkpoint. KL teacher와 batch action의 reference다.
- WN: We에서 기존 native writer를 한 번 적용한 실제 FP32 endpoint.
- DeltaN = WN − We: native의 전체 batch update.
- M: 현재 batch를 넣기 전의 history statistic.
- K: 기존 native write가 실제 사용한 request별 key 100개, shape [14336,100].
- U = Ub: 기존 native input basis의 orthonormal 표현, shape [14336,100], 실제 rank 100.

Full-Q rank 14326 공간은 사용하지 않는다. 이 진단은 native rank-100 공간 안에서 방향 계산과 barrier 효과를 본다. 공간 밖의 누적 손상을 이 방법으로 모두 고칠 수 있다고 가정하지 않는다.

Native K는 600 context를 모두 열로 둔 행렬이 아니다. 기존 source는 context-type 안에서 평균하고 type 사이를 평균한다. 현재 clean 1개와 generated 5개 recipe에서는 clean key 가중치가 1/2, generated 각각은 1/10이다. Actual-output NLL의 6-context 평균은 기존처럼 flat mean이다. 두 평균 규칙을 새로 통일하면 loss 또는 native metric이 달라지므로 이번에는 그대로 둔다.

**3. Physical warm start는 정확히 WN으로 유지한다.**

Refinement coefficient X를 사용한다.

\[
W(X)=W_N+XU^\top,\qquad X_0=0.
\]

X는 [4096,100]이다. `We + (DeltaN @ U) @ U.T`로 native update를 재구성해서 시작하지 않는다. 그렇게 하면 projection/rounding 차이가 시작점에 들어간다. 실제 저장된 WN을 anchor로 사용한다.

보존 penalty가 보는 전체 batch update는

\[
\Delta(X)=\Delta_N+XU^\top=W(X)-W_e
\]

다. X=0이라고 보존 penalty가 0이 되는 것이 아니다. Native initialization도 전체 batch update에 포함된다.

**4. 네 refinement 방법의 기본 objective와 gradient는 같다.**

\[
L(X)=L_E(W(X))+0.0625L_{\rm essence}(W(X);W_e)
+0.1\frac{J_N(\Delta(X))}{j_N},
\]

\[
J_N(\Delta)=\operatorname{tr}\{\Delta(M+L2I)\Delta^\top\},
\qquad j_N=J_N(\Delta_N),\quad L2=1.
\]

L_E는 기존 Current100의 target-token 평균 NLL을 context별, request별로 동일 가중 평균한 값이다. KL teacher는 We에서 만들고, 기존 essence 입력·방향·reduction을 유지한다. 모든 token 위치에 실제 수정 weight가 적용된 model output에서 gradient를 계산한다. Subject-position activation injection으로 대체하지 않는다.

기존 reduced metric을 공유하여 다음 항을 한 번 준비한다.

\[
H_R=U^\top(M+L2I)U,
\qquad C_N=\Delta_N(M+L2I)U.
\]

그러면 orthonormal 표현에서

\[
J_N(\Delta_N+XU^\top)
=j_N+2\langle X,C_N\rangle+
\operatorname{tr}(XH_RX^\top),
\]

\[
G_s=\nabla_X L(X_s)
=G_{E,s}+0.0625G_{{\rm essence},s}
+\frac{0.2}{j_N}(C_N+X_sH_R).
\]

기존 구현은 U의 orthonormal성을 이용해 H_R의 L2 항을 I100으로 계산한다. 두 방법 모두 같은 기존 reduction을 사용한다. 저장된 FP32 U의 Gram 오차는 기록하되 한 방법만 basis를 재직교화하거나 penalty를 달리 만들지 않는다. Physical action 측정과 reduced quadratic의 작은 수치 차이도 구분한다.

H 방향에서 M을 사용한다는 이유로 G_s의 action gradient를 빼지 않는다. Metric은 이동 경로를, penalty는 목적함수를 정한다. 이번에는 quadratic penalty의 implicit/proximal 처리까지 동시에 바꾸지 않는다.

**5. Native metric은 current key와 이전 history를 합친 100×100 행렬이다.**

\[
Z=U^\top K,
\qquad H_N=ZZ^\top+H_R.
\]

H_R는 공통 loss의 보존 비용이고, H_N은 방향 계산에 사용하는 native quadratic metric이다. H_N은 actual-output NLL의 정확한 Hessian이나 Fisher matrix가 아니다.

기존 native solve의 `P(KK.T+M)+I`는 일반적으로 비대칭이다. 이를 그대로 SPD metric이라고 부르거나 임의로 대칭화해 사용하지 않는다. 허용 basis 안의 위 H_N을 구성한다.

Prepared asset의 `MN`에는 current KK.T가 이미 한 번 추가되어 있다. 따라서 H_N을 `MN + KK.T`로 만들면 현재 batch가 이중 계산된다. 공통 loss의 M을 MN으로 바꾸는 것도 이번 동일-loss 조건을 깨뜨린다.

H_N은 entry마다 한 번 factorization한다. 이후 inverse를 명시적으로 만들지 않고 small solve를 사용한다. H_N의 구성·factorization은 FP64를 권장하고, actual model과 coefficient 적용은 기존 FP32 정책을 유지한다. 작은 행렬의 대칭 residual과 solve residual을 기록한다. I 항이 있는데 factorization에 문제가 생기면 입력·계산 정밀도를 먼저 확인하며, 결과를 보며 damping 계수를 탐색하지 않는다.

**6. Nominal step은 첫 이동량만 맞춘 후 고정 learning rate로 계산한다.**

Momentum=0, optimizer-state carryover 없음, decoupled weight decay 없음으로 둔다. 이전 Direct-B의 momentum이나 We에서 보정한 learning rate를 가져오지 않는다.

\[
T_E=I,\qquad T_H=H_N,
\qquad d^{\rm nom}_{m,s}=-\eta_mG_sT_m^{-1}.
\]

주 amplitude는 alpha=0.01이다. 해당 entry의 native update를 길이 단위로 사용한다.

\[
b_\alpha=\alpha\|\Delta_N\|_F,
\]

\[
\eta_E=\frac{b_\alpha}{\|G_0U^\top\|_F},
\qquad
\eta_H=\frac{b_\alpha}{\|(G_0H_N^{-1})U^\top\|_F}.
\]

U가 정확히 orthonormal이면 분모는 각각 coefficient Frobenius norm과 같다. 실제 구현에서는 작은 Gram U.T@U로 physical norm을 계산할 수 있다. 최종 FP32 weight에 적용된 첫 norm도 별도로 기록한다.

E+B는 E의 eta, H+B는 H의 eta를 그대로 사용한다. **Barrier 전 nominal 첫 step을 맞추며, barrier 후 실제 step을 재정규화하지 않는다.** 이후 step마다 gradient를 normalize하거나 path length를 강제로 맞추지 않는다.

이 비교는 초기 nominal 이동량을 맞춘 비교다. 이후 NLL, actual path length, endpoint norm, native action까지 동일하다는 뜻은 아니다. U가 정확히 orthonormal인 이상화에서는 같은 Frobenius 첫 step에서 E가 일차 loss 감소를 최대화한다. 실제 저장 U의 작은 Gram 오차는 앞서 정한 방식으로 기록한다. H의 목표를 무조건 더 큰 최초 loss 감소로 정하지 않는다.

**7. Barrier는 W0 기준 누적 Frobenius risk 한 개로 시작한다.**

\[
R_F(W)=\tfrac12\|W-W_0\|_F^2,
\qquad s_R=\tfrac12\|\Delta_N\|_F^2.
\]

보고와 slack 단위가 entry마다 지나치게 달라지지 않도록 다음 normalized risk를 쓴다.

\[
r(X)=\frac{R_F(W(X))-R_F(W_N)}{s_R}.
\]

이는 값의 offset과 단위만 바꾼 것이다. Risk의 중심은 계속 W0이며 WN으로 바뀌지 않는다. X=0에서 r=0이다.

주 budget은 beta=0으로 고정한다.

\[
h_s=\beta-r(X_s),\qquad \beta=0.
\]

물리적으로는 R_F(WN)을 refinement 동안의 cap으로 삼는다. 질문은 **native endpoint 이후의 refinement가 누적 변형을 더 키우는 것을 제어하면 유리한가**다. Native가 이미 만든 손상을 전부 복구하라고 요구하지 않는다. 또한 매 batch마다 이 cap을 새로 만들면 lifelong 전체에 하나의 고정 global budget을 준 것이 아니다.

Budget은 inner step마다 현재 risk로 reset하지 않는다. 성능이나 barrier 활성 빈도를 보고 실행 도중 cap을 자동 조정하지 않는다.

Frobenius는 이번 metric/barrier 결합을 명확히 확인하기 위한 첫 structural surrogate다. ABC에서는 operator/covariance 방향의 NS 효과가 더 컸으므로 Frobenius를 최종적으로 최선인 risk라고 주장하지 않는다. 여러 risk, replay, current-group 보호를 본실험에 동시에 추가하지 않는다.

**8. Barrier gradient와 finite-step 오차는 모델 backward 없이 계산한다.**

\[
D_N=W_N-W_0,\qquad B_U=U^\top U,
\]

\[
r(X)=\frac{\langle D_NU,X\rangle+
\tfrac12\operatorname{tr}(XB_UX^\top)}{s_R},
\]

\[
a_s=\nabla_Xr(X_s)=\frac{D_NU+X_sB_U}{s_R}.
\]

매 step 현재 X_s에서 a_s와 h_s를 갱신한다. 이 갱신에는 추가 request-gradient나 model forward가 필요 없다. Geometry K/U/M/H_N은 고정하고, 실제 output gradient와 risk state는 갱신한다.

추가 모델 실행 없이 이 basis에서의 geometric risk floor도 한 번 계산한다.

\[
\beta_{\min}=-\frac{\operatorname{tr}\{(D_NU)B_U^{-1}(D_NU)^\top\}}{2s_R}.
\]

이는 편집 loss를 무시하고 허용된 X 전체에서 얻을 수 있는 최저 r이다. Global displacement 중 이 basis에서 움직일 수 있는 부분이 얼마나 남아 있는지 보여준다. 뒤의 별도 restoration budget이 이 값보다 낮다면 그 support에서는 달성 불가능하다는 사실을 slack과 함께 해석한다. 실행을 막는 gate로 사용하지 않는다.

적용할 coefficient displacement d에 대해 선형화한 barrier 조건은

\[
\langle a_s,d\rangle\le\gamma h_s+\xi,
\qquad\gamma=0.25,\quad\xi\ge0.
\]

여기서 d는 learning rate를 이미 곱한 **한 iteration의 displacement**다. 우변에 eta를 다시 곱하지 않는다. Gamma는 iteration당 headroom 사용/초과분 복구 비율이다. 실제 초 단위 ODE 시간상수로 해석하지 않는다.

이는 다음 finite-step 조건을 일차화한 형태다.

\[
r_{s+1}\le(1-\gamma)r_s+\gamma\beta+\xi.
\]

실제 quadratic risk에는 양의 이차항이 남는다.

\[
r(X_s+d)-r(X_s)
=\langle a_s,d\rangle+
\frac{\|dU^\top\|_F^2}{2s_R}.
\]

따라서 일차 조건을 만족해도 실제 cap을 넘을 수 있다. 이 실험은 slack을 허용하는 선형화된 CBF 진단이며 hard finite-step safety certificate가 아니다. 실제 FP32 적용 weight에서 계산한 risk, 일차 예측, 이차항, residual을 함께 남긴다. 위반이 관측될 때마다 자동 backtracking이나 추가 trial forward를 붙이지 않는다.

**9. Barrier를 포함한 방향도 closed-form으로 계산한다.**

Metric과 learning rate를 결합한 effective stiffness를 정의한다.

\[
\mathcal B_m=T_m/\eta_m,
\qquad d^{\rm nom}_{m,s}=-G_s\mathcal B_m^{-1}.
\]

Barrier-on 방법은 매 step 다음 문제를 푼다.

\[
\begin{aligned}
\min_{d,\xi\ge0}\;&
\frac12\operatorname{tr}\{(d-d^{\rm nom})\mathcal B_m(d-d^{\rm nom})^\top\}
+\frac{\xi^2}{2\varepsilon}\\
\text{s.t.}\;&\langle a_s,d\rangle\le\gamma h_s+\xi.
\end{aligned}
\]

Raw T_m만 사용하지 않고 T_m/eta_m를 사용하는 이유는 metric의 임의 scalar scale이 barrier의 slack trade-off를 바꾸지 않게 하기 위해서다. H_N을 c배 하고 초기 calibration으로 eta_H가 c배 되면 effective stiffness와 barrier 해가 모두 그대로다.

\[
z_{m,s}=a_s\mathcal B_m^{-1},\qquad
q_{m,s}=\langle a_s,z_{m,s}\rangle,
\]

\[
v_{m,s}=\langle a_s,d^{\rm nom}_{m,s}\rangle-\gamma h_s,
\qquad
\lambda_{m,s}=\frac{[v_{m,s}]_+}{q_{m,s}+\varepsilon}.
\]

해는

\[
\boxed{
d_{m,s}=d^{\rm nom}_{m,s}-\lambda_{m,s}z_{m,s},
\qquad\xi_{m,s}=\varepsilon\lambda_{m,s}.
}
\]

그리고 X_{s+1}=X_s+d_{m,s}를 실제 weight에 적용한다. 비활성 상태에서는 nominal step과 정확히 같은 수식이다. Active 상태에서는 선택한 metric에서 가장 작은 수정과 slack 비용을 절충한다.

단일 scalar barrier이므로 고차원 generic QP나 수십 번의 inner optimization이 필요 없다. Existing factorization을 이용한 small solve, inner product, coefficient correction으로 계산한다. Actual-output gradient 계산은 여전히 필요하다.

**Elasticity는 E+B/H+B가 공유하는 한 번의 calibration으로 정한다.**

각 entry의 주 alpha=0.01 공통 시작점에서

\[
\bar q_0=(q_{E,0}+q_{H,0})/2,
\qquad\varepsilon=\bar q_0/4
\]

로 두고 해당 entry의 모든 trajectory 동안 고정한다. 두 barrier arm과 Middle의 추가 alpha들도 같은 epsilon을 사용한다. q가 qbar와 같으면 최초 선형 위반의 약 80%를 correction으로 줄이고 약 20%를 slack으로 남기는 설정이다. 이는 검증된 최적값이 아니라 첫 진단의 고정 설정이다.

Metric별 또는 매-step q에 맞춰 epsilon을 다시 정하지 않는다. 그렇게 하면 두 방법이 다른 slack objective를 풀게 된다. Middle alpha 감도에서도 primary에서 정한 epsilon을 유지하여 amplitude와 barrier penalty를 동시에 변경하지 않는다. Alpha가 달라지면 q와 실제 위반 교정 비율은 달라질 수 있으며 이를 결과로 기록한다.

a_s=0인 step에서는 z=q=0으로 barrier가 방향을 바꿀 수 없다. 그 경우 필요한 slack만 기록한다. 초기 qbar가 정확히 0인 퇴화 경우에는 공통 native-local gradient DeltaN U / s_R의 pooled q를 calibration 단위로 사용하고 이 fallback을 표시한다. Barrier normal a_s 자체를 이 gradient로 교체하지는 않는다. DeltaN=0이면 이동량과 risk 단위도 퇴화하므로 no-action 상태로 구분한다. G0=0이면 nominal 이동량 calibration이 불가능하므로 stationary branch로 표시하며, 강제로 같은 길이의 무작위 step을 만들지 않는다.

**10. 기본 실행은 16 trajectories, 128 logical steps다.**

| 구분 | 구성 | Trajectories | Steps |
| --- | --- | ---: | ---: |
| 주 비교 | Early/Middle/Late × E/H/E+B/H+B, alpha=0.01 | 12 | 96 |
| Barrier-on amplitude 감도 | Middle × E+B/H+B × alpha=0.003,0.03 | 4 | 32 |
| 합계 | N 세 endpoint는 재사용 | 16 | 128 |

모든 trajectory는 8 full-batch optimization steps다. Middle 감도는 barrier를 켠 두 방법에서 작은/큰 nominal step의 영향을 보는 것이다. Barrier off/on interaction은 네 방법이 모두 있는 alpha=0.01 주 비교에서 계산한다.

Learning-rate winner를 고르지 않는다. Alpha=0.01을 주표로 유지하고 .003/.03은 모두 감도로 공개한다. NS나 PS가 좋은 alpha로 주표를 교체하지 않는다. Gamma=0.25와 elasticity 설정도 이번에는 sweep하지 않는다.

예산을 줄일 때는 Middle amplitude 감도 네 개를 제외한 core12를 유지한다. 이 축소는 계산 예산에 따른 선택이며 특정 방법의 성능에 따른 제외가 아니다.

**11. 평가 panel과 시점은 기존 ABC를 재사용한다.**

Controller는 Current100, 기존 contexts와 essence inputs, native K/U/P, M, W0만 읽는다. Held-out paraphrase와 neighborhood는 gradient·budget·alpha 설정에 쓰지 않는다. 추가 historical functional replay는 넣지 않는다.

| Panel | 구성 | Full 분모 |
| --- | --- | --- |
| Current100 | 현재 batch 전체 | RS100 / PS200 / NS1000 |
| Fixed100 | 이전 B1의 동일 100 requests | RS100 / PS200 / NS1000 |
| Past100 | 기존 ABC의 age-stratified 과거 100 requests | RS100 / PS200 / NS1000 |

기존 panel IDs, candidate tokenization, tie 정책, metadata conflict strata를 그대로 재사용한다. RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이다. 이를 semantic free-generation accuracy라고 부르지 않는다.

| 상태 | 평가 |
| --- | --- |
| Step0 | 같은 WN의 기존 Full 결과 재사용 |
| Step1/2/4 | Curve1100쌍 |
| Step8 | Full3900쌍; curve rows는 여기서 추출 |
| Optimizer pre-state 0–7 | 공통 train loss 구성, 합산 gradient, step/metric/barrier 기록 |
| Terminal state8 | Loss 구성, risk, weight geometry를 forward-only로 기록; 추가 gradient 없음 |

Curve는 Current RS100/PS200, Fixed/Past RS 각100, 세 panel에서 각각 neighbor200쌍이다. Fixed/Past PS는 step8에서만 측정한다. 중간 PS를 보간하거나 측정한 것으로 표시하지 않는다.

Middle 주 비교 N/E/H/E+B/H+B에는 기존과 같은 20 rewrite + 40 rephrase 생성 panel을 추가한다. N의 기존 생성이 state와 decoding 설정까지 같으면 재사용하여 새 생성은 240 sequences다. 작은 literal-prefix 지표와 출력 원문을 보조로 제시한다.

**12. Metric, barrier 작동, 기능적 결과를 따로 측정한다.**

기능적 주 결과는 세 가지 비교다.

\[
Y(H)-Y(E),\quad Y(H+B)-Y(H),\quad Y(H+B)-Y(E+B).
\]

추가로 네 arm의 interaction을 계산한다.

\[
I_Y=[Y(H+B)-Y(H)]-[Y(E+B)-Y(E)].
\]

Y에는 Current PS/NLL/NS, Fixed/Past RS/PS/NS와 loss/recovery를 각각 넣는다. 단일 합성 점수로 합치지 않는다. N 대비 변화도 함께 보고해 refinement의 추가 비용과 효과를 확인한다.

Metric 진단에는 다음을 남긴다.

- H_N의 eigenvalue spread와 condition number.
- 같은 현재 상태의 G와 G H_N^-1 사이 cosine. 이후 두 trajectory 사이의 방향 차이와 구분한다.
- Eigenmode별 gradient energy와 실제 nominal step energy. Condition number가 커도 gradient가 한 구간에 몰리면 방향 변화가 작을 수 있다.
- Step별 q_K=||dZ||F², q_R=tr(d H_R d.T), 합 q_K+q_R.
- 전체 batch action의 변화. Increment cost와 native delta를 포함한 cross term을 구분한다.
- 실제 step norm, WN 이후 net norm, We 이후 전체 batch norm, path length.

작은 native quadratic step cost 자체는 metric이 만든 수학적 결과일 수 있다. 이를 locality 개선의 실증으로 대신하지 않는다.

Barrier 진단에는 다음을 남긴다.

- r_s, h_s, cap의 physical 값, a_s norm과 허용 basis 안으로 투영된 risk gradient 비중.
- Nominal predicted violation v_s, activation 여부, q_s, epsilon, lambda, slack xi.
- Nominal norm, correction norm, applied norm, correction/nominal 비율과 방향 cosine.
- 일차 risk 변화, 정확한 quadratic 이차항, 실제 FP32 risk 변화 및 cap 초과량.
- 수정이 risk 증가 방향을 얼마나 줄였는지와 Current/Past edit loss를 얼마나 바꿨는지.

E/H의 barrier-off 상태에서도 a_s와 해당 nominal의 predicted violation은 algebra-only로 기록한다. Counterfactual barrier correction의 벡터를 계산할 수는 있지만, 그 weight를 forward하지 않았다면 기능적 결과를 측정한 것으로 쓰지 않는다.

모든 arm은 actual-output gradient를 갱신한다. Barrier state를 새로 계산하는 비용은 작으므로 이번에는 frozen/refreshed risk를 다시 factorial로 나누지 않는다. 이 실험만으로 매-step risk refresh의 독립적 우월성을 주장하지 않는다.

**13. Strength matching과 통계는 결과 해석에 사용한다.**

첫 nominal step을 맞췄다고 동일 editing strength가 되는 것은 아니다. 모든 관측 snapshot에서 train NLL, canonical NLL, Current PS와 NS의 관계를 표시한다. 같은 RS100%라도 NLL과 PS가 다를 수 있다.

추천 그림은 네 개다.

1. Current train NLL ↔ Current/Past NS, actual snapshot만 표시.
2. Current PS ↔ Fixed/Past edit loss/recovery, full endpoints만 표시.
3. 누적 risk 변화 ↔ NS 변화, 색으로 방법과 entry 구분.
4. Step별 nominal/correction norm, slack, 실제 cap 초과량.

Common strength 범위가 겹치면 관측된 지점의 차이를 비교한다. 겹치지 않아도 방법을 제외하지 않는다. 관측하지 않은 중간 weight의 NS를 보간값으로 대신하지 않는다.

Request 단위로 rewrite/paraphrase/neighborhood와 네 arm의 값을 함께 묶어 2000회 paired bootstrap한다. Interaction의 CI도 네 arm을 함께 resample해서 직접 계산한다. 서로 다른 arm의 CI를 단순히 빼지 않는다.

세 entry는 history와 현재 batch가 함께 다르므로 age-only 효과가 아니다. 동일 Fixed100이 세 entry에 반복된 것을 독립 300 requests로 취급하지 않는다. Bootstrap은 이번 panel의 불확실성이며 optimizer seed나 edit order 모집단의 불확실성을 추정하지 않는다. 기존 개발 panel을 새 held-out benchmark라고 부르지 않는다.

**14. 구현은 기존 actual-output 경로와 준비 자산을 재사용한다.**

재사용 대상은 기존 ABC worktree의 prepared.pt, W0/We/WN/DeltaN/K/Ub/M/MN, contexts, native target provenance, panel 목록이다. 실제 tensor를 재사용할 때 object identity와 metadata를 한 번 확인하면 된다. Source receipt와 달리 실제 reload가 안 되는 경우는 구현/입력 문제로 구분한다.

변경할 책임 범위는 다음 정도다.

| 부분 | 필요한 작업 |
| --- | --- |
| Common preparation | 기존 later_stages의 WN anchor + cross-term penalty를 Q 대신 Ub에 적용 |
| Metric preparation | H_R/Z/H_N/Gram, H_N factorization, risk 초기 cross term 준비 |
| Optimizer step | 공통 gradient, 초기 eta, E/H nominal, elastic scalar-barrier closed-form |
| Evaluation/recording | 기존 evaluator 재사용, barrier 및 metric columns 추가 |

Teacher 생성 전에 live model은 We여야 하고, 실제 optimization forward에는 WN+XU.T를 전달한다. Teacher를 WN으로 만들면 이번 공통 loss가 바뀐다.

Native wrapper는 호출마다 We로 복원하고 cached z를 사용할 수 있으므로 inner loop에서 재호출하지 않는다. 새 z target, K, P, M을 step마다 다시 만들지 않는다. Single-layer output projection과 고정 teacher-forced inputs에서는 upstream K를 재사용할 수 있다.

M은 inner loop 동안 고정한다. 이번 독립 branch 진단에서 history를 step마다 append하지 않는다. 최종 continuation state를 저장한다면 M_next=M+KK.T가 한 번 반영된 값이어야 한다. Prepared MN을 쓰면 이미 그 한 번이 반영되어 있으므로 다시 더하지 않는다.

모든 step의 X와 scalar 기록을 저장하고, 0/1/2/4/8 state를 확실히 재구성할 수 있게 한다. U/WN의 원본 reference와 적용 dtype/순서를 함께 저장하면 모든 dense weight를 복제할 필요는 없다. 평가에는 실제 FP32로 materialize한 weight를 사용한다.

**15. 계산 예산은 barrier 때문에 추가 request Jacobian이 생기지 않는다는 점을 반영한다.**

동일 entry에서 최초 G0는 모든 branch가 공유한다. Logical optimizer step과 실제 gradient 계산 횟수는 다르다.

| 항목 | Core12 | 감도 포함16 |
| --- | ---: | ---: |
| Logical optimizer steps | 96 | 128 |
| 실제 full-gradient 계산 | 3+12×7=87 | 3+16×7=115 |
| Terminal step8 train objective, forward only | 12 | 16 |
| 추가 Curve evaluations | 36 | 48 |
| 추가 Full evaluations | 12 | 16 |
| 새 평가 prompt pairs | 86,400 | 115,200 |
| New/true candidate sequences | 172,800 | 230,400 |

공통 G0 공유는 WN/teacher/U/objective/reduction 순서가 같은 경우에만 한다. Step1/2/4의 train loss는 다음 step gradient 계산에서 얻은 값을 재사용할 수 있다. Terminal step8의 post-update loss는 forward-only 평가를 별도로 센다.

기존과 동일한 microbatch 경로에서 full objective당 forward/backward 각각 350회가 유지되면, 16개 구성의 nominal backward는 40,250회이고 terminal 포함 training forward는 45,850회다. Teacher 준비, panel evaluation, 생성, load 비용은 별도다. 실제 실행에서 달라지면 측정된 count를 보고한다.

16개 구성의 조건부 계획 예산은 **약 2.5–4 GPU시간 + 환경에 따른 준비 비용**이다. Core12는 약 2–3시간으로 본다. 기존 ABC의 step/evaluation 비용에 근거한 추산이며 같은 GPU·kernel·microbatch 조건을 전제한다. 최초 준비와 몇 step에서 실측한 비용으로 남은 시간을 갱신한다.

이 예산은 기존 native z와 WN/U를 재사용하는 refinement 진단이다. Cold compute-z나 전체 온라인 editor 비용을 포함하지 않는다. Native warm start와 평가도 포함한 전체 속도 우위로 해석하지 않는다.

**16. 실행 순서는 성능 gate 없이 한 묶음으로 진행한다.**

1. Middle 공통 자산을 복원하고, teacher/actual weight/penalty reference/gradient가 의도한 모델을 나타내는지 한 번 확인한다.
2. 작은 행렬 계산에서 metric solve, barrier KKT, risk 전개를 확인하고 Middle 네 주 arm을 실행한다.
3. Early/Late의 동일 네 arm을 실행한다. Middle 성능을 통과 조건으로 삼지 않는다.
4. Middle barrier-on alpha 감도 네 개와 계획된 terminal/생성 평가를 마친다.
5. 전체 결과를 한 번 종합해 nominal metric 효과, barrier 효과, interaction과 비용을 논의한다.

성능 하락, 작은 효과, barrier 비활성, positive slack, cap 초과, NLL 비단조성은 모두 과학적 관측이다. NaN/Inf·잘못된 입력·update 미적용·자원 실패처럼 계산이 성립하지 않는 문제는 고치거나 branch 상태로 기록한다. 실패한 branch를 결과 목록에서 없애지 않는다.

**17. 관측 패턴에 따라 다음 질문을 정한다.**

| 관측 | 이번 결과의 해석 | 이후 분리할 질문 |
| --- | --- | --- |
| H가 E보다 좋은 strength–locality 조합 | Native metric의 finite-step 방향 선택에 가치가 있음 | 짧은 실제 sequential continuation에서도 유지되는가? |
| H+B가 H보다 유리하고 E+B보다도 유리 | Barrier와 native metric의 결합에 추가 가치가 있음 | 같은 risk의 soft penalty로 설명되는가? |
| E+B와 H+B가 모두 비슷하게 개선 | Barrier의 기여가 크고 metric의 추가 기여는 작을 수 있음 | 다른 상태에서도 이 패턴이 반복되는가? |
| Risk는 줄지만 Current/Past edit 품질도 하락 | Risk 제어와 편집 강도 사이 trade-off | Risk surrogate 또는 제한된 support가 적절한가? |
| Barrier가 거의 비활성 | 현재 선형 조건에서 correction 요청이 적었음; 실제 risk/cap 초과는 이차항과 함께 별도 확인 | 선형화 오차, 더 긴 구간 또는 명시적인 복구 목표 중 무엇이 중요한가? |
| Slack이 크고 correction이 작음 | 편집 방향과 risk 조건의 충돌을 elasticity가 흡수함 | 더 강한 제어가 유용한가, 편집 비용만 커지는가? |
| 일차 조건은 맞지만 실제 cap 초과가 큼 | Finite-step quadratic 항을 무시하기 어려움 | 정확한 quadratic 제약이나 discrete correction이 필요한가? |
| Condition number는 크지만 방향 cosine이 거의 1 | 실제 gradient가 쓰는 공간에서는 metric 효과가 작음 | 현재 basis 안에 유용한 방향 차이가 충분한가? |

주 cap에서 barrier가 비활성이었다고 실행 중 budget을 낮추지 않는다. 별도 복구 질문을 시험한다면 예를 들어 Middle에서 beta=-0.1을 둔 E+B/H+B 한 쌍을 명시적으로 추가할 수 있다. 이는 누적 risk 전체의 10% 감소가 아니라 **native update 자체의 Frobenius energy s_R의 10%만큼 RN 아래로 내려가는 목표**다. 주 실험을 대체하지 않고 이번 기본 16개에도 포함하지 않는다.

본실험이 H+B의 유용성을 보여도 hard barrier의 필요성이나 lifelong 안정성까지 증명되지는 않는다. 이어서 같은 risk의 soft-penalty companion 또는 5-batch continuation 중 결과가 요구하는 질문 하나를 고른다. 연속 실험에서는 각 방법이 자기 endpoint/history를 다음 batch로 넘기고 그 상태에서 native initialization을 다시 계산한다. 공통 native chain으로 매번 돌아오지 않는다.

**18. 산출물은 실행 상태와 해석에 필요한 정보로 제한한다.**

- `run-index.csv`: entry, method, alpha, gamma, epsilon, risk cap, initialization/support/objective reference, state path, 완료/실행 오류.
- `trajectory.csv`: state index, pre/post 구분, loss 구성, gradient norm, nominal/applied/path norm, 전체 action, 실제 누적 risk, 시간.
- `metric-probes.csv`: spectrum, gradient-mode energy, 방향 cosine, qK/qR, small-solve residual.
- `barrier-probes.csv`: h/a/v/q/lambda/slack, predicted/actual risk, quadratic 항, activation, correction norm.
- `request-metrics.csv`: 기존 identity와 candidate NLL/margin/성공 판정.
- `paired-summary.csv`: N 대비, E/H 및 off/on 대비, interaction, request-cluster 구간, loss/recovery.
- `compute-summary.csv`: cached native 준비, physical gradient/evaluation/생성, small solve와 barrier 계산 비용.
- `diagnostic-discussion-ko.md`: 실제 관측, 가능한 설명, 미분리 요인과 다음 질문 한 개.

별도 scientific promotion flag나 다단계 승인표를 기본 산출물로 요구하지 않는다.

**근거와 설계 검산**

- [완료된 ABC 최종 진단 보고서](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/final/final-diagnostic-report-ko.md)
- [앞선 ABC 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-single-layer-cumulative-risk-diagnostic-design.md)
- [기존 native 준비와 reduced metric](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/runtime.py:68)
- [기존 actual-output objective](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/objective.py:70)
- [기존 warm-start cross-term 구성](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/later_stages.py:32)
- [기존 finite-change 측정](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/directions.py:57)
- [Ames et al., Control Barrier Function Based Quadratic Programs for Safety Critical Systems](https://arxiv.org/abs/1609.06408): CBF를 QP에 결합하는 일반적 출발점이다. 본 문서의 slack·finite-step 진단에 원 논문의 forward-invariance 보장을 그대로 적용하지 않는다.

작은 합성 SPD 행렬에서 검산한 residual은 KKT stationarity 약 7.3e-16, active constraint 약 5.6e-17, quadratic risk identity 약 1.2e-15, warm-start penalty expansion 약 1.1e-13이었다. 이는 수식/방향 검산이며 실제 LLM gradient·성능·실제 H_N의 condition number를 측정한 결과가 아니다.
