# FzCB-Edit

## Fixed-z Conditional-Completion Barrier Editing

## 2026-08-31 Method Pivot Proposal

| 항목 | 내용 |
|---|---|
| 문서 상태 | 2026-08-31 current primary method proposal |
| 직전 proposal | 2026-08-30 Fixed-z Functional Safe Write |
| 적용 위치 | MEMIT/AlphaEdit의 \(W\)-write stage |
| 고정 대상 | stock editor가 한 번 계산한 direct target \(z^\star\)와 shared intervention \(\delta^\star\) |
| 변화 대상 | 동일 fixed-\(z\) progress를 만드는 admissible weight trajectory |
| progress authority | full-model activation homotopy equality |
| 유일한 barrier | spent action과 conditional suffix action의 합에 대한 completion budget |
| controller 비입력 | output KL, reference facts, locality prompts, prior-edit prompts |
| 1차 editor | MEMIT |
| 2차 editor | hard projected AlphaEdit |
| 방법명 | FzCB-Edit |
| ODE claim | refreshed continuation이 strong static solve보다 추가 가치를 보일 때만 유지 |

> **한 문장 방법 정의**
>
> FzCB-Edit은 stock MEMIT/AlphaEdit이 계산한 fixed activation target을 full-model
> homotopy equality로 추적하면서, 동일한 activation progress를 만드는 weight velocity
> 중 이미 소비한 native action과 남은 target을 완성하는 데 필요한 conditional action의
> 합이 초기 completion budget을 넘지 않는 방향을 선택한다.

---

# 0. Method pivot

2026-08-30 proposal은 동일 \(z^\star\)를 실현하는 weight freedom 중
output-distribution drift가 작은 endpoint를 functional anchor와 KL constraint로 고르는
방향이었다. 이후 q-KL screen은 controller가 q-KL을 낮추면서 edit target strength를 함께
약화할 수 있음을 보였다. 이는 output KL 자체가 무의미하다는 뜻이 아니라, fixed target
realization을 hard하게 유지하지 않은 functional objective가 writer의 구조적 자유도와 edit
strength를 분리하지 못했다는 뜻이다.

본 pivot은 다음을 결정한다.

1. \(z^\star\)와 그 homotopy progress는 soft objective가 아니라 hard equality다.
2. Barrier는 output distribution, reference fact 또는 여러 heuristic score를 섞어 만들지
   않는다.
3. Barrier가 보호하는 유일한 상태는 남은 fixed-\(z\) edit의 conditional completion
   viability다.
4. Output KL, locality, efficacy, generalization과 previous-edit retention은 controller가
   보지 않는 held-out outcome이다.
5. MEMIT/AlphaEdit closed form은 final answer가 아니라 admissible instantaneous write
   basis와 metric을 제공한다.
6. 단순 Euler subdivision은 method contribution이 아니다.
7. 전체 알고리즘은 global convex optimizer가 아니다. 각 fixed-rank local state에서만
   affine equality와 하나의 convex quadratic barrier inequality를 푼다.

따라서 이전 proposal의 functional anchor bank, per-anchor KL CBF, multi-constraint active
set과 supersession ledger는 FzCB-Edit의 method core에서 제거한다. 해당 항목들은 평가 또는
별도 sequential extension으로만 남긴다.

---

# 1. 연구 질문과 claim boundary

## 1.1 중심 연구 질문

동일한 fixed activation target을 향해 같은 instantaneous progress를 만드는 여러 native
weight velocity가 있을 때, 현재 step을 수행한 뒤 남은 target이 더 비싸거나 unreachable해지는
velocity를 피할 수 있는가?

## 1.2 중심 가설

\[
\boxed{
\text{fixed-}z\text{ progress equality}
\quad+\quad
\text{conditional-completion budget barrier}
}
\]

동일한 progress equality를 만족하는 velocity들의 equality-null freedom 안에서,
conditional suffix action이 작은 state로 이동하면 다음이 가능하다는 것이 중심 가설이다.

- final \(z^\star\) closure를 edit strength와 교환하지 않는다.
- early-layer write가 remaining layers에 감당하기 어려운 residual을 넘기는 것을 줄인다.
- 마지막 layer action concentration과 nonlinear correction burden을 줄인다.
- 동일 \(z^\star\) endpoint 중 perturbation에 더 robust한 weight realization을 찾을 수 있다.
- output reference set을 controller에 넣지 않고도 held-out functional outcome이 개선될 수 있다.

마지막 항목은 보장이 아니라 empirical outcome이다.

## 1.3 barrier가 보장하지 않는 것

FzCB barrier는 다음을 직접 보장하지 않는다.

- unrelated prompt의 output distribution preservation
- 기존 factual knowledge의 population-level preservation
- paraphrase generalization
- neighborhood specificity
- sequential retention
- global minimum-action endpoint
- global nonlinear reachability

본 proposal에서 safety 또는 viability라는 표현은 다음으로 제한한다.

\[
\boxed{
\text{현재 native writer 안에서 남은 fixed-}z\text{ target을}
\atop
\text{제한된 action으로 계속 완성할 수 있는가}
}
\]

## 1.4 ODE는 종속 claim이다

FzCB-Edit은 수학적으로 state-dependent differential controller로 기술한다. 그러나 ODE가
논문의 독립적인 필요조건이라는 claim은 다음이 확인될 때만 허용한다.

- \(T(\theta)\), \(\mathcal C(\theta)\), rank 또는 completion value가 trajectory에서 실제로
  움직인다.
- Frozen-geometry split과 endpoint가 달라진다.
- 동일한 equality와 barrier를 푸는 strong static nonlinear solver보다 최종 endpoint,
  completion feasibility 또는 robustness에서 이점이 있다.

이 조건이 성립하지 않으면 방법의 본질은 ODE가 아니라
reachability-aware constrained writer다.

---

# 2. Stock direct-\(z\)의 정확한 target contract

## 2.1 하나의 absolute \(z^\star\)를 모든 context에 반복하지 않는다

Edit \(i\)의 rewrite context를 \(q_{i,p}\), fact-lookup 위치의 original activation을

\[
a_{i,p}^0
=
h_{\theta_0}^{L}(q_{i,p})
\]

라 하자. Stock direct-\(z\) optimization은 여러 rewrite context에서 동일한 trainable
intervention \(\delta_i\)를 activation에 더하고, target token likelihood와 stock
regularizer를 사용해 \(\delta_i^\star\)를 구한다.

Canonical context \(p=0\)에 대해 stock absolute target은

\[
z_i^\star
=
a_{i,0}^0+\delta_i^\star
\]

로 기록할 수 있다. 그러나 다른 context의 intervention target은

\[
\boxed{
z_{i,p}^{\star,\mathrm{ctx}}
=
a_{i,p}^0+\delta_i^\star
}
\]

이다. 일반적으로

\[
z_{i,p}^{\star,\mathrm{ctx}}
\ne
z_i^\star
\qquad (p\ne0).
\]

따라서 모든 context activation을 동일한 absolute \(z_i^\star\)에 맞추는 equality는 stock
oracle의 의미를 바꾸고 문제를 불필요하게 overconstrain한다.

## 2.2 Core registered target

초기 method core의 hard target map은 edit당 canonical fact-lookup activation 하나를
사용한다.

\[
\Phi(\theta)
=
\operatorname{vec}
\left[
h_\theta^L(q_{i,0})
\right]_{i=1}^{B},
\]

\[
Z^\star
=
\operatorname{vec}
\left[
z_i^\star
\right]_{i=1}^{B}.
\]

여러 native rewrite context를 hard equality에 포함하려면

\[
\Phi_{\mathrm{ctx}}(\theta)
=
\operatorname{vec}
\left[
h_\theta^L(q_{i,p})
\right]_{i,p},
\]

\[
Z_{\mathrm{ctx}}^\star
=
\operatorname{vec}
\left[
a_{i,p}^0+\delta_i^\star
\right]_{i,p}
\]

를 사용한다. 이 확장은 range feasibility와 effective free authority가 남는 경우에만
허용한다. Held-out paraphrase를 controller equality에 넣지 않는다.

## 2.3 Direct-\(z\) identity

한 run에서 다음은 한 번만 계산하고 immutable receipt로 고정한다.

- request identity와 rewrite templates
- lookup indices
- \(a_{i,p}^0\)
- \(\delta_i^\star\)
- canonical \(z_i^\star\)
- context-specific \(z_{i,p}^{\star,\mathrm{ctx}}\)
- direct-\(z\) tensor hash
- model, tokenizer, dtype와 seed

FzCB trajectory 중 \(z^\star\) 또는 \(\delta^\star\)를 다시 Adam으로 최적화하지 않는다.
Target가 unreachable하면 target을 바꾸지 않고 typed infeasibility로 기록한다.

---

# 3. Weight state와 native writer control basis

## 3.1 Editable state

Editable layer 집합을 \(\mathcal R\)라 하고, 각 rewrite weight를

\[
W_l\in\mathbb R^{d_{\mathrm{out},l}\times d_{\mathrm{in},l}}
\]

라 하자. 본문의 \(\operatorname{vec}\)은 column-major convention을 사용한다. 실제
row-major tensor 구현은 transpose-aware operator로 동일한 선형 map을 재현하고, explicit
Kronecker matrix를 만들지 않는다. 전체 state는

\[
\theta
=
\left\{
\operatorname{vec}(W_l)
\right\}_{l\in\mathcal R},
\qquad
\theta(0)=\theta_0.
\]

Trajectory는 coefficient control \(c\)를 통해서만 움직인다.

\[
\frac{d\theta}{ds}
=
T(\theta)c.
\]

여기서 \(s\in[0,1]\)은 wall-clock time이 아니라 fixed-\(z\) homotopy progress다.

## 3.2 MEMIT basis

현재 state에서 edit key matrix를

\[
K_l(\theta)\in\mathbb R^{d_{\mathrm{in},l}\times B}
\]

라 하고, frozen reference second moment를 \(C_l^0\)라 하자. MEMIT write operator의
right factor를

\[
B_l^{\mathrm M}(\theta)
=
K_l(\theta)^\top
\left(
\lambda_l C_l^0
+
K_l(\theta)K_l(\theta)^\top
\right)^{-1}
\]

로 둔다. Instantaneous residual coefficient를

\[
R_l(s)\in\mathbb R^{d_{\mathrm{out},l}\times B}
\]

라 하면 admissible weight velocity는

\[
\dot W_l
=
R_l B_l^{\mathrm M}.
\]

\(c_l=\operatorname{vec}(R_l)\)로 두면

\[
\operatorname{vec}(\dot W_l)
=
T_l^{\mathrm M}c_l,
\]

\[
\boxed{
T_l^{\mathrm M}
=
\left(B_l^{\mathrm M}\right)^\top
\otimes I_{d_{\mathrm{out},l}}.
}
\]

FzCB는 \(R_l\)을 기존 residual allocator에서 한 번 받아 고정하지 않는다. 모든 editable
layer의 \(R_l\)을 한 waypoint의 joint control variable로 푼다.

## 3.3 AlphaEdit basis

초기 AlphaEdit extension은 pinned stock solve와 fixed hard projector를 그대로 선형
연산자로 사용한다. Preserved-key projector를 \(P_l^0\), trajectory entry의 history cache를
\(C_{H,l}^0\)라 하자. Current edit key만 waypoint에서 refresh한다.

\[
A_l^\alpha(\theta)
=
P_l^0
\left(
K_l(\theta)K_l(\theta)^\top+C_{H,l}^0
\right)
+
\lambda_lI.
\]

Pinned AlphaEdit orientation 이전의 solve는

\[
A_l^\alpha X_l
=
P_l^0K_lR_l^\top
\]

이다. Weight orientation을 맞춘 velocity는

\[
\dot W_l
=
X_l^\top
=
R_lB_l^\alpha,
\]

\[
\boxed{
B_l^\alpha
=
K_l^\top
(P_l^0)^\top
(A_l^\alpha)^{-\top}.
}
\]

따라서 exact residual-to-update linear map의 vectorization을

\[
T_l^\alpha=(B_l^\alpha)^\top\otimes I
\]

로 사용한다. 구현에서는 inverse를 만들지 않고 pinned linear solve 자체를 operator로
호출한다.

\(P_l^0\)와 \(C_{H,l}^0\)는 한 edit trajectory 동안 고정한다. Current edit key를 Euler
waypoint마다 history cache에 다시 append하지 않는다. Sequential setting에서는 terminal
edit이 성공적으로 commit된 뒤에만 해당 edit key를 history cache에 한 번 반영한다.

이 parameterization은 pinned stock Alpha writer space 안에서만 움직인다. 다음은 core에서
제외한다.

- 매 waypoint projector rank를 다시 선택하는 dynamic \(P_l(\theta)\)
- \((I-P_l)\) 방향으로의 soft escape
- output constraint를 이용한 projector relaxation

이들은 hard Alpha basis가 structurally unreachable한 경우의 별도 extension이다.

## 3.4 Joint basis와 coefficient gauge 제거

모든 layer basis를 모아

\[
T(\theta)
=
\operatorname{blkdiag}
\left(
T_l(\theta)
\right)_{l\in\mathcal R},
\]

\[
c
=
\operatorname{concat}
\left(
c_l
\right)_{l\in\mathcal R}
\]

로 둔다.

\(T\)가 non-injective이면 \(\ker T\)에 속한 coefficient는 실제 weight를 전혀 움직이지 않는
pure gauge다. 이 방향에 ridge를 더해 가짜 control authority를 만들지 않는다. Rank-revealing
factorization으로

\[
T=\widetilde TQ^\top,
\qquad
\widetilde T\text{ has independent columns}
\]

를 만들고 reduced coefficient

\[
\widetilde c=Q^\top c
\]

에서 solver를 정의한다. 아래에서는 표기를 단순화하기 위해 reduced basis도 \(T,c\)로 쓴다.

## 3.5 Fixed weight-action metric

Action의 단위를 trajectory에 따라 바꾸지 않도록 original model에서 고정한 block metric을

\[
M_0
=
\operatorname{blkdiag}
\left(
M_l^0
\right)_{l\in\mathcal R}
\succ0
\]

로 둔다. Weight velocity action은

\[
\ell(c;\theta)
=
\frac12
\|T(\theta)c\|_{M_0}^2
=
\frac12c^\top H(\theta)c,
\]

\[
\boxed{
H(\theta)
=
T(\theta)^\top M_0T(\theta).
}
\]

MEMIT의 기본 metric은 covariance action

\[
\|\dot W_l\|_{C_l^0}^2
=
\operatorname{tr}
\left(
\dot W_l C_l^0\dot W_l^\top
\right)
\]

에서 유도한다. AlphaEdit은 hard projected basis를 admissible set으로 사용하고, 동일한
\(W_0\)-relative covariance metric을 공통 action 단위로 사용한다. AlphaEdit의 native
normal equation과 공통 action metric을 동일한 객체라고 주장하지 않고 둘을 별도 receipt로
보고한다.

Column-major vectorization에서는 예를 들어

\[
M_l^0
=
\left(C_l^0+\epsilon_M I\right)\otimes I_{d_{\mathrm{out},l}},
\qquad
\epsilon_M>0
\]

로 둔다. \(\epsilon_M\)은 output outcome을 보고 조정하지 않고 covariance의 precision
floor에서 preregister한다. 이 항은 새로운 preservation barrier가 아니라 action 단위를
nondegenerate하게 만드는 numerical metric definition이다.

Reduced \(T\)와 \(M_0\succ0\) 아래에서 \(H\succ0\)가 되어 local coefficient problem의
strict convexity가 확보된다.

---

# 4. Fixed-\(z\) homotopy equality

## 4.1 Scheduled target

초기 registered activation을

\[
\Phi_0=\Phi(\theta_0)
\]

라 하고 final displacement를

\[
d=Z^\star-\Phi_0
\]

로 둔다. Fixed oracle target으로 향하는 homotopy는

\[
\boxed{
z_d(s)
=
\Phi_0+sd,
\qquad
s\in[0,1].
}
\]

이는 waypoint마다 final \(Z^\star\)를 이미 만족한다는 뜻이 아니다. 하나의 고정된 final
oracle을 향하는 scheduled activation path다.

Tracking error는

\[
e(\theta,s)
=
z_d(s)-\Phi(\theta).
\]

## 4.2 Full-model control map

Registered activation의 full-model Jacobian을

\[
J_\Phi(\theta)
=
\frac{\partial\Phi(\theta)}{\partial\theta}
\]

라 하면 native coefficient가 만드는 instantaneous activation velocity는

\[
\frac{d\Phi}{ds}
=
J_\Phi(\theta)T(\theta)c.
\]

따라서 control map은

\[
\boxed{
\mathcal C(\theta)
=
J_\Phi(\theta)T(\theta).
}
\]

이 식은 local write key의 \(B_lK_l\)만 보는 것이 아니라, 해당 weight velocity가
attention, residual stream, LayerNorm과 later blocks를 통과해 실제 registered target
activation에 미치는 전체 first-order effect를 사용한다.

## 4.3 Tracking equality

Continuous exact trajectory는

\[
\boxed{
\mathcal C(\theta)c
=
d.
}
\]

이는

\[
\Phi(\theta(s))=z_d(s)
\]

를 \(s\)로 미분한 tangent equality다.

Finite-step implementation은 별도의 feedback gain을 선택하지 않고 다음 waypoint를 직접
사용한다. 현재 accepted state \((\theta_n,s_n)\)와
\(s_{n+1}=s_n+\Delta s_n\)에 대해

\[
\boxed{
b_n
=
\frac{
z_d(s_{n+1})-\Phi(\theta_n)
}{
\Delta s_n
},
\qquad
\mathcal C_nc=b_n.
}
\]

현재 state가 exact waypoint 위에 있으면 \(b_n=d\)다. Drift가 있으면 \(b_n\)이 다음
scheduled target까지의 correction을 자동 포함한다. Finite nonlinear closure는 actual
forward corrector가 담당한다.

Batch request의 equality를 평균 scalar로 줄이지 않는다. 각 request의 registered
full-vector target을 stack하고, worst-request residual을 별도로 기록한다.

## 4.4 Range feasibility

Continuous 분석에서는 \(b=d\), discrete step에서는 Section 4.3의 \(b_n\)을 사용한다.
Equality가 성립하려면

\[
b\in\operatorname{Range}(\mathcal C)
\]

여야 한다. Metric-whitened map을

\[
A
=
\mathcal C H^{-1/2}
=
U_r\Sigma_rV_r^\top
\]

로 분해하면 relative range residual은

\[
\boxed{
\epsilon_{\mathrm{range}}
=
\frac{
\left\|
\left(I-U_rU_r^\top\right)b
\right\|_2
}{
\|b\|_2+\varepsilon
}.
}
\]

\(\epsilon_{\mathrm{range}}\)가 numerical tolerance보다 크면 pseudoinverse로 residual을
숨기지 않는다. 해당 state는 equality-infeasible이다.

## 4.5 Minimum-action tracking velocity

Barrier가 없을 때의 nominal velocity는

\[
c_{\mathrm{eq}}
=
\arg\min_c
\frac12c^\top Hc
\quad
\text{s.t.}\quad
\mathcal Cc=b.
\]

Range-feasible하면

\[
\boxed{
c_{\mathrm{eq}}
=
H^{-1}\mathcal C^\top
\left(
\mathcal CH^{-1}\mathcal C^\top
\right)^\dagger b.
}
\]

Official layer sweep의 delta를 한 state의 vector field로 간주하거나
\(1/(1-s)\)로 rescale하지 않는다. Stock MEMIT/AlphaEdit은 endpoint baseline이며,
\(T,H\)만 instantaneous geometry로 승계한다.

## 4.6 실제 guide authority

Whitened control을 \(x=H^{1/2}c\)라 하면 equality는

\[
Ax=b,
\qquad
A=\mathcal CH^{-1/2}
\]

이다. \(N\)의 column을 \(\ker A\)의 orthonormal basis로 두면 모든
equality-preserving velocity는

\[
\boxed{
c
=
c_{\mathrm{eq}}
+
H^{-1/2}Ny.
}
\]

Raw coefficient에서 \(\ker T\) gauge를 미리 제거했더라도 실제 weight-space guide
authority를 직접 측정한다.

\[
\boxed{
d_{\mathrm{eff}}
=
\operatorname{rank}
\left(
TH^{-1/2}N
\right).
}
\]

Reduced injective \(T\)에서는 일반적으로
\(d_{\mathrm{eff}}=\dim\ker A\)다. \(d_{\mathrm{eff}}=0\)이면
barrier는 trajectory를 선택할 수 없고 feasibility checker로만 작동한다.

---

# 5. 단일 conditional-completion barrier의 유도

## 5.1 Exact ideal value

현재 state \((\theta,s)\)에서 남은 fixed-\(z\) target을 admissible writer로 완성하는 exact
cost-to-go는

\[
V^\star(\theta,s)
=
\inf_{c(\cdot)}
\int_s^1
\frac12
c(\sigma)^\top
H(\theta(\sigma))
c(\sigma)
\,d\sigma
\]

subject to

\[
\frac{d\theta}{d\sigma}
=
T(\theta)c,
\]

\[
\Phi(\theta(\sigma))
=
z_d(\sigma),
\]

\[
\theta(s)=\theta
\]

로 정의된다.

Spent action state를

\[
\frac{dE}{ds}
=
\frac12c^\top Hc,
\qquad
E(0)=0
\]

로 두고 총 budget을 \(A_{\max}\)라 하면 ideal barrier는

\[
h^\star(\theta,s,E)
=
A_{\max}-E-V^\star(\theta,s)
\]

이다. 이 값은 HJB problem이므로 LLM scale에서 직접 계산하지 않는다.

## 5.2 Frozen-geometry suffix value

Practical core는 exact value를 여러 heuristic barrier로 분해하지 않고, 하나의 local
conditional value로 근사한다.

남은 homotopy interval을

\[
\tau=1-s
\]

라 하고 final activation residual을

\[
r(\theta)
=
Z^\star-\Phi(\theta)
\]

로 둔다. 현재 \(T,H,\mathcal C\)를 남은 interval 동안 고정하고, coefficient displacement
\(q=\tau c\)로 residual을 닫는 문제를 생각한다.

\[
\widehat V_{\mathrm{suf}}(\theta,s)
=
\min_q
\frac{1}{2\tau}q^\top Hq
\]

subject to

\[
\mathcal Cq=r.
\]

Range-feasible하면

\[
\boxed{
\widehat V_{\mathrm{suf}}(\theta,s)
=
\frac{1}{2\tau}
r^\top
\left(
\mathcal CH^{-1}\mathcal C^\top
\right)^\dagger
r.
}
\]

\(r\notin\operatorname{Range}(\mathcal C)\)이면

\[
\widehat V_{\mathrm{suf}}=+\infty.
\]

Exact homotopy 위에서는

\[
r=\tau d
\]

이므로, instantaneous tracking action을

\[
V_{\mathrm{inst}}
=
\frac12
d^\top
\left(
\mathcal CH^{-1}\mathcal C^\top
\right)^\dagger
d
\]

라 할 때

\[
\boxed{
\widehat V_{\mathrm{suf}}
=
\tau V_{\mathrm{inst}}.
}
\]

따라서 \(V_{\mathrm{inst}}\)와 달리

\[
\lim_{s\to1}\widehat V_{\mathrm{suf}}=0
\]

이다.

## 5.3 자연적으로 유도되는 budget

초기 state에서

\[
\boxed{
A_0
=
\widehat V_{\mathrm{suf}}(\theta_0,0)
}
\]

를 계산한다. 이는 initial full-model linearization과 native writer metric이 예측하는 최소
fixed-\(z\) completion action이다.

FzCB의 유일한 barrier는

\[
\boxed{
h_{\mathrm{cc}}(\theta,s,E)
=
A_0-E-\widehat V_{\mathrm{suf}}(\theta,s)
\ge0.
}
\]

Pointwise reference trajectory, preservation fact bank 또는 output KL threshold로 envelope를
만들지 않는다. Scientific slack을 edit별로 tune하지 않는다. Numerical solver tolerance만
별도 receipt로 둔다.

더 완화된 scalar budget이 필요한 연구에서는 independent same-\(z\) static solver의 총
action을 \(A_{\max}\)로 사용할 수 있다. 단, 이 경우 해당 static solve의 비용을 method
compute에 포함하며, 그 solver의 pointwise trajectory를 envelope로 사용하지 않는다.

## 5.4 Frozen geometry에서의 exact negative control

\(T,H,\mathcal C\)가 고정되고 minimum-action control이

\[
c(s)=c_0
\]

로 일정하다고 하자. Exact homotopy에서는

\[
E(s)
=
s\frac12c_0^\top Hc_0,
\]

\[
\widehat V_{\mathrm{suf}}(s)
=
(1-s)\frac12c_0^\top Hc_0.
\]

따라서

\[
\boxed{
E(s)+\widehat V_{\mathrm{suf}}(s)
=
A_0.
}
\]

즉 frozen linear geometry에서 barrier는 path를 바꾸지 않으며, Euler step을 더 잘게
나누는 것만으로 이점을 만들 수 없다. FzCB correction은 state-dependent geometry 때문에
equality-only path가 future completion action을 증가시킬 때만 의미가 생긴다.

## 5.5 Continuous barrier condition

\[
h_{\mathrm{cc}}
=
A_0-E-\widehat V_{\mathrm{suf}}
\]

이므로

\[
\frac{dh_{\mathrm{cc}}}{ds}
=
-\frac12c^\top Hc
-\partial_s\widehat V_{\mathrm{suf}}
-\nabla_\theta\widehat V_{\mathrm{suf}}^\top Tc.
\]

Standard CBF condition

\[
\frac{dh_{\mathrm{cc}}}{ds}
\ge
-\kappa h_{\mathrm{cc}}
\]

은 다음 하나의 inequality가 된다.

\[
\boxed{
\frac12c^\top Hc
+
\partial_s\widehat V_{\mathrm{suf}}
+
\nabla_\theta\widehat V_{\mathrm{suf}}^\top Tc
\le
\kappa h_{\mathrm{cc}}.
}
\]

Core strict-budget variant는

\[
\kappa=0
\]

을 사용한다. \(\kappa>0\)은 positive slack을 사용할 수 있게 하는 별도 calibrated
relaxation이며 core claim과 혼합하지 않는다.

## 5.6 Barrier의 정확한 역할

Fixed-\(z\) equality와 barrier의 역할은 서로 다르다.

| 구성 | 역할 |
|---|---|
| \(\mathcal Cc=d\), discrete \(\mathcal C_nc=b_n\) | 지금 반드시 만들어야 하는 activation progress |
| \(h_{\mathrm{cc}}\ge0\) | 지금까지의 action과 남은 completion action의 총 budget |
| \(c=c_{\mathrm{eq}}+H^{-1/2}Ny\) | 같은 progress를 만드는 weight freedom |
| barrier correction \(H^{-1/2}Ny\) | 미래 completion을 어렵게 만드는 same-progress 방향 제거 |

Barrier는 target으로 끌어당기지 않는다. Target progress를 희생해 action을 줄일 수도 없다.
Correction은 equality-null direction 안에서만 발생한다.

---

# 6. Local convex control problem

## 6.1 Per-state QCQP

현재 state에서 equality-only minimum-action velocity \(c_{\mathrm{eq}}\)를 nominal로 둔다.
FzCB velocity는

\[
\boxed{
\begin{aligned}
c^\star
=
\arg\min_c\quad&
\frac12
(c-c_{\mathrm{eq}})^\top
H
(c-c_{\mathrm{eq}})
\\
\text{s.t.}\quad&
\mathcal Cc
=
b,
\\
&
\frac12c^\top Hc
+
\partial_s\widehat V_{\mathrm{suf}}
+
\nabla_\theta\widehat V_{\mathrm{suf}}^\top Tc
\le
\kappa h_{\mathrm{cc}}.
\end{aligned}
}
\]

Reduced writer basis에서 \(H\succ0\)이면 objective는 strictly convex다. Equality는 affine이고
barrier condition은 convex quadratic inequality다. 따라서 feasible한 한 local state의
해는 유일하다.

이는 global editing problem이 convex하다는 뜻이 아니다. 다음은 accepted step마다 바뀐다.

\[
T(\theta),
\quad
H(\theta),
\quad
J_\Phi(\theta),
\quad
\mathcal C(\theta),
\quad
\widehat V_{\mathrm{suf}}(\theta,s).
\]

## 6.2 Null-space form

\[
g
=
T^\top
\nabla_\theta\widehat V_{\mathrm{suf}}
\]

를 coefficient-space barrier gradient라 하자. Whitened null-space form

\[
c
=
c_{\mathrm{eq}}
+
H^{-1/2}Ny
\]

를 대입하면 equality는 자동 만족한다. Local problem은

\[
\min_y
\frac12\|y\|_2^2
\]

subject to 하나의 convex quadratic inequality로 줄어든다.

Projected barrier authority는

\[
\boxed{
g_{\mathrm{free}}
=
N^\top H^{-1/2}g.
}
\]

Equality-only point에서의 barrier residual을

\[
\psi_0
=
\frac12
c_{\mathrm{eq}}^\top Hc_{\mathrm{eq}}
+
g^\top c_{\mathrm{eq}}
+
\partial_s\widehat V_{\mathrm{suf}}
-
\kappa h_{\mathrm{cc}}
\]

라 하면 one-barrier constraint는 정확히

\[
\boxed{
\frac12\|y\|_2^2
+
g_{\mathrm{free}}^\top y
+
\psi_0
\le0
}
\]

가 된다.

따라서 local solution은 scalar rectification으로 닫힌형태를 갖는다.

### Barrier inactive

\[
\psi_0\le0
\quad\Longrightarrow\quad
y^\star=0,
\qquad
c^\star=c_{\mathrm{eq}}.
\]

### Barrier active and feasible

\[
\psi_0>0,
\qquad
\|g_{\mathrm{free}}\|_2^2\ge2\psi_0
\]

이면 minimum-norm correction은

\[
\boxed{
y^\star
=
-\alpha^\star g_{\mathrm{free}},
\qquad
\alpha^\star
=
1-
\sqrt{
1-
\frac{2\psi_0}
{\|g_{\mathrm{free}}\|_2^2}
}.
}
\]

### Local infeasibility

\[
\psi_0
-
\frac12\|g_{\mathrm{free}}\|_2^2
>0
\]

이면 equality affine space와 CBF sublevel이 교차하지 않는다.

\[
\|g_{\mathrm{free}}\|\approx0,
\qquad
\psi_0>0
\]

도 local infeasibility다. Completion value가 equality-null weight freedom을 구별하지 못하기
때문이다.

따라서 single-barrier core는 generic multi-constraint active-set solver를 요구하지 않는다.
Equality KKT, suffix KKT와 하나의 scalar rectification 계수로 구성된다.

## 6.3 Suffix KKT solve

Suffix value의 optimal coefficient displacement를 \(q^\star\)라 하면

\[
\mathcal L_{\mathrm{suf}}
=
\frac{1}{2\tau}q^\top Hq
+
\lambda^\top(\mathcal Cq-r).
\]

KKT system은

\[
\boxed{
\begin{bmatrix}
\tau^{-1}H & \mathcal C^\top
\\
\mathcal C & 0
\end{bmatrix}
\begin{bmatrix}
q^\star\\
\lambda^\star
\end{bmatrix}
=
\begin{bmatrix}
0\\
r
\end{bmatrix}.
}
\]

\(\mathcal C\)에 redundant row가 있으면 multiplier가 유일하지 않다. SVD로 independent
equality만 남긴다.

\[
\widetilde{\mathcal C}
=
U_r^\top\mathcal C,
\qquad
\widetilde r
=
U_r^\top r.
\]

KKT와 sensitivity는 \((\widetilde{\mathcal C},\widetilde r)\)에서 계산한다.

## 6.4 Envelope derivative

Fixed-rank neighborhood에서 optimal \(q^\star,\lambda^\star\)를 stop-gradient하면

\[
\boxed{
d\widehat V_{\mathrm{suf}}
=
\frac{1}{2\tau}
q^{\star\top}(dH)q^\star
+
\lambda^{\star\top}
\left[
(d\mathcal C)q^\star-dr
\right]
-
\frac{q^{\star\top}Hq^\star}{2\tau^2}
d\tau.
}
\]

\[
d\tau=-ds,
\qquad
dr=-d\Phi
\]

이다. 따라서 pseudoinverse 자체를 직접 autodiff할 필요는 없다.

그러나 \(d\mathcal C=d(J_\Phi T)\)에는 full-model Hessian-vector product와 \(dT\)가
포함된다. Envelope theorem은 second-order model cost를 제거하지 않는다. Production
method는 이를 JVP/VJP/HVP operator로 계산해야 한다.

## 6.5 Rank change

Fixed-rank 구간에서만 위 derivative와 local smoothness를 주장한다. Candidate에서 rank가
변했다는 이유만으로 자동 reject하지 않는다. Rank change 자체를 hidden second barrier로
사용하지 않기 위해 다음 순서를 따른다.

1. Candidate state에서 \(\mathcal C,H\)를 다시 계산한다.
2. Independent row space와 range residual을 다시 평가한다.
3. New rank에서 suffix value를 다시 푼다.
4. Actual equality, budget과 numerical stability가 만족되면 accept한다.
5. Range가 사라지거나 KKT residual이 numerical tolerance를 넘을 때만 reject한다.

---

# 7. Full predictor–corrector pipeline

## 7.1 Offline initialization

한 edit batch에 대해 다음을 먼저 수행한다.

### Step O1. Stock target capture

Stock MEMIT/AlphaEdit과 동일한 target optimizer를 정확히 한 번 실행한다.

\[
\{a_{i,p}^0\}_{i,p},
\qquad
\{\delta_i^\star,z_i^\star\}_i
\]

와 모든 lookup/context receipt를 저장한다.

### Step O2. Registered target construction

Canonical core이면

\[
\Phi(\theta)
=
\operatorname{vec}[h_\theta^L(q_{i,0})]_i,
\quad
Z^\star
=
\operatorname{vec}[z_i^\star]_i.
\]

Context extension이면

\[
\Phi_{\mathrm{ctx}}(\theta)
=
\operatorname{vec}[h_\theta^L(q_{i,p})]_{i,p},
\quad
Z^\star
=
\operatorname{vec}[a_{i,p}^0+\delta_i^\star]_{i,p}.
\]

이 branch에서는 아래 \(\Phi\) 표기가 \(\Phi_{\mathrm{ctx}}\)를 뜻한다. Canonical
\(\Phi\)와 stacked context \(Z^\star\)를 혼합하지 않는다.

### Step O3. Original action geometry

\[
\theta_0,
\quad
C_l^0,
\quad
M_0,
\quad
P_l^0\text{ and }C_{H,l}^0\text{ for AlphaEdit}
\]

를 고정한다. \(M_0\), hard Alpha projector와 trajectory-entry history cache는 run 중
변경하지 않는다.

### Step O4. Initial writer and reachability

\[
T_0,
\quad
H_0,
\quad
\mathcal C_0
\]

를 만들고

\[
d=Z^\star-\Phi_0
\]

에 대한 range residual을 계산한다. Initial final residual이 writer range 밖이면 method
instance를 시작하지 않는다.

### Step O5. Initial budget

\[
A_0
=
\widehat V_{\mathrm{suf}}(\theta_0,0)
\]

를 계산하고

\[
s_0=0,
\qquad
E_0=0,
\qquad
h_{\mathrm{cc},0}=0
\]

로 초기화한다.

### Step O6. Initial local viability

Initial equality, suffix KKT와 barrier sensitivity를 계산해

\[
\psi_{\min,0}
=
\psi_{0,0}
-
\frac12\|g_{\mathrm{free},0}\|_2^2
\]

를 검사한다. \(\psi_{\min,0}>\tau_{\mathrm{cbf}}\)이면 초기 boundary에서 admissible
velocity가 없으므로 method instance를 시작하지 않는다. Initial step size를 줄여도
continuous local feasible set은 바뀌지 않는다.

## 7.2 One accepted macro step

현재 accepted state를

\[
(\theta_n,s_n,E_n)
\]

이라 하고 proposed progress를

\[
s_{n+1}=s_n+\Delta s_n
\]

이라 하자.

### Step P1. Full forward refresh

현재 model에서 다음을 실제 forward로 다시 수집한다.

\[
\Phi_n,
\quad
K_{l,n},
\quad
B_{l,n},
\quad
T_n,
\quad
H_n.
\]

MEMIT이 layer sweep 중 key와 remaining residual을 refresh한다는 사실과 구분해야 한다.
FzCB의 refresh는 모든 editable layer를 동시에 active하게 유지하고, 각 homotopy waypoint에서
full-model control map을 다시 구성한다.

### Step P2. Tracking state

\[
z_d(s_{n+1})
=
\Phi_0+s_{n+1}d,
\]

\[
b_n
=
\frac{
z_d(s_{n+1})-\Phi_n
}{
\Delta s_n
}.
\]

### Step P3. Matrix-free control map

\[
\mathcal C_n=J_{\Phi,n}T_n
\]

을 explicit dense matrix로 만들지 않는다.

Forward operator

\[
v\mapsto\mathcal C_nv
\]

는 \(T_nv\) weight tangent를 만든 뒤 full-model JVP로 계산한다.

Adjoint operator

\[
u\mapsto\mathcal C_n^\top u
\]

는 activation VJP 뒤 \(T_n^\top\) projection으로 계산한다.

### Step P4. Range and authority

\[
\epsilon_{\mathrm{range},n},
\quad
\operatorname{rank}(\mathcal C_n),
\quad
d_{\mathrm{eff},n}
\]

을 계산한다. Analytical contract의 accepted waypoint는
\(\Phi_n=z_d(s_n)\)이므로 \(b_n=d\)다. Finite-precision 구현은 tangent demand \(d\)와
실제 feedback demand \(b_n\)의 range residual을 모두 계산한다. 이때

\[
d\notin\operatorname{Range}(\mathcal C_n)
\]

이면 step size와 무관한 typed local infeasibility다. \(d\)는 reachable하지만 \(b_n\)만
out-of-range이면 accepted-state tracking error를 다음 step에 넘긴 numerical failure다.
같은 accepted state에서 \(\Delta s_n\)만 줄이면 correction term이 오히려 커진다. 별도의
무상 fixed-\(s_n\) weight repair를 허용하지 않고, 저장된 이전 accepted state
\((\theta_{n-1},s_{n-1},E_{n-1})\)로 rollback한 뒤 직전 macro step을 더 작은 step으로
다시 푼다. Initial state에서 이 상황이 발생하면 numerical-contract failure다.

### Step P5. Equality-only nominal

\[
c_{\mathrm{eq},n}
=
\arg\min_c
\frac12c^\top H_nc
\quad
\text{s.t.}\quad
\mathcal C_nc=b_n.
\]

### Step P6. Conditional suffix solve

\[
r_n=Z^\star-\Phi_n,
\qquad
\tau_n=1-s_n
\]

에 대해 suffix KKT를 풀어

\[
q_n^\star,
\quad
\lambda_n^\star,
\quad
\widehat V_{\mathrm{suf},n}
\]

을 얻는다.

### Step P7. Barrier sensitivity

Envelope formula로

\[
\partial_s\widehat V_{\mathrm{suf},n},
\qquad
T_n^\top
\nabla_\theta\widehat V_{\mathrm{suf},n}
\]

을 계산한다. Equality-null projection이 numerical zero이면 barrier correction authority가
없음을 기록한다.

### Step P8. Local QCQP

\[
c_n^\star
\]

를 Section 6.1의 equality + single-barrier QCQP로 계산한다. 현재 accepted state에서
QCQP가 infeasible하면 \(\Delta s_n\)만 줄여 재시도하지 않는다. Initial state이면 typed
initial infeasibility이고, 이후 state이면 Section 7.5의 next-state viability 검사가 이전
candidate를 accept하지 못했어야 하는 numerical contract violation이다.

### Step P9. Euler predictor

\[
\theta_{n+1}^{\mathrm{pred}}
=
\theta_n
+
\Delta s_nT_nc_n^\star.
\]

이는 candidate일 뿐 authoritative accepted state가 아니다.

## 7.3 Nonlinear corrector

한 macro step 안에서는 entry writer basis \(T_n\)과 metric \(H_n\)을 고정하고, total
coefficient \(c\)를 보정한다. Candidate map을

\[
F_n(c)
=
\Phi
\left(
\theta_n+\Delta s_nT_nc
\right)
-
z_d(s_{n+1})
\]

로 둔다. 초기값은

\[
c^{(0)}=c_n^\star
\]

다. \(j\)번째 candidate에서 full forward로 \(F_n(c^{(j)})\)와 candidate Jacobian을 다시
계산하고, minimum-\(H_n\)-norm Newton correction을 푼다.

\[
\delta c^{(j)}
=
\arg\min_{\delta c}
\frac12
\delta c^\top H_n\delta c
\]

\[
\text{s.t.}\quad
\Delta s_n
J_\Phi
\left(
\theta_n+\Delta s_nT_nc^{(j)}
\right)
T_n\delta c
=
-F_n(c^{(j)}).
\]

\[
c^{(j+1)}
=
c^{(j)}
+
\eta_j\delta c^{(j)}
\]

로 update하며 \(\eta_j\)는 actual target residual과 completion budget을 함께 검사하는
backtracking line search로 정한다.

같은 macro step에서 \(T_n\)과 \(H_n\)을 refresh하지 않는 이유는 corrector가 무상의 별도
weight jump가 아니라, entry native writer 안에서 최종 accepted control을 보정하도록 하기
위해서다. 다음 accepted waypoint에 도달한 뒤 Step P1에서 새 \(T_{n+1},H_{n+1}\)를
구성한다.

Terminal corrector coefficient를 \(c_n^{\mathrm{corr}}\)라 한다. Corrector는 barrier 밖으로
별도 권한을 갖지 않는다. Corrected candidate가 completion budget을 위반하면 correction을
accept하지 않고 outer \(\Delta s_n\)를 줄인다. Corrector residual을 slack으로 숨기지
않는다.

## 7.4 Discrete action accounting

Predictor와 corrector는 실제 model에 최종 candidate를 찾기 위한 virtual computation이다.
Accepted candidate는

\[
\theta_{n+1}
=
\theta_n
+
\Delta s_nT_nc_n^{\mathrm{corr}}.
\]

따라서 corrector를 포함한 accepted macro action은

\[
\boxed{
\Delta E_n
=
\frac{\Delta s_n}{2}
c_n^{\mathrm{corr}\top}
H_n
c_n^{\mathrm{corr}}
=
\frac{1}{2\Delta s_n}
\left\|
\theta_{n+1}-\theta_n
\right\|_{M_0}^2.
}
\]

로 기록한다.

\[
E_{n+1}=E_n+\Delta E_n.
\]

Predictor만의 action을 기록하고 corrector displacement를 누락하지 않는다. 추가로 numerical
solver가 방문한 micro-candidate path length는 compute diagnostic으로 별도 보고하되,
method trajectory action과 혼합하지 않는다.

## 7.5 Actual candidate verification

Corrected candidate \(\theta_{n+1}\)에서 다음을 actual forward와 actual KKT solve로 검증한다.

### Tracking

\[
\left\|
\Phi(\theta_{n+1})
-
z_d(s_{n+1})
\right\|_2
\le
\tau_z.
\]

### Range

\(s_{n+1}<1\)이면 Section 4.4의 range residual을 next homotopy tangent \(b=d\)로
평가한다.

\[
\epsilon_{\mathrm{tan},n+1}
\le
\tau_{\mathrm{range}}.
\]

Suffix residual은 별도로

\[
\epsilon_{\mathrm{suf},n+1}
=
\frac{
\left\|
(I-U_rU_r^\top)r_{n+1}
\right\|_2
}{
\|r_{n+1}\|_2+\varepsilon
}
\le
\tau_{\mathrm{range}}
\]

로 검사하고 suffix KKT의 primal residual도 확인한다. Exact waypoint에서는
\(r_{n+1}=(1-s_{n+1})d\)이므로 두 검사는 같은 direction을 보지만, finite tracking
tolerance 때문에 둘 다 기록한다.

### Completion budget

\[
\boxed{
E_{n+1}
+
\widehat V_{\mathrm{suf}}(\theta_{n+1},s_{n+1})
\le
A_0+\epsilon_{\mathrm{num}}.
}
\]

### Solver residual

KKT primal, dual과 complementarity residual이 preregistered numerical tolerance 안에 있어야
한다.

### Next-state local viability

\(s_{n+1}<1\)이면 candidate geometry에서 equality-only control, suffix sensitivity와
projected authority를 다시 계산하고

\[
\boxed{
\psi_{\min,n+1}
=
\psi_{0,n+1}
-
\frac12
\|g_{\mathrm{free},n+1}\|_2^2
\le
\tau_{\mathrm{cbf}}
}
\]

를 확인한다. 이는 다음 infinitesimal fixed-\(z\) progress를 만드는 local CBF velocity가
적어도 하나 존재한다는 검사다. 단순히 \(h_{\mathrm{cc},n+1}\ge0\)인 candidate가 다음
step에서 dead end가 되는 것을 막는다.

모두 만족할 때만

\[
(\theta_n,s_n,E_n)
\leftarrow
(\theta_{n+1},s_{n+1},E_{n+1})
\]

로 accept한다.

## 7.6 Backtracking

Candidate가 tracking, range, budget, next-state viability 또는 numerical check를 위반하면

\[
\Delta s_n
\leftarrow
\beta\Delta s_n,
\qquad
0<\beta<1
\]

로 줄이고 Step P1부터 다시 시작한다.

Backtracking 때 이전 candidate의 \(T,H,\mathcal C\)를 재사용하지 않는다. Accepted model
state에서 다시 full forward한다.

Backtracking은 finite Euler/corrector candidate의 nonlinear 또는 budget failure를 해결하는
장치다. 이미 accepted된 현재 state의 tangent range failure나 continuous QCQP
infeasibility를 step-size 문제로 재분류하지 않는다.

다음은 허용하지 않는다.

- barrier slack을 결과 후 증가
- \(z^\star\) 재최적화
- context 평균 scalar equality로 몰래 축소
- unreachable singular direction의 pseudoinverse truncation으로 residual 은폐
- Alpha projector 밖으로 outcome-dependent escape

## 7.7 Terminal condition

Knowledge edit은 utility operating point에서 중간 종료하지 않는다. 반드시

\[
s=1
\]

까지 도달해야 한다.

\(s=1\)에서 suffix value는

\[
\widehat V_{\mathrm{suf}}(\theta,1)
=
\begin{cases}
0,
&
\|\Phi(\theta)-Z^\star\|\le\tau_z,
\\
+\infty,
&
\text{otherwise}
\end{cases}
\]

로 정의한다.

Terminal endpoint는

\[
\boxed{
\|\Phi(\theta_1)-Z^\star\|\le\tau_z,
\qquad
E_1\le A_0+\epsilon_{\mathrm{num}}.
}
\]

를 만족해야 한다.

Intermediate weight는 모두 virtual state다. 최종 \(\theta_1\)만 atomic transaction으로
commit하고 post-commit activation과 weight identity를 다시 확인한다.

---

# 8. Algorithm summary

## 8.1 Mathematical program

\[
\boxed{
\begin{aligned}
\frac{d\theta}{ds}
&=
T(\theta)c^\star(\theta,s,E),
\\
\frac{dE}{ds}
&=
\frac12c^{\star\top}H(\theta)c^\star,
\\
c^\star
&=
\arg\min_c
\frac12
(c-c_{\mathrm{eq}})^\top
H
(c-c_{\mathrm{eq}})
\\
\text{s.t.}\quad
&
J_\Phi(\theta)T(\theta)c
=
d,
\\
&
\frac12c^\top Hc
+
\partial_s\widehat V_{\mathrm{suf}}
+
\nabla_\theta\widehat V_{\mathrm{suf}}^\top Tc
\le
\kappa
\left[
A_0-E-\widehat V_{\mathrm{suf}}
\right].
\end{aligned}
}
\]

실제 finite step에서는 위 continuous tangent equality 대신

\[
\boxed{
J_\Phi(\theta_n)T_n c_n
=
\frac{
\Phi_0+s_{n+1}d-\Phi(\theta_n)
}{
\Delta s_n
}
}
\]

를 사용하고 nonlinear corrector로 다음 waypoint를 닫는다.

\[
\boxed{
\widehat V_{\mathrm{suf}}(\theta,s)
=
\frac{1}{2(1-s)}
r(\theta)^\top
\left[
\mathcal C(\theta)
H(\theta)^{-1}
\mathcal C(\theta)^\top
\right]^\dagger
r(\theta).
}
\]

## 8.2 Operational pseudocode

1. Stock target optimizer를 한 번 실행해 \(z^\star,\delta^\star\)를 freeze한다.
2. Canonical 또는 context-correct target map \((\Phi,Z^\star)\)를 만든다.
3. \(W_0\)-relative metric \(M_0\)와 fixed Alpha basis를 freeze한다.
4. Initial \(T,H,\mathcal C\)의 range를 검사하고 \(A_0\)를 계산한다.
5. 현재 accepted model에서 full forward한다.
6. Current keys와 native right factors로 \(T,H\)를 refresh한다.
7. JVP/VJP operator로 \(\mathcal C=J_\Phi T\)를 구성한다.
8. Tracking equality의 range와 effective null freedom을 검사한다.
9. Equality-only \(c_{\mathrm{eq}}\)를 푼다.
10. Suffix KKT로 \(\widehat V_{\mathrm{suf}}\)와 multiplier를 구한다.
11. Envelope derivative와 equality-null guide authority를 계산한다.
12. Whitened equality-null 좌표에서 single-barrier scalar rectification을 계산한다.
13. Euler predictor를 만들고 actual full-model corrector를 수행한다.
14. Net accepted displacement로 \(E_{n+1}\)를 계산한다.
15. Actual target, tangent/suffix range, suffix value, total budget과 next-state local
    viability를 검증한다.
16. 실패하면 step을 줄여 accepted state에서 재시작한다.
17. 통과하면 state를 accept하고 다음 waypoint로 간다.
18. \(s=1\)의 exact target과 action budget을 만족할 때만 atomic commit한다.

---

# 9. MEMIT과 AlphaEdit에서 baseline 대비 바뀌는 것

## 9.1 MEMIT

MEMIT은 layer \(l\)에서 current key와 remaining target-layer residual을 다시 계산하고,
한 번의 local closed-form write를 수행한 뒤 다음 layer로 이동한다. FzCB가 새롭게 주장할 수
있는 부분은 단순 refresh가 아니다.

FzCB의 차이는 다음이다.

- 모든 editable layer를 한 waypoint의 joint control로 푼다.
- 이미 지나간 shallow layer도 다음 waypoint에서 다시 조정할 수 있다.
- local key realization이 아니라 full-model registered activation equality를 사용한다.
- 균등 remaining-layer residual split을 고정하지 않는다.
- same-progress freedom을 conditional suffix action으로 선택한다.
- nonlinear corrector와 actual completion budget을 함께 검증한다.

## 9.2 AlphaEdit

Stock AlphaEdit의 eigenvalue-thresholded approximate projector를 한 trajectory의 fixed hard
admissible basis로 강제한다. 이는 계산된 key-space projection constraint이지 exact
null-space 또는 output/function preservation guarantee가 아니다.

Hard Alpha basis에서

\[
r\notin\operatorname{Range}(\mathcal C^\alpha)
\]

이면 현재 strict homotopy waypoint에는 admissible progress tangent가 없다. 이 경우 FzCB는
projector 밖으로 자동 escape하거나 step splitting으로 reachability가 생겼다고 간주하지
않고 typed local Alpha infeasibility를 반환한다. 여러 Euler step을 포함한 global
unreachability는 fixed basis가 전체 reachable set에서 target direction을 제거한다는 별도
조건 없이는 주장하지 않는다.

## 9.3 Closed form의 역할 변화

\[
\boxed{
\text{one-shot endpoint solver}
\quad\longrightarrow\quad
\text{state-dependent admissible writer geometry}
}
\]

Closed form은 제거되지 않는다. \(B_l,T_l,H_l\)를 통해 어떤 velocity가 writer-native인지와
그 action cost를 정의한다. 최종 residual coefficient와 layer allocation은 equality와
completion barrier가 현재 model state에서 다시 결정한다.

---

# 10. Related-work boundary

| 방법 | 이미 해결하는 것 | FzCB와의 구분 |
|---|---|---|
| MEMIT | covariance-regularized batched local write | full-model conditional-completion path를 풀지 않음 |
| AlphaEdit | preserved-key approximate null-space write | projected space 안의 future target viability를 제어하지 않음 |
| EMMET | batched local equality-constrained write; multi-layer variant는 MEMIT residual distribution 재사용 | receding full-model completion value와 pathwise budget을 사용하지 않음 |
| CAKE | causal layer selection과 residual allocation | 동일 progress weight freedom의 conditional suffix cost를 제어하지 않음 |
| BLUE | residual distribution의 weight-shift error를 피하는 boundary-layer update | FzCB는 fixed multi-layer writer에서 actual full-model equality와 suffix viability를 푼다 |
| MetaKE | downstream constraint feedback으로 upstream target representation을 bi-level 최적화 | FzCB는 stock \(z^\star\)를 바꾸지 않고 write trajectory만 선택한다 |
| DOW-KE | 실제 배포되는 multi-layer update의 anchor-free full-model end-to-end optimization | fixed-\(z\) conditional path viability와 다른 문제 |
| KLOD | target amplification을 bounded하게 멈추고 non-target/prefix output distribution을 보존 | FzCB controller는 output reference나 probability threshold를 사용하지 않음 |
| SPHERE | hyperspherical-energy 안정화와 sparse projection을 이용한 sequential preservation | FzCB barrier는 HE나 neuron-uniformity가 아니라 현재 edit의 suffix completion action이다 |
| EvoEdit/BetaEdit | evolving 또는 history-aware null-space로 sequential interference 완화 | FzCB core는 한 edit 안의 fixed hard writer basis에서 completion path를 선택한다 |
| ODESteer | activation density-ratio barrier field | inference-time activation steering이며 fixed endpoint writer가 아님 |
| ODE-M | parameter-space barrier-aware merging trajectory | utility operating point를 선택하지만 FzCB는 \(s=1\) endpoint가 필수 |

Dynamic refresh, multi-layer allocation 또는 exact equality 하나만으로 novelty를 주장하지
않는다. FzCB의 잠정 contribution은 다음 결합이다.

\[
\boxed{
\text{stock fixed oracle}
+
\text{full-model tracking equality}
+
\text{native writer geometry}
+
\text{single conditional-completion budget}
+
\text{receding nonlinear correction}
}
\]

Strong static endpoint solve와 direct-transcription path solver는 필수 comparator다. FzCB가
동일 목적의 static solver와 같으면 ODE necessity claim을 제거한다.

---

# 11. Evaluation contract

## 11.1 Controller-visible quantities

FzCB controller가 볼 수 있는 것은 다음으로 제한한다.

- registered edit activation \(\Phi\)
- frozen \(Z^\star,\delta^\star\)
- current edit keys
- native writer factors \(T,H\)
- full-model edit activation JVP/VJP
- range residual과 singular structure
- spent action \(E\)
- conditional suffix action \(\widehat V_{\mathrm{suf}}\)

## 11.2 Controller-invisible held-out outcomes

다음은 최종 평가에만 사용한다.

- rewrite efficacy
- paraphrase generalization
- neighborhood specificity
- output KL
- previous-edit retention
- general capability
- fluency
- small weight perturbation robustness
- small \(z\) perturbation robustness

q-KL, locality facts 또는 prior-edit prompts를 FzCB velocity에 넣지 않는다.

## 11.3 Mechanism receipts

각 accepted waypoint에서 다음을 기록한다.

- \(s_n,\Delta s_n\)
- full target residual과 worst-request residual
- context identity와 target semantics
- \(T_n,H_n,\mathcal C_n\) operator identity
- rank와 singular spectrum summary
- \(\epsilon_{\mathrm{range},n}\)
- \(d_{\mathrm{eff},n}\)
- \(\widehat V_{\mathrm{suf},n}\)
- \(E_n\)
- \(h_{\mathrm{cc},n}\)
- barrier dual과 active 여부
- \(\|c_n^\star-c_{\mathrm{eq},n}\|_H\)
- layer별 action
- corrector iteration과 net correction
- accepted/rejected/backtracked step
- KKT primal/dual residual
- full forward, JVP, VJP와 HVP 수
- wall-clock과 peak memory

## 11.4 Endpoint primary quantities

Mechanism primary는 다음이다.

\[
\text{canonical fixed-}z\text{ closure},
\]

\[
\text{context-correct intervention residual},
\]

\[
\text{realized total action }E_1,
\]

\[
\text{completion failure rate},
\]

\[
\text{last-layer action fraction},
\]

\[
\text{small perturbation robustness}.
\]

Output metrics가 좋아도 fixed-\(z\) closure가 다르면 같은 realization problem의 비교가 아니다.

---

# 12. Computational realization

## 12.1 Matrix-free solves

\(\mathcal C\)를 materialize하지 않는다.

- \(\mathcal Cv\): low-rank weight tangent \(Tv\)를 만든 뒤 JVP
- \(\mathcal C^\top u\): activation VJP 뒤 \(T^\top\)
- range residual: LSMR 또는 LSQR
- saddle KKT: MINRES 또는 reduced Schur complement
- spectrum: randomized SVD 또는 Lanczos

## 12.2 Exact sensitivity cost

\(\nabla_\theta\widehat V_{\mathrm{suf}}\)는 \(dJ_\Phi\)와 \(dT\)를 포함한다. 따라서 exact
controller는 HVP와 writer-factor sensitivity를 요구한다. 이를 단순 pseudoinverse-free
gradient라고만 표현하지 않는다.

## 12.3 Reduced-space engineering approximation

Full sensitivity를 구현하기 전 correctness prototype은 equality-null basis의 작은
subspace에서 actual suffix value를 finite difference할 수 있다.

\[
c
=
c_{\mathrm{eq}}
+
H^{-1/2}N_ky,
\qquad
N_k\in\mathbb R^{d_c\times k}.
\]

각 direction에 대해

\[
\frac{
\widehat V_{\mathrm{suf}}
\left(
\theta+\epsilon TH^{-1/2}N_ke_j,s
\right)
-
\widehat V_{\mathrm{suf}}
\left(
\theta-\epsilon TH^{-1/2}N_ke_j,s
\right)
}{
2\epsilon
}
\]

을 계산한다. 이는 mechanism prototype이며 final scalable method와 구분해 보고한다.

## 12.4 Short-horizon extension

One-step frozen suffix value가 실제 completion action을 예측하지 못할 때 여러 독립 barrier를
추가하지 않는다. 유일한 scalar value를 \(H\)-step conditional value로 교체할 수 있다.

\[
\widehat V_H(\theta,s)
=
\min_{c_0,\ldots,c_{H-1}}
\sum_{j=0}^{H-1}
\frac{\Delta s_j}{2}
c_j^\top H_jc_j
\]

subject to linearized fixed-\(z\) waypoint equalities다.

이 경우 natural initial budget도 같은 value family에서 다시 정의한다.

\[
A_{0,H}
=
\widehat V_H(\theta_0,0),
\qquad
h_H
=
A_{0,H}-E-\widehat V_H.
\]

그러나 horizon \(H\)가 새로운 method parameter이므로 core formulation에는 포함하지 않는다.

---

# 13. Local geometry와 시각화

한 waypoint의 coefficient space에서:

- fixed-\(z\) equality \(\mathcal Cc=b\)는 affine set이다.
- single CBF inequality는 convex quadratic sublevel이다.
- equality-only \(c_{\mathrm{eq}}\)가 budget을 악화시키지 않으면 그대로 사용한다.
- 악화시키면 equality affine set 위에서만 이동해 feasible boundary의 \(c^\star\)를 선택한다.

따라서 2-D conceptual figure는 다음을 보여야 한다.

1. 파란 affine line: 동일 activation progress
2. 타원 내부: completion-budget compatible velocity
3. \(c_{\mathrm{eq}}\): equality-only minimum-action point
4. \(c^\star\): equality를 유지한 barrier-rectified point
5. \(Ny\): progress를 바꾸지 않는 correction

Global weight trajectory는 하나의 고정 convex bowl 안에서 움직이지 않는다. Full forward 뒤
각 local affine set과 quadratic sublevel이 이동하므로 curved path가 생긴다.

Frozen geometry panel에서는 barrier-on과 barrier-off path가 일치해야 한다. Curved geometry
panel에서만 barrier가 future completion을 보존하는 방향으로 이동해야 한다.

---

# 14. Failure semantics

| Failure | 의미 | Method response |
|---|---|---|
| \(z^\star\) intervention 자체가 target behavior를 만들지 못함 | oracle target 문제 | writer가 해결한다고 주장하지 않음 |
| \(b\notin\operatorname{Range}(\mathcal C)\) | homotopy tangent unreachable | typed infeasibility |
| final \(r\notin\operatorname{Range}(\mathcal C)\) | suffix completion unreachable | \(\widehat V_{\mathrm{suf}}=+\infty\) |
| \(d_{\mathrm{eff}}=0\) | same-progress weight choice 없음 | barrier checker only |
| projected value gradient가 0 | barrier가 freedom을 구별하지 못함 | equality-only endpoint |
| Current-state QCQP infeasible | progress와 budget의 local conflict | typed local failure; step size로 은폐하지 않음 |
| corrector가 budget을 깨뜨림 | local predictor certificate 불충분 | candidate reject |
| candidate의 next-state QCQP가 infeasible | locally viable set의 경계/dead end | candidate reject와 outer backtrack |
| rank threshold에 민감 | value와 derivative 불안정 | independent-row recomputation과 sensitivity report |
| barrier가 항상 inactive | completion guidance가 실제 path를 바꾸지 않음 | barrier contribution claim 제거 |
| static solver와 동일 endpoint | ODE trajectory 불필요 | static constrained writer로 pivot |
| held-out locality 악화 | completion viability가 functional safety로 전이되지 않음 | locality claim 제거; output barrier를 사후 추가하지 않음 |

Feasibility 실패를 target relaxation, post-outcome context 삭제 또는 barrier slack으로 숨기지
않는다.

---

# 15. Reproducibility and numerical contract

각 run은 최소한 다음 identity를 남긴다.

- git commit와 dirty-state manifest
- model/tokenizer/revision/dtype
- request IDs, order와 seed
- canonical/native context text hash와 lookup index
- \(a_{i,p}^0,\delta_i^\star,z_i^\star\) tensor hash
- editable layer set
- \(C_l^0,M_0,\epsilon_M,P_l^0,C_{H,l}^0\) identity
- reduced \(T\) rank와 gauge removal receipt
- \(\kappa\), initial step과 backtracking factor
- range/rank tolerances
- KKT solver와 stopping residual
- waypoint별 \(E,\widehat V_{\mathrm{suf}},h_{\mathrm{cc}}\)
- waypoint별 \(\psi_{\min}\)과 next-state viability margin
- predictor와 corrected candidate identity
- virtual-versus-atomic-commit weight identity
- terminal activation closure
- controller-visible/held-out split identity

Numerical tolerance는 no-op, repeated solve와 precision floor에서 정한다. Held-out output 결과를
본 뒤 tolerance, rank threshold, \(\kappa\), context set 또는 endpoint rule을 변경하지 않는다.

---

# 16. Paper narrative boundary

현재 허용되는 proposal-level narrative는 다음이다.

> Existing closed-form editors select a local weight update using fixed associative-memory geometry.
> FzCB-Edit instead treats that geometry as a state-dependent admissible writer, enforces a frozen
> activation target through a full-model homotopy equality, and uses a single conditional-completion
> budget to select among equality-preserving weight velocities.

다음 표현은 금지한다.

- Euler subdivision이 MEMIT attenuation을 자동 해결한다.
- \(V_{\mathrm{inst}}\)가 exact remaining cost다.
- Barrier가 knowledge preservation을 보장한다.
- 모든 rewrite context가 동일 absolute \(z^\star\)를 가져야 한다.
- MEMIT은 layer 사이에서 key/residual을 refresh하지 않는다.
- AlphaEdit의 projector가 output function을 exact하게 보존한다.
- Local QCQP이 global editing problem을 convex하게 만든다.
- Rank change를 reject하면 continuous ODE guarantee가 유지된다.
- Same-\(z\) reference path가 자연스러운 barrier envelope를 제공한다.
- Intermediate trajectory가 안전하므로 최종 endpoint도 우월하다.

---

# 17. References

## ODE와 control

- [ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](https://arxiv.org/abs/2602.17560)
- [Unlocking the Potential of Continual Model Merging: An ODE Perspective](https://arxiv.org/abs/2605.19409)
- [Control Barrier Function Based Quadratic Programs for Safety Critical Systems](https://arxiv.org/abs/1609.06408)

## Knowledge editing

- [MEMIT: Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355)
- [A Unified Framework for Model Editing / EMMET](https://arxiv.org/abs/2403.14236)
- [CAKE: Causal-Guided Adaptive Knowledge Editing for LLMs](https://aclanthology.org/2026.acl-long.918/)
- [DOW-KE: Anchor-Free Multi-Layer Knowledge Editing via Direct End-to-End Weight Optimization](https://arxiv.org/abs/2608.16932)
- [KLOD: Locality-Preserving Knowledge Editing via Non-Target Distribution Preservation](https://arxiv.org/abs/2608.27839)
- [MetaKE: Meta-Learning for Knowledge Editing Toward a Better Accuracy-Editability Trade-off](https://arxiv.org/abs/2603.12677)
- [Rethinking Residual Distribution in Locate-then-Edit Model Editing / BLUE](https://arxiv.org/abs/2502.03748)
- [Energy-Regularized Sequential Model Editing on Hyperspheres / SPHERE](https://arxiv.org/abs/2510.01172)
- [EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](https://arxiv.org/abs/2510.13851)
- [BetaEdit: Null-Space Constrained Sequential Model Editing](https://arxiv.org/abs/2605.09285)

---

# 18. 2026-08-31 method decision

| 결정 | 상태 |
|---|---|
| ODE 적용 위치 | \(W\)-write stage |
| direct target | stock \(z^\star,\delta^\star\) 한 번 계산 후 고정 |
| context target | 동일 absolute \(z^\star\) 반복 금지; \(a_{i,p}^0+\delta_i^\star\) |
| progress | full-model hard homotopy equality |
| nominal | equality-only minimum native action |
| Official closed-form | endpoint가 아니라 \(T,H\) writer geometry로 재해석 |
| barrier 수 | 하나 |
| barrier state | spent action + conditional suffix action |
| budget | \(A_0=\widehat V_{\mathrm{suf}}(\theta_0,0)\) |
| output KL/reference facts | controller에서 제거; 평가 전용 |
| reference path envelope | 기각 |
| \(V_{\mathrm{inst}}\) | instantaneous action rate로만 사용 |
| core practical value | \(\widehat V_{\mathrm{suf}}=\frac{1}{2(1-s)}r^\top G^\dagger r\) |
| AlphaEdit | fixed hard projected basis부터 적용 |
| dynamic projector/soft escape | 후속 extension |
| solver | local equality + single convex quadratic inequality |
| finite-step authority | full-forward corrector와 actual budget verification |
| endpoint | \(s=1\), fixed target closure와 action budget 모두 필수 |
| output safety claim | 보류 |
| ODE necessity claim | strong static solve 대비 endpoint 이점 확인 전 보류 |

이 문서는 method formulation을 정의한다. 실험 순서, sample size, promotion/kill criterion은
별도 date-stamped execution plan에서 관리하며 본문 method 수식과 혼합하지 않는다.
