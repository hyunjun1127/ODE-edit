# ODE-BF Proposal

## Direct-\(z\) ODE와 동적 Layer Routing을 결합한 Transactional Knowledge Editing

- **Project:** ODE Edit
- **Method name:** **ODE-BF** — *ODE with Barrier-Filtered Dynamic Layer Routing*
- **Base editor:** AlphaEdit 계열 locate-then-edit editor
- **보존 범위:** historical edits와 pretrained/locality knowledge
- **배포 원칙:** inner ODE trajectory는 전부 virtual state에서 수행하고, 실제 model parameter는 terminal endpoint에서 정확히 한 번만 갱신

---

## 0. Proposal 요약

ODE-BF는 기존의

1. 독립적으로 direct-\(z\)를 먼저 산출하고,
2. 고정된 layer 순서와 residual 분배 규칙으로 write하는 구조

를 다음의 closed-loop flow로 대체한다.

```text
outer edit마다 하나의 direct-z ODE trajectory 초기화
                         ↓
accepted virtual waypoint에서만
현재 key / residual / layer별 Alpha proposal 재계산
                         ↓
structural H/P budget 아래에서 layer coefficient를 joint routing
                         ↓
provisional candidate에 historical / pretrained functional replay 적용
                         ↓
위반 시 persistent state를 건드리지 않고 rollback/backtracking
                         ↓
처음으로 edit efficacy와 H/P 보존을 동시에 만족하는 endpoint 선택
                         ↓
누적 low-rank update를 실제 model에 정확히 한 번 atomic commit
                         ↓
post-commit 검증 후 historical state를 정확히 한 번 append
```

최종 메커니즘은 다음 여섯 부분으로 구성된다.

### 1. Direct-\(z\) ODE

Direct target을 fixed endpoint로 두지 않고 outer edit 전체에서 유지되는 ODE state로 둔다. Target velocity는 단순 activation intervention loss만이 아니라, 실제 Alpha write를 virtual하게 통과한 뒤의 efficacy와 structural preservation risk를 보고 결정한다.

### 2. State-dependent layer proposals

Accepted waypoint마다 현재 virtual model에서 key, residual, rewrite efficiency, Alpha proposal을 다시 계산한다. 따라서 layer direction 자체가 trajectory를 따라 변한다.

### 3. Structural H/P routing

각 layer coefficient는 작은 convex QCQP로 구한다. 새 edit progress는 반드시 확보하되,

- active historical key subspace를 많이 건드리는 layer,
- pretrained key distribution에 큰 local disturbance를 주는 layer,
- cumulative load가 큰 layer

의 비중은 자동으로 줄어든다.

### 4. Functional replay filter

Covariance risk는 local structural proxy이므로 실제 보존을 보장하지 않는다. Provisional accepted candidate에서

- historical target margin CVaR,
- pretrained teacher에 대한 incremental KL

을 확인하고 위반 시 transition을 reject한다.

### 5. Transactional one-shot commit

Accepted waypoint도 deployed model과 persistent history를 변경하지 않는다. Terminal endpoint가 확정된 뒤 누적 update를 한 번만 적용하고, post-commit 검증이 성공한 뒤에만 history를 한 번 append한다.

### 6. Exact low-rank Woodbury execution

Dense historical matrix와 \(d\times d\) Alpha solve를 반복하지 않는다. Active history를 projected-key factor로 보관하고 Woodbury/Cholesky를 사용해 canonical Alpha linear system을 key-rank space에서 정확히 푼다.

핵심 claim은 단순히 ODE가 더 많은 early-stop 지점을 제공한다는 것이 아니다.

> **현재 model state가 사용 가능한 layer direction을 결정하고, preservation barrier가 각 direction에 실릴 rewrite progress를 결정하는 state-dependent target–write flow가 static direct-\(z\)와 fixed layer sharing보다 안전한 sequential edit endpoint를 찾을 수 있다.**

---

# 1. 문제 정의

Sequential edit index를 \(t\)라 하자.

- 최초 pretrained model: \(W^0\)
- edit \(t\) 직전 deployed model: \(W_{t-1}\)
- 새 edit request: \(e_t=(x_t,o_t^*)\)
- candidate rewrite layers: \(\mathcal L=\{l_1,\ldots,l_m\}\)
- 현재 유효한 committed historical edits: \(\mathcal H_t^{\mathrm{act}}\)

기존 multi-layer AlphaEdit 실행은 direct target을 먼저 구한 뒤, 정해진 layer order에 따라 residual을 나누어 write한다. 이 구조에는 네 가지 문제가 있다.

## 1.1 Target–write disconnect

Activation intervention에서는 효과적인 direct-\(z\)라도 실제 rewrite layer들이 이를 구현할 때는 큰 update norm, historical interference, pretrained drift를 요구할 수 있다.

즉,

\[
\text{semantic target feasibility}
\neq
\text{safe write feasibility}.
\]

## 1.2 Static layer allocation

동일한 residual share를 모든 edit에 적용하면 다음 차이를 반영하지 못한다.

- 현재 edit에 대한 layer별 rewrite efficiency
- 이전 edit key와의 충돌 정도
- pretrained key distribution disturbance
- layer별 누적 edit load

## 1.3 Proposal의 state dependence

한 waypoint가 virtual하게 적용되면 이후 hidden state, key, residual, layer proposal이 달라진다. 초기 model에서 만든 proposal bank를 끝까지 재사용하면 현재 trajectory의 geometry를 반영하지 못한다.

## 1.4 Dense solve와 mutable history 문제

Stock AlphaEdit를 waypoint마다 다시 호출하면 다음 비용과 오류가 발생한다.

- 거대한 \(d\times d\) matrix construction
- dense LU/solve 반복
- dense `cache_c` 유지
- rejected trial에서도 `cache_c += KK^\top`가 실행될 가능성

ODE-BF는 위 네 문제를 동시에 다루되, 실제 deployed parameter는 terminal에서 한 번만 변경한다.

---

# 2. Scope와 claim 경계

## 2.1 보존 대상

### Historical knowledge \(H\)

- 성공적으로 commit된 과거 edit
- 과거 target margin
- active historical key subspace

### Pretrained knowledge \(P\)

- fixed Wikipedia second moment가 표현하는 pretrained key distribution
- unrelated/locality calibration population에서의 pretrained-teacher incremental drift

## 2.2 의도적으로 제외하는 범위

본 proposal의 method/controller는 historical edit와 pretrained/locality preservation만 다룬다. Claim은 다음으로 제한한다.

\[
\boxed{
\text{new-edit efficacy}
+
\text{historical retention}
+
\text{pretrained/locality preservation}
}
\]

따라서 성능 향상의 원인을 외부 anchor design이 아니라 direct-\(z\)–write coupling과 dynamic layer routing에 귀속시킬 수 있다.

## 2.3 여기서 ODE가 의미하는 것

실제 model tensor를 매 step 갱신하는 것이 아니다. Continuous state는 virtual state다.

\[
S(s)=\bigl(z(s),\Delta W(s)\bigr),
\]

여기서

- \(z(s)\): direct-target state
- \(\Delta W(s)\): \(W_{t-1}\) 대비 누적 virtual low-rank update

이다.

Vector field는 다음처럼 state dependent하다.

\[
\dot z=F_z(z,\Delta W),
\]

\[
\dot W
=
\sum_{l\in\mathcal L}
 y_l^*(z,\Delta W)B_l(z,\Delta W).
\]

Constraint active set, backtracking, functional accept/reject가 바뀌므로 실제 구현은 엄밀히는 **piecewise-smooth constrained flow with discrete acceptance events**에 가깝다.

---

# 3. State와 transaction contract

## 3.1 Virtual waypoint state

Accepted waypoint \(s\)에서

\[
W_s=W_{t-1}+\Delta_s,
\qquad
S_s=(W_s,z_s),
\qquad
\Delta_0=0
\]

로 둔다.

\(W_s\)는 full model copy가 아니라 base model과 누적 low-rank hook의 합으로 평가한다.

## 3.2 Read-only static artifacts

Layer \(l\)마다 다음 artifact를 process 시작 시 한 번 load한다.

- Alpha projector \(\Pi_l\)
- pretrained non-centered second moment
  \[
  C_l^0=\mathbb E_{k\sim\mathcal D_0}[kk^\top]
  \]
- tokenizer, context template, rewrite-module metadata

Online edit path에서는 Wikipedia covariance, SVD, projector를 다시 계산하지 않는다.

## 3.3 Persistent mutable state

Persistent state는 다음을 포함한다.

- active historical solve factors
- weighted historical risk factors
- functional history metadata
- cumulative layer load
- subject–relation version metadata
- transaction ID와 history version

이 state는 terminal commit과 post-commit verification이 성공한 뒤에만 변경한다.

## 3.4 Transaction table

| Event | Virtual state | Deployed weights | Persistent history | Current field cache |
|---|---:|---:|---:|---:|
| coefficient/backtracking trial | 임시 생성 | 불변 | 불변 | 재사용 |
| rejected trial | 폐기 | 불변 | 불변 | 유지 |
| accepted nonterminal waypoint | 다음 virtual state로 이동 | 불변 | 불변 | 다음 waypoint에서 무효화 |
| terminal candidate confirmation | 임시 유지 | 불변 | 불변 | 유지 |
| terminal commit | endpoint 고정 | 정확히 1회 변경 | 아직 불변 | freeze |
| post-commit verification 성공 | 종료 | 유지 | 정확히 1회 갱신 | 삭제 |
| post-commit verification 실패 | rollback | 원복 | 불변 | 삭제 |

이 transaction 경계가 무너지면 rejected trial이 이후 edit geometry를 바꾸므로 알고리즘 자체가 달라진다.

---

# 4. Direct-\(z\) ODE

## 4.1 Outer edit당 하나의 persistent target trajectory

ODE-BF는 waypoint마다 native `compute_z`를 독립적으로 다시 실행하지 않는다. Outer edit 시작 시 target state를 한 번 초기화하고, 해당 edit가 종료될 때까지 같은 trajectory를 유지한다.

Target layer의 현재 lookup activation을

\[
z_{\mathrm{base}}
\]

라 하면 기본 초기값은

\[
z_0=z_{\mathrm{base}}
\]

이다.

Native direct-\(z\) endpoint를 초기값으로 사용하는 것은 warm-start ablation으로만 둔다. Main method에서는 target state가 fixed되지 않는다.

“direct-\(z\)를 edit당 한 번”이라는 계약은 다음을 의미한다.

- target trajectory를 한 번만 초기화
- layer마다 재시작하지 않음
- rejected trial에서 재시작하지 않음
- accepted waypoint마다 독립 direct-\(z\) solve를 수행하지 않음
- 하나의 persistent \(z_s\)를 계속 진화시킴

## 4.2 Target-only bootstrap

초기 \(z_0=z_{\mathrm{base}}\)에서는 residual이 0이므로 모든 write arm이 0일 수 있다. 따라서 coupled write field를 만들기 전에 동일한 direct-\(z\) trajectory 안에서 짧은 target-only bootstrap을 수행한다.

Activation intervention edit loss를 \(\Phi_E^{\mathrm{int}}(z)\)라 하면

\[
J_z^{\mathrm{boot}}(z)
=
\Phi_E^{\mathrm{int}}(z)
+
\lambda_z
\|z-z_{\mathrm{base}}\|_{G_z}^2
\]

이고 bootstrap velocity는

\[
v_z^{\mathrm{boot}}
=
-G_z^{-1}\nabla_zJ_z^{\mathrm{boot}}(z)
\]

이다.

Target trust radius 아래에서 projected Euler step을 적용한다.

\[
z\leftarrow z+\eta_zv_z^{\mathrm{boot}}.
\]

다음 중 하나가 만족되면 bootstrap을 종료한다.

- 적어도 한 layer에서 nonzero residual과 positive rewrite efficiency가 생성됨
- 최대 bootstrap step \(S_{\mathrm{boot}}\) 도달

Bootstrap은

- target state만 변경하고,
- weight proposal을 commit하지 않으며,
- deployed weight와 history를 변경하지 않고,
- 일반적으로 1–2 step으로 제한한다.

## 4.3 New-edit terminal objective

Canonical new-edit target margin을

\[
m_t(W)
=
\log p_W(o_t^*\mid x_t)
-
\max_{o\in\mathcal N_t}
\log p_W(o\mid x_t)
\]

로 두고

\[
h_E(W)=m_t(W)-m_E^{\min}
\]

를 정의한다.

Terminal efficacy condition은

\[
h_E(W)\ge0
\]

이다. 목적은 confidence를 무한히 높이는 것이 아니라 이 threshold를 처음 안전하게 넘는 endpoint를 찾는 것이다.

## 4.4 Write-aware target energy

현재 state \(S_s=(W_s,z_s)\)에서 routed coefficient를 \(y_s^*\)라 하자. One-step virtual write는

\[
\widehat W_s(z)
=
W_s+
\sum_l y_{s,l}^*B_l(W_s,z)
\]

이다.

Pilot에서는 \(y_s^*\)를 stop-gradient로 두고 exact Woodbury proposal을 통해 gradient를 전달한다. Stronger variant에서는 routing QCQP에 대한 implicit differentiation을 비교할 수 있다.

Target energy는

\[
\begin{aligned}
J_z(S_s)
=&\;
\Phi_E\bigl(\widehat W_s(z_s)\bigr)
+
\lambda_{zH}
\mathcal R_H^{\mathrm{str}}
\bigl(\widehat W_s(z_s)\bigr)
\\
&+
\lambda_{zP}
\mathcal R_P^{\mathrm{str}}
\bigl(\widehat W_s(z_s)\bigr)
+
\lambda_z
\|z_s-z_{\mathrm{base}}\|_{G_z}^2.
\end{aligned}
\]

중요한 점은 target gradient가 actual virtual write를 통과한다는 것이다.

\[
\frac{\partial J_z}{\partial z}
=
\frac{\partial J_z}{\partial B}
\frac{\partial B}{\partial R}
\frac{\partial R}{\partial z}
+
\text{direct target terms}.
\]

따라서 write solver가 구현하기 어렵거나 H/P risk가 큰 target direction은 target ODE 단계에서부터 억제된다.

## 4.5 Target velocity와 trust region

Preconditioned target velocity는

\[
u_{z,s}
=-G_z^{-1}\nabla_zJ_z(S_s)
\]

이다.

Target displacement를 제한하기 위해

\[
v_{z,s}
=
\arg\min_v
\frac12\|v-u_{z,s}\|_{G_z}^2
\]

subject to

\[
\|z_s+\eta_sv-z_{\mathrm{base}}\|_{G_z}
\le r_z
\]

를 푼다.

Joint transition에서

\[
z_s^{\mathrm{trial}}
=z_s+\beta_s\eta_sv_{z,s}
\]

를 사용한다. Write step이 reject되면 동일한 \(\beta_s\)를 적용한 target step도 함께 reject된다.

## 4.6 First-hit stopping

다음 조건을 처음 동시에 만족하는 waypoint에서 integration을 종료한다.

\[
h_E(W_s)\ge0,
\]

\[
D_H^{\mathrm{func}}(W_s)
\le B_H^{\mathrm{func}},
\]

\[
D_P^{\mathrm{func}}(W_s)
\le B_P^{\mathrm{func}}.
\]

즉,

\[
s^*
=
\min\left\{
 s:
 h_E(W_s)\ge0,
 D_H^{\mathrm{func}}(W_s)\le B_H^{\mathrm{func}},
 D_P^{\mathrm{func}}(W_s)\le B_P^{\mathrm{func}}
\right\}.
\]

First-hit 이후 target likelihood를 더 최적화하지 않는다. 이것이 direct-\(z\) over-optimization을 막는 주요 장치다.

---

# 5. State-dependent layer proposals

## 5.1 Accepted state에서만 field 재계산

Accepted waypoint \(S_s\)에서 모든 candidate layer에 대해 다음을 다시 계산한다.

- current key \(K_{s,l}\)
- current residual \(R_{s,l}(z_s)\)
- Alpha input-side factor \(Q_{s,l}\)
- raw low-rank proposal \(B_{s,l}\)
- rewrite efficiency \(a_{s,l}\)
- structural H/P coefficients

Rejected coefficient trial과 backtracking candidate에서는 위 field 전체를 재사용한다.

## 5.2 Raw full-residual Alpha arm

Weight orientation을

\[
W_l\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
\]

로 두고

\[
K_{s,l}\in\mathbb R^{d_{\mathrm{in}}\times b},
\qquad
R_{s,l}\in\mathbb R^{d_{\mathrm{out}}\times b}
\]

라 하자.

Layer \(l\)의 Alpha input-side system은

\[
\left[
\lambda I
+
\Pi_l
\left(
K_{s,l}K_{s,l}^{\top}
+C_{t,l}^{H,\mathrm{solve}}
\right)
\right]Q_{s,l}
=
\Pi_lK_{s,l}
\]

이다.

Raw proposal은

\[
B_{s,l}
=
R_{s,l}Q_{s,l}^{\top}
\]

이다.

Single request에서는 rank 1이고, batch size가 \(b\)이면 rank는 최대 \(b\)다.

Native처럼 residual을 \(1/(m-i)\)로 미리 나누지 않는다. 각 layer는 current full residual에 대한 arm을 만들고, 실제 share는 routing coefficient \(y_l\)가 결정한다.

## 5.3 Rewrite efficiency

New-edit loss를 \(\Phi_E\)라 하면 layer \(l\)의 first-order progress는

\[
a_{s,l}
=
\left[
-D\Phi_E(W_s)[B_{s,l}]
\right]_+
\]

이다.

JVP/VJP 또는 low-rank virtual forward로 계산할 수 있다. \(a_{s,l}=0\)인 arm은 current routing 문제에서 제외한다.

## 5.4 Joint routing

모든 layer arm은 같은 accepted state에서 계산하고 동시에 최적화한다. Shallow-to-deep, deep-to-shallow coordinate routing은 ablation으로만 둔다.

이를 통해 coefficient search가 다시 layer-order path dependence를 만드는 것을 방지한다.

---

# 6. Exact low-rank Woodbury Alpha solve

## 6.1 Historical factorization

Active historical key matrix를

\[
H_{t,l}
=
[K_{1,l}^{H},\ldots,K_{r,l}^{H}]
\]

라 하면

\[
C_{t,l}^{H,\mathrm{solve}}
=
H_{t,l}H_{t,l}^{\top}
\]

이다.

Current key와 history를 합쳐

\[
X_{s,l}=[K_{s,l},H_{t,l}]
\]

로 두면 Alpha matrix는

\[
A_{s,l}
=
\lambda I
+
\Pi_lX_{s,l}X_{s,l}^{\top}
\]

가 된다.

다음을 정의하자.

\[
Z_{s,l}=
\Pi_lX_{s,l},
\qquad
G_{s,l}=
\Pi_lK_{s,l}.
\]

\(\Pi_l\)가 symmetric idempotent projector이면

\[
X_{s,l}^{\top}\Pi_lX_{s,l}
=
Z_{s,l}^{\top}Z_{s,l}.
\]

Woodbury identity에 의해

\[
\boxed{
Q_{s,l}
=
\lambda^{-1}
\left[
G_{s,l}
-
Z_{s,l}
\left(
\lambda I+Z_{s,l}^{\top}Z_{s,l}
\right)^{-1}
Z_{s,l}^{\top}G_{s,l}
\right]
}
\]

이다.

따라서 다음이 필요 없다.

- dense \(C_l^H\)
- dense \(\Pi_lC_l^H\)
- \(d\times d\) LU factorization
- wide RHS dense solve

## 6.2 Projected-key registry

Commit된 active edit마다

\[
Z_{i,l}^{H}=
\Pi_lK_{i,l}^{H}
\]

를 terminal commit 이후 정확히 한 번 계산해 저장한다.

Alpha RHS와 small Gram은 projected historical key만으로 계산할 수 있다. Main implementation은 dense `cache_c` 대신 이 factor registry를 사용한다.

Projector exactness를 확인하기 위해

\[
\epsilon_{\mathrm{sym},l}
=
\frac{
\|\Pi_l-\Pi_l^{\top}\|_F
}{
\|\Pi_l\|_F
},
\]

\[
\epsilon_{\mathrm{idemp},l}
=
\frac{
\|\Pi_l^2-\Pi_l\|_F
}{
\|\Pi_l\|_F
}
\]

를 기록한다.

Tolerance를 넘으면 raw key도 저장하고 \(Z^{\top}Z\) 대신 정확한 \(X^{\top}Z\)를 사용하는 fallback을 둔다.

## 6.3 Cached history Cholesky

한 outer edit 동안 committed history는 immutable하다. 따라서

\[
G_{H,l}
=
\lambda I
+
(Z_{t,l}^{H})^{\top}Z_{t,l}^{H}
\]

의 Cholesky factor를 edit 시작 시 한 번 cache할 수 있다.

Accepted waypoint에서 바뀌는 것은 current key block뿐이므로 block Cholesky 또는 Schur complement로 small system을 갱신한다.

## 6.4 Full proposal materialization 제거

\[
B_{s,l}=R_{s,l}Q_{s,l}^{\top}
\]

이므로 hidden vector \(h\)에 대한 action은

\[
B_{s,l}h
=
R_{s,l}
(Q_{s,l}^{\top}h)
\]

이다.

Virtual hook은 \((R_{s,l},Q_{s,l},y_{s,l})\)만 저장한다. Trial 중 full \(d_{\mathrm{out}}\times d_{\mathrm{in}}\) update matrix를 만들지 않는다.

여러 accepted waypoint가 누적되면

\[
W_{s,l}^{\mathrm{virt}}h
=
W_{t-1,l}h
+
\sum_{j<s}
 y_{j,l}R_{j,l}
(Q_{j,l}^{\top}h)
\]

로 평가한다.

## 6.5 Exactness claim의 범위

Woodbury solve는 동일한

- projector
- current/history key
- residual
- regularization
- numerical precision

을 사용할 때 proposal이 정의한 per-layer dense Alpha system과 수학적으로 동일하다.

다만 전체 ODE-BF trajectory는 stock AlphaEdit와 동일하지 않다. ODE-BF는 의도적으로

- accepted state마다 proposal을 다시 계산하고,
- dynamic routing을 사용하며,
- obsolete active history를 제거할 수 있고,
- target와 write를 coupled하게 진화시킨다.

따라서 **exactness는 per-layer solve에 대한 claim**이다.


---

# 7. Historical structural preservation

## 7.1 Solve factor와 risk factor 분리

Alpha solve history와 routing history는 목적이 다르므로 별도로 유지한다.

### Solve factor

\[
Z_{t,l}^{H,\mathrm{solve}}
=
[\Pi_lK_{i,l}]_{i\in\mathcal H_t^{\mathrm{act}}}
\]

- active committed key를 unweighted로 보관
- exact Alpha linear system 정의
- obsolete version은 active set에서 제거

### Risk factor

\[
Z_{t,l}^{H,\mathrm{risk}}
=
\left[
\sqrt{\omega_i/Z_t}
\,\Pi_lK_{i,l}
\right]_{i\in\mathcal H_t^{\mathrm{act}}}
\]

- recency, reservoir, 중요도 weighting 적용 가능
- dynamic routing의 historical interference 계산에 사용

두 factor를 하나로 합치면

- weighted solve를 사용해 native-equivalent linear system을 잃거나,
- unweighted risk로 인해 recent/reservoir policy가 사라지는

문제가 생긴다.

## 7.2 Per-layer historical risk

Raw proposal \(B_{s,l}\)의 historical disturbance는

\[
c_{H,s,l}
=
\operatorname{tr}
\left(
B_{s,l}
C_{t,l}^{H,\mathrm{risk}}
B_{s,l}^{\top}
\right)
\]

이다.

Projected factor를 사용하면

\[
c_{H,s,l}
=
\left\|
B_{s,l}
Z_{t,l}^{H,\mathrm{risk}}
\right\|_F^2
\]

이다. Alpha input factor \(Q_{s,l}\)는 \(\Pi_l\)의 range에 있으므로

\[
Q_{s,l}^{\top}K_{i,l}
=
Q_{s,l}^{\top}\Pi_lK_{i,l}
\]

가 성립하며, historical risk 역시 projected key만으로 정확히 계산할 수 있다.

\[
B_{s,l}=R_{s,l}Q_{s,l}^{\top}
\]

이므로 실제 계산은

\[
c_{H,s,l}
=
\left\|
R_{s,l}
\left(
Q_{s,l}^{\top}
Z_{t,l}^{H,\mathrm{risk}}
\right)
\right\|_F^2
\]

로 수행한다.

Full \(B_l\) 또는 dense \(C_l^H\)를 만들 필요가 없다.

## 7.3 Cumulative historical risk

현재 step의 \(y_l^2c_{H,l}\)만 제한하면 이전 accepted virtual update와의 cross term을 놓친다.

Outer edit 시작점 대비 현재 cumulative update를 \(\Delta_{s,l}\)라 두고

\[
S_{H,s,l}
=
\Delta_{s,l}
Z_{t,l}^{H,\mathrm{risk}},
\]

\[
T_{H,s,l}
=
B_{s,l}
Z_{t,l}^{H,\mathrm{risk}}
\]

를 정의한다.

Candidate cumulative structural risk는

\[
\begin{aligned}
\mathcal R_H^{\mathrm{str}}(y)
&=
\sum_l
\|S_{H,s,l}+y_lT_{H,s,l}\|_F^2
\\
&=
 d_{H,s}
+2g_{H,s}^{\top}y
+y^{\top}M_{H,s}y
\end{aligned}
\]

이다.

여기서

\[
d_{H,s}
=
\sum_l\|S_{H,s,l}\|_F^2,
\]

\[
(g_{H,s})_l
=
\langle
S_{H,s,l},T_{H,s,l}
\rangle_F,
\]

\[
M_{H,s}
=
\operatorname{diag}
\left(
\|T_{H,s,1}\|_F^2,
\ldots,
\|T_{H,s,m}\|_F^2
\right).
\]

Main routing constraint는 이 cumulative form을 사용한다.

---

# 8. Pretrained structural preservation

## 8.1 Fixed pretrained geometry

Pretrained key distribution의 non-centered second moment를

\[
C_l^0
=
\mathbb E_{k\sim\mathcal D_0}[kk^{\top}]
\]

라 한다.

Proposal \(B_{s,l}\)의 평균 local disturbance는

\[
c_{P,s,l}
=
\operatorname{tr}
\left(
B_{s,l}C_l^0B_{s,l}^{\top}
\right)
\]

이고

\[
c_{P,s,l}
=
\mathbb E_{k\sim\mathcal D_0}
\|B_{s,l}k\|_2^2
\]

이다.

이는 임의의 소수 anchor가 아니라 pretrained key distribution 전체에 대한 평균 local disturbance approximation이다.

## 8.2 Low-rank covariance cost

\[
B_{s,l}=U_{s,l}V_{s,l}^{\top}
\]

이면

\[
\boxed{
\operatorname{tr}
(B_{s,l}C_l^0B_{s,l}^{\top})
=
\operatorname{tr}
\left[
(U_{s,l}^{\top}U_{s,l})
(V_{s,l}^{\top}C_l^0V_{s,l})
\right]
}
\]

이다.

Rank 1이면 \(C_l^0v_l\) matvec 한 번으로 계산된다. Current field에서 얻은 scalar risk는

- QCQP coefficient 변경
- line search
- backtracking
- rejected trial

동안 그대로 재사용한다.

## 8.3 Cumulative pretrained risk

Historical risk와 동일하게

\[
\mathcal R_P^{\mathrm{str}}(y)
=
 d_{P,s}
+2g_{P,s}^{\top}y
+y^{\top}M_{P,s}y
\]

로 둔다.

Main low-compute controller에서는 layer-local covariance risk를 사용하므로 \(M_{P,s}\)는 diagonal이다. Cumulative update \(\Delta_{s,l}\)도 accepted low-rank factor list로 유지하므로 \(d_{P,s}\)와 \(g_{P,s}\)는 factor Gram과 cached \(C_l^0v\) product만으로 계산하고 full \(\Delta_{s,l}\)를 materialize하지 않는다.

Stronger ablation에서는 candidate layer가 \(m\)개일 때 actuator-space Fisher/JVP Gram

\[
(M_{P,s}^{F})_{lr}
=b_{s,l}^{\top}F_0b_{s,r}
\]

을 사용할 수 있다. 이는 \(m\times m\) matrix만 필요하며 layer 간 상쇄와 증폭을 반영한다. 다만 초기 main table에는 필수로 넣지 않는다.

## 8.4 Normalization geometry 분리

Main method는 raw Alpha arm을 사용한다.

동일한 \(C_l^0\)로

\[
\widehat B_l
=
\frac{B_l}
{\sqrt{
\operatorname{tr}(B_lC_l^0B_l^{\top})
}}
\]

처럼 normalize한 뒤 다시 같은 geometry를 risk로 사용하면

\[
\operatorname{tr}
(\widehat B_lC_l^0\widehat B_l^{\top})
=1
\]

이 되어 layer별 pretrained routing signal이 사라진다.

따라서 다음 계약을 둔다.

> **Proposal normalization metric과 preservation-risk metric은 동일하게 두지 않는다.**

Numerical reparameterization이 필요하다면 \(y_l\), progress coefficient, structural coefficient를 모두 일관되게 변환해야 한다.

---

# 9. Dynamic layer routing

## 9.1 Routing objective

Layer \(l\)의 outer edit 시작 전 committed cumulative load를 \(\Omega_{t-1,l}\), 현재 virtual trajectory에서 이미 누적된 temporary load를 \(\omega_{s,l}\)라 한다.

\[
\omega_{s,l}
=
\sum_{j<s}
 y_{j,l}^2\|B_{j,l}\|_F^2.
\]

Routing cost를

\[
Q_{s,ll}
=
(1+\Omega_{t-1,l}+\omega_{s,l})
\left(
\|B_{s,l}\|_F^2+\epsilon_Q
\right)
\]

로 둔다.

Low-rank arm의 Frobenius norm은 full matrix를 만들지 않고

\[
\|B_{s,l}\|_F^2
=
\operatorname{tr}
\left[
(R_{s,l}^{\top}R_{s,l})
(Q_{s,l}^{\top}Q_{s,l})
\right]
\]

로 계산한다.

Progress constraint에 이미 \(a_{s,l}\)가 들어가므로 rewrite efficiency가 높은 layer는 같은 progress를 더 작은 \(y_l\)로 제공할 수 있다. Objective는 low-norm이며 committed/virtual load가 낮은 layer 조합을 선호한다.

## 9.2 Main convex QCQP

Accepted waypoint \(s\)에서

\[
\begin{aligned}
\min_{y\ge0}\quad&
\frac12y^{\top}Q_sy
\\
\text{s.t.}\quad&
 a_s^{\top}y\ge p_s,
\\
&
 d_{H,s}
+2g_{H,s}^{\top}y
+y^{\top}M_{H,s}y
\le B_H^{\mathrm{str}},
\\
&
 d_{P,s}
+2g_{P,s}^{\top}y
+y^{\top}M_{P,s}y
\le B_P^{\mathrm{str}},
\\
&
\sum_l
 y_l^2\|B_{s,l}\|_F^2
\le h_s^2,
\\
&
0\le y_l\le y_l^{\max}
\end{aligned}
\]

를 푼다.

Candidate layer가 4–6개라면 매우 작은 convex QCQP/SOCP다.

Main method에서 \(y_l\ge0\)를 사용하는 이유는 각 raw arm이 current residual을 줄이는 방향으로 정의되기 때문이다. Negative coefficient는 해당 arm을 역방향으로 적용해 edit progress를 훼손하거나 first-order cancellation을 악용할 수 있다. Signed routing은 별도 ablation으로만 다룬다.

## 9.3 Constraint의 의미

### Rewrite progress

\[
a_s^{\top}y\ge p_s
\]

새 edit 방향의 predicted progress를 반드시 확보한다.

### Historical structural budget

Active historical key를 많이 건드리는 arm의 비중을 줄인다.

### Pretrained structural budget

Pretrained key distribution에서 평균 disturbance가 큰 arm의 비중을 줄인다.

### Physical trust region

작은 coefficient지만 실제 matrix norm이 큰 ill-scaled arm을 막는다.

### Layer cap

한 layer가 비정상적인 scale 차이로 전체 step을 독점하는 것을 제한한다.

## 9.4 Infeasibility 처리

QCQP가 infeasible하면 다음 순서로 처리한다.

1. required progress \(p_s\) 감소
2. joint target/write step size 감소
3. 같은 accepted state에서 target direction 재조정
4. minimum step 아래까지 feasible direction이 없으면 outer edit 실패로 종료

이 과정에서 key, residual, proposal, covariance risk를 다시 계산하지 않는다. Accepted virtual state 또는 accepted target state가 바뀔 때만 field를 갱신한다.

---

# 10. Functional replay filter

Structural H/P cost는 routing proxy이며 실제 behavior 보존의 충분조건이 아니다. Functional replay는 provisional candidate를 검증하는 discrete safety filter다.

## 10.1 Active historical replay set

Functional history에는 다음 edit만 포함한다.

- outer edit가 성공적으로 terminal commit됨
- 현재 subject–relation의 active version임
- 현재 outer edit 시작점 \(W_{t-1}\)에서도 success 상태임

현재 이미 실패한 fact는 hard barrier가 아니라 repair/audit queue로 보낸다.

Benchmark held-out paraphrase는 controller에 넣지 않고 evaluation에만 사용한다.

## 10.2 Historical margin floor

Historical edit \(i\)의 margin을

\[
m_i(W)
=
\log p_W(o_i^*\mid x_i)
-
\max_{o\in\mathcal N_i}
\log p_W(o\mid x_i)
\]

라 한다.

Floor는

\[
m_i^{\mathrm{floor}}(t)
=
\max\left[
 m_H^{\min},
 \min\left(
 m_i^{\mathrm{post}},
 m_H^{\mathrm{cap}},
 m_i(W_{t-1})-
 \epsilon_H^{\mathrm{drop}}
 \right)
\right]
\]

로 둔다.

이 정의는

- 지나치게 높은 post-edit confidence 전체를 강제로 보존하지 않고,
- 현재 outer edit가 허용할 margin drop을 제한하며,
- outer-start feasibility를 유지한다.

## 10.3 Recent/reservoir CVaR

Recent와 reservoir를 한 평균으로 합치지 않고 별도 tail risk를 계산한다.

\[
D_H^{\mathrm{rec}}(W)
=
\operatorname{CVaR}_{\rho}
\left(
[m_i^{\mathrm{floor}}(t)-m_i(W)]_+
\right)_{i\in\mathcal H_t^{\mathrm{rec}}},
\]

\[
D_H^{\mathrm{res}}(W)
=
\operatorname{CVaR}_{\rho}
\left(
[m_i^{\mathrm{floor}}(t)-m_i(W)]_+
\right)_{i\in\mathcal H_t^{\mathrm{res}}}.
\]

최종 historical functional damage는

\[
D_H^{\mathrm{func}}(W)
=
\operatorname{smoothmax}
\left(
D_H^{\mathrm{rec}}(W),
D_H^{\mathrm{res}}(W)
\right)
\]

로 둔다.

\[
D_H^{\mathrm{func}}(W^{\mathrm{trial}})
>B_H^{\mathrm{func}}
\]

이면 candidate를 reject한다.

## 10.4 Obsolete target 처리

동일 subject–relation에 새 value가 commit되면 obsolete target은 다음 세 곳에서 함께 제거해야 한다.

1. functional hard history
2. historical solve factor
3. historical risk factor

Functional history에서만 제거하고 structural factor에는 남기면 새 target을 기능적으로는 허용하면서 key geometry로는 계속 막는 모순이 생긴다.

단, multi-valued relation은 단순 subject–relation 일치만으로 제거하지 않고 benchmark의 overwrite/version semantics를 따른다.

## 10.5 Pretrained functional replay

고정된 소수 사례 자체가 아니라 sampling contract를 seal한다.

- 큰 unrelated/locality calibration population
- population/version hash
- sampling seed
- prompt strata
- outer-edit/waypoint sampling schedule
- controller batch와 terminal-confirmation batch 분리

Outer edit \(t\)에서 pretrained incremental drift는

\[
\begin{aligned}
D_P^{\mathrm{func}}(W)
=
\mathbb E_x
\Bigg[
&
\operatorname{KL}
\left(
 p_{W^0}(\cdot\mid x)
 \Vert
 p_W(\cdot\mid x)
\right)
\\
&-
\operatorname{KL}
\left(
 p_{W^0}(\cdot\mid x)
 \Vert
 p_{W_{t-1}}(\cdot\mid x)
\right)
\Bigg]_+
\end{aligned}
\]

로 둔다.

이는 현재 edit가 추가한 drift만 측정하며, 이전 edit 전체를 pretrained model 방향으로 되돌리도록 강제하지 않는다.

Teacher top-\(k\) logits는 cache할 수 있다. Sample 수는 no-op/native pilot의 paired variance를 측정한 뒤 필요한 검출 해상도에 따라 정한다.

## 10.6 Compute-bounded replay schedule

| 평가 시점 | New-edit | Historical replay | Pretrained teacher-KL |
|---|---:|---:|---:|
| 모든 coefficient/backtracking trial | cheap canonical check | 수행하지 않음 | 수행하지 않음 |
| provisional accepted candidate | exact canonical margin | 수행 | 주기적 또는 uncertainty-triggered |
| first-hit terminal candidate | larger confirmation | larger confirmation | larger confirmation |
| post-commit verification | exact | exact | exact |

동일 waypoint의 모든 candidate와 backtracking에서는 같은 replay batch와 common random numbers를 사용한다.

Functional violation이 발생하면 우선 coefficient 또는 step을 backtrack한다. Field는 accepted state가 실제로 바뀐 뒤에만 다시 계산한다.

---

# 11. Joint waypoint transition

Accepted state \(S_s=(W_s,z_s)\)에서 다음 순서로 진행한다.

1. current field 계산
2. routing QCQP로 \(y_s\) 산출
3. routed virtual write를 통과해 target velocity \(v_{z,s}\) 계산
4. joint Euler candidate 생성

Write step은

\[
\Delta W_s^{\mathrm{step}}
=
\sum_l y_{s,l}B_{s,l}
\]

이다.

Backtracking coefficient를 \(\beta_s\in(0,1]\)라 하면

\[
W_s^{\mathrm{trial}}
=
W_s+
\beta_s\Delta W_s^{\mathrm{step}},
\]

\[
z_s^{\mathrm{trial}}
=
z_s+
\beta_s\eta_sv_{z,s}.
\]

Explicit Euler이므로 현재 \(z_s\)에서 계산한 field가 current write step을 만든다. Transition이 accept되면 \(W\)와 \(z\)가 함께 다음 state로 이동하고, 그때 field를 다시 계산한다.

## 11.1 Acceptance hierarchy

Candidate는 값싼 조건부터 검사한다.

1. QCQP feasibility
2. predicted progress와 structural H/P budget
3. actual canonical edit progress
4. historical functional replay
5. scheduled pretrained functional replay
6. edit threshold 도달 시 first-hit terminal confirmation

Cheap gate에서 실패한 candidate에는 functional replay 비용을 쓰지 않는다.

## 11.2 Terminal condition

Terminal candidate는 다음을 모두 만족해야 한다.

\[
m_t(W^{\mathrm{trial}})
\ge m_E^{\min},
\]

\[
D_H^{\mathrm{func}}(W^{\mathrm{trial}})
\le B_H^{\mathrm{func}},
\]

\[
D_P^{\mathrm{func}}(W^{\mathrm{trial}})
\le B_P^{\mathrm{func}},
\]

그리고 structural H/P budget과 trust region을 만족해야 한다.

가장 먼저 통과한 candidate를 endpoint로 선택한다.

---

# 12. Transactional one-shot commit

## 12.1 Commit protocol

First-hit terminal confirmation 이후 다음 순서로 commit한다.

1. terminal low-rank endpoint freeze
2. layer별 accepted factor aggregate
3. affected deployed weight만 snapshot
4. cumulative update를 실제 model에 정확히 한 번 적용
5. exact post-commit efficacy/H/P functional check
6. 실패 시 weight rollback, history 불변
7. 성공 시 persistent state를 정확히 한 번 갱신

## 12.2 History append protocol

Post-commit verification이 성공한 뒤에만 다음을 수행한다.

- final committed model에서 key 재계산
- dataset semantics에 따라 obsolete version 제거
- projected key를 `H_solve`에 append
- weighted factor와 metadata를 `H_risk`에 append
- canonical fact를 functional history에 append
- cumulative layer load \(\Omega_l\) 갱신
- transaction ID 저장

Rejected ODE trial과 accepted virtual waypoint는 stock

```python
cache_c += K @ K.T
```

경로를 절대 호출하면 안 된다.

## 12.3 Exactly-once semantics

각 outer edit에는 unique transaction ID를 부여한다. Finalization retry가 발생해도 동일 ID가 이미 commit되었다면 history를 다시 append하지 않는다.

History update는 pre-edit history version에 대한 compare-and-swap 형태로 처리한다.

---

# 13. 전체 알고리즘

```text
INPUT
  deployed model W_{t-1}
  new edit request e_t
  candidate layers L
  read-only projector Π_l and pretrained second moment C_l^0
  active projected-key registry H_solve / H_risk
  functional history
  pretrained sampling contract

BEGIN TRANSACTION
  deployed weights와 persistent-history version freeze
  Δ_0 = 0
  z_0 = current lookup activation
  virtual low-rank hooks 초기화

TARGET-ONLY BOOTSTRAP, 최대 S_boot step
  while viable nonzero layer arm이 없음:
      intervention-space edit gradient 계산
      projected target-only Euler step 수행
      deployed weight와 history는 불변
  S_boot 이후에도 viable arm이 없으면 no-op abort

FOR accepted coupled waypoint s = 0, ..., S_max - 1

  # accepted state에서만 field 생성
  current virtual key K_{s,l} 계산
  current residual R_{s,l}(z_s) 계산

  FOR each candidate layer jointly
      G_{s,l} = Π_l K_{s,l}
      exact Woodbury/Cholesky로 Q_{s,l} 계산
      B_{s,l} = R_{s,l} Q_{s,l}^T를 low-rank factor로 표현
      rewrite efficiency a_{s,l} 계산
      cumulative historical structural coefficient 계산
      cumulative pretrained structural coefficient 계산
  END

  current field cache
  small routing QCQP를 풀어 y_s 산출
  routed virtual write를 통해 direct-z velocity v_{z,s} 계산

  FOR β in {1, β0, β0^2, ...}, 최대 R_max
      cached field로 joint virtual trial 생성

      cheap canonical edit check
      실패하면 더 작은 β 또는 progress로 재시도

      provisional candidate에서 historical functional replay
      실패하면 field 재계산 없이 backtrack/reroute

      scheduled pretrained replay
      실패하면 field 재계산 없이 backtrack/reroute

      edit terminal threshold에 도달하면
          larger first-hit confirmation 수행
          통과하면 COMMIT으로 이동

      terminal이 아니지만 feasible하면 virtual waypoint accept
      (Δ_s, z_s)를 virtual하게 갱신
      field cache invalidate
      다음 accepted waypoint로 이동
  END

  어떤 trial도 accept되지 않으면
      deployed model/history를 건드리지 않고 infeasible abort
END

COMMIT
  accumulated low-rank update를 실제 model에 한 번 materialize
  post-commit verification 수행

  실패:
      deployed weight rollback
      history append 없음

  성공:
      obsolete active version 제거
      final projected key 정확히 한 번 append
      functional history 정확히 한 번 append
      layer load 정확히 한 번 갱신
      transaction finalize
```


---

# 14. 계산량과 메모리 설계

## 14.1 Naive waypoint-wise AlphaEdit가 무거운 이유

현재 Alpha solve의 down-projection input dimension은 대략 다음과 같다.

- Llama3-8B: \(d=14{,}336\)
- Qwen2.5-7B: \(d=18{,}944\)

Dense 구현을 waypoint마다 반복하면 layer마다 다음 비용을 다시 지불한다.

- \(\Pi_l(KK^{\top}+C_H)\) dense construction
- \(d\times d\) factorization
- wide RHS solve
- dense mutable historical matrix update

Accepted waypoint가 증가할수록 비용이 거의 선형으로 누적되고, 각 solve 자체가 매우 크다.

## 14.2 Factorized historical memory

FP32, single-request edit당 context-averaged key 1개, active historical key 100개, candidate layer 5개를 기준으로 projected-key factor memory는 다음과 같다.

| Model | Factorized history | Dense \(C_H\) |
|---|---:|---:|
| Llama3-8B | 약 27.3 MiB | 약 3.83 GiB |
| Qwen2.5-7B | 약 36.1 MiB | 약 6.68 GiB |

주의할 점은 이 수치가 **historical state memory만** 의미한다는 것이다.

- dense static projector \(\Pi_l\)
- dense pretrained moment \(C_l^0\)

를 GPU에 상주시킬 경우 해당 memory는 별도다.

Projected key만 저장하면 위 factor memory를 유지할 수 있다. Projector numerical error 때문에 raw key fallback까지 저장하면 historical factor memory는 대략 두 배가 된다.

## 14.3 Per-accepted-field complexity

다음을 두자.

- current request rank: \(b\)
- active history rank: \(r\)
- input dimension: \(d\)
- candidate layer 수: \(m\)

Dense \(\Pi_l\)를 사용할 때 accepted field마다 current projection은

\[
O(d^2b)
\]

이다.

History Cholesky를 cache하면 Woodbury small solve와 reconstruction은 대략

\[
O(drb+r^2b+b^3)
\]

이다.

즉 dense \(O(d^3)\) factorization은 제거되지만, dense projector matvec는 남는다.

Pretrained structural risk는 low-rank proposal factor마다 \(C_l^0v\) matvec를 추가한다. 이 값도 accepted field에서 한 번만 계산하고 모든 line search와 backtracking에서 재사용한다.

## 14.4 Rejected trial의 비용 계약

Rejected trial에서는 다음을 다시 계산하지 않는다.

- key extraction
- residual extraction
- \(\Pi_lK_l\)
- Woodbury solve
- \(C_l^0v\)
- structural H/P coefficient
- persistent history

Rejected trial이 지불하는 비용은 다음으로 제한한다.

- 작은 QCQP 재해결 또는 coefficient scaling
- low-rank virtual forward
- cheap canonical edit check
- cheap gate를 통과한 경우에만 functional replay

## 14.5 남는 실제 병목

Exact low-rank conversion 이후 예상되는 online bottleneck은 다음이다.

1. accepted state에서의 dense \(\Pi_lK_{s,l}\) matvec
2. accepted state에서의 \(C_l^0v\) matvec
3. write-aware target backward
4. historical/pretrained functional replay 빈도

따라서 “Wikipedia covariance 사용 자체”가 병목인 것은 아니다. 병목은 static artifact를 이용하는 accepted-state matvec와 functional evaluation이다.

## 14.6 Compute contract

Main implementation은 다음 upper bound를 명시해야 한다.

- target-only bootstrap 최대 step \(S_{\mathrm{boot}}\)
- accepted coupled waypoint 최대 수 \(S_{\max}\)
- field당 backtracking 최대 수 \(R_{\max}\)
- accepted field당 target backward 최대 1회
- rejected trial에서 field recomputation 0회
- historical replay는 provisional accepted candidate에서만
- pretrained teacher-KL은 periodic/terminal에서만
- dense \(C_H\), full \(B_l\), full trial-model copy 생성 금지

## 14.7 필수 compute logging

단순히 “ODE step 수”만 기록하면 실제 비용을 비교할 수 없다. Edit마다 다음을 기록해야 한다.

\[
\begin{aligned}
&N_{\mathrm{boot\_backward}},\\
&N_{\mathrm{target\_backward}},\\
&N_{\mathrm{accepted\_field}},\\
&N_{\mathrm{rejected\_trial}},\\
&N_{\Pi K},\\
&N_{C^0v},\\
&N_{\mathrm{small\_chol}},\\
&N_{H\text{-replay}},\\
&N_{P\text{-replay}},\\
&\text{wall-clock/edit},\\
&\text{peak GPU/CPU memory},\\
&r_{\mathrm{active}}.
\end{aligned}
\]

Accepted field recomputation과 coefficient backtracking을 분리해 보고해야 한다.

---

# 15. 실험 설계

## 15.1 Core main table

| Method | Target | Proposal | Layer routing | Functional filter | Commit |
|---|---|---|---|---|---|
| Native AlphaEdit | fixed native direct-\(z\) | native sequential | fixed residual sharing | 없음 | native |
| Native AlphaEdit-WB | fixed native direct-\(z\) | native와 동일 | native와 동일 | 없음 | native |
| Fixed-\(z\) Dynamic Routing | fixed native direct-\(z\) | state-dependent | structural H/P QCQP | H/P replay | one-shot |
| ODE-\(z\) + Static Sharing | direct-\(z\) ODE | state-dependent | equal/fixed sharing | H/P replay | one-shot |
| ODE-Alloc v1 ablation | fixed target | fixed basis | adaptive coefficient | H/P replay | one-shot |
| **ODE-BF Full** | write-aware direct-\(z\) ODE | state-dependent | structural H/P QCQP | H/P replay | transactional one-shot |

`Native AlphaEdit-WB`는 solver replacement의 계산량 효과만 분리하기 위한 필수 baseline이다. Target, residual sharing, layer order, history를 native와 동일하게 고정했을 때 numerical tolerance 내에서 같은 endpoint를 재현해야 한다.

## 15.2 Mechanism ablations

### Target 관련

- target-only ODE bootstrap vs native direct-\(z\) warm start
- fixed direct-\(z\) vs evolving direct-\(z\)
- intervention-only target gradient vs write-aware target gradient
- routing stop-gradient vs implicit differentiation
- first-hit stopping vs fixed waypoint count

### Proposal/routing 관련

- accepted state마다 proposal 재계산 vs initial proposal 고정
- joint routing vs shallow-to-deep coordinate routing
- joint routing vs deep-to-shallow coordinate routing
- cumulative structural risk vs step-only risk
- cumulative layer load penalty 제거

### Preservation 관련

- historical structural risk 제거
- pretrained structural risk 제거
- historical functional replay 제거
- pretrained functional replay 제거
- diagonal covariance risk vs small Fisher/JVP Gram

### Transaction/compute 관련

- dense Alpha solve vs exact Woodbury
- field cache 재사용 vs 모든 trial 재계산
- virtual low-rank hook vs trial full materialization

## 15.3 Primary metrics

### New edit

- canonical efficacy
- target margin
- held-out paraphrase generalization
- target-only bootstrap step 수
- first hit까지 accepted coupled waypoint 수
- terminal infeasibility rate

### Historical retention

- historical success rate
- recent history success
- reservoir history success
- CVaR margin deficit
- worst-decile margin drop
- obsolete version 처리 정확도

### Pretrained/locality preservation

- incremental teacher KL
- unrelated factual/locality accuracy
- neighborhood/locality change rate
- structural pretrained covariance risk

### Update geometry

- layer별 \(y_l\)
- layer별 raw proposal norm
- layer별 rewrite efficiency \(a_l\)
- layer별 \(c_{H,l}\), \(c_{P,l}\)
- cumulative layer load
- target displacement \(\|z_s-z_{\mathrm{base}}\|_{G_z}\)

### Compute

- target backward NFE
- accepted field count
- rejected/backtracking count
- projector/covariance matvec count
- historical/pretrained replay NFE
- wall-clock/edit
- peak GPU/CPU memory
- active key rank

## 15.4 Matched-compute comparison

ODE-BF와 generic penalty optimizer를 동일한 coefficient state와 동일한 field 위에서 비교한다.

최소한 다음 두 기준을 모두 맞춰야 한다.

- accepted field evaluation 수
- total forward/backward NFE 또는 wall-clock

Generic optimizer가 같은 compute에서 동일한 efficacy–preservation frontier를 보이면 contribution은 ODE 자체보다 constrained dynamic routing으로 재해석해야 한다.

## 15.5 권장 pilot configuration

초기 pilot은 다음 범위가 적절하다.

- candidate layer: 4–6개
- sequential stream: 10, 50, 100 edits
- \(S_{\mathrm{boot}}\): 1–2
- \(S_{\max}\): 8 또는 10
- \(R_{\max}\): field당 3
- target backward: accepted field당 최대 1회
- historical replay: recent 32 + reservoir 32, 가능한 범위에서
- CVaR tail: \(\rho\in\{0.1,0.2\}\)
- pretrained replay: accepted waypoint 2회마다 small rotating batch + terminal larger batch
- exactness linear algebra: FP32
- held-out paraphrase: evaluation only

Pretrained replay sample 수는 no-op/native paired variance를 측정한 뒤 정한다.

---

# 16. 필수 implementation invariants

## 16.1 Woodbury–dense equivalence

작은 model 또는 축소 dimension에서

\[
\frac{
\|Q_l^{\mathrm{WB}}-Q_l^{\mathrm{dense}}\|_F
}{
\|Q_l^{\mathrm{dense}}\|_F
}
\]

를 측정한다.

Full model에서는 항상 linear residual을 기록한다.

\[
\frac{
\|A_lQ_l^{\mathrm{WB}}-
\Pi_lK_l\|_F
}{
\|\Pi_lK_l\|_F
}.
\]

## 16.2 Native-Woodbury endpoint equivalence

Target, residual sharing, layer order, history를 native와 동일하게 두었을 때 `Native AlphaEdit-WB`가 native endpoint를 numerical tolerance 내에서 재현해야 한다.

이 검증 전에는 “exact low-rank replacement” claim을 사용하지 않는다.

## 16.3 Dense materialization 금지

Trial 중 다음 tensor가 생성되지 않는지 assertion을 둔다.

- dense \(C_l^H\)
- dense \(\Pi_lC_l^H\)
- full \(B_l\)
- full copied trial model

## 16.4 Rejected trial purity

Rejected trial 전후 다음 hash/version이 동일해야 한다.

- deployed weights
- solve/risk history factors
- functional history
- cumulative layer load
- transaction state
- sampling schedule state

## 16.5 Exactly-once history update

Successful outer edit 하나에 대해 다음이 성립해야 한다.

- transaction ID 1개
- active version append 1회
- obsolete version removal 1회 이하
- projected factor append 1회
- finalization retry 시 중복 append 0회

## 16.6 Field cache identity

Cache key에는 최소한 다음이 포함되어야 한다.

```text
outer_edit_id
accepted_waypoint_id
virtual_state_version
target_state_version
layer_id
proposal_version
history_version
```

Rejected trial은 cache key를 바꾸지 않는다. Accepted transition만 새로운 field version을 만든다.

## 16.7 Virtual endpoint와 committed endpoint 일치

Commit 전 virtual prediction과 commit 후 actual model output의 차이를 기록한다.

- new-edit logits
- historical replay margins
- pretrained replay KL
- affected layer weight delta

차이가 tolerance를 넘으면 low-rank hook 또는 materialization bug로 처리하고 history를 append하지 않는다.

## 16.8 Structural proxy calibration

다음을 edit/waypoint별로 log한다.

- predicted historical risk
- actual historical margin damage
- predicted pretrained risk
- actual incremental teacher KL

ODE-BF의 mechanism claim을 위해서는 최소한 다음 경향이 보여야 한다.

\[
\text{high }c_H
\Rightarrow
\text{larger historical damage tendency},
\]

\[
\text{high }c_P
\Rightarrow
\text{larger pretrained drift tendency}.
\]

Structural risk가 functional damage와 무관하면 routing signal로서 의미가 없다.

---

# 17. 예상 실패 모드와 대응

## 17.1 Initial target field가 형성되지 않음

**증상:** \(S_{\mathrm{boot}}\) 이후에도 모든 layer arm의 residual 또는 rewrite efficiency가 0에 가까움.

**대응:**

- bootstrap step size와 target trust radius 점검
- target token loss gradient 점검
- target-layer intervention 위치 점검
- native direct-\(z\) warm-start fallback을 명시적 ablation으로 사용

## 17.2 Structural risk가 functional damage를 예측하지 못함

**증상:** covariance risk는 낮지만 historical replay나 pretrained KL가 크게 악화됨.

**대응:**

- key token 위치와 context aggregation 일치 여부 확인
- functional replay 빈도 증가
- small Fisher/JVP Gram ablation
- structural budget을 더 보수적으로 설정

## 17.3 Target ODE가 write-infeasible region으로 이동

**증상:** target step 이후 QCQP infeasibility가 반복됨.

**대응:**

- \(J_z\)의 H/P structural term 강화
- target trust radius 축소
- 마지막 feasible target state로 rollback
- write-aware gradient와 intervention-only gradient 비교

## 17.4 Layer collapse

**증상:** 거의 모든 edit에서 한 layer가 \(y_l\approx y_l^{\max}\)를 차지함.

**대응 순서:**

1. raw proposal scale와 \(a_l\) 계산 검증
2. same-geometry normalization 여부 확인
3. cumulative load가 실제로 반영되는지 확인
4. 그 뒤에만 layer cap 또는 약한 diversification 추가

실제 edit별 layer specialization일 수 있으므로 entropy regularization을 처음부터 넣지 않는다.

## 17.5 Active history rank 증가

**증상:** long stream에서 small Gram과 factor memory가 커짐.

**대응:**

Main exact experiment에서는 benchmark horizon 전체 active factor를 유지한다. Rank compression, sketching, reservoir solve는 별도 long-horizon approximation으로 분리하고 exact라고 부르지 않는다.

## 17.6 Dense pretrained matvec가 병목

**증상:** \(C_l^0v\) 시간이 accepted field의 대부분을 차지함.

**대응:**

- layer별 streaming
- repeated factor cache
- CPU/GPU overlap
- fixed low-rank eigenspace approximation을 명시적 ablation으로 비교

Main exact covariance result를 조용히 approximation으로 바꾸지 않는다.

## 17.7 Functional replay가 총비용을 지배

**증상:** replay forward가 field construction보다 많음.

**대응:**

- rejected coefficient trial에서는 replay 생략
- historical replay는 provisional accepted candidate에서만
- pretrained KL은 periodic/terminal로 제한
- common random numbers와 teacher logits cache 사용

## 17.8 Woodbury numerical instability

**증상:** small Gram condition number 증가, Cholesky failure, linear residual 증가.

**대응:**

- FP32 Gram/Cholesky
- adaptive jitter
- condition number logging
- QR/SVD small solve fallback

Fallback도 동일 linear system을 풀어야 한다.

## 17.9 Functional filter만 성능을 만들고 routing은 무의미함

**증상:** structural routing을 제거해도 replay-based trial selection만으로 같은 성능.

**해석:** ODE-BF가 preservation-aware routing이 아니라 expensive candidate rejection method로 동작하는 것.

**필수 비교:** equal sharing + 동일 replay filter와 full ODE-BF를 직접 비교한다.

---

# 18. Go/No-Go 판정 기준

## 18.1 Technical Go

다음이 모두 성립해야 한다.

1. Woodbury와 dense per-layer solve equivalence 확인
2. rejected trial의 persistent mutation 0건
3. virtual endpoint와 committed endpoint 일치
4. history append exactly once
5. accepted state에서만 field recomputation
6. terminal replay schedule 재현 가능

하나라도 실패하면 scientific comparison 전에 구현을 수정한다.

## 18.2 Scientific Go

Matched new-edit efficacy에서 full ODE-BF가 다음을 보여야 한다.

- Native AlphaEdit보다 높은 historical retention
- Native AlphaEdit보다 낮은 pretrained/locality incremental drift
- Fixed-\(z\) Dynamic Routing보다 개선
- ODE-\(z\) + Static Sharing보다 개선
- Fixed proposal bank보다 state-dependent proposal recomputation의 이점
- matched-NFE generic coefficient optimizer보다 우수하거나 더 낮은 violation rate

## 18.3 결과별 해석

### Woodbury만 성공

`Native AlphaEdit-WB`는 빠르고 exact하지만 full method가 보존 성능을 개선하지 못하면, solver optimization은 유효하되 ODE-BF mechanism은 지지되지 않는다.

### Fixed-\(z\) routing과 동일

Direct-\(z\) ODE가 불필요하고 dynamic routing이 핵심이다.

### ODE-\(z\) + static sharing과 동일

Dynamic routing이 불필요하고 target construction이 핵심이다.

### Functional replay 제거 시만 붕괴

Structural proxy가 충분히 강하지 않으며 method의 실제 안전성은 replay filter에 의존한다.

### Generic penalty optimizer와 동일

ODE-specific claim을 줄이고 constrained state-dependent routing으로 framing한다.

### State-dependent recomputation 이점 없음

Initial proposal bank를 재사용하는 저비용 ODE-Alloc 계열이 더 적절할 수 있다.

---

# 19. 최종 proposal statement

ODE-BF는 한 번의 knowledge edit을 fixed direct target과 정적 residual sharing으로 보지 않는다. 대신 다음 closed loop로 정의한다.

\[
\boxed{
\text{current virtual model}
\rightarrow
\text{current direct-}z
\rightarrow
\text{current Alpha layer arms}
\rightarrow
\text{structural H/P routing}
\rightarrow
\text{functional H/P replay}
}
\]

Accepted waypoint에서만 model state, target state, key, residual, proposal이 갱신된다. Rejected trial에서는 같은 field와 risk를 재사용하고 deployed model과 history는 완전히 불변이다.

High-dimensional Alpha system은 dense하게 풀지 않는다. Active historical knowledge를 projected-key factor로 유지하고, exact Woodbury/Cholesky solve로 per-layer arm을 계산한다. Historical/pretrained structural risk 역시 low-rank factor에서 직접 계산한다.

최종 endpoint는

- new edit threshold,
- historical functional budget,
- pretrained functional budget,
- structural H/P budget

을 처음 동시에 만족하는 first hit이다. 그 전까지 모든 state는 virtual이며, terminal endpoint만 실제 model에 정확히 한 번 commit된다.

최종 방법은 다음으로 요약된다.

\[
\boxed{
\text{Direct-}z\text{ ODE}
+
\text{State-dependent layer proposals}
+
\text{Structural H/P routing}
+
\text{Functional replay filter}
+
\text{Transactional one-shot commit}
+
\text{Exact low-rank Woodbury solve}
}
\]

본 proposal을 main research direction으로 두고, 기존 fixed-target ODE-Alloc은 coefficient allocation contribution만 분리하는 component ablation으로 유지한다.
