# Fixed-z Functional Safe Write for Knowledge Editing

## 2026-08-30 Research Reset Proposal

> **문서 상태 변경 — 2026-08-31**
>
> 이 문서는 functional-anchor/KL controller branch의 historical proposal로 보존한다.
> Current primary method는
> [FzCB-Edit method pivot proposal](2026-08-31-fzcb-edit-method-pivot-proposal.md)이다.

| 항목 | 내용 |
|---|---|
| 문서 상태 | Historical FCW branch; 2026-08-31 FzCB pivot으로 superseded |
| 기준일 | 2026년 8월 30일 |
| 연구 단계 | Research reset / falsification-first |
| 1차 적용 | MEMIT write stage |
| 2차 적용 | AlphaEdit projected native basis |
| 고정 대상 | 기존 editor가 산출한 direct target \(z^\star\) |
| 연구 대상 | 동일 target을 실현하는 weight write의 functional safety |
| 초기 method 명칭 | Functional-Constrained Write, 약칭 FCW |
| 조건부 승격 명칭 | CBF-ODE-Write; ODE necessity gate 통과 후에만 사용 |

> **현재의 핵심 판단**
>
> 단순히 closed-form weight update를 여러 Euler step으로 나누는 것은 새로운 방법이
> 아니다. 연구 가치가 생기려면 동일한 \(z^\star\)를 만족하는 여러 weight realization이
> 실제로 서로 다른 functional damage를 만들고, key/null-space geometry만으로는 그 차이를
> 설명하지 못하며, state-dependent functional feedback을 사용한 multi-step writer가 동일한
> compute의 static constrained solver보다 더 좋은 frontier를 보여야 한다.

---

## 0. 연구 리셋 선언

이 문서는 2026-08-30 이전의 BF-ODE, fixed-E8, cold/warm target, dynamic layer routing,
functional-P, capacity barrier 관련 결과를 새 방법의 근거로 자동 승계하지 않는다.
이전 결과와 구현은 다음 역할로만 보존한다.

- 구현 및 transaction 자산의 재사용 가능성 확인
- 새로운 가설을 반증하는 negative evidence
- 동일 실험을 반복하지 않기 위한 역사 기록
- 새 method와 비교할 legacy arm

다음 주장은 현재 모두 보류한다.

- ODE가 knowledge editing에 필요하다는 주장
- fixed Euler step 자체가 성능을 개선한다는 주장
- first-hit point가 자연스럽거나 최적인 endpoint라는 주장
- functional-P 또는 임의 threshold가 model preservation을 보장한다는 주장
- CBF의 continuous-time 조건이 finite-precision discrete write의 safety를 보장한다는 주장
- 동일 direct-\(z\)가 실제로 유의미한 safe weight 자유도를 제공한다는 주장
- long-horizon 또는 lifelong superiority 주장

새 연구는 아래 세 질문을 순서대로 통과해야 한다.

```text
same-z endpoint multiplicity
  -> functional signal의 key-space 대비 추가 가치
  -> multi-step feedback의 static constrained solve 대비 추가 가치
```

앞 단계가 실패하면 이후 단계는 실행하지 않는다.

---

# 1. 연구 질문과 한 문장 가설

## 1.1 중심 연구 질문

Locate-then-edit editor가 산출한 동일한 semantic target \(z^\star\)에 대해,
target activation residual을 만족하는 여러 weight update가 존재할 때 실제 model output
기능을 가장 적게 훼손하는 realization을 어떻게 선택할 것인가?

## 1.2 조건부 중심 가설

> 동일한 \(z^\star\), target residual, update action, residual allocator, edit-time information,
> compute budget을 맞춘 조건에서, current model의 functional preservation contract를 매
> accepted step 재관측하여 nominal MEMIT/AlphaEdit write를 최소 개입으로 수정하면,
> key-space-only writer가 구별하지 못하는 safer weight realization을 선택할 수 있다.

## 1.3 ODE 가설은 종속 가설이다

ODE는 연구 출발점이 아니라 다음 조건부 가설이다.

> Functional constraints, edit keys, residuals 또는 usable write directions이 intermediate
> model state에 따라 유의하게 변할 때, multi-step relinearized feedback은 동일한
> functional constraints를 한 번만 선형화해 푸는 static solver보다 좋은
> efficacy--preservation--compute frontier를 만들 수 있다.

이 가설이 실패하면 FCW는 static functional constrained writer로 남고, ODE를 paper의
핵심 기여로 주장하지 않는다.

---

# 2. 적용 위치: direct-z가 아니라 W write stage

## 2.1 기존 locate-then-edit의 두 단계

MEMIT/AlphaEdit 계열은 대략 다음 두 단계로 나뉜다.

1. edit request가 원하는 semantic target activation \(z_i^\star\)를 Adam 등으로 최적화한다.
2. 여러 editable layer의 weight를 변경하여 해당 target을 model에 기록한다.

Direct-\(z\) 단계는 이미 iterative optimization, norm clamp, KL anchor 등을 포함한다. 또한
downstream write feasibility를 target optimization에 되먹임하는 방향은
[MetaKE](https://arxiv.org/abs/2603.12677)가 직접 다룬다. 따라서 본 연구의 주 방법은
\(z\)-space ODE가 아니다.

## 2.2 고정되는 것과 변화할 수 있는 것

본 연구에서 한 edit request 동안 다음은 고정한다.

- editor와 seed
- direct target \(z_i^\star\)
- target을 계산할 때 사용한 rewrite contexts
- editable layer 집합
- baseline별 residual allocation policy
- controller가 접근할 수 있는 prompt와 anchor bank

다음은 accepted write 뒤 재관측할 수 있다.

- current edit key와 hidden activation
- remaining target residual
- native low-rank proposal
- functional constraint value와 active set
- directional Jacobian과 local trust model

## 2.3 같은 z가 W endpoint를 고정하지 않는다

한 linear memory layer에서 edit key matrix \(K_E\)와 residual \(R_E\)가 주어지면 exact
local write 조건은

\[
\Delta K_E=R_E
\]

이다. 일관된 문제에서 일반해는

\[
\Delta(U)
=
R_EK_E^\dagger
+
U(I-K_EK_E^\dagger)
\]

이며

\[
\Delta(U)K_E=R_E.
\]

여기서 consistency는

\[
R_E=R_EK_E^\dagger K_E
\]

를 뜻한다. 이 조건이 실패한 request를 null-space 자유도 실험에 포함하지 않고, 별도의
typed algebraic infeasibility로 기록한다.

따라서 \(U\)가 만드는 edit-key null component는 target activation을 바꾸지 않으면서
다른 context의 동작을 바꿀 수 있다. 그러나 이 자유도가 실제 LLM에서 유의미한 functional
차이를 만드는지는 수학적 사실이 아니라 실험 가설이다. Gate 1에서 먼저 검증한다.

---

# 3. barrier 개념의 정리

## 3.1 서로 다른 세 개념

본 문서에서는 다음을 명시적으로 구분한다.

| 이름 | 정의와 역할 | 본 연구에서의 지위 |
|---|---|---|
| Path loss barrier | endpoint 보간 대비 trajectory 중간의 excess loss | ODE-M 분석 대상; 본 방법의 safety 정의가 아님 |
| Control Barrier Function, CBF | \(h(\theta)\ge0\) safe set의 forward invariance를 위한 velocity constraint | candidate continuous-time controller |
| Interior log barrier | \(-\sum_j\log h_j\)를 constrained objective에 넣는 수치 최적화 장치 | optional solver baseline; canonical하지 않음 |

[ODE-M](https://arxiv.org/html/2605.19409v1)은 full path가 target endpoint로 수렴하는
조건과 operating time을 분리한다. [ODESteer](https://arxiv.org/html/2602.17560)는
activation density ratio의 gradient로 state-dependent steering field를 만든다. 두 논문의
barrier와 본 연구의 functional preservation contract는 동일한 객체가 아니다.

## 3.2 log barrier를 canonical이라고 부르지 않는다

Safe set이

\[
\mathcal S=\bigcap_j\{\theta:h_j(\theta)\ge0\}
\]

로 주어졌다고 해도

\[
\phi(\theta)=-\sum_j\log h_j(\theta)
\]

는 고정된 inequality 목록에 대한 표준적 선택일 뿐, 집합 \(\mathcal S\)가 유일하게
정하는 함수가 아니다. Redundant constraint, group duplication, weighting, equivalent
reparameterization에 따라 central path와 analytic center가 달라진다.

[Boyd and Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
도 동일한 feasible set을 서로 다른 inequality 목록으로 표현하면 analytic center가 달라질
수 있음을 명시한다.

따라서 본 연구의 표현은 다음으로 제한한다.

> Preservation contract가 지킬 집합을 정의한다. Contract, local metric, rate function을
> 사전 등록하면 nominal velocity에 대한 minimum-intervention projection이 유도된다.

Contract 자체, budget, metric, aggregation까지 무선택적으로 생긴다고 주장하지 않는다.

## 3.3 analytic center를 endpoint로 사용하지 않는다

\(E_\tau\cap\mathcal S\)의 analytic center는 다음 이유로 자연스러운 edit endpoint가 아니다.

- constraint 표현과 중복 anchor에 의존한다.
- 시작 weight \(\theta_0\)에서 최소한으로 이동한다는 preference가 없다.
- transformer functional set은 weight에 대해 비볼록이고 disconnected할 수 있다.
- target manifold의 null directions 때문에 feasible set이 unbounded할 수 있다.
- AlphaEdit equality subspace는 ambient space에서 strict interior가 없다.
- exact target \(\tau=0\)이면 target inequality의 log-barrier domain이 비어 있을 수 있다.

Log-barrier central path는 제한된 local convex surrogate의 solver baseline으로만 허용한다.

---

# 4. 문제 정의

## 4.1 State와 target

편집되는 모든 layer weight를 묶어

\[
\theta=(W_l)_{l\in\mathcal L},
\qquad
\theta(0)=\theta_0
\]

로 둔다. Edit \(i\)의 fixed target은 \(z_i^\star\), current model에서 측정하는 target map은
\(F_i(\theta)\)다.

Per-edit target energy는

\[
E_i(\theta)
=
\frac12
\|F_i(\theta)-z_i^\star\|_{\Sigma_i^{-1}}^2.
\]

배치 전체를 하나의 scalar \(E=\sum_iE_i\)로만 제어하지 않는다. 어려운 edit 하나가 쉬운
edit에 가려지거나 반대로 전체 controller를 지배하지 않도록 per-edit constraint,
epigraph max 또는 preregistered group-CVaR를 사용한다.

## 4.2 Functional preservation contract

Controller anchor \(x_j\)에 대해

\[
D_j(\theta)
=
D_{\mathrm{KL}}
\left(
p_{\theta_0}(\cdot\mid x_j)
\Vert
p_{\theta}(\cdot\mid x_j)
\right)
\]

또는 preregistered logit/margin drift를 사용한다. Safe slack은

\[
h_j(\theta)=\epsilon_j-D_j(\theta)
\]

이며 contract는

\[
h_j(\theta)\ge0
\]

이다.

보존 단위는 다음 중 하나로 사전 등록한다.

- individual anchor constraint
- semantic group별 max 또는 CVaR
- general capability sentinel group
- active previous-edit margin group

평균 KL 하나로 worst-case preservation을 주장하지 않는다.

## 4.3 Contract budget calibration

\(\epsilon_j\)는 held-out 성능을 보며 조정하지 않는다. Calibration bank에서 다음 분포를
측정해 정한다.

- zero-write repeated evaluation floor
- authoritative execution dtype의 virtual/commit numerical drift
- benign paraphrase 또는 formatting perturbation drift
- native editor의 dev-only reference distribution

Raw slack은 unit과 noise scale이 다르므로 필요하면

\[
\bar h_j
=
\frac{\epsilon_j-D_j}{\sigma_j+\delta}
\]

로 정규화한다. \(\sigma_j\)와 \(\delta\)도 calibration-only로 고정한다.

초기 Gate 0--2의 authoritative execution dtype은 `FULL_FP32`로 고정한다. BF16 또는 다른
저정밀 deployment는 별도 사전등록된 study에서만 허용하며, 그 결과를 FULL-FP32 gate와
혼합하지 않는다.

## 4.4 네 개의 데이터 bank

| Bank | 역할 | Controller 접근 |
|---|---|---|
| \(Q_{\mathrm{ctrl}}\) | active constraint와 candidate acceptance | 허용 |
| \(Q_{\mathrm{cal}}\) | \(\epsilon\), normalization, method-common hyperparameter 결정 | training 전/개발 단계만 허용 |
| \(Q_{\mathrm{gate}}\) | Gate 1--4의 Go/No-go와 architecture selection | controller에는 금지; gate 판정에만 허용 |
| \(Q_{\mathrm{audit}}\) | 최종 held-out functional 평가 | 금지 |

동일 prompt 또는 paraphrase family가 여러 bank에 들어가지 않도록 source identity와 hash를
기록한다. \(Q_{\mathrm{gate}}\)를 한 번이라도 보고 method, arm, threshold, step 수 또는 실행
계속 여부를 정하면 해당 bank는 최종 audit으로 재사용할 수 없다. \(Q_{\mathrm{audit}}\)는
architecture, hyperparameter, endpoint rule과 모든 Gate 결정이 freeze된 뒤 정확히 한 번
평가하며, 그 결과를 보고 method를 고치거나 추가 subset을 선택하지 않는다.

## 4.5 Supersession-aware active ledger

시점 \(t\)에서 유효한 fact contract를

\[
\mathcal A_t
=
\{(subject,relation,value,version):\text{currently valid}\}
\]

로 둔다. 같은 \((subject,relation)\)의 새 version이 들어오면 이전 target은 다음에서
삭제한다.

- past-edit target margin
- history key set \(K_p\)
- functional replay target
- active edit-retention denominator

단, 같은 subject의 unrelated relation과 general neighborhood probe는 유지한다. 평가에서는
current-version success, obsolete-value suppression, non-superseded retention을 분리한다.
Repeated overwrite benchmark 자체는
[sLKE/ARM](https://aclanthology.org/2025.acl-long.1492/)이 이미 제안했으므로 novelty로
주장하지 않는다.

---

# 5. Endpoint와 trajectory의 역할

## 5.1 Online method의 endpoint claim

FCW의 online endpoint는 target tube와 functional contract를 만족하는 trajectory event다.

\[
E_i(\theta_T)\le\tau_i,
\qquad
h_j(\theta_T)\ge0.
\]

이 endpoint를 analytic center, global optimum 또는 canonical realization이라고 부르지 않는다.
Vector field, initial state, active-set rule, integration tolerance가 고정된 deterministic
operating point로만 해석한다.

First-hit을 사용할 경우 다음을 필수 보고한다.

- step size와 solver에 따른 endpoint convergence
- accepted/rejected step 수
- event overshoot
- terminal target residual
- terminal functional slack
- trajectory 중 best-feasible point와 last point의 차이

## 5.2 Offline minimum-action oracle

Endpoint characterization을 검증하기 위한 offline oracle은 다음 constrained problem이다.

\[
\theta^\dagger
=
\arg\min_\theta
\frac12\|\theta-\theta_0\|_{G_0}^2
\]

subject to

\[
E_i(\theta)\le\tau_i,
\qquad
D_j(\theta)\le\epsilon_j.
\]

여기서 \(G_0\)는 preregistered editor-native action metric이다. Oracle은 direct SQP,
trust-constr, augmented Lagrangian 등으로 coefficient space에서 먼저 푼다. Nonconvex
full-model 문제에서 global optimum이라고 주장하지 않고, multi-start local constrained
solution으로 보고한다.

Online FCW가 oracle과 같은 endpoint를 찾아야 한다고 가정하지 않는다. 대신 다음을 묻는다.

- FCW endpoint가 oracle frontier에서 얼마나 떨어지는가?
- FCW의 intermediate safety가 oracle의 direct path보다 좋은가?
- 동일 compute에서 FCW가 더 자주 feasible point를 찾는가?

## 5.3 조건부 KKT-continuation 경로

Endpoint characterization과 genuine continuation ODE가 모두 필요할 경우에만 target homotopy를
사용한다.

\[
z_i(s)
=
(1-s)F_i(\theta_0)+sz_i^\star,
\qquad s\in[0,1].
\]

각 \(s\)에서 minimum-action constrained problem의 perturbed KKT residual을
\(\mathcal F(y,s)=0\)라 두면, regularity 조건 아래 local path는

\[
\frac{dy}{ds}
=
-\left[\partial_y\mathcal F(y,s)\right]^{-1}
\partial_s\mathcal F(y,s)
\]

로 추적할 수 있다. 이 경로는 analytic center가 아니라 parametric constrained optimum의
continuation이다. Transformer에서는 다음이 발생할 수 있으므로 global guarantee를 금지한다.

- Hessian/KKT matrix singularity
- local branch 전환
- disconnected feasible set
- \(s^\star<1\)에서 safe reachability 종료

이 route는 Gate 0--3 통과 뒤 이론 확장으로만 검토한다.

---

# 6. Nominal writer와 minimum-intervention safety filter

## 6.1 MEMIT nominal direction

Frozen single-layer linear setting에서 target residual \(R\), edit key \(K\), reference covariance
\(C_0\)에 대한 MEMIT update는

\[
\Delta_{\mathrm{MEMIT}}
=
RK^\top(C_0+KK^\top)^{-1}.
\]

이는

\[
\min_\Delta
\|\Delta K-R\|_F^2
+
\operatorname{tr}(\Delta C_0\Delta^\top)
\]

의 one-shot solution이다. FCW의 nominal velocity는 local linear 조건에서 이 방향으로
환원되어야 한다. 이 환원은 baseline 포함관계이자 negative control이다.

## 6.2 AlphaEdit 적용

AlphaEdit projector를 \(P_l\)라 하면 initial MVP에서는 native factor를 proposal 생성 시 한 번
projection한다.

\[
B_l^\alpha=B_l^{\mathrm{native}}P_l.
\]

이후 controller는 \(B_l^\alpha\)의 coefficient만 조절한다. Full functional gradient마다 dense
\(P_l\)를 다시 적용하지 않는다. Approximate projector이므로

\[
\|\Delta K_0\|_F
\]

를 별도로 측정하고 exact null-space guarantee를 주장하지 않는다.

Alpha subspace의 exact local feasibility도 별도로 검사한다. Orthonormal-column basis
\(N\)에 대해 exact projector가 \(P=NN^\top\)일 때
\(\Delta=BN^\top\)라면 target equation은

\[
B(N^\top K_E)=R_E
\]

이며 feasible condition은

\[
R_E
=
R_E(N^\top K_E)^\dagger(N^\top K_E)
\]

이다. 이 조건을 만족하지 않는 request는 결과에서 제거하지 않고
target-infeasible-under-registered-Alpha-subspace로 기록한다. ODE 또는 CBF가
algebraic infeasibility를 해결한다고 주장하지 않는다.

## 6.3 Continuous-time CBF/CLF candidate

Nominal velocity를 \(u(\theta)\), positive-definite local metric을 \(M\)이라 두면 candidate
controller는

\[
\begin{aligned}
(v^\star,s^\star)
=\arg\min_{v,s\ge0}\quad&
\frac12\|v-u(\theta)\|_M^2
+
\frac{\rho}{2}\|s\|_2^2\\
\text{s.t.}\quad&
\nabla h_j(\theta)^\top v
\ge
-\alpha_j(h_j(\theta)),\\
&
\nabla E_i(\theta)^\top v
\le
-c_iE_i(\theta)+s_i.
\end{aligned}
\]

여기서

- functional safety constraint에는 slack을 허용하지 않는다.
- edit-progress CLF에만 slack \(s_i\)를 허용한다.
- \(s_i>0\)는 local progress-safety conflict이지 전역 infeasibility certificate가 아니다.
- Alpha exact-local protection을 주장할 경우 \(V_lK_{0,l}=0\)를 별도 equality로 둔다.

## 6.4 KL constraint의 zero-gradient 문제

Reference distribution을 teacher로 쓰는 KL은 시작점에서

\[
\nabla_\theta
D_{\mathrm{KL}}
(p_{\theta_0}\Vert p_\theta)
\big|_{\theta=\theta_0}
=0.
\]

따라서 첫 write에서 1차 CBF constraint가 nominal direction을 제한하지 못한다. Log barrier로
바꾸어도 \(\nabla h=0\)이면 동일하다. Continuous feedback는 boundary에 접근하며 활성화될 수
있지만, finite Euler step은 boundary를 한 번에 넘을 수 있다.

따라서 actual discrete method는 2차 trust model을 포함한다.

\[
D_j(q+\delta q)
\approx
D_j(q)
+g_j(q)^\top\delta q
+\frac12\delta q^\top H_j(q)\delta q.
\]

Gauss--Newton/Fisher PSD approximation을 사용해

\[
D_j(q)
+g_j^\top\delta q
+\frac12\delta q^\top H_j\delta q
\le\epsilon_j
\]

를 local trust cap으로 둔다. 이 cap도 authority가 아니며, candidate model의 actual functional
value가 최종 accept/reject를 결정한다.

## 6.5 Initial infeasibility

Safe start에서는 CBF forward-invariance 조건을 검토할 수 있다. 그러나 sequential 시작점에서
\(h_j(\theta_0)<0\)이면

\[
\dot h_j\ge-\kappa h_j
\]

가 finite-time recovery를 보장하지 않는다. 등호인 경우

\[
h_j(t)=h_j(0)e^{-\kappa t}<0
\]

이므로 finite time에 safe set으로 들어오지 않는다.

각 edit 전 다음 audit을 수행한다.

1. active contract의 actual value 재평가
2. numerical drift와 supersession ledger 확인
3. safe start면 main writer 실행
4. unsafe start면 separate Phase-I restoration 실행
5. restoration infeasible 또는 budget 초과면 edit reject

Phase-I 성공을 새 edit efficacy에 포함하지 않고 별도 비용과 failure rate로 보고한다.

---

# 7. 계산 가능한 parameterization

## 7.1 Full W-space에서 시작하지 않는다

Llama-3-8B의 \(4096\times14336\) down projection 5개는 약 2.94억 weight 변수를 갖는다.
Dual QP의 constraint 수가 작아도 다음이 병목이다.

- per-anchor functional derivative 수집
- covariance/Fisher inverse action
- dense Alpha projector 적용
- active gradient factor rank 증가
- virtual candidate와 authoritative-dtype commit identity 확인

따라서 full W-space functional CBF를 MVP로 구현하지 않는다.

## 7.2 Stage-1 coefficient state

현재 editor가 만든 layer별 native low-rank factor를 \(B_l\)라 두고

\[
\Delta W_l(q)=q_lB_l,
\qquad
q\in\mathbb R^m,
\qquad m=|\mathcal L|\approx5
\]

로 parameterize한다.

Coordinate identity는 구현 전에 고정한다.

\[
\theta(q)
=
\theta_{\mathrm{entry}}
+
\sum_{l=1}^{m}q_lB_l.
\]

Native endpoint를 재현하는 coefficient를 \(q_{\mathrm{nat}}\)라 두고, proposal factor 정의상
\(q_{\mathrm{nat}}=\mathbf1\)인지 별도 scale이 필요한지 source-backed test로 확정한다.
\(q_{\mathrm{nat}}\)의 weight bytes, logits, rewrite event가 native terminal write와 일치하지
않으면 모든 coefficient-space 실험을 시작하지 않는다.

이 단계의 의미는 다음과 같다.

- full W gradient 없이 functional sensitivity를 검증한다.
- cross-layer functional allocation이 존재하는지 본다.
- static constrained solve와 relinearized feedback을 공정하게 비교한다.
- CAKE/BLUE와 겹치는 residual allocation 효과를 분리하기 위해 base residual allocator는 arm
  내부에서 고정한다.

Coefficient-space metric은

\[
R_{ab}
=
\langle B_a,B_b\rangle_M
\]

의 작은 Gram matrix로 구성한다.

이 Gram matrix는 native factors의 중복 또는 거의 선형 종속 때문에 positive semidefinite일
수 있다. 따라서 preregistered rank tolerance로 numerical range를 한 번 고정하고 그
range에서만 solve한다. Null directions, rank, spectrum과 condition estimate를 receipt에
기록하며, 결과를 본 뒤 ridge를 추가하거나 rank threshold를 바꾸지 않는다.

## 7.3 Per-anchor directional Jacobian

Functional scalar \(D_j\)의 layer direction \(B_l\)에 대한 derivative는

\[
J_{jl}
=
\left\langle
\nabla_{W_l}D_j,
B_l
\right\rangle.
\]

Linear layer input과 output gradient의 batch axis를 보존하면 dense \(\nabla_WD_j\)를 만들지
않고 \(J_{jl}\)를 계산할 수 있다. 단, scalar loss를 합친 ordinary backward 한 번은
microbatch의 **합산 gradient**만 주며 per-anchor row를 자동으로 분리하지 않는다. Exact
per-anchor row는 다음 중 사전 등록된 한 방식으로 계산한다.

- batched VJP 또는 `vmap(grad)`로 anchor별 cotangent를 분리한다.
- memory 때문에 위 방식이 불가능하면 anchor 또는 고정 group별 backward를 수행한다.

두 경우 모두 실제 VJP/backward 수와 grouping을 기록하고, 합산 gradient를 per-anchor
Jacobian으로 재해석하지 않는다.

다음 표현은 금지한다.

- 모든 sequence-level gradient가 rank-1이다.
- 200개의 full weight gradient를 한 backward로 무료 획득한다.
- ordinary parameter `.grad`에 per-anchor gradient가 남는다.

Sequence-level KL/margin gradient는 causal dependency 때문에 일반적으로 rank \(\le T_j\)다.

## 7.4 MVP 비용 계수

다음 기호를 사용한다.

\[
m=\text{editable layer 수},\quad
r_l=\text{layer factor rank},\quad
b=\text{anchor microbatch},\quad
T=\text{token length},\quad
a=\text{active constraint 수}.
\]

| 항목 | MVP 비용과 정책 |
|---|---|
| Native factor 생성 | baseline MEMIT/Alpha 비용; edit당 1회 |
| Hook 추가 연산 | batched VJP 또는 anchor/group별 backward의 실제 횟수와 함께 \(\sum_l O(bTr_l(d_l+d_{\mathrm{out},l}))\) 보고 |
| Hook activation 저장 | 대략 \(\sum_l O(bT(d_l+d_{\mathrm{out},l}))\) |
| Coefficient QP | \(O(m^3+am^2)\); \(m\approx5\)에서는 무시 가능 |
| Candidate functional check | active anchor forward; 실제 반복 병목 |
| Full controller screen | \(\lceil N_{\mathrm{screen}}/b\rceil\) microbatch forward |
| Dense \(C^{-1}\) action | MVP에서 금지 |
| Alpha \(P_l\) action | factor 생성 시 right factor에 한 번만 적용 |

`one backward per microbatch`는 per-anchor row가 필요 없는 aggregate-only ablation에만 허용한다.
Main per-anchor controller의 비용은 batched VJP 수 또는 anchor/group별 backward 수로 센다.

## 7.5 Active constraint working set

Stage-1에서는 controller bank 전체를 매 backward 사용하지 않는다.

1. no-grad 또는 cached-score screen으로 normalized slack을 계산한다.
2. smallest-slack anchor 4--16개를 선택한다.
3. deterministic capability sentinel을 항상 포함한다.
4. accepted step 또는 trust failure 때 active set을 refresh한다.
5. terminal commit 전 \(Q_{\mathrm{ctrl}}\) 전체 actual value를 검사한다.

Constraint guarantee는 검사한 controller bank에만 한정한다. Population-wide 또는 모든 기존
지식 보존을 주장하지 않는다.

## 7.6 조건부 corrective direction 확장

Coefficient basis가 functional benefit을 보인 뒤에만 edit-key null correction을 추가한다.

\[
N_lK_{E,l}=0.
\]

Candidate state는

\[
\Delta W_l(q,a)
=
q_lB_l+a_lN_l
\]

로 확장할 수 있다. \(N_l\)은 fixed target activation을 local하게 보존하면서 functional risk를
줄이는 direction을 제공해야 한다.

확장 순서는 다음으로 고정한다.

1. MEMIT, active anchor \(\le4\)
2. diagonal/KFAC/low-rank preconditioner
3. explicit rank cap과 compression
4. compression 뒤 mandatory actual functional recheck
5. AlphaEdit corrective gradient는 cheap null-basis action이 확보될 때까지 보류

---

# 8. Discrete safe-write algorithm

## 8.1 Authority hierarchy

Candidate transition의 판단 권한은 다음 순서다.

1. cheap \(C\)-metric/spectral proxy: proposal radius와 refresh trigger
2. local quadratic functional model: candidate generation
3. active controller anchor의 actual finite functional value: step acceptance
4. controller bank 전체 actual value: terminal commit approval
5. held-out audit bank: scientific evaluation only

Cheap proxy를 actual functional safety의 upper bound라고 부르지 않는다.

## 8.2 Pseudocode

```text
Input:
  current model theta0
  frozen direct target z_star
  native factors {B_l}
  controller/calibration/gate/audit bank identities
  active version ledger A_t

Preflight:
  audit current contract values
  run Phase-I or reject if start is unsafe
  compute/cache z_star, keys, native factors, teacher logits
  initialize coefficient q = q0
  initialize best feasible point

for macro step k = 1 .. K_max:
  1. Build authoritative-dtype-faithful virtual model theta(q)
  2. Screen Q_ctrl and select smallest-slack anchors + sentinels
  3. If refresh required:
       compute per-anchor coefficient Jacobian by microbatch hooks
       construct small metric Gram and quadratic functional model
  4. Compute nominal edit-progress direction
  5. Solve small minimum-intervention constrained subproblem
  6. Propose q_trial = q + eta * delta_q
  7. Apply cheap cumulative-action/trust screen
  8. Measure actual target and functional values at q_trial
  9. If any hard contract is violated:
       backtrack eta or reject
     else:
       accept q_trial
       update active set/trust state
       optionally relinearize native factors
 10. Record best feasible point and all receipts
 11. Stop on preregistered target event, stagnation, or failure budget

Before commit:
  evaluate all Q_ctrl constraints
  verify virtual-versus-commit authoritative-dtype identity
  choose best feasible point by preregistered endpoint rule

Commit:
  apply exactly one atomic terminal write
  run post-commit identity and functional checks
  evaluate Q_audit exactly once without controller or gate feedback
  update active ledger, removing superseded contracts
```

## 8.3 Step acceptance

Accepted step은 최소한 다음을 만족한다.

\[
E_i(q_{k+1})
\le
E_i(q_k)+\text{preregistered progress tolerance}
\]

또는 target tube 방향의 progress condition과

\[
D_j(q_{k+1})\le\epsilon_j
\quad
\forall j\in\mathcal A_k^{\mathrm{ctrl}}
\]

를 동시에 만족해야 한다. Numerical tolerance, authoritative-dtype rounding allowance,
maximum retry 수는 calibration 뒤 model-common 값으로 고정한다.

---

# 9. Falsification-first experiment gates

## Gate 0. Frozen linear equivalence

### 질문

ODE step 자체가 closed-form writer와 다른 결과를 만드는가?

### 비교

- MEMIT closed form
- 동일 \(K,R,C\)와 frozen \(\Delta\)의 exact algebraic \(N\)-step split
- 동일 quadratic objective의 Euler/gradient flow와 analytic finite-time flow
- direct static QP

### 통과 조건

Frozen \(\Delta\)를 \(N\)등분해 같은 authoritative arithmetic order로 합친 split은 native
closed-form endpoint와 numerical tolerance 내에서 일치해야 한다. 반면 quadratic
gradient flow는 일반적으로 **유한 terminal time에 minimizer와 같지 않다**. 이 arm은
analytic finite-time solution 및 step-refinement convergence와 비교하고, 충분히 긴 시간의
limit에서만 closed-form minimizer 접근을 검사한다. Direct static QP는 별도 endpoint
baseline이며 Euler와의 동일성을 전제하지 않는다.

### Kill condition

단순 분할의 성능 차이를 ODE 효과로 해석하는 모든 claim을 폐기한다.

---

## Gate 1. Same-z endpoint multiplicity

### 질문

동일한 local activation endpoint를 만족하는 weight realization 사이에 실제 functional
preservation 차이가 존재하는가?

### 구성

한 layer, 한 edit batch에서 평균 key가 아니라 모든 authorized rewrite context key를
\(K_E\)에 쌓는다. 일관된 request에 대해

\[
\Delta(U)
=
R_EK_E^\dagger
+
U(I-K_EK_E^\dagger)
\]

를 사용한다.

### 최소 pilot

- 모델별 128 request
- 3 random seed
- request별 16--32개 \(U\) direction
- 동일 null-component \(G\)-action
- candidate 간 matched \(\|\Delta\|_F\) 또는 matched covariance action
- local endpoint relative error \(\le10^{-5}\)
- virtual application과 authoritative commit 뒤 full-model fixed-\(z\) realization check

\(U\) direction, matching rule과 noise-floor threshold는 \(Q_{\mathrm{cal}}\)만 사용해
outcome-independent하게 생성·고정한다. \(Q_{\mathrm{gate}}\)의 functional score를 보고
candidate를 재생성하거나 subset을 제거하지 않는다.

### 평가

\[
L_{\mathrm{key}}(\Delta)
=
\operatorname{tr}(\Delta C_{\mathrm{ref}}\Delta^\top)
\]

\[
L_{\mathrm{func}}(\Delta)
=
\operatorname{CVaR}_{0.95,x\in Q_{\mathrm{gate}}}
D_{\mathrm{KL}}
(p_{\theta_0}\Vert p_{\theta_0+\Delta}).
\]

추가로 edit paraphrase, neighborhood locality, downstream capability sentinel을 측정한다.

### Go condition

Target residual과 action을 맞춘 후보 사이에서 noise floor보다 큰 gate-bank functional spread가
두 모델 또는 두 dataset 조건에서 재현된다.

여기서 local \(\Delta K_E=R_E\)는 layer-local equality일 뿐이다. Full-model target event,
rewrite activation과 logits가 preregistered tolerance 안에 유지되지 않은 후보는 `same-z`로
분류하지 않는다.

### No-go condition

Functional spread가 작거나 update action만으로 거의 전부 설명되면 same-z safe-selection 가설을
폐기한다.

---

## Gate 2. Functional signal의 conditional incremental value

### 질문

Functional feedback이 AlphaEdit/EvoEdit/BetaEdit의 key/null-space protection보다 독립적인
정보를 제공하는가?

### 두 risk

\[
r_i^K
=
\|\Delta_iK_0\|_F
\quad\text{또는}\quad
\operatorname{tr}(\Delta_iC_0\Delta_i^\top),
\]

\[
r_i^F
=
\operatorname{CVaR}_{0.95,x\in Q_{\mathrm{ctrl}}}
\|J_{\log p}(x)\Delta_i\|_{F_x}^2.
\]

### Conditional strata

| | Functional low | Functional high |
|---|---:|---:|
| Key risk low | sanity cell | 핵심 차별화 cell |
| Key risk high | key method 우세 예상 | attribution이 혼합된 cell |

핵심은 key-low / functional-high request가 실제로 존재하고, functional writer가 해당 subset의
gate-bank drift를 줄이는가다.

### 비교 arm

1. native MEMIT/AlphaEdit
2. BetaEdit/EvoEdit key/history control
3. key-only constrained writer
4. functional-only constrained writer
5. key + functional writer
6. 동일 step 수의 barrier-off split

### Matching

다음 세 축을 각각 고정한 conditional comparison을 보고한다.

- 동일 target residual
- 동일 key-risk 또는 update action
- 동일 wall-clock/NFE budget

### Go condition

Key-risk와 update magnitude를 통제한 뒤에도 functional writer의 gate-bank CVaR drift 개선에 대한
bootstrap confidence interval이 zero improvement를 제외하고, efficacy non-inferiority가
preregistered bound를 만족한다.

### No-go condition

Conditional matching 뒤 효과가 사라지면 functional barrier는 더 비싼 key-risk proxy로
판정하고 method 확장을 중단한다.

---

## Gate 3. Coefficient-space functional constrained writer

### 질문

현재 low-rank native basis만으로도 functional constraints가 endpoint를 바꾸고 gate-bank
preservation을 개선하는가?

### Main pilot arms

- static native coefficient scaling
- static functional constrained QP
- multi-step coefficient update, no relinearization
- multi-step coefficient update, functional value refresh only
- multi-step coefficient update, Jacobian/native factor relinearization

### Go condition

Static native scaling보다 gate-bank efficacy--preservation frontier가 개선되고, active constraint가
실제로 발생하며, barrier-on/off의 terminal point가 달라진다.

### No-go condition

Constraint가 거의 활성화되지 않거나 gate-bank 이득이 없으면 functional controller를 중단한다.

---

## Gate 4. ODE necessity

### 질문

Multi-step trajectory가 동일 constrained endpoint problem을 직접 푸는 것보다 필요한가?

### 필수 비교

1. \(N=1\) static functional constrained QP
2. \(N>1\), frozen \(K,R,J\): step-only
3. \(N>1\), refreshed \(K,R,J\): feedback-only
4. refreshed + CBF/trust filter
5. offline minimum-action SQP oracle
6. Euler step-size sweep와 RK4 diagnostic

### ODE Go

다음을 모두 만족해야 한다.

- 동일 compute에서 static solver에 비지배적이거나 우월한 Pareto point
- functional constraint가 trajectory 중 실제 활성화
- refreshed field가 frozen field와 실질적으로 다름
- endpoint가 step refinement에 따라 안정적으로 수렴
- controller bank가 아닌 frozen gate bank에서도 이득 유지

### ODE No-go / pivot

다음 중 하나면 ODE 명칭을 폐기한다.

- \(N=1\) constrained solve와 동일 endpoint/성능
- feedback 이득이 단순 residual redistribution으로 설명됨
- step 수가 늘어도 field 또는 active set이 거의 변하지 않음
- 이득이 더 많은 forward/backward budget에서만 발생
- direct SQP가 더 싸고 같거나 더 좋은 frontier를 만듦

이 경우 최종 방법은 static Functional-Constrained Write다.

---

## Gate 5. Batch, sequential, supersession

Gate 0--4를 통과한 뒤에만 long-horizon을 실행한다.

### Batch contention

- batch size \(B\in\{1,10,100\}\)
- total edit fact 수 고정
- batch 사이 model reset
- non-conflicting fact만 사용
- per-edit target residual, feasibility, rank, functional drift 기록

Residual allocator는 arm 내부에서 고정하여
[CAKE](https://aclanthology.org/2026.acl-long.918/)와
[BLUE](https://arxiv.org/abs/2502.03748)의 allocation 효과와 섞이지 않게 한다.

### Append-only sequential

- pilot: 100 events
- main: 2,000 events
- stress: 10,000 events
- 모든 과거 edit가 여전히 유효한 stream
- 최소 3 order seed와 adversarial high-association order
- 100/500 event마다 retention, capability, functional drift, key leakage, runtime 측정

### Supersession

- 동일 \((subject,relation)\)의 value version이 반복 교체되는 stream
- expired target을 active ledger와 history protection에서 제거
- current success, old suppression, unrelated retention 분리

Long-horizon 자체는 [LyapLock](https://aclanthology.org/2025.emnlp-main.327/),
[EvoEdit](https://arxiv.org/abs/2510.13851/),
[BetaEdit](https://arxiv.org/abs/2605.09285/)가 이미 다루므로 novelty가 아니다.

---

# 10. Baselines와 novelty boundary

## 10.1 반드시 포함할 editing baseline

- MEMIT
- AlphaEdit
- AlphaEdit history-aware variant
- BLUE-compatible boundary-layer allocation
- CAKE causal residual allocation
- SPHERE
- EvoEdit
- BetaEdit
- LyapLock 또는 공개 구현 가능한 long-term constrained wrapper
- FCW barrier-off 및 step-only controls
- direct static constrained solver

## 10.2 최신 연구와의 경계

| 연구 | 이미 다루는 핵심 | 본 연구에서 금지할 중복 claim | 남는 비교 질문 |
|---|---|---|---|
| [MetaKE](https://arxiv.org/abs/2603.12677) | downstream-aware \(z\) bi-level optimization | feasibility-aware target이 최초 | fixed \(z\)의 safe W realization이 추가로 필요한가 |
| [BLUE](https://arxiv.org/abs/2502.03748) | residual distribution error와 boundary layer | multi-layer residual 재분배가 최초 | allocator 고정 뒤 functional feedback 효과가 남는가 |
| [CAKE](https://aclanthology.org/2026.acl-long.918/) | causal layer selection과 adaptive residual allocation | adaptive layer weight가 최초 | causal efficacy와 functional preservation이 다른 신호인가 |
| [SPHERE](https://arxiv.org/abs/2510.01172) | hyperspherical energy와 weight geometry | geometry-based stability가 최초 | functional behavior contract가 geometry를 넘어서는가 |
| [EvoEdit](https://arxiv.org/abs/2510.13851) | evolving null-space history alignment | dynamic projector가 최초 | key-safe이지만 function-unsafe인 case가 존재하는가 |
| [BetaEdit](https://arxiv.org/abs/2605.09285) | pseudo-null leakage와 history penalty | leakage control이 최초 | output-space constraint의 incremental value가 있는가 |
| [LyapLock](https://aclanthology.org/2025.emnlp-main.327/) | long-term constrained editing | 최초 장기 preservation 보장 | per-write local functional contract가 별도 가치를 갖는가 |
| [ODESteer](https://arxiv.org/abs/2602.17560) | activation density-ratio ODE steering | 최초 barrier-guided ODE | fixed-z W write의 functional safe set은 다른 문제인가 |
| [ODE-M](https://arxiv.org/abs/2605.19409) | barrier-aware continual merging path | 최초 parameter ODE | edit-specific target manifold와 functional contract가 다른가 |

## 10.3 허용되는 잠정 novelty statement

Gate 0--4 통과 전에는 다음보다 강하게 쓰지 않는다.

> We study whether state-dependent functional constraints can select safer weight realizations for
> a frozen edit target beyond key-space and residual-allocation controls.

Gate 통과 뒤에도 다음은 금지한다.

- first ODE-based model editor
- first constrained sequential editor
- all knowledge preservation guarantee
- canonical or naturally unique barrier
- globally optimal safe endpoint
- functional barrier가 population worst-case를 보장

---

# 11. Evaluation과 공정 비교

## 11.1 Primary outcome

단일 composite score를 primary로 사용하지 않는다. 다음 Pareto surface를 보고한다.

\[
x=\text{held-out functional CVaR drift},
\]

\[
y=\text{edit efficacy/generalization},
\]

\[
z=\text{wall-clock 또는 full-model NFE}.
\]

Conditional slices는 다음과 같다.

- fixed compute에서 edit quality 대 functional risk
- fixed efficacy에서 functional risk 대 runtime
- fixed functional budget에서 efficacy/generalization 대 runtime
- fixed key-risk에서 functional risk 대 efficacy

## 11.2 Metrics

### Edit

- efficacy
- paraphrase generalization
- target activation residual
- per-edit failure distribution
- native non-inferiority

### Preservation

- controller/audit mean KL 분리
- individual worst and CVaR KL
- neighborhood locality
- active previous-edit retention
- current-version success와 old-version suppression
- MMLU 및 preregistered capability panel

### Mechanism

- \(\|\Delta K_0\|_F\)
- covariance action
- active constraint 수와 identity
- minimum slack trajectory
- CLF slack
- accepted/rejected/backtracked step 수
- key/residual/Jacobian refresh delta
- barrier-on/off endpoint distance
- endpoint solver/step sensitivity

### Compute

- total GPU seconds/edit
- peak allocated/reserved memory
- full-model forward 수
- backward/JVP/VJP 수
- barrier screen/actual check 수
- z optimization 비용과 W-write 비용 분리
- virtual trial과 physical commit 수

## 11.3 Hyperparameter fairness

\(\tau\), \(\epsilon\), \(\kappa\), trust radius, step 수는 efficacy--preservation trade-off를
직접 바꾼다. 단일 best point 비교를 금지하고 다음 sweep을 같은 dev budget에서 수행한다.

- MEMIT covariance/regularization strength
- AlphaEdit/BetaEdit projector 및 penalty parameter
- FCW target tube와 functional budget
- static solver tolerance와 FCW integration tolerance

Main point는 preregistered calibration rule로 하나를 고정하고, 전체 Pareto curve 및
hypervolume을 함께 보고한다. Held-out 결과로 main point를 바꾸지 않는다.

Gate 1--4의 선택에는 \(Q_{\mathrm{gate}}\)만 사용한다. 모든 선택이 끝난 뒤
\(Q_{\mathrm{audit}}\)를 한 번 열어 final estimate를 만들며, audit 결과가 나쁘더라도
동일 study 안에서 threshold, arm, subset 또는 endpoint rule을 고치지 않는다.

## 11.4 Compute matching

Functional writer가 anchor forward를 더 사용한다면 key-only baseline에도 동일 wall-clock cap과
사전 등록된 optimization budget을 준다. Dummy forward로 비용만 맞추지 않는다. 다만 결과를
본 뒤 baseline에 candidate, refresh 또는 hyperparameter evaluation을 추가하는 것도 금지한다.
각 arm의 candidate 수, refresh 수, forward/backward/VJP 수와 wall-clock을 함께 보고하며,
compute-matched 결과와 fixed-algorithm 결과를 분리한다.

W-write arm에서는 \(z^\star\)를 cache해 같은 값을 공유한다. Full-system 표에서는 z 비용을
다시 더해 별도로 보고한다.

---

# 12. 필수 ablation matrix

| 축 | Arm |
|---|---|
| Step | one-shot / split-only / feedback |
| State refresh | none / functional value / Jacobian / key+residual+factor |
| Constraint | none / key only / functional only / key+functional |
| Solver | static QP / PGD-Adam / CBF-QP / SQP oracle |
| Endpoint | last / best feasible / target first-hit / minimum-action oracle |
| Metric | identity / covariance / diagonal Fisher-KFAC approximation |
| Basis | native coefficient / projected Alpha basis / null corrective extension |
| History | none / active append-only / supersession-aware |
| Numerical solver | Euler step sweep / RK4 diagnostic |

Main paper의 모든 대규모 arm을 한 번에 실행하지 않는다. Gate별 최소 arm만 실행하고 통과한
구성만 다음 단계로 확장한다.

---

# 13. Failure taxonomy와 대응

| Failure | 해석 | 대응 |
|---|---|---|
| same-z 후보의 functional spread가 없음 | endpoint selection premise 실패 | 연구 중단 또는 static minimum-action으로 축소 |
| functional benefit이 key-risk matching 뒤 소멸 | BetaEdit 대비 추가 가치 없음 | functional CBF 중단 |
| KL gradient가 시작점에서 0 | 1차 CBF 한계 | quadratic trust model + actual finite check |
| active constraint가 한 번도 작동하지 않음 | barrier mechanism evidence 없음 | threshold를 결과 후 변경하지 말고 negative result 보고 |
| controller만 개선, audit 악화 | constraint overfitting | method no-go; anchor 확대는 새 preregistered study |
| CBF-QP infeasible | local rate conflict | Phase-I/reject; safety slack으로 숨기지 않음 |
| initial \(h<0\) | invariance 전제 실패 | 별도 restoration 또는 reject |
| Euler actual violation | discretization error | backtracking, exact check, step refinement |
| static QP와 ODE 동일 | trajectory 불필요 | ODE claim 폐기 |
| direct SQP가 더 효율적 | ODE solver 불필요 | static FCW로 pivot |
| full-space correction rank 폭발 | 계산 불가능 | active set/rank cap; coefficient basis로 복귀 |
| Alpha projector 적용 비용 폭발 | dense projection bottleneck | projected native factor만 사용 |
| virtual/commit mismatch | acceptance certificate 무효 | commit 금지, authoritative-dtype identity gate 실패 기록 |
| superseded target가 보호됨 | contract ledger 오류 | run invalidation 후 ledger 수정 |

---

# 14. Go/No-go decision table

## 연구 전체 Go

다음을 모두 만족해야 한다.

1. 동일 activation endpoint의 matched-action W 후보가 gate-bank functional drift 차이를 만든다.
2. Functional signal이 key/null-space risk를 통제한 뒤에도 추가 설명력과 intervention value를
   가진다.
3. Low-rank coefficient writer가 static native scaling보다 Pareto frontier를 개선한다.
4. Multi-step refreshed writer가 static constrained solver보다 compute-matched 추가 가치를
   가진다.
5. Gate 1--4와 모든 hyperparameter를 freeze한 뒤 controller/gate-bank 개선이 final audit bank에서
   한 번 재현된다.
6. Sequential/supersession에서 expired contract를 보호하지 않고 active fact를 유지한다.

## Static pivot

Gate 1--3은 통과하지만 Gate 4가 실패하면 method는 static FCW로 정리한다.

## Full kill

다음 중 하나면 핵심 연구를 종료한다.

- Gate 1 functional multiplicity 부재
- Gate 2 conditional incremental value 부재
- held-out audit에서 반복적인 역효과
- feasibility를 유지하려면 efficacy가 baseline보다 실질적으로 낮아짐
- compute frontier가 모든 strong baseline에 지배됨

No-go 결과를 threshold rescue, model-specific policy, outcome-dependent subset 제거로 뒤집지
않는다.

---

# 15. 시각화 계획

Static convex bowl에서 Euler step이 analytic center로 들어가는 그림은 사용하지 않는다.
그 그림은 direct convex solve가 더 적합하다는 반론을 강화하고 실제 nonconvex functional
problem을 잘못 표현한다.

Primary conceptual figure는 actual update subspace의 2-D slice로 구성한다.

- edit target tube 또는 same-z solution manifold
- controller functional safe region
- Alpha projected/equality subspace
- one-shot MEMIT/Alpha endpoint
- split-only trajectory
- functional feedback trajectory
- static constrained solution과 offline minimum-action oracle

하단 panel에는 다음을 함께 그린다.

- \(E_i(t)\)
- \(\min_j h_j(t)\)
- active constraint count
- accepted/rejected step
- controller 대 audit functional drift

핵심 시각 메시지는 다음이다.

> Euler가 안정적이어서가 아니라, 동일 target을 만족하는 weight freedom 안에서 실제
> functional boundary를 재관측할 때 path와 terminal realization이 달라진다.

---

# 16. 현재 repository 자산의 처리

## 16.1 재사용 후보

| 자산 | 재사용 범위 |
|---|---|
| [easyedit_bridge.py](../run_scripts/ode_edit_motivation/easyedit_bridge.py) | frozen direct-\(z\), native MEMIT factor/covariance lineage |
| [memit_adapter.py](../run_scripts/ode_edit_method/memit_adapter.py), [hooks.py](../run_scripts/ode_edit_method/hooks.py) | low-rank factor adapter와 accepted-write transaction |
| [derivatives.py](../run_scripts/ode_edit_method/derivatives.py) | directional derivative 기반; batch-preserving coefficient Jacobian으로 확장 |
| [diagnostic_math.py](../run_scripts/ode_edit_motivation/diagnostic_math.py) | low-rank \(C\)-inner product와 Gram correctness |
| [functional_trial.py](../run_scripts/ode_edit_method/functional_trial.py) | virtual low-rank trial과 authoritative-dtype-faithful finite candidate check |
| [projector_adapter.py](../run_scripts/ode_edit_motivation/projector_adapter.py) | Alpha projector를 proposal factor에 한 번 적용 |
| [controller.py](../run_scripts/ode_edit_method/controller.py), [runtime.py](../run_scripts/ode_edit_method/runtime.py) | small solver, trust, instrumentation 골격 |
| transaction/receipt code | atomic terminal commit, rollback, immutable identity |

all_layer_directional_derivatives의 dense target-weight path는 correctness oracle로만 유지한다.
현재 rewrite-only trust ratio는 functional constraint 전체의 vector-valued actual post-trial
check로 확장해야 한다.

## 16.2 그대로 승계하지 않는 구성

- fixed-E8을 method-defining scientific mechanism으로 사용하는 것
- cold/warm target pilot의 결과를 새 가설 근거로 재사용하는 것
- arbitrary hard functional-P threshold
- capacity load만으로 functional safety를 주장하는 것
- soft routing 결과를 formal CBF로 부르는 것
- first-hit을 최적 endpoint로 부르는 것

## 16.3 필요한 최소 구현 변경

1. directional hook의 batch-preserving per-anchor coefficient Jacobian
2. controller/calibration/gate/audit bank identity enforcement
3. normalized functional slack과 active-set screen
4. small coefficient-space QP/QCQP
5. quadratic functional trust model
6. actual vector-valued functional candidate check
7. supersession-aware active ledger
8. Gate 0--2 diagnostic scripts

이 변경은 Gate 0--2가 통과하기 전에는 full scientific method implementation으로 확장하지
않는다.

---

# 17. 실행 순서

## Phase A. 문서·수식·데이터 계약

- 본 proposal freeze
- source/dataset split manifest 작성
- direct-\(z\) cache identity 정의
- no-op/calibration protocol 작성
- Gate 0--2 preregistered thresholds와 statistics plan 작성

## Phase B. 최소 falsification

1. Gate 0 frozen equivalence
2. Gate 1 same-z multiplicity
3. Gate 2 key/function conditional strata

이 단계에서 GPU 규모를 확대하지 않는다.

## Phase C. Coefficient-space MVP

- MEMIT 한 모델, 한 dataset
- active anchors 4--16
- static QP 대 multi-step feedback
- controller/gate/audit 완전 분리
- FULL_FP32 terminal identity

## Phase D. ODE audit

- fixed/relinearized field 비교
- static constrained solver와 matched compute
- step refinement와 solver sensitivity
- ODE Go/No-go 판정

## Phase E. Strong baselines와 확장

- AlphaEdit projected native factors
- BetaEdit/EvoEdit/SPHERE/CAKE/BLUE
- null corrective direction
- second model/dataset replication

## Phase F. Sequential

- 100-event pilot
- 2K main
- 조건부 10K stress
- append-only와 supersession 분리

---

# 18. Reproducibility receipts

각 run은 최소한 다음을 기록한다.

- git commit와 dirty-state manifest
- model/tokenizer/revision hash
- dataset row IDs와 split hash
- edit order와 seed
- direct-\(z\) tensor hash
- editable layer와 native factor hash
- controller/calibration/gate/audit bank hash
- active ledger before/after
- all hyperparameters와 solver tolerances
- step별 target/functional actual values
- predicted versus actual trust ratio
- active constraint identity
- forward/backward/JVP/VJP counts
- wall-clock, memory, retry 수
- virtual/commit/post-commit identity
- terminal selection rule와 selected step

Server-head 보고는 `PROTOCOL.md`의 factual-only policy를 따른다. 과학적 해석과 Go/No-go
판정은 global report에서 분리한다.

---

# 19. 최종 paper narrative의 조건부 형태

Gate 0--4가 모두 통과했을 때만 다음 narrative를 검토한다.

1. Fixed semantic target does not determine a unique parameter realization.
2. Key-space preservation does not fully predict functional behavior drift.
3. Functional contracts induce a minimum-intervention safe-write filter.
4. State-dependent relinearization changes the active safety geometry.
5. The resulting trajectory improves the compute-matched frontier over static constrained writes.

Gate 4가 실패하면 4--5를 제거하고 static constrained writer paper로 축소한다.

---

# 20. 참고문헌

## ODE와 control

- [ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](https://arxiv.org/abs/2602.17560)
- [Unlocking the Potential of Continual Model Merging: An ODE Perspective](https://arxiv.org/abs/2605.19409)
- [Control Barrier Function Based Quadratic Programs for Safety Critical Systems](https://arxiv.org/abs/1609.06408)
- [Boyd and Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)

## Knowledge editing

- [MEMIT: Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355)
- [Rethinking Residual Distribution in Locate-then-Edit Model Editing / BLUE](https://arxiv.org/abs/2502.03748)
- [Energy-Regularized Sequential Model Editing on Hyperspheres / SPHERE](https://arxiv.org/abs/2510.01172)
- [EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](https://arxiv.org/abs/2510.13851)
- [MetaKE: Meta-learning Aligned Knowledge Editing via Bi-level Optimization](https://arxiv.org/abs/2603.12677)
- [CAKE: Causal-Guided Adaptive Knowledge Editing for LLMs](https://aclanthology.org/2026.acl-long.918/)
- [BetaEdit: Null-Space Constrained Sequential Model Editing](https://arxiv.org/abs/2605.09285)
- [LyapLock: Bounded Knowledge Preservation in Sequential Large Language Model Editing](https://aclanthology.org/2025.emnlp-main.327/)
- [Serial Lifelong Editing via Mixture of Knowledge Experts](https://aclanthology.org/2025.acl-long.1492/)

---

# 21. 2026-08-30 결정 요약

| 결정 | 상태 |
|---|---|
| ODE 적용 위치 | W write stage |
| direct-\(z\) | 고정; infeasibility fallback 외 재최적화 금지 |
| canonical log barrier | 기각 |
| analytic-center endpoint | 기각 |
| functional contract | 채택하되 calibration과 finite-bank 한계 명시 |
| 1차 CBF 단독 safety | 기각; quadratic trust + actual check 필수 |
| full W-space QP | 초기 구현에서 기각 |
| coefficient-space MVP | 채택 |
| first-hit optimality | 주장 금지 |
| offline minimum-action solver | oracle baseline으로 채택 |
| ODE contribution | Gate 4 통과 전 보류 |
| long-horizon 실행 | Gate 0--4 통과 뒤로 보류 |
| 다음 실험 | Gate 0, Gate 1, Gate 2 순서 |

이 표가 변경되려면 새로운 experiment report 또는 audit 근거와 함께 본 문서의 후속
date-stamped revision을 작성한다. 이전 결과를 근거 없이 새 proposal에 소급 적용하지 않는다.
