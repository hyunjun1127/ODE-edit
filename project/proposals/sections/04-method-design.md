# ODE-Edit Method Session — layer-synchronous capacity-aware flow

- 초안 개시: **2026-08-03 12:04 KST**
- 정식 개시: **2026-08-03 14:20 KST**, 사용자 canonical objective 반영
- 방법명: **`ODE-Edit`** (`BF`는 방법명이 아니라 layer-synchronous execution 원칙)
- 상태: **`FORMAL_METHOD_DESIGN_OPEN; EXECUTION_HOLD_PENDING_LOCKED_SPEC_AND_SH_ONBOARDING`**
- fast primary execution spec:
  [`../../../plans/global/2026-08-03-session02-fast-main-table-spec.md`](../../../plans/global/2026-08-03-session02-fast-main-table-spec.md)
- inherited Motivation verdict: **`CLOSED_DIRECTIONAL_POSITIVE; STRONG_METHOD_GATE_FAIL`**
- 대상 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- actuator family: MEMIT, canonical projector/history AlphaEdit
- policy 원칙: 두 모델·두 family 공통 controller, model-specific rescue 금지
- large/lifelong 상태: strong pilot 전까지 **closed**

## 0. Canonical Method objective

### 사용자 canonical input

기존 locate-then-edit editor는 한 edit request의 fixed direct-z를 semantic target으로
삼고 layer를 정해진 순서로 방문해 local low-rank write를 전량 commit한다. MEMIT은 앞
layer write 뒤의 residual과 key를 뒤 layer 계산에는 전달하지만, 이미 방문한 layer를
새 joint model state에서 다시 비교·조정하지 않는다. 따라서 layer 축에서 한 번의
ordered Gauss--Seidel-like sweep으로 볼 수 있다.

ODE-Edit은 같은 fixed direct-z 아래에서 모든 edit layer의 low-rank proposal을 동일한
current model snapshot에서 synchronous하게 다시 선형화한다. 허용된 rewrite context의
predicted progress와 editor-native cumulative capacity geometry를 사용해 layer별 actual
write coefficient를 공동 결정하고, joint partial step 뒤 같은 과정을 반복한다. Rewrite
event를 처음 만족하면 종료해 동일 semantic target을 만족하는 여러 parameter solution 중
낮은 cumulative capacity와 좋은 retention frontier를 갖는 endpoint를 탐색한다.

### MEMIT과의 한 문장 차이

> MEMIT은 바뀐 model state를 **뒤 layer 계산에만** 전달하지만, ODE-Edit은 바뀐 joint
> state를 **모든 layer의 다음 proposal과 allocation 계산에 다시 전달한다.**

### ODE가 필요한 이유

ODE 표현의 이유는 write를 천천히 적용하는 데 있지 않다. Joint partial update가 다음
순간의 residual, key, proposal direction, rewrite efficiency와 marginal capacity cost를
바꾸므로, model state에 종속된 edit vector field를 repeated feedback integration으로
풀어야 한다는 가설 때문이다. Best frozen/static controller가 같은 frontier를 만들면 ODE
necessity는 기각하고 static capacity routing으로 단순화한다.

### 네 범주

- **Proposal에서 온 내용:** fixed direct-z의 parameter realization은 하나가 아닐 수 있고,
  state-refreshed layer-synchronous routing이 lower-capacity endpoint를 찾을 수 있다는
  가설이다.
- **Repo/protocol에서 확인한 사실:** C3는 share--magnitude separation의 필요성만 확인했고,
  applied hard barrier, adaptive trust/rollback, first-hit과 ODE necessity는 검증하지 않았다.
- **GH 추정:** 다음 핵심 estimand는 waypoint 수가 아니라 full dynamic controller와 best
  frozen/static controller의 matched-terminal frontier 차이다.
- **사용자 확인 필요:** 없음. 사용자가 Method section의 정식 개시와 위 objective를 직접
  승인했다. 다만 구체 hyperparameter와 실행 resource는 별도 locked execution spec이
  승인되기 전까지 열지 않는다.

## 1. Motivation에서 상속한 설계 요구

C3는 완성된 ODE editor의 superiority를 보인 실험이 아니다. C1 BF allocation
coefficient가 relative layer share와 global update magnitude를 동시에 결정해 under-write를
만든 원인을 분리했고, 동일 BF allocation algorithm과 independent path magnitude를
결합했을 때 네 model×family cell이 모두 C1보다 회복한다는 directional signal을 남겼다.

상세 evidence와 claim boundary는
[`../../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md`](../../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)에 있다.

Method는 다음 세 원칙으로 시작한다.

1. **Allocation–magnitude separation**

   어느 layer가 semantic progress를 담당할지와 전체적으로 얼마나 이동할지를 분리한다.
   Raw QP norm이 우연히 actual global update magnitude가 되지 않게 한다.

2. **Applied-step feasibility**

   QP가 제약하고 최적화한 바로 그 coefficient를 실제로 적용한다. C3처럼 사후 radial
   amplification으로 load optimum과 trust contract를 깨지 않는다.

3. **Free trajectory duration**

   Native distance 또는 고정 \(K\)가 아니라 semantic rewrite event와 trust accuracy가
   step size와 trajectory length를 결정한다.

이 section이 여는 것은 method design과 common strong pilot이지, 구현 완료나 native
superiority claim이 아니다.

## 2. Baseline과 방법의 정확한 차이

MEMIT은 edit당 direct-z를 한 번 계산하고 layer를 ascending order로 방문한다. 앞 layer의
full write 뒤 뒤 layer의 current key와 remaining residual을 다시 측정하므로 완전히
고정된 open-loop update는 아니다. 그러나 앞서 full write한 layer를 새로운 joint state에서
다시 방문하지 않는 one-pass ordered Gauss–Seidel에 가깝다.

ODE-Edit은 다음 READ–PROPOSE–CONTROL–TRIAL–ACCEPT 구조를 반복한다.

1. **READ:** 같은 current snapshot에서 모든 edit layer의 key와 current residual을 읽는다.
2. **PROPOSE:** 모든 layer의 local low-rank actuator를 weight 변경 없이 만든다.
3. **CONTROL:** rewrite progress와 cumulative capacity를 사용해 actual layer coefficient를
   공동 결정한다.
4. **TRIAL:** joint partial update의 actual rewrite progress와 barrier를 측정한다.
5. **ACCEPT/REJECT:** trust ratio에 따라 commit 또는 exact rollback하고 step size를 조절한다.
6. **EVENT:** semantic rewrite goal을 처음 만족하면 즉시 종료한다.

핵심은 waypoint 수가 아니라 partial joint update 후 editor 전체를 current state에서 다시
선형화한다는 점이다.

## 3. Fixed direct-z target과 evolving state

Outer edit \(t\)가 시작될 때 current stream model을 \(W_{t,0}\), request를 \(r_t\)라 한다.
Direct-z는 한 번만 계산한다.

\[
Z_t^\star=\operatorname{DirectZ}(W_{t,0},r_t).
\]

\(Z_t^\star\)는 edit 안에서 고정된 exogenous representation target이다. ODE state는
\(Z_t^\star\)가 아니라 editable weights \(W_t(\tau)\)다.

Waypoint \(s\)에서 current residual은

\[
E_{t,s}=Z_t^\star-H^L(W_{t,s})
\]

이고 layer key와 proposal은

\[
K_{t,s,l}=K_l(W_{t,s}),
\]

\[
B_{t,s,l}
=\operatorname{EditorProposal}_l
\left(W_{t,s},E_{t,s},K_{t,s,l}\right)
\]

이다. Covariance metric으로 unit actuator를 만든다.

\[
\widehat B_{t,s,l}
=\frac{B_{t,s,l}}
{\|B_{t,s,l}\|_{C_l}+\epsilon_B}.
\]

여기서 editable weight와 같은 shape의 \(A,B\)에 대해 사용하는 covariance geometry는

\[
\langle A,B\rangle_{C_l}
=\operatorname{tr}(A C_l B^\top),
\qquad
\|A\|_{C_l}^{2}=\langle A,A\rangle_{C_l}
\]

로 고정한다. \(C_l\)는 EasyEdit actuator solve가 이미 사용하는 precomputed object를
read-only로 재사용하며, ODE-Edit을 위해 새 corpus에서 재계산하지 않는다.

정리하면:

- fixed direct-z: trajectory 중 semantic target drift를 막는다.
- current residual: 같은 target까지 남은 functional deficit이다.
- current key/proposal: 현재 weights에서 residual을 줄이는 local actuator다.
- ODE adaptivity: direct-z를 매번 다시 최적화하는 것이 아니라 actuator field를 현재
  state에서 다시 선형화하는 것이다.

Main method에서는 direct-z를 매 hop 다시 계산하지 않는다. Direct-z refresh는 target
drift와 state relinearization을 섞으므로 필요하면 별도 ablation으로만 둔다.

## 4. Rewrite event와 layer efficiency

Controller가 읽을 수 있는 정보는 rewrite prompt, subject, target object, 허용된 MEMIT
context prefix, current hidden/key와 precomputed covariance에 한정한다. Evaluation prompt,
locality outcome, held-out answer는 controller에 들어가지 않는다.

빠른 MEMIT main track의 event는 target sequence \(o^\star\)와 old sequence
\(o^{\rm old}\)의 length-normalized teacher-forced log-likelihood를 비교한다.

\[
\ell_c(o;W)
=\frac1{|o|}
\sum_{j=1}^{|o|}
\log P_W(o_j\mid c,o_{<j}),
\]

\[
m_c(W)=\ell_c(o^\star;W)-\ell_c(o^{\rm old};W),
\qquad
\Phi_{\rm event}(W)=\max_{c\in\mathcal C}[-m_c(W)].
\]

Rewrite goal은 \(\Phi_{\rm event}(W)\le0\)이다. 즉 direct-z construction에 허용된 모든
context에서 target sequence가 old sequence보다 높은 첫 상태다. Length normalization은
서로 다른 object-token 길이의 합산 log-likelihood bias를 막고, zero threshold는 별도
margin tuning을 없앤다. 이는 controller-side event이지 evaluation efficacy나
autoregressive exact-match claim이 아니다.

Controller gradient에는 normalized smooth surrogate를 별도로 사용한다.

\[
\widetilde\Phi_{\rm ctrl}(W)
=\tau\log\left[
\frac1{|\mathcal C|}
\sum_{c\in\mathcal C}
\exp\left(\frac{-m_c(W)}{\tau}\right)
\right].
\]

Layer \(l\)의 instantaneous semantic efficiency는

\[
a_{t,s,l}
=\left[-D\widetilde\Phi_{\rm ctrl}(W_{t,s})
[\widehat B_{t,s,l}]\right]_+
\]

로 둔다. Exact all-vocabulary token별 top-1 event는 첫 main table의 terminal이 아니라
후속 robustness ablation으로 남긴다.

## 5. Monotone outer-edit load와 signed displacement diagnostic

Stream origin의 layer weight를 \(W_{0,l}\)라 하고 normalization을

\[
D_l=\|W_{0,l}\|_{C_l}^{2}+\epsilon_D
\]

로 둔다. Outer edit \(i\)가 first-hit terminal에서 남긴 net write를

\[
U_{i,l}=W_{i,T_i,l}-W_{i,0,l}
\]

라 하면 main controller의 historical load는

\[
\Omega_{t,l}
=\sum_{i<t}\frac{\|U_{i,l}\|_{C_l}^{2}}{D_l}
\]

이다. \(\Omega\)는 모든 completed outer edit의 net write에 양의 비용을 부여하고,
과거 edit를 지우는 방향을 capacity 회복으로 보상하지 않는다. 같은 outer edit를 더 많은
micro-step으로 나눴다는 이유로 비용이 작아지지 않도록, first-hit terminal에서 정확히
한 번만

\[
\Omega_{t+1,l}
=\Omega_{t,l}
+\frac{\|W_{t,T_t,l}-W_{t,0,l}\|_{C_l}^{2}}{D_l}
\]

로 갱신한다. Current edit 안에서는 \(\Omega_{t,l}\)를 freeze한다.

원래 base-relative displacement

\[
\Psi_{t,s,l}
=\frac{\|W_{t,s,l}-W_{0,l}\|_{C_l}^{2}}{D_l}
\]

와 그 signed increment

\[
\Delta\Psi_l(y_l)
=m_{t,s,l}y_l+\frac{y_l^2}{D_l}
\]

는 diagnostic으로 계속 기록한다. \(\Delta\Psi_l<0\)은 base weight 쪽 cancellation일 수
있지만 past-edit retention을 뜻하지 않는다. C3에서 \(\Psi\)/concentration 감소와
functional retention이 분리됐으므로, 첫 main track에서는 signed \(\Psi\)를 reward하거나
hard capacity envelope로 사용하지 않는다. Signed displacement objective와 hard barrier는
100-edit에서 \(\Omega\)-retention 정렬이 확인된 뒤 ablation으로만 연다.

## 6. ODE-Edit state equation

Continuous-time 표현은

\[
\frac{dW_{t,l}(\tau)}{d\tau}
=v_{t,l}(\tau)
\widehat B_l(W_t(\tau),Z_t^\star),
\qquad v_{t,l}(\tau)\ge0
\]

이다.

Accepted discrete step은

\[
W_{t,s+1,l}
=W_{t,s,l}+y_{t,s,l}\widehat B_{t,s,l},
\]

\[
y_{t,s,l}=h_{t,s}v_{t,s,l}
\]

로 쓴다. 실제 applied coefficient는 \(y\)이며, 이를 구한 뒤 별도 normalization하지
않는다. Applied magnitude와 share는 결과적으로

\[
h_{t,s}^{\rm applied}=\|y_{t,s}\|_2,
\qquad
s_{t,s}=\frac{y_{t,s}}{\|y_{t,s}\|_2}
\]

로 해석한다.

## 7. Actual applied-step controller

각 waypoint에서 실제 적용할 coefficient \(y\)를 직접 푼다. Progress slack과
load objective를 한 scalar objective에 섞지 않고 feasible progress와 minimum
monotone-load cost를 두 단계로 분리한다.

\[
\mathcal F_s(h_s)
=\left\{
y\ge0:
\|y\|_2\le h_s,\quad
y_l=0\ \text{if }a_{s,l}\le\epsilon_a
\right\}.
\]

마지막 support constraint는 positive predicted rewrite slope가 없는 actuator에 historical
load만을 이유로 weight를 쓰는 것을 막는다.

먼저 current trust radius 안에서 낼 수 있는 최대 predicted progress를 계산한다.

\[
\bar p_s
=\max_{y\in\mathcal F_s(h_s)}a_s^Ty.
\]

\(\bar p_s\le\epsilon_p\)이면 positive local actuator가 없는 것으로 판정하고 step을
적용하지 않는다. 그렇지 않으면 model-common \(\beta\in(0,1)\)로 feasible interior를
남겨 requested progress를 정한다.

\[
p_s
=\min\left\{
\kappa[\Phi_{\rm event}(W_s)]_+,\quad
\beta\bar p_s
\right\}.
\]

그 뒤 실제 applied coefficient를 minimum-load QP로 구한다.

\[
\begin{aligned}
\min_{y\in\mathcal F_s(h_s)}\quad &
\sum_l(1+\Omega_{t,l})\frac{y_l^2}{D_l},\\
\text{s.t.}\quad &
a_s^Ty=p_s.
\end{aligned}
\]

구현에서는 solver tolerance \(\epsilon_{\rm qp}\) 안의 equality band를 사용한다.
`>=`만 두면 required progress를 넘는 coefficient가 들어갈 수 있으므로 main arm에서는
허용하지 않는다. \(h_s\), \(\beta\), solver tolerance와 failure fallback은 두 모델에
공통으로 preregister한다. Standard accepted step에는 progress slack을 두지 않으며
infeasibility를 silent accept하지 않는다.

QP가 반환한 \(y\)를 그대로 trial에 적용한다. Share를 구한 뒤 native/global norm으로
radial amplification하거나 별도 normalization하지 않는다.

이 구조에서만 다음 local claim을 열 수 있다.

> 고정된 current-state actuator와 local quadratic model 아래에서 accepted applied
> update는 required rewrite progress를 만족하는 feasible layer allocation 중
> monotone historical-load-weighted write energy를 최소화한다.

## 8. Adaptive trust, reject, rollback

Trial step의 predicted progress는

\[
\widehat{\Delta\Phi}_s=a_s^Ty_s
\]

이고 actual progress는

\[
\Delta\Phi_s
=\widetilde\Phi_{\rm ctrl}(W_s)-
\widetilde\Phi_{\rm ctrl}(W_s+y_s\widehat B_s)
\]

다. Trust ratio를

\[
r_s
=\frac{\Delta\Phi_s}
{\widehat{\Delta\Phi}_s+\epsilon_r}
\]

로 둔다.

Controller contract는 model-common threshold로 preregister한다. Smooth surrogate가
개선되더라도 hard event deficit이 tolerance 밖으로 악화되는 trial은 accept하지 않는다.

- \(r_s<\eta_{\rm reject}\), \(\Delta\Phi_s\le0\), 또는
  \(\Phi_{\rm event}(W_{\rm trial})>\Phi_{\rm event}(W_s)+\epsilon_{\rm event}\): exact rollback,
  \(h_s\leftarrow\gamma_\downarrow h_s\)
- accept 범위: commit하고 다음 \(h\) 유지 또는 축소
- 높은 trust: commit하고
  \(h_{s+1}\leftarrow\gamma_\uparrow h_s\)

Accepted step만 parameter lineage에 반영한다. \(\Omega\)와 Alpha history는 outer edit의
first-hit terminal에서 정확히 한 번 반영한다. Rejected step은 weight hash, RNG contract,
history가 entry state와 정확히 같아야 한다.

## 9. First-hitting terminal과 adaptive K

Native endpoint utility 또는 native C-distance가 terminal condition이 아니다.

\[
T_t^\star
=\inf\{\tau:\Phi_{\rm event}(W_t(\tau))\le0\}.
\]

Discrete accepted round 수는

\[
K_t
=\inf\{s:\Phi_{\rm event}(W_{t,s})\le0\}
\]

로 trajectory 결과가 된다.

- 첫 trial에서 성공하면 즉시 종료한다.
- field가 nonlinear하면 현재 state에서 다시 proposal/controller를 푼다.
- rejected trial은 \(K_t\)의 accepted count에 포함하지 않는다.
- \(S_{\max}\)까지 성공하지 못하면 명시적 fail/fallback으로 종료한다.
- native \(D\)를 끝까지 소비하지 않는다.

이 free terminal이 들어가야 동일 semantic target을 만족하는 여러 endpoint 중 excess
write가 적은 endpoint를 선택한다는 ODE-Edit claim이 성립한다.

## 10. MEMIT과 AlphaEdit instantiation

Controller와 integrator는 actuator family와 분리한다.

### MEMIT actuator

- current state에서 current residual/key를 사용한 same-snapshot low-rank proposal
- precomputed covariance를 C metric과 solver에 read-only 사용

### AlphaEdit actuator

- 동일 current residual/key를 사용
- canonical precomputed projector 안에서 proposal solve
- accepted past-edit key history를 solver에 반영
- history append는 outer edit의 accepted terminal 후 정확히 한 번

Alpha projection은 preservation 결과 자체가 아니라 admissible subspace다. ODE-Edit
controller는
그 subspace 안에서 layer별 progress/capacity trade-off를 결정한다.

첫 10/100-edit main track은 ODE identity를 projection/history confound 없이 빠르게
식별하기 위해 MEMIT만 사용한다. AlphaEdit은 MEMIT 100-edit에서 Full ODE-Edit이 best
simple/static baseline보다 살아남은 뒤 extension으로 연다. Extension에서도 두 family에
서로 다른 threshold, sign rule 또는 architecture-specific fallback을 두지 않는다.

## 11. Runtime invariant

구현과 evaluator는 최소한 다음을 강제해야 한다.

1. direct-z compute exactly once/edit
2. 모든 layer proposal은 동일 current snapshot에서 생성
3. proposal 생성 중 weight mutation 없음
4. current residual/key/proposal/slope/load/share의 state ID와 hash 기록
5. evaluation prompt/outcome는 controller action 확정 전 unavailable
6. actual applied coefficient가 QP output과 동일; 사후 radial rescale 없음
7. applied step에서 smooth progress와 hard-event non-worsening을 다시 검증
8. rejected trial은 parameter/history/RNG lineage 무변경
9. first-hit 이후 write 없음
10. \(\Omega\)는 completed outer edit의 terminal net write로 한 번만 증가
11. accepted edit만 outer history에 append
12. covariance와 후속 Alpha projector/Wikipedia artifact read-only 재사용
13. EasyEdit source 미수정

## 12. ODE identity와 필수 ablation

ODE는 명칭으로 유지되는 것이 아니라 state dependence가 performance에 필요함을 보여야
한다. 첫 10/100-edit main track의 다섯 arm은 다음으로 제한한다.

| Arm | 실행 계약 | 반론 또는 원인 |
|---|---|---|
| Native MEMIT | canonical ordered full write | EasyEdit baseline |
| Scalar first-hit | entry에서 계산한 native MEMIT multi-layer endpoint direction을 freeze하고 \(\alpha\in[0,1]\)에서 같은 event의 first hit | 단순히 native를 덜 쓰면 되는가 |
| Static synchronous | entry same-snapshot proposal과 allocation을 freeze하고 같은 trust/first-hit 적용 | initial static routing이면 충분한가 |
| Ordered adaptive | canonical ascending layer를 cyclic하게 하나씩 허용하고 매 coordinate 뒤 current state를 refresh; 같은 progress/load/trust/event 사용 | 좋은 controller인가, joint synchronous execution인가 |
| Full ODE-Edit | 모든 layer를 same snapshot에서 refresh하고 joint coefficient를 solve/apply | full method |

Scalar first-hit의 \(\Delta W_{\rm MEMIT}\)은 entry에서 canonical ordered MEMIT을 한 번
실행해 얻은 layer별 terminal write를 원상복구한 뒤 freeze한다. Prelocked increasing
\(\alpha\) grid에서 첫 event-hit bracket을 찾고 그 bracket 안에서만 bisection한다.
Evaluation metric을 보며 \(\alpha\)를 고르지 않는다. \([0,1]\) 안에서 hit를 찾지
못하면 명시적 fail이며 \(\alpha>1\) rescue를 허용하지 않는다. Event가
non-monotone이면 그 사실과 grid resolution을 함께 보고한다.

Ordered adaptive는 ODE-Edit처럼 이미 방문한 layer도 다음 cycle에서 재평가할 수 있지만,
한 micro-step에는 canonical order의 현재 layer 하나만 \(y_l>0\)일 수 있다. 이 비교로
state refresh/controller 효과와 simultaneous joint allocation 효과를 분리한다.

Main table 뒤 mechanism appendix에만 다음을 연다.

- Frozen K-split: fixed update subdivision negative control
- dynamic-share-only와 dynamic-direction-only
- all-vocabulary top-1 event
- signed \(\Psi\) objective와 hard capacity barrier
- AlphaEdit actuator extension

핵심 판정은 세 비교다.

\[
\text{Full}-\text{Scalar first-hit}
\quad\text{(단순 under-write 반론)},
\]

\[
\text{Full}-\text{Static synchronous}
\quad\text{(ODE refresh 필요성)},
\]

\[
\text{Full}-\text{Ordered adaptive}
\quad\text{(joint synchronous execution 필요성)}.
\]

모든 adaptive arm은 같은 terminal, allowed information, trust rule과 compute accounting을
사용한다. Full이 Static과 같으면 ODE necessity를 kill하고, Full이 Ordered와 같으면
joint synchronous claim을 제거하며, Scalar가 Full과 같으면 scalar early-stop으로
pivot한다.

## 13. Non-stationarity panel

각 accepted/rejected waypoint에 다음을 기록한다.

- residual norm과 cosine: \(E_{s+1}\) vs \(E_s\)
- layer key C/Euclidean cosine
- layer actuator C-cosine:
  \(\cos_C(B_l^{(s+1)},B_l^{(s)})\)
- rewrite efficiency vector의 Spearman/Kendall ranking turnover
- share cosine와 active support change
- monotone outer-edit load \(\Omega_l\), signed displacement \(\Psi_l\), max-load share와 Gini
- active constraint와 QP equality residual
- predicted/actual progress와 trust ratio
- first-hit, accepted/rejected flag
- selected layer pair \(i,j\)의 ordered non-commutativity:
  \(\|\mathcal U_i(\mathcal U_j(W))-\mathcal U_j(\mathcal U_i(W))\|_C\)

단순히 hash가 다르다는 사실이 아니라 field가 얼마나 변했고 그 refresh가 frontier 개선을
매개했는지 분석한다.

## 14. Direct-z fidelity panel

Rewrite utility를 direct-z fidelity와 동일시하지 않는다. Native와 각 ablation에 대해
다음을 보고한다.

\[
R_z
=\frac{\|H^L(W_T)-Z^\star\|}
{\|H^L(W_0)-Z^\star\|+\epsilon},
\]

\[
C_z=\cos(H^L(W_T),Z^\star),
\]

그리고 허용 prefix별 residual distribution, target-token margin, exact controller-side
rewrite event success를
함께 보고한다. Direct-z fidelity가 높아도 preservation이 좋아진다는 보장은 없으므로
efficacy–fidelity–retention의 3축 frontier를 사용한다. 다만 첫 main table에서는
direct-z fidelity를 mechanism diagnostic으로 보고하며 main success gate로 사용하지 않는다.

## 15. Complexity와 fallback

보고할 compute는 다음을 포함한다.

- direct-z optimization count/time
- proposal build/NFE per trial
- accepted/rejected round 수
- QP time
- evaluation을 제외한 controller wall time
- total GPU-hours와 peak memory
- first-hit이 제거한 proposal/write 수

평균 accepted macro-round 목표는 proposal의 `1.5--2`, hard maximum은 별도
preregister한다. No-positive-slope, repeated trust rejection, \(S_{\max}\) 도달,
native/scalar event failure를 서로 다른 failure type으로 기록한다. Fallback은 scientific
arm을 숨기지 않도록 별도 결과로 센다.

## 16. Fast main-table progression gate

### MT — Technical identity

Deterministic contract test 뒤 두 모델×MEMIT fresh 1--2 edits만 실행한다. Native와 Full
path를 실제 GPU에서 확인하고, 나머지 baseline은 동일 artifact schema/unit contract를
통과해야 한다.

- post-QP rescale 0; applied coefficient와 solver output exact
- active-slope support와 progress-equality residual이 tolerance 안
- negative-gain/hard-event-worsening accepted step 0
- rejected-step parameter/history/RNG rollback과 hash exact
- direct-z once/edit, same-snapshot proposal, information firewall pass
- first-hit 이후 proposal/write 0
- \(\Omega\) outer-terminal single append
- 동일 common policy on both models
- EasyEdit와 precomputed covariance read-only

하나라도 실패하면 outcome을 해석하지 않고 기술 수리 후 MT만 반복한다.

### MI — 10-edit ODE identity table

- `Llama3-8B-Instruct`와 `Qwen2.5-7B-Instruct`
- MEMIT only, fresh 10 edits, 공유된 두 locked order
- Native, Scalar first-hit, Static synchronous, Ordered adaptive, Full ODE-Edit
- 동일 request/order/event/controller threshold와 evaluation code
- proposal drift, ranking turnover, allocation drift, ordered non-commutativity panel
- NFE/edit와 wall time/edit 포함

100-edit open을 위한 lenient gate는 다음이다.

1. Full의 current acquisition이 Native 대비 common locked tolerance 안에서 non-collapse
2. `Full - Scalar`와 `Full - Static`의 prior-retention AUC 방향이 두 모델 어느 쪽에서도
   material하게 음수가 아니고 pooled paired direction은 strict positive
3. `Full - Ordered`가 두 모델 공통으로 material하게 열세가 아님
4. proposal/ranking/allocation drift가 numerical noise를 넘음
5. 평균 proposal refresh가 `<=2`이거나 추가 NFE를 정당화하는 retention frontier signal

두 모델 중 하나만 살리는 threshold, event, step, sign 또는 fallback은 금지한다.
Full이 Scalar/Static보다 두 모델 공통으로 열세면 ODE를 kill한다. Full과 Static이
사실상 같으면 static routing, Full과 Ordered가 같으면 joint synchronous claim 제거,
Full과 Scalar가 같으면 scalar early-stop으로 각각 pivot한다.

Exact tolerance, trust threshold, scalar search, \(S_{\max}\), order와 case ID는 결과를
보기 전 execution spec에 고정한다.

### MS — 100-edit first main table

MI gate 통과 뒤 동일 두 모델×MEMIT×다섯 arm×두 locked order로 바로 확장한다. 첫 main
table의 canonical column은 다음이다.

| Method | Current rewrite ↑ | Prior-retention AUC ↑ | Final prior retention ↑ | Paraphrase ↑ | Neighborhood ↑ | Max-layer \(\Omega\) load ↓ | NFE/edit ↓ | Time/edit ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Checkpoint별 lightweight fixed downstream proxy와 perplexity는 보조 표로 두고 full
downstream suite는 미룬다. \(\Psi\), direct-z fidelity, Gini와 endpoint displacement는
mechanism diagnostic이며 main success를 대신하지 않는다.

Main success는 current acquisition non-collapse, Scalar/Static 대비 prior-retention
AUC와 final retention의 두 모델 공통 개선 방향, paraphrase/neighborhood non-worsening,
감당 가능한 NFE/time을 함께 요구한다. \(\Omega\) concentration 감소가 retention과
정렬되지 않으면 load geometry claim을 kill하거나 재정의한다.

### MA — AlphaEdit extension

MEMIT 100-edit에서 Full ODE-Edit이 Scalar/Static baseline보다 살아남은 뒤에만 canonical
projector/history AlphaEdit 100-edit를 연다. Projector와 Wikipedia artifact는 read-only로
재사용하며 controller rule은 MEMIT과 동일하다.

### ML — Large/lifelong

MS와 필요한 MA를 통과한 뒤에만 1K+를 열고, 10K/full stream은 별도 승인한다. 그 전에는
lifelong superiority, preservation guarantee 또는 deployable efficiency를 열지 않는다.

## 17. Local guarantee와 claim boundary

Method가 올바르게 구현되면 열 수 있는 local claim은 제한적이다.

- fixed current-state actuators, feasible-progress target \(p_s\), local quadratic model
  아래에서 actual applied update의 monotone-load-weighted energy optimality
- accepted step의 smooth controller deficit 감소와 hard event deficit non-worsening
- first-hit 이후 excess write 없음

다음은 empirical evidence가 없으면 열지 않는다.

- global convergence
- direct-z fidelity가 preservation을 보장한다는 주장
- static synchronous 대비 ODE necessity
- native 대비 retention/downstream 우위
- lifelong collapse 방지

## 18. Method opening statement

> ODE-Edit은 edit당 direct-z target을 고정하되 current model state에서 editor actuator
> field를 반복적으로 재선형화한다. 각 waypoint에서는 rewrite progress, trust region,
> monotone outer-edit load를 함께 고려해 실제 적용할 layer coefficient를 직접 선택하고,
> baseline-defined update budget을 소비하는 대신 rewrite goal을 처음 만족하는 상태에서
> 종료한다.
