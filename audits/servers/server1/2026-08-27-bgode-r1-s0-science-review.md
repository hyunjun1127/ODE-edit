# BGODE-R1 — Sequence-Event Barrier-Guided ODE AlphaEdit Flow

- instruction: `ODEEDIT-S05-BGODE-R1-SEQUENCE-EVENT-BARRIER-GUIDED-ODE-SCIENCE-R1`
- family / registry: `BGODE` / `BGODE-R1`
- display name: **Sequence-Event Barrier-Guided ODE AlphaEdit Flow**
- 단계: `S0_CPU_SCIENCE_CORE_PASS; MODEL_GPU_SLURM_HOLD`
- 기존 문서 관계: 이 문서는 BGODE-R1의 새 과학 명세이며
  [`04-method-design.md`](04-method-design.md)를 덮어쓰거나 수정하지 않는다.

## 1. 문제와 가장 중요한 제한

BGODE-R1은 한 atomic edit의 source sequence를 target sequence로 바꾸는 진행률을
first-departure event distribution에서 정의하고, AlphaEdit가 제공하는 다섯 ordered
factor 방향 안에서 그 진행률을 정확한 1차 equality로 만든다. 그 equality를 만족하는
속도 중 moving forward-KL barrier의 instantaneous derivative와 categorical Fisher
kinetic cost의 합을 최소화한다.

BGODE-R1의 atomic batch size는 **정확히 `B=1`**이다. 요청마다 진행 민감도
`a_i`가 다르므로 `B>1`에서 하나의 공용 5-layer coefficient `u`가 모든
`a_i^T u=1`을 만족한다고 일반적으로 말할 수 없다. 하나의 scalar localization `rho`도
여러 nonlinear request progress를 동시에 맞출 수 없다. 따라서 `B>1` 호출은 성능이
나쁜 configuration이 아니라 typed scientific boundary다. Request-labelled mixture 또는
aggregate progress law가 정의되기 전의 batched method는 future `BGODE-R2`다.

## 2. 네 종류의 명제

| 구분 | BGODE-R1 S0의 내용 |
|---|---|
| 사용자/GH가 잠근 결정 | `B=1`, fixed native `z*`, physical `W(t)`, raw joint event probability, equality-only Rayleighian, genuine P-inside AlphaEdit factors |
| 수학/CPU test로 닫힌 lemma | trie partition normalization, forward-KL reference, on-manifold decomposition, Fisher PSD, equality KKT, `beta=h u`, affine frozen-field identity |
| 구현 approximation | native ordered rollout은 earlier factor를 coefficient 1로 적용해 later factor를 만든 dictionary이며 controlled prefix와 다를 수 있음 |
| 모델 실험으로만 검증할 hypothesis | Full barrier의 load-bearing 여부, dynamic relinearization 이득, empirical rewrite/generalization/locality/retention |

## 3. State와 AlphaEdit actuator dictionary

Atomic edit entry를 `W0`, request를 `R=(x,S,Y)`라 한다.

\[
z^\star=\operatorname{ComputeZ}_{native}(W_0,R)
\]

는 정확히 한 번 계산하고 Euler trajectory 내 recompute count는 0이다. ODE state는
`z*`가 아니라 `W(t)`다. Node `n`에서는 기존
`AlphaEditProposalAdapter.propose_ordered` 인터페이스가 genuine P-inside solve로 만든

\[
\mathcal B_n=\{B_{n,1},\ldots,B_{n,5}\}
\]

를 소비한다. BGODE 코어는 AlphaEdit normal equation을 복사하거나 EasyEdit를 수정하지
않는다. Dictionary build 중 temporary unit-prefix writes는 반환 전에 exact restore되어야
하고 history는 entry snapshot으로 고정된다. Euler node append는 0, future accepted terminal
append만 1이다.

이 dictionary는 **native-ordered-rollout-derived actuator dictionary**다. Actual action의
`beta_j=h rho u_j`와 내부 unit-prefix가 다르므로 predictor mismatch는 outer `h`를
줄인다고 자동 소멸하지 않는다.

## 4. Joint first-departure partition

고정 termination token `b`를 붙인 `Sbar=(S,b)`, `Ybar=(Y,b)`로 joint trie를 만든다.
Event는 exact source leaf `eS`, exact target leaf `eY`, 각 internal prefix `u`에서 trie
child가 아닌 token `v`를 처음 고르는 `e_(u,v)`다. Off-child 이후 suffix는 marginalize한다.

\[
\pi_W(e_Y)=\prod_i p_W(\bar y_i\mid x,\bar y_{<i}),\quad
\pi_W(e_S)=\prod_i p_W(\bar s_i\mid x,\bar s_{<i})
\]

\[
\pi_W(e_{u,v})=P_W(u\mid x)p_W(v\mid x,u).
\]

이는 full suffix completion distribution이 아니라 completion space의 disjoint
first-departure partition이다. Raw joint probabilities만 사용하며 length normalization은
0이다. S0 brute-force evaluator는 tiny vocabulary에서 모든 off-child event를 materialize해
`logsumexp(log pi)=0`을 검사한다. Production vocabulary reducer는 동일 node semantics를
유지하되 event Python object 전체를 만들지 않는 별도 interface다.

Source와 target tokenization이 같으면 두 distinguished leaf를 만들 수 없으므로 reject한다.
Boundary token이 distinguished sequence 내부에 이미 나타나도 reject한다. Termination
convention은 endpoint 결과로 고르는 hyperparameter가 아니며 S1 전에 semantic protocol로
고정해야 한다.

## 5. Single-coordinate reference와 barrier

\[
r(W)=\log\pi_W(e_Y)-\log\pi_W(e_S),\qquad \dot r=1.
\]

`pY0=pi0(eY)`, `pS0=pi0(eS)`, `c0=pY0+pS0`, `r0=log(pY0/pS0)`라 두면

\[
\pi_t^\star(e_Y)=c_0\sigma(r_0+t),\qquad
\pi_t^\star(e_S)=c_0[1-\sigma(r_0+t)],
\]

\[
\pi_t^\star(e)=\pi_0(e),\quad e\notin\{e_Y,e_S\}.
\]

이는 `log pY/pS=r0+t` 제약 아래 `KL(pi0 || pi)`의 forward-KL optimum이다. Global
potential은 항상

\[
\mathcal B(W,t)=D_{KL}(\pi_t^\star\Vert\pi_W)
\]

다. `KL(pi0||piW)-Dmin(t)=B(W,t)`는 exact progress manifold에서만 성립한다. S0 test는
on-manifold identity와 off-manifold 반례를 별도로 고정한다.

Pair-mass ceiling `pY(t)<=c0`는 diagnostic이다. R1 core에 pair-mass equality 또는
two-coordinate flow를 넣지 않는다. Two-coordinate case에서 single-coordinate KL
identity를 복사하면 outside conditional term의 weight가 달라지는 반례도 고정했다.

## 6. Event score와 Fisher pullback

각 event와 ordered actuator에 대해

\[
S_{e,j}=D\log\pi_W(e)[B_j],\qquad
a_j=S_{Y,j}-S_{S,j}.
\]

\[
G=\sum_e\pi_W(e)S_eS_e^T,qquad
g_t=-\sum_e\pi_t^\star(e)S_e.
\]

`G`는 categorical Fisher–Gauss–Newton pullback이며 anchored KL의 exact Hessian이라고
부르지 않는다. Test는 autograd Jacobian과 central finite difference로 `S`를 확인하고,
`sum pi S≈0`, `G=G^T`, `G>=0`, `g0(W0)≈0`을 검증한다.

`g_t-g_0=-delta_t a`이므로 exact equality `a^T u=1` 위에서는 두 gradient가 objective에
상수 차이만 만든다. Progress를 penalty로 바꾸면 optimizer equivalence가 깨지는 negative
test가 있다.

## 7. 왜 이 속도인가

BGODE-R1 속도는 heuristic Euler velocity가 아니라 다음 constrained Rayleighian의
유일한 numerical solution이다.

\[
u^\star=\arg\min_u\left[g_t^Tu+\frac12u^TGu\right]
\quad\text{s.t.}\quad a^Tu=1.
\]

이는 prescribed sequence-log-odds progress를 내는 속도 중 instantaneous barrier
derivative와 Fisher kinetic cost를 함께 최소화한다. Finite action은

\[
\beta_h=h u^\star
\]

다. Unscaled beta-QP는 `h→0`에서 action이 사라지지 않을 수 있어 금지한다.

Solver는 symmetric eigendecomposition의 FP32 scale과 machine epsilon에서 deterministic
pinv tolerance를 정한다. `g,a in range(G)`, positive equality Schur denominator/full rank,
PSD, equality residual, KKT stationarity residual, finite objective를 모두 검사한다. Damping,
ridge, cap/sign QP, pair-mass equality, retry/fallback은 없다. 실패는 typed
`NumericalRankBoundary`다.

### 명시적 nonclaim

이 optimization은 barrier의 monotone decrease, classical CBF forward invariance, global
locality 또는 convergence를 증명하지 않는다. Exact path/W0에서는 Full과 Fisher-only가
같을 수 있다. Full barrier가 load-bearing이라는 말은 off-reference collateral drift에서
matched progress/action diagnostics 아래 Full이 Fisher-only와 Plain보다 excess를 실제로
줄일 때만 가능하다.

## 8. H1–H3와 S0 증거

### H1 EVENT VALIDITY

Prefix/non-prefix, 두 boundary token variant, extreme logits에서 partition은 disjoint하고
log-space sum이 1이다. 이로 synthetic correctness는 닫혔지만 model tokenizer termination
semantics는 아직 open이다.

### H2 BARRIER ATTRIBUTION

고정 categorical synthetic case에서 off-reference state의 한 local step 결과는
`KL(Full)=0.01885999`, `KL(Fisher)=0.02280380`, `KL(Plain)=0.02226517`이었다. Full velocity와
Fisher velocity도 달랐다. 이는 attribution implementation이 구분 가능한지를 증명하는
fixture이지 Llama/AlphaEdit에서 Full이 이긴다는 evidence가 아니다.

### H3 ODE ATTRIBUTION

Affine/frozen field에서 one-step과 any subcycle Euler는 같다. State-dependent categorical
geometry에서는 dynamic relinearization과 frozen split endpoint가 달랐고, fixed-time
Euler의 `N=1,2,4,8,16` endpoint error가 `N=128` reference 쪽으로 단조 감소했다. 따라서
H3를 판별할 구현은 준비됐지만 model field에서 이득이 있는지는 미검증이다.

## 9. Native reduction과 금지 입력

`N=1, barrier-off`는 controller/root를 거치지 않고 `u_j=1,rho=1`을 반환하는 explicit
bypass다. 다른 configuration은 이 branch를 호출할 수 없다.

Controller 입력은 canonical rewrite prompt, source/target, pre-edit model distribution,
fixed native target, inherited read-only P/history뿐이다. Locality/neighborhood/heldout/evaluator,
external teacher/calibration, length-normalized score의 influence count는 모두 0이다.

## 10. 모델 실행 전 열린 정의

1. **termination semantics:** EOS/EOT, delimiter 등 하나의 semantic protocol과 tokenizer
   mapping. Performance로 선택 금지.
2. **B=1 sample unit:** sealed canonical order에서 어떤 exact single request를 pilot unit으로
   봉인할지.
3. **`T` when native `T_AE<=0`:** 이미 target log-odds가 높거나 native endpoint가 progress를
   역행할 때 horizon을 임의 양수화하지 않는 typed rule.
4. **JVP runtime:** node×vocabulary×5 actuator score를 serial JVP, batched tangent, hook 중 어떤
   동일성-검증 backend로 계산할지.
5. **ordered dictionary compute cost:** current adapter의 layer별 key/terminal capture가 runtime
   graph에서 실제 몇 physical forward인지 profiler 없이 확정하지 않음.

이 정의들이 봉인되기 전 model load와 S1 submission은 HOLD다.

## 11. Future S1 최소 pilot 제안 — 실행 금지

### Outcome-free sample rule

Sealed canonical request order의 첫 request를 그대로 사용한다. Tokenization 후 source와
target이 동일하거나 frozen boundary token을 내부에 포함하면 이를 결과 기반으로 대체하지
않고 `S1_SAMPLE_SEMANTIC_BOUNDARY`로 보고한다. Success/NLL/native endpoint 결과를 보고 다음
request를 고르지 않는다. 한 request가므로 모든 arm은 exact same `W0`, request bytes,
tokenization, context, evaluator-off controller를 공유한다.

### 최소 attribution matrix

| Arm | Dictionary/geometry refresh | Barrier gradient | 목적 |
|---|---|---|---|
| Native AlphaEdit | native ordered one-shot | 없음 | official fidelity/reference |
| Plain dynamic | node마다 dictionary refresh, native/equal coefficient | 없음 | dynamic writer 자체 |
| Fisher-only | node마다 `B,G,a` refresh | `g=0` | information allocation |
| Full barrier | node마다 `B,G,a,g` refresh | moving `g_t` | H2 |
| One-step barrier | entry dictionary/geometry 한 번 | entry Full | one-step control |
| Frozen-field split | entry `B,G,a,g` 고정, 동일 T 분할 | entry Full | 단순 subdivision negative control |

이는 H1–H3에 필요한 6-arm matrix다. Pair-mass/two-coordinate/cap/sign/token-level/full-vocab
ablation은 R1 최소 pilot에 넣지 않는다. Scientific promotion은 자동 0이다.

## 12. 현재 판정

`S0_CPU_SCIENCE_CORE_PASS; H1_H2_H3_TESTABLE; B1_ONLY; S1_MODEL_EXECUTION_HOLD`

관련 red-team 문서:
[`2026-08-27-bgode-r1-s0-science-red-team.md`](2026-08-27-bgode-r1-s0-science-red-team.md)
