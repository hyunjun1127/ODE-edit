# Ordered Response-Barrier ODE-Edit

## Full-residual official-form actuators with ordered realized-state feedback

| 항목 | 내용 |
|---|---|
| 기준일 | 2026-09-03 |
| 문서 상태 | Current primary research proposal; falsification-first; implementation not yet authorized |
| 방법명 | **Ordered Response-Barrier ODE-Edit** |
| 적용 위치 | Official MEMIT/AlphaEdit의 multi-layer weight-write stage |
| 고정 대상 | Entry state에서 baseline이 한 번 계산한 \(z^\star\), request/token order, covariance, projector, committed history/cache |
| 변화 대상 | Current virtual model에서 재구성되는 residual, key, official-form writer direction, terminal response와 layer workload |
| 주 적분 | Architectural shallow-to-deep ordered Lie--Trotter splitting, multi-sweep |
| 주 제어 | Target-realization potential의 current directional response로 계산한 \(h\)-independent velocity multiplier |
| primary horizon | \(T_{\max}=Nh=1\), \(N=4\), \(h=0.25\) |
| action geometry | Controller constraint가 아닌 mechanism telemetry only |
| terminal | Target-new strict batch first-hit; 미도달 시 (T_{\max}=1) 또는 `FLOW_STALLED` finite endpoint commit |
| transaction | Non-no-op terminal에서 materialization/commit 각 1회; AlphaEdit history append 1회, MEMIT static COV mutation 0; entry-hit no-op은 모두 0회 |
| controller 비입력 | External causal/reference bank, rephrase/neighborhood/locality outcome, prior-edit replay, learned selector, reward/probe |
| 현재 claim | Mechanism proposal only; barrier/ODE/locality/retention benefit는 아직 미검증 |
| 원안 identity | `5e7c02dcdba3e7523b1b3c2f838e2c228be0aa32d649a79b58d5f96459a683c5` / 41,195 bytes / 1,667 lines |
| final-seal review input identity | `1444cde58af069edb3abed2ab88dfd4faf1c5609b8587c09f0e29e80b30f6825` / 14,957 bytes / 597 lines |

> **한 문장 정의**
>
> Ordered Response-Barrier ODE-Edit은 prescribed remaining-layer residual share를 제거하고,
> full residual로 만든 official-form closed-form writer를 current-state actuator로 사용한다.
> 각 actuator의 terminal activation response가 target-realization potential을 줄이는 범위에서
> velocity를 정하고, 실제 upstream transition 이후 downstream residual, key, response와 writer를
> 모두 다시 구성하면서 shallow-to-deep 순서로 적분한다.

---

# 0. 설계 판정과 원안 교정

제안의 중심 동기와 비교 구도는 채택한다. 다만 원안의 step coefficient를 그대로 구현하면
동일한 ODE의 수치 적분이 되지 않으므로, 본 문서가 canonical proposal이 되기 전에 다음을
교정한다.

1. 원안의
   \[
   \alpha^\star
   =
   \frac{[\langle\bar R,\bar\psi\rangle]_+}{\|\bar\psi\|^2}
   \]
   는 약한 response를 줄이는 계수가 아니라 **linearized inverse-gain line minimizer**다.
   \(\psi=\rho R\)이면 \(\alpha^\star=1/\rho\)이므로, 약한 positive response에서는 오히려
   커진다.
2. \(\alpha=\min\{h,h/\sqrt{q_{\rm rel}},\alpha^\star\}\)를 쓰면 \(h\to0\)에서
   response term이 effective vector field에서 사라진다. 본 proposal은 먼저
   \(h\)-독립적인 multiplier \(u(W)\)를 정의하고 \(\alpha=hu(W)\)로 적분한다.
3. \(V_z\)는 classical hard safety barrier보다 target-tracking Lyapunov potential에 가깝다.
   본문의 `barrier`는 entry sublevel set을 향한 **local one-sided response barrier**라는 제한된
   뜻으로만 사용한다. Finite Euler iterate의 invariance는 주장하지 않는다.
4. Action geometry와 terminal residual energy의 직접 비율은 dimensionally 충분히 닫혀 있지
   않다. 따라서 action \(q\)는 main controller와 terminal predicate에서 완전히 제거하고
   mechanism telemetry로만 둔다. Action-aware controller는 별도 future ablation이다.
5. `full-residual native writer`라는 표현은 쓰지 않는다. Divisor를 1로 바꾸는 순간 official
   method 자체가 아니므로 **official-form full-residual actuator**라고 부른다.
6. Convergence는 \(T_{\max}=Nh=1\)을 고정하고 \(h\)를 반으로 줄일 때 full ordered sweep 수
   \(N\)을 두 배로 늘려 검증한다. Semantic first-hit은 이 fixed-horizon convergence gate와
   분리한다.
7. Main numerical seal은 \(T_{\max}=1\), \(N=4\), \(h=0.25\)다. \(N=2\)는 compute-reduced
   ablation, \(N=8\)은 B1/B10 refinement audit으로만 사용한다.
8. Output/history/action barrier, QP, line search, retry와 fallback은 main에서 제거한다. Additional
   reference bank 없이 locality/retention은 evaluation-only로 둔다.

이 여덟 교정은 원안의 핵심을 약화시키는 것이 아니라, response control, ODE claim과
barrier claim을 실제로 검증 가능한 형태로 만든다.

---

# 1. Proposal lineage와 supersession

본 문서는 다음 방향을 current primary research contract로 제안한다.

- [FzCB-Edit method pivot](2026-08-31-fzcb-edit-method-pivot-proposal.md)의 fixed \(z^\star\),
  official writer geometry, immutable inner history와 terminal one-shot transaction 자산은
  계승한다.
- FzCB의 hard homotopy equality, spent-plus-suffix completion budget, equality-null QCQP,
  predictor--corrector/backtracking은 본 method의 과학 수식으로 계승하지 않는다.
- [Fixed-z Functional Safe Write](2026-08-30-fixed-z-functional-safe-write-proposal.md)의
  same-\(z\) write-stage 문제의식은 역사적 선행이다. Functional anchor bank, KL constraint와
  accept/reject controller는 본 method에서 제거한다.
- [이전 dynamic-layer method](ODE_BF_Dynamic_Layer_Proposal.md)의 virtual low-rank state,
  state-dependent rebuild와 transaction accounting은 구현 자산으로만 재사용한다.
- [BGODE-R2 section](sections/05-bgode-r2-rho-free-prefix-event-fp64-two-equality.md)의
  FP64 reduction, singular-aware numerical boundary와 ordered-dictionary fidelity는 engineering
  precedent다. Prefix-event/two-equality controller는 본 수식의 근거가 아니다.

기존 proposal, source, result와 report byte는 수정하거나 재해석하지 않는다. Supersession은
새 실험의 research direction에만 적용된다.

---

# 2. Empirical motivation과 evidence boundary

## 2.1 Official writer의 세 객체

편집 layer를 \(\ell_1<\cdots<\ell_m\)이라 하고 \(n_j=m-j+1\)이라 하자. Official
multi-layer writer는 대략 다음 세 객체를 연결한다.

\[
R_j=z^\star-H(W^{j-1}),
\qquad
A_j=\frac{R_j}{n_j},
\]

\[
\Delta W_j
=
\operatorname{Writer}_{\ell_j}
\left(W^{j-1},A_j,K_j;C_j,P_j,\mathcal H\right),
\]

\[
Y_j
=
H(W^{j-1}+\Delta W_j)-H(W^{j-1}),
\qquad
E_j=A_j-Y_j.
\]

따라서 exact recurrence는

\[
\boxed{
R_{j+1}
=
\left(1-\frac1{n_j}\right)R_j+E_j
}
\]

이다. 여기에는 서로 다른 공간의 세 객체가 있다.

```text
allocated activation command A
  -> parameter-space action Delta W
  -> realized full-model activation response Y
```

## 2.2 Baseline에 대한 정확한 표현

Official MEMIT/AlphaEdit은 앞 layer write 뒤의 실제 changed state에서 downstream residual과
key를 다시 측정한다. 그러므로 baseline이 cross-layer effect를 전혀 보지 않는다는 표현은
틀리다.

정확한 표현은 다음이다.

> **The official writer is state-reactive but schedule-open-loop.**

즉 upstream realization error는 다음 residual에 들어가지만, 그 error를 이용해 현재 step의
크기를 줄이거나 건너뛰고, workload rule을 다시 정의하거나, 이미 방문한 layer를 새 joint
state에서 재방문하지 않는다. One-pass \(R/n_j\) schedule은 유지된다.

## 2.3 Phase A가 실제로 보여준 것

[Realization Debt Phase A](../../experiment-reports/servers/server1/realization-debt-phase-a-lifelong-v6-2026-09-03-v1/realization-debt-phase-a-lifelong-v6-detailed-factual-ko.md)는
40,000 edit requests와 200,000 request-layer observations에서 다음을 기록했다.

- allocated response와 realized response 사이의 gap은 layer·model·method에 따라 크게 다르다.
- 일부 arm의 mean/CVaR은 sparse L4 extremes와 큰 orthogonal component에 지배된다.
- \(\rho\), \(\tau\), `debt_parallel_native`, `debt_orthogonal`은 단일 scalar under-write보다
  더 복잡한 action--realization mismatch를 보인다.
- immediate semantic endpoint와 direct-\(z\) mismatch는 같은 값이 아니다.

다음 관찰 보고서도 context evidence로만 사용한다.

- [Independent B10 observer](../../experiment-reports/servers/server4/official-layer-realization-debt-b10x10-2026-09-01-v1/official-layer-realization-debt-b10x10-factual-ko.md)
- [Sequential B10x10 observer](../../experiment-reports/servers/server4/official-layer-realization-debt-sequential-b10x10-2026-09-01-v1/official-layer-realization-debt-sequential-b10x10-factual-ko.md)
- [Lifelong CounterFact v6](../../experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v6/official-layer-debt-lifelong-counterfact-metrics-v6-factual-ko.md)

이 evidence는 **realization mismatch가 존재한다**는 motivation이다. 다음은 아직 보이지 않는다.

- realization debt가 forgetting 또는 locality damage의 원인이라는 인과 관계
- barrier feedback이 debt tail을 줄인다는 사실
- ODE integration이 static controller보다 낫다는 사실
- 특정 layer가 universal hotspot이라는 결론

특히 semantic success와 큰 terminal direct-\(z\) mismatch가 공존한 hotspot은 downstream
layer가 debt를 상환했다는 사례가 아니다. 가능한 정확한 결론은 다음뿐이다.

> Large persistent direct-\(z\) mismatch can coexist with semantic rewrite success.

## 2.4 두 개의 독립된 질문

Phase A에서 바로 따라오는 질문은 둘이다.

1. **Realization fidelity:** allocated activation command가 parameter action을 거쳐 실제로 어떤
   response를 만드는가?
2. **Allocation validity:** 각 layer가 서로 다른 key, covariance/projector geometry와 downstream
   response를 갖는데 residual을 remaining-layer count로 균등 분할할 이유가 있는가?

두 번째 질문은 \(R/n\)이 전역적으로 나쁘다는 결론이 아니다. 단지 그 symmetry 가정을
실험 대상으로 올린다.

---

# 3. Research questions와 scope

본 proposal은 다음을 순서대로 검증한다.

1. Current official-form writer direction의 local terminal response를 안정적으로 측정할 수 있는가?
2. Prescribed \(R/n\) quota를 제거하고 response-guided ordered flow로 workload를 만들 수 있는가?
3. Realized upstream transition 뒤 downstream key와 writer를 재구성하는 것이 frozen/Jacobi
   field와 실제로 다른가?
4. Multi-sweep revisit가 one-pass schedule보다 mechanism 또는 endpoint를 개선하는가?
5. Semantic first-hit이 fixed \(z^\star\) overtracking을 줄이면서 efficacy를 유지하는가?
6. B100과 lifelong에서 controller가 보지 않은 RS/PS/NS와 retention이 개선되는가?

본 proposal이 해결하지 않는 문제는 다음과 같다.

- 잘못되었거나 underfit된 \(z^\star\)의 재학습
- globally optimal layer allocation
- canonical locality 또는 retention의 hard guarantee
- arbitrary finite-step barrier invariance
- architecture-independent layer importance

---

# 4. Information firewall

## 4.1 Controller가 사용할 수 있는 baseline-native artifact

- Official editor가 entry state에서 한 번 계산한 \(z^\star\)
- Current edit requests, `target_new`, stock compute-\(z\) contexts와 tokenization
- Current virtual model의 terminal activations와 current edit keys
- Official MEMIT/AlphaEdit closed-form solve와 regularization
- Existing Wikipedia covariance/statistics
- AlphaEdit projector와 previously committed history/cache
- Editable layer inventory와 official architectural order

## 4.2 Controller에 들어갈 수 없는 정보

- CAKE-style causal reference prompts, AIE 또는 corruption statistics
- New locality, calibration, sentinel 또는 general-capability prompt bank
- CounterFact rephrase/neighborhood prompts와 PS/NS outcome
- Historical request replay bank와 prior-edit success
- Learned layer selector, reward model, probe 또는 manual importance prior
- Future retention, final lifelong score 또는 held-out efficacy를 이용한 \(h\)/horizon tuning

따라서 이 방법을 `reference-free`라고 부르지 않는다.

> **No additional external reference bank beyond baseline-native artifacts.**

이것이 허용되는 가장 정확한 표현이다.

---

# 5. Frozen anchor와 exact virtual state

Outer edit 또는 edit batch entry를 \(W_0\)라 한다. Baseline direct target은 정확히 한 번
계산한다.

\[
Z^\star
=
\operatorname{ComputeZ}_{\rm baseline}(W_0,\mathcal E).
\]

Inner trajectory에서 다음은 immutable이다.

- \(Z^\star\)와 direct-\(z\) tensor hash
- request, context와 target token order
- baseline hyperparameters
- covariance, AlphaEdit projector와 entry committed history/cache
- editable module inventory

State는 persistent model이 아니라 exact virtual low-rank overlay \(W(t)\)다. Current-state
forward, key extraction, response JVP와 writer solve는 누적 overlay를 모두 포함해야 한다.
같은 weight에 여러 sweep factor가 쌓일 수 있으므로 factor를 deterministic하게
group/concatenate할 수 있어야 하며, persistent dense weights는 terminal transaction 전까지
byte-identical해야 한다.

Request \(i\)의 shared terminal activation을 \(H_i(W)\)라 하고 residual을

\[
R_i(W)=Z_i^\star-H_i(W)
\]

로 둔다. 모든 edit layer는 동일한 terminal readout 의미를 사용한다.

---

# 6. Official-form full-residual actuator

Current virtual state에서 layer \(\ell\)의 key를 \(K_\ell(W)\)라 한다. Main actuator는

\[
\boxed{
B_\ell(W)
=
\operatorname{OfficialFormWriter}_\ell
\left(
W,
R(W),
K_\ell(W);
C_\ell,P_\ell,\mathcal H_{\rm committed}
\right)
}
\]

이다. 핵심은 다음과 같다.

- RHS residual divisor는 1이다.
- Current full residual과 current key를 사용한다.
- Official covariance, regularizer, projector와 committed history semantics를 유지한다.
- Inner trajectory에서 history/cache를 append하지 않는다.
- State version이 바뀌면 \(R,K,B\)를 모두 폐기하고 다시 만든다.

이 방향은 official equation family를 사용하지만 official baseline update는 아니다. Official
parity는 별도 \(R/n\) mode에서 검증한다.

Frozen state에서 writer solve가 RHS에 선형이라면

\[
B_\ell(R/n)=\frac1n B_\ell(R)
\]

이어야 한다. 이 identity는 가정하지 않고 native-fidelity gate로 검증한다.

---

# 7. Ordered state-dependent flow

각 \(F_\ell(W)\)는 전체 parameter state 중 layer \(\ell\) block에만 작용하는 field다.

\[
F_\ell(W)
=
\mathcal I_\ell\!\left(u_\ell(W)B_\ell(W)\right),
\]

\[
\boxed{
\frac{dW}{dt}
=
\sum_{\ell\in\{4,5,6,7,8\}}F_\ell(W)
}
\]

로 combined flow를 정의한다. 한 numerical sweep은 Lie--Trotter 순서

\[
L4\rightarrow L5\rightarrow L6\rightarrow L7\rightarrow L8
\]

로 각 block field의 Euler subflow를 적용한다. 한 sweep 뒤 global pseudo-time은 \(h\)만큼
증가한다. Fixed horizon은

\[
\boxed{T_{\max}=Nh=1}
\]

다. 여기서 \(N\)은 layer update 수가 아니라 full ordered sweep 수이고, 최대 layer stage 수는
\(5N\)이다. Primary lock은 \(N=4,h=0.25\)다. \(N=2\)는 compute-reduced ablation,
\(N=8\)은 B1/B10 refinement audit으로만 사용한다. \(h\)를 반으로 줄이는 convergence run은
\(N\)을 두 배로 늘려 같은 \(T_{\max}=1\)을 유지하며 partial final sweep을 만들지 않는다.

Upstream step 이후 downstream writer는 exact changed state에서 다시 만든다. Expansion의
첫 cross term은 개념적으로

\[
h^2 D F_j(W)[F_i(W)]
\]

를 포함한다. Frozen/Jacobi field는 이 항을 보지 못한다.

순서는 transformer architecture의 shallow-to-deep execution order이지 causal importance
order가 아니다.

---

# 8. Normalized realization potential

Entry residual scale을 request별로

\[
s_i=\|R_i(W_0)\|_2
\]

로 고정한다. \(s_i=0\)이면 다음처럼 처리한다.

- 해당 request의 entry semantic predicate도 true: batch solve에는 남기되 normalized potential
  active set에서만 제외
- 해당 request의 semantic predicate가 false: `ANCHOR_DEGENERATE`; batch-level typed scientific
  boundary

전체 batch predicate가 true일 때만 batch 전체를 `ENTRY_ALREADY_HIT` no-op terminal로 처리한다.
이때 virtual factor, materialization, weight commit과 AlphaEdit history append는 모두 0회다.
Active set \(\mathcal A=\{i:s_i>0\}\)는 trajectory 동안 고정한다. 임의 epsilon은 넣지 않으며,
nonzero이지만 매우 작은 \(s_i\)에 대해서는 entry-scale dynamic range와 normalized JVP
conditioning을 outcome-blind gate로 검사한다. Active requests에 대해

\[
\bar R_i(W)=\frac{R_i(W)}{s_i},
\]

\[
\boxed{
V_z(W)
=
\frac1{2|\mathcal A|}
\sum_{i\in\mathcal A}
\|\bar R_i(W)\|_2^2
}
\]

로 둔다. Entry sublevel slack은

\[
b_z(W)=V_z(W_0)-V_z(W)
\]

다. 이는 direct-\(z\) realization path만 본다. Semantic correctness, prior knowledge,
CounterFact NS 또는 general locality safety를 뜻하지 않는다.

---

# 9. Current writer response와 canonical velocity multiplier

Layer actuator가 shared terminal activation에 만드는 current directional response는

\[
\psi_{i,\ell}(W)
=
D H_i(W)
\left[
\mathcal I_\ell(B_\ell(W))
\right]
\]

이고

\[
\bar\psi_{i,\ell}=\psi_{i,\ell}/s_i
\]

다. Batch inner product와 squared response를

\[
g_\ell(W)
=
\frac1{|\mathcal A|}
\sum_i
\langle\bar R_i,\bar\psi_{i,\ell}\rangle,
\]

\[
r_\ell(W)
=
\frac1{|\mathcal A|}
\sum_i
\|\bar\psi_{i,\ell}\|_2^2
\]

로 둔다. Linearized potential

\[
\widehat V_z(a)
=
\frac1{2|\mathcal A|}
\sum_i
\|\bar R_i-a\bar\psi_{i,\ell}\|_2^2
\]

의 nonnegative line minimizer는

\[
a_\ell^\star
=
\begin{cases}
0,&r_\ell=0,\\[1mm]
\dfrac{[g_\ell]_+}{r_\ell},&r_\ell>0.
\end{cases}
\]

본 proposal은 이를 Euler step 자체로 사용하지 않는다. \(h\)-independent response velocity를

\[
\boxed{
u_\ell(W)
=
\begin{cases}
0,&r_\ell=0\text{ or }g_\ell\le0,\\[1mm]
\min\{1,a_\ell^\star\},&g_\ell>0
\end{cases}
}
\]

로 정의한다.

여기서 상한 1은 `no-amplification prior`다. 즉 official-form full-residual actuator를 local
response rule만으로 1배보다 증폭하지 않는다. Arbitrary performance threshold는 아니지만
분명한 velocity prior이며 숨기지 않는다.

따라서 Euler coefficient는 \(\alpha_{n,\ell}=h u_\ell(W_{n,\ell})\)이다. Isolated layer
subflow에서는

\[
\frac{dV_z}{dt}
=
-u_\ell g_\ell
\le0.
\]

이고 combined simultaneous-field notation에서는

\[
\frac{dV_z}{dt}
=
-\sum_\ell u_\ell g_\ell
\le0
\]

다. 실제 integrator는 simultaneous sum이 아니라 §7의 ordered splitting을 사용한다.

해석은 다음과 같다.

- opposite/non-descending response: \(u=0\)
- weak aligned response: \(a^\star\)가 커지므로 기본적으로 \(u=1\); 작아진다고 해석하지 않음
- excessive aligned gain: \(a^\star<1\)이면 overshoot를 피하도록 축소
- large orthogonal response: \(r_\ell\)이 커져 \(a^\star\)와 \(u\)를 축소할 수 있음

이것은 arbitrary debt threshold가 아니라 current response model의 inverse-gain cap이다.
각 run은 \(u=0\), \(0<u<1\), \(u=1\)인 layer-visit 비율을 별도로 보고한다.

---

# 10. Residual excursion과 parameter-action telemetry

초안에서 같은 \(q\) 기호로 부르던 residual excursion과 parameter action은 서로 다른 객체다.
둘 다 mechanism을 관측하지만 main controller, Euler coefficient와 terminal predicate에는 들어가지
않는다.

## 10.1 Residual excursion

Potential active set \(i\in\mathcal A\)의 request별 normalized residual excursion은

\[
q_i^{\rm res}(W)
=
\frac{\|R_i(W)\|_2}{\|R_i(W_0)\|_2}
\]

다. 별도의 \(q_i^{\rm res}\le q_{\max}\) hard cap을 두지 않는다. 각 run은 최소한

\[
\max_{i,n,\ell}q_i^{\rm res},
\qquad
q_{i,\rm terminal}^{\rm res},
\qquad
\Pr[q_i^{\rm res}>1]
\]

과 distribution tail을 보고한다. \(s_i=0\)으로 \(\mathcal A\)에서 제외된 request에는 이 비율을
정의하지 않고 `ZERO_ANCHOR_EXCLUDED_FROM_QRES` count와 별도 semantic trajectory만 기록한다.
이 telemetry는 \(V_z\)의 request-level 거동을 설명하지만 semantic truth나 finite-step safety
certificate가 아니다.

## 10.2 Baseline-native parameter action

Method-specific regularized covariance geometry를 \(C_\ell^{\rm reg}\)라 할 때 current actuator의
directional action은

\[
A_{n,\ell}^{\rm dir}
=
\operatorname{tr}
\left(
B_{n,\ell}C_\ell^{\rm reg}B_{n,\ell}^{\top}
\right)
\]

로 기록한다. 이는 canonical locality/NS bound가 아니며 architecture, layer와 covariance
regularization에 따라 scale이 달라진다. 그러므로

\[
\boxed{
u=u(R,\psi),
\qquad
A^{\rm dir}\notin u,\alpha,\operatorname{Sem}
}
\]

를 main firewall로 고정한다. Per-step action과 resolution-stable path action은 각각

\[
\Delta W_{n,\ell}=h u_{n,\ell}B_{n,\ell},
\]

\[
A_{n,\ell}^{\rm step}
=
(h u_{n,\ell})^2 A_{n,\ell}^{\rm dir},
\]

\[
\boxed{
A_{\rm path}
=
\sum_{n,\ell}
h u_{n,\ell}^{2}A_{n,\ell}^{\rm dir}
}
\]

다. 단순 \(\sum A^{\rm step}\)는 \(h\to0\)에서 인위적으로 0으로 가므로 numerical-resolution
비교의 primary action statistic으로 사용하지 않는다. Terminal net action은

\[
A_{\rm net}
=
\sum_\ell
\operatorname{tr}\!\left(
\Delta W_{\ell}^{\rm net}
C_\ell^{\rm reg}
(\Delta W_{\ell}^{\rm net})^\top
\right),
\qquad
\Delta W_{\ell}^{\rm net}=\sum_n\Delta W_{n,\ell}
\]

로 별도 보고한다. 이는 canonical edited-weight orientation
\(\Delta W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}\)을 사용한 표기다. Adapter 내부 factor
orientation이 다르면 `match_shape` 이후의 \(\Delta W\)로 계산한다.
\(C_\ell^{\rm reg}\)의 exact official-object binding, symmetry/PSD numerical
receipt와 finite-value 검사는 telemetry fidelity gate다. 이 gate의 실패는 action telemetry를
invalid로 만들 뿐, 유효한 response controller를 사후 action constraint로 바꾸지 않는다.

따라서 final seal은 다음과 같다.

\[
\boxed{q\text{와 parameter action은 main constraint가 아니라 mechanism telemetry다.}}
\]

Action-aware controller는 threshold, feasibility와 attribution을 새로 도입하는 별도 future
ablation으로만 설계한다.

---

# 11. Euler advance와 exact realized feedback

한 layer visit에서

\[
\boxed{
W^+
=
W+h u_\ell(W)\mathcal I_\ell(B_\ell(W))
}
\]

로 정확히 한 번 advance한다. Candidate evaluation, accept/reject, coefficient shrink, retry와
backtracking은 없다.

Step 뒤 exact forward로

\[
Y_{i,\ell}^{\rm act}
=
H_i(W^+)-H_i(W)
\]

를 관측한다. Linear prediction, command, realization error와 model error는

\[
Y_{i,\ell}^{\rm pred}
=
h u_\ell\psi_{i,\ell},
\]

\[
A_{i,\ell}=h u_\ell R_i,
\]

\[
E_{i,\ell}=A_{i,\ell}-Y_{i,\ell}^{\rm act},
\qquad
M_{i,\ell}=Y_{i,\ell}^{\rm act}-Y_{i,\ell}^{\rm pred}
\]

로 기록한다. Exact next residual은

\[
\boxed{
R_i(W^+)=R_i(W)-Y_{i,\ell}^{\rm act}
}
\]

다. Actual error를 별도로 상환한다고 주장하지 않는다. Actual changed state가 다음 residual,
key, response와 writer direction을 바꾸는 것이 feedback이다.

> **The realized error is incorporated into the next state and therefore changes every downstream
> residual, key, response model, and writer direction.**

---

# 12. Local barrier guarantee와 finite-step non-guarantee

\(0<h\le1\), \(0\le u\le a^\star\)이므로 linearized model에서는

\[
\widehat V_z(hu)\le V_z(W)
\]

이다. 그러나 nonlinear model은

\[
V_z(W+huB)
=
\widehat V_z(hu)
+O\!\left((hu)^2\|B\|^2\right)
\]

인 local expansion만 갖는다. Remainder constant는 current curvature, ill-conditioned solve와
큰 \(\|B\|\)에서 커질 수 있으므로 uniform \(O(h^2)\) bound로 해석하지 않는다. Actual
finite-step monotonicity는 보장되지 않는다. 반드시 다음을 기록한다.

\[
\delta_{s,\ell}^{\rm disc}
=
[V_z(W^+)-V_z(W)]_+,
\]

\[
\delta_{s,\ell}^{\rm entry}
=
[V_z(W^+)-V_z(W_0)]_+,
\]

- stepwise defect count, sum, p90, CVaR90와 max
- entry safe-set crossing count, cumulative and maximum violation
- actual reduction / predicted reduction ratio; predicted reduction이 locked near-zero bound 이하이면
  ratio는 `NOT_APPLICABLE_ZERO_PREDICTION`으로 기록하고 나누지 않음
- per-request potential worsening rate와 tail
- \(\|M\|\) 및 normalized model error
- \(h,h/2,h/4\)에서 defect의 convergence order

허용되는 표현은 다음이다.

> The response barrier shapes the local velocity; the exact realized transition determines the next
> field.

금지되는 표현은 다음이다.

> Every finite Euler iterate is barrier-certified or safe.

---

# 13. Semantic first-hit

\(z^\star\)는 direction과 realization potential의 frozen anchor다. Exact
\(H(W_T)=Z^\star\)는 mandatory terminal이 아니다.

초기 구현의 semantic event는 baseline compute-\(z\)가 만든 **target-bearing contexts만** 사용해
모든 batch request와 모든 `target_new` token이 teacher-forced strict top-1을 만족하는
batch-level predicate다. KL-only context는 제외한다.

각 method adapter는 outcome을 보기 전에 ordered semantic inventory를 봉인한다. 한 event row는
`(request_ordinal, context_ordinal, target_token_ordinal)`로 식별하고, stock compute-\(z\) loss
builder가 실제 target loss에 사용한 모든 context와 모든 target token position을 정확히 한 번
포함한다. Context/input token IDs, target IDs, attention mask, position IDs와 row order는 hash로
묶는다. KL-only context count는 별도로 기록하되 semantic inventory에는 넣지 않는다. Llama와
Qwen, MEMIT과 AlphaEdit은 각자의 stock context builder를 사용하며 서로의 context set을
대체하지 않는다.

\[
\operatorname{Sem}(W)=1
\iff
\log p_W(y_{i,c,r}^{\rm new}\mid c_{i,c},y_{i,c,<r}^{\rm new})
>
\max_{v\ne y_{i,c,r}^{\rm new}}
\log p_W(v\mid c_{i,c},y_{i,c,<r}^{\rm new})
\quad\forall i,c,r.
\]

정책은

\[
\boxed{
\text{batch-level first semantic hit or fixed horizon }T
}
\]

다.

실제 관측 가능한 event는 continuous-time crossing이 아니라 각 exact post-layer state에서의
`ordered split-state first hit`이다. Post-step forward 직후 predicate를 평가하고 최초
`sweep/layer/state_version`을 latch한다. 마지막 L8 또는 최종 sweep의 hit도 누락하지 않는다.

- Entry에서 batch 전체가 predicate를 만족하면 `ENTRY_ALREADY_HIT` no-op으로 기록한다.
- Batch 일부만 entry-hit이면 해당 count를 공개하되 main v1에서는 그 request만 제거하지 않는다.
  Mixed batch는 전체 batch predicate가 false인 동안 동일 batch solve를 유지한다.
- 일부 request만 성공했다고 active set에서 제거하지 않는다. Per-request removal은 batch key와
  solve를 바꾸는 별도 method다.
- Entry와 매 nonzero layer update 직후에만 검사한다. Logit tie는 strict predicate false이며,
  numerical/method failure로 분류하지 않는다.
- Predicate는 sealed inventory의 모든 \((i,c,r)\)에 대한 conjunction이며 missing, duplicate 또는
  reordered event는 technical invalid execution이다.
- First hit이 없으면 fixed-horizon endpoint를 반환하고 strict-event miss로 계산한다. 이는
  canonical CounterFact RS/PS failure와 동치가 아니며 endpoint evaluator가 별도로 판정한다.
- 성공 case만 골라 commit하거나 실패 endpoint를 숨기지 않는다.
- 이 predicate는 canonical CounterFact RS가 아니라 controller-side target-new strict event다.
  Evaluation RS/PS/NS와 명확히 분리한다.
- `target_true`, rephrase, neighborhood, PS/NS와 별도 margin \(\gamma\)는 predicate에 들어가지
  않는다. \(V_z(W)\le V_z(W_0)\)도 terminal gate로 추가하지 않는다.

Fixed-\(T\), first-hit OFF arm을 먼저 수치 convergence authority로 사용하고, first-hit ON/OFF를
별도 paired ablation으로 비교한다. 그래야 action/compute 감소를 stopping policy에 귀속할 수
있다.

잘못되거나 underfit된 \(z^\star\)를 first-hit이 수정한다고 주장하지 않는다.

---

# 14. Multi-sweep와 emergent workload

Semantic hit이 없고 horizon이 남으면 L4부터 다시 방문한다. 각 layer workload는 사전에
simplex weight로 정하지 않는다.

\[
\Omega_\ell
=
\sum_n h u_{n,\ell},
\]

\[
\Lambda_\ell^{\rm progress}
=
\sum_s
\left[V_z(W_{s,\ell})-V_z(W_{s,\ell}^{+})\right],
\]

\[
A_{\ell}^{\rm path}
=
\sum_n h u_{n,\ell}^2 A_{n,\ell}^{\rm dir},
\qquad
A_{\ell}^{\rm net}
=
\operatorname{tr}\!\left(
\Delta W_{\ell}^{\rm net}
C_\ell^{\rm reg}
(\Delta W_{\ell}^{\rm net})^\top
\right).
\]

\(T_{\max}=1\)과 \(0\le u\le1\) 때문에 각 layer의 nominal exposure는
\(\Omega_\ell\le1\)이지만, 이는 FLOP 또는 parameter-action matching을 뜻하지 않는다.
이 값들은 effective layer workload를 사후적으로 보여준다. \(h\), fixed order와 response gate도
implicit prior이므로 globally optimal allocation을 발견했다고 말하지 않는다.

가능한 정확한 claim은 다음이다.

> We remove the explicit remaining-layer residual quota and let effective layer workload emerge from
> repeated current-state actuator responses under a fixed architectural visitation order.

---

# 15. Transaction과 failure semantics

Inner integration 동안 persistent weights와 committed history/cache는 변경하지 않는다. Entry에서
편집 대상 weight bytes/pointers와 method state를 exact recovery snapshot으로 봉인한다. Terminal
transaction은 다음 순서를 정확히 한 번 수행한다.

1. accumulated factors를 persistent model과 분리된 FULL_FP32 shadow buffers에 materialize
2. virtual/shadow weight, activation과 logit identity 확인
3. 모든 edited weights와 method state를 하나의 guarded system transaction으로 dense commit
4. method-specific finalization:
   - AlphaEdit: current edit key/history를 final state에서 정확히 한 번 append
   - MEMIT: request-history append 없음; static `COV_CACHE` identity와 mutation 0을 확인

Commit 또는 finalization 중 interruption/exception이 발생하면 recovery snapshot에서 weight bytes,
pointers와 method state를 전부 복구하고 exact identity를 확인한 뒤 technical invalid execution으로
종료한다. 이는 partial persistent mutation을 막는 **system-level atomic recovery**이며, scientific
candidate를 거절하거나 더 작은 coefficient로 재시도하는 algorithmic rollback이 아니다.

Failure semantics는 다음처럼 고정한다.

| 분류 | 상태 | 처리 |
|---|---|---|
| no-op | entry batch predicate true | W0 endpoint의 RS/PS/NS 등 evaluation은 수행하고 intent-to-edit/no-op denominator에 포함; factor/materialization/commit/history append 0 |
| technical invalid execution | stale-state access, wrapper/counter mismatch, corrupted factor, runtime/hardware corruption, shadow/materialized identity 또는 atomic-recovery failure | no-commit 또는 exact system recovery; scientific endpoint 분모 제외; technical-attempt 분모에는 포함 |
| locked numerical method boundary | Official/parity oracle, reference solve/JVP, clean rerun과 필요시 FP64 observer로 implementation fault를 배제한 뒤에도 sealed input에서 재현되는 solve singularity/range failure, nonfinite writer/JVP/forward 또는 anchor degeneracy | pre-commit abort; terminal-valid endpoint는 없지만 attempted scientific denominator와 method kill-rate에 포함; fault 배제 전에는 technical invalid |
| finite zero-action endpoint | 첫 sweep에서 모든 \(u=0\), accumulated action 0, strict event false | `FLOW_STALLED_ZERO_ACTION`; byte-identical current endpoint를 output으로 확정/evaluate하되 materialization/commit/history append 0; strict-event miss와 intent-to-edit denominator에 포함 |
| finite post-action endpoint | prior nonzero action 뒤 한 full sweep에서 모든 \(u=0\) | `FLOW_STALLED`; 남은 시간은 zero-flow로 간주하고 current endpoint commit/evaluate; method-specific finalization 1회; fallback 없음 |
| finite scientific endpoint | fixed horizon에서 strict event false | finite endpoint commit/evaluate, `HORIZON_SEMANTIC_MISS`로 포함; RS/PS/NS는 독립 평가하고 outcome 누락 없음 |
| observed defect | actual barrier defect 또는 entry-set crossing | step을 유지하고 telemetry 기록; reject/shrink/retry 없음 |
| system interruption | scheduler/hardware/user interruption before commit | atomic abort; technical exclusion; algorithmic rollback으로 세지 않음 |

Staged factor 폐기는 persistent model이 아직 바뀌지 않은 transaction abort이며 algorithmic
candidate rollback이 아니다. Total accumulated action이 0인 첫-sweep stall은 byte-identical
no-op endpoint로 기록하되 materialization, dense commit과 AlphaEdit history append를 모두 0회로
둔다. 이전 sweep에 nonzero action이 있었던 뒤의 stall은 current finite endpoint를 commit하고
method-specific finalization을 1회 수행한다. Main에는 backtracking oracle, best-waypoint selection, fallback과
candidate retry가 없다. 어떤 boundary도 successful endpoint 분모에서 조용히 제거하지 않고
technical-attempt, attempted-scientific, terminal-valid와 kill-rate를 함께 공개한다.

---

# 16. Canonical pseudocode

```text
Input:
    entry model W0
    edit batch E
    ordered layers [L4, L5, L6, L7, L8]
    baseline-native C/P and committed history
    fixed T_max = 1, primary sweeps N = 4, Euler step h = 1/N

1. z_star = baseline_compute_z(W0, E) exactly once
2. freeze request/token order, target-bearing compute-z contexts,
   C/P/committed-history and module inventory
3. compute entry residual scales and entry semantic state
4. initialize exact virtual overlay Wv = W0
5. bind method-specific C_reg for telemetry only
   seal exact weight/method-state recovery snapshot
   total_nonzero_action = 0

6. if target-new strict predicate is true at entry:
       evaluate W0 endpoint with canonical RS/PS/NS and include all intent-to-edit requests
       return ENTRY_ALREADY_HIT with materialization/commit/history append all zero

7. for sweep = 1..N:
       sweep_nonzero_flow = false
       for layer in [L4, L5, L6, L7, L8]:
           read shared terminal activation on current Wv
           R = z_star - H(Wv)
           K = current_key(layer, Wv)
           B = official_form_writer(layer, Wv, full_residual=R,
                                    committed_history_only=True)
           psi = JVP_H(Wv; inject_layer(B))

           compute g = <R_bar, psi_bar> and r = ||psi_bar||^2
           if r == 0 or g <= 0:
               u = 0
           else:
               u = min(1, g / r)

           if u == 0:
               record skip and continue

           advance once without a trial:
               Wv = Wv + h * u * B
               sweep_nonzero_flow = true
               total_nonzero_action += 1

           run exact post-step observation
           record actual Y, E, M, residual excursion, potential defects,
                  A_dir, A_step, A_path accumulator and semantic state
           increment state version
           invalidate R/K/B/psi/action objects

           evaluate semantic predicate on the exact post-step state
           if this is the first batch-level hit:
               latch first_hit_sweep/layer/state_version immediately
               terminate trajectory

       if sweep_nonzero_flow == false:
           mark FLOW_STALLED
           treat remaining horizon as zero-flow
           terminate trajectory at current Wv

8. if total_nonzero_action == 0:
       publish byte-identical FLOW_STALLED_ZERO_ACTION endpoint
       materialization/commit/history append = 0
       evaluate it as a strict-event miss and stop
9. materialize accumulated factors to detached FULL_FP32 shadow buffers exactly once
10. verify virtual/shadow weights, activations and logits once
11. atomically commit the finite endpoint once, including post-action FLOW_STALLED and horizon miss
    on commit/finalization error, restore exact recovery snapshot and fail technical
12. finalize method state for every committed endpoint:
        AlphaEdit -> append current edit history once
        MEMIT -> append N/A, verify static COV cache unchanged
13. if no strict hit, mark HORIZON_SEMANTIC_MISS or FLOW_STALLED
14. evaluate every intent-to-edit request, including strict-event misses;
    compute RS/PS/NS independently
```

---

# 17. Prior-work boundary

## 17.1 ODESteer와 ODE-M

[ODESteer](https://arxiv.org/html/2602.17560)는 current activation에서 barrier-gradient field를
반복 평가한다. Continual model merging 방법인
[ODE-M](https://arxiv.org/html/2605.19409)은 incoming endpoint velocity의 risk-aligned
component를 current loss gradient로 rectification한다.

본 proposal이 가져오는 것은 다음 구조적 영감뿐이다.

- current state에서 field/response를 다시 평가한다.
- local risk 또는 potential response가 instantaneous velocity를 조절한다.

두 연구의 theorem, calibration data, invariance 또는 bounded-barrier guarantee를 본 방법으로
이전하지 않는다.

## 17.2 CAKE

[CAKE](https://aclanthology.org/2026.acl-long.918/)는 causal-tracing-derived layer scores/weights로
residual allocation을 조절한다. 본 방법의 차별점은 external bank 부재 하나가 아니라 current
concrete writer JVP, realized-state rebuild와 revisit에 있다. 본 방법은 causal score,
softmax temperature 또는 simplex residual share를 사용하지 않는다.

Novelty 후보는 `adaptive allocation 최초`가 아니라 다음 결합이다.

- concrete official-form writer의 current terminal-response JVP
- prescribed \(R/n\) quota 제거
- actual upstream state 뒤 downstream field rebuild
- ordered multi-sweep revisit
- external causal bank 없는 semantic first-hit transaction

CAKE도 downstream residual을 다시 계산할 수 있으므로 `최초로 layer마다 residual을 재계산`은
금지 claim이다.

## 17.3 MetaKE

[MetaKE](https://arxiv.org/abs/2603.12677)는 downstream realization feedback을 upstream target
optimization으로 전달한다. 본 방법은 \(z^\star\)를 고정하고 meta-gradient를 전달하지 않는다.
따라서 target adaptation을 해결했다고 주장할 수 없다.

---

# 18. Implementation map과 missing primitives

현재 재사용 후보는 다음과 같다. 모두 그대로 main loop가 되는 것이 아니라 oracle, factor
primitive 또는 transaction pattern으로만 재사용한다.

- [`p1_backend.py`](../run_scripts/ode_bf/p1_backend.py): cumulative-factor state에서 current
  residual/key를 다시 읽는 AlphaEdit-side reference. Standard path의 layer-local activation을
  shared terminal \(H\)로 자동 해석하지 않는다.
- [`functional.py`](../run_scripts/ode_bf/functional.py): factor ordering과 virtual transaction
  pattern. Existing BF16/dense-effective-weight path를 FULL_FP32 repeated-JVP main으로 그대로
  사용하지 않는다.
- [`euler_writer.py`](../run_scripts/barrier_guided_ode/fixed_basis_ode/euler_writer.py): frozen-field comparator와 Euler accounting
- Official MEMIT/AlphaEdit wrappers와 latest layer-debt observer: exact method-specific activation
  return, terminal readout, key/solve/padding/cache lifecycle의 oracle

기존 fixed-basis Euler를 state-dependent main으로 부르지 않는다. 신규 primitive는 다음으로
제한한다.

1. Frozen \(z^\star\), shared terminal \(H\), current key, full-residual factor build와 finalization을
   통일하는 method-generic MEMIT/AlphaEdit adapter
2. 같은 weight의 repeated factors를 dense reassembly 없이 처리하는 FULL_FP32 grouped virtual overlay
3. Exact current overlay에서 vector \(D H(W)[B_\ell(W)]\)를 계산하는 terminal JVP
4. Residual/key columns와 request identity의 exact repeated-column mapping
5. Every state mutation 뒤 residual/key/writer/JVP/action-telemetry object를 무효화하는 version ledger
6. Batch-level semantic predicate, fixed-\(T\) clock과 first-hit state machine
7. Method-specific terminal commit: AlphaEdit final history append, MEMIT static COV preservation
8. No-reject/no-retry failure receipt와 barrier-defect telemetry
9. Fixed-\(T\) convergence runner와 deterministic arm scheduler

구현은 다음 순서로만 확장한다.

```text
method adapter + exact Official one-step oracle
  -> FULL_FP32 grouped virtual overlay/materialization parity
  -> vector terminal JVP + multi-epsilon FD
  -> one-layer/one-request state-versioned step
  -> five-layer one-sweep
  -> fixed-T multi-sweep
  -> B1 scientific arms
```

앞 block이 닫히기 전에 뒤 block을 GPU scientific failure로 분류하지 않는다.

가장 큰 B100 compute/memory risk는 layer×sweep마다 current key extraction, dense/closed-form
solve, directional JVP와 exact post-step forward가 반복되고, virtual low-rank rank가 sweep마다
누적되는 것이다. Exact semantic hit가 activation forward와 결합되지 않으면 target-token logit
forward가 추가된다. Reverse-mode로 vector \(\psi\)의 모든 output을 따로 미분하는 것은
허용하지 않고, forward-mode tangent compatibility를 gate한다. B1에서 다음을 반드시 산출한다.

- forwards/JVPs/FD probes/solves per node and per request
- accumulated factor rank와 peak overlay memory
- materialization cost와 peak allocated GPU memory
- wall time 대비 Official 및 frozen-field comparator

---

# 19. Pre-B1 mandatory gates

## G0-A — Official fidelity와 scaling

- Stock \(R/n\) mode가 Official endpoint를 selected weight에서 bitwise 또는 locked ULP로 재현
- Fixed state에서 \(B(R/n)=B(R)/n\) RHS scaling identity
- All layers가 동일 terminal readout과 exact request order를 사용
- Direct-\(z\) compute 1회, recompute 0회

## G0-B — JVP fidelity

- \(\psi=DH[B]\)와 multi-epsilon central finite difference의 sign/norm/elementwise agreement
- Current virtual overlay를 포함한 JVP와 materialized temporary model JVP identity
- Llama/Qwen left-padding, attention mask와 position semantics parity
- zero response와 nonfinite response의 typed boundary
- Entry residual scale dynamic range와 near-zero normalized-JVP conditioning

## G0-C — Sequential state fidelity

- L4 step 뒤 L5 key와 writer hash가 changed state를 반영
- State-version change 뒤 stale \(R,K,B,\psi,q\) consumption 0회
- Second-sweep L4가 first full sweep의 accumulated state를 읽음
- Ordered path와 frozen/Jacobi comparator가 독립 namespace를 사용

## G0-D — Virtual/physical transaction

- Virtual and materialized selected weights byte/ULP lock
- Terminal activations와 logits identity
- Factor application order identity
- Inner persistent history/cache mutation, retry, rejected step와 rollback 각 0회
- Non-no-op terminal materialization/commit 각 정확히 1회; entry-hit no-op은 0회
- Non-no-op AlphaEdit final-state history append 정확히 1회; entry-hit no-op은 0회
- MEMIT request-history append `N/A`, static `COV_CACHE` identity와 mutation 0

## G0-E — Action telemetry fidelity

- Family-specific \(C_\ell^{\rm reg}\) official-object binding, symmetry/PSD와 solve-conditioning receipt
- End-to-end request-order invariance와 finite-value audit
- \(A^{\rm step}=(hu)^2A^{\rm dir}\), \(A_{\rm path}=\sum h u^2A^{\rm dir}\) identity
- Terminal accumulated factors에서 \(A_{\rm net}\) 독립 재계산 identity
- Same-layer coordinate/rescaling sensitivity를 공개하고 cross-model raw action magnitude를 직접
  비교하지 않음
- Gate 실패는 해당 telemetry를 invalid로 표시하며 \(u\), \(\alpha\), endpoint를 바꾸지 않음

## G0-F — Fixed-time numerical contract

- First-hit OFF, same \(T_{\max}=1\)에서 \(N=2,4,8\) 및 \(h=1/2,1/4,1/8\)
- 모든 grid에서 \(Nh=1\); partial final sweep 0
- Weight, activation, semantic logit와 \(V_z\) endpoint convergence
- Stepwise defect가 \(h\) 감소에 따라 줄어드는지 확인
- Same sweep count로 서로 다른 horizon을 비교하지 않음

Field에는 \(\min\), \(g=0\) switch와 solve-conditioning boundary가 있어 classical smooth
first-order rate를 선험적으로 강제하지 않는다. Outcome-blind distance와 ratio를 잠근 empirical
Cauchy gate로 endpoint convergence를 판정하고, observed order는 사실값으로 보고한다.

G0-A/B/C/D/F 중 하나라도 실패하면 B1 endpoint 성능으로 method를 평가하지 않는다. G0-E만
실패하면 controller endpoint 자체는 유효할 수 있으므로 B1 response/semantic 평가는 진행할 수
있지만 action telemetry를 `TELEMETRY_INVALID`로 표시하고 action-based claim과 promotion package는
보류한다. Telemetry를 고치기 위해 \(u\), \(h\), terminal 또는 endpoint를 바꾸지 않는다.

---

# 20. Falsification-first experiment ladder

## 20.1 Stage B1 — mechanism pilot

최소 scientific arms는 다음과 같다.

| Arm | 정확한 역할 |
|---|---|
| Official \(R/n\), one pass | Stock baseline |
| No-quota fixed-speed ordered, \(N=4\) | Full-residual actuator, \(u=1\), \(h=0.25\); canonical main과 동일한 strict first-hit ON으로 response-controller 효과 분리 |
| No-quota response convergence family, \(N=1,2,4,8\) | Current-response \(u\), \(T_{\max}=1\), first-hit OFF; \(N=1\) coarse one-sweep, \(N=4\) fixed-horizon authority, \(N=2/8\) compute/refinement audit |
| Full response-barrier, \(N=4\) | Canonical main: current-response \(u\), ordered exact refresh, strict semantic first-hit ON |
| Strict-z terminal diagnostic | Same \(N=4\) flow에서 semantic first-hit을 끄고 locked strict-z diagnostic terminal을 사용; main/promotion endpoint 아님 |

`Official R/n one-pass`와 stock-parity wrapper는 별도 scientific arm이 아니다. Parity wrapper는
G0 engineering gate다. 위 다섯 arm family가 first-pass seal이다. Canonical \(N=4\) ON과
convergence-family \(N=4\) OFF가 first-hit의 필수 paired comparison이다. \(N=1\)과 \(N=4\)의
차이는 discretization과 revisit를 함께 바꾸므로 `multi-sweep integrated effect`로만 해석한다.

Ordered revisit 자체를 claim하려면 같은 \(h=0.25\), 20 layer stages와 first-hit OFF에서
`[L4,L5,L6,L7,L8]×4`와 preregistered grouped/no-revisit ordering
`[L4×4,L5×4,L6×4,L7×4,L8×4]`을 비교한다. Ordered state rebuild를 claim하려면 같은 schedule의
frozen/Jacobi comparator도 promotion 전 필수다. Repeated-quota closed-loop는 quota attribution이
필요한 경우에만 secondary ablation으로 추가한다. 이 arm은 매 sweep에서 shallow-to-deep
\(n_\ell=m-j+1\)을 다시 적용하고 \(u=1\), 같은 \(T,h\)로 맞춘다. Official과 비교할 때는
fractional stepping, re-solve와 revisit도 함께 달라지므로 `동일 quota에서 feedback만 분리`했다고
주장하지 않는다. Frozen/static comparator와 compute-count matching rule은 결과 전에 고정한다.

B1 개발 sample과 confirmatory sample을 분리한다. \(h\) 선택은 fixed-\(T\) stability,
JVP/model error, defect order, efficacy와 compute로만 한다. PS/NS는 보지 않는다.

## 20.2 Stage B10 — ordered cross-effect

다음을 직접 측정한다.

\[
\delta K_{j\mid i}
=
\frac{\|K_j(W+\Delta W_i)-K_j(W)\|}{\|K_j(W)\|},
\]

\[
\delta B_{j\mid i}
=
\frac{\|B_j(W+\Delta W_i)-B_j(W)\|}{\|B_j(W)\|}.
\]

비교는 다음을 포함한다.

- ordered refreshed vs frozen/Jacobi
- one sweep vs fixed-\(T\) multi-sweep
- response barrier OFF vs ON
- semantic first-hit OFF vs ON

Ordered refreshed가 frozen/Jacobi와 구분되지 않으면 cross-layer adaptation을 contribution의
중심으로 삼지 않는다.

## 20.3 Stage B100 — frozen confirmatory

B10 결과를 보기 전에 다음을 고정한다.

- \(T_{\max}=1\), primary \(N=4,h=0.25\), refinement grid와 layer order
- response normalization과 telemetry-only action schema
- semantic predicate와 fixed-horizon policy
- failure semantics와 telemetry schema
- model-common controller; model-specific rescue 금지

각 node와 request에서 최소 다음을 기록한다.

- exact/predicted \(V_z\), \(g,r,a^\star,u\)와 \(u=0,(0,1),1\) counts
- \(R,B,\psi,A,Y,E,M\) norm과 cosine
- Phase-A-compatible \(\rho,\tau\), parallel/orthogonal debt
- \(q^{\rm res}\) maximum/terminal/tail, \(A^{\rm dir}\), \(A^{\rm step}\),
  \(A_{\rm path}\)와 \(A_{\rm net}\)
- stepwise/entry barrier defect와 per-request worsening
- layer skip/revisit, sweep and first-hit ordinal
- solve condition, factor rank, path/terminal action
- forward/JVP/solve count, runtime와 peak memory
- target NLL/RS tail과 worst decile

Outcome table은 mean만 쓰지 않고 median, p90, p99, CVaR90와 max를 함께 낸다.

## 20.4 Lifelong

B100 mechanism gate 이후에만 existing v6 protocol의 exact sample/order/evaluator를 사용한다.
Primary evaluation은 controller가 보지 않은 다음 지표다.

- current-B100 RS/PS/NS
- cumulative seen-prefix RS/PS/NS
- fixed age cohorts와 old/recent gap
- final \(W_{10000}\) full-10k metrics
- target-new/target-true NLL advantage distributions

Rephrase, neighborhood, prior-edit prompts와 future retention은 controller input으로 사용하지 않는다.

---

# 21. Decision table

| 질문 | 필요한 비교 | 해석 가능한 결론 |
|---|---|---|
| Repeated partial revisit가 one-pass와 다른가? | Repeated-quota closed-loop vs Official | Repeated per-visit quota를 포함한 integrated policy의 차이; feedback 단독 귀속 금지 |
| \(R/n\) quota가 필요한가? | No-quota vs Repeated-quota, same \(T,h\) | Repeated per-visit prescribed share 제거의 operational effect |
| Response barrier가 필요한가? | Full response \(N=4\) vs no-quota \(u=1,N=4\) | Current JVP coefficient의 추가 가치 |
| Multi-sweep integrated policy가 필요한가? | First-hit OFF response \(N=1,2,4,8\), same \(T=1\) | Discretization과 revisit를 합친 resolution effect; revisit 단독 귀속 금지 |
| Revisit 자체가 필요한가? | \(h=.25\), 20 stages의 ordered-four-sweep vs grouped/no-revisit | Matched stage-count에서 revisit/order effect |
| Ordered rebuild가 필요한가? | Same schedule ordered-current vs frozen/Jacobi | Cross-layer state adaptation의 가치 |
| First-hit이 필요한가? | Response \(N=4\), first-hit ON vs OFF | Overtracking/action/compute effect |
| ODE가 필요한가? | Fixed-\(T\) dynamic flow vs preregistered static/frozen comparator | State feedback의 독립 가치 |

이 비교 없이 단일 arm의 성능 향상을 barrier, ODE 또는 allocation에 귀속하지 않는다.
Static/frozen comparator와 compute/action matching rule은 B1 outcome 전에 고정하며, 결과를 본 뒤
`best` comparator를 선택하지 않는다.

---

# 22. Claims and nonclaims

## 22.1 현재 허용되는 motivation claim

> Official multi-layer writers can exhibit large, layer-dependent gaps between allocated and realized
> activation progress. The observed mismatch includes sparse heavy tails and orthogonal components,
> while semantic endpoint success can coexist with substantial terminal direct-target mismatch.

## 22.2 현재 허용되는 method claim

> We remove the explicit remaining-layer residual quota and treat each official-form closed-form writer
> as a state-dependent actuator driven by the full current residual. A response-linearized realization
> potential defines an \(h\)-independent velocity multiplier, and ordered shallow-to-deep splitting
> rebuilds every downstream actuator after the realized upstream transition.

## 22.3 실험 성공 후에만 가능한 claim

> Within the same official writer family and at matched immediate efficacy, response-guided ordered
> integration reduces realization/action tails or improves held-out retention relative to no-quota,
> frozen-field and official baselines.

## 22.4 금지 claim

- Baseline은 cross-layer state를 전혀 보지 않는다.
- Residual debt를 최초로 발견했다.
- CAKE는 residual을 layer마다 재계산하지 않는다.
- \(R/n\)은 일반적으로 또는 전역적으로 최적이 아니다.
- Effective workload는 globally optimal allocation이다.
- Direct-\(z\) debt가 forgetting의 원인이다.
- \(C_\ell/P_\ell\) geometry가 canonical locality를 보장한다.
- No-rollback Euler가 finite-step hard barrier를 보장한다.
- Semantic first-hit이 잘못된 \(z^\star\)를 수정한다.
- 이 방법은 완전히 reference-free다.
- Phase A가 barrier 또는 ODE benefit을 이미 입증했다.

---

# 23. Promotion and kill criteria

다음 중 하나면 해당 claim을 닫는다.

- JVP가 multi-epsilon FD 또는 virtual/materialized identity를 통과하지 못함
- Fixed-\(T\)에서 \(h\) 감소에 따른 endpoint/defect convergence가 없음
- Response multiplier active fraction이 사실상 0이거나 no-quota arm과 동일
- Ordered refreshed가 frozen/Jacobi와 구분되지 않음
- Multi-sweep가 one sweep 대비 mechanism/endpoint 차이를 만들지 못함
- First-hit ON이 matched efficacy에서 action/compute를 줄이지 못함
- B100에서 response arm의 realization/action tail이 no-quota arm보다 악화
- Held-out RS/PS/NS 또는 lifelong retention이 matched efficacy에서 개선되지 않음

마지막 항목이 실패하면 realization response는 writer-mechanics diagnostic으로만 남기고 locality
또는 retention method로 promotion하지 않는다.

---

# 24. Open items before implementation release

아래 항목은 코드 작성 전에 outcome-blind numerical lock으로 닫아야 한다.

1. MEMIT과 AlphaEdit의 \(C_\ell^{\rm reg}\) exact-object binding 및 action telemetry numerical receipt
2. Fixed \(T_{\max}=1\), \(N=2,4,8\) empirical Cauchy distance/ratio tolerance
3. Target-bearing compute-\(z\) context extraction과 batch-level strict multi-token aggregation identity
4. Strict-z diagnostic arm의 outcome-blind tolerance와 main-result exclusion seal
5. Virtual factor rank ceiling과 memory fail-close boundary
6. JVP backend의 exactness, FP32/FP64 reduction boundary와 FD epsilon grid
7. MEMIT/AlphaEdit 공통 telemetry schema와 family-specific solve/finalization receipt

이 open item은 held-out result를 본 뒤 바꾸지 않는다.

---

# 25. Final research position

본 proposal의 핵심은 layer별 heuristic coefficient를 하나 더 만드는 것이 아니다. 또한
Official writer를 frozen update로 잘게 나누는 것도 아니다.

\[
\boxed{
\begin{aligned}
&\text{frozen baseline }z^\star\\
&+\ \text{official-form full-residual current-state actuators}\\
&+\ \text{current response only로 정한 }u(W),\quad \alpha=hu\\
&+\ \text{architectural ordered rebuild and multi-sweep revisit}\\
&+\ \text{one-way Euler observation without algorithmic rollback}\\
&+\ T_{\max}=Nh=1,\ N=4,\ h=0.25\\
&+\ \text{target-new strict first-hit or committed finite miss/stall}\\
&+\ \text{residual/action telemetry outside the controller}\\
&+\ \text{terminal one-shot commit and method-specific finalization}
\end{aligned}
}
\]

Phase A는 이 연구 질문을 여는 evidence이지 정답이 아니다. 첫 scientific objective는
`barrier가 좋다`를 보이는 것이 아니라 다음 세 항을 분리해 반증 가능하게 만드는 것이다.

1. prescribed quota 제거가 필요한가,
2. current response multiplier가 필요한가,
3. ordered realized-state rebuild와 revisit가 필요한가.

세 비교가 닫힌 뒤에만 Barrier-Guided ODE-Edit이라는 이름을 성능 claim으로 승격한다.
