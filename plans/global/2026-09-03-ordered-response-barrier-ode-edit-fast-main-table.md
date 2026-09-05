# Ordered Response-Barrier ODE-Edit fast main-table 실행 계획

## 0. 상태와 목적

- 기준일: `2026-09-03`
- plan ID: `ODEEDIT-ORBODE-FAST-MAIN-V0`
- proposal:
  `project/proposals/2026-09-03-ordered-response-barrier-ode-edit-proposal.md`
- proposal identity: SHA256
  `03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be`,
  55,900 bytes, 1,359 lines
- 실행 성격: 빠른 mechanism main-table pilot와 이어지는 paired B100 confirmatory
- scientific promotion: `false` until the full paired table and audits close

사용자 우선순위에 따라 장기간의 별도 G0 audit를 선행하지 않는다. 대신 각 GPU cell이
모델을 한 번 load한 직후 짧은 fail-close runtime preamble을 수행하고, 통과하면 같은
process에서 즉시 B100 실험으로 이어간다. 이 preamble은 생략 가능한 형식 검사가 아니라
명백히 잘못된 JVP, stale state, arm contamination 또는 partial commit으로 GPU main table 전체가
무효가 되는 것을 막는 최소 실행 검증이다.

이번 fast table이 직접 답할 질문은 다음 세 가지다.

1. 같은 fixed-time repeated schedule에서 explicit (R/n) quota를 유지할 때와 제거할 때 실제
   trajectory와 endpoint가 다른가?
2. Full-residual actuator를 (u=1)로 쓰는 것보다 current terminal response로 정한
   (u(W))가 realization tail을 줄이는가?
3. Upstream actual virtual transition 뒤 downstream field를 다시 만드는 per-visit refresh가
   sweep-entry Jacobi field와 다른가?

Fixed-(z) overtracking은 동일 response trajectory에서 얻는 target-context strict first-hit
endpoint를 fixed-horizon endpoint와 비교하는 secondary question으로 둔다. Strict-(z) tolerance
arm, CAKE causal weighting, external prompt bank, output/history barrier와 lifelong run은 이번 첫
submission에 넣지 않는다.

---

## 1. 변하지 않는 method contract

### 1.1 Controller와 integrator

현재 virtual state (W)에서 request (i)의 frozen direct-target residual을

\[
R_i(W)=z_i^\star-H_i(W),
\qquad
s_i=\|R_i(W_0)\|_2
\]

로 둔다. Layer ℓ의 official-form full-residual writer direction은

\[
B_\ell(W)=\operatorname{Writer}^{\rm official\mbox{-}form}_\ell
\bigl(W,R(W),K_\ell(W);C_\ell,P_\ell,\mathcal H_{\rm entry}\bigr)
\]

이다. JVP는 writer construction을 미분하지 않고 detached direction에 대한 terminal
activation directional derivative만 계산한다.

\[
\psi_{i,\ell}(W)
=D_{\Delta=0}H_i\bigl(W\oplus_\ell\Delta\bigr)
\left[B_\ell(W)^{\rm stopgrad}\right].
\]

FP64 reduction으로

\[
g_\ell=\frac1B\sum_i
\left\langle\frac{R_i}{s_i},\frac{\psi_{i,\ell}}{s_i}\right\rangle,
\qquad
r_\ell=\frac1B\sum_i
\left\|\frac{\psi_{i,\ell}}{s_i}\right\|_2^2
\]

를 계산하고,

\[
u_\ell(W)=
\begin{cases}
0,&r_\ell=0\text{ or }g_\ell\le0,\\
\min\{1,g_\ell/r_\ell\},&g_\ell>0
\end{cases}
\]

로 둔다. Integrator만 다음 coefficient를 만든다.

\[
\boxed{\alpha_{n,\ell}=h u_\ell(W_{n,\ell})}.
\]

Main clock은 모든 ODE arm에서

\[
\boxed{T_{\max}=Nh=1,\qquad N=4,\qquad h=0.25}
\]

로 고정한다. (N)은 layer visit 수가 아니라 ordered full-sweep 수이며, 다섯 layer를 쓰므로
fixed-horizon arm은 최대 20번의 layer operator를 적용한다.

### 1.2 Ordered state propagation

순서는 항상

```text
[L4, L5, L6, L7, L8] x 4 sweeps
```

이다. Per-visit arm은 한 layer step을 실제 virtual overlay에 advance한 직후 state version을
올리고, 다음 layer의 (R,K,B,\psi,u)를 새 state에서 모두 다시 계산한다. 이전 version에서 만든
객체의 소비는 technical invalid다.

### 1.3 Barrier의 제한된 의미

이번 main의 barrier는 external output/history inequality 집합이 아니다. (V_z)의 current
directional response가 non-descending이면 (u=0)으로 만드는 local response barrier다.
No-rollback finite Euler step의 hard invariance는 주장하지 않으며 actual potential 증가와
entry-set crossing은 telemetry로 기록한다.

### 1.4 (q)와 action

Residual excursion (q_i^{\rm res}), parameter action과 factor rank는 controller, α, terminal
predicate에 절대 들어가지 않는다. Nonfinite만 numerical invalid다. (C_\ell^{\rm reg})의 exact
binding이 fast implementation에서 준비되지 않으면 endpoint 실행을 막지 않고 native action
열을 `TELEMETRY_WITHHELD`로 둔다. 다음은 항상 계산한다.

- per-layer and net ΔW Frobenius norm
- accumulated factor rank
- (q_i^{\rm res}) maximum/terminal distribution
- forward/JVP/solve count, runtime와 peak memory

Raw action magnitude는 architecture 간 우열 비교에 사용하지 않는다.

### 1.5 Semantic endpoint

Fixed-horizon response trajectory가 numerical authority다. Compute-(z) loss builder가 실제
target loss에 사용한 target-bearing contexts의 모든 target-new token에 대해 strict
teacher-forced top-1을 검사하고, 이를 정확히

```text
TARGET_CONTEXT_STRICT_HIT
```

이라고 부른다. 이는 canonical RS 또는 일반 semantic correctness가 아니다.

Response fixed-horizon trajectory는 이 predicate와 무관하게 (T=1)까지 적분한다. 각 post-layer
prefix state에서 predicate를 관측하여 최초 hit factor prefix를 보존한다. 직접 early-stop run과
prefix-derived endpoint가 B1 smoke에서 weight/activation/logit identity를 통과하면 별도 ON
trajectory를 다시 적분하지 않고 같은 trajectory에서 first-hit endpoint를 정확히 materialize한다.

### 1.6 Failure policy

- candidate reject, coefficient shrink, retry, backtracking, best-waypoint selection과 official
  fallback: 모두 0회
- finite horizon miss/stall: 해당 finite endpoint를 평가하고 intention-to-edit denominator에 포함
- locked numerical boundary: persistent mutation 없이 해당 cell을 중단; silent no-op continuation 금지
- technical runtime interruption: 이미 완료된 immutable round artifact는 보존하되 자동 scientific
  rerun 금지
- experimental arm/round isolation을 위한 W0/cache restore는 method rollback이 아니라 harness reset으로
  별도 count한다.

---

## 2. 데이터, models와 병렬화

### 2.1 공통 표본

기존 sealed stream만 사용한다.

- loader: `project/run_scripts/ode_bf/p1r52_b100x10_stream.py`
- lock: `project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json`
- stream root: `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a`
- order root: `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3`
- structure: ten sealed B100 cohorts

이번 mechanism table에서 각 cohort는 동일 entry (W_0)와 동일 entry method state에서 시작한다.
즉 열 개 cohort 사이에 edit를 누적하지 않는다. Lifelong stream이 아니다.

Round 0 완료 즉시 `FAST-B100-V0` interim table을 발행하고, 같은 jobs가 round 1–9를 이어서
`PAIRED-B100x10-V1` table을 완성한다. Scientific outcome이 나쁘다는 이유로 round를 중단하지
않으며, technical invalid boundary만 cell을 중단한다.

### 2.2 Four cells

| Array cell | Model | Writer family |
|---:|---|---|
| 0 | Llama-3-8B-Instruct locked snapshot | MEMIT |
| 1 | Llama-3-8B-Instruct locked snapshot | AlphaEdit |
| 2 | Qwen2.5-7B-Instruct locked snapshot | MEMIT |
| 3 | Qwen2.5-7B-Instruct locked snapshot | AlphaEdit |

`0-3%4`, cell당 GPU 1개로 병렬 실행한다. 각 cell은 model을 한 번 load하고 ten cohorts를 끝까지
처리한다. Model, method, stream, hparams, Wikipedia moments와 AlphaEdit projector/history는
기존 sealed baseline artifacts만 사용한다. Cache miss download/recompute는 fail-close한다.

---

## 3. Fast main-table arms

모든 non-Official arm의 first-hit은 trajectory dynamics에 영향을 주지 않도록 fixed-horizon OFF로
실행한다.

| ID | Residual command | Response | Refresh | (N,h,T) | 목적 |
|---|---|---|---|---|---|
| `O` | stock remaining-layer (R/n_\ell) | stock | official one pass | stock | 실용 baseline |
| `QCL` | current (R/n_\ell) | (u=1) | per visit | (4,.25,1) | repeated quota policy |
| `NQFIX` | full current (R) | (u=1) | per visit | (4,.25,1) | explicit quota 제거 |
| `ORBFH` | full current (R) | current (u(W)) | per visit | (4,.25,1) | canonical fixed-horizon response flow |
| `JAC` | full sweep-entry (R) | sweep-entry (u(W)) | per sweep | (4,.25,1) | within-sweep realized-state rebuild 제거 |
| `ORBHit` | `ORBFH`의 exact prefix | same path | same path | first hit | stopping-only derived endpoint |

`JAC`는 fully entry-frozen arm이 아니다. 각 sweep 시작 state에서 다섯 layer의
(R,K,B,\psi,u)를 모두 생성하고, 그 sweep 안에서만 frozen field를 적용한다. 다음 sweep
시작에는 전부 refresh한다.

`QCL`에서는 매 sweep의 shallow-to-deep visit마다 (n_\ell=(5,4,3,2,1))을 다시 적용하고
(B_\ell(R/n_\ell))에 (h=0.25)를 곱한다. 이 arm은 이번 표에서 필수다. 이를 빼면
`NQFIX` 변화가 (R/n) 제거 때문인지 repeated integration 때문인지 답할 수 없다.

Strict-(z), fully-entry-frozen, grouped ordering, (q)-constrained, CAKE와 external-reference
arms는 fast table에서 제외한다.

---

## 4. 한 process 안의 실행 순서

각 model×method cell은 다음 순서를 고정한다.

1. session/repository, source manifest, HF snapshot, stream, hparams, Wikipedia moments,
   projector와 evaluator identity 확인
2. model (W_0) 한 번 load 및 selected-weight bytes/pointers와 method state snapshot
3. 각 cohort에서 family-native compute-(z)를 (W_0)에서 정확히 한 번 수행하고 arm들이 같은
   tensor와 request/context order를 공유
4. round 0의 첫 request로 아래 runtime preamble 수행
5. arm order를 `O -> QCL -> NQFIX -> ORBFH -> JAC -> ORBHit(materialize only)`로 고정
6. 각 arm은 entry W0/cache에서 시작하고 terminal evaluation 후 harness reset
7. round별 raw-free metrics와 immutable artifact receipt를 즉시 flush
8. round 0 뒤 interim summary 작성, 이어서 round 1–9 수행
9. terminal aggregate report 작성; result를 보고 재실행하거나 arm 순서를 바꾸지 않음

Arm isolation restore count와 method-internal rollback count를 서로 다른 열로 기록한다.

---

## 5. 최소 runtime preamble

별도 장기 audit job은 만들지 않는다. 다음 검사를 각 array cell의 실제 model process에서 수행한다.

### P0. Official/scaling

- `O` wrapper와 direct stock call의 selected weight/activation/logit parity
- fixed state에서 (B(R/n)=B(R)/n) relative error receipt
- compute-(z) count 1, request/context order identity

### P1. JVP

- 첫 nonzero response layer에서 detached (B) forward JVP와 central finite difference 비교
- FD ε grid: (2^{-7},2^{-8},2^{-9}); primary receipt (2^{-8})
- JVP/FD cosine, relative (L_2), sign과 virtual/materialized response 기록
- writer solve, residual, key, (C/P), history와 (u)를 통한 gradient path 0
- nonfinite 또는 direction/sign contradiction이면 cell fail-close

Preamble은 exhaustive all-layer certification이 아니므로 보고서에서 `FAST_RUNTIME_PREAMBLE`로
부르고 full G0 authority라고 쓰지 않는다.

### P2. State/transaction

- L4 advance 뒤 L5 key/writer/state-version이 changed state를 읽음
- stale object consumption 0
- same-layer repeated factors를 포함한 FP32 overlay와 one-time shadow materialization parity
- inner persistent weight/cache/history mutation, algorithmic retry/rollback 0
- AlphaEdit은 terminal model에서 모든 layer key를 다시 계산해 native history delta를 한 번 만들고,
  intermediate/path key는 append하지 않음
- MEMIT static `COV_CACHE` identity 유지

### P3. One-case resolution smoke

- first-hit OFF, (T=1), 동일 첫 request에서 (N=2,4,8) endpoint와 defect를 기록
- 이 결과는 catastrophic resolution failure를 찾는 smoke이며 B100 (N=4)의 정식 convergence
  claim이 아님

---

## 6. Telemetry와 main tables

### 6.1 항상 정의되는 request-level mechanism metrics

한 visit의 α가 0이면 relative debt를 0으로 넣지 않고 `SKIPPED_ZERO_COMMAND`로 둔다. (s_i=0)은
`ZERO_ANCHOR`로 분리한다. 항상 정의 가능한 entry-normalized 값은 다음이다.

\[
D_{i,k}^{R_0}
=\frac{\|\alpha_kR_{i,k}-Y_{i,k}^{\rm act}\|^2}{s_i^2},
\]

\[
D_{i,k}^{\rm model,R_0}
=\frac{\|Y_{i,k}^{\rm act}-\alpha_k\psi_{i,k}\|^2}{s_i^2}.
\]

Batch-global (u_\ell)가 positive여도 request별

\[
g_{i,\ell}=\langle\bar R_i,\bar\psi_{i,\ell}\rangle
\]

가 nonpositive일 수 있다. 따라서 다음을 반드시 저장한다.

- batch (g,r,u)와 (u=0,(0,1),1) visit counts
- request별 (g_i\le0) fraction when batch (g>0)
- request별 actual ΔV worsening fraction
- (D^{R_0}), (D^{\rm model,R_0}), orthogonal response와 (q^{\rm res}) tails
- zero-command인데 batch coupling으로 (Y_i\ne0)인 cross-response count

이 표는 request-specific controller를 주장하지 않는다. Controller는 layer당 batch-global scalar
하나이며, 개선 여부는 request-tail 결과로만 판단한다.

### 6.2 Table A — endpoint

각 model×method×arm에 대해 다음을 intention-to-edit denominator로 보고한다.

- canonical current-B100 RS, PS, NS
- target-new/target-true NLL advantage
- target-new strict와 `TARGET_CONTEXT_STRICT_HIT` coverage
- `ENTRY_ALREADY_HIT`, first hit, horizon miss, stall, numerical/technical boundary count
- terminal-valid / scientific-attempted / technical-attempted denominators

PS/NS, rephrase와 neighborhood output은 controller나 run-time branching에 사용하지 않는다.

### 6.3 Table B — mechanism

다음을 mean 하나로 요약하지 않고 median, p90, p99, CVaR90와 max로 보고한다.

- maximum/terminal (q^{\rm res})
- (D^{R_0}), (D^{\rm model,R_0})
- actual/predicted response cosine와 ratio
- orthogonal response energy
- negative actual ΔV rate와 discrete barrier defect
- layer workload Ωℓ, skip/revisit와 first-hit ordinal

### 6.4 Table C — compute/action

- compute-(z), state forward, semantic forward, JVP, FD, writer solve counts
- end-to-end/write-stage wall time
- peak allocated/reserved GPU memory
- factor count/rank, materialization/commit/history-finalization count
- path/net Frobenius action
- native (C^{\rm reg}) action or `TELEMETRY_WITHHELD`

Diagnostic FD와 smoke compute는 main controller compute와 분리한다.

---

## 7. 비교와 허용되는 해석

| 비교 | 직접 답하는 질문 | 금지 해석 |
|---|---|---|
| `QCL` vs `NQFIX` | 같은 repeated clock에서 explicit (R/n) quota 제거 효과 | globally optimal allocation 발견 |
| `NQFIX` vs `ORBFH` | current response multiplier의 추가 효과 | ODE 전체 효과 또는 causal layer importance |
| `ORBFH` vs `JAC` | within-sweep upstream-to-downstream field rebuild 효과 | 모든 cross-layer interaction의 인과 식별 |
| `ORBFH` vs `ORBHit` | 동일 path에서 first-hit stopping 효과 | (z^\star) 자체를 수정함 |
| `O` vs others | stock editor 대비 operational frontier | 단일 구성요소 attribution |

Main table에서 response tail이 개선되어도 locality/retention의 원인이라고 쓰지 않는다. Canonical
NS와 PS 개선이 관측되면 held-out association으로만 보고하고, lifelong causal claim은 별도
sealed run 전까지 금지한다.

---

## 8. Fast continuation rule

### Interim `FAST-B100-V0`

Round 0의 네 cells가 terminal이면 즉시 표를 만든다. 한 cell이 technical invalid면 나머지 cells의
수치는 보존하되 four-cell main table은 `INCOMPLETE_TECHNICAL`로 표시한다.

### Full `PAIRED-B100x10-V1`

각 cell의 runtime preamble이 통과하면 scientific outcome을 기다리지 않고 같은 process에서
round 1–9를 계속한다. 다음은 hard stop이다.

- nonfinite model/JVP/writer response
- stale state consumption
- persistent inner mutation 또는 arm contamination
- overlay/shadow identity failure
- source/stream/evaluator identity mismatch
- GPU OOM 또는 resource cap violation

낮은 RS, 큰 debt 또는 zero response는 기술 실패가 아니므로 기록하고 계속한다.

### 다음 단계

Full table 후에만 다음을 결정한다.

- quota claim: `QCL` vs `NQFIX`
- response-barrier claim: `NQFIX` vs `ORBFH`
- ordered rebuild claim: `ORBFH` vs `JAC`
- first-hit claim: `ORBFH` vs `ORBHit`

Mechanism 신호가 없으면 lifelong을 실행하지 않는다. 신호는 있으나 PS/NS가 개선되지 않으면
writer-mechanics result로 유지한다. Immediate efficacy와 mechanism tail이 함께 살아남을 때만
동일 v6 protocol의 1k smoke를 별도 승인한다.

---

## 9. 구현 및 산출물 경계

신규 구현은 historical controller에 덧대지 않고 독립 package로 둔다.

```text
project/run_scripts/ordered_response_barrier_ode/
  contracts.py
  adapters.py
  fp32_overlay.py
  terminal_jvp.py
  semantic.py
  integrator.py
  telemetry.py
  runtime.py
  preflight.py
  dry_plan.py
  report.py
  locks/
  tests/
```

재사용 대상은 interface/engineering primitive에 한정한다.

- sealed stream loader: `ode_bf/p1r52_b100x10_stream.py`
- AlphaEdit full-current-residual writer primitive: `ode_bf/p1_backend.py`
- official four-cell runtime/evaluator adapters from current `origin/main`
- JVP implementation pattern: `barrier_guided_ode/s1_alphaedit_runtime.py`
- semantic token/span primitives: `ode_bf/target_new_nll.py`
- transaction pattern: `ode_bf/transaction.py`
- endpoint evaluator: `ode_bf/scalable_batched_evaluator.py`

다음 historical scientific objects는 import하거나 복사하지 않는다.

- CAKE causal scores/weights 또는 external tracing set
- BGODE event/Fisher/two-equality controllers
- FzCB homotopy/QP/backtracking controller
- previous debt-priority, learned routing, oracle selector와 reference prompt bank

Raw outputs와 logs는

```text
local/ordered-response-barrier-ode/fast-main-v0/
```

에만 둔다. Git에는 source, numerical/source locks, compact metrics, path/checksum manifest, factual
server report와 global interpretation report만 둔다.

