# Fixed-z Functional Safe Write — Fast Falsification Plan

- 작성 기준: **2026-08-30 KST**
- 상태: **`RESEARCH_DIRECTION_LOCK_V2_MAIN_ALIGNED; EXECUTION_NOT_STARTED`**
- 확인한 remote main: `edf2ea2c98c5` (`proposal fixed-z-functional-safe-write by head-server1-gh`)
- 적용한 main audit: `audits/global/2026-08-30-fixed-z-functional-safe-write-proposal-review.md`
- 상위 proposal:
  [`2026-08-30-fixed-z-functional-safe-write-proposal.md`](../../project/proposals/2026-08-30-fixed-z-functional-safe-write-proposal.md)
- 현재 session 역할: **연구 방향, 실험 계약, Go/No-go 해석 전용**
- 1차 paired screen: `llama3-8b-inst`, `qwen2.5-7b-inst`
- 1차 editor: MEMIT
- 1차 범위: single edit, single editable layer, frozen direct-\(z\)

이 문서는 큰 방법을 먼저 구현하지 않고 가장 싼 반증부터 수행하기 위한 실행 방향
lock이다. GPU 실행 승인, source 구현 완료 또는 scientific claim을 뜻하지 않는다.

---

# 1. 이 session의 역할

현재 Codex session은 다음만 담당한다.

- 연구 질문과 claim boundary 정리
- 실험 arm, 데이터 방화벽, metric과 decision gate 설계
- 결과가 도착한 뒤 과학적 해석과 다음 방향 결정
- proposal 또는 후속 plan 갱신

다음은 별도 실행 흐름으로 넘긴다.

- model/GPU job 제출과 모니터링
- source 구현과 test
- raw artifact 생성
- server별 factual report

이 session에서 방향을 바꿀 때에는 outcome을 본 뒤 기존 threshold를 고치는 대신 새로운
date-stamped revision을 작성한다.

---

# 2. 가장 빠른 연구 질문

첫 질문은 ODE가 아니다.

> 동일한 direct-\(z\) progress, edit/history key response와 covariance action을 갖는 실제 W
> 후보들이 held-out model function에서는 서로 다르게 동작하는가?

두 번째 질문은 다음이다.

> Controller prompt에서 계산한 functional tangent가 gate-validation prompt에서도 더 안전한 후보를
> 선택하며, 이 차이가 이미 고정한 key-space geometry를 넘어서는가?

세 번째 질문만 ODE와 관련된다.

> 같은 constrained endpoint를 static solver가 직접 찾을 수 있는데도, state-dependent
> functional field를 재측정하는 trajectory가 compute-matched 추가 가치를 갖는가?

따라서 실행 순서는 고정한다.

```text
F0  algebra/identity
  -> F1  same-z functional multiplicity
  -> F2  functional signal의 gate-bank incremental value
  -> F3  static functional safe write
  -> F4  refreshed trajectory의 필요성
  -> 그 뒤에만 multi-layer MEMIT/AlphaEdit와 sequential
```

F1 또는 F2가 실패하면 barrier와 ODE method 구현을 시작하지 않는다.

---

# 3. 가장 작은 식별 가능한 testbed

## 3.1 Equality set

마지막 MEMIT editable layer 하나에서 모든 authorized rewrite context의 lookup-position edit
key를

\[
K_E\in\mathbb R^{d_{in}\times n_E}
\]

로 쌓는다. Fixed direct target \(z^\star\)에 필요한 context별 output residual을

\[
R_E\in\mathbb R^{d_{out}\times n_E}
\]

로 둔다.

Primary F1/F2 shared screen에서는 controller가 접근 가능한 finite unedited reference/history
key bank \(K_H\)도 함께 고정한다. 이 bank는 실제 sequential history 전체가 아니라
Alpha/Beta 계열 key-protection을 모사하는 finite registered set이다.

\[
A_0=[K_E,K_H],
\qquad
B_0=[R_E,0].
\]

Closed-form reference를 만든 뒤 canonical rewrite의 모든 teacher-forced target-token position에서
해당 layer input key \(K_T\)를 capture한다. Tangent equality matrix는

\[
A=[K_E,K_H,K_T]
\]

로 확장한다. Base update는 \(\Delta_C A_0=B_0\), 모든 null correction은 \(NA=0\)을
만족한다. Target-token position을 prefix order로 모두 고정하므로 algebraically candidate끼리
canonical teacher-forced target path의 layer output과 downstream target logits도 동일해야 한다.
실제 FULL_FP32 실행에서는 preregistered numerical tolerance와 byte/identity receipt로 이를
검사한다. 따라서 유효 후보끼리는 다음이 동일하다.

- direct-\(z\) local activation progress
- authorized edit-context response
- registered finite history-key response
- canonical target-token path response와 target logits

이 equality는 모든 지식 보존 보장이 아니라 **검사한 finite key bank에 한정된 identity**다.

## 3.2 Closed-form reference

Raw reference covariance를 \(C_0\)라 하고 native MEMIT과 동일하게 outcome 전에 고정한
regularization을 사용해

\[
C=C_0+\lambda I\succ0
\]

로 둔다. 이후 이 문서의 C-metric과 inverse action은 모두 이 regularized \(C\)를 뜻한다.
Minimum-\(C\)-action exact write는

\[
\Delta_C
=
B_0(A_0^\top C^{-1}A_0)^{-1}A_0^\top C^{-1}.
\]

이는

\[
\min_\Delta \operatorname{tr}(\Delta C\Delta^\top)
\quad\text{subject to}\quad
\Delta A_0=B_0
\]

의 해다. Rank deficiency, solve residual 또는 condition failure는 ODE로 숨기지 않고
`registered-equality-infeasible`로 기록한다.

구현에서는 \(C^{-1}\) 또는 Gram inverse를 dense tensor로 materialize하지 않는다. 고정된
factorization/solve backend로 inverse action을 계산하고, backend, numerical rank, residual,
spectrum과 condition estimate를 receipt에 기록한다.

## 3.3 Same-endpoint tangent

Null correction \(N\)은

\[
NA=0
\]

를 만족한다. 이때

\[
\Delta_\pm=\Delta_C\pm\gamma N
\]

은 동일 equality endpoint를 갖는다. \(N\)을 \(C\)-normalize하면 minimum-action 해의
직교성으로

\[
\langle\Delta_C,N\rangle_C=0
\]

이고, 따라서 같은 \(\gamma\)의 모든 후보는

\[
\|\Delta_\pm\|_C^2
=
\|\Delta_C\|_C^2+\gamma^2\|N\|_C^2
\]

로 action도 동일하다.

Primary tangent energy는

\[
\gamma^2\|N\|_C^2
=
\rho_{\mathrm{tan}}\|\Delta_C\|_C^2,
\qquad \rho_{\mathrm{tan}}=0.05
\]

로 lock한다. \(\rho_{\mathrm{tan}}\in\{0.01,0.10\}\)은 technical two-case sensitivity panel에서만
기록하며 main screen의 값을 결과에 맞춰 바꾸지 않는다.

## 3.4 Functional tangent

Controller risk를 \(D_{ctrl}\)라 하고 closed-form reference endpoint
\(\theta_0+\Delta_C\)에서 계산한 weight gradient를 \(G\)라 하면, \(-GC^{-1}\)를 equality
tangent에 \(C\)-projection한 방향을 functional-safe candidate axis로 사용한다. W0에서는
teacher KL gradient가 0이므로 functional sign을 W0의 1차 gradient에서 만들지 않는다.

Projection은 다음 두 조건을 실제 tensor identity로 검사해야 한다.

\[
N_F A=0,
\qquad
\langle G,N_F\rangle<0.
\]

반대 부호 \(-N_F\)는 paired adverse control이다. Safe/adverse 명칭은 gate/audit outcome이
아니라 controller derivative의 부호만으로 candidate receipt 전에 고정한다.

Functional scalar의 MVP는 fixed generic prompt에서 teacher model 대비 **next-token KL**다.
Sequence-wide KL, margin group과 capability panel은 F3 이후에만 확장한다.

F1/F2의 한 개 aggregate functional axis는 aggregate \(D_{ctrl}\)의 gradient이므로 ordinary
backward 한 번을 사용할 수 있다. 이를 per-anchor Jacobian이라고 부르지 않는다. F3에서
anchor/group별 constraint row가 필요해지면 다음 중 한 방식을 outcome 전에 고정한다.

- batched VJP 또는 `vmap(grad)`
- anchor/group별 backward

실제 VJP/backward 수와 grouping을 compute receipt에 기록한다.

---

# 4. 데이터 방화벽

## 4.1 여섯 데이터 역할

| 집합 | 역할 | candidate 선택 접근 |
|---|---|---|
| \(Q_{edit}\) | direct-\(z\), \(K_E\), edit efficacy | 허용 |
| \(Q_{hist}\) | finite key identity \(K_H\) | 허용 |
| \(Q_{cal}\) | no-op, duplicate candidate, numerical floor | outcome 전 허용 |
| \(Q_{ctrl}\) | functional gradient와 candidate 선택 | 허용 |
| \(Q_{gate}\) | F1--F4 Go/No-go와 architecture selection | controller에는 금지 |
| \(Q_{audit}\) | 모든 선택 뒤 최종 estimate | 완전 금지; 마지막에 한 번만 open |

`Q_gate` prompt text와 teacher logits는 모든 candidate tensor, sign, scale, action과 receipt가
고정된 뒤에만 연다. `Q_gate`를 본 뒤에는 같은 prompt family를 final audit에 재사용하지 않는다.
`Q_audit`은 architecture, method, arm, threshold, endpoint rule과 모든 Gate 판정이 freeze된 뒤
정확히 한 번만 연다. 그 결과를 보고 같은 study의 방법을 고치지 않는다.

## 4.2 Split 원칙

- CounterFact source hash와 deterministic salted rank를 기록한다.
- F1/F2 scientific screen의 edit case는 기존 사용 범위 `[0:148]`과 분리한다.
- 1차 fresh edit block은 salted ranks `[148:156]`, 모델별 동일 8 case, 총 16 model-case다.
- Technical development에는 기존 `[132:140]` artifact를 사용할 수 있지만 scientific
  denominator에 넣지 않는다.
- Primary F1/F2 shared run은 처음부터 \(K_E\)와 32개 \(K_H\)를 함께 equality에 넣는다.
  F1-T에서 edit-only와 edit+history solve를 모두 점검해 latter의 feasibility를 먼저 확인한다.
- `Q_hist`, `Q_cal`, `Q_ctrl`, `Q_gate`, `Q_audit`는 edit block과 row, source/semantic family,
  subject--relation, exact prompt
  hash가 겹치지 않게 별도 고정한다.
- Generic bank는 충분히 뒤쪽의 deterministic rank stream에서 필요한 수만 순서대로
  선택한다. Collision row는 outcome을 보기 전에 manifest builder가 기계적으로 건너뛴다.
- F3/F4는 F1/F2 gate 결과를 본 뒤 설계가 영향을 받을 수 있으므로 **새 edit block과 새 gate
  bank**를 사용한다. Final `Q_audit`은 어느 Gate에도 사용하지 않는다.

## 4.3 Fast screen 크기

| 집합 | F1/F2 paired screen |
|---|---:|
| fresh edits | 8/model; 16 model-case |
| history keys | 32 |
| calibration prompts | 8 |
| controller prompts | 8 |
| gate-validation prompts | 32 |
| final audit prompts | sealed; screen에서 0회 open |
| random tangent axes | 3 |
| functional tangent axes | 1 |
| signs per axis | 2 |
| candidates per edit | 8 |

\(\Delta_C\) reference endpoint도 edit당 한 번 평가하지만, tangent energy가 없는 reference이므로
matched-action 8-candidate spread denominator에는 넣지 않는다.

32개 gate prompt에서 `CVaR_0.95`는 표본이 너무 작으므로 screen primary tail metric은
`CVaR_0.875`로 고정한다. Canonical confirmatory stage에서는 gate prompt를 128개로 늘리고
`CVaR_0.95`를 사용한다.

---

# 5. F0 — algebra와 split identity

## 목적

GPU 없이 수식 구현이 맞고 단순 step 분할이 새 효과를 만들지 않음을 확인한다.

## 범위

- float64 synthetic \(C,K_E,K_H,R_E\)
- C-minimum exact write
- random/functional tangent projection
- equal target, equal history response와 equal action
- frozen \(\Delta\)의 one-shot 대 exact algebraic split endpoint
- quadratic objective의 analytic finite-time gradient flow 대 Euler refinement

## mandatory checks

1. \(\|\Delta_C A_0-B_0\|/\|B_0\|\le10^{-10}\)
2. \(\|N A\|/(\|N\|\|A\|)\le10^{-10}\)
3. \(|\langle\Delta_C,N\rangle_C|\) relative error \(\le10^{-10}\)
4. paired candidate C-action relative difference \(\le10^{-10}\)
5. frozen algebraic split endpoint relative difference \(\le10^{-8}\)
6. Euler endpoint가 같은 finite \(T\)의 analytic flow로 step refinement 수렴

하나라도 실패하면 GPU HOLD다. Split-only 차이가 발생하면 ODE signal이 아니라 구현 또는
objective mismatch로 판정한다. Quadratic gradient flow는 일반적으로 finite \(T\)에 closed-form
minimizer와 같지 않으며, \(T\to\infty\) limit에서만 minimizer 접근을 검사한다.

---

# 6. F1-T — two-case real-model technical smoke

## 범위

- model: `llama3-8b-inst`, `qwen2.5-7b-inst`
- cases: 기존 `[132:140]` 중 같은 identity의 첫 1개/model
- layer: model hparams의 마지막 MEMIT editable layer
- direct-\(z\): W0에서 case당 정확히 한 번
- Gate 0--2 authoritative dtype: `FULL_FP32`
- scientific comparison: 금지

## 확인할 것

- actual EasyEdit key/residual orientation
- \(A_0^\top C^{-1}A_0\) rank와 solve residual
- local \(\Delta_C A_0=B_0\), tangent \(NA=0\), virtual full-model rewrite activation과
  fixed-\(z\) realization의 분리 identity
- target event와 target-token logit/margin realization
- virtual trial의 no-mutation/rollback identity
- \(\rho_{\mathrm{tan}}=0.01,0.05,0.10\)의 finite behavior
- candidate당 controller/gate forward 수와 peak memory
- 별도 low-precision technical panel의 FP32 대비 endpoint discrepancy

Primary F1/F2 estimator와 identity authority는 `FULL_FP32` virtual/commit이다. BF16 또는 다른
low-precision panel은 별도 study ID의 technical secondary panel이며 Gate score, noise floor,
threshold 또는 case validity에 섞지 않는다. Low precision이 candidate identity를 무너뜨리면
그 결과를 숨기지 않고 F3 뒤 별도 deployment gate를 설계한다. FULL_FP32 결과를 곧바로
low-precision deployable weight guarantee로 부르지 않는다.

\(A_0^\top C^{-1}A_0\)의 numerical range tolerance, rank rule과 solve backend는 `Q_cal`만 사용해
outcome 전에 고정한다. Gate 결과를 본 뒤 ridge 또는 rank threshold를 변경하지 않는다.

예상 screen 비용이 model당 `8 A6000-equivalent GPU-hours`를 넘으면 fresh paired job을 열지
않고 anchor batching과 teacher-logit cache를 먼저 줄인다. 두 model task는 resource가
허용되면 같은 submission batch에서 병렬 실행한다.

---

# 7. F1 — same-z functional multiplicity screen

## Candidate set

각 fresh edit에 대해 다음 8개를 candidate receipt에 먼저 기록한다.

1. random tangent axis 1의 \(+/-\)
2. random tangent axis 2의 \(+/-\)
3. random tangent axis 3의 \(+/-\)
4. controller functional tangent의 safe/adverse pair

모든 candidate는 동일한 \(K_E\), \(K_H\), target residual과 C-action contract를 통과해야
한다. Candidate sign, seed, factor hash와 metric은 `Q_gate` open 전에 immutable receipt에 쓴다.
Random axis seed, projection과 matching rule은 `Q_cal`만 사용해 고정하며 gate score를 본 뒤
candidate를 재생성하거나 invalid subset을 제거하지 않는다.

`same-z` validity는 세 층으로 분리해 기록한다.

1. algebraic base equality \(\Delta_C A_0=B_0\)와 tangent equality \(NA=0\)
2. FULL_FP32 virtual/commit 뒤 full-model rewrite-position activation realization
3. preregistered target event와 target-token logit/margin realization

1만 통과하고 2 또는 3이 실패한 candidate를 same-z denominator에 넣지 않는다. Failure 자체는
typed count로 남기며 outcome-dependent replacement candidate를 만들지 않는다.

## Primary multiplicity statistic

Edit \(i\)의 gate-bank risk spread를

\[
S_i
=
Q_{0.9}\{D_{gate}(\Delta_{i,c})\}_c
-
Q_{0.1}\{D_{gate}(\Delta_{i,c})\}_c
\]

로 둔다. Duplicate candidate와 no-op repeat로 얻은 numerical spread를 \(S_i^{noise}\)라 둔다.

## F1 screen Go

다음을 모두 만족해야 한다.

1. 모델별 최소 7/8 case에서 8개 candidate equality/action contract가 유효하다.
2. 모델별 최소 6/8 case에서 \(S_i>3S_i^{noise}\)다.
3. 모델별 median \(S_i>5\,median\,S_i^{noise}\)다.
4. candidate target residual 또는 target logit variation은 gate-bank spread의 원인으로 남지 않는다.
5. functional spread가 mean KL 하나가 아니라 gate mean과 tail 중 최소 하나에서 재현된다.

이는 canonical Gate 1 통과가 아니라 **다음 구현 준비를 정당화하는 engineering premise
screen**이다.

## F1 No-go

- exact equality candidate를 안정적으로 만들 수 없음
- matched-action functional spread가 numerical floor 수준
- spread가 target mismatch 또는 action mismatch로 설명됨

이 경우 fixed-z safe-realization 연구를 중단한다. ODE, barrier와 sequential 실험은 열지
않는다.

---

# 8. F2 — key-space 밖 functional signal screen

F1과 같은 run artifact를 사용하되 F1 판정을 먼저 freeze한 뒤 F2를 연다.

## Primary contrast

Controller derivative가 정한 safe candidate와 adverse candidate의 gate-bank 차이를

\[
A_i
=
D_{gate}(\Delta_i^{adverse})
-D_{gate}(\Delta_i^{safe})
\]

로 둔다. \(A_i>0\)이면 controller functional sign이 unseen gate bank에 일반화된 것이다.

두 candidate는 다음이 동일하다.

- fixed-z local target
- \(K_E\) response
- registered \(K_H\) response
- C-action
- tangent amplitude

따라서 이 contrast는 finite registered key geometry를 넘어서는 functional information의
가장 작은 직접 검정이다.

## F2 screen Go

1. \(A_i>0\)인 case가 모델별 최소 6/8이다.
2. 모델별 mean과 median \(A_i\)가 모두 양수다.
3. 모델별 median \(|A_i|\)가 duplicate-candidate noise의 3배보다 크다.
4. safe candidate가 preregistered target-success event를 만족한 case가 모델별 최소 7/8이다.
5. controller 개선만 있고 gate bank가 악화하는 pattern이 아니다.

## F2 No-go

- safe/adverse sign이 gate bank에서 chance 수준
- history-key null constraint를 추가하면 효과가 사라짐
- gate-bank 개선이 target success 붕괴와 교환됨
- controller bank에만 맞는 overfit

F2가 실패하면 functional barrier는 비싼 key-risk proxy로 판정한다. Static FCW와 ODE를 모두
중단한다.

## 조건부 replication

두 모델의 paired 8-case F1/F2 screen이 모두 Go이면 다음을 동시에 연다.

- canonical 128-case/model, 세 tangent seed, request별 16--32 direction manifest 준비
- F3 static prototype 구현

Implementation prototype을 빠르게 시작할 수는 있지만 scientific F3 job은 canonical Gate 1과
Gate 2가 통과하기 전에 열지 않는다. 16-case 결과는 `screen-supported hypothesis`를 넘지
않으며, 상위 proposal의 128-request/3-seed Gate와 strong key/history baseline 비교를
대체하지 않는다. Final `Q_audit`은 이 단계에서 계속 sealed 상태다.

---

# 9. F3 — closed form이 유도하는 상대적 barrier

절대 KL threshold를 임의로 선택하지 않는다. 먼저 closed-form reference path 자체가 matched
semantic progress에서 허용하는 functional envelope를 사용한다.

## 9.1 Reference path

\[
\theta_{ref}(s)
=
\theta_0+s\Delta_C,
\qquad s\in[0,1].
\]

Single-layer testbed에서는 input key가 해당 weight에 의해 바뀌지 않으므로

\[
(s\Delta_C)K_E=sR_E
\]

가 exact semantic-progress path다.

## 9.2 Baseline-relative functional barrier

Controller group \(g\)의 functional risk를 \(D_g\), calibration-only numerical allowance를
\(\nu_g\)라 하면

\[
b_g(s)
=
D_g(\theta_{ref}(s))+\nu_g,
\]

\[
h_g(\theta,s)
=
b_g(s)-D_g(\theta).
\]

Action tube도 reference에서 유도한다.

\[
h_C(\theta,s)
=
(1+\rho_{\mathrm{tan}})s^2\|\Delta_C\|_C^2
-\|\theta-\theta_0\|_C^2.
\]

이 설계에서 다음은 임의 선택이 아니다.

- semantic progress: fixed direct-\(z\)와 closed-form equality가 결정
- functional envelope: 같은 progress의 closed-form reference가 결정
- numerical allowance: \(Q_{cal}\) no-op/duplicate floor가 결정
- action radius: F1에서 lock한 matched tangent budget이 결정

단, anchor group, covariance metric과 \(\rho_{\mathrm{tan}}\)까지 우주적으로 유일하게 유도된다고 주장하지
않는다. 이 barrier의 정확한 이름은 **baseline-relative non-worsening functional
contract**이며 population-wide absolute safety 보장이 아니다.

Teacher KL은 W0에서 1차 gradient가 0이고 finite step은 continuous boundary를 넘을 수 있다.
따라서 F3의 실제 authority는 다음 순서로 고정한다.

1. quadratic/Fisher local model로 candidate 생성
2. active controller prompt의 actual finite value로 accept/reject
3. terminal 전에 controller bank 전체 actual value 검사

1차 CBF inequality만으로 discrete safety를 주장하지 않는다.

## 9.3 왜 closed form만으로 끝나지 않는가

Closed form은 \(\Delta_C\) 하나를 제공하지만, \(NA=0\)인 tangent 방향의 실제 output behavior를
사용하지 않는다. F2가 통과하면 functional gradient는 동일 equality/action set 안에서 더
안전한 candidate를 선택하는 추가 정보를 제공한다.

Static endpoint에서는 scalar trade-off 대신 다음 lexicographic order를 사용한다.

1. target/history equality 만족
2. action tube 만족
3. controller group의 worst normalized functional risk 최소화
4. 동률에서 \(\Delta_C\)로부터의 C-distance 최소화

F3의 첫 method 명칭은 **Static Functional-Constrained Write**다. 아직 ODE라고 부르지 않는다.

---

# 10. F3/F4 최소 비교 arm

F1/F2와 분리한 fresh edit/gate block에서 다음만 실행한다.

| Arm | Functional field | Steps | 역할 |
|---|---|---:|---|
| `CF-REF` | 없음 | 1 | C-min closed-form reference |
| `STATIC-Q` | entry에서 1회 | 1 | direct static constrained solve |
| `SPLIT-FROZEN` | entry field 고정 | 4 | Euler subdivision negative control |
| `REFRESH-BARRIER` | accepted state마다 갱신 | 4 | feedback + relative barrier |
| `REFRESH-NOBARRIER` | 갱신 | 4 | barrier mechanism ablation |
| `STATIC-ORACLE` | multi-start/SQP | compute matched | ODE에 가장 강한 static comparator |

모든 arm은 같은 direct-\(z\), equality set, action cap, controller/gate split을 사용한다.

`Q_gate`를 열기 전에 technical/calibration profile만으로 arm별 최대 wall-clock,
gradient/forward/VJP와 candidate budget을 고정한다. `STATIC-Q`와 `STATIC-ORACLE`은 이
prelocked compute cap 안에서 실제 optimization iteration과 multi-start를 사용할 수 있다.
Dummy forward로 비용만 태우지 않으며, 결과를 본 뒤 static search budget을 추가하지 않는다.

## F3 Go

- `STATIC-Q`가 `CF-REF`보다 gate-bank efficacy--functional frontier를 개선
- improvement가 target/action mismatch로 설명되지 않음
- controller와 gate 방향이 일치

F3가 실패하면 functional safe-write method를 kill한다.

## F4 ODE Go

다음을 모두 만족할 때만 method를 `CBF-ODE-Write` 후보로 승격한다.

1. `REFRESH-BARRIER`가 best compute-matched static arm에 지배되지 않는다.
2. gate-bank frontier에서 static보다 재현 가능한 이득이 있다.
3. functional Jacobian 또는 active group이 accepted state에서 실제로 변한다.
4. `SPLIT-FROZEN`과 endpoint가 다르다.
5. barrier가 적어도 일부 case에서 candidate 또는 step을 바꾼다.
6. step refinement에서 endpoint/metric이 안정화된다.

Step refinement는 primary four-step 결과를 본 뒤 선택하지 않는다. 모든 valid technical case의
고정 subset에서 \(K\in\{1,2,4,8\}\)을 preregistered diagnostic으로 수행한다.

## F4 ODE No-go

- `STATIC-Q` 또는 `STATIC-ORACLE`이 같거나 더 좋음
- frozen split과 refreshed path가 같음
- field/active set이 사실상 stationary
- 이득이 추가 compute로만 설명됨
- barrier가 한 번도 작동하지 않음

이 경우 최종 방향은 static FCW다. ODE를 살리기 위한 threshold 변경이나 model-specific rescue를
금지한다.

## Final one-shot audit

F4 판정, architecture, arm, threshold, step rule과 모든 hyperparameter를 고정한 뒤에만
`Q_audit`을 정확히 한 번 연다. Final audit에는 선택된 method와 preregistered comparator만
평가한다. Audit 결과가 gate-bank 개선을 재현하지 못해도 같은 study에서 anchor, threshold,
case, subset 또는 method를 고치지 않는다. 후속 변경은 새 audit bank를 가진 별도 revision이다.

---

# 11. 빠른 실행 진행표

| Stage | Model/case | 최대 확장 조건 | 연구 판정 |
|---|---|---|---|
| F0 | CPU synthetic | mandatory identity pass | GPU open 여부 |
| F1-T | Llama/Qwen, old 1 case/model | projected cost `<=8 GPU-h/model` | technical only |
| F1/F2-S | Llama/Qwen, same fresh 8/model | 두 모델 F1/F2 screen Go | paired premise screen |
| F1/F2-C | 128/model, 3 seeds, 16--32 directions | canonical proposal 조건 | confirmatory Gate |
| F3 | fresh 16/model | static frontier gain | method premise |
| F4 | fresh 16--64/model | static 대비 gain | ODE/pivot 판정 |
| Final audit | frozen method, new bank | one-shot only | final estimate |

F1/F2-S가 끝나기 전에 AlphaEdit, multi-layer ODE, 100-edit, sequential 또는 downstream full
suite를 열지 않는다.

---

# 12. Compute와 receipt

## 필수 counter

- direct-\(z\) optimization 수와 시간
- covariance solve 수와 시간
- teacher-logit cache forward
- controller forward/backward
- candidate actual forward
- gate-validation forward
- final audit forward
- batched VJP 또는 anchor/group backward 수
- factor rank와 temporary peak elements
- wall-clock, GPU seconds와 peak memory

## 필수 identity

- source/model/tokenizer/revision hash
- edit/history/cal/controller/gate/audit manifest hash
- direct-\(z\) tensor hash
- \(K_E,K_H,K_T,R_E,C\) identity
- candidate factor/sign/seed/action hash
- candidate receipt 시각이 gate-open보다 앞서는지
- 모든 Gate/method freeze 시각이 final audit-open보다 앞서는지
- target/history equality residual
- FULL_FP32 virtual/commit identity
- 별도 low-precision study의 FP32 대비 endpoint discrepancy
- model parameter와 RNG no-mutation identity

Raw key, prompt, hidden state, gradient, logits와 direct-\(z\) tensor는 ignored `local/` 아래에 둔다.
Git에는 scalar receipt, manifest hash, compact factual report만 기록한다.

---

# 13. 재사용 자산과 최소 구현 범위

| 자산 | 재사용 |
|---|---|
| [`direct_z_possibility.py`](../../project/run_scripts/ode_edit_motivation/direct_z_possibility.py) | frozen direct-\(z\), CounterFact selection, teacher evaluator 골격 |
| [`easyedit_bridge.py`](../../project/run_scripts/ode_edit_motivation/easyedit_bridge.py) | native keys/residual/covariance lineage |
| [`diagnostic_math.py`](../../project/run_scripts/ode_edit_motivation/diagnostic_math.py) | C-inner product, solve와 low-rank identity |
| [`derivatives.py`](../../project/run_scripts/ode_edit_method/derivatives.py) | activation/gradient low-rank hook 골격 |
| [`functional_trial.py`](../../project/run_scripts/ode_edit_method/functional_trial.py) | no-mutation FULL_FP32 virtual trial; low precision은 별도 study |
| [`hooks.py`](../../project/run_scripts/ode_edit_method/hooks.py) | factor identity와 transaction primitives |

최소 새 구현은 다음뿐이다.

1. C-minimum equality write와 null-tangent builder
2. batch-preserving next-token KL gradient factor
3. edit/history/cal/controller/gate/audit manifest firewall
4. eight-candidate virtual evaluator
5. F1/F2 deterministic analyzer

Canonical F1/F2 Go 전에는 general CBF-QP, multi-layer runtime 또는 sequential ledger를
구현하지 않는다.

---

# 14. Remote main evidence의 취급

`edf2ea2c98c5`까지의 remote main을 확인했다. 다음 결과는 새 Gate의 통과 근거로 자동
승계하지 않지만, 반복하면 안 되는 negative control과 설계 경계를 제공한다.

## 14.1 BGODE fixed-basis barrier

`experiment-reports/servers/server1/2026-08-28-bgode-fbp-minimal-fixed-basis-barrier-ode-v1/`
에서 다음이 이미 관측됐다.

- 2 models × 8 cases에서 static N4 split과 Native AlphaEdit endpoint/logit identity가 일치했다.
- q-KL half-space barrier는 event/locality drift를 줄였다.
- 동시에 target strength가 크게 약해졌고 Llama rewrite exact는 N2/N4 모두 0/8이었다.
- node 0 q-KL gradient는 0이었고 barrier는 이후 node에서 활성화됐다.

따라서 본 plan은 단순 split parity나 one-sided q-KL projection을 새 contribution으로 반복하지
않는다. Exact target/history equality와 matched action을 먼저 강제하는 이유가 바로 이
strength--preservation confounding을 제거하기 위해서다.

## 14.2 Cache-aware projected AlphaEdit B100

`experiment-reports/servers/server4/cache-aware-qkl-projected-alphaedit-b100-2026-08-29-v2/`
에서는 N2가 두 모델의 locality를 비열세 또는 소폭 개선했지만 Llama/Qwen rephrase prompt
success가 Official AlphaEdit보다 각각 낮았다. N4는 더 큰 preservation 이동과 더 큰 strength
loss를 함께 보였다.

이 결과도 same-z/history-null matched-action contrast가 아니며, controller/gate/final-audit
분리로 functional signal의 독립 가치를 검정하지 않았다. F1/F2를 대신하지 않지만, efficacy
equality 없이 barrier만 강화하는 arm을 재실행하지 않게 한다.

## 14.3 BGODE R1--R3 numerical boundary

- R1 Llama는 exact progress localization 부재로 matched-progress barrier attribution이
  열리지 않았다.
- R1 Qwen과 R2 일부 topology는 Fisher range/equality numerical boundary에 닿았다.
- R2/R3는 FP64 event/equality 수학과 natural topology를 강화했지만 scientific promotion은
  없고, R3 G2는 all-prefix JVP/FD numerical boundary에서 terminal 0/2였다.

따라서 본 plan은 singular/range failure를 ridge나 fallback으로 숨기지 않고 typed failure로
남기며, main Gate에서는 FULL_FP32 identity와 outcome-independent rank rule을 요구한다.

## 14.4 Direct-z possibility 자산

기존 direct-z possibility panel은 다음 기술 자산을 제공한다.

- frozen direct-\(z\) lineage
- two-model evaluator와 gate/final-audit open 순서
- low-rank virtual write와 rollback
- state refresh만으로 preservation이 자동 개선되지 않았다는 negative evidence

하지만 기존 panel은 다음 이유로 F1/F2 scientific denominator에 넣지 않는다.

- exact same-activation tangent candidate 비교가 아님
- functional signal로 후보를 선택하지 않음
- key/history response와 C-action을 동시에 동일화하지 않음
- ordered native 대 refreshed controller의 복합 대비임

따라서 재사용은 source와 technical smoke에 한정한다.

---

# 15. 즉시 handoff할 첫 작업

방향 확정 뒤 별도 실행 session에 넘길 첫 작업은 **F0 + F1-T 구현과 dry-run**이다.

Handoff 범위는 다음으로 제한한다.

- 새 probe module과 CPU tests
- 기존 2 case technical manifest
- outcome-free command renderer
- 예상 시간/메모리 profiler
- GPU scientific outcome 0건

F0 identity와 F1-T technical receipt를 이 연구 방향 session이 검토한 뒤에만 fresh `[148:156]`
F1/F2-S 실행 여부를 결정한다.

---

# 16. 현재 결정 요약

| 항목 | 결정 |
|---|---|
| 가장 먼저 할 실험 | single-layer same-z/history-null functional multiplicity |
| 첫 model | Llama/Qwen paired |
| 첫 scientific case 수 | 8/model; 16 model-case |
| direct-\(z\) | W0에서 1회 계산 후 고정 |
| key protection | edit key + finite history key exact equality |
| action matching | C-min reference + C-orthogonal tangent, \(\rho_{\mathrm{tan}}=0.05\) |
| functional metric | controller/gate/final-audit 분리 next-token teacher KL |
| Gate 0--2 dtype | FULL_FP32 authority; low precision은 별도 study |
| barrier | closed-form matched-progress reference가 유도하는 relative envelope |
| static solver | ODE보다 먼저 검증 |
| ODE 이름 | F4 통과 전 사용 금지 |
| Alpha/multi-layer/sequential | F1--F4 뒤로 보류 |

이 fast plan의 목적은 긍정 결과를 빨리 만드는 것이 아니라, 가장 비싼 잘못된 방향을 가장
빨리 제거하는 것이다.
