# Session 01 — Motivation Validation: 최종 연구 판정

- 최종 갱신: **2026-08-03 12:04 KST**
- 상태: **`CLOSED_DIRECTIONAL_POSITIVE; STRONG_METHOD_GATE_FAIL`**
- terminal experiment: C3 BF-share magnitude-only cause isolation
- causal verdict: **`LOW_UPDATE_IMPLEMENTATION_CAUSE_CONFIRMED_CROSS_MODEL`**
- 현재 방법명: `ODE-Edit`
- 다음 section: [`04-method-design.md`](04-method-design.md)

## 1. 최종 supersession

2026-08-02 C1의 `CAPACITY_HISTORY_HARM_SIGNAL`과 C2의 bundled share/magnitude 결과는
C3에 의해 최종 supersede됐다.

C3는 C1과 같은 BF allocation algorithm을 사용하되 raw QP coefficient가 global update
magnitude까지 결정하지 않도록 분리했다. 실제 write는 매 round C1-compatible relative
share를 만들고 branch-local ordered-native C-distance의 정확한 `D/4`를 적용했다.

그 결과 C3 current efficacy는 Llama/Qwen×MEMIT/Alpha 네 cell 모두 C1보다 회복했고,
16개 edit 중 15개에서 C1보다 좋아졌다.

| Model | Family | C1 current | C3 current | C3−C1 |
|---|---|---:|---:|---:|
| Llama | MEMIT | `-4.032392` | `-0.361365` | `+3.671027` |
| Llama | Alpha-history | `-2.495123` | `-0.563717` | `+1.931406` |
| Qwen | MEMIT | `-0.397837` | `-0.251563` | `+0.146274` |
| Qwen | Alpha-history | `+0.029707` | `+0.555242` | `+0.525536` |

따라서 C1 harm의 주요 원인은 BF relative routing 자체가 아니라 relative share와 actual
global magnitude를 동일 raw coefficient에 결합한 under-write implementation confound로
판정한다. 이는 동일 4-case, one-order, one-seed panel의 mechanistic RCA이며 통계적
population claim이 아니다.

## 2. Motivation에서 확인된 것

### 2.1 Survive

- 같은 current snapshot에서 layer actuator를 비교하는 BF 구조
- partial joint write 뒤 current residual/key/proposal을 다시 계산하는 relinearization
- layer별 rewrite utility와 capacity heterogeneity
- cumulative capacity와 layer-load concentration을 routing signal로 사용할 가능성
- BF relative share와 global update magnitude를 독립적으로 결정해야 한다는 원칙
- MEMIT과 projected/history AlphaEdit에 공통 controller를 둘 가능성

### 2.2 Kill

- fixed initial update를 \(K\)등분하는 waypoint-only ODE narrative
- central-probe G selector
- unconditional always-refresh full-distance sequential execution
- C1 raw absolute coefficient writer
- C2 changed-share 결과를 BF share의 최종 verdict로 사용하는 해석
- model-specific K/threshold/sign/policy rescue
- C3를 native superiority 또는 lifelong evidence로 사용하는 해석

## 3. Fixed direct-z와 state-dependent flow의 구분

Edit entry state에서 direct-z target은 한 번 계산해 고정한다.

\[
Z_t^\star=\operatorname{DirectZ}(W_{t,0},r_t).
\]

그러나 waypoint \(s\)의 current residual, key, proposal과 layer velocity는 다시 계산한다.

\[
E_{t,s}=Z_t^\star-H^L(W_{t,s}),
\]

\[
K_{t,s,l}=K_l(W_{t,s}),
\qquad
B_{t,s,l}=B_l(W_{t,s},E_{t,s},K_{t,s,l}),
\]

\[
v_{t,s}=v(W_{t,s},B_{t,s},D\Phi_{\rm rw},\Psi_{t,s}).
\]

Fixed direct-z는 fixed parameter endpoint를 의미하지 않는다. \(Z_t^\star\)는 trajectory의
외생 target이고 state는 \(W(t)\)다. 동일 semantic target을 만족하는 여러 parameter
realization 중 낮은 cumulative cost를 갖는 endpoint를 찾는 것이 Method의 목적이다.

초기 MEMIT delta를 고정한 채

\[
W^{(k+1)}=W^{(k)}+\frac1K\Delta W_{\rm MEMIT}(W^{(0)})
\]

를 반복하면 native와 같은 endpoint이므로 의미 있는 ODE가 아니다. C3는 매 hop current
state에서 residual/key/proposal/slope/capacity/share를 다시 만드는 구조라 이 negative
control과 다르다. 다만 dynamic이 static보다 실제로 낫다는 matched ablation은 아직 하지
않았다.

## 4. Motivation evidence chain

| 질문 | 관측 | 판정 |
|---|---|---|
| EasyEdit hook이 native behavior를 보존하는가 | exact tensor/logit/rollback fidelity | pass |
| 작은 step 자체가 개선을 만드는가 | fixed delta split은 native endpoint | waypoint-only claim kill |
| same-snapshot allocation signal이 있는가 | 두 모델 local signal | survive |
| partial write 뒤 direction refresh signal이 있는가 | atomic MEMIT/Alpha local positive | survive |
| 간단 adaptive G selector가 model-common한가 | selection/endpoint 불안정 | kill |
| unconditional 4-edit refresh가 보존에 좋은가 | capacity/locality proxy 양성, efficacy/retention 악화 | safeguard 필요 |
| capacity/history QP가 바로 method가 되는가 | C1 severe under-write | implementation diagnosis |
| exact distance가 C1을 회복시키는가 | C2 회복, share/cap change로 cause 혼합 | 불충분 |
| C1 share algorithm과 magnitude를 분리하면 회복하는가 | C3 4/4 cell, 15/16 edit 회복 | final RCA pass |

## 5. C3가 남긴 양성 신호와 강한 음성 경계

### 양성

- 네 cell 모두 C1 current efficacy보다 회복
- 네 cell 모두 C2보다 회복
- capacity reduction, max-layer-share reduction, layer-Gini reduction 네 cell 공통
- Qwen Alpha current `+0.555242`, prior retention `-0.014337`의 local point

### 음성 및 미해결

- strict native current floor `-0.10`은 Qwen Alpha만 통과
- pair passing family `[]`
- final prior retention과 retention AUC는 네 cell 모두 native보다 낮음
- neighborhood/generation KL 방향 model-common 아님
- applied coefficient cap exceed `140/320`
- radial scale 평균 `2.106×`, 최대 `35.927×`
- negative trust step `1/64`
- compute native 대비 `4.80--6.93×`
- direct-z hidden fidelity 미측정

따라서 capacity/Gini 감소는 장기 routing 가능성의 proxy이지 preservation guarantee가
아니다.

## 6. C3가 proposal을 구현한 범위

C3는 다음 proposal skeleton을 구현했다.

- direct-z once/edit
- same-snapshot layer proposal
- current residual/key/proposal refresh
- unit-C actuator
- cumulative \(W^0\)-relative capacity
- non-negative BF relative allocation
- MEMIT/Alpha 공통 controller

다음 deployable Method 요소는 kill-test 범위에서 의도적으로 구현하지 않았다.

- actual applied coefficient를 직접 푸는 constrained QP
- applied full-step hard capacity barrier
- semantic smooth-max Φ와 first-hitting terminal
- trust reject, exact rollback, adaptive \(h\)
- adaptive \(K\)와 numerical refinement
- dynamic-vs-static ODE necessity ablation
- 100–1K+ sequential/downstream validation

C3는 proposal의 motivation mechanism을 구현한 cause-isolation control이지 완성된
BF-ODE Method가 아니다.

## 7. Motivation 최종 claim boundary

### 현재 가능한 claim

1. BF layer allocation과 global update magnitude는 분리해야 한다.
2. 이 분리가 동일 4-case panel에서 C1 under-write와 efficacy collapse를 두 모델·두
   family에 걸쳐 회복했다.
3. Branch-local native C-path budget을 거의 맞춘 상태에서도 ex-post capacity와
   concentration 감소 신호가 남았다.
4. Fixed direct-z 아래의 current-state BF relinearization을 Method로 검정할 방향성이 있다.

### 현재 불가능한 claim

1. ODE가 static BF보다 필수다.
2. BF가 DF/uniform/static routing보다 우월하다.
3. Direct-z hidden state를 baseline보다 더 충실히 write한다.
4. Applied update가 minimum-capacity solution 또는 hard-safe다.
5. Native보다 retention/downstream이 좋다.
6. Large/lifelong sequential collapse를 방지한다.
7. Alpha-BF가 일반적으로 MEMIT보다 좋다.

## 8. Motivation closure decision

Session 01에서는 추가 K/share/threshold/model-specific Motivation rescue를 하지 않는다.
C1/C2/C3를 더 retune해 native strong gate를 만들지 않는다. Motivation은 다음 상태로
확정한다.

> **`CLOSED_DIRECTIONAL_POSITIVE`** — state refresh + BF relative share + independent
> global step을 완성 Method로 검정할 이유는 확인됐다.

Strong method gate가 실패한 사실은 그대로 유지한다.

> **`STRONG_METHOD_GATE_FAIL`** — native superiority, broad preservation, applied hard
> barrier, ODE necessity, lifelong robustness와 compute efficiency는 미확립이다.

## 9. Method handoff

> **2026-08-03 compute-aware amendment:** 이후 method-direction/compute review에서 signed
> base-relative capacity 감소가 past-edit retention을 뜻하지 않는다는 C3 경계를
> 반영했다. 첫 MEMIT 10/100-edit track은 completed outer-edit terminal net write로
> 정의한 monotone load (Omega)를 soft routing cost로 사용하고 hard barrier는 후속
> ablation으로 미룬다. Canonical 실행 방향은
> [`Session 02 compute-aware main-table spec`](../../../plans/global/2026-08-03-session02-compute-aware-main-table-spec.md)을
> 따른다.

다음 research section은 [`04-method-design.md`](04-method-design.md)다. Method는 다음을
동일 model-common controller로 결합해야 한다.

1. Actual applied coefficient를 직접 푸는 progress/capacity QP
2. Applied hard barrier
3. Semantic first-hitting terminal
4. Trust reject/rollback과 adaptive \(h\)
5. Emergent \(K\)
6. Current residual/key/direction/efficiency/capacity/velocity non-stationarity panel
7. Best static/frozen controller와 matched ODE necessity ablation

10-edit common strong pilot을 통과하기 전 100+, 1K+, lifelong execution은 열지 않는다.

## 10. Canonical evidence

- 최종 closure와 Method handoff:
  [`../../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md`](../../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)
- C3 pair synthesis:
  [`../../../experiment-reports/global/2026-08-02-session01-caphist-pair-c3-v1-synthesis.md`](../../../experiment-reports/global/2026-08-02-session01-caphist-pair-c3-v1-synthesis.md)
- C3 preregistered spec:
  [`../../../plans/global/2026-08-02-session01-bf-share-magnitude-control-c3-spec.md`](../../../plans/global/2026-08-02-session01-bf-share-magnitude-control-c3-spec.md)
- C3 post-run audit:
  [`../../../audits/global/2026-08-02-session01-bf-share-magnitude-control-c3-postrun.md`](../../../audits/global/2026-08-02-session01-bf-share-magnitude-control-c3-postrun.md)
- Motivation historical closure index:
  [`../../../experiment-reports/global/2026-08-02-session01-motivation-closure-gh.md`](../../../experiment-reports/global/2026-08-02-session01-motivation-closure-gh.md)
- Related-work/novelty boundary:
  [`02-related-work-and-novelty-boundary.md`](02-related-work-and-novelty-boundary.md)
- Direct-z atomic/de-bundling design history:
  [`03-direct-z-review-and-motivation-closure-design.md`](03-direct-z-review-and-motivation-closure-design.md)

과거 MV-2/C1 closed-negative 문서는 당시 decision과 preregistration을 보존하는 historical
evidence다. 현재 Session 01 판정은 이 section과 2026-08-03 final closure를 따른다.
