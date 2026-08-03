# Session 01 Motivation 최종 종결 및 Method handoff

- 최종 갱신: **2026-08-03 12:04 KST**
- 방법명: **ODE-Edit**
- 대상 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- actuator family: MEMIT, canonical projector/history AlphaEdit
- 최종 Motivation 판정: **`CLOSED_DIRECTIONAL_POSITIVE`**
- Method 판정: **`OPEN; STRONG_NATIVE_GATE_NOT_YET_PASSED`**
- C3 technical 판정: **pass**
- C3 four-case implementation-RCA 판정: **pass**
- strict native method 판정: **fail; passing family `[]`**
- claim boundary: 4-case, one-order, one-seed mechanism/implementation diagnostic

## 0. Executive decision

Session 01 Motivation은 여기서 종료한다.

C3는 C1의 capacity-aware BF allocation algorithm을 유지하면서 layer share와 실제
global update magnitude를 분리했다. 이 intervention으로 Llama/Qwen과 MEMIT/Alpha
네 model×family cell 모두에서 C1 current efficacy가 회복됐고, 16개 edit 중 15개가
C1보다 좋아졌다. 동시에 cumulative capacity, max-layer concentration, layer Gini의
감소 방향이 네 cell에 공통으로 남았다.

따라서 다음은 확인됐다.

1. C1의 심한 악화는 BF 방향 자체를 kill하는 증거가 아니라, raw QP coefficient를
   relative layer share와 actual update magnitude에 동시에 사용한 under-write가 주요
   원인이었던 동일 4-case panel의 implementation-RCA다.
2. fixed direct-z target 아래에서 current residual, key, local low-rank proposal,
   rewrite efficiency, cumulative capacity, layer velocity를 매 waypoint 재계산하는
   state-refreshed BF trajectory는 Method Session으로 넘길 충분한 directional signal이
   있다.
3. 그러나 C3는 adaptive step, applied hard barrier, trust reject/rollback,
   first-hitting terminal을 구현하지 않았고 native 대비 retention도 네 cell 모두
   음수다. 따라서 deployable method, preservation superiority, lifelong superiority는
   아직 열 수 없다.

이 결론은 다음처럼 한 문장으로 요약한다.

> **Motivation은 “state-relinearized BF path를 더 개발할 이유가 있는가”라는 질문에는
> positive로 닫혔다. 다음 단계는 추가 Motivation rescue가 아니라, fixed-`D/4 × 4`
> heuristic을 제거하고 actual applied velocity를 직접 제약하는 adaptive BF-ODE
> Method를 설계·검증하는 것이다.**

## 1. 이 문서의 지위와 evidence source

이 문서는 새 실험 결과를 만드는 보고서가 아니다. Session 01 전체 논의와 C3
terminal evidence를 하나의 최종 연구 판정으로 통합하고 다음 Method section을 여는
canonical decision report다.

수치와 실행 무결성의 source-of-truth는 다음 순서를 따른다.

1. tracked scientific synthesis:
   [`2026-08-02-session01-caphist-pair-c3-v1-synthesis.md`](2026-08-02-session01-caphist-pair-c3-v1-synthesis.md)
2. post-run integrity:
   [`../../audits/global/2026-08-02-session01-bf-share-magnitude-control-c3-postrun.md`](../../audits/global/2026-08-02-session01-bf-share-magnitude-control-c3-postrun.md)
3. preregistered causal intervention:
   [`../../plans/global/2026-08-02-session01-bf-share-magnitude-control-c3-spec.md`](../../plans/global/2026-08-02-session01-bf-share-magnitude-control-c3-spec.md)
4. ignored local compact source:
   `local/results/raw/session01_motivation/caphist_combined_{llama,qwen}_c3_v1/analysis.json`,
   `local/results/raw/session01_motivation/caphist_pair_c3_v1/analysis.json`

이 문서의 proposal-side handoff는
[`../../project/proposals/sections/04-method-design.md`](../../project/proposals/sections/04-method-design.md)다.

## 2. 최종 연구 명제

### 2.1 Baseline을 정확히 기술한 출발점

MEMIT을 모든 layer update를 처음부터 고정하는 완전한 open-loop editor라고 부르면
정확하지 않다. MEMIT은 edit당 direct-z target을 한 번 계산하고 layer를 정해진
순서로 방문하며, 앞 layer의 full write 뒤 뒤 layer의 key와 remaining residual을
다시 측정한다. 따라서 one-pass ordered Gauss–Seidel에 가깝다.

문제는 각 layer의 local prescription을 전량 commit한 뒤 해당 layer를 새로운 joint
state에서 다시 검토하지 않는다는 점이다. 유한한 write가 여러 layer에 들어가면
다음이 달라진다.

- current direct-z residual
- layer key와 downstream activation
- layer별 low-rank actuator direction
- semantic rewrite progress에 대한 layer별 efficiency
- 이미 누적된 layer capacity와 barrier slack

그 결과 하나의 고정 layer order와 full-write sweep은 가능한 여러 parameter
realization 중 특정 order-dependent endpoint를 선택하고, 장기 stream에서는 write
load를 일부 layer에 집중시킬 수 있다.

### 2.2 ODE-Edit이 주장하려는 차이

Edit \(i\)의 direct-z target은 edit entry state에서 한 번 계산해 고정한다.

\[
Z_i^\star = \operatorname{DirectZ}(W_i^{(0)}, r_i).
\]

그러나 이 target을 실현하는 vector field는 현재 parameter state에 의존한다.

\[
E_i^{(k)} = Z_i^\star-H^L(W_i^{(k)}),
\]

\[
K_{i,l}^{(k)}=K_l(W_i^{(k)}),
\qquad
B_{i,l}^{(k)}=B_l(W_i^{(k)},E_i^{(k)},K_{i,l}^{(k)}),
\]

\[
v_i^{(k)}
=v\!\left(W_i^{(k)},B_i^{(k)},D\Phi_{\rm rw},\Psi_i^{(k)}\right).
\]

ODE-Edit의 core update는 다음이어야 한다.

\[
W_{i,l}^{(k+1)}
=W_{i,l}^{(k)}
+h_i^{(k)}v_{i,l}^{(k)}\widehat B_{i,l}^{(k)}.
\]

그리고 모든 예정 step을 적용하는 것이 아니라 rewrite-authorized semantic goal을 처음
만족하는 시점에서 종료해야 한다.

\[
T_i^\star
=\inf\{t:\Phi_{\rm rw}(W_i(t);r_i)\le0\}.
\]

고정 \(Z_i^\star\)는 고정 parameter endpoint를 의미하지 않는다. \(Z_i^\star\)는
ODE의 state가 아니라 edit-conditioned target이고, 실제 state는 \(W(t)\)다. 동일
semantic target을 만족하는 parameter 집합

\[
\mathcal M_i
=\{W:\Phi_{\rm rw}(W;r_i)\le0\}
\]

에는 일반적으로 여러 endpoint가 존재한다. ODE-Edit은 그중 낮은 누적 capacity와
좋은 retention frontier를 갖는 endpoint를 trajectory control로 찾는 방법이어야 한다.

## 3. 의미 없는 split과 의미 있는 state-dependent trajectory

초기 MEMIT update를 고정하고 단순히 나누면

\[
W^{(k+1)}=W^{(k)}+\frac1K\Delta W_{\rm MEMIT}(W^{(0)}),
\]

\[
W^{(K)}=W^{(0)}+\Delta W_{\rm MEMIT}(W^{(0)}).
\]

최종 weight는 native와 같다. Waypoint 수만 늘었을 뿐 ODE mechanism이 아니다.

C3는 이 negative control과 다르다. C3의 실제 path는

\[
W^{(k+1)}
=W^{(k)}
+\frac D4
\sum_l
\frac{x_l^\star(W^{(k)})}{\|x^\star(W^{(k)})\|_2}
\widehat B_l(W^{(k)},Z^\star)
\]

이다. 매 hop의 source snapshot과 proposal action hash는 current descendant state에
bind되며, 앞 hop을 permanent apply한 뒤 다음 residual/key/proposal/panel/QP를 다시
만든다. 따라서 C3는 fixed initial delta subdivision이 아니며 native endpoint와 같다는
제약도 없다.

다만 endpoint가 다르다는 사실만으로 ODE의 과학적 가치가 증명되지는 않는다. 다음 두
조건을 Method Session에서 식별해야 한다.

1. partial write 뒤 key, residual, proposal direction, utility ranking 또는 active
   barrier가 실질적으로 변한다.
2. 그 변화에 반응한 refreshed BF가 동일 terminal goal, path/endpoint budget,
   compute 조건의 best static/frozen controller보다 좋은 frontier를 만든다.

## 4. Motivation evidence ladder

| 단계 | 질문 | 관측 | 최종 역할 |
|---|---|---|---|
| EasyEdit fidelity | ODE-side hook이 native writer를 보존하는가 | exact tensor/logit/rollback fidelity 통과 | baseline 신뢰성 확보 |
| exact split negative control | 작은 step 자체가 개선을 만드는가 | fixed delta split은 native endpoint/practical floor | waypoint-only claim kill |
| same-snapshot allocation | layer utility/share heterogeneity가 있는가 | 두 모델에서 local allocation signal | H1 survive |
| atomic MEMIT refresh | partial write 뒤 refresh signal이 있는가 | Llama/Qwen local positive | H2 survive |
| projected Alpha refresh | projection 뒤에도 signal이 남는가 | 두 모델·8/8 positive local panel | family transfer 가능성 |
| central-probe G selector | 간단 selector가 model-common한가 | model별 선택/endpoint 불안정 | selector kill |
| 4-edit always-refresh | unconditional refreshed path가 보존을 개선하는가 | capacity/locality proxy 감소, efficacy/retention 악화 | safeguard 필요성 확인 |
| C0/C1 capacity history | cumulative BF controller가 바로 동작하는가 | C1 큰 under-write와 native gap | implementation diagnosis 필요 |
| C2 exact-quarter | magnitude를 복원하면 회복하는가 | 회복하지만 cap/share까지 바뀌어 cause 혼합 | 불충분 control |
| C3 magnitude-only | C1 allocation algorithm과 speed를 분리하면 회복하는가 | 4 cell, 15/16 edit 회복 | implementation-RCA pass |

Motivation에서 kill된 것은 다음이다.

- fixed initial update를 나눠 쓰는 waypoint-only ODE narrative
- central-probe G selector
- unconditional always-refresh full-distance sequential skeleton
- C1 raw absolute coefficient writer
- C2의 changed-share 결과를 BF share의 최종 verdict로 사용하는 해석
- model-specific threshold/K/policy rescue

살아남은 것은 다음이다.

- current-state relinearization의 local mechanism
- layer별 utility와 capacity에 따른 relative BF routing
- share와 speed를 독립적으로 정해야 한다는 설계 원칙
- MEMIT과 Alpha actuator에 공통 controller를 둘 가능성
- 장기 sequential에서 누적 cross term과 load concentration을 제어할 가능성

## 5. C3 intervention과 기술 무결성

C3의 목적은 C1의 raw QP coefficient \(x^\star\)가 layer ratio와 global magnitude를
동시에 결정하던 문제를 분리하는 것이었다. C1-compatible raw allocation을 다시 푼 뒤

\[
s^{(k)}=\frac{x^{\star(k)}}{\|x^{\star(k)}\|_2},
\qquad
y^{(k)}=\frac D4s^{(k)}
\]

로 actual coefficient를 만들었다.

기술 계약은 통과했다.

- job `15891`, 4 GPU, elapsed `01:29:01`, exit `COMPLETED 0:0`
- controller/evaluator `16/16` terminal/pass
- checkpoint `32/32`
- QP `16` edits, `64` hops
- max path/hop/share/radial identity error:
  `1.802e-16 / 2.715e-16 / 2.220e-16 / 1.388e-17`
- direct-z branch/edit당 1회
- covariance, Alpha projector/history, Wikipedia artifact read-only 재사용
- EasyEdit source 미수정
- C1/C2/C3 native current arrays 동일: version comparison의 native drift 없음

“C1 share를 유지했다”는 표현은 algorithm-level로만 정확하다. 첫 edit/첫 round의 C1과
C3 coefficient는 정확히 동일하지만 첫 C3 hop의 magnitude가 달라지는 순간 state가
갈라지고 후속 direction/share도 다시 계산된다. 따라서 C3는 전체 C1 trajectory를
post-hoc rescale한 순수 replay가 아니라, 동일 allocation algorithm에 대한 magnitude
policy의 total effect다.

## 6. C3 결과

모든 efficacy/retention 값은 `C3 QP − native`다.

| Model | Family | C1 current | C2 current | C3 current | C3−C1 | C3−C2 | Final prior retention | Retention AUC | Capacity 감소 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | MEMIT | `-4.032392` | `-1.369350` | `-0.361365` | `+3.671027` | `+1.007986` | `-0.292871` | `-0.2643` | `5.76%` |
| Llama | Alpha-history | `-2.495123` | `-1.137219` | `-0.563717` | `+1.931406` | `+0.573502` | `-0.782910` | `-0.5410` | `12.57%` |
| Qwen | MEMIT | `-0.397837` | `-0.797188` | `-0.251563` | `+0.146274` | `+0.545625` | `-0.295380` | `-0.4312` | `53.38%` |
| Qwen | Alpha-history | `+0.029707` | `-0.400738` | `+0.555242` | `+0.525536` | `+0.955980` | `-0.014337` | `-0.1958` | `71.03%` |

### 6.1 통과한 gate

- 양 모델 family-average C3−C1 current recovery가 양수다.
- 네 cell 모두 aggregate current가 C1보다 회복했다.
- per-edit 기준 15/16이 C1보다 회복했다.
- capacity, max-layer share, layer Gini reduction은 네 cell 공통이다.
- Qwen Alpha는 current `+0.555242`, prior retention `-0.014337`의 local strong point다.

이는 statistical superiority gate가 아니라 사전 명시한 implementation-RCA와 lenient
directional Motivation gate다. Cell당 edit 4개, 하나의 order와 seed이며 case가
model/family 사이에서 반복되므로 confidence interval이나 population generalization을
주장하지 않는다.

### 6.2 통과하지 못한 gate

- strict current non-collapse floor `-0.10`은 Qwen Alpha만 통과했다.
- strict pair passing family는 `[]`다.
- final prior retention과 retention AUC는 네 cell 모두 native보다 낮다.
- neighborhood/generation KL의 개선 방향은 model-common하지 않다.
- Qwen MEMIT cumulative Frobenius는 native보다 악화됐다.
- controller compute는 native의 `4.80--6.93×`다.

따라서 capacity/concentration proxy 감소를 preservation superiority로 바꿔 말하지
않는다.

## 7. Direct-z와 state refresh 감사

### 7.1 실제로 고정된 것

- edit entry state에서 계산한 direct-z tensor \(Z_i^\star\)
- target token identity
- precomputed covariance/projector

Direct-z를 매 hop 다시 최적화하지 않는 것은 proposal의 의도된 초기 design이다. Moving
target confound와 큰 compute를 피하면서 같은 semantic/representation target을 향한
서로 다른 parameter path를 비교하기 위함이다.

### 7.2 매 hop 다시 계산된 것

- current last-critical-layer activation \(H^L(W^{(k)})\)
- current residual \(Z^\star-H^L(W^{(k)})\)
- current layer keys
- current MEMIT/Alpha low-rank proposal
- unit-C proposal direction
- finite-difference rewrite efficiency slope
- cumulative capacity와 cross term
- raw QP allocation과 applied relative velocity

Controller는 앞 hop을 permanent apply한 뒤 다음 synchronous proposal을 현재 model에서
새로 만든다. 따라서 C3는 구조적으로 meaningful state-dependent flow다.

다만 compact evaluator는 state lineage와 proposal artifact를 강하게 검증하지만 residual,
key, finite-difference slope tensor를 독립 재실행해 attestation하지는 않는다. 또한
round 간 direction C-cosine, key drift, rank turnover는 canonical metric으로 측정하지
않았다. 이 non-stationarity panel은 Method identity test에서 추가해야 한다.

### 7.3 Direct-z fidelity claim의 현재 경계

C3가 측정한 것은 direct-z compute count/hash/lineage와 rewrite utility다. 다음 hidden
representation fidelity는 측정하지 않았다.

\[
\|H^L(W_T)-Z^\star\|,
\qquad
\cos(H^L(W_T),Z^\star).
\]

Rewrite utility progress ratio

\[
\rho_U
=\frac{U_{\rm C3}-U_0}{U_{\rm native}-U_0}
\]

는 Llama MEMIT `0.970`, Llama Alpha `0.957`, Qwen MEMIT `0.979`, Qwen Alpha
`1.038`이다. 이는 native utility progress의 회복이지 direct-z tensor를 더 충실히 썼다는
증거가 아니다. `1.038`도 fidelity가 아니라 overshoot일 수 있다.

또한 edit 2부터 native와 QP branch state가 달라져 각 branch가 새로 계산하는 direct-z와
native distance도 달라진다. 후속 edit 비교는 동일 \(Z^\star\) writer의 paired 비교가
아니라 동일 request stream을 각자 current state에서 처리하는 closed-loop comparison이다.

## 8. C3가 원 proposal을 구현한 범위

| Proposal 요소 | C3 판정 |
|---|---|
| edit당 direct-z 1회, inner trajectory에서 고정 | implemented |
| 같은 snapshot에서 모든 layer proposal 생성 | implemented |
| current residual/key/proposal refresh | implemented |
| unit-C layer actuator | implemented |
| cumulative \(W^0\)-relative capacity | implemented |
| non-negative BF layer share | implemented |
| MEMIT/Alpha 공통 controller | implemented |
| exact smooth-max Φ와 directional derivative | partial; mean utility + finite difference |
| soft-slack minimum-capacity QP | partial; raw allocation에 한정 |
| applied hard capacity barrier | intentionally absent |
| adaptive trust \(h\), reject/rollback | intentionally absent |
| first-hitting terminal | diagnostic-only |
| adaptive \(K\), Euler–Heun/refinement | absent |
| dynamic-vs-static ODE necessity ablation | absent |
| 1K–10K/full sequential/downstream | absent |

따라서 C3는 proposal의 motivation mechanism을 구현한 kill test지만 deployable BF-ODE
method를 구현한 실험은 아니다.

## 9. Fixed `D/4 × 4` heuristic의 역할과 한계

C3의 \(D\)는 direct-z distance가 아니라 각 QP branch의 current state에서 계산한 ordered
native proposal의 C-distance다.

\[
D=\|\Delta W_{\rm native}\|_C,
\qquad
h=D/4.
\]

### 9.1 Kill test에서의 역할

- C1 under-write를 제거한다.
- BF share와 global magnitude를 분리한다.
- 공통 \(K=4\)로 model-specific retune을 막는다.
- exact path-length audit가 가능하다.
- state refresh를 여러 번 관찰할 최소 waypoint를 제공한다.

### 9.2 Method로 사용할 수 없는 이유

1. **Native scale 상속:** \(D\)는 semantic goal의 intrinsic distance가 아니므로 native의
   over/under-write를 그대로 상속한다.
2. **Path와 endpoint 혼동:** 보장한 것은
   \(\sum_k\|\Delta W^{(k)}\|_C=D\)이지
   \(\|W^{(4)}-W^{(0)}\|_C=D\)가 아니다.
3. **Raw speed 삭제:** near-zero QP 해도 full \(D/4\)로 확대한다. Overall radial scale은
   평균 `2.106×`, 최대 `35.927×`다.
4. **QP optimum 파괴:** raw objective
   \(b^Tx+x^TQx\)는 radial scale 뒤
   \(\alpha b^Tx+\alpha^2x^TQx\)가 되어 같은 optimum/barrier를 보존하지 않는다.
5. **Applied barrier 부재:** raw allocation cap violation은 0이지만 radial 적용 뒤
   `140/320` layer-hop coefficient가 cap을 넘었다.
6. **Forced over-integration:** rewrite target 또는 native utility를 일찍 만족해도 네
   hop을 모두 적용한다.
7. **Always commit:** Qwen MEMIT의 rewrite gain `-0.057425`, trust ratio
   `-0.248366` step도 적용했다.
8. **고정 numerical resolution:** curvature와 무관하게 항상 네 proposal/probe/write를
   실행해 native 대비 큰 compute를 쓴다.

전체 chain QP/native path ratio는 Llama MEMIT `0.998178`, Llama Alpha `0.999789`,
Qwen MEMIT `1.002677`, Qwen Alpha `1.002477`로 가깝다. 그러나 이것은 endpoint norm
또는 endpoint parameter equality가 아니라 branch-local ordered-native path budget이
유사하다는 뜻뿐이다.

## 10. Method에 넣어야 하는 요소와 현재 개선 signal

| Method 요소 | C3에서 관찰된 필요 signal | 현재 강도 |
|---|---|---|
| actual share와 speed의 joint solve | radial 최대 `35.927×` | 매우 강함 |
| applied hard capacity constraint | cap exceed `140/320` | 매우 강함 |
| first-hitting stop | edit별 first-hit `3/4, 4/4, 4/4, 4/4` | 강함 |
| adaptive \(K\) | 다수 edit가 round 4 전에 hit | 강함 |
| trust reject/rollback | negative trust `1/64` | 직접 signal |
| adaptive \(h\) | raw near-zero allocation 강제 확대 | 매우 강함 |
| exact semantic event와 control surrogate 분리 | native utility match는 `0/4,0/4,1/4,2/4` | 필요 |
| feasible-progress frontier와 no-silent-slack policy | no-slack QP round `0/16,0/16,6/16,8/16` | 매우 강함 |
| dynamic-vs-static ablation | share/state가 변하나 performance causality 미확인 | 필수 |
| capacity proxy validation | capacity 공통 양성, retention/KL mixed | 필수 |
| direct-z fidelity panel | utility proxy만 존재 | 필수 |

C3의 first-hit은 teacher-forced exact top-1 diagnostic이다. Method section은
hard worst-case \(\Phi_{\rm event}\)를 terminal로, normalized smooth
\(\widetilde\Phi_{\rm ctrl}\)을 gradient/trust surrogate로 분리한다. C3는 이 최종
event/surrogate contract를 구현하지 않았으므로 위 first-hit 수치는 free-terminal의
필요성을 보여주는 signal로만 사용하고 Method stopping 성능으로 재사용하지 않는다.

Qwen Alpha의 aggregate positive는 per-edit `-0.201/-0.575/+0.767/+2.230`이며 마지막
positive가 `35.927×` radial amplification과 함께 나타났다. 유망한 local signal이지만
현재 form의 robust strong result로 사용하지 않는다.

## 11. ODE가 필수 톱니바퀴로 남는 조건

Adaptive BF share만으로는 논리적으로 ODE가 필수는 아니다. 동일한 controller를 discrete
trust-region 또는 receding-horizon optimization으로 표현할 수 있다. ODE가 research
contribution의 필수 톱니바퀴가 되려면 state dependence와 그에 반응한 trajectory
control이 실제 성능에 필요함을 보여야 한다.

1. layer vector field가 state-dependent다.
   \[
   B_l(W^{(k+1)},Z^\star)\ne B_l(W^{(k)},Z^\star).
   \]
2. partial joint write 뒤의 relinearization이 best static/frozen controller보다 실제
   frontier를 개선한다.
3. step refinement 또는 Euler–Heun 비교에서 endpoint/metric이 안정적이다.
4. fixed round가 아니라 adaptive trust와 first-hitting terminal이 trajectory length를
   결정한다.

Separated-layer order sensitivity까지 claim하려면 추가로
\(\operatorname{LieBracket}(F_i,F_j)(W)\ne0\) 또는 empirical order effect를 진단할
수 있다. Nonzero Lie bracket는 해당 order-sensitivity claim의 충분한 mechanistic
signal이지 joint state-dependent ODE의 일반적 필요조건은 아니다.

위 조건 중 state-dependent 코드 경로와 local signal은 살아 있지만 dynamic-vs-static
frontier, solver stability와 free terminal은 C3에서 검증하지 않았다. 따라서 현재
claim은 “ODE necessary”가 아니라 “ODE necessity를 Method identity test로 검정할
정당성이 생겼다”다.

## 12. Large-scale sequential에서 기대하는 우위

Outer edit \(t\) 이전 layer displacement와 새 write를

\[
\Delta_{t,l}=W_{t,l}-W_{0,l},
\qquad
u_{t,l}=W_{t+1,l}-W_{t,l}
\]

라 하면 capacity 증분은

\[
\Psi_{t+1,l}-\Psi_{t,l}
=\frac{
2\langle\Delta_{t,l},u_{t,l}\rangle_{C_l}
+\|u_{t,l}\|_{C_l}^2
}{D_l}.
\]

Atomic/small-scale에서는 quadratic update-size 항이 주로 보이지만 edit가 누적되면
cross term이 커진다. 같은 layer와 같은 방향으로 반복 write하면 idealized capacity는
`O(T^2)`로 증가할 수 있다. 여러 대체 가능한 layer로 load를 분산하면 concentration을
`O(T^2/L)` 방향으로 낮출 수 있고, cumulative displacement와 C-orthogonal한 actuator를
선택할 수 있다면 `O(T)`에 가까운 성장도 가능하다.

따라서 ODE-Edit이 native 대비 노려야 하는 우위는 “항상 더 작은 atomic update”가
아니다.

- 누적 load가 큰 layer의 marginal cost 상승
- 저부하/고효율 layer로 semantic progress rerouting
- max-layer load와 Gini 감소
- stale key/residual/proposal의 current-state refresh
- first-hit을 통한 edit당 excess write 제거
- negative-tail step rollback
- Alpha null-space/history 안에서 가능한 layer의 동적 선택
- edit age가 커져도 past-edit margin을 더 오래 유지

AlphaEdit은 small scale에서 projection 자유도 제한 때문에 MEMIT보다 efficacy가 낮을 수
있다. 이는 long-horizon에서 projection/history와 BF routing이 상보적인지의 질문과
구분한다. 비교는 `Alpha-BF vs canonical-history Alpha`와 `MEMIT-BF vs native MEMIT`을
각 family 안에서 먼저 수행하고, 이후 공통 efficacy–retention–compute frontier를
비교해야 한다.

현재 large-scale superiority는 미검증이다. Capacity proxy가 actual retention과
정렬되지 않거나 layer actuator가 대체 불가능하거나 Alpha feasible subspace가 고갈되면
예상 우위는 사라질 수 있다. Long horizon은 현재 약한 sign이 자동으로 뒤집힌다는 면책
논리가 아니며, 잘못된 forced step과 rare negative step도 함께 누적시킨다.

## 13. 최종 claim ledger

### 13.1 현재 열 수 있는 claim

1. BF relative layer allocation과 global update magnitude는 분리해야 한다.
2. 이 분리는 동일 4-case panel에서 C1 under-write와 efficacy collapse를 두 모델·두
   actuator family에 걸쳐 크게 회복했다.
3. Branch-local native C-path budget을 거의 맞춘 상태에서도 ex-post cumulative capacity와
   layer-load concentration 감소가 공통으로 남았다.
4. Fixed direct-z target 아래에서도 current residual/key/proposal/share를 refresh하면
   native와 다른 state-dependent trajectory와 endpoint를 구성할 수 있다.
5. State refresh + BF share + independent adaptive step을 완성 Method로 검증할 방향성이
   있다.

### 13.2 현재 열 수 없는 claim

1. ODE가 static BF 또는 discrete controller보다 필수다.
2. BF가 DF/uniform/static allocation보다 우월하다.
3. C3가 direct-z hidden representation을 native보다 더 충실히 write한다.
4. Actual C3 write가 per-step minimum-capacity solution이다.
5. Applied hard capacity barrier가 유지된다.
6. Native보다 preservation과 downstream capability가 좋다.
7. Lifelong/large-scale sequential에서 collapse를 늦춘다.
8. Alpha-BF가 일반적으로 MEMIT보다 좋다.
9. 현재 fixed `D/4 × 4` policy를 그대로 scale하면 된다.

## 14. Motivation closure와 Method opening

### 14.1 Motivation 종료 규칙

- Session 01 안에서 K/share/threshold/model-specific rescue를 더 하지 않는다.
- C1/C2/C3를 추가로 retune해 strict native superiority를 만들지 않는다.
- C3의 directional signal을 lifelong claim으로 확장하지 않는다.
- fixed split 또는 waypoint count를 ODE contribution으로 사용하지 않는다.

### 14.2 Method Session의 첫 과제

> **Post-closure design amendment (2026-08-03):** 이 subsection은 Motivation 종료
> 시점에 필요하다고 본 완성형 safeguard 목록을 보존한다. 이후 fast-main-table review는
> C3의 capacity–retention 불일치를 근거로 signed (Psi) hard barrier를 primary에서
> 내리고, outer-edit terminal net-write (Omega) soft routing을 먼저 검증하도록
> 순서를 바꿨다. 후속 compute review는 평균 accepted round 목표를 제거하고 field-cost를
> co-primary로 올렸다. 최신 실행 source는
> [`2026-08-03-session02-compute-aware-main-table-spec.md`](../../plans/global/2026-08-03-session02-compute-aware-main-table-spec.md)다.

다음 Method section은
[`../../project/proposals/sections/04-method-design.md`](../../project/proposals/sections/04-method-design.md)에서 연다.
첫 구현 단위는 다음을 하나의 model-common controller로 결합해야 한다.

1. Actual applied coefficient를 직접 푸는 capacity/progress constrained optimization
2. Applied full-step hard barrier
3. Semantic rewrite deficit와 first-hitting terminal
4. Actual/predicted progress trust ratio, reject와 exact rollback
5. Adaptive \(h\)와 emergent \(K\)
6. Direct-z residual/key/direction/efficiency/capacity/velocity non-stationarity log
7. Dynamic-vs-static matched ODE necessity ablation

### 14.3 Stage gate

Method는 곧바로 lifelong scale로 가지 않는다.

- **M0 identity:** 같은 entry state에서 fixed split, frozen direction/static share,
  refreshed direction/static share, frozen direction/dynamic share, full dynamic BF를 비교
- **M1 strong pilot:** 두 모델×두 family의 10-edit multi-order pilot에서 native current
  non-collapse와 retention/capacity의 model-common 축 확인
- **M2 medium stream:** 100-edit에서 edit-age retention, collapse onset, load
  concentration, compute frontier 확인
- **M3 large stream:** M1/M2 통과 후에만 1K+와 downstream evaluation 개방

각 arm은 동일 semantic stopping rule, efficacy target, allowed prompt/context, model-common
hyperparameter를 사용해야 한다. Path length만 맞추지 말고 endpoint displacement,
accepted NFE, wall time도 함께 보고한다.

## 15. Reproducibility, resource, artifact boundary

- C3 source commit: `bbd3d0ec5b8c41df96f367a0503d29823d7d7b89`
- C3 reporting commit: `6145406`
- run/job: `15891` / `odeedit_capacity_history_pair_c3_v1`
- server/resource: server1, GPU 4, CPU 32, `260000M`
- raw artifact root: ignored
  `local/results/raw/session01_motivation/caphist_*_c3_v1/`
- compact aggregate SHA-256:
  `bc2d3da7fe3cb7ae0859e72e6bcdb144c53f0307f0d013be5bacc511d4203127`
- raw factors, checkpoints, full logs, datasets와 model weights는 Git에 추가하지 않았다.

본 closure 작성 과정에서는 새 실험을 실행하지 않았고 실행 코드·EasyEdit source·raw
artifact를 수정하지 않았다. C3 proposal/spec/report/code에 범위를 제한한 독립 read-only
검토를 병렬로 수행해 수치, proposal fidelity, ODE/direct-z 해석, 문서 supersession을
교차검토했다.

## 16. 최종 판정

Motivation은 다음 상태로 확정한다.

> **`CLOSED_DIRECTIONAL_POSITIVE`** — BF share와 global magnitude 분리의 필요성과
> state-refreshed BF trajectory를 Method로 넘길 이유는 확인됐다.

Method는 다음 상태로 연다.

> **`OPEN; STRONG_NATIVE_GATE_NOT_YET_PASSED`** — adaptive step, actual barrier,
> trust rollback, first-hit, ODE necessity ablation을 구현·통과하기 전에는 native
> preservation 또는 lifelong superiority를 주장하지 않는다.
