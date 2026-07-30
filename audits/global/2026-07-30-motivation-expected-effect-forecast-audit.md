# ODE-Edit Motivation 기대효과 예측 measurement audit

- 작성일: 2026-07-30
- 문서 성격: 독립 red/measurement audit
- 적용 범위: Session 01 — Motivation Validation의 MEMIT 기반 MV-1~MV-4
- 현재 판정: **조건부 PASS**
- 숫자 상태: **현재 성능 개선 수치는 식별 불가** — 이 문서는 수치를 만드는
  문서가 아니라, 어떤 관측치가 쌓였을 때 어느 수준의 기대효과까지 방어
  가능하게 예측할지를 고정한다.

## 1. 감사 결론

ODE-Edit의 기대효과는 하나의 숫자로 미리 예측하면 안 된다. 다음 네 양을
서로 다른 claim으로 순서대로 추정해야 한다.

1. `candidate-set oracle allocation - best static`:
   현재 action set 안에 routing opportunity가 얼마나 있는지 보여주는
   **사후적 경험 상한**
2. `held-out/cross-fitted controller - best static`:
   event outcome을 보지 않는 controller가 실제로 회수할 수 있는
   **achievable allocation gain**
3. `refreshed direction - fixed direction`:
   allocation 효과와 분리한 **dynamic refresh incremental gain**
4. matched-progress displacement와 retention:
   같은 rewrite progress에서 parameter cost를 줄이는지, 그리고 그 감소가
   short stream retention으로 실제 이어지는지 보는 **mechanism-side gain**

현재 proposal, plan, helper code만으로는 위 네 숫자의 부호나 크기를 알 수
없다. 특히 MV-0 fidelity 또는 descriptive layer heterogeneity를 이용해
“몇 % 향상 예상”을 쓰는 것은 근거가 없다. 가장 이른 정량 forecast 시점은
MV-1 confirmatory 결과가 나온 뒤이며, retention forecast는 MV-4 intervention
전에는 금지한다.

`20 calibration / 60 confirmatory / 20 untouched` 구조는 유지할 수 있다.
다만 calibration 승자 성능, event-wise oracle, confirmatory에서 결과를 본 뒤
고른 controller를 섞지 않도록 아래 lock 절차가 먼저 필요하다. 이를
지키면 큰 stream을 시작하기 전에 작은 paired experiment로 opportunity,
achievability, refresh, displacement를 차례로 kill할 수 있다.

## 2. 네 범주

### Proposal에서 온 내용

- 동일 snapshot의 layer proposal을 local actuator로 비교하고, rewrite
  utility와 editor-native geometry로 allocation을 정한 뒤 partial update
  후 proposal을 refresh한다는 가설
- static layer scaling이 동등하거나 partial update 뒤 decision이 유의하게
  변하지 않으면 ODE가 불필요하다는 falsification condition
- paraphrase, neighborhood/locality, downstream prompt를 controller에서
  차단하는 information firewall
- 같은 rewrite goal에서 더 낮은 cumulative displacement를 선택하면 장기
  retention에 유리할 수 있다는, 아직 검증되지 않은 mechanism 가설

### Repo/protocol에서 확인한 사실

- canonical plan은 model별 deterministic 100 case를
  `calibration 20 / confirmatory 60 / untouched 20`으로 나눈다.
- case 선택은 evaluation outcome이나 row field가 아니라 salted
  `case_id` hash만 사용한다.
- 고정 backbone은
  `meta-llama/Meta-Llama-3-8B-Instruct@8afb486...`와
  `Qwen/Qwen2.5-7B-Instruct@a09a354...` 두 개다.
- Session 01의 primary baseline은 local EasyEdit MEMIT이다. 그 구현은
  ascending-order Gauss–Seidel-style one-pass construction이며, 완전한
  open-loop baseline이 아니다.
- package에는 현재 다음 pure measurement primitive가 있다.
  - factor-only `C` inner product, norm, cosine
  - detached `exact_top1_stop`
  - differentiable `smooth_target_utility`
  - central finite-difference calibration
  - joint additivity, rank turnover, reroutability
- `smooth_target_utility`는 context/token별 tensor를 반환하지만, confirmatory
  분석에 쓸 aggregation rule과 normalization은 아직 이 helper가 결정하지
  않는다.
- `rank_turnover`는 descriptive statistic이고, `reroutability`의
  `best_route`는 관측 outcome으로 사후 선택된다. 둘 다 그대로 achievable
  controller 효과가 되지 않는다.
- bootstrap, sequential boundary, frontier AUC, stream retention 추정기는
  현재 읽은 metric helper에 구현돼 있지 않다. 구현 전에는 해당 분석이
  “준비 완료”라고 간주할 수 없다.

### GH 추정

- 기대효과의 가장 방어 가능한 forecast는
  `oracle ceiling / held-out achievable / refresh increment / displacement`
  네 칸을 나란히 보여주는 형태다. 이들을 더한 단일 예상 향상률은 interaction
  때문에 일반적으로 식별되지 않는다.
- calibration 20은 5개 layer의 연속 allocation과 다수 hyperparameter를
  학습하기에는 작다. 초기 controller는 utility와 `C` cost를 입력으로 하는
  저차원·단조 rule 또는 작은 사전 지정 후보군이어야 한다.
- 두 model만으로 backbone population의 random-effect를 추정할 수 없다.
  pooled 값은 두 pinned snapshot의 equal-weight summary일 뿐이다.
- `C`-weighted displacement가 낮아졌다는 사실만으로 retention이 개선된다고
  번역할 수 없다. MV-4 arm intervention이 살아남아야 한다.

### 사용자 확인 필요

- 즉시 필요한 확인은 없다. canonical plan의 “두 model 모두 생존” 기준을
  따른다.
- 추가 NFE와 효과를 하나의 scalar utility로 환산하려면 “GPU 시간 1단위와
  rewrite/retention 1단위의 교환가치”가 별도로 필요하다. 사용자가 이를
  정하지 않는 동안에는 임의 가중합을 만들지 않고 effect–NFE–wall-clock
  Pareto 표로 판정한다.

## 3. 공통 관측 단위와 metric lock

Model `m`, state-edit event `i`, arm `a`에 대해 모든 paired arm은 동일한
`pre_edit_state_hash`, sanitized request, frozen context, direct-z, covariance,
dtype, RNG state에서 갈라져야 한다.

다음 값을 event-arm 단위 raw trace에 남긴다.

| 기호 | 관측치 | 사용 |
| --- | --- | --- |
| `U_before`, `U_after(a)` | calibration에서 고정한 smooth rewrite utility aggregation | 실제 progress |
| `P_i(a)=U_after(a)-U_before` | higher-is-better realized progress | MV-1/MV-2 |
| `S_i(a)` | 모든 허용 context/token의 exact top-1 first-hit 여부 | reach/failure |
| `D_state_i(a)` | base 대비 normalized `C`-weighted state displacement | MV-3 proxy |
| `D_path_i(a)` | accepted update의 non-negative `C` expenditure 합 | path cost |
| `NFE_i(a)` | forward/backward/solve 및 accepted/rejected round | compute match |
| `T_i(a)` | wall-clock과 GPU/host-memory peak | compute frontier |
| `F_i(a)` | non-finite, rejection, target-not-reached, rollback 등 status | ITD denominator |

`exact_top1_stop`은 terminal 판정에만 쓰고 gradient loss로 쓰지 않는다.
`smooth_target_utility`의 다음 항목은 calibration artifact를 열기 전에
manifest에 고정한다.

- target tokenization과 teacher-forced position
- context/token aggregation
- temperature
- worst-context와 mean-context 중 primary
- multi-token mask와 denominator
- higher-is-better sign

Model logit scale가 다르므로 raw `P_i`는 model별 primary로 유지한다. pooled
summary가 필요하면 calibration-only scale

```text
s_m = max(MAD(calibration native-MEMIT progress),
          calibration replay-noise envelope)
```

를 한 번 고정하고 `P_i/s_m`를 사용한다. event outcome으로 매번 나누는
“percent deficit closed”는 작은 denominator에서 폭발하므로 primary로 쓰지
않는다. exact reach-rate difference는 model 간 비교 가능한 별도
percentage-point metric으로 보고한다.

## 4. Baseline과 유한 action set

Motivation forecast는 EasyEdit MEMIT proposal family 안에서 먼저 만든다.
초기 finite action set은 다음으로 제한한다.

1. native MEMIT ordered construction
2. fixed native delta `K`-step negative control
3. global alpha/first-hit
4. `C`-normalized uniform allocation
5. calibration-fixed static layer allocation
6. best single-layer/WilKE-style proxy
7. analytic utility allocation
8. fixed-direction dynamic coefficient
9. refreshed-direction refreshed-coefficient allocation
10. Gauss–Seidel small-step

`WilKE-style proxy`는 WilKE reproduction이라고 부르지 않는다. AlphaEdit,
NAS, ENCORE와 외부 방법은 MEMIT Motivation이 생존하기 전 기대효과 추정에
섞지 않는다.

연속 simplex를 촘촘히 탐색해 “oracle”을 키우는 것은 금지한다. calibration
20에서 action grid, step budget, tie rule을 고정하고 confirmatory/untouched에서
그 유한 집합만 평가한다. 필요하면 다섯 single-layer probe, uniform,
static, analytic controller를 먼저 평가하고 pairwise mixture는 MV-3
reroutability gate가 살아난 뒤에만 연다.

`best static`은 event별 winner가 아니다. Model별 calibration split에서
공통으로 한 번 선택해 confirmatory 전체에 고정하는 policy다. 선택 목적은
고정 `C` budget에서의 mean progress이며, replay near-tie 안에서는 더 단순하고
compute가 싼 policy를 택한다. calibration에서 본 apparent winner gain은
보고하지 않는다.

## 5. 네 기대효과 estimand

### 5.1 Candidate-set oracle allocation gap

Locked finite set을 `A_lock`, calibration-fixed best static을 `a_static,m`이라
두면:

```text
O_mi = max_{a in A_lock} P_mi(a)
G_oracle,m = E_i[O_mi - P_mi(a_static,m)]
```

로 정의한다.

이 값은 실제 controller가 event별 arm outcome을 미리 안다고 가정한
retrospective ceiling이다. 따라서:

- `candidate-set empirical upper bound`라고만 부른다.
- true continuous-allocation oracle 또는 achievable method gain이라고 부르지
  않는다.
- calibration replay로 정한 near-tie envelope 안의 arm은 동률 처리한다.
- candidate 수와 각 arm의 failure를 함께 공개한다.
- max-selection에 의한 upward bias가 있으므로 point estimate보다 paired
  interval과 near-tie-adjusted sensitivity를 함께 낸다.
- oracle branch 결과는 controller decision/hash를 먼저 기록한 뒤 분석
  process에 공개한다.

Oracle gap의 upper confidence bound가 replay/null envelope를 넘지 못하면
controller를 더 잘 설계해도 현재 action set에서 회수할 routing opportunity가
없다는 뜻이므로 가장 빠른 kill 근거가 된다.

### 5.2 Achievable allocation gain

허용된 pre-step feature만 보는 controller를 `f_m(x_i)`라 하면:

```text
G_alloc,m =
  E_i[P_mi(f_m(x_i)) - P_mi(a_static,m)]
```

가 실제 기대효과의 첫 primary다. `x_i`에는 현재 analytic utility,
`C` norm/cost, allowed rewrite deficit만 들어갈 수 있고 actual branch
outcome, paraphrase, locality, retention은 들어갈 수 없다.

가장 엄격한 primary 절차는 다음이다.

1. calibration 20에서 feature transform, controller family, hyperparameter,
   missing/failure fallback, static comparator를 모두 고정한다.
2. confirmatory event마다 controller action을 실제 arm outcome을 열기 전에
   파일과 hash로 기록한다.
3. confirmatory 60에서 한 번 평가한다.
4. untouched 20에는 threshold나 controller를 다시 맞추지 않고 그대로
   적용한다.

학습형 controller가 필요하면 calibration에서 family와 hyperparameter를
고정한 뒤 confirmatory 60을 사전 지정 5-fold로 cross-fit할 수 있다. 각
held-out 12-case fold의 action은 그 fold outcome을 전혀 보지 않은 fit에서
나와야 하며, fold split도 `case_id` hash로 미리 고정한다. 이 결과는
`cross-fitted achievable estimate`로 별도 표기한다. calibration-only frozen
controller 결과와 섞지 않는다. 최종 untouched test에는 사전에 명시한
fitting rule만 한 번 적용한다.

Flexible classifier, per-relation lookup, layer별 unconstrained coefficient
regression은 calibration 20에서 과적합 위험이 높으므로 초기 lane에서
block한다.

### 5.3 Refresh incremental gain

Allocation 자체와 refresh를 분리하기 위해 canonical MV-2 contrast를 그대로
쓴다.

```text
G_direction-refresh,m =
  E_i[P_mi(refreshed direction + refreshed coefficient)
      - P_mi(fixed initial direction + refreshed coefficient)]

G_coefficient-refresh,m =
  E_i[P_mi(fixed direction + refreshed coefficient)
      - P_mi(fixed direction + fixed coefficient)]
```

두 arm은 같은 첫 partial joint step, direct-z, step-size schedule, maximum
round, NFE budget, reject/fallback rule을 가져야 한다. 한 arm만 target을
일찍 맞힌 경우에도 success/failure와 사용 NFE를 ITD에 그대로 남긴다.

`G_direction-refresh`가 ODE/relinearization 기대효과다.
`G_coefficient-refresh`만 positive면 fixed-direction dynamic controller로
pivot한다. 서로 다른 experiment에서 얻은 `G_alloc`과 `G_direction-refresh`
point estimate를 단순 합산해 “총 예상 향상”을 만들지 않는다. 총 효과는
동일 factorial arm 안에서 `refreshed joint - best static`으로 직접 측정한다.

### 5.4 Matched-progress displacement와 retention gain

MV-3에서는 calibration에서 공통 달성 support와 progress grid `Q`를 먼저
고정한다. 각 arm의 first-hitting displacement frontier를:

```text
D_a(q) = minimum displacement when progress q is first reached
```

로 두고, higher-is-better gain을:

```text
G_frontier,m =
  restricted_AUC_q[D_static(q)] - restricted_AUC_q[D_ODE(q)]
```

로 정의한다. `D_state`와 `D_path`는 분리한다.

Target-not-reached를 제외한 conditional 평균만 내면 under-edit arm이 유리해질
수 있다. 따라서 primary report는 반드시 다음을 같이 둔다.

- ITD reach-rate difference at each `q`
- 모든 event를 포함하는 restricted frontier score
- 양 arm이 모두 도달한 event의 conditional displacement difference
- target-not-reached, rejected, failed count

Percent displacement reduction은 양 arm이 모두 도달하고 static denominator가
noise floor보다 큰 event에서만 secondary로 보고한다. Common support는
confirmatory outcome을 보고 다시 자르지 않는다.

MV-4의 retention 기대효과는:

```text
G_retention,m =
  retention(displacement-reducing arm)
  - retention(global-alpha matched-progress arm)
```

이며 동일 window start state와 cumulative achieved rewrite progress에서
측정한다. 이는 non-overlapping paired window가 primary unit이다. 두 order,
current-edit underfitting, failed edits, total norm, target difficulty를
함께 공개한다.

MV-3 전에는 `matched-progress displacement reduction forecast`까지만 허용한다.
MV-4에서 arm intervention이 실제 displacement 감소와 retention 증가를 둘 다
보인 뒤에만 retention 기대효과를 말한다. Arm-level
`Delta retention / Delta displacement` ratio는 denominator가 zero에서
분리되고 bootstrap interval이 안정적인 경우에만 sensitivity로 제시하며,
일반적인 causal mediation slope라고 부르지 않는다.

## 6. 불확실성과 선택편향 방지

### 6.1 Split 역할

| split | 허용 | 금지 |
| --- | --- | --- |
| calibration 20 | noise envelope, step/C budget, action grid, metric aggregation, controller family/hyperparameter 고정 | 성능 claim, oracle 기대효과 claim |
| confirmatory 60 | locked primary contrasts와 CI | 결과를 본 threshold/controller/arm 변경 |
| untouched 20 | exact locked replication | 실패 뒤 retuning, favorable case만 pooling |

Untouched 결과를 한 번 본 뒤에는 더 이상 untouched라고 부르지 않는다.
Controller code hash, analysis code hash, action/fold manifest를 그 전에
고정한다.

### 6.2 Paired uncertainty

- MV-1~3의 primary unit은 event다. layer, context, token을 독립 sample로
  세지 않는다.
- model별 mean paired effect, median, sign fraction, trimmed mean과
  denominator flow를 함께 보고한다.
- CI는 case-ID cluster paired bootstrap으로 계산한다.
- Llama와 Qwen이 같은 case ID를 쓰면 pooled bootstrap도 case ID를 두
  model에서 함께 resample하여 cross-model correlation을 보존한다.
- 여러 primary contrast는 simultaneous interval 또는 Holm correction을
  적용한다.
- Oracle max의 interval은 ordinary arm contrast보다 낙관적일 수 있으므로
  near-tie-adjusted 결과와 candidate count를 병기한다.
- MV-4는 edit를 iid로 bootstrap하지 않고 non-overlapping window 또는 paired
  moving-block bootstrap을 쓴다. 두 order를 sample 수 2배로 부풀리지 않는다.
- technical failure, non-finite, zero utility, pre-satisfied, rejected step,
  target-not-reached를 사후 제거하지 않는다.

### 6.3 Model별 및 pooled 판정

모든 primary effect는 Llama와 Qwen을 먼저 따로 판정한다.

Pooled 값은 calibration-fixed scale로 표준화한 두 model effect의 equal-weight
평균과 exact reach-rate의 equal-weight 평균만 허용한다. Model이 두 개뿐이므로
random-effects “backbone generalization” CI를 만들지 않는다.

| 결과 | 판정 |
| --- | --- |
| 두 model 모두 positive이고 locked untouched에서도 같은 sign | cross-model Motivation signal |
| 한 model만 positive | architecture-conditional/inconclusive; pooled 값으로 rescue 금지 |
| 두 model 모두 null/noise-equivalent | 해당 mechanism kill |
| 두 model sign 반대 | pooled 평균 보고는 가능하나 GO 근거로 사용 금지 |

Canonical Motivation closure에는 두 model 생존을 요구한다. 한 model의 큰
gain이 다른 model의 harm을 상쇄한 pooled 평균은 통과가 아니다.

### 6.4 Overclaim 방지 문구

허용:

> Locked CounterFact candidate set에서 event-wise oracle은 static policy보다
> 여지가 있었고, outcome을 보지 않은 controller가 두 pinned model의 held-out
> events에서 그 일부를 회수했다.

금지:

- oracle gap을 “ODE-Edit 예상 향상”이라고 부르기
- confirmatory를 본 뒤 고른 controller의 같은 split 성능을 held-out이라고
  부르기
- displacement 감소를 retention 증가로 자동 환산하기
- 두 model pooled 값을 일반 LLM backbone 평균으로 부르기
- CounterFact 100-case 결과를 long-horizon superiority로 외삽하기

## 7. 비용 효율적인 sequential design

두 model은 dependency와 red gate가 모두 PASS한 뒤 각각 GPU 1개로 같은
wave에 동시에 실행한다. Llama 결과를 본 뒤 Qwen 제출 여부를 정하지 않는다.
이는 wall-clock을 줄일 뿐 아니라 model별 선택적 실행 bias도 막는다.

### 7.1 Case ladder

| wave | model별 누적 case | 목적 | 허용 결정 |
| --- | ---: | --- | --- |
| C0 | calibration 20 | noise, budget, action/controller lock | claim 없음 |
| C1 | confirmatory 20 | conservative futility look | early kill만, early GO 금지 |
| C2 | confirmatory 40 | 두 번째 futility look | early kill만 |
| C3 | confirmatory 60 | primary final estimate | go/kill/pivot |
| U | untouched 20 | locked replication | cross-model closure |

Interim look을 쓸 경우 boundary와 bootstrap seed를 C0에서 고정한다. C1/C2의
positive result로 성공 선언하지 않는다. 빠른 kill은 두 model 모두에서
oracle과 achievable effect의 conservative upper bound가 calibration
replay/near-tie envelope를 넘지 못할 때만 허용한다. 한 model이라도 아직
불확실하면 C3까지 진행한다. Interim 종료 사실과 사용한 case 수를 그대로
보고하며, 중도 종료 estimate를 full-sample estimate처럼 쓰지 않는다.

### 7.2 Experiment ladder

1. MV-1 cheap probes:
   reusable direct-z와 same-snapshot factor를 한 번 만들고 single-layer
   finite-step, uniform, static, analytic arm을 paired branch한다.
2. Oracle opportunity가 양 model에서 없으면 즉시 routing을 kill한다.
3. Opportunity는 있으나 locked controller가 회수하지 못하면 현재 controller를
   kill하거나 static routing으로 pivot한다. 더 유연한 learner를 사후 도입해
   같은 confirmatory split을 재사용하지 않는다.
4. Achievable allocation gain이 살아야 MV-2 fixed/refreshed branches를 연다.
5. Refresh가 살아야 MV-3 full frontier, layer-cap compensation,
   Gauss–Seidel small-step를 연다.
6. Matched-progress displacement가 살아야 MV-4 100-edit two-order microstream을
   연다.
7. MEMIT ladder가 생존하기 전 AlphaEdit projector lane, external baseline,
   long stream은 제출하지 않는다.

Actionable precondition에서 탈락한 event는 ITD row를 남기되, 사전에 정한
규칙에 따라 expensive downstream branch를 생략할 수 있다. 특정 arm outcome이
나빠 보인다는 이유로 그 event의 다른 branch를 생략하는 것은 금지한다.

### 7.3 Compute-adjusted report

각 forecast 표에는 effect 옆에 다음을 둔다.

- extra forward/backward/solve NFE
- accepted/rejected macro-round
- seconds/edit와 GPU-hours
- GPU/host-memory peak
- matched-NFE contrast

ODE 명칭은 unmatched compute에서만 gain이 있거나 평균 accepted round가
1~2에 그치며 fixed-direction과 같으면 유지하지 않는다. 임의의
effect-minus-latency scalar를 만들지 않고 Pareto dominance와 matched-NFE
결과로 판단한다.

## 8. 최소 kill / go / pivot 규칙

### Layer allocation

- **Kill**: 두 model 모두에서 near-tie-adjusted oracle upper bound가 replay
  envelope를 넘지 않음
- **Current controller kill**: oracle opportunity는 있으나 held-out
  `G_alloc`이 두 model 모두 noise-equivalent
- **Proceed**: model별 achievable gain이 positive이고 untouched에서 같은 sign
- **Architecture-conditional**: 한 model만 생존

### Dynamic refresh

- **ODE kill**: 두 model의 `G_direction-refresh`가 noise-equivalent이거나
  matched-NFE에서 사라짐
- **Pivot**: `G_coefficient-refresh`만 positive이면 fixed-direction dynamic
  coefficient controller
- **Proceed**: refreshed-direction gain, 최소 두 accepted refresh, step-size
  refinement가 함께 생존

### Capacity/displacement

- **Rerouting kill**: global alpha/static과 restricted frontier가 동등하거나
  reach-rate가 악화
- **Capacity claim kill**: displacement는 줄지만 MV-4 retention intervention이
  없음
- **Pivot**: progress/refresh gain은 있으나 displacement/retention이 없으면
  capacity-free state-adaptive scheduler
- **Proceed**: matched-progress state/path displacement와 short-window retention
  intervention이 양 model에서 같은 방향

Canonical baseline damage가 short MV-4 horizon에서 replay/order noise를 넘지
않으면 capacity effect를 바로 kill하지 않고
`inconclusive_at_this_horizon`으로 기록한다. 더 긴 stream 제출은 그때의
추정 effect size와 compute budget을 보고 별도 결정한다.

## 9. 기대효과 보고서 형식

실험 뒤 model별로 아래 한 줄을 채운다. 관측 전에는 빈칸을 수치로 채우지
않는다.

| model | oracle gap | achievable allocation | direction refresh | total refreshed-vs-static | displacement frontier | retention | extra NFE/time | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Llama3-8B-Instruct | TBD | TBD | TBD | TBD | TBD | MV-4 전 금지 | TBD | TBD |
| Qwen2.5-7B-Instruct | TBD | TBD | TBD | TBD | TBD | MV-4 전 금지 | TBD | TBD |
| equal-weight pooled | TBD | TBD | TBD | TBD | TBD | MV-4 전 금지 | 해당 없음 | model별 판정 대체 불가 |

각 cell에는 point estimate 하나만 쓰지 말고
`mean [paired CI] / median / sign fraction / denominator`를 연결한다.
Oracle은 `upper bound`, achievable은 `held-out estimate`, refresh는
`incremental`, displacement는 `mechanism forecast`, retention은
`MV-4 intervention`이라고 명시한다.

## 10. 최종 red 판정

기대효과 예측을 Motivation 단계에 포함하는 것은 타당하다. 단, 현재 허용되는
예측은 다음 순서뿐이다.

```text
MV-1: routing opportunity ceiling
   -> MV-1: held-out achievable allocation gain
   -> MV-2: refresh incremental gain
   -> MV-3: matched-progress displacement forecast
   -> MV-4: direct retention effect
```

이 순서를 지키고 양 model을 같은 wave에서 실행하면 작은 100-case budget
안에서 연구가 회수할 수 있는 최대 여지, 실제 controller가 회수하는 비율,
ODE refresh의 추가 가치, retention으로 이어질 가능성을 구분해 예측할 수
있다. 반대로 oracle, achievable, refresh, retention을 하나의 선제적
“예상 개선율”로 합치는 분석은 selection bias와 overclaim 때문에
`BLOCK`한다.
