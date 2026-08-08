# ODE-BF Main-Table Promotion and Compute-Reduction Future Work

- 작성 시각: **2026-08-08 21:51:19 KST (+09:00)**
- GH task: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 상태: **FUTURE_WORK; DESIGN_ONLY; NO_EXECUTION_AUTHORITY**
- 현재 method 기준: **Cold-FR-E8, full residual, target-new NLL, no arbitrary H/P hard budget**

이 문서는 현재 B10 gate가 끝난 뒤 main table로 넘어가는 순서와, fixed-E8의
계산량을 줄이기 위한 후속 설계를 기록한다. 기존 2026-08-03 compute spec은 당시의
adaptive/first-hit controller를 전제로 하므로, 현재 Cold-FR-E8 경로에는 이 문서를 우선해
해석한다.

## 1. 현재 B10의 역할

현재 실행 중인 R10은 과거 warm-up에서 사용한 ordered B10을 재사용한
`REUSED_WARMUP_SEAL_CAUSAL_REGRESSION`이다. 따라서 다음을 확인하는 데는 적합하다.

- common cold-coordinate 수정이 양 모델에서 정상적으로 작동하는지
- 정확히 8회의 Euler transition으로 `tau=1`에 도달하는지
- Neutral과 Soft가 technical certificate와 transaction gate를 통과하는지
- Native 대비 efficacy/generalization/locality, absolute NLL과 routing trajectory가
  어느 방향으로 움직이는지

그러나 동일 B10은 이전 결과를 보고 선택된 표본이므로, 그 결과만으로 unseen main-table
성능을 승격하지 않는다. R10은 causal/engineering gate이고, scientific promotion에는 같은
계약으로 새롭게 봉인한 unseen B10 확인이 한 번 더 필요하다.

## 2. Main-table 승격 순서

### G0. 현재 R10 terminal review

사용자가 명시적으로 결과 확인을 지시한 뒤에만 pending jobs의 terminal artifact를 읽는다.
다음 네 endpoint를 같은 seal에서 비교한다.

1. `W0`
2. `N32 Native`
3. `Cold fixed-E8 Neutral/no-soft/no-hard-budget`
4. `Cold fixed-E8 Soft/no-hard-budget`

R10이 기술적으로 완주하지 못하면 B100으로 넘어가지 않고, 공통 오류만 먼저 고친다.

### G1. Fresh B10 promotion gate

두 모델 모두 동일한 unseen ordered B10, 동일 evaluator/span/context 계약으로 실행한다.
기본 promotion 조건은 다음과 같다.

- genuine joint `edit_batch_size=10`
- `K=8`, `h=1/8`, 정확히 8 accepted transitions, `tau_8=1`
- scientific retry/reject가 없고 최종 transaction/rollback이 exact
- 공식 efficacy, generalization, locality가 각각 matched Native보다 낮지 않음
- count metric은 동일 denominator에서 기본 `delta=0`
- absolute target-new/target-old NLL, Native-relative teacher-KL, update capacity,
  direct-target-to-write realization과 layer-allocation trajectory를 함께 기록

Soft가 Neutral과 efficacy/generalization/locality가 같고 보존 이득도 없다면, main sequential
run에는 Neutral만 남기고 Soft는 ablation으로 내린다. Soft의 작은 proxy 개선만으로 두 arm을
모두 B100에 유지하지 않는다.

### G2. Ordered B100 historical gate

Fresh B10에서 두 모델 모두 matched-Native 수준을 유지하면, 다음 단계는 **100 requests를
10개의 ordered joint B10 transaction으로 처리하는 실험**이다.

| 항목 | 고정값 |
|---|---|
| total requests | 100 |
| edit batch size | 10 |
| sequential batches | 10 |
| within-batch Euler steps | 8 |
| batch pseudo-time | 1 |
| primary arms | Native, selected Ours |
| optional ablation | Soft, 단 G1에서 보존 이득이 확인된 경우만 |

각 arm은 같은 `W0`에서 독립적으로 시작한다. 한 arm 안에서는 model state, edit history,
capacity state를 B10-1부터 B10-10까지 유지한다. 다른 arm의 state/history는 절대 공유하지
않는다.

Historical-H는 B10-1에서 history가 비어 있어 decision-vacuous하다. **B10-2부터 비로소
historical preservation mechanism을 검증할 수 있다.** 따라서 100-sample experiment는
단순 규모 확대가 아니라 historical claim을 처음 직접 측정하는 gate다.

각 batch가 끝날 때 다음을 기록한다.

- 현재 batch efficacy/generalization/locality
- 이전 모든 accepted edit의 cumulative retention과 forgetting count
- historical target-new NLL, ranking margin과 age별 retention
- fixed pretrained-anchor teacher-KL과 locality
- cumulative update capacity, per-layer utilization/Omega와 concentration
- editor GPU seconds, wall seconds, forward/backward/JVP/QP/commit counter
- batch별 peak memory와 amortized cost/edit

Controller는 held-out generalization/locality 결과를 보지 않는다. 평가 결과는 batch action이
freeze된 뒤에만 연다.

### G3. Main table 판정

B100 결과에서 다음을 분리한다.

- **성능:** current efficacy, generalization, locality, retention AUC, final retention
- **안정성:** catastrophic forgetting, Native-success loss, layer collapse, capacity growth
- **비용:** GPU sec/edit, wall sec/edit, model forward/edit, peak memory

Ours가 B10에서는 Native 수준이지만 B100에서 빠르게 붕괴하면 lifelong claim을 열지 않는다.
반대로 B100에서 retention 이득이 있어도 비용이 과도하면 deployable-efficiency claim을 열지
않고 아래 계산량 절감 gate를 먼저 통과한다.

## 3. 계산량이 큰 현재 지점

Persistent model write는 batch endpoint에서 한 번만 수행하도록 줄였지만, inner trajectory의
virtual compute는 여전히 크다.

- batch마다 8회의 state-dependent field build
- 각 field의 target/weight directional derivative
- functional soft routing을 위한 baseline과 layer probe
- Neutral/Soft 두 live arm의 중복 실행
- 다단 QP와 numerical polish
- step별 full evaluator를 열 경우 발생하는 추가 forward
- historical item을 무제한 추가할 경우 증가하는 H replay/geometry

특히 이전 fixed-E8 Soft 설계의 `baseline + 5 layer probes`는 arm당
`8 x 6 = 48` functional-basis endpoints를 만들었다. 이것은 persistent write 횟수와 별개의
주요 overhead다.

## 4. 계산량 절감 설계

### Tier A. Scientific semantics를 바꾸지 않는 최적화

1. **Main live arm 하나만 유지한다.** G1에서 이긴 Neutral 또는 Soft만 B100에 넣고 다른
   arm은 작은 ablation subset에서만 실행한다.
2. **Official evaluator는 Euler step마다 열지 않는다.** Inner step에는 controller NLL,
   progress, routing/fidelity telemetry만 기록하고, efficacy/generalization/locality는 action
   freeze 뒤 batch endpoint에서 한 번 측정한다.
3. **Outer-entry reference를 cache한다.** W0/Native reference logits, target spans,
   pretrained-anchor baseline, key/projector/covariance identity를 같은 arm 안에서 재사용한다.
4. **Context forward와 layer hook을 fuse한다.** 한 controller batch forward/backward에서
   5개 layer의 signed slope와 필요한 activation을 함께 얻는다.
5. **Virtual overlay를 유지한다.** 8개 inner step에서 dense weight copy/commit을 만들지 않고,
   endpoint에서만 한 번 atomic BF16 commit한다.
6. **평가와 compute counter를 분리한다.** Editor cost에 post-freeze held-out evaluator 시간을
   섞지 않고 양쪽을 별도 표로 보고한다.

### Tier B. 동등성 검증이 필요한 구현 최적화

1. **Functional basis를 batched JVP/VJP로 계산한다.** `baseline + L`개의 완전한 endpoint
   forward 대신, baseline 한 번과 layer-direction batched JVP로 동일한 first-order score를
   얻는다. 기존 48-endpoint oracle과 두 모델에서 수치·routing identity를 먼저 검증한다.
2. **QP를 step당 하나로 제한한다.** Neutral은 single minimum-capacity QP만 사용한다.
   Soft가 최종 method가 되면 epigraph를 포함한 single certified minimax QP로 합치고 별도
   stage-2 polish를 제거한다. 목표는 arm당 mathematical QP `<=8`이다.
3. **동일 field의 tensor를 공유한다.** target gradient, full residual, key, Q-factor와
   structural sufficient statistics를 solver/probe/evaluator 경로가 중복 생성하지 않게 한다.
4. **Functional probe invocation 목표를 고정한다.** 정확한 JVP identity가 성립하면 arm당
   full-model functional forward equivalent를 현재 48에서 `<=16`으로 줄이는 것을 1차
   목표로 둔다.

이 tier는 outcome을 보고 근사 오차를 조정하지 않는다. Oracle과 coefficient, score,
endpoint event가 사전 tolerance 안에서 일치하지 않으면 main backend로 승격하지 않는다.

### Tier C. Historical cost를 edit 수와 무관하게 제한

모든 과거 key/prompt를 계속 추가하면 historical structural matrix와 functional replay 비용은
edit 수에 따라 증가한다. B100 이전에 다음 bounded-memory 계약을 봉인한다.

- recent window + deterministic reservoir의 최대 history item 수를 고정
- layer별 projected-key sufficient statistics/Gram을 incremental update
- historical functional replay도 같은 bounded recent/reservoir 표본만 controller에 사용
- terminal scientific audit에서는 전체 active history를 streaming 평가하되 controller에는
  다시 넣지 않음
- fixed reservoir에서 controller geometry 비용은 edit count에 대해 `O(1)`, terminal audit은
  보고용 `O(N_history)`로 분리

Reservoir 크기는 결과를 본 뒤 조절하지 않고 B100 전에 공통값으로 lock한다. 전체 history
audit에서 작은 reservoir가 retention miss를 놓치면 method limitation으로 보고한다.

### Tier D. Method-level 단순화 판정

- Soft가 Neutral보다 official preservation을 개선하지 않으면 Neutral을 final 후보로 둔다.
- refreshed field가 frozen/one-refresh보다 이득이 없다면 8회 rebuild를 유지하지 않는다.
- 이후 별도 attribution에서만 `K=4/8/16`과 refreshed-vs-frozen field를 비교한다.
- B100 primary run의 K나 probe 주기는 결과를 본 뒤 늘리지 않는다.

## 5. 목표 operation ceiling

선택된 Ours arm의 batch당 목표는 다음과 같다.

| 항목 | 목표 |
|---|---:|
| persistent commits | 1 |
| Euler transitions | 8 |
| scientific retries | 0 |
| field builds | 8 |
| mathematical QPs | <=8 |
| live Ours arms in B100 | 1 |
| full-model functional forward equivalents | <=16 |
| official endpoint evaluations | 1 per batch |
| controller history size | fixed bounded reservoir |

이 ceiling을 만족하지 못하면 실제 GPU sec/edit과 retention 이득을 함께 보고하고, main table
확장 전에 어느 component가 지배적인지 profiler로 분리한다.

## 6. 의사결정 요약

| 관찰 | 다음 행동 |
|---|---|
| Reused B10 technical failure | 공통 technical/RCA만 수정; B100 금지 |
| Reused B10 pass, fresh B10 Native floor fail | under-edit/coordinate RCA; B100 금지 |
| Fresh B10에서 양 모델 Native floor pass | ordered B100 historical gate 진행 |
| B100에서 historical retention 이득 없음 | historical claim 제거 또는 method 단순화 |
| Soft가 Neutral보다 보존 이득 없음 | Neutral만 main 후보로 유지 |
| 이득은 있으나 compute ceiling 초과 | batched JVP/single-QP/bounded-history 최적화 후 재평가 |
| B100 retention과 compute frontier 모두 유리 | larger sequential main table/1K gate 검토 |

핵심적으로, **B10 Native-level 확인은 B100으로 넘어가기 위한 필요조건이지만 충분조건은
아니다.** fresh B10 parity 뒤 ordered B100에서 B10-2부터 작동하는 historical mechanism과
비용 frontier를 함께 검증해야 main-table/lifelong claim으로 넘어갈 수 있다.
