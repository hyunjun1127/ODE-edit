# Session 01 capacity/history pair c0_v3 — superseded under-edit diagnostic

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- analyzer verdict: **`CAPACITY_HISTORY_HARM_SIGNAL`**
- technical validity: **pass**
- scientific status: **implementation-contract failure; c1로 superseded**
- claim boundary: 4-edit same-policy diagnostic only

## 결론 먼저

Cumulative-capacity QP는 MEMIT과 canonical-history AlphaEdit 양쪽에서 더 작은
capacity, Frobenius, layer concentration과 KL drift를 만들었다. 그러나 두 모델·두
family 모두 current edit utility와 prior retention이 크게 악화해 매우 느슨한
non-collapse floor `-0.10`조차 통과하지 못했다. 4 edits 동안 overload observation과
reroute round도 모두 0이었다.

따라서 이 결과는 “capacity-aware routing이 잘 작동했다”거나 “capacity routing이
실패했다”는 판정이 아니라 **native보다 훨씬 약하게 편집한 under-edit 구현 진단**이다.
Canonical Alpha history도 strength mismatch를 제거하지 못했다. 사용자 지시에 따라
이 report는 c1 matched-quarter 재실험의 RCA evidence로 보존하며 final method kill로
사용하지 않는다.

## 네 범주

### Proposal에서 온 내용

- 동일 direct-z의 여러 layer realization 중 cumulative capacity marginal cost가
  낮은 곳으로 progress를 재배분할 수 있다는 가설.
- 많이 찬 layer의 positive write를 막고 다른 feasible layer로 넘기면서 rewrite
  progress를 유지해야 한다는 controller 요구.
- MEMIT에서 시작하고 AlphaEdit null-space/history proposal로 transfer하되,
  short diagnostic이 살아남기 전 lifelong claim을 열지 않는다는 단계 원칙.

### Repo/protocol에서 확인한 사실

- 두 모델은 case/order/seed/layer/QP/trust/first-hit policy가 동일하다.
- EasyEdit source와 global `cache_c`는 수정하지 않았다. Existing covariance,
  projector와 Wikipedia artifact만 read-only로 사용했다.
- Controller 8개와 evaluator 8개가 각각 4/4 checkpoint로 terminal/pass다.
- Alpha native/QP는 accepted edit 뒤 current key를 정확히 한 번 append했고
  history count가 edit index `1--4`와 일치한다.
- Pair analyzer는 technical pass지만 두 family 모두 non-collapse 실패로
  `CAPACITY_HISTORY_HARM_SIGNAL`을 반환했다.

### GH 추정

- 현재 harm의 가장 단순한 설명은 QP solver failure가 아니라 objective mismatch다.
  Step마다 local gain은 양수였으나 native와 같은 margin/progress를 맞추는 constraint가
  없었고, first-hit는 낮은-margin endpoint도 성공으로 받았다.
- Capacity/KL/Frobenius 감소는 matched-efficacy frontier가 아니므로 method benefit으로
  해석할 수 없다.
- Overload가 0회라 hard barrier/rerouting hypothesis는 이 scale에서 검증되지 않았다.
  Vacuous barrier pass를 empirical mechanism pass로 세면 안 된다.

### 사용자 확인 필요

- 없음. 사용자가 c0 구현을 잘못된 것으로 보고 c1 재구현·재실험을 지시했다.
- Direct-z direction×share `M-CLOSE-DZ-2x2`는 설계만 잠겼고 실행 권한은 없다.
  실제 실행은 별도 사용자 지시가 필요하다.
- Lifelong/100+ edit, model별 tuning, 새 threshold/case sweep는 현재 근거로 제출하지
  않는다.

## Pair metric

모든 수치는 QP minus native다. Reduction 축의 양수는 QP cost/proxy가 더 작다는
뜻이다.

| Model | Family | current utility | final all-edit utility | retention AUC | capacity reduction | neighborhood KL reduction | generation KL reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama | MEMIT | `-6.428714` | `-6.443999` | `-4.944429` | `+0.000127` | `+0.064093` | `+2.033320` |
| Llama | Alpha-history | `-5.663860` | `-5.659343` | `-4.220787` | `+0.000949` | `+0.155460` | `+1.797439` |
| Qwen | MEMIT | `-3.710800` | `-3.708018` | `-3.680108` | `+0.000938` | `+0.283246` | `+1.744179` |
| Qwen | Alpha-history | `-3.321125` | `-3.316262` | `-3.363195` | `+0.002020` | `+0.329390` | `+1.525872` |

두 모델에 공통으로 양수인 축은 `capacity_reduction`,
`cumulative_frobenius_reduction`, `generation_kl_reduction`,
`layer_gini_reduction`, `max_layer_share_reduction`,
`neighborhood_kl_reduction`, `target_true_nll_drift_reduction`이다. 그러나 네
cell 모두 current non-collapse가 false이므로 common-axis rule에 들어가기 전에
gate가 닫힌다.

## Routing mechanism 판정

| Model | Family | overload observations | suppressed | reroute rounds | max barrier violation |
|---|---|---:|---:|---:|---:|
| Llama | MEMIT | `0` | `0` | `0` | `1.69e-21` |
| Llama | Alpha-history | `0` | `0` | `0` | `3.39e-21` |
| Qwen | MEMIT | `0` | `0` | `0` | `0` |
| Qwen | Alpha-history | `0` | `0` | `0` | `0` |

`all_observed_positive_overloads_suppressed=true`는 네 cell 모두 분모가 0인
vacuous truth다. Soft capacity objective가 allocation/concentration을 바꾸기는
했지만 “이미 찬 layer의 write를 억제해 다른 layer로 넘겼다”는 proposal 핵심
mechanism은 이 run에서 한 번도 관측되지 않았다.

## Compute와 실용성

- Native는 model/family당 proposal 4회, accepted round 4회다.
- Llama QP는 family당 proposal 15회, accepted round 11회; wall ratio는 MEMIT
  `4.30×`, Alpha `3.20×`다.
- Qwen QP는 family당 proposal 11회, accepted round 7회; wall ratio는 MEMIT
  `2.88×`, Alpha `2.24×`다.
- Reject는 모든 QP cell에서 0회다. Trust-ratio가 local worsening은 막았지만
  native-relative under-edit를 막지는 못했다.

추가 compute를 지불하고도 edit/retention frontier가 악화했으므로 NAS/ENCORE나
history-on native보다 복잡한 controller를 정당화하지 못한다.

## 실행·복구 ledger

| Job | 상태 | scientific 사용 여부 |
|---|---|---|
| `15813` | Qwen MEMIT worker technical exit; evaluator 전 | 제외 |
| `15817` | five-action probe integration mismatch; QP/eval 전 | 제외 |
| `15819` | controller 8개 pass; docs-only HEAD change로 evaluator Git gate fail | controller action만 canonical |
| `15823` | W0 baseline 뒤 metadata key sanitizer collision; edit replay 전 | 제외; 빈 dirs 보존 |
| `15824` | evaluator-only recovery `COMPLETED 0:0`, 6m42s | terminal evaluation evidence |

Job `15824`는 server1에서 4 GPU, 32 CPU, 260000M을 요청했고 active cap `4`를
넘지 않았다. Llama/Qwen은 네 1-GPU worker로 동시에 시작했다. Recovery는
controller action을 재실행하지 않았고, metadata key `evaluation`을
`metric_protocol`로 바꾼 tracked entrypoint만 사용했다. Sanitizer는 완화하지
않았으며 wrapper/evaluator hash와 no-rerun 표식은 evaluator policy hash에 묶였다.

## Artifact와 재현성

- Llama analysis SHA-256:
  `4931235b2dcb224fb93bef739cc408252ea343ea122a4a67aa72b7f85e33c0c0`.
- Qwen analysis SHA-256:
  `70cb345924da0cfac875c335fa42a4af31766534ef034c48b7302e747f21ea5a`.
- Pair analysis SHA-256:
  `d962c0fb783dffe10268d4036d0c0d8e82ac605d29f73587263b5291325a9aad`.
- 새 evaluator/combined/pair 29 files의 aggregate SHA-256:
  `cbf614289b26aba66194b29e2536d859b7e22cb8dbb719087afdcb324e53a934`.
- 8 evaluator source/locked copy가 recursive byte-identical이고, summary artifact
  hash 16/16이 실제 file hash와 일치한다.
- Combined payload는 네 isolated branch stream의 exact concatenation이다.
- 독립 local re-analysis가 model/pair JSON과 byte-identical했다.
- Compact artifact와 gate에 NFE field는 없다.

Terra Ultra Llama/Qwen/pair analyst 세 개를 각각 관련 JSON 하나에만 격리했으나
runtime metadata를 검증할 수 없어 모두 무열람 종료했다. 독립 analysis pass는
0건이며 GH fallback 결과를 agent 결과로 표기하지 않는다.

## Related-work와 novelty 경계

Canonical Alpha history가 이미 strong accumulated-constraint baseline이라는 점과,
OTE–SE alignment/NAS/ENCORE의 단순화 반론이 더 강해졌다. Current QP는 동일 efficacy를
유지하지 못했으므로 cumulative constraint, null space, dynamic layer selection 또는
capacity metric을 ODE-Edit novelty/benefit으로 주장할 수 없다.

## Supersession decision

- **Kill:** policy `capacity-qp-history-k3-trust-d4-v1`의 current progress/first-hit
  formulation.
- **보류:** overloaded-layer rerouting, capacity/preservation benefit, method-superiority
  claim. c0로 긍정·부정 어느 쪽도 확정하지 않는다.
- **보존:** exact cumulative-cost algebra, canonical Alpha history-on baseline, 두 모델
  raw result, under-edit RCA.
- **허용된 단일 후속:** 같은 case/model/family에 model-common
  `capacity-qp-history-k4-native-progress-v2`를 적용한 c1 matched-quarter rerun.
- **계속 금지:** model별 branch, threshold/case retune, edit-count 확대, lifelong claim.

Capacity/history Motivation은 c1 결과 전까지 open이다. Direct-z 임시 세션 결과는
이 report나 c1 gate에 합치지 않는다.
