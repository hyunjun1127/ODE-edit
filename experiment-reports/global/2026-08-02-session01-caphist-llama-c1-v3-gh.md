# Session 01 capacity/history — Llama c1_v3 GH 분석

- 날짜: 2026-08-02
- 모델: `llama3-8b-inst`
- 방법명: **ODE-Edit**
- policy: `capacity-qp-history-k4-native-progress-v2`
- 판정: **technical pass; c0보다 회복했으나 두 family 모두 harm signal**
- claim boundary: same-policy 4-edit Motivation diagnostic only

## 결론

c1은 c0의 고정 `75%` progress request, 최대 `3D/4`, exact-top1 조기종료를
제거했다. 네 편집 모두 매 round에서 남은 ordered-native rewrite utility 전체를
요청했고, MEMIT과 Alpha-history 모두 4 rounds를 실제 적용했다. 따라서 이번 결과는
c0와 달리 구현 계약을 통과한 과학적 결과다.

Llama 손실은 c0보다 줄었지만 충분하지 않았다. QP-minus-native current-edit utility는
MEMIT `-4.032392`, Alpha-history `-2.495123`으로 lenient floor `-0.10`을 크게
실패했다. KL, capacity, layer concentration은 낮아졌으나 이는 efficacy와 retention을
보존한 개선이 아니다. Llama에서는 현재 controller를 계속 밀 근거가 없다.

## 네 범주

### Proposal에서 온 내용

- current state마다 layer-synchronous proposal direction을 다시 계산한다.
- 남은 rewrite progress를 만족시키면서 cumulative covariance-capacity cost가 낮은
  realization을 선택한다.
- MEMIT과 AlphaEdit projector/history actuator에 같은 controller가 작동해야 한다.

### Repo/protocol에서 확인한 사실

- EasyEdit source는 수정하지 않았고 ODE-edit hook만 사용했다.
- precomputed Wikipedia covariance와 Alpha projector를 read-only로 재사용했다.
- controller/evaluator 네 branch가 각각 4/4 terminal, `all_pass=true`다.
- controller는 evaluation field를 보지 않았고 evaluator technical false는 0개다.
- native/QP 양쪽 Alpha history는 accepted edit마다 정확히 한 번 append됐다.

### GH 추정

- c1이 c0보다 utility를 회복한 것은 old under-edit bug를 고친 효과와 일치한다.
- 남은 Llama harm는 fixed fractional request가 아니라, capacity/preservation 제약형
  QP가 총거리 상한을 다 쓰지 못하면서 native endpoint utility에 도달하지 못한
  trade-off로 보는 것이 가장 단순하다.
- hard overload observation이 0이므로 Llama에서 overloaded-layer rerouting은
  검증되지 않았다.

### 사용자 확인 필요

- 없음. 사용자가 구현 수정과 재실험을 명시적으로 지시했다.
- 이 결과로 model-specific Llama rescue나 threshold retune을 열지 않는다.

## c1 QP-minus-native 결과

양수인 `*_reduction`은 QP가 native보다 해당 cost/proxy를 줄였다는 뜻이다.

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-4.032392` | `-2.495123` |
| final all-edit utility delta | `-4.041824` | `-2.492299` |
| final prior-retention delta | `-3.862633` | `-2.708148` |
| retention AUC delta | `-2.585957` | `-2.029810` |
| neighborhood KL reduction | `+0.026321` | `+0.055909` |
| generation KL reduction | `+0.972480` | `+1.027351` |
| target-true NLL drift reduction | `+3.029000` | `+2.890667` |
| capacity reduction | `+0.000087` | `+0.000660` |
| max-layer-share reduction | `+0.296844` | `+0.284283` |
| layer-Gini reduction | `+0.368359` | `+0.355861` |
| cumulative Frobenius reduction | `+0.542119` | `+0.650509` |

현재 utility의 절대 평균은 MEMIT native `4.014310` 대 QP `-0.018082`,
Alpha-history native `6.183269` 대 QP `3.688146`이다. First-hit도 MEMIT native/QP
`3/4` 대 `2/4`, Alpha native/QP `4/4` 대 `3/4`다.

## 구현 수정 효과와 realized path

| Family | c0 path/native | c1 path/native | c0 current delta | c1 current delta |
|---|---:|---:|---:|---:|
| MEMIT | `0.454052` | `0.671819` | `-6.428714` | `-4.032392` |
| Alpha-history | `0.457809` | `0.686245` | `-5.663860` | `-2.495123` |

c1은 family마다 accepted round `16`, proposal build `20`, reject `0`이다. 모든
편집이 남은 native gap 전체를 요청했지만 native reference match는 `0/4`이고,
네 편집 모두 `budget_exhausted=true`다. Realized path는 native의 약 `66.4--69.6%`
범위다. Nominal 총 trust cap은 `D`지만 exact-length를 강제한 실험은 아니다.

QP/native wall ratio는 MEMIT `6.34×`, Alpha-history `4.65×`다. 더 많은 계산을
사용하고도 Llama efficacy/retention frontier를 회복하지 못했다.

## Agent와 artifact

Terra Ultra Llama analyst를 c1 Llama JSON/feature에만 격리 호출했으나 runtime
metadata를 검증할 수 없어 파일을 읽지 않고 종료했다. 독립 agent analysis로 세지
않으며 이 문서는 GH fallback이다.

- analysis SHA-256:
  `42b63463198c53c39880f7bef1d11d3cd93c9f1fed1a2ea6404b469c7469b78b`
- raw root: ignored
  `local/results/raw/session01_motivation/caphist_*_llama_c1_v3/`

## 최종 판정

- MEMIT: `CAPACITY_QP_HARM_SIGNAL`.
- Alpha-history: `CAPACITY_QP_HARM_SIGNAL`.
- model: `CAPACITY_HISTORY_MODEL_NO_SIGNAL`.

c1은 old implementation bug를 고쳤고 c0보다 Llama를 개선했지만, Llama Motivation
gate를 열 정도는 아니다. 이 4-edit 결과로 lifelong 또는 broad method-superiority
claim을 만들지 않는다.
