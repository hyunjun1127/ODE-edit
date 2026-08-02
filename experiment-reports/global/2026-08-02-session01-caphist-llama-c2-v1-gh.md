# Session 01 capacity/share exact-quarter — Llama c2_v1 GH 분석

- 날짜: 2026-08-02
- 모델: `llama3-8b-inst`
- 방법명: **ODE-Edit**
- policy: `capacity-share-history-exact-quarter-k4-v3`
- 판정: **technical pass; c1 low-update confound를 크게 회복했으나 native 대비 harm은 잔존**
- claim boundary: same-policy 4-edit Motivation 원인분리 diagnostic only

## 결론

c2는 QP coefficient를 update 크기로 직접 쓰지 않고 layer별 상대 share로만 사용한 뒤,
매 hop의 전체 C-distance를 정확히 `D/4`로 맞췄다. Llama current-edit utility는 c1 대비
MEMIT `+2.663042`, Alpha-history `+1.357904` 회복했다. 따라서 c1의 낮은 실제 update가
Llama 악화의 큰 구현 원인이었다는 사용자 가설은 강하게 지지된다.

다만 c2도 native 대비 current utility가 MEMIT `-1.369350`, Alpha-history
`-1.137219`이고 prior retention도 각각 `-1.477975/-1.531226`이다. Exact magnitude가
Llama를 상당히 살렸지만 현재 layer routing이 native endpoint를 이기거나 보존하지는
못했다.

## 네 범주

### Proposal에서 온 내용

- current state에서 direction과 layer velocity를 다시 계산한다.
- cumulative capacity가 큰 layer의 update share를 줄이고 여유 layer로 재배분한다.
- 같은 controller가 MEMIT과 AlphaEdit history/projector actuator에 작동해야 한다.

### Repo/protocol에서 확인한 사실

- MEMIT/Alpha controller와 evaluator가 모두 terminal, `all_pass=true`다.
- 각 QP edit는 4개 exact-quarter hop을 적용했고 총 path는 native C-distance `D`다.
- 32 Llama QP hop에서 applied share L2 norm과 hop/path distance 오차는 numerical zero
  수준이며 barrier violation은 0이다.
- EasyEdit는 수정하지 않았고 covariance/projector/Wikipedia artifact를 precomputed
  read-only로 재사용했다.

### GH 추정

- c1→c2의 큰 회복은 direction refresh의 과학적 실패보다 global update shrink가 먼저
  결과를 지배했다는 설명과 일치한다.
- 남은 native 대비 손실은 exact magnitude 자체의 실패로 확정할 수 없다. c2가 c1의
  common-frontier share 규칙까지 바꿨기 때문이다.

### 사용자 확인 필요

- 없음. 사용자는 구현 문제를 우선 가설로 두고 재구현·원인분리 실험을 지시했다.

## c2 QP-minus-native 결과

양수인 `*_reduction`은 QP가 native보다 해당 drift/cost를 낮췄다는 뜻이다.

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-1.369350` | `-1.137219` |
| final all-edit utility delta | `-1.371123` | `-1.120088` |
| final prior-retention delta | `-1.477975` | `-1.531226` |
| retention AUC delta | `-1.144366` | `-1.077396` |
| neighborhood KL reduction | `+0.004894` | `-0.003902` |
| generation KL reduction | `+0.441412` | `+0.442004` |
| target-true NLL drift reduction | `+1.245907` | `+1.562021` |
| capacity reduction | `+0.000041` | `+0.000380` |
| max-layer-share reduction | `+0.273536` | `+0.247129` |
| layer-Gini reduction | `+0.315877` | `+0.285053` |
| cumulative Frobenius reduction | `+0.237710` | `+0.371691` |

## c1에서의 회복

| Family | c1 current delta | c2 current delta | 회복량 |
|---|---:|---:|---:|
| MEMIT | `-4.032392` | `-1.369350` | `+2.663042` |
| Alpha-history | `-2.495123` | `-1.137219` | `+1.357904` |

Prior retention도 MEMIT `-3.862633→-1.477975`, Alpha-history
`-2.708148→-1.531226`으로 회복했다. 이는 작은 update가 efficacy뿐 아니라 최종
sequential state 자체를 크게 바꿨다는 증거다.

## Compute와 agent

- proposal build: QP family별 `20`, native family별 `4`.
- accepted round: QP family별 `16`, native family별 `4`.
- QP/native controller wall ratio: MEMIT 약 `6.03×`, Alpha-history 약 `4.47×`.
- analysis SHA-256:
  `8053ac2c753ed6bd7059709b478cf0a7e8558dcbdb3820d57b328e3cae731e18`.

별도 Terra Ultra Llama analyst는 runtime metadata에서 지정 profile을 검증하지 못해
파일을 열지 않고 `BLOCK` 종료했다. 독립 분석으로 세지 않으며 이 문서는 GH fallback이다.

## 판정

Llama에서는 **low-update implementation confound가 확인**됐다. 그러나 c2가 layer-share
정의도 함께 바꿨으므로, 남은 손실을 BF routing 또는 exact magnitude 중 하나에
귀속하지 않는다. 기존 BF share를 고정한 magnitude-only 대조가 최종 원인분리 조건이다.
