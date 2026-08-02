# Session 01 capacity/share exact-quarter — Qwen c2_v1 GH 분석

- 날짜: 2026-08-02
- 모델: `qwen2.5-7b-inst`
- 방법명: **ODE-Edit**
- policy: `capacity-share-history-exact-quarter-k4-v3`
- 판정: **technical pass; c1 대비 mixed-negative, magnitude effect 식별 불가**
- claim boundary: same-policy 4-edit Motivation 원인분리 diagnostic only

## 결론

Qwen에도 Llama와 byte-identical한 exact `D/4 × 4` controller를 적용했다. Technical
계약은 모두 통과했지만 current-edit utility는 c1 대비 MEMIT `-0.399352`,
Alpha-history `-0.430445` 악화했다. Native 대비로도 `-0.797188/-0.400738`이다.

이 결과는 “update를 크게 하면 Qwen이 나빠진다”는 증거가 아니다. c2는 magnitude뿐
아니라 common-frontier cap과 overload 처리, 따라서 실제 layer share까지 바꿨다.
첫 edit의 c1/c2 normalized share cosine도 Alpha-history에서 round별
`0.4222/0.8679/0.2107/0.7061`로 크게 달랐다. Qwen은 magnitude와 routing이 묶인
bundled intervention에 negative였다고만 판정한다.

## 네 범주

### Proposal에서 온 내용

- finite edit를 여러 current-state waypoint로 나누고 direction/layer allocation을
  다시 계산한다.
- capacity-aware layer share가 efficacy를 보존하면서 cost를 낮춰야 한다.
- 모델별 다른 controller나 rescue는 허용하지 않는다.

### Repo/protocol에서 확인한 사실

- MEMIT/Alpha controller와 evaluator가 모두 terminal, `all_pass=true`다.
- QP edit마다 4개 exact-quarter hop, total path `D`, applied share norm 1을 만족했다.
- Qwen은 MEMIT 8개, Alpha-history 4개의 positive overload observation을 모두
  suppression했고 barrier violation은 0이다.
- Llama와 같은 case order, seed, layers, policy hash, code를 사용했다.

### GH 추정

- c2의 Qwen 악화는 exact magnitude 단독 효과보다 share 변경과의 interaction일 수 있다.
- Qwen c1 Alpha의 작은 positive가 사라진 것도 magnitude와 share를 분리하지 않으면
  해석할 수 없다.

### 사용자 확인 필요

- 없음. 모델별 threshold/K/share rescue는 사용자 원칙상 계속 금지한다.

## c2 QP-minus-native 결과

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-0.797188` | `-0.400738` |
| final all-edit utility delta | `-0.795757` | `-0.416153` |
| final prior-retention delta | `-0.645751` | `-0.363292` |
| retention AUC delta | `-1.001022` | `-0.906707` |
| neighborhood KL reduction | `+0.084538` | `+0.112947` |
| generation KL reduction | `+0.124315` | `-0.173040` |
| target-true NLL drift reduction | `+1.424733` | `+1.251480` |
| capacity reduction | `+0.000803` | `+0.001816` |
| max-layer-share reduction | `+0.032280` | `+0.127529` |
| layer-Gini reduction | `+0.027861` | `+0.139618` |
| cumulative Frobenius reduction | `-16.132178` | `+1.392715` |

## c1 대비 변화

| Family | c1 current delta | c2 current delta | c2-c1 |
|---|---:|---:|---:|
| MEMIT | `-0.397837` | `-0.797188` | `-0.399352` |
| Alpha-history | `+0.029707` | `-0.400738` | `-0.430445` |

Prior retention은 MEMIT `-0.579252→-0.645751`로 소폭 악화했고, Alpha-history는
`-0.431721→-0.363292`로 소폭 회복했다. 단일 축의 일관된 worsening은 아니다.

## Compute와 agent

- QP/native controller wall ratio: MEMIT 약 `6.01×`, Alpha-history 약 `4.68×`.
- analysis SHA-256:
  `60086eeb7be8882f636e0fdf7630160ff9a3ccdfa7fdc3ccc8c7140842791a7f`.

별도 Terra Ultra Qwen analyst는 runtime metadata를 검증하지 못해 파일을 열지 않고
`BLOCK` 종료했다. 독립 분석으로 세지 않으며 이 문서는 GH fallback이다.

## 판정

Qwen c2는 scientific harm으로 닫지 않는다. 정확한 update 크기와 바뀐 layer share가
동시에 개입한 식별 실패다. 기존 BF share를 보존한 magnitude-only 대조에서 Llama와
같은 policy로 다시 판정한다.
