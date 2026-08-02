# Session 01 BF-share magnitude-only — Llama c3_v1 GH 분석

- 날짜: 2026-08-02
- 모델: `llama3-8b-inst`
- 방법명: **ODE-Edit**
- policy: `bf-common-frontier-share-radial-exact-quarter-k4-v1`
- 판정: **low-update implementation cause confirmed; directional signal, not native superiority**
- claim boundary: same-policy 4-edit Motivation cause-isolation only

## 결론

c1과 같은 BF layer allocation을 유지하고 update norm만 exact `D/4 × 4`로 보정하자
Llama current efficacy가 크게 회복했다. MEMIT은 c1 `-4.032392`에서
`-0.361365`로 `+3.671027`, Alpha-history는 `-2.495123`에서 `-0.563717`로
`+1.931406` 개선됐다. 이는 c1 Llama harm의 주요 원인이 BF의 과학적 실패가 아니라
QP coefficient를 실제 update 크기로도 사용한 under-update 구현이었다는 강한 증거다.

하지만 native 대비 current gap은 두 family 모두 남아 있고 prior retention도
MEMIT `-0.292871`, Alpha `-0.782910`이다. Llama에서는 구현 원인은 확인했지만
method superiority나 preservation guarantee는 아직 성립하지 않는다.

## 네 범주

### Proposal에서 온 내용

- current state마다 direction과 layer velocity를 다시 계산한다.
- cumulative capacity를 기준으로 layer별 update share를 다르게 둔다.
- MEMIT과 AlphaEdit history/projector에 같은 controller를 적용한다.

### Repo/protocol에서 확인한 사실

- c3 Llama controller/evaluator 8개 branch-cell이 모두 terminal/pass다.
- QP family별 4 edits/16 hops가 exact total path `D`를 만족했다.
- 첫 edit/round의 c3 allocation coefficients는 c1 coefficients와 두 family 모두
  원소별 max abs diff `0`이고 native distance도 동일했다.
- allocation QP cap 위반과 trust failure는 두 family 모두 0이다.
- EasyEdit와 precomputed covariance/projector/Wikipedia artifact는 read-only였다.

### GH 추정

- c1→c3 회복량이 c2→c3 회복량보다도 크므로, c2의 changed cap/share보다 c1 BF
  relative weighting + independent magnitude가 Llama에 더 적합하다.
- 남은 native gap은 under-update만으로 설명되지 않으며 direction refresh와 BF share,
  fixed four-hop terminal의 method-stage 문제다.

### 사용자 확인 필요

- 없음. Motivation에서는 이 결과로 닫고 추가 model-specific retune을 하지 않는다.

## c3 QP-minus-native 결과

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-0.361365` | `-0.563717` |
| final all-edit utility delta | `-0.360966` | `-0.551857` |
| final prior-retention delta | `-0.292871` | `-0.782910` |
| retention AUC delta | `-0.264297` | `-0.540969` |
| neighborhood KL reduction | `-0.014038` | `-0.036208` |
| generation KL reduction | `+0.156987` | `+0.275865` |
| target-true NLL drift reduction | `-0.024014` | `+0.782261` |
| capacity reduction | `+0.000009` | `+0.000150` |
| max-layer-share reduction | `+0.288449` | `+0.279204` |
| layer-Gini reduction | `+0.345577` | `+0.345291` |
| cumulative Frobenius reduction | `+0.062477` | `+0.149602` |

Neighborhood KL은 두 family에서 native보다 소폭 나쁘므로 broad locality claim은
금지한다. 반면 capacity와 layer concentration은 efficacy가 크게 회복된 상태에서도
공통으로 감소했다.

## 구현 수정 효과

| Family | c1 | c2 | c3 | c3-c1 | c3-c2 |
|---|---:|---:|---:|---:|---:|
| MEMIT current | `-4.032392` | `-1.369350` | `-0.361365` | `+3.671027` | `+1.007986` |
| Alpha current | `-2.495123` | `-1.137219` | `-0.563717` | `+1.931406` | `+0.573502` |
| MEMIT prior retention | `-3.862633` | `-1.477975` | `-0.292871` | `+3.569762` | `+1.185104` |
| Alpha prior retention | `-2.708148` | `-1.531226` | `-0.782910` | `+1.925237` | `+0.748316` |

c1 native gap 중 MEMIT 약 `91.0%`, Alpha 약 `77.4%`를 회복했다. 네 개별 edit도
두 family 모두 c1보다 개선됐다. Family-average current recovery는 `+2.801217`이다.

## Magnitude integrity와 compute

- MEMIT radial scale: mean `1.769×`, max `2.970×`.
- Alpha radial scale: mean `1.675×`, max `2.706×`.
- max path/hop relative error: `1.802e-16`.
- max share error: `2.220e-16`; max radial identity error `1.735e-18`.
- QP/native controller wall ratio: MEMIT `6.93×`, Alpha `4.83×`.
- analysis SHA-256:
  `fdf52b7c5f1db78d1a12e8092988816bccbadffdab64d9842f401673ab43765f`.

Terra Ultra Llama analyst는 runtime metadata를 검증하지 못해 지정 파일을 읽지 않고
`BLOCK` 종료했다. 독립 분석으로 세지 않으며 이 문서는 GH fallback이다.

## 판정

Llama low-update implementation cause는 **confirmed**다. 하지만 strong native
non-collapse floor `-0.10`은 두 family 모두 실패한다. Motivation-sized directional
signal로만 보존하고, native efficacy/retention을 동시에 회복하는 Method Session 전에는
large-scale/lifelong 또는 superiority claim을 열지 않는다.
