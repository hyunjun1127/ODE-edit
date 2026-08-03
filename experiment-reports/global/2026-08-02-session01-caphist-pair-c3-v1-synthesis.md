# Session 01 BF-share magnitude-only c3_v1 — final pair synthesis

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- policy: `bf-common-frontier-share-radial-exact-quarter-k4-v1`
- technical verdict: **pass**
- causal verdict: **`LOW_UPDATE_IMPLEMENTATION_CAUSE_CONFIRMED_CROSS_MODEL`**
- Motivation verdict: **`CLOSED_DIRECTIONAL_POSITIVE; STRONG_METHOD_GATE_FAIL`**
- claim boundary: same-policy 4-edit Motivation signal only

## 결론

사용자의 판단이 맞았다. BF layer weight를 다르게 주면서 그 QP coefficient를 실제
update magnitude로도 사용한 것이 c1 성능 악화의 주요 구현 원인이었다. c1 BF share를
생성하는 allocation algorithm을 복원하고 joint norm만 exact `D/4`로 맞춘 c3에서
current efficacy가 네 model×family cell 모두 c1보다 회복했다. 첫 edit/첫 round
coefficient는 c1과 exact-identical하지만, 후속 share는 달라진 state에서 재계산된다.

| Model | Family | c1 current | c2 current | c3 current | c3-c1 | c3 prior retention |
|---|---|---:|---:|---:|---:|---:|
| Llama | MEMIT | `-4.032392` | `-1.369350` | `-0.361365` | `+3.671027` | `-0.292871` |
| Llama | Alpha-history | `-2.495123` | `-1.137219` | `-0.563717` | `+1.931406` | `-0.782910` |
| Qwen | MEMIT | `-0.397837` | `-0.797188` | `-0.251563` | `+0.146274` | `-0.295380` |
| Qwen | Alpha-history | `+0.029707` | `-0.400738` | `+0.555242` | `+0.525536` | `-0.014337` |

Family-average c1→c3 recovery는 Llama `+2.801217`, Qwen `+0.335905`다. 사전 고정한
lenient cross-model implementation gate와 direction-aligned viability gate는
통과한다. 그러나 Llama 두 family와 Qwen MEMIT은 native non-collapse `-0.10`을
실패하므로 strong method gate와 old pair analyzer gate는 실패한다.

따라서 Motivation은 **구현 원인 확인 및 model-common 방향 신호로 닫고**, deployable
ODE-Edit superiority, preservation guarantee, lifelong claim은 열지 않는다.

## 네 범주

### Proposal에서 온 내용

- finite edit를 current-state-dependent trajectory로 보고 direction을 재계산한다.
- layer velocity를 cumulative capacity와 rewrite utility에 따라 다르게 배분한다.
- MEMIT과 AlphaEdit null-space/history에서 같은 policy가 작동해야 한다.

### Repo/protocol에서 확인한 사실

- job `15891`은 4 GPU에서 네 worker를 동시에 실행해 `COMPLETED 0:0`, elapsed
  `01:29:01`로 끝났다.
- controller 8/8, evaluator 8/8 terminal/pass; checkpoint 32/32다.
- 네 model×family 첫 edit/round에서 c3 allocation coefficients와 c1 coefficients의
  max abs diff는 모두 `0`이고 native distance도 동일하다.
- QP 16 edits/64 hops의 exact magnitude와 radial share identity가 통과했다.
- c3는 source commit `bbd3d0e`의 clean pushed main에서 실행됐다.
- EasyEdit 및 precomputed covariance/projector/history는 read-only였다.

### GH 추정

- c1 negative의 대부분은 scientific direction failure가 아니라 BF share/global
  magnitude 결합이었다.
- c3가 c2보다 네 cell 모두 좋으므로 c2의 overload-only cap/share 변경은 현재 panel에서
  도움이 되지 않았다. c1 BF relative share-generation rule을 보존한 것이 더 낫다.
- 남은 Llama gap과 Qwen MEMIT trust failure는 method-stage safeguard와 objective가
  아직 필요함을 뜻한다.

### 사용자 확인 필요

- Motivation 내 추가 실험은 없음. 이 result로 Motivation을 닫는다.
- 다음 large/lifelong scale은 Method Session의 common controller가 strong pilot gate를
  통과한 뒤에만 사용자 승인 대상으로 올린다.

## Technical integrity

- max path relative error: `1.802e-16`.
- max hop relative error: `2.715e-16`.
- max share L2 error: `2.220e-16`.
- max applied/allocation radial identity error: `1.388e-17`.
- allocation cap max violation: `0`; compact allocation barrier residual max
  `1.355e-20`.
- radial scale: overall mean `2.106×`, max `35.927×`.
- applied coefficient가 allocation cap을 넘은 observation은 `140/320`이다. 이는 cap을
  share-generator로만 쓴 사전 명시된 intervention이며 hard-barrier evidence가 아니다.
- positive overload observation `7/7` zero-suppressed.
- trust diagnostic failure `1/64`; Qwen MEMIT edit 1 round 4이며 scale `1.0×`다.

## Scientific gate

### 통과

- 네 cell 모두 c1 current efficacy보다 회복.
- 네 cell 모두 c1 prior retention보다 회복.
- 양 모델 family-average recovery 양수.
- capacity reduction, layer-Gini reduction, max-layer-share reduction은 두 family와
  두 모델에 공통.
- Qwen Alpha는 current `+0.555242`, final all-edit `+0.546818`, prior retention
  `-0.014337`의 strong local point.

### 미통과

- old strong non-collapse floor: Llama MEMIT/Alpha, Qwen MEMIT fail.
- neighborhood/generation KL은 모델 공통 방향이 아님.
- Llama prior retention과 efficacy는 native보다 낮음.
- compute는 native 대비 약 `4.80--6.93×`.
- radial scale `35.927×`와 trust failure 1회 때문에 현재 form을 그대로 deploy할 수 없음.

Old analyzer의 `CAPACITY_HISTORY_HARM_SIGNAL`과 passing family `[]`는 strict absolute
native gate 결과로 그대로 보존한다. c3 causal spec의 lenient gate는 **c1 implementation
cause와 model-common 방향**을 묻기 때문에 동시에 통과할 수 있다. 이를 method
superiority로 바꾸지 않는다.

## Motivation 최종 결정

- **Confirmed:** BF relative layer allocation과 global update magnitude는 분리해야 한다.
- **Confirmed:** 동일 분리가 Llama/Qwen과 MEMIT/Alpha 네 cell 모두 c1을 개선한다.
- **Survive:** state refresh + BF relative share + independent global step의 Motivation.
- **Kill:** c1 absolute-coefficient writer와 c2 changed-share 결과를 scientific method
  verdict로 사용하는 것.
- **Not established:** native superiority, broad preservation, hard capacity barrier,
  lifelong, compute efficiency.
- **No more Motivation retune:** K/share/threshold/model-specific rescue를 추가하지 않는다.

## Method Session으로 넘길 최소 과제

1. Unit-norm layer share를 QP 자체에서 직접 최적화해 near-zero solution을 사후
   `35.9×` 증폭하지 않도록 한다.
2. Negative trust step을 rollback/first-hit로 막되 두 모델에 동일 rule을 쓴다.
3. Applied full step에서의 capacity constraint와 efficacy progress를 동시에 명시한다.
4. Llama native gap과 Qwen MEMIT을 같은 policy로 회복한 뒤에만 scale을 늘린다.

## Agent와 artifact

Llama, Qwen, pair/red Terra Ultra agents는 runtime metadata에서
`gpt-5.6-terra / ultra`를 검증하지 못해 지정 파일을 읽지 않고 모두 `BLOCK`
종료했다. active agent는 남아 있지 않으며 독립 review pass로 세지 않는다.

- Llama analysis SHA-256:
  `fdf52b7c5f1db78d1a12e8092988816bccbadffdab64d9842f401673ab43765f`
- Qwen analysis SHA-256:
  `45b2e7b807d7a7622aea7c4843259b75d109124d01f85bd1cc8073bb5785637a`
- Pair analysis SHA-256:
  `cf9c320fff1492a2d9854de0c57811cdc5788fda152a4b80d3b21b04fb1e00d6`
- evaluator/combined/pair compact 29 files aggregate SHA-256:
  `bc2d3da7fe3cb7ae0859e72e6bcdb144c53f0307f0d013be5bacc511d4203127`

Direct-z 임시 session의 report/artifact는 이 수치나 gate에 합치지 않았고 모니터링하지
않았다.
