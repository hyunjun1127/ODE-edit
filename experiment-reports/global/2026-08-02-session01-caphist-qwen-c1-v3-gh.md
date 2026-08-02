# Session 01 capacity/history — Qwen c1_v3 GH 분석

- 날짜: 2026-08-02
- 모델: `qwen2.5-7b-inst`
- 방법명: **ODE-Edit**
- policy: `capacity-qp-history-k4-native-progress-v2`
- 판정: **technical pass; Alpha-history만 lenient signal, MEMIT은 harm signal**
- claim boundary: same-policy 4-edit Motivation diagnostic only

## 결론

Qwen c1은 c0의 under-edit 구현을 크게 회복했다. Alpha-history의 QP-minus-native
current-edit utility는 `+0.029707`이고 capacity/KL/concentration 축도 함께 줄어
model-level lenient signal을 냈다. 반면 MEMIT current utility는 `-0.397837`로
non-collapse floor `-0.10`을 실패했다. 같은 모델에서도 editor family에 따라 gate가
갈리므로 ODE-Edit 공통 method 우위로 승격할 수 없다.

## 네 범주

### Proposal에서 온 내용

- current state에서 방향과 layer allocation을 다시 계산한다.
- ordered native endpoint의 남은 rewrite utility 전체를 progress request로 쓴다.
- AlphaEdit null-space/history에서도 MEMIT과 동일 policy가 작동해야 한다.

### Repo/protocol에서 확인한 사실

- 네 controller와 네 evaluator가 각각 4/4 terminal, `all_pass=true`다.
- Llama와 동일 case/order/seed/layers/threshold/controller code를 사용했다.
- precomputed covariance/projector만 read-only로 사용했고 EasyEdit를 수정하지 않았다.
- raw evaluation text/logit/token을 저장하지 않았고 controller firewall이 통과했다.

### GH 추정

- Qwen Alpha-history의 작은 양성은 canonical historical term과 capacity-QP가 결합될
  가능성을 보여주지만 4 cases의 point signal일 뿐이다.
- MEMIT 손실과 Alpha prior-retention 손실이 남아 있으므로 “AlphaEdit보다 ODE가
  우월하다” 또는 “preservation을 보장한다”는 결론은 불가능하다.
- Qwen에서 관측된 overload suppression 1회는 Llama에 재현되지 않아 cross-model
  routing mechanism evidence가 아니다.

### 사용자 확인 필요

- 없음. 모델별 method 분기나 Qwen-only rescue는 사용자 원칙상 허용하지 않는다.

## c1 QP-minus-native 결과

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-0.397837` | `+0.029707` |
| final all-edit utility delta | `-0.387928` | `+0.024499` |
| final prior-retention delta | `-0.579252` | `-0.431721` |
| retention AUC delta | `-0.667779` | `-0.478392` |
| neighborhood KL reduction | `+0.025233` | `+0.067991` |
| generation KL reduction | `+0.263864` | `+0.101347` |
| target-true NLL drift reduction | `+1.070763` | `+0.770046` |
| capacity reduction | `+0.000729` | `+0.001679` |
| max-layer-share reduction | `+0.084778` | `+0.285425` |
| layer-Gini reduction | `+0.134872` | `+0.289573` |
| cumulative Frobenius reduction | `-6.551545` | `+3.946488` |

현재 utility 절대 평균은 MEMIT native/QP `5.079090/4.681254`, Alpha-history
native/QP `4.979270/5.008976`이다. First-hit은 네 branch 모두 `4/4`이므로 Qwen에서는
top-1 자체보다 margin/retention과 cost frontier가 더 중요한 구분자다.

## 구현 수정 효과와 realized path

| Family | c0 path/native | c1 path/native | c0 current delta | c1 current delta |
|---|---:|---:|---:|---:|
| MEMIT | `0.399978` | `0.863935` | `-3.710800` | `-0.397837` |
| Alpha-history | `0.409667` | `0.742208` | `-3.321125` | `+0.029707` |

MEMIT은 16 accepted rounds, Alpha-history는 native reference를 한 편집에서 조기
match해 15 rounds다. Native reference match는 두 family 각각 `1/4`; 나머지
`3/4`는 `budget_exhausted=true`다. Realized path 범위는 MEMIT `0.728--0.969D`,
Alpha-history `0.521--0.958D`다.

QP/native wall ratio는 MEMIT `6.09×`, Alpha-history `4.41×`다. Alpha의 작은
point gain만으로 이 계산비를 정당화하지 않는다.

## Routing과 agent

MEMIT과 Alpha-history에서 overload observation/suppression/reroute가 각각 1회씩
관측됐고 barrier violation은 numerical zero다. 하지만 Llama의 해당 count는 0이라
proposal의 model-common overloaded-layer routing 가설은 아직 성립하지 않는다.

Terra Ultra Qwen analyst는 runtime metadata를 검증하지 못해 지정 파일을 열지 않고
종료했다. 독립 agent analysis로 세지 않으며 이 문서는 GH fallback이다.

- analysis SHA-256:
  `58b25722f69d1e11de3c8360365632a9b5321cd7e805c6b42184934234ee400f`
- raw root: ignored
  `local/results/raw/session01_motivation/caphist_*_qwen_c1_v3/`

## 최종 판정

- MEMIT: `CAPACITY_QP_HARM_SIGNAL`.
- Alpha-history: `CAPACITY_QP_MODEL_SIGNAL`.
- model: `CAPACITY_HISTORY_MODEL_SIGNAL`.

Qwen Alpha-history는 Motivation-sized possibility signal이지만, Llama와 MEMIT에
공통되지 않는다. 따라서 model-specific rescue 없이 pair gate를 열 수 없다.
