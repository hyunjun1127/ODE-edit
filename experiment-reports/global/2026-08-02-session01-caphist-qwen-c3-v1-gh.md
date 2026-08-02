# Session 01 BF-share magnitude-only — Qwen c3_v1 GH 분석

- 날짜: 2026-08-02
- 모델: `qwen2.5-7b-inst`
- 방법명: **ODE-Edit**
- policy: `bf-common-frontier-share-radial-exact-quarter-k4-v1`
- 판정: **cross-model low-update cause supported; Alpha positive, MEMIT mixed**
- claim boundary: same-policy 4-edit Motivation cause-isolation only

## 결론

Qwen에서도 c1 BF share를 유지한 magnitude-only c3가 c1보다 회복했다. MEMIT current
delta는 `-0.397837→-0.251563`으로 `+0.146274`, Alpha-history는
`+0.029707→+0.555242`로 `+0.525536` 개선됐다. 두 family 평균 current delta는
native 대비 `+0.151840`이다. 따라서 low-update 구현 원인은 Llama에만 국한되지 않고
Qwen 방향에서도 지지된다.

Alpha-history는 final all-edit `+0.546818`, prior retention `-0.014337`로 가장 강한
Motivation point를 냈다. MEMIT은 native 대비 current `-0.251563`, retention
`-0.295380`으로 남아 Qwen 전체 method superiority는 아직 아니다.

## 네 범주

### Proposal에서 온 내용

- BF layer share와 global integration step을 분리한다.
- current-state refresh와 cumulative capacity allocation을 함께 사용한다.
- 모델별 다른 policy 없이 Llama/Qwen에 같은 controller를 적용한다.

### Repo/protocol에서 확인한 사실

- Qwen controller/evaluator 네 branch가 모두 terminal/pass다.
- 첫 edit/round에서 c3 allocation coefficients는 c1 coefficients와 MEMIT/Alpha 모두
  max abs diff `0`이고 native distance도 동일하다.
- allocation cap 위반은 0이며 positive overload 7개는 모두 zero-suppression됐다.
- MEMIT 한 round의 trust diagnostic만 negative였고 fixed-distance policy상 action은
  보존됐다. Alpha trust failure는 0이다.

### GH 추정

- c2의 Qwen worsening은 update 확대 자체보다 changed cap/share policy 영향이었다.
  c3가 c2보다 MEMIT `+0.545625`, Alpha `+0.955980` 높기 때문이다.
- Alpha의 큰 positive는 BF relative share와 independent magnitude 결합 가능성을
  보여주지만 4-edit point signal이다.

### 사용자 확인 필요

- 없음. Qwen-only branch나 threshold를 만들지 않고 common policy를 유지한다.

## c3 QP-minus-native 결과

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-0.251563` | `+0.555242` |
| final all-edit utility delta | `-0.256171` | `+0.546818` |
| final prior-retention delta | `-0.295380` | `-0.014337` |
| retention AUC delta | `-0.431188` | `-0.195831` |
| neighborhood KL reduction | `+0.016724` | `+0.053091` |
| generation KL reduction | `-0.195487` | `-0.124116` |
| target-true NLL drift reduction | `+0.950084` | `+0.412229` |
| capacity reduction | `+0.000529` | `+0.001504` |
| max-layer-share reduction | `+0.086491` | `+0.217269` |
| layer-Gini reduction | `+0.122472` | `+0.210097` |
| cumulative Frobenius reduction | `-4.257640` | `+2.487057` |

Generation KL은 두 family에서 나빠졌고 MEMIT Frobenius도 나빠졌다. Alpha positive
efficacy만으로 broad preservation을 주장하지 않는다.

## 구현 수정 효과

| Family | c1 | c2 | c3 | c3-c1 | c3-c2 |
|---|---:|---:|---:|---:|---:|
| MEMIT current | `-0.397837` | `-0.797188` | `-0.251563` | `+0.146274` | `+0.545625` |
| Alpha current | `+0.029707` | `-0.400738` | `+0.555242` | `+0.525536` | `+0.955980` |
| MEMIT prior retention | `-0.579252` | `-0.645751` | `-0.295380` | `+0.283872` | `+0.350371` |
| Alpha prior retention | `-0.431721` | `-0.363292` | `-0.014337` | `+0.417385` | `+0.348955` |

Family-average c1→c3 current recovery는 `+0.335905`, retention recovery는
`+0.350629`다. MEMIT은 edit별 recovery가 mixed지만 평균 방향은 양수이고 Alpha는
4/4 edit가 c1보다 개선됐다.

## Magnitude/trust caveat와 compute

- MEMIT radial scale: mean `1.413×`, max `4.226×`.
- Alpha radial scale: mean `3.565×`, max `35.927×`.
- Alpha edit 4 round 4의 `35.927×`는 allocation norm이 거의 0에 가까웠다는 뜻이다.
  Exact magnitude 결과는 유효하지만 deployable controller에서는 unit-share를 직접
  최적화하거나 lower-bound/rollback safeguard가 필요하다.
- MEMIT edit 1 round 4는 rewrite gain `-0.057425`, trust ratio `-0.248366`으로
  diagnostic trust gate를 1회 실패했다. 이 round의 radial scale은 `1.0×`여서
  magnitude amplification 때문은 아니다.
- QP/native controller wall ratio: MEMIT `6.59×`, Alpha `4.80×`.
- analysis SHA-256:
  `45b2e7b807d7a7622aea7c4843259b75d109124d01f85bd1cc8073bb5785637a`.

Terra Ultra Qwen analyst는 runtime metadata를 검증하지 못해 지정 파일을 읽지 않고
`BLOCK` 종료했다. 독립 분석으로 세지 않으며 이 문서는 GH fallback이다.

## 판정

Qwen은 cross-model implementation signal을 지지하며 Alpha-history는 strong local
point signal이다. MEMIT은 strong floor를 실패하고 trust failure 1회가 있어 model
전체 superiority는 열지 않는다. 모델별 분기 없이 Method Session의 common safeguard로
넘긴다.
