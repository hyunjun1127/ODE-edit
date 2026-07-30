# Qwen2.5-7B-Instruct MV-1 untouched 독립 분석

- 작성일: 2026-07-31 KST
- run: `mv1mix_qwen_untouched_v1`
- model: `qwen2.5-7b-inst`
- execution commit: `a7c929f11da540c7435f9b0f3d367c5009c1ff2d`
- Slurm: job `15586`, `odeedit_mv1mix_untouched_pair_v1`, `devbox`
- 판정 범위: Qwen 단일 모델의 §10.9 gate input까지만 허용

## 네 범주 구분

### Proposal에서 온 내용

이 보고서는 proposal의 최종 method claim을 검증하지 않는다. Motivation에서
필요한 최소 질문, 즉 고정된 matched-energy 조건에서 event-wise `score_mix`가
frozen `frozen_static_mix`보다 양의 progress를 내는지만 본다. Primary estimand는
`progress(score_mix)-progress(frozen_static_mix)`로 고정되어 있다.

### Repo/protocol에서 확인한 사실

- canonical untouched split exact 20 case, seed `17`, `q=1/256`, six-arm panel이다.
- 20/20 case가 완료되었고 feature/action/event/receipt가 각각 20개,
  outcome이 120개다. 실패·abort·미실행 case는 0개다.
- 모든 rollback이 exact이고 matched-`C`, outcome firewall, durable action
  receipt 검증이 모두 통과했다. `direct_z`는 event당 한 번 고정되어 20개지만
  이 분석에서는 tensor를 열지 않았다.
- raw prompt, target, logits, weights, generation, activation은 이 보고서에
  노출하지 않았다. Git output도 생성하지 않았다.
- execution은 commit `a7c929f11da540c7435f9b0f3d367c5009c1ff2d`에서
  tracked-clean으로 실행되었다. 사용한
  `mv1_score_mix_followup_analysis.py`는 이 commit부터 점검 시점 HEAD
  `51d85971f825b237d7a79320c9c8eb9cf9cf92bd`까지 diff가 없다.

### GH 추정

Qwen에서는 event-adaptive routing의 방향 신호가 재현되었다. 다만 D1 forecast는
untouched 평균보다 낙관적이었다. 다음 단계 기대효과의 중심을 D1의
`+0.0845763`으로 두기보다, 현재 untouched 관측치 `+0.0483206`과 그 bootstrap
구간을 보수적 기준으로 삼는 편이 타당하다. 이는 ODE-Edit 전체 method gain,
다른 모델 일반화, 장기 trajectory 개선을 뜻하지 않는다.

### 사용자 확인 필요

이 단일 모델 보고서만으로 사용자 선택이 필요한 사항은 없다. 다른 모델의
독립 분석 뒤 별도 red pair audit이 cross-model 판정과 최소 MV-2 개방 여부를
결정해야 한다. 본 보고서는 그 결정을 하지 않는다.

## Artifact 및 재현 lock

| 항목 | 확인값 |
| --- | ---: |
| planned / attempted / pass / failure | `20 / 20 / 20 / 0` |
| features / actions / events / receipts | `20 / 20 / 20 / 20` |
| outcomes | `120` (`20 × 6`) |
| exact rollback | `true` |
| equal-`C` + rollback validation | `true` |
| outcome firewall + receipt validation | `true` |
| replay envelope | `1e-12` |
| bootstrap | seed `20260731`, `4000` replicates |

Source identity:

- analysis SHA-256:
  `f401d62e57e33d27bfac6623ac0d89f0f063a2fdf7819f21ddcbc2cb21ae8947`
- manifest SHA-256:
  `e93605704088b458e78cbc302d1406e66eff352cd7693b9b8ad5d235f2ad43d2`
- raw summary SHA-256:
  `dd12913804becaa669d09d478cadcc871177d7b3ad3fc3fe6cd433182f034e6b`

## Primary 및 oracle

Case-level 값은 공개하지 않고 aggregate만 기록한다.

| metric | primary: adaptive − frozen static | finite-panel oracle |
| --- | ---: | ---: |
| mean | `0.04832061529159546` | `0.04847902059555054` |
| trimmed mean 20% | `0.03188145160675049` | `0.03188145160675049` |
| median | `0.021752119064331055` | `0.021752119064331055` |
| positive sign | `20/20` | `20/20` |
| paired bootstrap mean 95% CI | `[0.027076385319232944, 0.0734122896194458]` | `[0.0264308500289917, 0.07327890038490295]` |

Primary mean, trim20, median, sign count, bootstrap CI가 모두 replay envelope의
양의 방향이다. Oracle과 primary의 mean 차이는 약 `0.0001584`로 작아, 이
finite panel 안에서는 고정 static 대비 이용 가능한 기회의 대부분을
`score_mix`가 포착했다. 이는 panel 밖 최적성 주장은 아니다.

## D0+D1 calibration prior와 untouched casewise forecast

D0+D1 calibration에서 outcome 개봉 전에 고정한 population-level prior와,
untouched 20 case의 frozen feature에 같은 forecast policy를 적용한 casewise
predicted mean은 서로 다른 수치다.

| 항목 | 값 |
| --- | ---: |
| D0+D1 calibration prior: expected realized gap | `0.11201531043591464` |
| D0+D1 calibration prior: case-bootstrap 95% interval | `[0.06476441445994054, 0.17629666495305207]` |
| untouched realized − calibration prior | `-0.06369469514431918` |
| untouched realized / calibration prior | 약 `0.431374` |
| untouched casewise frozen predicted mean | `0.0845762937360485` |
| untouched realized mean | `0.04832061529159546` |
| casewise mean error (realized − predicted) | `-0.03625567844445304` |
| MAE | `0.038683593051440535` |
| median AE | `0.018002594769061388` |
| zero-intercept slope | `0.563926864638651` |
| Pearson | `0.8164494890719834` |
| positive-direction concordance | `1.0` |

Untouched 실현 평균은 D0+D1 calibration prior의 약 `43.14%`이며 prior
interval의 하한보다도 작다. 별도로, untouched casewise frozen predicted mean
대비로는 약 `57.1%`가 실현되었다. 방향 일치는 유지되고 case 간 순위 상관도
높지만 두 기준 모두 magnitude가 낙관적이었음을 보인다. 따라서 Qwen의 보수적
motivation 기대효과는 현 시점에서 primary mean `+0.0483`, 불확실성 범위는
bootstrap mean 95% CI `[+0.0271, +0.0734]`로 읽는다.

## Compute 및 resource

| 항목 | 값 |
| --- | ---: |
| visible GPU | `1` |
| wall time | `8411.8998 s` (`2.3366 h`) |
| GPU peak allocated | `43,798,774,272 B` (`40.7908 GiB`) |
| GPU peak reserved | `47,605,350,400 B` (`44.3359 GiB`) |
| host max RSS | `17,342,640 KiB` (`16.5392 GiB`) |

GPU peak reserved는 단일 A6000 visible memory 범위 안이며 OOM, failure,
resource abort는 없었다.

## §10.9 locked 단일 모델 gate

고정 gate를 aggregate에서 직접 재계산했다.

- clear: `mean > e`이고 trim20, median, 또는 sign `>=11/20` 중 하나 이상
- kill: mean, trim20, oracle가 모두 `<=e`이고 sign `<=10/20`
- controller pivot: primary kill 조건이지만 oracle만 양의 방향
- gray: 위 셋 어디에도 속하지 않음

Qwen 입력은 다음과 같다.

| gate input | 판정 |
| --- | --- |
| `clear_continue_input` | `true` |
| `scientific_kill_input` | `false` |
| `controller_pivot_input` | `false` |
| `gray_input` | `false` |
| architecture-conditional tolerance input | `0.03236109868826046` |
| pair-level decision computed | `false` |
| MV-2 decision computed | `false` |

주의: generic analyzer JSON의 follow-up gate boolean은 C1 외 wave에서
비결정값 `false`로 고정되는 현재 구현이다. 위 표는 threshold를 바꾸지 않고
§10.9 untouched 규칙을 aggregate에 직접 적용한 authoritative single-model
input이다. 이 차이를 pair verdict로 오해해서는 안 된다.

## Claim boundary

Qwen fixed-model untouched 20-case에서 adaptive-minus-static motivation 신호가
§10.9 단일 모델 clear input을 충족했다. Pair/cross-model verdict, method gain,
MV-2 개방, ODE refresh/trajectory claim은 모두 별도 red pair audit 전까지
금지한다.
