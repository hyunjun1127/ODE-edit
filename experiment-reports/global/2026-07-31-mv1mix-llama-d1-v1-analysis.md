# MV-1 score-mix D1 독립 분석: llama3-8b-inst

- run ID: `mv1mix_llama_d1_v1`
- 분석 상태: `d1_descriptive_complete`
- artifact/panel: `valid=True`, `complete=True`
- 해석 경계: D1 calibration과 frozen static comparator 생성만 수행한다. 양수 결과도 방법 개선 claim, GO, confirmatory success 또는 MV-2 진입을 허용하지 않는다.
- D1 replay envelope: `1e-12`

## Case별 descriptive outcome

| case | P(score_mix) | P(uniform) | G_mix | e_D1 초과 |
| --- | ---: | ---: | ---: | :---: |
| `2699` | 0.62903595 | 0.58279896 | 0.046236992 | yes |
| `17994` | 0.8897264 | 0.8812387 | 0.0084877014 | yes |
| `16979` | 1.0465937 | 1.0381351 | 0.0084586143 | yes |
| `19018` | 0.51847935 | 0.51642466 | 0.0020546913 | yes |
| `5545` | 1.0209837 | 1.0166445 | 0.0043392181 | yes |
| `10299` | 1.1785316 | 1.1770239 | 0.0015077591 | yes |
| `14991` | 0.92964697 | 0.92003202 | 0.0096149445 | yes |
| `13486` | 0.71388531 | 0.71331692 | 0.00056838989 | yes |
| `968` | 0.48484516 | 0.4814806 | 0.003364563 | yes |
| `6334` | 1.0458775 | 1.0424151 | 0.0034623146 | yes |
| `9735` | 0.34815502 | 0.34774399 | 0.00041103363 | yes |
| `5160` | 0.1668787 | 0.16421127 | 0.0026674271 | yes |

## Frozen static policy

- 입력: independently valid D0 `[3:8]`와 D1 `[8:20]`의 17개 feature에서 다섯 single-layer slope만 사용했다.
- outcome field 사용: 없음
- policy hash: `e37443fa075245aa52828773ffc062217d983fb46254d1f40010d396be3d5e06`

| layer action | sbar | frozen weight |
| --- | ---: | ---: |
| `layer_4` | 446.56816 | 0.52924976 |
| `layer_5` | 388.35476 | 0.46025822 |
| `layer_6` | 366.04768 | 0.43382101 |
| `layer_7` | 337.70347 | 0.4002289 |
| `layer_8` | 337.16551 | 0.39959133 |

- 이 policy는 confirmatory comparator 후보를 freeze한 것이며 positive efficacy evidence가 아니다.
- pair-level 또는 confirmatory 결정은 이 단일 모델 문서에서 계산하지 않는다.
