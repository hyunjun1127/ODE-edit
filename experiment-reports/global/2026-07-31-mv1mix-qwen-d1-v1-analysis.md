# MV-1 score-mix D1 독립 분석: qwen2.5-7b-inst

- run ID: `mv1mix_qwen_d1_v1`
- 분석 상태: `d1_descriptive_complete`
- artifact/panel: `valid=True`, `complete=True`
- 해석 경계: D1 calibration과 frozen static comparator 생성만 수행한다. 양수 결과도 방법 개선 claim, GO, confirmatory success 또는 MV-2 진입을 허용하지 않는다.
- D1 replay envelope: `1e-12`

## Case별 descriptive outcome

| case | P(score_mix) | P(uniform) | G_mix | e_D1 초과 |
| --- | ---: | ---: | ---: | :---: |
| `2699` | 1.0726891 | 0.9842205 | 0.088468552 | yes |
| `17994` | 2.3700945 | 2.3494208 | 0.020673752 | yes |
| `16979` | 4.1236296 | 3.8536806 | 0.26994896 | yes |
| `19018` | 2.7011609 | 2.5213459 | 0.17981505 | yes |
| `5545` | 6.0900741 | 5.8843694 | 0.20570469 | yes |
| `10299` | 9.4645084 | 9.2071664 | 0.25734198 | yes |
| `14991` | 2.4236329 | 2.4066775 | 0.016955376 | yes |
| `13486` | 0.46042633 | 0.42610168 | 0.034324646 | yes |
| `968` | 1.6650963 | 1.517664 | 0.14743233 | yes |
| `6334` | 5.2591755 | 5.1529975 | 0.10617805 | yes |
| `9735` | 3.3677135 | 2.9245105 | 0.44320297 | yes |
| `5160` | 2.8027735 | 2.6232681 | 0.17950535 | yes |

## Frozen static policy

- 입력: independently valid D0 `[3:8]`와 D1 `[8:20]`의 17개 feature에서 다섯 single-layer slope만 사용했다.
- outcome field 사용: 없음
- policy hash: `381e22334e0e3d07c37f07db16c22e8cdd1b689deba66ec7b032dd32ba0b2955`

| layer action | sbar | frozen weight |
| --- | ---: | ---: |
| `layer_4` | 54.251434 | 0.30583144 |
| `layer_5` | 66.309229 | 0.37380481 |
| `layer_6` | 91.002316 | 0.51300708 |
| `layer_7` | 85.557427 | 0.48231263 |
| `layer_8` | 92.334058 | 0.5205145 |

- 이 policy는 confirmatory comparator 후보를 freeze한 것이며 positive efficacy evidence가 아니다.
- pair-level 또는 confirmatory 결정은 이 단일 모델 문서에서 계산하지 않는다.
