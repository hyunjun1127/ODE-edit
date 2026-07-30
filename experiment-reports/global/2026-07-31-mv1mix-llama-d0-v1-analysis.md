# MV-1 score-mix D0 독립 분석: llama3-8b-inst

- run ID: `mv1mix_llama_d0_v1`
- 분석 상태: `d0_descriptive_complete`
- artifact/panel: `valid=True`, `complete=True`
- 해석 경계: D0 calibration 및 continuous-routing early-kill 입력만이다. 양의 event는 paired D1만 허용하며 GO, 방법 개선 claim, confirmatory success 또는 MV-2 진입을 뜻하지 않는다.

## Metric과 판정

- `P_i(V)`는 runner가 기록한 `progress = after.utility - before.utility`다.
- `G_mix = P_i(V_mix) - P_i(V_uniform)`이며 두 arm은 같은 `q=1/256` 및 remeasured equal-`C` budget을 사용한다.
- model replay envelope `e_m = 1e-12`는 `max(1e-12, 5개 no_op_replay의 max(abs(progress)))`다.
- event 판정은 `G_mix <= e_m`, continue는 `G_mix > e_m`다. `1e-12`는 exact-zero JSON/부동소수점 산술용 절대 floor이며 효과 허용폭으로 한 번 더 더하지 않는다.

## Case별 결과

| case | P(score_mix) | P(uniform) | G_mix | e_m 내 | predicted slope gap | realized slope gap |
| --- | ---: | ---: | ---: | :---: | ---: | ---: |
| `20313` | 1.6716976 | 1.6527452 | 0.01895237 | no | 37.055753 | 18.152131 |
| `12241` | 0.92990208 | 0.88412666 | 0.045775414 | no | 38.017051 | 47.38888 |
| `19008` | 3.0060501 | 2.9197054 | 0.086344719 | no | 77.235968 | 76.31609 |
| `19110` | 0.90700817 | 0.900877 | 0.0061311722 | no | 7.1145798 | 5.4499718 |
| `9316` | 1.2676435 | 1.2598825 | 0.0077610016 | no | 3.3222414 | 7.3924523 |

## 단일 모델 결론

- all-five nonpositive-within-envelope: `False`
- envelope 초과 event 수: `5`
- pair-level early kill은 이 문서에서 계산하지 않는다. 독립 검증된 Llama/Qwen 두 analyzer 출력이 모두 all-five 조건을 만족할 때만 GH가 결합 판정한다.
