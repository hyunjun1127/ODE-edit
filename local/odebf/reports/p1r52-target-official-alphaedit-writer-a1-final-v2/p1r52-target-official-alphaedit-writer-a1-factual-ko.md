# P1R52 Target + Official AlphaEdit Writer A1 최종 사실 보고서

## 1. 실행 상태와 범위

- 상태: **COMPLETE_NEGATIVE_SEQUENTIAL_TRANSFER_RESULT**
- Phase A: 동일한 P1R52 accepted-z와 동일한 writer-entry W를 P1R52 writer 및 공식 EasyEdit AlphaEdit writer에 각각 적용한 10개 독립 B100 쌍을 완료했다.
- Phase B: Phase A의 사전 고정된 gap 방향이 충족되어, fresh W0에서 P1R52 target + Official AlphaEdit writer의 sequential 10×B100(1,000요청)을 완료했다.
- Phase A source: `1e5bee6a555c1a7cc1a1c5b034ecf004d18d475c`. Phase B source: `f072ee73d11fbc704dc8b750ec187923250439c0`.
- Phase A/B W0 복원 PASS, imputation=0, retry=0, native AlphaEdit `compute_z` 호출=0, R52 Structural-H/P/energy-capacity/PIR/PIR-U/FPiQ 영향=0.
- 해석 경계: Phase A만 **DIRECT_SAME_ACCEPTED_Z_PAIRED_WRITER_COMPARATOR**다. Phase B의 봉인 기준들은 source/trajectory가 다른 결합 sequential 비교이며 동일-head 단일변수 인과 비교가 아니다.

## 2. 핵심 결과

Phase A에서 Official AlphaEdit writer는 동일 P1R52 z에 대해 rephrase W-z NLL gap을 P1R52 writer보다 평균 **-0.033913** 낮췄고 10개 중 **7개** case에서 gap이 더 작았다. 이 raw paired 신호로 Phase B가 RELEASE되었다.

Phase B에서는 hybrid의 immediate rephrase NLL이 **2.550638**, final-W10 NLL이 **2.535034**였다. 봉인 R52 H-off의 immediate/final 값 **2.490271/2.423899**보다 낮아지지 않았다. LOC는 hybrid immediate **8674/10000 (0.8674)**, final **8523/10000 (0.8523)**로 R52 H-off immediate/final **8574/10000 (0.8574)/8367/10000 (0.8367)**보다 높았다.

따라서 이 패키지의 기록은 **동일-z 단기 writer gap 감소는 관찰되었으나 sequential 절대 GEN/NLL 개선으로 이어지지 않은 negative result**다. `TRANSFER_GAIN_WITH_LOCALITY_TRADEOFF` 조건은 충족되지 않았다. writer가 유일하거나 주된 병목이라는 결론은 기록하지 않는다.

## 3. Phase A — 동일-z paired writer 비교

| 지표 | 값 |
|---|---:|
| completed endpoints | 20/20 (10 cases × 2 writers) |
| accepted-z hash exact | 10/10 |
| writer-entry W hash exact | 10/10 |
| cross-arm W/cache contamination | 0/0 |
| Official−R52 rephrase gap mean | -0.033913062 |
| median / p90 / min / max | -0.023391690 / 0.039708151 / -0.148942871 / 0.068597336 |
| Official wins / R52 wins / ties | 7 / 3 / 0 |
| Phase-B gate | RELEASE |

음수 delta는 Official writer의 W-z gap이 더 작음을 뜻한다. case별 절대 z/W 지표와 delta는 `p1r52-target-official-alphaedit-writer-phase-a-per-case.json`에 수록했다.

## 4. Phase B — hybrid sequential 절대값

| 패널 | Rewrite success | Rewrite acc | Rephrase success | Rephrase strict success | Rephrase acc | Rephrase strict acc | LOC | Rephrase new NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accepted z, 10×B100 | 999/1000 (0.9990) | 997/1000 (0.9970) | 1952/2000 (0.9760) | 959/1000 (0.9590) | 1408/2000 (0.7040) | 550/1000 (0.5500) | 8776/10000 (0.8776) | 1.349481 |
| immediate W, 10×B100 | 999/1000 (0.9990) | 996/1000 (0.9960) | 1685/2000 (0.8425) | 742/1000 (0.7420) | 992/2000 (0.4960) | 300/1000 (0.3000) | 8674/10000 (0.8674) | 2.550638 |
| final W10, B1000 | 999/1000 (0.9990) | 995/1000 (0.9950) | 1684/2000 (0.8420) | 745/1000 (0.7450) | 1009/2000 (0.5045) | 311/1000 (0.3110) | 8523/10000 (0.8523) | 2.535034 |

- requestwise immediate rephrase writer-realization gap: mean **1.201157**, median **0.586761**, p90 **3.522530**, max **13.103638**.
- z strict-rephrase success 후 W strict-rephrase failure: **222/1000** requests.
- batch update energy: mean **25.254777**, median **23.433852**, p90 **34.742886**, max **39.774690**.
- requestwise final-minus-immediate rephrase NLL: mean **-0.015604**, median **0.000000**, p90 **0.234546**.
- cache_c는 0→1000으로 연속 누적되었고 static P identity는 `8cf69120aad901a6796f676193e11b46912067ef6ca064ed4a2d3670f12531e8`다.

## 5. 봉인 sequential 기준 비교

| 방법·패널 | Rewrite success | Rewrite acc | Rephrase success | Rephrase strict | Rephrase acc | Rephrase acc strict | LOC | Rephrase new NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| memit immediate | 996/1000 (0.9960) | 991/1000 (0.9910) | 1825/2000 (0.9125) | 862/1000 (0.8620) | 1332/2000 (0.6660) | 512/1000 (0.5120) | 8056/10000 (0.8056) | 1.750168 |
| memit final W10 | 942/1000 (0.9420) | 819/1000 (0.8190) | 1777/2000 (0.8885) | 838/1000 (0.8380) | 1287/2000 (0.6435) | 512/1000 (0.5120) | 7214/10000 (0.7214) | 1.753286 |
| alphaedit immediate | 1000/1000 (1.0000) | 1000/1000 (1.0000) | 1922/2000 (0.9610) | 937/1000 (0.9370) | 1495/2000 (0.7475) | 606/1000 (0.6060) | 8177/10000 (0.8177) | 1.230092 |
| alphaedit final W10 | 998/1000 (0.9980) | 996/1000 (0.9960) | 1909/2000 (0.9545) | 929/1000 (0.9290) | 1479/2000 (0.7395) | 598/1000 (0.5980) | 7763/10000 (0.7763) | 1.262916 |
| r52_h_on immediate | 998/1000 (0.9980) | 995/1000 (0.9950) | 1687/2000 (0.8435) | 748/1000 (0.7480) | 1019/2000 (0.5095) | 321/1000 (0.3210) | 8586/10000 (0.8586) | 2.505594 |
| r52_h_on final W10 | 999/1000 (0.9990) | 992/1000 (0.9920) | 1696/2000 (0.8480) | 751/1000 (0.7510) | 1023/2000 (0.5115) | 324/1000 (0.3240) | 8394/10000 (0.8394) | 2.430740 |
| r52_h_off immediate | 998/1000 (0.9980) | 995/1000 (0.9950) | 1680/2000 (0.8400) | 737/1000 (0.7370) | 1018/2000 (0.5090) | 315/1000 (0.3150) | 8574/10000 (0.8574) | 2.490271 |
| r52_h_off final W10 | 998/1000 (0.9980) | 992/1000 (0.9920) | 1696/2000 (0.8480) | 748/1000 (0.7480) | 1032/2000 (0.5160) | 324/1000 (0.3240) | 8367/10000 (0.8367) | 2.423899 |
| hybrid immediate | 999/1000 (0.9990) | 996/1000 (0.9960) | 1685/2000 (0.8425) | 742/1000 (0.7420) | 992/2000 (0.4960) | 300/1000 (0.3000) | 8674/10000 (0.8674) | 2.550638 |
| hybrid final W10 | 999/1000 (0.9990) | 995/1000 (0.9950) | 1684/2000 (0.8420) | 745/1000 (0.7450) | 1009/2000 (0.5045) | 311/1000 (0.3110) | 8523/10000 (0.8523) | 2.535034 |

공식 AlphaEdit full pipeline은 자체 native target을 사용하므로 hybrid와 동일 accepted-z 비교가 아니다. 봉인 R52 H-on/H-off 역시 서로 다른 source/trajectory다. 위 표는 절대값의 matched stream reference이며 causal isolation으로 사용하지 않는다.

## 6. 배치별 전달 궤적

| B | cache width | z Rephrase NLL | W Rephrase NLL | W-z gap | W Rephrase success | W strict | LOC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0→100 | 1.342022 | 2.293479 | 0.951458 | 179/200 (0.8950) | 82/100 (0.8200) | 889/1000 (0.8890) |
| B2 | 100→200 | 1.356797 | 2.257081 | 0.900284 | 177/200 (0.8850) | 81/100 (0.8100) | 911/1000 (0.9110) |
| B3 | 200→300 | 1.224506 | 2.375576 | 1.151070 | 170/200 (0.8500) | 73/100 (0.7300) | 889/1000 (0.8890) |
| B4 | 300→400 | 1.363806 | 2.276013 | 0.912207 | 174/200 (0.8700) | 79/100 (0.7900) | 846/1000 (0.8460) |
| B5 | 400→500 | 1.493324 | 2.643810 | 1.150486 | 164/200 (0.8200) | 71/100 (0.7100) | 875/1000 (0.8750) |
| B6 | 500→600 | 1.293977 | 2.632825 | 1.338848 | 166/200 (0.8300) | 72/100 (0.7200) | 833/1000 (0.8330) |
| B7 | 600→700 | 1.332521 | 2.684099 | 1.351578 | 171/200 (0.8550) | 76/100 (0.7600) | 860/1000 (0.8600) |
| B8 | 700→800 | 1.476486 | 2.907560 | 1.431074 | 160/200 (0.8000) | 67/100 (0.6700) | 827/1000 (0.8270) |
| B9 | 800→900 | 1.115606 | 2.223788 | 1.108182 | 172/200 (0.8600) | 76/100 (0.7600) | 860/1000 (0.8600) |
| B10 | 900→1000 | 1.495771 | 3.212151 | 1.716380 | 152/200 (0.7600) | 65/100 (0.6500) | 884/1000 (0.8840) |

## 7. 무결성 및 계산 경계

- Phase A scheduler terminal: COMPLETED, exit 0:0, completed cases 10/10, requests 1000/1000.
- Phase B scheduler terminal: COMPLETED, exit 0:0, completed batches 10/10, requests 1000/1000.
- Phase B materialization authoritative count=10, W persistence PASS, final W10 hash `90609bcc9eb5fd9da5971ce5707cba7d9d07d58d007a4bb54de9f6dd244c8710`, terminal W0 pointer/bytes restore PASS.
- cache_c continuity: entry/consume/append widths are batch table에 기록; final hash `d3f7eb1e56fadd59c818947fb690bbe2041bf217f69bee9846888965dcdf91c7`.
- `native_alphaedit_compute_z_call_count=0`; target는 P1R52 accepted-z만 사용했다.
- Structural-H, R52 P barrier, energy/capacity barrier, PIR/PIR-U/FPiQ, R52 historical-risk/router decision influence count=0.
- success와 accuracy는 별도 집계했다. batch-entry method-specific evaluator count=0. missing/imputation=0.
- target objective 집계: model forward **4000**, backward **4000**, corrector forward **2050**, generation **0**, processed tokens **851856**.
- terminal evaluator: accepted-z/immediate-W/final-W10 model forward **1000/1000/1000**, backward/generation **0/0**.
- Official writer materialization **10**, edit-core wall **352.318168s**. Writer 내부 model F/B는 별도 counter가 없어 **NOT_RECORDED_SEPARATELY**로 유지했다.
- Phase A/B terminal `job_compute` 원문과 위 source-backed 세부 집계는 `p1r52-target-official-alphaedit-writer-core-comparison.json`에 함께 보존했다.

## 8. 결론 분류

- Phase A: **SAME_Z_OFFICIAL_WRITER_GAP_REDUCTION_OBSERVED**.
- Phase B: **SEQUENTIAL_ABSOLUTE_GEN_IMPROVEMENT_NOT_OBSERVED**.
- 최종: **COMPLETE_NEGATIVE_SEQUENTIAL_TRANSFER_RESULT**.
- 이 결과는 P1R52 preservation target + Official AlphaEdit writer라는 패키지 결과다. KL/decay/clamp 각각의 고립 인과 효과나 writer 단독 병목을 주장하지 않는다.
