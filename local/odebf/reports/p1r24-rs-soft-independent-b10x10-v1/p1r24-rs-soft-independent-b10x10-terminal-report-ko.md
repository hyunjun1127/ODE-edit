# P1R24 RS-Soft 독립 B10×10 Atomic 최종 보고서

## 1. 범위와 무결성

- instruction: ODEEDIT-S05-P1R24-RS-SOFT-INDEPENDENT-B10X10-V1
- 과학 소스는 ce8c6c36348752f1407f7d713d30e6b5c727379b; wrapper-only TECH-R1 실행 HEAD는 88954efa04b44c3db29f2a4c7da947736ae80f27이다.
- 동일 frozen 10×B10 stream seal/order는 74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6 / abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c이다.
- 각 B10은 동일 W0에서 독립 시작했고, K8/tau1 후 action-freeze/evaluator를 거쳐 exact W0 pointer+bytes restore를 확인했다. cross-case weight/controller/history state는 0이다.
- scheduler: Llama 19018 COMPLETED 0:0, 00:23:12, MaxRSS 10,883,872 KiB; Qwen 19019 COMPLETED 0:0, 00:20:16, MaxRSS 7,301,924 KiB.
- raw-free 검증: case 20/20, top/case identity, terminal↔manifest SHA, action-freeze, 8 accepted receipts, H/history0, one-h, second-division0, W0 restore 전부 PASS.

## 2. 기술 경계

- 최초 wrapper checkpoint cdd6468의 job 19012는 과학 동작 전에 ScalableObjectivePlan.context_ordinals 접근으로 20/20 case가 실패했다. 각 case W0 restore는 PASS이고 과학 endpoint는 없었다.
- TECH-R1은 wrapper certificate를 source-compatible raw_free_payload().context_count==6 및 capture row ordinals {0..5} 검사로만 바꿨다. 과학 체크포인트/수치/샘플은 바꾸지 않았다.
- replacement 19018/19019만 이번 endpoint 분석에 사용한다. retry, rescue, imputation은 0이다.

## 3. 전체 집계

| model | coverage | W0 E/G/L | P1R24 RS-Soft E/G/L | Eff new NLL | Gen new NLL | Loc new NLL | z8 NLL | terminal P | functional KL | capacity | realization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | 10/10 | 13/100, 28/200, 892/1000 | 99/100, 177/200, 873/1000 | 0.179537 | 2.377230 | 10.407288 | 0.084685 | 0.006847 | 0.021350 | 4.016993 | 0.549 |
| Qwen | 10/10 | 11/100, 36/200, 849/1000 | 100/100, 163/200, 844/1000 | 0.492203 | 3.599686 | 10.113464 | 0.294569 | 0.191940 | 0.019761 | 17.720959 | 0.283 |

동일 W0·10개 B10·frozen evaluator의 Official AlphaEdit를 primary baseline으로 함께 두면 다음과 같다. W0의 연속 NLL 전체 집계는 이번 raw-free 분석 입력에 직렬화되지 않아 `NOT_RECORDED`로 남긴다.

| model | method | coverage | E/G/L | Eff new NLL | Gen new NLL | Loc new NLL |
|---|---|---:|---:|---:|---:|---:|
| Llama | W0 | 10/10 | 13/100, 28/200, 892/1000 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| Llama | Official AlphaEdit baseline | 10/10 | 100/100, 185/200, 859/1000 | 0.001169 | 1.762698 | 10.153035 |
| Llama | P1R24 RS-Soft | 10/10 | 99/100, 177/200, 873/1000 | 0.179537 | 2.377230 | 10.407288 |
| Qwen | W0 | 10/10 | 11/100, 36/200, 849/1000 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| Qwen | Official AlphaEdit baseline | 10/10 | 100/100, 192/200, 828/1000 | 0.032646 | 1.772163 | 9.911450 |
| Qwen | P1R24 RS-Soft | 10/10 | 100/100, 163/200, 844/1000 | 0.492203 | 3.599686 | 10.113464 |

- AlphaEdit 대비 P1R24는 Llama에서 Eff -1/100, Gen -8/200, Loc +14/1000이고, Qwen에서 Eff 동률, Gen -29/200, Loc +16/1000이다. Loc count가 더 높더라도 Eff/Gen과 absolute target-new NLL strength가 일치하지 않으므로 이를 곧바로 preservation 우위로 해석하지 않는다.

## 4. B10별 endpoint

| model | batch | Eff | Gen | Loc | Eff new NLL | Eff margin | z8 NLL | W-z gap | P | functional KL | capacity | realization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | 01 | 10/10 | 19/20 | 92/100 | 0.128774 | 10.118101 | 0.030361 | 0.098413 | 0.007212 | 0.006010 | 4.878587 | 0.459 |
| Llama | 02 | 10/10 | 17/20 | 86/100 | 0.030848 | 13.831652 | 0.049866 | -0.019018 | 0.004980 | 0.112475 | 2.786369 | 0.613 |
| Llama | 03 | 10/10 | 19/20 | 87/100 | 0.015981 | 12.227769 | 0.047868 | -0.031888 | 0.004835 | 0.003083 | 3.011135 | 0.676 |
| Llama | 04 | 10/10 | 16/20 | 92/100 | 0.017834 | 9.869666 | 0.018625 | -0.000792 | 0.009281 | 0.057729 | 4.867644 | 0.506 |
| Llama | 05 | 10/10 | 17/20 | 86/100 | 0.260707 | 11.054918 | 0.201690 | 0.059017 | 0.006714 | 0.004577 | 4.195894 | 0.509 |
| Llama | 06 | 10/10 | 17/20 | 83/100 | 0.005390 | 14.400860 | 0.037999 | -0.032608 | 0.005283 | 0.006194 | 3.026039 | 0.593 |
| Llama | 07 | 9/10 | 15/20 | 95/100 | 0.827915 | 7.618181 | 0.013974 | 0.813941 | 0.010335 | 0.008598 | 6.464696 | 0.591 |
| Llama | 08 | 10/10 | 20/20 | 80/100 | 0.091056 | 10.665194 | 0.007113 | 0.083943 | 0.004528 | 0.002090 | 2.520978 | 0.443 |
| Llama | 09 | 10/10 | 19/20 | 92/100 | 0.004362 | 11.265950 | 0.005512 | -0.001150 | 0.006563 | 0.005845 | 3.558762 | 0.517 |
| Llama | 10 | 10/10 | 18/20 | 80/100 | 0.412498 | 13.062502 | 0.433843 | -0.021345 | 0.008734 | 0.006901 | 4.859827 | 0.581 |
| Qwen | 01 | 10/10 | 17/20 | 92/100 | 0.080707 | 14.488043 | 0.031787 | 0.048920 | 0.137750 | 0.000874 | 12.930951 | 0.332 |
| Qwen | 02 | 10/10 | 19/20 | 77/100 | 0.042227 | 16.317148 | 0.030914 | 0.011313 | 0.238672 | 0.184211 | 21.184255 | 0.280 |
| Qwen | 03 | 10/10 | 16/20 | 77/100 | 1.197165 | 10.973147 | 0.127907 | 1.069258 | 0.118436 | 0.001151 | 11.548501 | 0.266 |
| Qwen | 04 | 10/10 | 15/20 | 98/100 | 1.437869 | 10.005881 | 1.255147 | 0.182723 | 0.209317 | 0.001809 | 19.132561 | 0.321 |
| Qwen | 05 | 10/10 | 17/20 | 82/100 | 0.459452 | 9.606173 | 0.118081 | 0.341371 | 0.174815 | 0.001283 | 16.445096 | 0.220 |
| Qwen | 06 | 10/10 | 16/20 | 85/100 | 0.019517 | 13.558608 | 0.013242 | 0.006276 | 0.158125 | 0.002283 | 14.798712 | 0.252 |
| Qwen | 07 | 10/10 | 17/20 | 96/100 | 0.628380 | 11.790668 | 0.643217 | -0.014837 | 0.249406 | 0.001850 | 22.260072 | 0.296 |
| Qwen | 08 | 10/10 | 15/20 | 74/100 | 0.012604 | 12.012396 | 0.032543 | -0.019939 | 0.188998 | 0.001519 | 17.215773 | 0.304 |
| Qwen | 09 | 10/10 | 16/20 | 86/100 | 0.026056 | 12.523944 | 0.018600 | 0.007456 | 0.179044 | 0.001823 | 16.851990 | 0.277 |
| Qwen | 10 | 10/10 | 15/20 | 77/100 | 1.018053 | 9.894447 | 0.674253 | 0.343800 | 0.264835 | 0.000804 | 24.841683 | 0.278 |

## 5. 동일 batch 비교

| model | method | coverage | conditional E/G/L | Eff new NLL | Gen new NLL | Loc new NLL |
|---|---|---:|---:|---:|---:|---:|
| Llama | P1R24 RS-Soft | 10/10 | 99/100, 177/200, 873/1000 | 0.179537 | 2.377230 | 10.407288 |
| Llama | P1R23 RS-Soft | 9/10 | 90/90, 163/180, 788/900 | 0.118163 | 1.898085 | 10.524275 |
| Llama | Official AlphaEdit | 10/10 | 100/100, 185/200, 859/1000 | 0.001169 | 1.762698 | 10.153035 |
| Qwen | P1R24 RS-Soft | 10/10 | 100/100, 163/200, 844/1000 | 0.492203 | 3.599686 | 10.113464 |
| Qwen | P1R23 RS-Soft | 10/10 | 99/100, 164/200, 845/1000 | 0.332862 | 3.328141 | 10.129306 |
| Qwen | Official AlphaEdit | 10/10 | 100/100, 192/200, 828/1000 | 0.032646 | 1.772163 | 9.911450 |

- P1R23 Llama RS-Soft는 9/10만 endpoint가 있어 그 행은 성공 endpoint 조건부 집계다. P1R24의 열 번째 값을 채워 넣지 않았다.
- Llama P1R24는 99/100 Eff로 AlphaEdit 100/100에 1건 뒤지고, Gen 177/200 대 185/200, Loc 873/1000 대 859/1000이다.
- Qwen P1R24는 Eff 100/100으로 AlphaEdit와 같지만 Gen 163/200 대 192/200이며, Loc는 844/1000 대 828/1000이다. P1R23 RS-Soft와는 100/163/844 대 99/164/845로 거의 같은 count 수준이나 Eff/Gen 연속 NLL은 P1R24가 더 나쁘다.
- 원래 outcome-selected 단일 B10 job18983 RS-Soft는 Llama 10/20/81(E/G/L), Eff new NLL 0.026463; Qwen 10/20/80, Eff new NLL 0.021383이었다. 이번 10-batch 결과는 그 한 B10 복제가 아니라 별도 frozen stream의 분포 특성이다.

## 6. strength, realization, P

| model | rho/pred coverage | q mean | mean realization | P step-mean AUC | terminal P | P↔functional corr | target frozen rows/trajectory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | 1.000 | 9.326921 | 0.549 | 0.002628 | 0.006847 | -0.071 | 6.6 |
| Qwen | 1.000 | 14.089796 | 0.283 | 0.094658 | 0.191940 | 0.340 | 15.9 |

- 모든 160 step에서 predicted-strength equality와 positive-direction solve가 통과했다. typed scientific/numerical failure는 0/20 batches다.
- actual/predicted realization은 Llama 평균 0.549, Qwen 0.283으로 둘 다 under-realized이며 Qwen이 더 심하다. 따라서 WRITE_UNDER_REALIZED가 두 모델에 적용된다.
- cumulative Structural-P cross+self는 모든 case에 기록됐고 H는 완전 0이다. 다만 이 실험은 Soft-only여서 Neutral 대비 P 효과를 인과적으로 식별하지 못한다. P_NO_SIGNAL 또는 SOFT_UNDER_EDIT를 새로 확정하지 않는다.
- terminal Structural-P와 functional teacher-KL의 batch 간 상관은 관찰값일 뿐 동일-state proxy 정렬 증명이 아니다. 상관만으로 P_PROXY_MISMATCH를 단정하지 않는다.

## 7. compute

| model | mean edit-core/B10 | total edit-core | mean eval/B10 | mean F/B | mean tokens | job elapsed | MaxRSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | 93.299s | 932.986s | 2.411s | 180/125 | 28701 | 00:23:12 | 10,883,872 KiB |
| Qwen | 107.501s | 1075.009s | 2.259s | 180/125 | 26286 | 00:20:16 | 7,301,924 KiB |

- 각 B10은 180 model forwards, 125 backwards, materialization 8회를 receipt했다. model load는 장기 job당 1회이며 case별 edit-core 합과 scheduler elapsed는 서로 다른 집계 층위다.
- AlphaEdit와 formal pure-edit ratio는 phase counter 호환성이 없어 계산하지 않았다.

## 8. 판정

### FACT

- P1R24 RS-Soft 독립 B10 coverage는 Llama/Qwen 모두 10/10, W0 restore 20/20, history/H/retry 0이다.
- NUMERICAL_FAIL은 이번 replacement 결과에는 적용되지 않는다.
- absolute efficacy strength는 높지만 완전하지 않다: Llama 99/100, Qwen 100/100. Gen은 특히 Qwen에서 AlphaEdit보다 크게 낮다.
- 두 모델 모두 predicted demand 대비 실제 W-only 실현이 낮아 WRITE_UNDER_REALIZED다.

### INFERENCE

- Llama는 거의 전 batch에서 강한 efficacy를 회복했으나 batch07에서 9/10이고 Gen 총계가 AlphaEdit보다 낮다.
- Qwen은 efficacy count는 회복했지만 Gen 및 연속 NLL이 P1R23/AlphaEdit 대비 약해, target만의 약함보다 writer realization/일반화 문제가 더 유력하다.
- Soft-only 설계이므로 batch 간 P/functional-P 동행은 preservation의 원인 효과가 아니다.

### NOT_RECORDED

- 동일 10-batch P1R24 Neutral counterfactual, 따라서 matched-strength Soft-vs-Neutral 인과효과
- Official AlphaEdit와 호환되는 내부 F/B phase 및 formal pure-edit ratio
- fresh promotion seal, sequential/Historical evidence

## 9. Claim boundary

- 이 10-batch stream은 기존 outcome-selected 개발 계열을 재사용한 mechanistic/descriptive evidence다.
- 성공 endpoint 조건부 집계와 all-10 coverage를 분리했다. 실패 endpoint를 대체하거나 impute하지 않았다.
- 최종 신뢰성 요약: RS_SOFT_ATOMIC_EXECUTION_RELIABLE_20_OF_20, strength 요약: WRITE_UNDER_REALIZED, scientific_promotion=false.
