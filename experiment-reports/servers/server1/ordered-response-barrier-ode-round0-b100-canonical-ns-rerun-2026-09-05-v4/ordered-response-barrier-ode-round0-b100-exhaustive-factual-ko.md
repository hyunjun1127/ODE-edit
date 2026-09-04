# Ordered Response-Barrier ODE-Edit — Server1 round0 B100 기존 분석 + canonical-NS 재실행 통합 사실 보고서

## 0. 판정과 실행 경계

- 상태: `ANALYSIS_ONLY_TERMINAL_PASS`; canonical 실행은 Slurm array `36613`, round0 B100 네 cell이다.
- scheduler child: `36613_0→36614`, `36613_1→36615`, `36613_2→36840`, `36613_3→36613`; 모두 `COMPLETED/0:0`이다.
- canonical 과학 분모: `4 cells × 5 primary arms × 100 requests = 2,000 request-arm endpoints`; 입력 request는 cell별 100, 전체 실행 관측 400이다.
- `ORBHit`은 ORBFH 경로의 derived materialized prefix라 primary arm이나 2,000 분모에 중복 포함하지 않았다.
- 분석 중 새 model/GPU/Slurm/editing run/retry/imputation은 모두 0이며 remaining-nine은 HOLD다.
- 모든 비교는 동일 round0 request/order 안의 기술적·서술적 비교다. 인과, 보편성, 자동 promotion을 주장하지 않는다. `scientific_promotion=false`.

## 1. 지표 정의와 정확한 분모

- NLL은 teacher-forced target token의 평균 negative log likelihood다. `target-new`는 새 사실 정답, `target-true`는 원래 사실 정답이며 NLL 수치 자체는 낮을수록 해당 target에 더 높은 확률을 준다.
- `RS`는 100 rewrite pair에서 `NLL(new) < NLL(true)`인 strict count/rate다. tie는 failure다.
- `PS`는 200 rephrase prompt pair에서 같은 strict 비교를 한 count/rate다. `strict PS`는 request별 두 rephrase가 모두 성공한 100-request count/rate다.
- `canonical NS`는 1,000 neighborhood prompt pair에서 `NLL(target-true) < NLL(target-new)`인 strict count/rate다. tie는 failure다.
- `PP-token`은 endpoint token prediction이 PRE_EDIT과 같은 token 수/전체 target token 수다. target 길이 때문에 분모가 1,010이며 canonical NS가 아니다. `PP-prompt`의 분모는 1,000이다.
- rewrite/rephrase accuracy는 target token을 모두 맞힌 prompt 수로, pairwise RS/PS와 다른 secondary 지표다.
- PRE_EDIT은 endpoint 이전 W0 관측이며 이번 v2 evaluator에서 canonical NS 분모 1,000을 직접 기록했다. PRE_EDIT에는 전후 비교인 PP-token/PP-prompt가 적용되지 않는다.
- `ENTRY_ALREADY_HIT`은 계약상 정상 W0 no-op endpoint로 분모에 들어가지만 이번 네 cell에서는 0건이었다.

## 2. 핵심 endpoint 표 — PRE_EDIT, O, QCL, NQFIX, ORBFH, JAC

| model | writer | stage | status | RS | PS | strict PS | canonical NS | PP-token | RW new acc | RW true acc | RP new acc | RP true acc | LOC true acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | PRE_EDIT | ENTRY_REFERENCE | 13/100 (13.00%) | 28/200 (14.00%) | 8/100 (8.00%) | 893/1000 (89.30%) | N/A (entry reference) | 0/100 (0.00%) | 19/100 (19.00%) | 1/200 (0.50%) | 36/200 (18.00%) | 208/1000 (20.80%) |
| llama3-8b-inst | MEMIT | O | TERMINAL_VALID | 98/100 (98.00%) | 164/200 (82.00%) | 77/100 (77.00%) | 887/1000 (88.70%) | 951/1010 (94.16%) | 95/100 (95.00%) | 1/100 (1.00%) | 97/200 (48.50%) | 7/200 (3.50%) | 205/1000 (20.50%) |
| llama3-8b-inst | MEMIT | QCL | HORIZON_SEMANTIC_MISS | 96/100 (96.00%) | 156/200 (78.00%) | 69/100 (69.00%) | 888/1000 (88.80%) | 952/1010 (94.26%) | 89/100 (89.00%) | 1/100 (1.00%) | 88/200 (44.00%) | 8/200 (4.00%) | 206/1000 (20.60%) |
| llama3-8b-inst | MEMIT | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 183/200 (91.50%) | 87/100 (87.00%) | 880/1000 (88.00%) | 941/1010 (93.17%) | 100/100 (100.00%) | 0/100 (0.00%) | 114/200 (57.00%) | 6/200 (3.00%) | 202/1000 (20.20%) |
| llama3-8b-inst | MEMIT | ORBFH | HORIZON_SEMANTIC_MISS | 98/100 (98.00%) | 162/200 (81.00%) | 75/100 (75.00%) | 885/1000 (88.50%) | 940/1010 (93.07%) | 96/100 (96.00%) | 1/100 (1.00%) | 97/200 (48.50%) | 8/200 (4.00%) | 205/1000 (20.50%) |
| llama3-8b-inst | MEMIT | JAC | HORIZON_SEMANTIC_MISS | 98/100 (98.00%) | 163/200 (81.50%) | 76/100 (76.00%) | 883/1000 (88.30%) | 936/1010 (92.67%) | 96/100 (96.00%) | 1/100 (1.00%) | 96/200 (48.00%) | 8/200 (4.00%) | 204/1000 (20.40%) |
| llama3-8b-inst | AlphaEdit | PRE_EDIT | ENTRY_REFERENCE | 13/100 (13.00%) | 28/200 (14.00%) | 8/100 (8.00%) | 893/1000 (89.30%) | N/A (entry reference) | 0/100 (0.00%) | 19/100 (19.00%) | 1/200 (0.50%) | 36/200 (18.00%) | 208/1000 (20.80%) |
| llama3-8b-inst | AlphaEdit | O | TERMINAL_VALID | 100/100 (100.00%) | 184/200 (92.00%) | 89/100 (89.00%) | 866/1000 (86.60%) | 877/1010 (86.83%) | 100/100 (100.00%) | 0/100 (0.00%) | 120/200 (60.00%) | 7/200 (3.50%) | 210/1000 (21.00%) |
| llama3-8b-inst | AlphaEdit | QCL | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 169/200 (84.50%) | 79/100 (79.00%) | 874/1000 (87.40%) | 891/1010 (88.22%) | 99/100 (99.00%) | 0/100 (0.00%) | 108/200 (54.00%) | 7/200 (3.50%) | 206/1000 (20.60%) |
| llama3-8b-inst | AlphaEdit | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 93/100 (93.00%) | 859/1000 (85.90%) | 868/1010 (85.94%) | 100/100 (100.00%) | 0/100 (0.00%) | 126/200 (63.00%) | 6/200 (3.00%) | 209/1000 (20.90%) |
| llama3-8b-inst | AlphaEdit | ORBFH | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 168/200 (84.00%) | 79/100 (79.00%) | 875/1000 (87.50%) | 890/1010 (88.12%) | 99/100 (99.00%) | 0/100 (0.00%) | 107/200 (53.50%) | 7/200 (3.50%) | 206/1000 (20.60%) |
| llama3-8b-inst | AlphaEdit | JAC | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 169/200 (84.50%) | 79/100 (79.00%) | 872/1000 (87.20%) | 888/1010 (87.92%) | 100/100 (100.00%) | 0/100 (0.00%) | 107/200 (53.50%) | 7/200 (3.50%) | 208/1000 (20.80%) |
| qwen2.5-7b-inst | MEMIT | PRE_EDIT | ENTRY_REFERENCE | 11/100 (11.00%) | 39/200 (19.50%) | 10/100 (10.00%) | 849/1000 (84.90%) | N/A (entry reference) | 0/100 (0.00%) | 13/100 (13.00%) | 1/200 (0.50%) | 32/200 (16.00%) | 153/1000 (15.30%) |
| qwen2.5-7b-inst | MEMIT | O | TERMINAL_VALID | 100/100 (100.00%) | 192/200 (96.00%) | 95/100 (95.00%) | 826/1000 (82.60%) | 898/1010 (88.91%) | 100/100 (100.00%) | 0/100 (0.00%) | 130/200 (65.00%) | 2/200 (1.00%) | 146/1000 (14.60%) |
| qwen2.5-7b-inst | MEMIT | QCL | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 189/200 (94.50%) | 93/100 (93.00%) | 833/1000 (83.30%) | 905/1010 (89.60%) | 99/100 (99.00%) | 0/100 (0.00%) | 130/200 (65.00%) | 2/200 (1.00%) | 150/1000 (15.00%) |
| qwen2.5-7b-inst | MEMIT | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 189/200 (94.50%) | 93/100 (93.00%) | 815/1000 (81.50%) | 857/1010 (84.85%) | 100/100 (100.00%) | 0/100 (0.00%) | 132/200 (66.00%) | 2/200 (1.00%) | 127/1000 (12.70%) |
| qwen2.5-7b-inst | MEMIT | ORBFH | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 821/1000 (82.10%) | 889/1010 (88.02%) | 100/100 (100.00%) | 0/100 (0.00%) | 133/200 (66.50%) | 2/200 (1.00%) | 141/1000 (14.10%) |
| qwen2.5-7b-inst | MEMIT | JAC | HORIZON_SEMANTIC_MISS | 99/100 (99.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 818/1000 (81.80%) | 875/1010 (86.63%) | 98/100 (98.00%) | 0/100 (0.00%) | 131/200 (65.50%) | 2/200 (1.00%) | 141/1000 (14.10%) |
| qwen2.5-7b-inst | AlphaEdit | PRE_EDIT | ENTRY_REFERENCE | 11/100 (11.00%) | 39/200 (19.50%) | 10/100 (10.00%) | 849/1000 (84.90%) | N/A (entry reference) | 0/100 (0.00%) | 13/100 (13.00%) | 1/200 (0.50%) | 32/200 (16.00%) | 153/1000 (15.30%) |
| qwen2.5-7b-inst | AlphaEdit | O | TERMINAL_VALID | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 817/1000 (81.70%) | 866/1010 (85.74%) | 99/100 (99.00%) | 0/100 (0.00%) | 132/200 (66.00%) | 2/200 (1.00%) | 136/1000 (13.60%) |
| qwen2.5-7b-inst | AlphaEdit | QCL | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 823/1000 (82.30%) | 889/1010 (88.02%) | 99/100 (99.00%) | 0/100 (0.00%) | 130/200 (65.00%) | 2/200 (1.00%) | 150/1000 (15.00%) |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 814/1000 (81.40%) | 838/1010 (82.97%) | 99/100 (99.00%) | 0/100 (0.00%) | 133/200 (66.50%) | 2/200 (1.00%) | 124/1000 (12.40%) |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 822/1000 (82.20%) | 881/1010 (87.23%) | 99/100 (99.00%) | 0/100 (0.00%) | 134/200 (67.00%) | 2/200 (1.00%) | 142/1000 (14.20%) |
| qwen2.5-7b-inst | AlphaEdit | JAC | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 820/1000 (82.00%) | 877/1010 (86.83%) | 100/100 (100.00%) | 0/100 (0.00%) | 132/200 (66.00%) | 2/200 (1.00%) | 144/1000 (14.40%) |

각 cell에서 primary rate의 단순 최댓값(선택·promotion 규칙 아님):

| model | writer | metric | max arm(s) | rate |
| --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | RS | NQFIX | 100.00% |
| llama3-8b-inst | MEMIT | PS | NQFIX | 91.50% |
| llama3-8b-inst | MEMIT | canonical NS | QCL | 88.80% |
| llama3-8b-inst | AlphaEdit | RS | O,QCL,NQFIX,ORBFH,JAC | 100.00% |
| llama3-8b-inst | AlphaEdit | PS | NQFIX | 95.00% |
| llama3-8b-inst | AlphaEdit | canonical NS | ORBFH | 87.50% |
| qwen2.5-7b-inst | MEMIT | RS | O,QCL,NQFIX,ORBFH | 100.00% |
| qwen2.5-7b-inst | MEMIT | PS | O | 96.00% |
| qwen2.5-7b-inst | MEMIT | canonical NS | QCL | 83.30% |
| qwen2.5-7b-inst | AlphaEdit | RS | O,QCL,NQFIX,ORBFH,JAC | 100.00% |
| qwen2.5-7b-inst | AlphaEdit | PS | O,QCL,ORBFH | 95.50% |
| qwen2.5-7b-inst | AlphaEdit | canonical NS | QCL | 82.30% |

## 2.1 기존 v1/v2 보고서 통합과 재실행 parity

기존 `exhaustive-v1`은 mechanics·RS/PS·PP-token을 상세 분석했지만 endpoint locality target-new NLL이 없었다. `baseline-inclusive-v2`는 별도 PRE_EDIT canonical NS만 보완했으며 endpoint canonical NS는 schema gap으로 남았다. 이 통합판은 두 package를 immutable reference로 결속하고, 새 v2 evaluator raw에서 PRE_EDIT와 O/QCL/NQFIX/ORBFH/JAC 전체 canonical NS를 다시 계산했다.

| shared metric | paired rows | matched N/A | bitwise equal | raw nonzero | pinned equal | outside tol | Δ mean | Δ median | Δ p90 | max \|Δ\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rewrite_success_rate | 24 | 0 | 24 | 0 | 24 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| rephrase_success_rate | 24 | 0 | 21 | 3 | 24 | 0 | 1.388e-17 | 0.000e+00 | 7.772e-17 | 1.110e-16 |
| strict_rephrase_success_rate | 24 | 0 | 24 | 0 | 24 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| locality_prediction_preservation_rate | 20 | 4 | 9 | 11 | 20 | 0 | 3.886e-17 | 0.000e+00 | 1.110e-16 | 1.110e-16 |
| rewrite_target_new_accuracy_rate | 24 | 0 | 24 | 0 | 24 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| rewrite_target_true_accuracy_rate | 24 | 0 | 24 | 0 | 24 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| rephrase_target_new_accuracy_rate | 24 | 0 | 20 | 4 | 24 | 0 | 1.619e-17 | 0.000e+00 | 9.437e-17 | 1.110e-16 |
| rephrase_target_true_accuracy_rate | 24 | 0 | 20 | 4 | 24 | 0 | 1.532e-17 | 0.000e+00 | 8.327e-17 | 1.006e-16 |
| locality_target_true_accuracy_rate | 24 | 0 | 6 | 18 | 24 | 0 | 6.245e-17 | 8.327e-17 | 8.327e-17 | 8.327e-17 |

여기서 delta는 `canonical-NS rerun - 기존 exhaustive-v1`이다. `pinned equal`은 outcome-independent `32·eps_FP64·max(1,|old|,|new|)` envelope 안의 동등성이고, raw nonzero는 CSV decimal→binary 재파싱의 최하위 표현 차이까지 포함한다. canonical NS 자체는 기존 endpoint schema에 없었으므로 parity 대상으로 만들지 않았다. 기존 두 보고서와 manifest/receipt의 exact SHA는 §15 input inventory에 포함된다.

## 3. Rewrite NLL 분포

| model | writer | stage | new mean | new median | new p90 | new max | true mean | true median | true p90 | true max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | PRE_EDIT | 11.1027 | 10.9094 | 16.5967 | 20.8539 | 4.9430 | 4.1574 | 10.3774 | 15.2707 |
| llama3-8b-inst | MEMIT | O | 0.3951 | 0.0031 | 0.1950 | 13.5315 | 12.0018 | 12.0787 | 17.3699 | 22.5877 |
| llama3-8b-inst | MEMIT | QCL | 0.7116 | 0.0129 | 1.1631 | 14.4560 | 10.2374 | 10.1974 | 15.9626 | 20.8145 |
| llama3-8b-inst | MEMIT | NQFIX | 0.0169 | 0.0009 | 0.0055 | 0.9135 | 14.2715 | 13.6577 | 19.5415 | 25.7539 |
| llama3-8b-inst | MEMIT | ORBFH | 0.3693 | 0.0032 | 0.1642 | 13.5977 | 11.9208 | 11.8335 | 17.3629 | 22.8297 |
| llama3-8b-inst | MEMIT | JAC | 0.3722 | 0.0030 | 0.1599 | 13.7227 | 11.9372 | 11.8691 | 17.4279 | 22.1953 |
| llama3-8b-inst | AlphaEdit | PRE_EDIT | 11.1027 | 10.9094 | 16.5967 | 20.8539 | 4.9430 | 4.1574 | 10.3774 | 15.2707 |
| llama3-8b-inst | AlphaEdit | O | 0.0013 | 0.0006 | 0.0016 | 0.0178 | 14.9738 | 14.6476 | 20.4057 | 26.3722 |
| llama3-8b-inst | AlphaEdit | QCL | 0.0328 | 0.0015 | 0.0083 | 2.5831 | 13.3236 | 13.0246 | 18.9229 | 25.9647 |
| llama3-8b-inst | AlphaEdit | NQFIX | 0.0021 | 0.0006 | 0.0021 | 0.0387 | 14.7861 | 14.1299 | 20.0044 | 25.9565 |
| llama3-8b-inst | AlphaEdit | ORBFH | 0.0350 | 0.0015 | 0.0091 | 2.7753 | 13.2511 | 12.9921 | 18.7156 | 25.8251 |
| llama3-8b-inst | AlphaEdit | JAC | 0.0246 | 0.0016 | 0.0096 | 1.8075 | 13.3294 | 13.0900 | 18.6659 | 25.8763 |
| qwen2.5-7b-inst | MEMIT | PRE_EDIT | 10.3713 | 10.1680 | 15.4705 | 19.0751 | 5.3252 | 4.9493 | 9.5116 | 14.0711 |
| qwen2.5-7b-inst | MEMIT | O | 0.0390 | 0.0103 | 0.0467 | 1.1937 | 14.9166 | 15.5281 | 21.2534 | 29.3200 |
| qwen2.5-7b-inst | MEMIT | QCL | 0.0737 | 0.0153 | 0.0696 | 3.4048 | 14.7329 | 14.8783 | 21.2445 | 27.6073 |
| qwen2.5-7b-inst | MEMIT | NQFIX | 0.0331 | 0.0108 | 0.0641 | 0.6276 | 14.7718 | 15.2234 | 21.2852 | 30.8516 |
| qwen2.5-7b-inst | MEMIT | ORBFH | 0.0432 | 0.0099 | 0.0459 | 0.9926 | 14.8945 | 14.9800 | 21.1588 | 29.1891 |
| qwen2.5-7b-inst | MEMIT | JAC | 0.0779 | 0.0117 | 0.0479 | 3.0483 | 14.8741 | 14.8230 | 21.6275 | 29.1549 |
| qwen2.5-7b-inst | AlphaEdit | PRE_EDIT | 10.3713 | 10.1680 | 15.4705 | 19.0751 | 5.3252 | 4.9493 | 9.5116 | 14.0711 |
| qwen2.5-7b-inst | AlphaEdit | O | 0.0419 | 0.0113 | 0.0566 | 1.1355 | 14.7101 | 15.1598 | 20.7021 | 31.3564 |
| qwen2.5-7b-inst | AlphaEdit | QCL | 0.0416 | 0.0118 | 0.0472 | 1.1690 | 15.0584 | 15.1949 | 21.2051 | 29.8987 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | 0.0765 | 0.0105 | 0.0653 | 3.6044 | 14.5872 | 14.6441 | 20.6363 | 32.1757 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | 0.0378 | 0.0093 | 0.0416 | 1.2374 | 15.0188 | 15.3262 | 20.8045 | 30.4389 |
| qwen2.5-7b-inst | AlphaEdit | JAC | 0.0375 | 0.0102 | 0.0455 | 1.1636 | 15.0655 | 15.2011 | 20.5545 | 30.1299 |

## 4. Rephrase NLL 분포

| model | writer | stage | new mean | new median | new p90 | new max | true mean | true median | true p90 | true max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | PRE_EDIT | 9.8210 | 9.6705 | 14.6847 | 20.9183 | 5.0023 | 4.2287 | 9.8321 | 18.1133 |
| llama3-8b-inst | MEMIT | O | 3.0947 | 1.4865 | 8.9486 | 17.2183 | 7.9768 | 7.9385 | 13.3704 | 18.0923 |
| llama3-8b-inst | MEMIT | QCL | 3.6153 | 2.1668 | 10.1654 | 17.3806 | 7.2735 | 7.1518 | 12.4829 | 16.5134 |
| llama3-8b-inst | MEMIT | NQFIX | 2.1117 | 0.5892 | 6.1896 | 15.0221 | 9.1384 | 9.2774 | 14.1831 | 20.5744 |
| llama3-8b-inst | MEMIT | ORBFH | 3.1472 | 1.5287 | 8.9809 | 17.3481 | 7.8820 | 7.8540 | 13.2000 | 18.2441 |
| llama3-8b-inst | MEMIT | JAC | 3.1097 | 1.5098 | 8.9186 | 17.4019 | 7.9344 | 7.9606 | 13.1314 | 18.3269 |
| llama3-8b-inst | AlphaEdit | PRE_EDIT | 9.8210 | 9.6705 | 14.6847 | 20.9183 | 5.0023 | 4.2287 | 9.8321 | 18.1133 |
| llama3-8b-inst | AlphaEdit | O | 2.0318 | 0.4268 | 5.9239 | 15.7287 | 9.3790 | 9.4459 | 14.7782 | 21.4626 |
| llama3-8b-inst | AlphaEdit | QCL | 2.5862 | 0.9004 | 7.0584 | 16.6496 | 8.3972 | 8.4206 | 13.8247 | 19.5546 |
| llama3-8b-inst | AlphaEdit | NQFIX | 1.8499 | 0.4664 | 4.9635 | 14.9350 | 9.6012 | 9.6479 | 14.7525 | 21.1951 |
| llama3-8b-inst | AlphaEdit | ORBFH | 2.6447 | 0.9726 | 7.5881 | 16.7943 | 8.3181 | 8.2812 | 13.6782 | 19.6024 |
| llama3-8b-inst | AlphaEdit | JAC | 2.5932 | 0.9168 | 7.4264 | 16.8188 | 8.3904 | 8.3078 | 13.7559 | 19.5524 |
| qwen2.5-7b-inst | MEMIT | PRE_EDIT | 10.0196 | 10.0804 | 14.6812 | 17.7831 | 5.5393 | 4.7003 | 10.9602 | 16.5844 |
| qwen2.5-7b-inst | MEMIT | O | 1.9100 | 0.4195 | 6.0387 | 13.5393 | 11.6759 | 11.4646 | 17.7422 | 25.1206 |
| qwen2.5-7b-inst | MEMIT | QCL | 1.9569 | 0.4685 | 6.2993 | 12.9441 | 11.3621 | 11.1511 | 17.4006 | 25.1492 |
| qwen2.5-7b-inst | MEMIT | NQFIX | 1.9171 | 0.4210 | 6.3086 | 13.0912 | 11.7618 | 11.4632 | 17.9480 | 24.4320 |
| qwen2.5-7b-inst | MEMIT | ORBFH | 1.9112 | 0.3878 | 6.0406 | 13.8478 | 11.6900 | 11.4985 | 17.7778 | 25.0702 |
| qwen2.5-7b-inst | MEMIT | JAC | 1.9183 | 0.4069 | 6.1174 | 13.1626 | 11.5546 | 11.4294 | 17.6530 | 24.9623 |
| qwen2.5-7b-inst | AlphaEdit | PRE_EDIT | 10.0196 | 10.0804 | 14.6812 | 17.7831 | 5.5393 | 4.7003 | 10.9602 | 16.5844 |
| qwen2.5-7b-inst | AlphaEdit | O | 1.9398 | 0.4309 | 6.1332 | 13.3278 | 11.7392 | 11.5548 | 17.6553 | 24.4833 |
| qwen2.5-7b-inst | AlphaEdit | QCL | 1.8736 | 0.4053 | 5.9025 | 13.7232 | 11.7123 | 11.6733 | 17.7420 | 24.9734 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | 1.9685 | 0.4724 | 6.2174 | 12.2726 | 11.6945 | 11.6411 | 17.4665 | 24.4258 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | 1.8956 | 0.3998 | 6.2019 | 14.0841 | 11.7697 | 11.6518 | 17.6639 | 24.8505 |
| qwen2.5-7b-inst | AlphaEdit | JAC | 1.8697 | 0.4113 | 6.1607 | 13.9522 | 11.6934 | 11.5934 | 17.5532 | 25.0375 |

## 4.1 Neighborhood locality NLL 분포와 canonical NS 입력

| model | writer | stage | new mean | new median | new p90 | new max | true mean | true median | true p90 | true max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | PRE_EDIT | 10.8794 | 10.7644 | 15.8983 | 22.3823 | 4.8875 | 4.2978 | 9.9810 | 20.4680 |
| llama3-8b-inst | MEMIT | O | 10.5825 | 10.3759 | 15.7055 | 22.4458 | 4.7555 | 4.1811 | 9.8464 | 19.9395 |
| llama3-8b-inst | MEMIT | QCL | 10.5662 | 10.3941 | 15.6863 | 22.3752 | 4.7546 | 4.1608 | 9.8663 | 19.9453 |
| llama3-8b-inst | MEMIT | NQFIX | 10.4995 | 10.3942 | 15.7116 | 22.4487 | 4.7882 | 4.1557 | 9.9078 | 19.7709 |
| llama3-8b-inst | MEMIT | ORBFH | 10.5234 | 10.3634 | 15.6446 | 22.3474 | 4.7478 | 4.1631 | 9.8778 | 19.7695 |
| llama3-8b-inst | MEMIT | JAC | 10.4646 | 10.2685 | 15.6884 | 22.2820 | 4.7443 | 4.1557 | 9.9099 | 19.6985 |
| llama3-8b-inst | AlphaEdit | PRE_EDIT | 10.8794 | 10.7644 | 15.8983 | 22.3823 | 4.8875 | 4.2978 | 9.9810 | 20.4680 |
| llama3-8b-inst | AlphaEdit | O | 10.1265 | 10.0089 | 15.3344 | 22.5042 | 4.7087 | 4.0522 | 9.9323 | 18.6552 |
| llama3-8b-inst | AlphaEdit | QCL | 10.1650 | 10.0223 | 15.2957 | 22.3593 | 4.6929 | 4.1102 | 9.8768 | 19.0190 |
| llama3-8b-inst | AlphaEdit | NQFIX | 9.9962 | 9.8623 | 15.1775 | 22.0566 | 4.7641 | 4.1395 | 10.0363 | 17.9928 |
| llama3-8b-inst | AlphaEdit | ORBFH | 10.1689 | 10.0173 | 15.2911 | 22.2556 | 4.6840 | 4.1195 | 9.8560 | 18.9806 |
| llama3-8b-inst | AlphaEdit | JAC | 10.0695 | 9.8909 | 15.2988 | 22.2673 | 4.6922 | 4.1257 | 9.8016 | 18.7959 |
| qwen2.5-7b-inst | MEMIT | PRE_EDIT | 10.1990 | 10.0613 | 15.3509 | 22.1653 | 5.2400 | 4.5635 | 10.1612 | 21.8171 |
| qwen2.5-7b-inst | MEMIT | O | 9.7325 | 9.6773 | 14.8467 | 22.1568 | 5.3001 | 4.4398 | 10.2958 | 26.0146 |
| qwen2.5-7b-inst | MEMIT | QCL | 9.6738 | 9.6047 | 14.8008 | 22.5492 | 5.2735 | 4.5145 | 10.3386 | 25.0366 |
| qwen2.5-7b-inst | MEMIT | NQFIX | 9.6261 | 9.5330 | 14.7848 | 22.2483 | 5.3963 | 4.5915 | 10.3765 | 27.2687 |
| qwen2.5-7b-inst | MEMIT | ORBFH | 9.6239 | 9.5455 | 14.7440 | 22.3846 | 5.3413 | 4.4379 | 10.4407 | 25.5229 |
| qwen2.5-7b-inst | MEMIT | JAC | 9.4772 | 9.3379 | 14.5869 | 22.7716 | 5.3092 | 4.4632 | 10.3796 | 25.7496 |
| qwen2.5-7b-inst | AlphaEdit | PRE_EDIT | 10.1990 | 10.0613 | 15.3509 | 22.1653 | 5.2400 | 4.5635 | 10.1612 | 21.8171 |
| qwen2.5-7b-inst | AlphaEdit | O | 9.7895 | 9.6252 | 14.8583 | 22.3564 | 5.4296 | 4.6832 | 10.4263 | 27.3060 |
| qwen2.5-7b-inst | AlphaEdit | QCL | 9.7275 | 9.6843 | 14.7930 | 22.7851 | 5.3724 | 4.5692 | 10.4229 | 26.6879 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | 9.6862 | 9.4749 | 14.6671 | 21.8533 | 5.4573 | 4.7434 | 10.4079 | 26.9987 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | 9.7279 | 9.6479 | 14.9051 | 22.7604 | 5.3872 | 4.5352 | 10.5096 | 26.6613 |
| qwen2.5-7b-inst | AlphaEdit | JAC | 9.6043 | 9.5052 | 14.7192 | 23.2237 | 5.3561 | 4.4863 | 10.3501 | 25.0774 |

이 표의 각 stage는 target-new 1,000행과 target-true 1,000행을 같은 `(request, prompt_index)`로 결속한다. canonical NS count는 두 NLL의 strict 비교에서 직접 계산하며 token prediction-preservation과 섞지 않는다.

`prompt-nll.csv.gz`는 62,400개 PRE_EDIT/endpoint prompt 행을 모두 보존하고, `request-endpoint-metrics.csv.gz`는 2,400개 request-stage 행에서 rewrite, 두 rephrase, 열 neighborhood target-new/target-true 관측을 결속한다.

## 5. 동일 request의 Official O 대비 paired delta

delta는 `ours - O`다. NLL은 음수가 더 낮은 수치, target-new margin과 NS는 양수가 더 높은 수치다. equal은 tolerance를 만들지 않은 exact arithmetic equality다.

| model | writer | ours | metric | n | Δ mean | Δ median | Δ p90 | Δ max | better/equal/worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | QCL | rewrite_target_new_nll | 100 | 0.3165 | 0.0087 | 0.9267 | 5.8327 | 0/0/100 |
| llama3-8b-inst | MEMIT | QCL | rephrase_target_new_nll | 200 | 0.5206 | 0.2602 | 1.4142 | 3.6276 | 14/0/186 |
| llama3-8b-inst | MEMIT | QCL | rewrite_margin_true_minus_new | 100 | -2.0809 | -1.6345 | -0.8211 | -0.2008 | 0/0/100 |
| llama3-8b-inst | MEMIT | QCL | rephrase_margin_true_minus_new | 200 | -1.2239 | -1.0013 | -0.2247 | 0.5504 | 5/0/195 |
| llama3-8b-inst | MEMIT | QCL | locality_prediction_preservation_rate | 100 | 0.0010 | 0.0000 | 0.0000 | 0.1000 | 6/89/5 |
| llama3-8b-inst | MEMIT | QCL | canonical_ns_rate | 100 | 0.0010 | 0.0000 | 0.0000 | 0.1000 | 2/97/1 |
| llama3-8b-inst | MEMIT | NQFIX | rewrite_target_new_nll | 100 | -0.3782 | -0.0017 | -0.0002 | 0.0367 | 98/0/2 |
| llama3-8b-inst | MEMIT | NQFIX | rephrase_target_new_nll | 200 | -0.9830 | -0.2148 | 0.0117 | 0.7521 | 175/0/25 |
| llama3-8b-inst | MEMIT | NQFIX | rewrite_margin_true_minus_new | 100 | 2.6478 | 1.5162 | 5.2441 | 30.4521 | 99/0/1 |
| llama3-8b-inst | MEMIT | NQFIX | rephrase_margin_true_minus_new | 200 | 2.1446 | 1.5972 | 4.7653 | 12.6742 | 189/0/11 |
| llama3-8b-inst | MEMIT | NQFIX | locality_prediction_preservation_rate | 100 | -0.0100 | 0.0000 | 0.0000 | 0.2000 | 7/79/14 |
| llama3-8b-inst | MEMIT | NQFIX | canonical_ns_rate | 100 | -0.0070 | 0.0000 | 0.0000 | 0.0000 | 0/93/7 |
| llama3-8b-inst | MEMIT | ORBFH | rewrite_target_new_nll | 100 | -0.0258 | 0.0001 | 0.0010 | 0.3204 | 36/0/64 |
| llama3-8b-inst | MEMIT | ORBFH | rephrase_target_new_nll | 200 | 0.0526 | 0.0032 | 0.3987 | 1.3789 | 75/0/125 |
| llama3-8b-inst | MEMIT | ORBFH | rewrite_margin_true_minus_new | 100 | -0.0552 | -0.0534 | 0.3676 | 3.6673 | 41/0/59 |
| llama3-8b-inst | MEMIT | ORBFH | rephrase_margin_true_minus_new | 200 | -0.1473 | -0.1176 | 0.3595 | 1.2102 | 62/0/138 |
| llama3-8b-inst | MEMIT | ORBFH | locality_prediction_preservation_rate | 100 | -0.0110 | 0.0000 | 0.0000 | 0.1000 | 3/84/13 |
| llama3-8b-inst | MEMIT | ORBFH | canonical_ns_rate | 100 | -0.0020 | 0.0000 | 0.0000 | 0.0000 | 0/98/2 |
| llama3-8b-inst | MEMIT | JAC | rewrite_target_new_nll | 100 | -0.0230 | 0.0000 | 0.0027 | 0.4011 | 41/0/59 |
| llama3-8b-inst | MEMIT | JAC | rephrase_target_new_nll | 200 | 0.0150 | 0.0000 | 0.4219 | 1.7616 | 100/0/100 |
| llama3-8b-inst | MEMIT | JAC | rewrite_margin_true_minus_new | 100 | -0.0417 | -0.0210 | 0.4135 | 3.4213 | 48/0/52 |
| llama3-8b-inst | MEMIT | JAC | rephrase_margin_true_minus_new | 200 | -0.0574 | -0.0680 | 0.6066 | 2.5272 | 88/0/112 |
| llama3-8b-inst | MEMIT | JAC | locality_prediction_preservation_rate | 100 | -0.0150 | 0.0000 | 0.0000 | 0.1000 | 3/81/16 |
| llama3-8b-inst | MEMIT | JAC | canonical_ns_rate | 100 | -0.0040 | 0.0000 | 0.0000 | 0.0000 | 0/96/4 |
| llama3-8b-inst | AlphaEdit | QCL | rewrite_target_new_nll | 100 | 0.0316 | 0.0010 | 0.0062 | 2.5653 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | QCL | rephrase_target_new_nll | 200 | 0.5544 | 0.1215 | 1.8245 | 9.3421 | 19/0/181 |
| llama3-8b-inst | AlphaEdit | QCL | rewrite_margin_true_minus_new | 100 | -1.6818 | -1.3289 | -0.5046 | 0.0032 | 1/0/99 |
| llama3-8b-inst | AlphaEdit | QCL | rephrase_margin_true_minus_new | 200 | -1.5363 | -1.0816 | -0.2124 | 1.3558 | 13/0/187 |
| llama3-8b-inst | AlphaEdit | QCL | locality_prediction_preservation_rate | 100 | 0.0135 | 0.0000 | 0.1000 | 0.2000 | 20/72/8 |
| llama3-8b-inst | AlphaEdit | QCL | canonical_ns_rate | 100 | 0.0080 | 0.0000 | 0.0000 | 0.2000 | 9/89/2 |
| llama3-8b-inst | AlphaEdit | NQFIX | rewrite_target_new_nll | 100 | 0.0008 | 0.0000 | 0.0007 | 0.0285 | 30/1/69 |
| llama3-8b-inst | AlphaEdit | NQFIX | rephrase_target_new_nll | 200 | -0.1819 | -0.0015 | 0.2376 | 2.5441 | 121/0/79 |
| llama3-8b-inst | AlphaEdit | NQFIX | rewrite_margin_true_minus_new | 100 | -0.1885 | -0.0481 | 0.3938 | 0.9225 | 40/0/60 |
| llama3-8b-inst | AlphaEdit | NQFIX | rephrase_margin_true_minus_new | 200 | 0.4040 | 0.2732 | 1.6578 | 4.8747 | 128/0/72 |
| llama3-8b-inst | AlphaEdit | NQFIX | locality_prediction_preservation_rate | 100 | -0.0085 | 0.0000 | 0.0000 | 0.2000 | 8/74/18 |
| llama3-8b-inst | AlphaEdit | NQFIX | canonical_ns_rate | 100 | -0.0070 | 0.0000 | 0.0000 | 0.2000 | 5/84/11 |
| llama3-8b-inst | AlphaEdit | ORBFH | rewrite_target_new_nll | 100 | 0.0337 | 0.0009 | 0.0067 | 2.7576 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | ORBFH | rephrase_target_new_nll | 200 | 0.6129 | 0.1462 | 2.0497 | 9.9136 | 21/0/179 |
| llama3-8b-inst | AlphaEdit | ORBFH | rewrite_margin_true_minus_new | 100 | -1.7565 | -1.4285 | -0.5449 | -0.0372 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | ORBFH | rephrase_margin_true_minus_new | 200 | -1.6738 | -1.1838 | -0.2469 | 2.0703 | 11/0/189 |
| llama3-8b-inst | AlphaEdit | ORBFH | locality_prediction_preservation_rate | 100 | 0.0125 | 0.0000 | 0.1000 | 0.2000 | 19/73/8 |
| llama3-8b-inst | AlphaEdit | ORBFH | canonical_ns_rate | 100 | 0.0090 | 0.0000 | 0.0100 | 0.2000 | 10/88/2 |
| llama3-8b-inst | AlphaEdit | JAC | rewrite_target_new_nll | 100 | 0.0234 | 0.0008 | 0.0061 | 1.7898 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | JAC | rephrase_target_new_nll | 200 | 0.5615 | 0.0973 | 1.9930 | 9.7019 | 27/0/173 |
| llama3-8b-inst | AlphaEdit | JAC | rewrite_margin_true_minus_new | 100 | -1.6678 | -1.3166 | -0.5373 | -0.0282 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | JAC | rephrase_margin_true_minus_new | 200 | -1.5502 | -1.1117 | -0.0429 | 2.3496 | 20/0/180 |
| llama3-8b-inst | AlphaEdit | JAC | locality_prediction_preservation_rate | 100 | 0.0110 | 0.0000 | 0.1000 | 0.2000 | 22/66/12 |
| llama3-8b-inst | AlphaEdit | JAC | canonical_ns_rate | 100 | 0.0060 | 0.0000 | 0.0000 | 0.2000 | 8/89/3 |
| qwen2.5-7b-inst | MEMIT | QCL | rewrite_target_new_nll | 100 | 0.0347 | 0.0005 | 0.0140 | 2.8970 | 40/0/60 |
| qwen2.5-7b-inst | MEMIT | QCL | rephrase_target_new_nll | 200 | 0.0469 | 0.0052 | 0.6295 | 4.0418 | 88/0/112 |
| qwen2.5-7b-inst | MEMIT | QCL | rewrite_margin_true_minus_new | 100 | -0.2185 | -0.1480 | 1.0086 | 4.0985 | 45/0/55 |
| qwen2.5-7b-inst | MEMIT | QCL | rephrase_margin_true_minus_new | 200 | -0.3607 | -0.2894 | 0.9518 | 3.4942 | 71/0/129 |
| qwen2.5-7b-inst | MEMIT | QCL | locality_prediction_preservation_rate | 100 | 0.0070 | 0.0000 | 0.1000 | 0.4000 | 15/71/14 |
| qwen2.5-7b-inst | MEMIT | QCL | canonical_ns_rate | 100 | 0.0070 | 0.0000 | 0.0000 | 0.3000 | 7/90/3 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rewrite_target_new_nll | 100 | -0.0059 | 0.0000 | 0.0119 | 0.1735 | 50/0/50 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rephrase_target_new_nll | 200 | 0.0071 | -0.0008 | 0.3717 | 2.1207 | 103/0/97 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rewrite_margin_true_minus_new | 100 | -0.1389 | -0.1397 | 0.8079 | 2.5106 | 37/0/63 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rephrase_margin_true_minus_new | 200 | 0.0787 | 0.0603 | 1.0226 | 4.9216 | 105/0/95 |
| qwen2.5-7b-inst | MEMIT | NQFIX | locality_prediction_preservation_rate | 100 | -0.0405 | 0.0000 | 0.0000 | 0.1000 | 8/61/31 |
| qwen2.5-7b-inst | MEMIT | NQFIX | canonical_ns_rate | 100 | -0.0110 | 0.0000 | 0.0000 | 0.1000 | 4/89/7 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rewrite_target_new_nll | 100 | 0.0042 | -0.0001 | 0.0053 | 0.4547 | 54/0/46 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rephrase_target_new_nll | 200 | 0.0011 | 0.0001 | 0.2021 | 4.1748 | 100/0/100 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rewrite_margin_true_minus_new | 100 | -0.0263 | 0.0009 | 0.5839 | 1.4701 | 51/0/49 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rephrase_margin_true_minus_new | 200 | 0.0129 | -0.0398 | 0.6678 | 3.0930 | 95/0/105 |
| qwen2.5-7b-inst | MEMIT | ORBFH | locality_prediction_preservation_rate | 100 | -0.0090 | 0.0000 | 0.1000 | 0.1000 | 12/70/18 |
| qwen2.5-7b-inst | MEMIT | ORBFH | canonical_ns_rate | 100 | -0.0050 | 0.0000 | 0.0000 | 0.1000 | 2/93/5 |
| qwen2.5-7b-inst | MEMIT | JAC | rewrite_target_new_nll | 100 | 0.0389 | -0.0000 | 0.0136 | 3.0114 | 51/0/49 |
| qwen2.5-7b-inst | MEMIT | JAC | rephrase_target_new_nll | 200 | 0.0083 | 0.0004 | 0.4050 | 3.9475 | 98/0/102 |
| qwen2.5-7b-inst | MEMIT | JAC | rewrite_margin_true_minus_new | 100 | -0.0814 | -0.0292 | 1.1552 | 2.8048 | 47/0/53 |
| qwen2.5-7b-inst | MEMIT | JAC | rephrase_margin_true_minus_new | 200 | -0.1296 | -0.1393 | 0.9429 | 3.9019 | 81/0/119 |
| qwen2.5-7b-inst | MEMIT | JAC | locality_prediction_preservation_rate | 100 | -0.0230 | 0.0000 | 0.1000 | 0.4000 | 15/56/29 |
| qwen2.5-7b-inst | MEMIT | JAC | canonical_ns_rate | 100 | -0.0080 | 0.0000 | 0.0000 | 0.3000 | 3/88/9 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rewrite_target_new_nll | 100 | -0.0003 | -0.0005 | 0.0054 | 0.4113 | 62/0/38 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rephrase_target_new_nll | 200 | -0.0662 | -0.0011 | 0.3822 | 1.6652 | 107/0/93 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rewrite_margin_true_minus_new | 100 | 0.3486 | 0.3568 | 1.4520 | 4.8924 | 67/0/33 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rephrase_margin_true_minus_new | 200 | 0.0393 | -0.0580 | 1.3818 | 5.2916 | 93/0/107 |
| qwen2.5-7b-inst | AlphaEdit | QCL | locality_prediction_preservation_rate | 100 | 0.0225 | 0.0000 | 0.1000 | 0.2000 | 34/51/15 |
| qwen2.5-7b-inst | AlphaEdit | QCL | canonical_ns_rate | 100 | 0.0060 | 0.0000 | 0.1000 | 0.2000 | 11/83/6 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rewrite_target_new_nll | 100 | 0.0346 | -0.0000 | 0.0112 | 2.8468 | 52/0/48 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rephrase_target_new_nll | 200 | 0.0287 | 0.0002 | 0.3016 | 5.0587 | 97/0/103 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rewrite_margin_true_minus_new | 100 | -0.1574 | 0.0455 | 0.5691 | 2.2383 | 54/0/46 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rephrase_margin_true_minus_new | 200 | -0.0735 | -0.0484 | 0.8013 | 2.3317 | 93/0/107 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | locality_prediction_preservation_rate | 100 | -0.0275 | 0.0000 | 0.0000 | 0.2000 | 9/64/27 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | canonical_ns_rate | 100 | -0.0030 | 0.0000 | 0.0000 | 0.1000 | 6/85/9 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rewrite_target_new_nll | 100 | -0.0041 | -0.0005 | 0.0038 | 0.1019 | 63/0/37 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rephrase_target_new_nll | 200 | -0.0442 | -0.0009 | 0.3599 | 2.0836 | 106/0/94 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rewrite_margin_true_minus_new | 100 | 0.3128 | 0.2674 | 1.3032 | 3.4084 | 65/0/35 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rephrase_margin_true_minus_new | 200 | 0.0746 | -0.0546 | 1.0913 | 5.2884 | 93/0/107 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | locality_prediction_preservation_rate | 100 | 0.0145 | 0.0000 | 0.1000 | 0.2000 | 29/56/15 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | canonical_ns_rate | 100 | 0.0050 | 0.0000 | 0.0100 | 0.2000 | 10/83/7 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rewrite_target_new_nll | 100 | -0.0044 | -0.0006 | 0.0077 | 0.4060 | 64/0/36 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rephrase_target_new_nll | 200 | -0.0701 | -0.0007 | 0.4533 | 2.0099 | 104/0/96 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rewrite_margin_true_minus_new | 100 | 0.3598 | 0.2843 | 1.8825 | 4.1917 | 62/0/38 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rephrase_margin_true_minus_new | 200 | 0.0243 | -0.0576 | 1.4118 | 7.5473 | 92/0/108 |
| qwen2.5-7b-inst | AlphaEdit | JAC | locality_prediction_preservation_rate | 100 | 0.0105 | 0.0000 | 0.1000 | 0.4000 | 29/48/23 |
| qwen2.5-7b-inst | AlphaEdit | JAC | canonical_ns_rate | 100 | 0.0030 | 0.0000 | 0.1000 | 0.3000 | 11/79/10 |

모든 target-new/target-true NLL, rewrite/rephrase margin, canonical NS와 PP-token의 128개 paired summary는 `paired-official-deltas.csv`; QCL→NQFIX→ORBFH→JAC의 96개 predeclared adjacent contrast는 `method-contrast-deltas.csv`에 있다.

## 6. 다섯 primary arm의 구현상 차이

- `O`: stock Official MEMIT 또는 AlphaEdit wrapper. B100 family-native fixed z 100개를 cohort entry에서 계산하고 stock entrypoint를 한 번 실행한다.
- `QCL`: 각 sweep의 L4→L8 visit마다 current residual을 remaining-layer count `n_l=(5,4,3,2,1)`로 나눈 `R/n_l`, fixed `u=1`, per-visit rebuild.
- `NQFIX`: 같은 20-visit clock에서 divisor를 제거한 full current residual `R`, fixed `u=1`, per-visit rebuild.
- `ORBFH`: full current residual과 current response-derived `u(W)`, 실제 upstream transition 뒤 downstream factor/key/response를 per visit rebuild.
- `JAC`: 각 sweep entry에서 full residual과 response `u(W)`를 정하고, sweep 내부에서는 그 command reference를 유지하는 per-sweep arm. fully entry-frozen arm은 아니다.
- 이 차이는 matched descriptive attribution axis일 뿐, 단일 round0 결과만으로 causal layer importance나 universal superiority를 주장하지 않는다.

## 7. Mechanism terminal 및 hit 사실

| model | writer | arm | status | endpoint | steps | nonzero | zero | terminal strict req | global hit | \|\|ΔW\|\|F | path \|\|·\|\|F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | O | TERMINAL_VALID | TERMINAL_VALID | 0 | 0 | 0 | 88 | NONE | 8.6915 | N/A |
| llama3-8b-inst | MEMIT | QCL | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 80 | NONE | 7.3626 | 7.7799 |
| llama3-8b-inst | MEMIT | NQFIX | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 97 | NONE | 12.9892 | 15.1458 |
| llama3-8b-inst | MEMIT | ORBFH | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 88 | NONE | 8.4611 | 9.3820 |
| llama3-8b-inst | MEMIT | JAC | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 18 | 2 | 88 | NONE | 8.6589 | 10.2720 |
| llama3-8b-inst | AlphaEdit | O | TERMINAL_VALID | TERMINAL_VALID | 0 | 0 | 0 | 99 | NONE | 9.8190 | N/A |
| llama3-8b-inst | AlphaEdit | QCL | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 93 | NONE | 7.5698 | 8.4890 |
| llama3-8b-inst | AlphaEdit | NQFIX | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 99 | NONE | 11.9105 | 15.5449 |
| llama3-8b-inst | AlphaEdit | ORBFH | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 91 | NONE | 7.5232 | 8.6508 |
| llama3-8b-inst | AlphaEdit | JAC | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 17 | 3 | 92 | NONE | 7.9246 | 9.9474 |
| qwen2.5-7b-inst | MEMIT | O | TERMINAL_VALID | TERMINAL_VALID | 0 | 0 | 0 | 97 | NONE | 72.0947 | N/A |
| qwen2.5-7b-inst | MEMIT | QCL | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 97 | NONE | 51.1288 | 56.2098 |
| qwen2.5-7b-inst | MEMIT | NQFIX | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 96 | NONE | 134.4719 | 180.5156 |
| qwen2.5-7b-inst | MEMIT | ORBFH | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 98 | NONE | 53.1411 | 65.5246 |
| qwen2.5-7b-inst | MEMIT | JAC | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 16 | 4 | 96 | NONE | 45.8523 | 64.8308 |
| qwen2.5-7b-inst | AlphaEdit | O | TERMINAL_VALID | TERMINAL_VALID | 0 | 0 | 0 | 98 | NONE | 47.7971 | N/A |
| qwen2.5-7b-inst | AlphaEdit | QCL | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 98 | NONE | 41.9012 | 50.3831 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 20 | 0 | 94 | NONE | 56.9870 | 89.3218 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 19 | 1 | 99 | NONE | 41.0694 | 50.9640 |
| qwen2.5-7b-inst | AlphaEdit | JAC | HORIZON_SEMANTIC_MISS | TERMINAL_VALID | 20 | 15 | 5 | 98 | NONE | 49.5245 | 67.0492 |

16개 dynamic endpoint 모두 20 visits를 완료하고 기술적으로 valid endpoint를 냈지만, 100 requests가 동시에 strict가 되는 global prefix는 한 번도 없었다. 따라서 16/16 상태는 `HORIZON_SEMANTIC_MISS`; 이는 계약상 과학 관측이지 기술 실패가 아니다. JAC 및 한 Qwen AlphaEdit ORBFH에서 zero-command visit이 기록됐고 나머지는 nonzero였다. per-step strict-request count와 residual 궤적은 `mechanism-step-summary.csv` 및 그림에 있다. per-request first-hit identity는 raw schema에 없어 재구성하지 않았다.

## 8. Layer별 terminal net update energy share (%)

| model | writer | arm | L4 | L5 | L6 | L7 | L8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | JAC | 0.75 | 3.26 | 13.28 | 42.88 | 39.82 |
| llama3-8b-inst | MEMIT | NQFIX | 32.39 | 23.05 | 18.23 | 14.97 | 11.35 |
| llama3-8b-inst | MEMIT | O | 9.09 | 9.03 | 11.95 | 20.25 | 49.69 |
| llama3-8b-inst | MEMIT | ORBFH | 1.65 | 5.34 | 15.81 | 42.99 | 34.21 |
| llama3-8b-inst | MEMIT | QCL | 6.07 | 6.50 | 9.28 | 18.16 | 59.99 |
| llama3-8b-inst | AlphaEdit | JAC | 0.54 | 1.73 | 6.49 | 26.06 | 65.18 |
| llama3-8b-inst | AlphaEdit | NQFIX | 29.31 | 23.41 | 19.30 | 16.41 | 11.56 |
| llama3-8b-inst | AlphaEdit | O | 8.71 | 9.52 | 13.28 | 23.08 | 45.41 |
| llama3-8b-inst | AlphaEdit | ORBFH | 1.09 | 3.24 | 9.01 | 28.92 | 57.74 |
| llama3-8b-inst | AlphaEdit | QCL | 4.73 | 5.51 | 8.36 | 18.50 | 62.90 |
| qwen2.5-7b-inst | MEMIT | JAC | 16.68 | 4.47 | 11.98 | 37.17 | 29.71 |
| qwen2.5-7b-inst | MEMIT | NQFIX | 81.32 | 14.69 | 1.75 | 1.47 | 0.77 |
| qwen2.5-7b-inst | MEMIT | O | 61.59 | 18.22 | 4.15 | 7.05 | 8.99 |
| qwen2.5-7b-inst | MEMIT | ORBFH | 43.15 | 12.71 | 10.89 | 20.19 | 13.06 |
| qwen2.5-7b-inst | MEMIT | QCL | 48.43 | 13.34 | 3.84 | 9.09 | 25.30 |
| qwen2.5-7b-inst | AlphaEdit | JAC | 1.54 | 2.27 | 7.47 | 39.19 | 49.52 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | 51.85 | 31.32 | 6.56 | 7.03 | 3.25 |
| qwen2.5-7b-inst | AlphaEdit | O | 24.55 | 31.32 | 9.89 | 18.71 | 15.54 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | 6.33 | 10.55 | 14.66 | 34.42 | 34.04 |
| qwen2.5-7b-inst | AlphaEdit | QCL | 8.94 | 15.68 | 6.22 | 19.02 | 50.14 |

표는 서로 다른 parameter block의 squared Frobenius terminal-net share다. 모델/방법 사이 raw residual norm을 직접 비교하지 않았으며, absolute update와 path energy는 `layer-update-summary.csv`에 별도로 있다.

## 9. Compute/time/memory

| model | writer | arm | wall s | Δs vs O | ratio vs O | forwards | keys | builds | solves | JVP | writes | JVP s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | O | 173.82 | 0.00 | 1.000 | 217 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| llama3-8b-inst | MEMIT | QCL | 663.13 | 489.31 | 3.815 | 528 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | MEMIT | NQFIX | 660.03 | 486.21 | 3.797 | 528 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | MEMIT | ORBFH | 742.67 | 568.84 | 4.273 | 548 | 20 | 20 | 20 | 20 | 1 | 66.35 |
| llama3-8b-inst | MEMIT | JAC | 691.82 | 518.00 | 3.980 | 510 | 20 | 20 | 20 | 20 | 1 | 66.18 |
| llama3-8b-inst | AlphaEdit | O | 221.32 | 0.00 | 1.000 | 242 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| llama3-8b-inst | AlphaEdit | QCL | 738.58 | 517.26 | 3.337 | 553 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | AlphaEdit | NQFIX | 735.05 | 513.73 | 3.321 | 553 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | AlphaEdit | ORBFH | 817.24 | 595.92 | 3.693 | 573 | 20 | 20 | 20 | 20 | 1 | 66.90 |
| llama3-8b-inst | AlphaEdit | JAC | 762.16 | 540.84 | 3.444 | 524 | 20 | 20 | 20 | 20 | 1 | 66.67 |
| qwen2.5-7b-inst | MEMIT | O | 195.75 | 0.00 | 1.000 | 217 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | MEMIT | QCL | 650.58 | 454.83 | 3.323 | 528 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | MEMIT | NQFIX | 657.70 | 461.95 | 3.360 | 528 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | MEMIT | ORBFH | 723.85 | 528.10 | 3.698 | 548 | 20 | 20 | 20 | 20 | 1 | 57.49 |
| qwen2.5-7b-inst | MEMIT | JAC | 644.66 | 448.90 | 3.293 | 488 | 20 | 20 | 20 | 20 | 1 | 57.30 |
| qwen2.5-7b-inst | AlphaEdit | O | 218.24 | 0.00 | 1.000 | 242 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | AlphaEdit | QCL | 740.00 | 521.76 | 3.391 | 553 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | 730.95 | 512.71 | 3.349 | 553 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | 782.61 | 564.37 | 3.586 | 562 | 20 | 20 | 20 | 20 | 1 | 57.60 |
| qwen2.5-7b-inst | AlphaEdit | JAC | 695.88 | 477.64 | 3.189 | 502 | 20 | 20 | 20 | 20 | 1 | 57.26 |

위 표의 `wall s`는 endpoint edit-core 관측 시간이고 `Δs/ratio`는 같은 model/writer cell의 Official O 대비다. JVP 시간은 endpoint wall의 구성 요소이며 별도 합산하지 않는다.

| model | writer | arm | terminal captures | stock logical solves | stock actual solves | JVP forwards | shadow mats | endpoint eval | derived eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | O | 0 | 5.0 | NOT_INSTRUMENTED_STOCK_SOURCE | 0 | nan | 1 | 0 |
| llama3-8b-inst | MEMIT | QCL | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| llama3-8b-inst | MEMIT | NQFIX | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| llama3-8b-inst | MEMIT | ORBFH | 41 | nan | N/A | 20 | 1.0 | 1 | 0 |
| llama3-8b-inst | MEMIT | JAC | 23 | nan | N/A | 20 | 1.0 | 1 | 0 |
| llama3-8b-inst | AlphaEdit | O | 0 | 5.0 | NOT_INSTRUMENTED_STOCK_SOURCE | 0 | nan | 1 | 0 |
| llama3-8b-inst | AlphaEdit | QCL | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| llama3-8b-inst | AlphaEdit | NQFIX | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| llama3-8b-inst | AlphaEdit | ORBFH | 41 | nan | N/A | 20 | 1.0 | 1 | 0 |
| llama3-8b-inst | AlphaEdit | JAC | 22 | nan | N/A | 20 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | MEMIT | O | 0 | 5.0 | NOT_INSTRUMENTED_STOCK_SOURCE | 0 | nan | 1 | 0 |
| qwen2.5-7b-inst | MEMIT | QCL | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | MEMIT | NQFIX | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | MEMIT | ORBFH | 41 | nan | N/A | 20 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | MEMIT | JAC | 21 | nan | N/A | 20 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | AlphaEdit | O | 0 | 5.0 | NOT_INSTRUMENTED_STOCK_SOURCE | 0 | nan | 1 | 0 |
| qwen2.5-7b-inst | AlphaEdit | QCL | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | 41 | nan | N/A | 0 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | 40 | nan | N/A | 20 | 1.0 | 1 | 0 |
| qwen2.5-7b-inst | AlphaEdit | JAC | 20 | nan | N/A | 20 | 1.0 | 1 | 0 |

| model | writer | model load s | job total s | peak alloc GiB | peak reserved GiB | arm forwards sum | arm JVP sum | arm wall sum s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | 3.61 | 4935.70 | 39.494 | 44.070 | 2331 | 40 | 2931.47 |
| llama3-8b-inst | AlphaEdit | 3.77 | 5314.89 | 37.525 | 40.402 | 2445 | 40 | 3274.34 |
| qwen2.5-7b-inst | MEMIT | 3.91 | 5005.96 | 42.612 | 46.693 | 2309 | 40 | 2872.55 |
| qwen2.5-7b-inst | AlphaEdit | 3.65 | 5294.55 | 39.439 | 41.650 | 2412 | 40 | 3167.68 |

Official stock 내부 actual solve call은 intercept하지 않아 logical expected 5만 기록됐고, dynamic arms는 adapter-intercepted 20 solves/builds가 기록됐다. `endpoint eval`은 각 primary endpoint의 evaluator 호출, `derived eval`은 ORBHit observation-only 평가다. canonical NS v2는 같은 1,000 neighborhood prompt에 target-new와 target-true를 모두 실행하므로 그 비용은 현재 job의 actual model-forward/wall 카운터에 포함된다. 이전 schema와의 FLOP 차이를 시간으로 역산하지 않는다.

peak GPU memory는 arm별이 아니라 cell-level allocated/reserved다: llama3-8b-inst/MEMIT 42.41/47.32GB; llama3-8b-inst/AlphaEdit 40.29/43.38GB; qwen2.5-7b-inst/MEMIT 45.75/50.14GB; qwen2.5-7b-inst/AlphaEdit 42.35/44.72GB. model load는 cell별 1회이며 wave 분리 시 reload가 필요하다. exact FLOPs는 raw schema에 없으므로 만들지 않았고, 실제 forward/key/factor/solve/JVP/capture/write/materialization/evaluator 호출 수와 CUDA 동기화된 wall time을 계산량의 재현 가능한 대리 계정으로 보고한다.

## 10. Runtime B1 preamble와 integrity

| model | writer | status | stock fidelity | FD pass | min cosine | max abs err | changed-state key | inner mutation | W0 restore | ORBHit audit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | FAST_RUNTIME_PREAMBLE_PASS | True | 3/3 | 0.9999993 | 0.00009231 | True | 0 | True | ORBHit_IDENTITY_PASS |
| llama3-8b-inst | AlphaEdit | FAST_RUNTIME_PREAMBLE_PASS | True | 3/3 | 0.9999997 | 0.00011851 | True | 0 | True | ORBHit_IDENTITY_PASS |
| qwen2.5-7b-inst | MEMIT | FAST_RUNTIME_PREAMBLE_PASS | True | 3/3 | 0.9999996 | 0.00366211 | True | 0 | True | ORBHit_IDENTITY_PASS |
| qwen2.5-7b-inst | AlphaEdit | FAST_RUNTIME_PREAMBLE_PASS | True | 3/3 | 0.9999999 | 0.00150764 | True | 0 | True | ORBHit_IDENTITY_PASS |

모든 cell은 stock O wrapper↔direct exact fidelity, multi-epsilon JVP/FD, changed-state observation, overlay/shadow/transaction, ORBHit direct-prefix audit를 통과했다. W0 pointer/bytes, cache/method-state content, arm contamination, forbidden evaluator influence, dynamic-z, retry, inner history/cache mutation gate도 PASS/0이다.

FULL-FP32 claim scope는 `MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH`다. 모든 model parameter/storage와 ORBODE controller/write path는 FP32, autocast/TF32/BF16/FP16/quantization/numeric storage cast는 0이다. MEMIT 두 cell은 stock native solve가 일시적 FP64이므로 `unqualified all-algorithm FP32`는 false이며 receipt가 이를 명시한다; AlphaEdit 두 cell은 true다.

MEMIT의 method state는 `MEMIT_STATIC_COV_CACHE_CONTENT_SHA256`; AlphaEdit은 `ALPHA_CACHE_C_CONTENT_SHA256`이다. 모든 arm에서 entry/after content identity가 같고 inner append/mutation은 0이다. AlphaEdit primary endpoint당 terminal history append 1이 관측되며 transaction 종료 후 entry content로 restore됐다. MEMIT을 Alpha cache로 부르지 않는다.

## 11. 기술 시도, B1 pilot, canonical B100 계보

| parent | task | child | namespace | state | exit | source | denominator | exception | detail | W0 restore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 36603 | 0 | 36604 | round0-tech-r1 | FAILED | 2:0 | 15fd95798216 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | - | - | NOT_RECORDED_CANCELLED_PARTIAL |
| 36603 | 1 | 36605 | round0-tech-r1 | FAILED | 2:0 | 15fd95798216 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | - | - | NOT_RECORDED_CANCELLED_PARTIAL |
| 36603 | 2 | 36608 | round0-tech-r1 | FAILED | 2:0 | 15fd95798216 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | - | - | NOT_RECORDED_CANCELLED_PARTIAL |
| 36603 | 3 | 36603 | round0-tech-r1 | FAILED | 2:0 | 15fd95798216 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | - | - | NOT_RECORDED_CANCELLED_PARTIAL |
| 36415 | 0 | 36416 | b1-tech-r1 | COMPLETED | 0:0 | 15fd95798216 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 36415 | 1 | 36417 | b1-tech-r1 | COMPLETED | 0:0 | 15fd95798216 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 36415 | 2 | 36490 | b1-tech-r1 | COMPLETED | 0:0 | 15fd95798216 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 36415 | 3 | 36415 | b1-tech-r1 | COMPLETED | 0:0 | 15fd95798216 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 36613 | 0 | 36614 | round0-tech-r2 | COMPLETED | 0:0 | 15fd95798216 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |
| 36613 | 1 | 36615 | round0-tech-r2 | COMPLETED | 0:0 | 15fd95798216 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |
| 36613 | 2 | 36840 | round0-tech-r2 | COMPLETED | 0:0 | 15fd95798216 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |
| 36613 | 3 | 36613 | round0-tech-r2 | COMPLETED | 0:0 | 15fd95798216 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |

이전 기술 시도와 B1 pilot은 B100 과학 분모에서 제외했다. canonical 과학 분모는 source `15fd95798216…`, job `36613`만 사용한다.

## 12. ORBHit derived endpoint

| model | writer | status | factors | RS | PS | strict PS | canonical NS | PP-token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | HORIZON_SEMANTIC_MISS | 20 | 98/100 (98.00%) | 162/200 (81.00%) | 75/100 (75.00%) | 885/1000 (88.50%) | 940/1010 (93.07%) |
| llama3-8b-inst | AlphaEdit | HORIZON_SEMANTIC_MISS | 20 | 100/100 (100.00%) | 168/200 (84.00%) | 79/100 (79.00%) | 875/1000 (87.50%) | 890/1010 (88.12%) |
| qwen2.5-7b-inst | MEMIT | HORIZON_SEMANTIC_MISS | 20 | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 821/1000 (82.10%) | 889/1010 (88.02%) |
| qwen2.5-7b-inst | AlphaEdit | HORIZON_SEMANTIC_MISS | 19 | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 822/1000 (82.20%) | 881/1010 (87.23%) |

global hit이 없었으므로 derived prefix는 horizon prefix를 materialize했다. 이 표는 observation-only이며 primary arm 분모와 aggregate에 합치지 않았다.

## 13. 기록 한계와 금지된 추정

| field | status |
| --- | --- |
| exact FLOPs | NOT_RECORDED_SCHEMA_GAP; 호출 수와 actual wall time만 보고 |
| per-layer factor condition number | NOT_RECORDED_SCHEMA_GAP |
| per-layer SVD spectrum | NOT_RECORDED_SCHEMA_GAP |
| stock Official actual torch.linalg.solve call count | NOT_RECORDED_STOCK_SOURCE; expected logical layer count only |
| per-arm peak GPU memory | NOT_RECORDED_SCHEMA_GAP; cell-level peak only |
| per-request first-hit sweep/layer | NOT_RECORDED_SCHEMA_GAP; only global all-request hit and per-step strict counts |
| post-hit drift | NOT_APPLICABLE_NO_GLOBAL_FIRST_HIT |
| exact C/P matrix telemetry | NOT_RECORDED_SCHEMA_GAP |
| raw generation text/tokens | INTENTIONALLY_NOT_PUBLISHED |

누락 필드는 시간이나 aggregate 이름으로 역추정하지 않았고 imputation/reconstruction/rerun은 0이다.

## 14. Figures와 재현성

- `primary-rs-ps-ns.png`: PRE_EDIT/O/QCL/NQFIX/ORBFH/JAC의 primary RS/PS/canonical NS.
- `paired-target-new-nll-deltas.png`: 동일 prompt의 ours−O target-new NLL.
- `hit-action-to-go-trajectories.png`: 20 visits의 residual-after와 strict-request count.
- `layer-update-energy-share.png`: L4–L8 terminal-net squared-Frobenius share.
- `compute-tradeoff.png`: endpoint wall ratio와 recorded forward count.

모든 그림은 repository Python CLI를 같은 입력으로 두 번 실행해 PNG bytes/SHA가 동일함을 검증했다. Codex image/visualization/manual edit는 사용하지 않았다. 명령·환경·입출력 SHA는 `plot-reproduction.json`에 있다.

## 15. Raw identity와 hard gates

| kind | absolute path | bytes | mode | sha256 |
| --- | --- | --- | --- | --- |
| authoritative_document | /mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-03-ordered-response-barrier-ode-edit-proposal.md | 55900 | 0664 | 03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be |
| authoritative_document | /mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-03-ordered-response-barrier-ode-edit-gh-fast-main-prompt.md | 13143 | 0664 | 8548d016fda343f8b8917bcbe43647b58ad99ab6f796ece2fe4ef661faa4c431 |
| authoritative_document | /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-03-ordered-response-barrier-ode-edit-fast-main-table.md | 18113 | 0664 | e134ac708c556482b912d7103e12a12347a78958943e7fa3d3da0a473908aa40 |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/core-performance-summary.csv | 6779 | 0664 | 6169a4a5827dfcd50675a0cb4c755454e086792738b8243b58af10b0c798df41 |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/ordered-response-barrier-ode-round0-b100-exhaustive-factual-ko.md | 48228 | 0664 | 54ef00e3944413a9908dd0ccabd33a487831c878b5a43bfbd55955ba03fa0a82 |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/analysis-manifest.json | 15839 | 0664 | 78b6217e952f03d446089d773bac365ace29ede806653ed72436d60ebd279a55 |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/rooted-analysis-receipt.json | 3772 | 0664 | d0fdeb5a1c4fd4e055265a1833576ff81ae3bce5093207bad92e531e4a059342 |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/baseline-inclusive-performance.csv | 9646 | 0664 | 7f5ed478faba2a12f444921b6d1324fa00646876418af1724c68d18cd8dd37b5 |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/ordered-response-barrier-ode-round0-b100-baseline-inclusive-factual-ko.md | 57221 | 0664 | 0067446cc19308991aa6fcb3ed5f2e9b6c2af2bb08af6d11477b0be4a6b4a84e |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/analysis-manifest.json | 21131 | 0664 | 5ebfc9df241e6ab22b68a7e0bf1f1992994b758111588b29c5106bf910f8c7af |
| immutable_prior_report_reference | experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/rooted-analysis-receipt.json | 1658 | 0664 | 9fde6a3527b6e578c4de6939eb64bf50b1e65e178568fea897e4c9606c4b720d |
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-0/result.json | 13299777 | 0600 | 68bb5ea3f0b455e3a3e892760de2b0ee40e12332e42a46894d02940eeee05553 |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-0/round-00-result.json | 13295072 | 0600 | be14adf153986c7a86a1451a83e5c2b20879352bb3eae899be243ff5a79928d9 |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-0/terminal-receipt.json | 2086 | 0600 | 1cfb5052a796c5bb546519cf5d62affda662bbe44e5087cc2fce68b913fefb74 |
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-1/result.json | 13296531 | 0600 | 2d988ec2f53f437f8897261c3ff21220425d0caff160a2985f5bc3c64a85256c |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-1/round-00-result.json | 13291816 | 0600 | fe1a5300dd6c864e8f49563934315529b914dab2b6806e085d901113c7192324 |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-1/terminal-receipt.json | 2093 | 0600 | bb0925cd51877a29620ad53b79c6b5bd6998c7ba8067eb8fad4120bc8ab86412 |
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-2/result.json | 13291646 | 0600 | 7c80c75b778ea6d58d32219058b2ab01a6dd611175122ddafe12be374291a8c9 |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-2/round-00-result.json | 13286949 | 0600 | 8574d548b31a5b0734b853abcb4aa4a2986fa7de09c3768d06fba90817fe56f8 |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-2/terminal-receipt.json | 2088 | 0600 | d4195863d31870093bbf67e8adfcd69d73aff346fcdce072f326fdd484430bd3 |
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-3/result.json | 13282763 | 0600 | 941f97dc3ddb1082927abdc941ee045cbc9e701ebc5ef1432c7b5d570d2fddd2 |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-3/round-00-result.json | 13278055 | 0600 | d1212141b68e6bb8e540a30d80cd58c1f182f8ad4f7188691d2520eeadc268ea |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-canonical-ns-rerun-v1/round0-tech-r2/task-3/terminal-receipt.json | 2095 | 0600 | 4af589d067cd604204e003b05e8e4cd7e1f501706934599d35105ba5c6706f35 |

| gate | value |
| --- | --- |
| analysis_only_gpu_action_count | 0 |
| analysis_only_model_action_count | 0 |
| analysis_only_slurm_submit_count | 0 |
| artifact_inventory_row_count | 50 |
| canonical_cell_count | 4 |
| canonical_request_count | 400 |
| compute_row_count | 20 |
| derived_orbhit_primary_denominator_influence_count | 0 |
| derived_orbhit_row_count | 4 |
| dynamic_z_recompute_count | 0 |
| entry_already_hit_count | 0 |
| evaluation_entry_binding_failure_count | 0 |
| evaluation_prompt_row_count | 62400 |
| evaluation_request_row_count | 2400 |
| evaluation_summary_recompute_failure_count | 0 |
| forbidden_controller_evaluator_access_count | 0 |
| full_fp32_scope | MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH |
| global_first_hit_count | 0 |
| history_append_count | 10 |
| horizon_semantic_miss_dynamic_endpoint_count | 16 |
| immutable_prior_report_reference_count | 8 |
| imputation_count | 0 |
| inner_cache_mutation_count | 0 |
| inner_history_append_count | 0 |
| layer_update_row_count | 100 |
| lineage_row_count | 12 |
| mechanism_arm_aggregate_row_count | 16 |
| mechanism_arm_row_count | 20 |
| mechanism_request_step_row_count | 32000 |
| mechanism_step_row_count | 320 |
| method_contrast_summary_row_count | 96 |
| method_state_content_mismatch_count | 0 |
| nll_summary_row_count | 144 |
| nonfinite_count | 0 |
| official_memit_ephemeral_fp64_exception_cell_count | 2 |
| official_wrapper_direct_fidelity_failure_count | 0 |
| paired_official_summary_row_count | 128 |
| performance_summary_row_count | 24 |
| plot_byte_reproduction_failure_count | 0 |
| plot_count | 5 |
| primary_endpoint_count | 2000 |
| prior_rerun_parity_metric_count | 9 |
| prior_rerun_parity_outside_pinned_tolerance_count | 0 |
| prior_rerun_parity_raw_nonzero_delta_count | 40 |
| raw_input_sha_before_after_unchanged | True |
| remaining_nine_submit_count | 0 |
| rerun_count_for_analysis | 0 |
| retry_count | 0 |
| runtime_preamble_failure_count | 0 |
| runtime_preamble_row_count | 4 |
| same_pre_edit_within_model | True |
| same_request_order_cross_cell | True |
| scientific_failure_count | 0 |
| scientific_promotion | False |
| source_head_tree_pass | True |
| squeue_checked | True |
| stream_order_root_pass | True |
| task_owned_active_job_count | 0 |
| technical_failure_count | 0 |
| unqualified_all_algorithm_full_fp32_cell_count | 2 |

전체 local state/log의 SHA/bytes/mode inventory는 `artifact-inventory.csv`; canonical raw→derived table→PNG→manifest/receipt 결속은 `analysis-manifest.json`과 `rooted-analysis-receipt.json`에 있다.
