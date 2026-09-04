# Ordered Response-Barrier ODE-Edit — Server1 round0 B100 상세 사실 보고서

## 0. 판정과 실행 경계

- 상태: `ANALYSIS_ONLY_TERMINAL_PASS`; canonical 실행은 Slurm array `35694`, round0 B100 네 cell이다.
- scheduler child: `35694_0→35737`, `35694_1→35752`, `35694_2→35889`, `35694_3→35694`; 모두 `COMPLETED/0:0`이다.
- canonical 과학 분모: `4 cells × 5 primary arms × 100 requests = 2,000 request-arm endpoints`; 입력 request는 cell별 100, 전체 실행 관측 400이다.
- `ORBHit`은 ORBFH 경로의 derived materialized prefix라 primary arm이나 2,000 분모에 중복 포함하지 않았다.
- 분석 중 새 model/GPU/Slurm/editing run/retry/imputation은 모두 0이며 remaining-nine은 HOLD다.
- 모든 비교는 동일 round0 request/order 안의 기술적·서술적 비교다. 인과, 보편성, 자동 promotion을 주장하지 않는다. `scientific_promotion=false`.

## 1. 지표 정의와 정확한 분모

- NLL은 teacher-forced target token의 평균 negative log likelihood다. `target-new`는 새 사실 정답, `target-true`는 원래 사실 정답이며 NLL 수치 자체는 낮을수록 해당 target에 더 높은 확률을 준다.
- `RS`는 100 rewrite pair에서 `NLL(new) < NLL(true)`인 strict count/rate다. tie는 failure다.
- `PS`는 200 rephrase prompt pair에서 같은 strict 비교를 한 count/rate다. `strict PS`는 request별 두 rephrase가 모두 성공한 100-request count/rate다.
- `NS`는 endpoint locality prompt에서 PRE_EDIT token prediction이 보존된 token count/전체 token count다. prompt 1,000개의 all-token preserved count도 CSV에 있다.
- rewrite/rephrase accuracy는 target token을 모두 맞힌 prompt 수로, pairwise RS/PS와 다른 secondary 지표다.
- PRE_EDIT은 endpoint 이전 W0 관측이다. NS는 전후 비교가 없어 `N/A`; locality target-true accuracy만 secondary로 보존했다.
- `ENTRY_ALREADY_HIT`은 계약상 정상 W0 no-op endpoint로 분모에 들어가지만 이번 네 cell에서는 0건이었다.

## 2. 핵심 endpoint 표 — PRE_EDIT, O, QCL, NQFIX, ORBFH, JAC

| model | writer | stage | status | RS | PS | strict PS | NS | RW new acc | RW true acc | RP new acc | RP true acc | LOC true acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | PRE_EDIT | ENTRY_REFERENCE | 13/100 (13.00%) | 28/200 (14.00%) | 8/100 (8.00%) | N/A (entry reference) | 0/100 (0.00%) | 19/100 (19.00%) | 1/200 (0.50%) | 36/200 (18.00%) | 208/1000 (20.80%) |
| llama3-8b-inst | MEMIT | O | TERMINAL_VALID | 98/100 (98.00%) | 164/200 (82.00%) | 77/100 (77.00%) | 951/1010 (94.16%) | 95/100 (95.00%) | 1/100 (1.00%) | 97/200 (48.50%) | 7/200 (3.50%) | 205/1000 (20.50%) |
| llama3-8b-inst | MEMIT | QCL | HORIZON_SEMANTIC_MISS | 96/100 (96.00%) | 156/200 (78.00%) | 69/100 (69.00%) | 952/1010 (94.26%) | 89/100 (89.00%) | 1/100 (1.00%) | 88/200 (44.00%) | 8/200 (4.00%) | 206/1000 (20.60%) |
| llama3-8b-inst | MEMIT | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 183/200 (91.50%) | 87/100 (87.00%) | 941/1010 (93.17%) | 100/100 (100.00%) | 0/100 (0.00%) | 114/200 (57.00%) | 6/200 (3.00%) | 202/1000 (20.20%) |
| llama3-8b-inst | MEMIT | ORBFH | HORIZON_SEMANTIC_MISS | 98/100 (98.00%) | 162/200 (81.00%) | 75/100 (75.00%) | 940/1010 (93.07%) | 96/100 (96.00%) | 1/100 (1.00%) | 97/200 (48.50%) | 8/200 (4.00%) | 205/1000 (20.50%) |
| llama3-8b-inst | MEMIT | JAC | HORIZON_SEMANTIC_MISS | 98/100 (98.00%) | 163/200 (81.50%) | 76/100 (76.00%) | 936/1010 (92.67%) | 96/100 (96.00%) | 1/100 (1.00%) | 96/200 (48.00%) | 8/200 (4.00%) | 204/1000 (20.40%) |
| llama3-8b-inst | AlphaEdit | PRE_EDIT | ENTRY_REFERENCE | 13/100 (13.00%) | 28/200 (14.00%) | 8/100 (8.00%) | N/A (entry reference) | 0/100 (0.00%) | 19/100 (19.00%) | 1/200 (0.50%) | 36/200 (18.00%) | 208/1000 (20.80%) |
| llama3-8b-inst | AlphaEdit | O | TERMINAL_VALID | 100/100 (100.00%) | 184/200 (92.00%) | 89/100 (89.00%) | 877/1010 (86.83%) | 100/100 (100.00%) | 0/100 (0.00%) | 120/200 (60.00%) | 7/200 (3.50%) | 210/1000 (21.00%) |
| llama3-8b-inst | AlphaEdit | QCL | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 169/200 (84.50%) | 79/100 (79.00%) | 891/1010 (88.22%) | 99/100 (99.00%) | 0/100 (0.00%) | 108/200 (54.00%) | 7/200 (3.50%) | 206/1000 (20.60%) |
| llama3-8b-inst | AlphaEdit | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 93/100 (93.00%) | 868/1010 (85.94%) | 100/100 (100.00%) | 0/100 (0.00%) | 126/200 (63.00%) | 6/200 (3.00%) | 209/1000 (20.90%) |
| llama3-8b-inst | AlphaEdit | ORBFH | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 168/200 (84.00%) | 79/100 (79.00%) | 890/1010 (88.12%) | 99/100 (99.00%) | 0/100 (0.00%) | 107/200 (53.50%) | 7/200 (3.50%) | 206/1000 (20.60%) |
| llama3-8b-inst | AlphaEdit | JAC | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 169/200 (84.50%) | 79/100 (79.00%) | 888/1010 (87.92%) | 100/100 (100.00%) | 0/100 (0.00%) | 107/200 (53.50%) | 7/200 (3.50%) | 208/1000 (20.80%) |
| qwen2.5-7b-inst | MEMIT | PRE_EDIT | ENTRY_REFERENCE | 11/100 (11.00%) | 39/200 (19.50%) | 10/100 (10.00%) | N/A (entry reference) | 0/100 (0.00%) | 13/100 (13.00%) | 1/200 (0.50%) | 32/200 (16.00%) | 153/1000 (15.30%) |
| qwen2.5-7b-inst | MEMIT | O | TERMINAL_VALID | 100/100 (100.00%) | 192/200 (96.00%) | 95/100 (95.00%) | 898/1010 (88.91%) | 100/100 (100.00%) | 0/100 (0.00%) | 130/200 (65.00%) | 2/200 (1.00%) | 146/1000 (14.60%) |
| qwen2.5-7b-inst | MEMIT | QCL | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 189/200 (94.50%) | 93/100 (93.00%) | 905/1010 (89.60%) | 99/100 (99.00%) | 0/100 (0.00%) | 130/200 (65.00%) | 2/200 (1.00%) | 150/1000 (15.00%) |
| qwen2.5-7b-inst | MEMIT | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 189/200 (94.50%) | 93/100 (93.00%) | 857/1010 (84.85%) | 100/100 (100.00%) | 0/100 (0.00%) | 132/200 (66.00%) | 2/200 (1.00%) | 127/1000 (12.70%) |
| qwen2.5-7b-inst | MEMIT | ORBFH | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 889/1010 (88.02%) | 100/100 (100.00%) | 0/100 (0.00%) | 133/200 (66.50%) | 2/200 (1.00%) | 141/1000 (14.10%) |
| qwen2.5-7b-inst | MEMIT | JAC | HORIZON_SEMANTIC_MISS | 99/100 (99.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 875/1010 (86.63%) | 98/100 (98.00%) | 0/100 (0.00%) | 131/200 (65.50%) | 2/200 (1.00%) | 141/1000 (14.10%) |
| qwen2.5-7b-inst | AlphaEdit | PRE_EDIT | ENTRY_REFERENCE | 11/100 (11.00%) | 39/200 (19.50%) | 10/100 (10.00%) | N/A (entry reference) | 0/100 (0.00%) | 13/100 (13.00%) | 1/200 (0.50%) | 32/200 (16.00%) | 153/1000 (15.30%) |
| qwen2.5-7b-inst | AlphaEdit | O | TERMINAL_VALID | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 866/1010 (85.74%) | 99/100 (99.00%) | 0/100 (0.00%) | 132/200 (66.00%) | 2/200 (1.00%) | 136/1000 (13.60%) |
| qwen2.5-7b-inst | AlphaEdit | QCL | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 889/1010 (88.02%) | 99/100 (99.00%) | 0/100 (0.00%) | 130/200 (65.00%) | 2/200 (1.00%) | 150/1000 (15.00%) |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 838/1010 (82.97%) | 99/100 (99.00%) | 0/100 (0.00%) | 133/200 (66.50%) | 2/200 (1.00%) | 124/1000 (12.40%) |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 881/1010 (87.23%) | 99/100 (99.00%) | 0/100 (0.00%) | 134/200 (67.00%) | 2/200 (1.00%) | 142/1000 (14.20%) |
| qwen2.5-7b-inst | AlphaEdit | JAC | HORIZON_SEMANTIC_MISS | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 877/1010 (86.83%) | 100/100 (100.00%) | 0/100 (0.00%) | 132/200 (66.00%) | 2/200 (1.00%) | 144/1000 (14.40%) |

각 cell에서 primary rate의 단순 최댓값(선택·promotion 규칙 아님):

| model | writer | metric | max arm(s) | rate |
| --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | RS | NQFIX | 100.00% |
| llama3-8b-inst | MEMIT | PS | NQFIX | 91.50% |
| llama3-8b-inst | MEMIT | NS | QCL | 94.26% |
| llama3-8b-inst | AlphaEdit | RS | O,QCL,NQFIX,ORBFH,JAC | 100.00% |
| llama3-8b-inst | AlphaEdit | PS | NQFIX | 95.00% |
| llama3-8b-inst | AlphaEdit | NS | QCL | 88.22% |
| qwen2.5-7b-inst | MEMIT | RS | O,QCL,NQFIX,ORBFH | 100.00% |
| qwen2.5-7b-inst | MEMIT | PS | O | 96.00% |
| qwen2.5-7b-inst | MEMIT | NS | QCL | 89.60% |
| qwen2.5-7b-inst | AlphaEdit | RS | O,QCL,NQFIX,ORBFH,JAC | 100.00% |
| qwen2.5-7b-inst | AlphaEdit | PS | O,QCL,ORBFH | 95.50% |
| qwen2.5-7b-inst | AlphaEdit | NS | QCL | 88.02% |

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

`prompt-nll.csv.gz`는 38,400개 PRE_EDIT/endpoint prompt 행을 모두 보존하고, `request-endpoint-metrics.csv.gz`는 2,400개 request-stage 행에서 rewrite, 두 rephrase, 열 locality 관측을 결속한다.

## 5. 동일 request의 Official O 대비 paired delta

delta는 `ours - O`다. NLL은 음수가 더 낮은 수치, target-new margin과 NS는 양수가 더 높은 수치다. equal은 tolerance를 만들지 않은 exact arithmetic equality다.

| model | writer | ours | metric | n | Δ mean | Δ median | Δ p90 | Δ max | better/equal/worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | QCL | rewrite_target_new_nll | 100 | 0.3165 | 0.0087 | 0.9267 | 5.8327 | 0/0/100 |
| llama3-8b-inst | MEMIT | QCL | rephrase_target_new_nll | 200 | 0.5206 | 0.2602 | 1.4142 | 3.6276 | 14/0/186 |
| llama3-8b-inst | MEMIT | QCL | rewrite_margin_true_minus_new | 100 | -2.0809 | -1.6345 | -0.8211 | -0.2008 | 0/0/100 |
| llama3-8b-inst | MEMIT | QCL | rephrase_margin_true_minus_new | 200 | -1.2239 | -1.0013 | -0.2247 | 0.5504 | 5/0/195 |
| llama3-8b-inst | MEMIT | QCL | locality_prediction_preservation_rate | 100 | 0.0010 | 0.0000 | 0.0000 | 0.1000 | 6/89/5 |
| llama3-8b-inst | MEMIT | NQFIX | rewrite_target_new_nll | 100 | -0.3782 | -0.0017 | -0.0002 | 0.0367 | 98/0/2 |
| llama3-8b-inst | MEMIT | NQFIX | rephrase_target_new_nll | 200 | -0.9830 | -0.2148 | 0.0117 | 0.7521 | 175/0/25 |
| llama3-8b-inst | MEMIT | NQFIX | rewrite_margin_true_minus_new | 100 | 2.6478 | 1.5162 | 5.2441 | 30.4521 | 99/0/1 |
| llama3-8b-inst | MEMIT | NQFIX | rephrase_margin_true_minus_new | 200 | 2.1446 | 1.5972 | 4.7653 | 12.6742 | 189/0/11 |
| llama3-8b-inst | MEMIT | NQFIX | locality_prediction_preservation_rate | 100 | -0.0100 | 0.0000 | 0.0000 | 0.2000 | 7/79/14 |
| llama3-8b-inst | MEMIT | ORBFH | rewrite_target_new_nll | 100 | -0.0258 | 0.0001 | 0.0010 | 0.3204 | 36/0/64 |
| llama3-8b-inst | MEMIT | ORBFH | rephrase_target_new_nll | 200 | 0.0526 | 0.0032 | 0.3987 | 1.3789 | 75/0/125 |
| llama3-8b-inst | MEMIT | ORBFH | rewrite_margin_true_minus_new | 100 | -0.0552 | -0.0534 | 0.3676 | 3.6673 | 41/0/59 |
| llama3-8b-inst | MEMIT | ORBFH | rephrase_margin_true_minus_new | 200 | -0.1473 | -0.1176 | 0.3595 | 1.2102 | 62/0/138 |
| llama3-8b-inst | MEMIT | ORBFH | locality_prediction_preservation_rate | 100 | -0.0110 | 0.0000 | 0.0000 | 0.1000 | 3/84/13 |
| llama3-8b-inst | MEMIT | JAC | rewrite_target_new_nll | 100 | -0.0230 | 0.0000 | 0.0027 | 0.4011 | 41/0/59 |
| llama3-8b-inst | MEMIT | JAC | rephrase_target_new_nll | 200 | 0.0150 | 0.0000 | 0.4219 | 1.7616 | 100/0/100 |
| llama3-8b-inst | MEMIT | JAC | rewrite_margin_true_minus_new | 100 | -0.0417 | -0.0210 | 0.4135 | 3.4213 | 48/0/52 |
| llama3-8b-inst | MEMIT | JAC | rephrase_margin_true_minus_new | 200 | -0.0574 | -0.0680 | 0.6066 | 2.5272 | 88/0/112 |
| llama3-8b-inst | MEMIT | JAC | locality_prediction_preservation_rate | 100 | -0.0150 | 0.0000 | 0.0000 | 0.1000 | 3/81/16 |
| llama3-8b-inst | AlphaEdit | QCL | rewrite_target_new_nll | 100 | 0.0316 | 0.0010 | 0.0062 | 2.5653 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | QCL | rephrase_target_new_nll | 200 | 0.5544 | 0.1215 | 1.8245 | 9.3421 | 19/0/181 |
| llama3-8b-inst | AlphaEdit | QCL | rewrite_margin_true_minus_new | 100 | -1.6818 | -1.3289 | -0.5046 | 0.0032 | 1/0/99 |
| llama3-8b-inst | AlphaEdit | QCL | rephrase_margin_true_minus_new | 200 | -1.5363 | -1.0816 | -0.2124 | 1.3558 | 13/0/187 |
| llama3-8b-inst | AlphaEdit | QCL | locality_prediction_preservation_rate | 100 | 0.0135 | 0.0000 | 0.1000 | 0.2000 | 20/72/8 |
| llama3-8b-inst | AlphaEdit | NQFIX | rewrite_target_new_nll | 100 | 0.0008 | 0.0000 | 0.0007 | 0.0285 | 30/1/69 |
| llama3-8b-inst | AlphaEdit | NQFIX | rephrase_target_new_nll | 200 | -0.1819 | -0.0015 | 0.2376 | 2.5441 | 121/0/79 |
| llama3-8b-inst | AlphaEdit | NQFIX | rewrite_margin_true_minus_new | 100 | -0.1885 | -0.0481 | 0.3938 | 0.9225 | 40/0/60 |
| llama3-8b-inst | AlphaEdit | NQFIX | rephrase_margin_true_minus_new | 200 | 0.4040 | 0.2732 | 1.6578 | 4.8747 | 128/0/72 |
| llama3-8b-inst | AlphaEdit | NQFIX | locality_prediction_preservation_rate | 100 | -0.0085 | 0.0000 | 0.0000 | 0.2000 | 8/74/18 |
| llama3-8b-inst | AlphaEdit | ORBFH | rewrite_target_new_nll | 100 | 0.0337 | 0.0009 | 0.0067 | 2.7576 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | ORBFH | rephrase_target_new_nll | 200 | 0.6129 | 0.1462 | 2.0497 | 9.9136 | 21/0/179 |
| llama3-8b-inst | AlphaEdit | ORBFH | rewrite_margin_true_minus_new | 100 | -1.7565 | -1.4285 | -0.5449 | -0.0372 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | ORBFH | rephrase_margin_true_minus_new | 200 | -1.6738 | -1.1838 | -0.2469 | 2.0703 | 11/0/189 |
| llama3-8b-inst | AlphaEdit | ORBFH | locality_prediction_preservation_rate | 100 | 0.0125 | 0.0000 | 0.1000 | 0.2000 | 19/73/8 |
| llama3-8b-inst | AlphaEdit | JAC | rewrite_target_new_nll | 100 | 0.0234 | 0.0008 | 0.0061 | 1.7898 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | JAC | rephrase_target_new_nll | 200 | 0.5615 | 0.0973 | 1.9930 | 9.7019 | 27/0/173 |
| llama3-8b-inst | AlphaEdit | JAC | rewrite_margin_true_minus_new | 100 | -1.6678 | -1.3166 | -0.5373 | -0.0282 | 0/0/100 |
| llama3-8b-inst | AlphaEdit | JAC | rephrase_margin_true_minus_new | 200 | -1.5502 | -1.1117 | -0.0429 | 2.3496 | 20/0/180 |
| llama3-8b-inst | AlphaEdit | JAC | locality_prediction_preservation_rate | 100 | 0.0110 | 0.0000 | 0.1000 | 0.2000 | 22/66/12 |
| qwen2.5-7b-inst | MEMIT | QCL | rewrite_target_new_nll | 100 | 0.0347 | 0.0005 | 0.0140 | 2.8970 | 40/0/60 |
| qwen2.5-7b-inst | MEMIT | QCL | rephrase_target_new_nll | 200 | 0.0469 | 0.0052 | 0.6295 | 4.0418 | 88/0/112 |
| qwen2.5-7b-inst | MEMIT | QCL | rewrite_margin_true_minus_new | 100 | -0.2185 | -0.1480 | 1.0086 | 4.0985 | 45/0/55 |
| qwen2.5-7b-inst | MEMIT | QCL | rephrase_margin_true_minus_new | 200 | -0.3607 | -0.2894 | 0.9518 | 3.4942 | 71/0/129 |
| qwen2.5-7b-inst | MEMIT | QCL | locality_prediction_preservation_rate | 100 | 0.0070 | 0.0000 | 0.1000 | 0.4000 | 15/71/14 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rewrite_target_new_nll | 100 | -0.0059 | 0.0000 | 0.0119 | 0.1735 | 50/0/50 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rephrase_target_new_nll | 200 | 0.0071 | -0.0008 | 0.3717 | 2.1207 | 103/0/97 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rewrite_margin_true_minus_new | 100 | -0.1389 | -0.1397 | 0.8079 | 2.5106 | 37/0/63 |
| qwen2.5-7b-inst | MEMIT | NQFIX | rephrase_margin_true_minus_new | 200 | 0.0787 | 0.0603 | 1.0226 | 4.9216 | 105/0/95 |
| qwen2.5-7b-inst | MEMIT | NQFIX | locality_prediction_preservation_rate | 100 | -0.0405 | 0.0000 | 0.0000 | 0.1000 | 8/61/31 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rewrite_target_new_nll | 100 | 0.0042 | -0.0001 | 0.0053 | 0.4547 | 54/0/46 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rephrase_target_new_nll | 200 | 0.0011 | 0.0001 | 0.2021 | 4.1748 | 100/0/100 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rewrite_margin_true_minus_new | 100 | -0.0263 | 0.0009 | 0.5839 | 1.4701 | 51/0/49 |
| qwen2.5-7b-inst | MEMIT | ORBFH | rephrase_margin_true_minus_new | 200 | 0.0129 | -0.0398 | 0.6678 | 3.0930 | 95/0/105 |
| qwen2.5-7b-inst | MEMIT | ORBFH | locality_prediction_preservation_rate | 100 | -0.0090 | 0.0000 | 0.1000 | 0.1000 | 12/70/18 |
| qwen2.5-7b-inst | MEMIT | JAC | rewrite_target_new_nll | 100 | 0.0389 | -0.0000 | 0.0136 | 3.0114 | 51/0/49 |
| qwen2.5-7b-inst | MEMIT | JAC | rephrase_target_new_nll | 200 | 0.0083 | 0.0004 | 0.4050 | 3.9475 | 98/0/102 |
| qwen2.5-7b-inst | MEMIT | JAC | rewrite_margin_true_minus_new | 100 | -0.0814 | -0.0292 | 1.1552 | 2.8048 | 47/0/53 |
| qwen2.5-7b-inst | MEMIT | JAC | rephrase_margin_true_minus_new | 200 | -0.1296 | -0.1393 | 0.9429 | 3.9019 | 81/0/119 |
| qwen2.5-7b-inst | MEMIT | JAC | locality_prediction_preservation_rate | 100 | -0.0230 | 0.0000 | 0.1000 | 0.4000 | 15/56/29 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rewrite_target_new_nll | 100 | -0.0003 | -0.0005 | 0.0054 | 0.4113 | 62/0/38 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rephrase_target_new_nll | 200 | -0.0662 | -0.0011 | 0.3822 | 1.6652 | 107/0/93 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rewrite_margin_true_minus_new | 100 | 0.3486 | 0.3568 | 1.4520 | 4.8924 | 67/0/33 |
| qwen2.5-7b-inst | AlphaEdit | QCL | rephrase_margin_true_minus_new | 200 | 0.0393 | -0.0580 | 1.3818 | 5.2916 | 93/0/107 |
| qwen2.5-7b-inst | AlphaEdit | QCL | locality_prediction_preservation_rate | 100 | 0.0225 | 0.0000 | 0.1000 | 0.2000 | 34/51/15 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rewrite_target_new_nll | 100 | 0.0346 | -0.0000 | 0.0112 | 2.8468 | 52/0/48 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rephrase_target_new_nll | 200 | 0.0287 | 0.0002 | 0.3016 | 5.0587 | 97/0/103 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rewrite_margin_true_minus_new | 100 | -0.1574 | 0.0455 | 0.5691 | 2.2383 | 54/0/46 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | rephrase_margin_true_minus_new | 200 | -0.0735 | -0.0484 | 0.8013 | 2.3317 | 93/0/107 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | locality_prediction_preservation_rate | 100 | -0.0275 | 0.0000 | 0.0000 | 0.2000 | 9/64/27 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rewrite_target_new_nll | 100 | -0.0041 | -0.0005 | 0.0038 | 0.1019 | 63/0/37 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rephrase_target_new_nll | 200 | -0.0442 | -0.0009 | 0.3599 | 2.0836 | 106/0/94 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rewrite_margin_true_minus_new | 100 | 0.3128 | 0.2674 | 1.3032 | 3.4084 | 65/0/35 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | rephrase_margin_true_minus_new | 200 | 0.0746 | -0.0546 | 1.0913 | 5.2884 | 93/0/107 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | locality_prediction_preservation_rate | 100 | 0.0145 | 0.0000 | 0.1000 | 0.2000 | 29/56/15 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rewrite_target_new_nll | 100 | -0.0044 | -0.0006 | 0.0077 | 0.4060 | 64/0/36 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rephrase_target_new_nll | 200 | -0.0701 | -0.0007 | 0.4533 | 2.0099 | 104/0/96 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rewrite_margin_true_minus_new | 100 | 0.3598 | 0.2843 | 1.8825 | 4.1917 | 62/0/38 |
| qwen2.5-7b-inst | AlphaEdit | JAC | rephrase_margin_true_minus_new | 200 | 0.0243 | -0.0576 | 1.4118 | 7.5473 | 92/0/108 |
| qwen2.5-7b-inst | AlphaEdit | JAC | locality_prediction_preservation_rate | 100 | 0.0105 | 0.0000 | 0.1000 | 0.4000 | 29/48/23 |

모든 target-new/target-true NLL, rewrite/rephrase margin, NS의 112개 paired summary는 `paired-official-deltas.csv`; QCL→NQFIX→ORBFH→JAC의 84개 predeclared adjacent contrast는 `method-contrast-deltas.csv`에 있다.

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
| llama3-8b-inst | MEMIT | O | 159.24 | 0.00 | 1.000 | 154 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| llama3-8b-inst | MEMIT | QCL | 647.71 | 488.47 | 4.068 | 465 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | MEMIT | NQFIX | 648.08 | 488.85 | 4.070 | 465 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | MEMIT | ORBFH | 732.28 | 573.04 | 4.599 | 485 | 20 | 20 | 20 | 20 | 1 | 67.03 |
| llama3-8b-inst | MEMIT | JAC | 680.30 | 521.06 | 4.272 | 447 | 20 | 20 | 20 | 20 | 1 | 66.76 |
| llama3-8b-inst | AlphaEdit | O | 204.21 | 0.00 | 1.000 | 179 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| llama3-8b-inst | AlphaEdit | QCL | 725.49 | 521.28 | 3.553 | 490 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | AlphaEdit | NQFIX | 725.52 | 521.31 | 3.553 | 490 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| llama3-8b-inst | AlphaEdit | ORBFH | 797.45 | 593.24 | 3.905 | 510 | 20 | 20 | 20 | 20 | 1 | 65.69 |
| llama3-8b-inst | AlphaEdit | JAC | 737.97 | 533.76 | 3.614 | 461 | 20 | 20 | 20 | 20 | 1 | 65.62 |
| qwen2.5-7b-inst | MEMIT | O | 182.77 | 0.00 | 1.000 | 154 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | MEMIT | QCL | 643.16 | 460.39 | 3.519 | 465 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | MEMIT | NQFIX | 641.95 | 459.18 | 3.512 | 465 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | MEMIT | ORBFH | 706.26 | 523.49 | 3.864 | 485 | 20 | 20 | 20 | 20 | 1 | 57.38 |
| qwen2.5-7b-inst | MEMIT | JAC | 623.26 | 440.50 | 3.410 | 425 | 20 | 20 | 20 | 20 | 1 | 57.03 |
| qwen2.5-7b-inst | AlphaEdit | O | 201.35 | 0.00 | 1.000 | 179 | 0 | 5 | 0 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | AlphaEdit | QCL | 705.47 | 504.13 | 3.504 | 490 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | AlphaEdit | NQFIX | 702.68 | 501.33 | 3.490 | 490 | 20 | 20 | 20 | 0 | 1 | 0.00 |
| qwen2.5-7b-inst | AlphaEdit | ORBFH | 750.84 | 549.49 | 3.729 | 499 | 20 | 20 | 20 | 20 | 1 | 56.79 |
| qwen2.5-7b-inst | AlphaEdit | JAC | 670.97 | 469.62 | 3.332 | 439 | 20 | 20 | 20 | 20 | 1 | 56.52 |

Official stock 내부 actual solve call은 intercept하지 않아 logical expected 5만 기록됐고, dynamic arms는 adapter-intercepted 20 solves/builds가 기록됐다. peak GPU memory는 arm별이 아니라 cell-level이다: Llama/MEMIT 42.41/47.32GB allocated/reserved, Llama/AlphaEdit 40.29/43.38GB, Qwen/MEMIT 45.75/50.14GB, Qwen/AlphaEdit 42.35/44.72GB. model load는 cell별 1회이며 wave 분리 시 reload가 필요하다.

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
| 35520 | 0 | 35521 | round0-tech-r1 | FAILED | 1:0 | c747b3ea67c4 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | RuntimeError | self.stride(-1) must be 1 to view Float as Byte (different element sizes… | True |
| 35520 | 1 | 35522 | round0-tech-r1 | FAILED | 1:0 | c747b3ea67c4 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TypeError | get_module_input_output_at_words() got an unexpected keyword argument 't… | True |
| 35520 | 2 | 35523 | round0-tech-r1 | FAILED | 1:0 | c747b3ea67c4 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | RuntimeError | self.stride(-1) must be 1 to view Float as Byte (different element sizes… | True |
| 35520 | 3 | 35520 | round0-tech-r1 | FAILED | 1:0 | c747b3ea67c4 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TypeError | get_module_input_output_at_words() got an unexpected keyword argument 't… | True |
| 35567 | 0 | 35568 | round0-tech-r2 | FAILED | 1:0 | 396496710c43 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TechnicalBoundary | custom dynamic R/n one-pass did not reproduce stock Official endpoint | True |
| 35567 | 1 | 35570 | round0-tech-r2 | FAILED | 1:0 | 396496710c43 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TechnicalBoundary | custom dynamic R/n one-pass did not reproduce stock Official endpoint | True |
| 35567 | 2 | 35571 | round0-tech-r2 | FAILED | 1:0 | 396496710c43 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TechnicalBoundary | custom dynamic R/n one-pass did not reproduce stock Official endpoint | True |
| 35567 | 3 | 35567 | round0-tech-r2 | FAILED | 1:0 | 396496710c43 | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TechnicalBoundary | custom dynamic R/n one-pass did not reproduce stock Official endpoint | True |
| 35582 | 0 | 35583 | round0-tech-r4 | FAILED | 1:0 | 346a592257be | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TechnicalBoundary | fixed-state B(R/n)=B(R)/n scaling parity failed | True |
| 35582 | 1 | 35584 | round0-tech-r4 | FAILED | 1:0 | 346a592257be | EXCLUDED_TECHNICAL_INVALID_DENOMINATOR_ZERO | TechnicalBoundary | fixed-state B(R/n)=B(R)/n scaling parity failed | True |
| 35582 | 2 | 35587 | round0-tech-r4 | CANCELLED by 1025 | 0:0 | 346a592257be | EXCLUDED_USER_CANCELLED_PARTIAL_DENOMINATOR_ZERO | - | - | NOT_RECORDED_CANCELLED_PARTIAL |
| 35582 | 3 | 35582 | round0-tech-r4 | CANCELLED by 1025 | 0:0 | 346a592257be | EXCLUDED_USER_CANCELLED_PARTIAL_DENOMINATOR_ZERO | - | - | NOT_RECORDED_CANCELLED_PARTIAL |
| 35615 | 0 | 35616 | b1-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 35615 | 1 | 35617 | b1-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 35615 | 2 | 35619 | b1-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 35615 | 3 | 35615 | b1-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | EXCLUDED_SEPARATE_B1_PILOT_PROVENANCE_ONLY | - | - | True |
| 35694 | 0 | 35737 | round0-b1-gated-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |
| 35694 | 1 | 35752 | round0-b1-gated-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |
| 35694 | 2 | 35889 | round0-b1-gated-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |
| 35694 | 3 | 35694 | round0-b1-gated-tech-r1 | COMPLETED | 0:0 | 84e6cde74db9 | INCLUDED_CANONICAL_SCIENTIFIC_DENOMINATOR | - | - | True |
| NOT_SUBMITTED | -1 | NOT_APPLICABLE | round0-tech-r3 | DRY_PLAN_ONLY | NOT_APPLICABLE | ed627c93e136 | EXCLUDED_NOT_SUBMITTED_DENOMINATOR_ZERO | - | - | NOT_APPLICABLE_NO_RUN |

`35520/35567`은 preamble 기술 실패 4/4, `35582`는 2개 기술 실패 후 2개 task 사용자 취소다. `round0-tech-r3`는 dry-plan-only이며 제출되지 않았다. `35615`는 4/4 완료한 별도 B1 pilot이라 B100 분모에서 제외했다. canonical 과학 분모는 source `84e6cde…`, job `35694`만 사용한다.

## 12. ORBHit derived endpoint

| model | writer | status | factors | RS | PS | strict PS | NS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | MEMIT | HORIZON_SEMANTIC_MISS | 20 | 98/100 (98.00%) | 162/200 (81.00%) | 75/100 (75.00%) | 940/1010 (93.07%) |
| llama3-8b-inst | AlphaEdit | HORIZON_SEMANTIC_MISS | 20 | 100/100 (100.00%) | 168/200 (84.00%) | 79/100 (79.00%) | 890/1010 (88.12%) |
| qwen2.5-7b-inst | MEMIT | HORIZON_SEMANTIC_MISS | 20 | 100/100 (100.00%) | 190/200 (95.00%) | 94/100 (94.00%) | 889/1010 (88.02%) |
| qwen2.5-7b-inst | AlphaEdit | HORIZON_SEMANTIC_MISS | 19 | 100/100 (100.00%) | 191/200 (95.50%) | 94/100 (94.00%) | 881/1010 (87.23%) |

global hit이 없었으므로 derived prefix는 horizon prefix를 materialize했다. 이 표는 observation-only이며 primary arm 분모와 aggregate에 합치지 않았다.

## 13. 기록 한계와 금지된 추정

| field | status |
| --- | --- |
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

- `primary-rs-ps-ns.png`: PRE_EDIT/O/QCL/NQFIX/ORBFH/JAC의 primary RS/PS/NS.
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
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-0/result.json | 8999580 | 0600 | fcc6d54d04696c62e4438e108b236138a6dfe122bcefec8d88f4ed172ced7607 |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-0/round-00-result.json | 8995236 | 0600 | 1cab6512fdad9f4fb3f20777fca9f74be596541db1d92dcf9968341910d4775e |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-0/terminal-receipt.json | 1723 | 0600 | c83d1f2da67ed6d9a7075c8b753ff8f1f05829e4cde866ed258ac0d29fa585cf |
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-1/result.json | 8998043 | 0600 | fa3f12037fb4af0dd53ba0882638d5ab1f80236cfe53b430c4a1c431a527dfbd |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-1/round-00-result.json | 8993689 | 0600 | 0fe47dfc75a42b2cea3411d1d4b39604967ffb84560d4697559353e349ae31fe |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-1/terminal-receipt.json | 1730 | 0600 | 4158012f3b215e59404646ad6b8c7c94697ba6dfeaea3915faa0c77e74f1094a |
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-2/result.json | 8992013 | 0600 | 6798fee4195f016ec0f7429a71bfea6fe8a98c40e349e042702ca97f89764832 |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-2/round-00-result.json | 8987678 | 0600 | df46f1dd66cdba74a9af5d1c2f291c97503d1fc7dbe721069fe306fba34a94b7 |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-2/terminal-receipt.json | 1725 | 0600 | 9a69e7bf5ada5ad65a7660d6a944e508a2835d63559ad610ae524e87d6c7ca56 |
| canonical_result | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-3/result.json | 8983591 | 0600 | 50060bdb29ac4c6135fe6c8a64580f6f33c809b4128d7ce080b41ac2e51d487c |
| canonical_round | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-3/round-00-result.json | 8979244 | 0600 | 3771f20b8635670127edf2873e629f109c60563015a6f676c90e083a4ea054dc |
| terminal_receipt | /mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1/round0-b1-gated-tech-r1/task-3/terminal-receipt.json | 1732 | 0600 | 155d9cf550df82a14024b65c67db644c3ffc35e2834a3876266a538cc0a8a622 |

| gate | value |
| --- | --- |
| analysis_only_gpu_action_count | 0 |
| analysis_only_model_action_count | 0 |
| analysis_only_slurm_submit_count | 0 |
| artifact_inventory_row_count | 93 |
| canonical_cell_count | 4 |
| canonical_request_count | 400 |
| compute_row_count | 20 |
| derived_orbhit_primary_denominator_influence_count | 0 |
| derived_orbhit_row_count | 4 |
| dynamic_z_recompute_count | 0 |
| entry_already_hit_count | 0 |
| evaluation_entry_binding_failure_count | 0 |
| evaluation_prompt_row_count | 38400 |
| evaluation_request_row_count | 2400 |
| evaluation_summary_recompute_failure_count | 0 |
| forbidden_controller_evaluator_access_count | 0 |
| full_fp32_scope | MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH |
| global_first_hit_count | 0 |
| history_append_count | 10 |
| horizon_semantic_miss_dynamic_endpoint_count | 16 |
| imputation_count | 0 |
| inner_cache_mutation_count | 0 |
| inner_history_append_count | 0 |
| layer_update_row_count | 100 |
| lineage_row_count | 21 |
| mechanism_arm_aggregate_row_count | 16 |
| mechanism_arm_row_count | 20 |
| mechanism_request_step_row_count | 32000 |
| mechanism_step_row_count | 320 |
| method_contrast_summary_row_count | 84 |
| method_state_content_mismatch_count | 0 |
| nll_summary_row_count | 120 |
| nonfinite_count | 0 |
| official_memit_ephemeral_fp64_exception_cell_count | 2 |
| official_wrapper_direct_fidelity_failure_count | 0 |
| paired_official_summary_row_count | 112 |
| performance_summary_row_count | 24 |
| plot_byte_reproduction_failure_count | 0 |
| plot_count | 5 |
| primary_endpoint_count | 2000 |
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
