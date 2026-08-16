# P1R43 Rho-Free Semantic-First Strength Recovery B10×10 사실 보고서

- 정책: `SH_FACTUAL_ONLY_REPORTING`
- `scientific_promotion=false`
- 시도/endpoint/typed failure: `40 / 40 / 0`
- 각 endpoint: B10, K8, 독립 W0 시작/복원
- source HEAD: `11508b6da11d606521b703037034e1814b70d8a8`
- exact P1R42 parent: `ca68a4f459fd7303a4d5abbde2e1bf7aee0d805c`
- stream root: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6`
- stream order: `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`

## Scheduler 및 제출 상태

| Model | Arm | Job | Index | State | Exit | Elapsed | MaxRSS KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 19844 | 0 | COMPLETED | 0:0 | 00:28:30 | 7470956 |
| llama3-8b-inst | Soft | 19845 | 1 | COMPLETED | 0:0 | 00:26:35 | 7419696 |
| qwen2.5-7b-inst | Neutral | 19846 | 2 | COMPLETED | 0:0 | 00:30:59 | 10783264 |
| qwen2.5-7b-inst | Soft | 19843 | 3 | COMPLETED | 0:0 | 00:32:38 | 10853092 |

- array parent: `19843`, topology: `0-3%4`, node: `devbox`
- intent SHA-256: `826053d8b08915d316d6535341d0827e756ea734a8b3bb4afd9758a86ff66a90`
- submission receipt SHA-256: `a59182760b2939511a03a00e08f9fcf49be823bbcaac393f9149a7fbd299a820`
- held inspection SHA-256: `14e4455f95147ad1a1e80020f5dd70859cbd4ae330a907358ed83d639fb59a97`

## 기계적 무결성

| Check | PASS/total |
|---|---:|
| K8/tau1 endpoint | 40/40 |
| action freeze | 40/40 |
| W0 pointer restore | 40/40 |
| W0 byte restore | 40/40 |
| controller/cross-case state reset | 40/40 |
| retry=0 | 40/40 |
| history append=0 | 40/40 |
| heldout Gen controller access=0 | 40/40 |

## Model × Arm terminal 집계

| Model | Arm | Endpoints | Eff | Gen | Loc | Eff NLL | Eff margin | Gen NLL | Gen margin | z8 full6 NLL | W8 full6 NLL | W−z gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 10/10 | 100/100 | 181/200 | 868/1000 | 0.289230 | 9.639364 | 2.406026 | 5.300134 | 0.236418 | 0.315244 | 0.078826 |
| llama3-8b-inst | Soft | 10/10 | 99/100 | 181/200 | 869/1000 | 0.247503 | 9.789372 | 2.390440 | 5.334373 | 0.174896 | 0.269520 | 0.094624 |
| qwen2.5-7b-inst | Neutral | 10/10 | 96/100 | 154/200 | 845/1000 | 1.021298 | 9.342409 | 4.062368 | 3.435845 | 0.789759 | 0.894635 | 0.104875 |
| qwen2.5-7b-inst | Soft | 10/10 | 96/100 | 152/200 | 845/1000 | 0.945882 | 9.800118 | 4.169287 | 3.372588 | 0.753546 | 0.821042 | 0.067496 |

## Terminal 네 패널

| Model | Arm | Eff z-inject | Gen z-inject | Eff W | Gen W |
|---|---|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 100/100 | 182/200 | 100/100 | 181/200 |
| llama3-8b-inst | Soft | 100/100 | 184/200 | 99/100 | 181/200 |
| qwen2.5-7b-inst | Neutral | 96/100 | 161/200 | 96/100 | 154/200 |
| qwen2.5-7b-inst | Soft | 96/100 | 157/200 | 96/100 | 152/200 |

| Model | Arm | Eff z NLL mean/med/p90/worst | Eff W NLL mean/med/p90/worst | Eff W−z mean | Eff z/W margin mean | Gen z NLL mean/med/p90/worst | Gen W NLL mean/med/p90/worst | Gen W−z mean | Gen z/W margin mean |
|---|---|---|---|---:|---|---|---|---:|---|
| llama3-8b-inst | Neutral | 0.216326/0.027405/0.314520/7.750000 | 0.289230/0.063721/0.513672/8.875000 | 0.072904 | 10.475705/9.639364 | 2.148835/1.050781/5.637500/15.687500 | 2.406026/1.300781/5.909375/15.500000 | 0.257191 | 5.871847/5.300134 |
| llama3-8b-inst | Soft | 0.159384/0.024658/0.255762/5.343750 | 0.247503/0.055298/0.530859/6.218750 | 0.088119 | 10.510616/9.789372 | 2.163182/1.062500/5.596875/15.937500 | 2.390440/1.378906/5.846875/16.375000 | 0.227258 | 5.836767/5.334373 |
| qwen2.5-7b-inst | Neutral | 0.883366/0.043419/3.134375/12.375000 | 1.021298/0.080566/3.510938/12.437500 | 0.137932 | 10.384253/9.342409 | 3.821950/2.507812/9.643750/17.500000 | 4.062368/2.851562/9.381250/17.500000 | 0.240419 | 3.924242/3.435845 |
| qwen2.5-7b-inst | Soft | 0.829520/0.041748/2.432813/12.625000 | 0.945882/0.077148/2.814063/12.562500 | 0.116362 | 10.789582/9.800118 | 3.887184/2.718750/9.475000/17.375000 | 4.169287/3.007812/10.025000/17.375000 | 0.282104 | 3.902646/3.372588 |

네 패널의 new/old NLL 및 margin per-request/prompt 분포는 `p1r43-per-panel.json`에 기록되어 있다.

## Target/corrector 및 full-strength routing

| Model | Arm | Primary accepts | Rescue eligible | Rescue accepts | Current holds | Corrector F | α_req sum | α_apply sum | Fallbacks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 785 | 15 | 7 | 8 | 65 | 131.377482 | 131.377482 | 0 |
| llama3-8b-inst | Soft | 788 | 12 | 6 | 6 | 55 | 131.362564 | 131.362564 | 0 |
| qwen2.5-7b-inst | Neutral | 739 | 61 | 24 | 37 | 195 | 134.038309 | 134.038309 | 0 |
| qwen2.5-7b-inst | Soft | 749 | 51 | 18 | 33 | 170 | 132.200995 | 132.200995 | 0 |

## Writer/P/capacity/compute

| Model | Arm | Predicted sum | Actual sum | Realization | Negative steps | P terminal mean | Capacity mean | Energy mean | Functional-P mean | F | B | Tokens | Mat | Edit-core s | Eval s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 131.377482 | 100.732527 | 0.766741 | 0 | 0.008612 | 0.148756 | 0.226339 | 0.027326 | 2265 | 1250 | 385585 | 80 | 1104.925 | 363.510 |
| llama3-8b-inst | Soft | 131.362564 | 101.189769 | 0.770309 | 1 | 0.008348 | 0.092765 | 0.140454 | 0.025820 | 2255 | 1250 | 383504 | 80 | 979.935 | 376.942 |
| qwen2.5-7b-inst | Neutral | 134.038309 | 95.598442 | 0.713217 | 0 | 0.266502 | 1.234746 | 0.685289 | 0.003999 | 2395 | 1250 | 378988 | 80 | 1167.939 | 422.996 |
| qwen2.5-7b-inst | Soft | 132.200995 | 96.334366 | 0.728696 | 0 | 0.189922 | 0.288446 | 0.201308 | 0.003989 | 2370 | 1250 | 374149 | 80 | 1270.000 | 415.150 |

## Exact case-paired 비교

모든 P1R42/P1R39/Official 비교는 동일 model+batch key와 package-bound stream/order/evaluator identity로 join했다. Official AlphaEdit에는 arm이 없으며 model+batch 공통 참조로만 사용했다.

| Model | Arm | Comparator | Matched cases | Mean-case ΔEff correct | Mean-case ΔGen correct | Mean-case ΔLoc correct | Mean-case ΔEff NLL | Mean-case ΔEff margin | Mean-case Δz8 NLL | Mean-case ΔW8 NLL | Mean-case Δgap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | OfficialAlphaEdit | 10/10 | 0.000000 | -0.400000 | 0.900000 | 0.288061 | -5.112280 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| llama3-8b-inst | Neutral | P1R39 | 10/10 | 0.000000 | 0.200000 | -0.700000 | 0.141205 | -2.094018 | 0.144135 | 0.171033 | 0.026897 |
| llama3-8b-inst | Neutral | P1R42 | 10/10 | 0.000000 | 0.200000 | -0.700000 | 0.137995 | -1.464713 | 0.148078 | 0.158614 | 0.010536 |
| llama3-8b-inst | Soft | OfficialAlphaEdit | 10/10 | -0.100000 | -0.400000 | 1.000000 | 0.246334 | -4.962272 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| llama3-8b-inst | Soft | P1R39 | 10/10 | -0.100000 | 0.000000 | -0.600000 | 0.121882 | -2.155944 | 0.092715 | 0.140660 | 0.047945 |
| llama3-8b-inst | Soft | P1R42 | 10/10 | -0.100000 | 0.100000 | -0.400000 | 0.133480 | -1.442074 | 0.086949 | 0.135489 | 0.048540 |
| qwen2.5-7b-inst | Neutral | OfficialAlphaEdit | 10/10 | -0.400000 | -3.800000 | 1.700000 | 0.988652 | -4.940023 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| qwen2.5-7b-inst | Neutral | P1R39 | 10/10 | 0.000000 | -0.100000 | 0.100000 | 0.192093 | -1.115360 | 0.109248 | 0.160477 | 0.051229 |
| qwen2.5-7b-inst | Neutral | P1R42 | 10/10 | 0.000000 | -0.700000 | 0.100000 | 0.268592 | -1.079769 | 0.206548 | 0.262351 | 0.055803 |
| qwen2.5-7b-inst | Soft | OfficialAlphaEdit | 10/10 | -0.400000 | -4.000000 | 1.700000 | 0.913236 | -4.482314 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| qwen2.5-7b-inst | Soft | P1R39 | 10/10 | 0.000000 | -0.400000 | 0.000000 | 0.175452 | -1.013300 | 0.136733 | 0.152739 | 0.016007 |
| qwen2.5-7b-inst | Soft | P1R42 | 10/10 | 0.100000 | -0.300000 | -0.100000 | 0.134539 | -0.786369 | 0.097812 | 0.121697 | 0.023885 |

Official AlphaEdit의 latent-z, W8 full-six NLL, P/capacity/energy 및 realization 필드는 `NOT_RECORDED`이다.

## Machine-readable 산출물

- `p1r43-per-case.json`: 40 rows
- `p1r43-per-step.json`: 320 rows
- `p1r43-per-request.json`: 3200 rows
- `p1r43-per-panel.json`: 160 rows
- `p1r43-panel-aggregates.json`: 16 rows
- `p1r43-case-paired-comparisons.json`: 120 rows
- `p1r43-cell-aggregates.json`: 4 rows
- `analysis-manifest.json`, `analysis-receipt.json`: 파일 및 rooted digest

## 기록 경계

- Peak GPU memory: `NOT_RECORDED` in per-case scientific receipts.
- Slurm MaxRSS는 scheduler 표에 task-level 값으로 기록했다.
- Stepwise heldout Eff/Gen/Loc: `NOT_RECORDED`; terminal-only evaluator가 사용되었다.
- raw prompt/target/generation/tensor/weight/model/cache/runtime log: 포함하지 않았다.
- 과학적 해석, 인과 주장, 우월성 판단, 추천: 포함하지 않았다.
