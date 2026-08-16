# P1R33 P1R24 Full-Residual Remaining-Horizon Atomic — terminal 분석

## 결론

**`SCIENTIFIC_FAIL`: remaining-horizon pivot은 P1R24 strength를 재현하지 못했다.** Llama는 Neutral/Soft 모두 4/10 Eff, 7/20 Gen으로 크게 약해졌고, Qwen은 Neutral 9/10·16/20, Soft 8/10·12/20이었다. 마지막 단계가 remaining=1이어도 큰 writer lag가 자동으로 닫히지 않았다. 따라서 결과는 terminal evidence이며 `scientific_promotion=false`다.

## 범위와 무결성 (FACT)

- source `4078eeab6da2abe0aa43c0bfae3ebf8586eb9939`; scientific base `ce8c6c36348752f1407f7d713d30e6b5c727379b`; order `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`.
- 2 model × RS × Neutral/Soft = 4 arm, accepted transition 32/32, K8/tau1/materialization8 전부 PASS.
- 최대 coordinate/equality residual `2.220e-16`; 모든 step에서 remaining division=1, physical h=1, second-h/debt/lag-readdition/barrier attenuation=0.
- action-freeze 뒤 official evaluator 1회/arm; W0 pointer restore 4/4 PASS; history/retry/backtracking/imputation=0.
- k0~k8 표의 E/G/Loc는 k8 terminal만 기록했다. k0~k7 heldout는 계약상 `NOT_RECORDED`; 별도 evaluator/F/B/materialization을 추가하지 않았다.

## Terminal endpoint

| model | arm | Eff | Gen | Loc | Eff new-NLL | Gen new-NLL | z8 full6 NLL | W8 full6 NLL | cumulative P | capacity | edit-core(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 4/10 | 7/20 | 83/100 | 4.3887 | 5.4773 | 0.05398 | 4.2104 | 0.00017936 | 0.12784 | 90.41 |
| llama3-8b-inst | SOFT | 4/10 | 7/20 | 83/100 | 4.2914 | 5.4266 | 0.05120 | 4.1483 | 0.000180257 | 0.131698 | 98.82 |
| qwen2.5-7b-inst | NEUTRAL | 9/10 | 16/20 | 80/100 | 1.1100 | 3.7730 | 0.04241 | 0.8844 | 0.0238026 | 2.71707 | 102.59 |
| qwen2.5-7b-inst | SOFT | 8/10 | 12/20 | 80/100 | 2.5710 | 4.7129 | 0.01563 | 2.0928 | 0.0120879 | 1.20704 | 109.30 |

## k0~k8 단계별 내부 성능 변화

다음 표는 추가 heldout 없이 scientific path가 이미 계산한 full-six z/controller NLL, W-only NLL 및 전이 영수증만 사용한다. `actual`은 다음 refreshed W field(마지막은 terminal W objective)에서 측정됐다.

### llama3-8b-inst / NEUTRAL

| k | rem | z full6 NLL | W full6 NLL | z−W | residual mean | required mean | rho/pred | actual | realization | cum.P | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 8.3655 | 8.3655 | 0.0000 | 0.7929 | 0.0991 | 0.6038/0.6038 | 0.1960 | 0.325 | 0.000003 | 0.000916 |
| 1 | 7 | 5.2216 | 8.1695 | -2.9479 | 1.3094 | 0.1871 | 0.7079/0.7079 | 0.5075 | 0.717 | 0.000013 | 0.001738 |
| 2 | 6 | 2.8655 | 7.6620 | -4.7964 | 1.6403 | 0.2734 | 0.5923/0.5923 | 0.5727 | 0.967 | 0.000031 | 0.001721 |
| 3 | 5 | 1.2138 | 7.0892 | -5.8754 | 1.8691 | 0.3738 | 0.4730/0.4730 | 0.5282 | 1.117 | 0.000051 | 0.001339 |
| 4 | 4 | 0.5458 | 6.5610 | -6.0152 | 1.9738 | 0.4934 | 0.2179/0.2179 | 0.2388 | 1.096 | 0.000062 | 0.000251 |
| 5 | 3 | 0.3771 | 6.3223 | -5.9451 | 2.0399 | 0.6800 | 0.2822/0.2822 | 0.3383 | 1.199 | 0.000077 | 0.000464 |
| 6 | 2 | 0.1895 | 5.9839 | -5.7944 | 2.0420 | 1.0210 | 0.2304/0.2304 | 0.2635 | 1.144 | 0.000091 | 0.000300 |
| 7 | 1 | 0.1336 | 5.7204 | -5.5868 | 2.0209 | 2.0209 | 1.1597/1.1597 | 1.5100 | 1.302 | 0.000179 | 0.007214 |
| 8 | — | 0.0540 | 4.2104 | -4.1564 | — | — | —/— | — | — | 0.000179 | — |

- r=1 직전(k7) full residual mean `2.0209`, required mean `2.0209`; 마지막 actual `1.5100`, realization `1.302`.
- stepwise Eff/Gen/Loc: `NOT_RECORDED_BY_CONTRACT`; terminal E/G/L은 위 endpoint 표에만 있다.

### llama3-8b-inst / SOFT

| k | rem | z full6 NLL | W full6 NLL | z−W | residual mean | required mean | rho/pred | actual | realization | cum.P | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 8.3655 | 8.3655 | 0.0000 | 0.7929 | 0.0991 | 0.6038/0.6038 | 0.2007 | 0.332 | 0.000003 | 0.000916 |
| 1 | 7 | 5.2302 | 8.1648 | -2.9346 | 1.3090 | 0.1870 | 0.7079/0.7079 | 0.5102 | 0.721 | 0.000013 | 0.001745 |
| 2 | 6 | 2.8674 | 7.6546 | -4.7873 | 1.6407 | 0.2735 | 0.5935/0.5935 | 0.5763 | 0.971 | 0.000030 | 0.001722 |
| 3 | 5 | 1.2114 | 7.0784 | -5.8670 | 1.8723 | 0.3745 | 0.4719/0.4719 | 0.5199 | 1.102 | 0.000050 | 0.001336 |
| 4 | 4 | 0.5441 | 6.5584 | -6.0143 | 1.9772 | 0.4943 | 0.2190/0.2190 | 0.2397 | 1.094 | 0.000060 | 0.000255 |
| 5 | 3 | 0.3734 | 6.3188 | -5.9453 | 2.0412 | 0.6804 | 0.2824/0.2824 | 0.3331 | 1.180 | 0.000075 | 0.000465 |
| 6 | 2 | 0.1906 | 5.9857 | -5.7951 | 2.0417 | 1.0209 | 0.2307/0.2307 | 0.2726 | 1.181 | 0.000089 | 0.000303 |
| 7 | 1 | 0.1448 | 5.7131 | -5.5683 | 2.0222 | 2.0222 | 1.2081/1.2081 | 1.5648 | 1.295 | 0.000180 | 0.007874 |
| 8 | — | 0.0512 | 4.1483 | -4.0971 | — | — | —/— | — | — | 0.000180 | — |

- r=1 직전(k7) full residual mean `2.0222`, required mean `2.0222`; 마지막 actual `1.5648`, realization `1.295`.
- stepwise Eff/Gen/Loc: `NOT_RECORDED_BY_CONTRACT`; terminal E/G/L은 위 endpoint 표에만 있다.

### qwen2.5-7b-inst / NEUTRAL

| k | rem | z full6 NLL | W full6 NLL | z−W | residual mean | required mean | rho/pred | actual | realization | cum.P | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 9.3092 | 9.3092 | 0.0000 | 8.6648 | 1.0831 | 1.3464/1.3464 | 0.6316 | 0.469 | 0.000157 | 0.008700 |
| 1 | 7 | 5.9201 | 8.6776 | -2.7575 | 13.2988 | 1.8998 | 0.8497/0.8497 | 0.6662 | 0.784 | 0.000544 | 0.008416 |
| 2 | 6 | 2.9810 | 8.0114 | -5.0304 | 16.1882 | 2.6980 | 1.0401/1.0401 | 1.0954 | 1.053 | 0.001741 | 0.024913 |
| 3 | 5 | 1.2530 | 6.9160 | -5.6630 | 17.7674 | 3.5535 | 0.8197/0.8197 | 0.9469 | 1.155 | 0.003273 | 0.018141 |
| 4 | 4 | 0.4922 | 5.9690 | -5.4768 | 18.5054 | 4.6263 | 0.8408/0.8408 | 0.9952 | 1.184 | 0.005369 | 0.020021 |
| 5 | 3 | 0.5135 | 4.9739 | -4.4604 | 18.6098 | 6.2033 | 1.7332/1.7332 | 2.0376 | 1.176 | 0.010896 | 0.071205 |
| 6 | 2 | 0.1282 | 2.9363 | -2.8081 | 17.5556 | 8.7778 | 0.8875/0.8875 | 0.8882 | 1.001 | 0.014617 | 0.020890 |
| 7 | 1 | 0.5567 | 2.0480 | -1.4913 | 16.8421 | 16.8421 | 1.4476/1.4476 | 1.1636 | 0.804 | 0.023803 | 0.082796 |
| 8 | — | 0.0424 | 0.8844 | -0.8420 | — | — | —/— | — | — | 0.023803 | — |

- r=1 직전(k7) full residual mean `16.8421`, required mean `16.8421`; 마지막 actual `1.1636`, realization `0.804`.
- stepwise Eff/Gen/Loc: `NOT_RECORDED_BY_CONTRACT`; terminal E/G/L은 위 endpoint 표에만 있다.

### qwen2.5-7b-inst / SOFT

| k | rem | z full6 NLL | W full6 NLL | z−W | residual mean | required mean | rho/pred | actual | realization | cum.P | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 9.3092 | 9.3092 | 0.0000 | 8.6648 | 1.0831 | 1.3464/1.3464 | 0.6213 | 0.461 | 0.000135 | 0.006543 |
| 1 | 7 | 5.9170 | 8.6880 | -2.7710 | 13.3038 | 1.9005 | 0.8520/0.8520 | 0.6828 | 0.801 | 0.000464 | 0.006247 |
| 2 | 6 | 2.9830 | 8.0051 | -5.0222 | 16.1656 | 2.6943 | 1.0460/1.0460 | 1.0909 | 1.043 | 0.001489 | 0.018611 |
| 3 | 5 | 1.2587 | 6.9142 | -5.6555 | 17.6833 | 3.5367 | 0.8215/0.8215 | 0.9637 | 1.173 | 0.002798 | 0.013364 |
| 4 | 4 | 0.5172 | 5.9505 | -5.4333 | 18.3314 | 4.5829 | 0.8595/0.8595 | 1.0207 | 1.188 | 0.004620 | 0.015044 |
| 5 | 3 | 0.7792 | 4.9298 | -4.1506 | 18.0478 | 6.0159 | 1.5992/1.5992 | 1.8625 | 1.165 | 0.008902 | 0.042977 |
| 6 | 2 | 0.0881 | 3.0673 | -2.9792 | 16.6286 | 8.3143 | 0.3694/0.3694 | 0.3864 | 1.046 | 0.010046 | 0.002496 |
| 7 | 1 | 0.0613 | 2.6810 | -2.6196 | 16.4446 | 16.4446 | 0.5745/0.5745 | 0.5882 | 1.024 | 0.012088 | 0.006043 |
| 8 | — | 0.0156 | 2.0928 | -2.0771 | — | — | —/— | — | — | 0.012088 | — |

- r=1 직전(k7) full residual mean `16.4446`, required mean `16.4446`; 마지막 actual `0.5882`, realization `1.024`.
- stepwise Eff/Gen/Loc: `NOT_RECORDED_BY_CONTRACT`; terminal E/G/L은 위 endpoint 표에만 있다.

## P1R24 동일 샘플 대비

P1R24 comparator는 동일 order/objective-plan/capture-plan/evaluator-case 및 동일 W0 metric payload를 재검증했다. 해시 외피 차이는 action-freeze provenance 차이이고 W0 metrics는 exact-equal이다.

| model | arm | P1R24 E/G/L | P1R33 E/G/L | Eff NLL 24→33 | z8 NLL 24→33 | terminal P 24→33 |
|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 10/20/82 | 4/7/83 | 0.0218→4.3887 | 0.03969→0.05398 | 0.00342759→0.00017936 |
| llama3-8b-inst | SOFT | 10/20/81 | 4/7/83 | 0.0265→4.2914 | 0.03991→0.05120 | 0.00353982→0.000180257 |
| qwen2.5-7b-inst | NEUTRAL | 10/19/80 | 9/16/80 | 0.5381→1.1100 | 0.04837→0.04241 | 0.253839→0.0238026 |
| qwen2.5-7b-inst | SOFT | 10/20/80 | 8/12/80 | 0.0214→2.5710 | 0.01490→0.01563 | 0.126967→0.0120879 |

## Neutral–Soft 및 추적 해석

- FACT: k0에서 Neutral/Soft required bytes와 rho가 각 모델 내 exact-equal이고, Soft는 같은 predicted strength에서 즉시 P proxy를 낮췄다.
- FACT: Llama terminal은 Soft가 NLL을 소폭 개선했지만 count/P/capacity 보존 이득은 없었다(4/7/83 동일, P와 capacity 소폭 증가).
- FACT: Qwen Soft는 capacity와 cumulative P를 줄였지만 Neutral보다 Eff/Gen가 1/4 낮고 NLL도 악화됐다. 즉 P proxy 개선이 semantic endpoint 보존으로 이어지지 않았다.
- INFERENCE: `full residual/(K-k)`는 k0에서 P1R24보다 총 demand를 줄이고, 누적 under-realization이 후반 residual에 재등장해도 남은 한 단계가 BF16 writer의 비선형/물리적 lag를 충분히 닫지 못했다.
- SCIENTIFIC_FAIL: Llama 양 arm과 Qwen 양 arm 모두 P1R24 strength region을 유지하지 못했다. 큰 terminal residual은 숨기지 않고 writer-tracking failure로 분류한다.
- P1R32는 immediate full-residual 계수 혼합 때문에 구조적으로 유효하지 않은 negative reference일 뿐이며, P1R33과의 causal merge로 해석하지 않는다.

## Compute

- arm당 180 model-forward / 125 backward / 8 materialization. processed tokens는 Llama 27,207, Qwen 24,687이다.
- edit-core: Llama 90.41/98.82s(N/S), Qwen 102.59/109.30s. terminal evaluator 약 0.99–1.07s/arm.
- scheduler elapsed/MaxRSS: Llama job19404 272s / 7,114,016 KiB; Qwen job19403 307s / 10,489,608 KiB. model-load-only wall은 `NOT_RECORDED`.

## 경계

- FACT: 기술 무결성 PASS, 과학 결과 FAIL. typed technical failure 0, scientific endpoint denominator 4/4.
- NOT_RECORDED: k0~k7 heldout E/G/Loc, 별도 stepwise official evaluator, fresh-sample/Sequential/Historical/B100.
- reused outcome-selected B10의 mechanistic/descriptive 결과이며 `scientific_promotion=false`. 동일 run tuning/rescue/pivot는 수행하지 않았다.
