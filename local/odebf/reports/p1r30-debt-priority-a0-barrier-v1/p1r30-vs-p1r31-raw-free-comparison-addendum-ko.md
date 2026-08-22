# P1R30 ↔ P1R31 raw-free 비교 부록

- 작성 시각: 2026-08-13 Asia/Seoul
- P1R30 terminal report: `p1r30-a2-full-matrix-diagnostic-terminal-report-ko.md`, SHA `426c78287cad904fbcf13502d5bdad150a5badc5f008123820dec0573c865de7`
- P1R31 package: `/mnt/raid5/janghj/ODE-edit/local/source-handoff/P1R31_P1R24_INDEPENDENT_B10X10_DETAILED_RESULTS_SH1_V1`
- P1R31 package manifest SHA/root: `ff804cd9f22a4d493c13324bf74fa908b7787689a9d78a13744140cd30b7e86e` / `8df18d6de8d1e065b12d607ee3e17e5257545b109801356026172a9d90628318`
- 분석 중 model/evaluator/GPU/Slurm/source/result mutation: 모두 `0`

## 결론

`P1R31_PACKAGE_VERIFICATION=PASS`  
`P1R30_P1R31_EXACT_SAMPLE_MATCH=false`  
`COMPARISON_CLASS=DESCRIPTIVE_AGGREGATE_ONLY_NOT_CASE_MATCHED`  
`PREDECLARED_NEUTRAL_RECOVERY_GATE=FAIL_UNCHANGED`  
`P1R30_FULL_MATRIX_DIAGNOSTIC=PARTIAL_6_ENDPOINTS_2_TYPED_K4_PREFIXES`  
`SCIENTIFIC_PROMOTION=false`

P1R31은 P1R24를 10개의 독립 Atomic B10에서 반복한 8-cell audit이고, P1R30은 원래 R13/P1R19의 단일 sealed B10에서 실행한 8-cell 진단이다. evaluator와 aggregator source는 같지만 request order와 evaluation-case identity가 다르다. 따라서 이 부록은 model×allocation×arm 수준의 **비매칭 집계 비교**이며, case-paired delta나 인과효과를 주장하지 않는다.

두 실험을 함께 보면 다음 세 가지가 남는다.

1. P1R31의 same-stream Official 비교는 특히 Qwen에서 P1R24 target z 자체가 약하고, BF16 writer가 다시 추가 결손을 만든다는 것을 보여준다.
2. P1R30은 debt를 총 write magnitude가 아니라 request priority에만 사용했으므로 이 두 결손을 증폭해 갚지 않는다. 실제로 Qwen RS의 debt와 z→W gap은 크고, Qwen BG는 양 arm 모두 k4 뒤 `WRITER_NO_POSITIVE_DIRECTION`에 도달했다.
3. Soft/Barrier는 Llama BG에서만 일관된 local mechanistic signal을 보였고, Llama RS는 혼합, Qwen은 fallback 또는 typed boundary로 비식별이다. 보편적 recovery나 preservation 개선으로 승격할 수 없다.

## 1. 패키지 무결성 및 비교 identity

P1R31 package의 payload 10개와 package manifest/handoff receipt를 합친 12개 파일을 독립 재검증했다.

- directory/files mode `0700/0600`, 모두 regular non-symlink
- manifest의 payload SHA/size/mode 10/10 PASS; JSON 9개 parse PASS
- canonical root 재계산 PASS:
  - package `8df18d6de8d1e065b12d607ee3e17e5257545b109801356026172a9d90628318`
  - handoff `763af0e499802236cebb8264055c348674eb935ad978ea9eaa6358a47174238e`
  - analysis manifest `81dc3d007ae79a4d6ab4e82a4d151844bfff74cb893a3449cb187c076223e419`
  - analysis receipt `dbbe0801db3396a279cf1291330599f337486c750ed6ece4a7480b4a79199b7d`
- 80 attempts, 79 endpoints, typed failure 1; endpoint imputation 0
- raw prompts/targets/generations/tensors/weights/cache/log payload 0

| identity | P1R30 | P1R31 | 판정 |
|---|---|---|---|
| 범위 | original atomic B10 1개/cell | independent atomic B10 10개/cell | cardinality 다름 |
| request order | `984fe6ec…015b` | `abe62c07…cd5c` | 불일치 |
| stream/seal | `3d38b76d…f628` | `74d68965…89e6` | 불일치 |
| evaluation-case identity | `04012bcc…4354` | 서로 다른 10개 identity | match 0/10 |
| evaluator source | `25c3f490…e145` | `25c3f490…e145` | exact match |
| aggregator source | `64f009b2…45c0` | `64f009b2…45c0` | exact match |

P1R31의 10개 B10 request-order hash 및 evaluation-case-group hash 가운데 P1R30과 일치하는 것은 각각 0/10이다. raw-free package에는 per-request raw ID가 없으므로 개별 request overlap은 `NOT_RECORDED`; 정확한 matched-case claim은 0건이다. P1R31은 B10을 history로 이어 붙인 sequential 실행이 아니라, 매번 W0에서 시작한 10개의 독립 Atomic 편집이다.

Source-owner transport receipt의 SHA/root는 GH가 전달했으나 receipt bytes가 server2 package 또는 sibling path에 없어 `NOT_INDEPENDENTLY_REVERIFIED`이다. 이는 locally present한 package payload/manifest/root PASS와 분리한다.

## 2. Endpoint와 continuous metric

P1R31 count는 10개 B10의 attempted denominator다. Llama BG-Soft는 9/10 endpoint이며 continuous 값은 성공한 9개 조건부 평균이다. P1R30은 한 B10의 10/20/100 denominator다. 표본이 다르므로 수치 차이를 paired delta로 읽으면 안 된다.

| Cell | P1R31 endpoint/attempt; E/G/L | P1R31 Eff new NLL; Gen new NLL | P1R30 endpoint; E/G/L | P1R30 Eff new NLL; Gen new NLL |
|---|---|---:|---|---:|
| Llama RS N | 10/10; 99/100, 178/200, 873/1000 | .143048; 2.366033 | 1/1; 10/10, 20/20, 81/100 | .023298; .985429 |
| Llama RS S | 10/10; 99/100, 177/200, 873/1000 | .179537; 2.377230 | 1/1; 10/10, 20/20, 81/100 | .026079; 1.082293 |
| Llama BG N | 10/10; 98/100, 175/200, 871/1000 | .418802; 2.509655 | 1/1; 10/10, 20/20, 81/100 | .743001; 1.448070 |
| Llama BG S | 9/10; 89/100, 164/200, 787/1000 | .263082; 2.407429 | 1/1; 10/10, 20/20, 81/100 | .607318; 1.224545 |
| Qwen RS N | 10/10; 98/100, 155/200, 843/1000 | .585113; 3.820958 | 1/1; 10/10, 19/20, 79/100 | 1.399106; 2.900348 |
| Qwen RS S | 10/10; 100/100, 163/200, 844/1000 | .492203; 3.599686 | 1/1; 10/10, 19/20, 79/100 | 1.399106; 2.900348 |
| Qwen BG N | 10/10; 94/100, 165/200, 843/1000 | 1.293311; 3.836427 | 0/1; k4 typed prefix | endpoint N/R |
| Qwen BG S | 10/10; 94/100, 164/200, 842/1000 | 1.273214; 3.950246 | 0/1; k4 typed prefix | endpoint N/R |

P1R31 package에는 Gen target-old/margin과 Loc continuous NLL/margin이 직렬화되지 않았다. P1R30에는 해당 값이 있으나 비대칭이므로 그 축의 cross-run continuous comparison은 `NOT_RECORDED`다.

## 3. Target/z와 W realization

### 3.1 P1R31 안의 exact same-stream Official 비교

P1R31은 79개 유효 endpoint마다 Official AlphaEdit와 evaluator/case/order/W0 score identity를 확인했다. 이 비교에서 Official은 direct-z-driven terminal endpoint이며 standalone latent-z trajectory는 기록되지 않았다.

- Llama Official: E/G/L `100/185/859`, Eff new NLL `.001169`
- Qwen Official: E/G/L `100/192/828`, Eff new NLL `.032646`
- Llama P1R24 z는 Official 대비 Gen이 주로 1–6 낮고 Eff NLL은 +.074–.292 높다.
- Qwen P1R24 z는 Official 대비 Eff 1–5, Gen 21–30 낮고 Eff NLL은 +.302–.829 높다.
- z에서 W로 추가되는 Eff NLL은 Llama +.030–.126, Qwen +.106–.510이다.

따라서 P1R31에서 Qwen under-edit은 `target-side + writer-side` 혼합으로 직접 분해된다. Llama RS는 target 결손이 더 작고, Llama BG는 target과 writer가 모두 관여한다.

### 3.2 P1R30과 P1R31의 full-six z→W 관측

아래 gap은 같은 row 안에서만 의미가 있다. P1R30과 P1R31 사이의 gap 크기 차이는 표본이 달라 paired effect가 아니다.

| Cell | P1R31 full-six W / z / gap | P1R30 full-six W / z / gap |
|---|---:|---:|
| Llama RS N | .145918 / .083357 / .062561 | .049419 / .039342 / .010077 |
| Llama RS S | .187498 / .084685 / .102812 | .050746 / .039926 / .010819 |
| Llama BG N | .439991 / .328960 / .111031 | .771678 / .078068 / .693610 |
| Llama BG S | .289079 / .265387 / .023692 | .551921 / .087832 / .464089 |
| Qwen RS N | .560281 / .287876 / .272405 | 1.281940 / .056804 / 1.225136 |
| Qwen RS S | .426689 / .294569 / .132120 | 1.281940 / .056804 / 1.225136 |
| Qwen BG N | 1.113824 / .840274 / .273550 | endpoint N/R |
| Qwen BG S | 1.158396 / .764909 / .393487 | endpoint N/R |

P1R30 Qwen RS는 terminal z target-new NLL은 낮지만 W gap이 `1.2251`로 크고 realization은 `.2312`다. 즉 P1R30의 이 B10에서는 writer under-realization이 명확하다. 반면 P1R30에는 official evaluator의 z E/G count가 없어 Official direct-z 대비 target sufficiency를 P1R31처럼 완전히 분해할 수 없다.

## 4. Predicted/actual progress와 debt

P1R30은 k1–k8 actual을 모두 기록해 cumulative predicted/actual/debt를 계산했다. P1R31은 k1–k8 `rho=predicted`를 저장했지만 actual은 성공 endpoint마다 k8 한 번만 non-null이다. 따라서 P1R31의 `actual_sum` 필드명은 cumulative actual로 해석하면 안 되며, 두 실험의 actual/realization 수치를 직접 빼지 않는다. P1R31/P1R24에는 semantic debt state가 없다.

| Cell | P1R31 mean Σrho / recorded k8 actual / mean step realization | P1R30 cumulative Σpred / Σactual / realization / debt mean(max) |
|---|---:|---:|
| Llama RS N | 16.7936 / .1017 / .3475 | 13.3830 / 8.3161 / .6214 / 5.0290 (11.9914) |
| Llama RS S | 16.9892 / .0630 / .2845 | 13.4135 / 8.3147 / .6199 / 5.0668 (11.9076) |
| Llama BG N | 23.0183 / .3025 / .3338 | 15.5190 / 7.5938 / .4893 / 7.9515 (25.6628) |
| Llama BG S | 21.9098 / .3023 / .3588 | 15.5718 / 7.8136 / .5018 / 7.7720 (26.8620) |
| Qwen RS N | 32.5694 / .4058 / .0334 | 34.7234 / 8.0273 / .2312 / 26.8230 (65.5739) |
| Qwen RS S | 32.9669 / .2066 / .2022 | 34.7234 / 8.0273 / .2312 / 26.8230 (65.5739) |
| Qwen BG N | 44.2458 / .2522 / .0633 | k1–k4: 27.5727 / 8.4435 / .3062 / 19.3278 (68.9647) |
| Qwen BG S | 46.3781 / −.1231 / −.0040 | N과 같은 k1–k4 prefix 뒤 typed boundary |

P1R30 debt는 audit/priority state이며 total update magnitude influence가 0이었다. 따라서 큰 Qwen debt는 미회수 semantic deficit을 정직하게 보여주지만, 그 자체로 write strength를 증폭해 회수하지 않는다. P1R31에서도 Qwen은 낮거나 음의 realization과 15개 negative-actual step을 보여 writer-side 불안정이 broad stream에서 반복된다.

## 5. Routing, Structural-P, capacity

표기는 `P-AUC / endpoint P / cumulative BF16 capacity / router raw capacity / energy`다. P1R30 Qwen BG는 k4 prefix 값이며 endpoint와 비교하지 않는다.

| Cell | P1R31 mean | P1R30 |
|---|---:|---:|
| Llama RS N | .020261 / .006220 / 3.7651 / .1759 / .2791 | .011712 / .003427 / 2.5036 / .1201 / .2049 |
| Llama RS S | .021021 / .006847 / 4.0170 / .2640 / .4264 | .011774 / .003542 / 2.6110 / .1435 / .2449 |
| Llama BG N | .034027 / .011251 / 6.8465 / .3202 / .4199 | .015768 / .005989 / 4.3777 / .6194 / 1.0172 |
| Llama BG S | .023982 / .008104 / 4.6645 / .2813 / .4152 | .015146 / .005381 / 3.8384 / .3985 / .6571 |
| Qwen RS N | 1.220162 / .374913 / 39.9999 / 35.2169 / 7.5004 | 1.028464 / .529561 / 58.9364 / 50.2470 / 29.1718 |
| Qwen RS S | .757265 / .191940 / 17.7210 / 2.2851 / 1.3053 | 1.028464 / .529561 / 58.9364 / 50.2470 / 29.1718 |
| Qwen BG N | 2.247097 / .997737 / 108.7054 / 517.3210 / 58.7182 | k4 .153471 / .069329 / 7.8198 / 1.5947 / 1.6395 |
| Qwen BG S | 1.205466 / .465210 / 42.8490 / 64.9872 / 18.9556 | N과 같은 k4 prefix |

P1R30 routing fact:

- Neutral은 exact A0 `c=c0`, debt magnitude influence 0.
- Llama RS Soft: selected 5/8, fallback 3/8. same-state local P는 낮췄지만 별도 Neutral trajectory 대비 endpoint P `+3.35%`, BF16 capacity `+4.29%`, Eff/Gen NLL도 소폭 악화했다.
- Llama BG Soft: selected 6/8, fallback 2/8. 별도 Neutral 대비 endpoint P `−10.15%`, BF16 capacity `−12.32%`, Eff/Gen NLL과 realization이 함께 개선됐다.
- Qwen RS Soft: fallback 8/8로 Neutral과 동일. Qwen BG Soft: prefix fallback 4/4 뒤 Neutral과 같은 typed boundary.

P1R31 routing fact:

- Llama RS Soft는 P/capacity와 endpoint count를 개선하지 않았다.
- Llama BG Soft는 Structural-P/capacity를 낮췄지만 1/10 typed failure가 있고 attempted-denominator E/G/L이 낮았다.
- Qwen RS Soft는 P/capacity와 E/G/L을 개선했지만 functional-P는 높아졌다.
- Qwen BG Soft는 P/capacity를 크게 낮췄으나 E는 동률, G/L은 −1/−1이고 mean realization은 음수였다.

공통 결론은 Structural-P proxy 감소가 endpoint preservation 또는 functional-P 개선으로 보편적으로 변환되지 않는다는 것이다. Llama BG는 두 표본에서 proxy 감소 방향이 반복되지만 endpoint robustness는 sample/state dependent다. Qwen RS의 P1R31 positive signal은 P1R30에서 8/8 fallback이므로 P1R30 barrier 효과의 증거가 아니다.

## 6. 실패와 compute

### 실패 분모

- P1R30: 8/8 cells attempted, 6 endpoints. Qwen BG Neutral/Soft는 각 k4 뒤 `WRITER_NO_POSITIVE_DIRECTION`; imputation 0.
- P1R31: 80 attempts, 79 endpoints. Llama BG-Soft case06은 k5 뒤 typed `ODEBFContractError`; imputation 0, 이후 case 계속 수행.

### Edit-core

| Cell | P1R31 mean edit-core, F/B, tokens | P1R30 edit-core, F/B, tokens |
|---|---:|---:|
| Llama RS N | 84.171s, 180/125, 28,701 | 73.479s, 180/125, 27,207 |
| Llama RS S | 80.180s, 180/125, 28,701 | 67.173s, 180/125, 27,207 |
| Llama BG N | 86.792s, 180/125, 28,701 | 72.017s, 180/125, 27,207 |
| Llama BG S | 81.077s, 180/125, 28,759 | 106.138s, 180/125, 27,207 |
| Qwen RS N | 102.534s, 180/125, 26,286 mean | 98.928s, 180/125, 24,687 |
| Qwen RS S | 99.320s, 180/125, 26,286 mean | 76.562s, 180/125, 24,687 |
| Qwen BG N/S | 105.892 / 104.389s | terminal compute N/R |

유효 trajectory는 두 실험 모두 8 materializations이며 P1R30 Soft router added model F/B=0이다. P1R31은 post-freeze k0..k8 stepwise evaluation에 평균 416–474초를 별도로 썼고, P1R30은 stepwise external evaluator 0/terminal 1이다. server1/server2, 표본 길이, evaluator workload가 달라 wall 차이를 speedup 또는 router causal cost로 해석하지 않는다.

## 7. Baseline 경계

Official Native comparator를 섞지 않는다.

1. P1R30 Official Native는 original sealed B10/order/W0/evaluator에 대한 AlphaEdit endpoint다. Llama `10/20/79`, Qwen `10/20/80`; P1R30의 직접 causal A0 control은 아니다.
2. P1R31 Official Native는 P1R31의 10-batch independent stream과 exact same-stream인 AlphaEdit aggregate다. Llama `100/185/859`, Qwen `100/192/828`; 79개 P1R24 endpoint와 identity PASS다.
3. 두 Official 결과도 sample이 다르므로 서로 하나의 공통 paired denominator로 합치지 않는다.

## 8. FACT / INFERENCE / NOT_RECORDED

### FACT

- P1R31 package payload와 rooted receipts는 독립 무결성 PASS다.
- P1R30과 P1R31은 evaluator/aggregator는 같지만 sample/order/evaluation-case identity가 다르다.
- P1R31 same-stream Official 비교에서 under-edit은 target-side와 writer-side가 함께 만든다. Qwen에서 두 결손이 가장 크다.
- P1R30 Qwen RS는 큰 z→W gap/debt/낮은 realization을 보였고, Qwen BG는 endpoint 전 writer direction 경계에 도달했다.
- P1R30 Soft barrier는 Llama BG에서만 positive mechanistic signal, Llama RS는 mixed, Qwen은 non-identifiable이다.

### INFERENCE

- P1R31의 broad-stream evidence는 P1R30의 predeclared Qwen Neutral recovery failure를 뒤집지 않는다.
- debt priority만으로 총 write magnitude를 바꾸지 않는 P1R30 설계는 target weakness나 writer under-realization을 직접 회수하지 못했다.
- Llama BG의 proxy 감소는 재현 방향성이 있으나 typed failure와 endpoint 불일치 때문에 universal preservation evidence가 아니다.
- Atomic 수준에서 다음 병목은 target sufficiency와 BF16 writer realization을 분리해 해결해야 하며, Structural-P 감소만으로 성공을 판정하면 안 된다.

### NOT_RECORDED / NON-COMPARABLE

- P1R30↔P1R31 individual request overlap 및 case-paired delta
- P1R31 cumulative k1–k8 actual progress와 semantic debt
- P1R31 Gen margin 및 Loc continuous NLL/margin
- P1R30 official-evaluator z E/G count
- P1R30 online/terminal functional-P 및 fKL
- Qwen BG P1R30 terminal/action-freeze/W0 endpoint
- cross-server runtime parity와 authoritative GPU utilization
- Official AlphaEdit latent-z k0..k8 trajectory
- source-owner transport receipt bytes의 server2 독립 재검증

## 최종 판정

`PACKAGE_VERIFICATION=PASS`  
`TARGET_WRITER_DECOMPOSITION=P1R31_SUPPORTS_TARGET_PLUS_WRITER_DEFICIT`  
`P1R30_DEBT_PRIORITY_STRENGTH_RECOVERY=FAIL`  
`P1R30_BARRIER_EFFECT=DESCRIPTIVE_MECHANISTIC_MIXED`  
`LLAMA_BG_SIGNAL=LOCAL_POSITIVE_NOT_ROBUSTLY_PROMOTABLE`  
`LLAMA_RS_SIGNAL=MIXED_NEGATIVE_AT_PACKAGE_LEVEL`  
`QWEN_SIGNAL=FALLBACK_OR_TYPED_BOUNDARY_NOT_IDENTIFIABLE`  
`PREDECLARED_NEUTRAL_RECOVERY_GATE=FAIL_UNCHANGED`  
`SCIENTIFIC_PROMOTION=false`  
`FOLLOWUP_MODEL_GPU_SLURM_SOURCE_ACTION=0`
