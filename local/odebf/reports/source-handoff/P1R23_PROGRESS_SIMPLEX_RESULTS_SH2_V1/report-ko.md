# P1R23 Progress-Simplex B10 원자적 terminal 검토

## 1. 결론

FACT:

- `18697`의 네 paired task, 여덟 trajectory가 모두 `COMPLETED 0:0`, K=8, tau=1로 끝났다. scientific-invalid=0이며 W0 pointer/bytes restore와 persistent commit=0을 모두 통과했다.
- Simplex-Neutral은 4/4 model×allocation에서 기존 P1R23 all-cap Neutral의 endpoint metric과 최종 effective BF16 weight hash를 정확히 재현했다.
- Simplex-Soft의 32/32 field는 DOF=4, Neutral과 다른 coefficient, 실제 P/H objective influence, 같은-state predicted-progress equality, Neutral-relative global-energy envelope를 모두 만족했다.
- Soft는 32/32 field에서 자체 정규화 P/H worst-risk를 Neutral shadow보다 낮췄다. 따라서 이전 P1R23의 `NO_ROUTING_OPPORTUNITY/NOT_IDENTIFIABLE`은 해소됐고 local router verdict는 `ACTIVE_LOCAL_ROUTER_SUCCESS`다.
- endpoint Eff/Gen count는 여덟 arm 모두 10/10, 20/20이었다. Loc는 Llama 네 arm 81/100, Qwen RS 두 arm 81/100, Qwen BG Neutral 80/100→Soft 81/100이었다.

INFERENCE:

- local optimization은 성공했지만 atomic preservation의 package-level 개선은 혼합적이다. Qwen BG에서 Loc +1이 관측됐고 나머지는 count tie이며 continuous NLL/margin은 축별로 작은 양·음 변화가 섞였다.
- Soft는 단순 strength attenuation이 아니다. 각 Soft field에서 q를 정확히 유지했고, 여러 layer의 v>1을 허용하면서 global energy를 Neutral envelope 안에 유지했다. 다만 서로 다른 trajectory는 k1 이후 state drift가 있으므로 cross-arm 누적 q 자체를 동일-strength 증명으로 사용하지 않는다.
- 재사용된 outcome-selected B10의 원자적 결과이므로 universal preservation 또는 Historical H 효과로 승격할 수 없다.

## 2. 실행·무결성

- scientific checkpoint: `a343d1f6967ef37763009b92d227ade85cd93de0`
- execution head: `47941b0bdfc50f48515f86bdded97e3cc4938d65`
- numerical lock SHA/root: `e07055eb247276039f9e5349b72ac151cf1c2ed906fc4271103da077bb4b20ca` / `9ac7d598fa19d01d363459fe158f92da558839f9524795c6c361d5b32615f490`
- source manifest root: `27b19cbaeb595a28cd287ce05089f161d31322f8ef271c414dce6b5f37131a93`
- request order: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- method: `P1R23-DYNAMIC-PROGRESS-SIMPLEX-PHYSICAL-WONLY-ROUTER-V1`
- routing solver tolerances were frozen before outcome access; temperature/cap/trust tuning and retries were zero.

| task | role | state | elapsed | MaxRSS |
|---|---|---:|---:|---:|
| 18697_0 | Llama BG pair | COMPLETED 0:0 | 265 s | 6.76 GiB |
| 18697_1 | Llama RS pair | COMPLETED 0:0 | 268 s | 6.73 GiB |
| 18697_2 | Qwen BG pair | COMPLETED 0:0 | 301 s | 10.30 GiB |
| 18697_3 | Qwen RS pair | COMPLETED 0:0 | 327 s | 10.11 GiB |

## 3. Pre-edit W0

| model | Eff | Gen | Loc | Eff new/old/margin | Gen new/old/margin | Loc new/old/margin |
|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 2/10 | 3/20 | 84/100 | 8.720313/3.928027/-4.792285 | 8.651563/4.104883/-4.546680 | 8.616563/4.301951/4.314612 |
| qwen2.5-7b-inst | 0/10 | 4/20 | 80/100 | 8.978125/4.123096/-4.855029 | 9.350000/4.335059/-5.014941 | 8.884375/5.164048/3.720327 |

## 4. Terminal endpoint

NLL 표기는 `mean new / old / receipt-defined margin`이다.

| model | alloc | arm | Eff | Gen | Loc | Eff NLL | Gen NLL | Loc NLL | edit-core | nominal/native* |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | BG | NEUTRAL | 10/10 | 20/20 | 81/100 | 0.031889/10.309375/10.277486 | 0.844440/9.442187/8.597748 | 8.196406/4.322847/3.873560 | 97.430s | 2.616x |
| llama3-8b-inst | BG | SOFT | 10/10 | 20/20 | 81/100 | 0.021565/10.425000/10.403435 | 0.807475/9.287500/8.480025 | 8.211250/4.307705/3.903545 | 101.023s | 2.712x |
| llama3-8b-inst | RS | NEUTRAL | 10/10 | 20/20 | 81/100 | 0.020693/16.187500/16.166807 | 0.526387/12.743750/12.217363 | 8.343594/4.297285/4.046309 | 103.041s | 2.766x |
| llama3-8b-inst | RS | SOFT | 10/10 | 20/20 | 81/100 | 0.020660/16.137500/16.116840 | 0.537638/12.667188/12.129550 | 8.338594/4.288025/4.050569 | 102.685s | 2.757x |
| qwen2.5-7b-inst | BG | NEUTRAL | 10/10 | 20/20 | 80/100 | 0.053979/12.965625/12.911646 | 1.906824/10.566406/8.659583 | 8.832812/5.177129/3.655684 | 111.782s | 2.421x |
| qwen2.5-7b-inst | BG | SOFT | 10/10 | 20/20 | 81/100 | 0.051495/13.046875/12.995380 | 2.095862/10.134375/8.038513 | 8.814375/5.171914/3.642461 | 114.369s | 2.477x |
| qwen2.5-7b-inst | RS | NEUTRAL | 10/10 | 20/20 | 81/100 | 0.037970/18.481250/18.443280 | 1.465430/11.296875/9.831445 | 8.847188/5.169692/3.677495 | 127.135s | 2.754x |
| qwen2.5-7b-inst | RS | SOFT | 10/10 | 20/20 | 81/100 | 0.037681/18.912500/18.874819 | 1.487311/11.139062/9.651752 | 8.841875/5.179419/3.662456 | 124.647s | 2.700x |

`*` repaired Optimized Native equivalence는 Qwen Loc 79 vs Official 80 때문에 fail-closed다. 따라서 이 비율은 timing-only nominal ratio이며 formal rho가 아니다.

## 5. Soft−Neutral endpoint contrast

delta는 Soft−Neutral이다.

| model | alloc | ΔE/G/L count | ΔEff new/margin | ΔGen new/margin | ΔLoc new/margin | Δactual progress | Δterminal fP | Δcapacity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | BG | 0/0/0 | -0.010324/+0.125949 | -0.036964/-0.117723 | +0.014844/+0.029985 | +0.012768 | -0.000182 | +0.088159 |
| llama3-8b-inst | RS | 0/0/0 | -0.000033/-0.049967 | +0.011251/-0.087813 | -0.005000/+0.004260 | +0.002231 | +0.001136 | +0.066115 |
| qwen2.5-7b-inst | BG | 0/0/1 | -0.002484/+0.083734 | +0.189038/-0.621069 | -0.018437/-0.013223 | +0.050786 | -0.010464 | -6.428104 |
| qwen2.5-7b-inst | RS | 0/0/0 | -0.000290/+0.431540 | +0.021880/-0.179693 | -0.005313/-0.015039 | +0.001757 | -0.013016 | -7.440236 |

## 6. Router identifiability와 local objective

| model | alloc | DOF/opportunity | risk↓ | allocation changed | influence | risk Δ min/med/mean/max | coeff L1 min/med/max | v>1 steps/count/max | energy max | equality max | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| llama3-8b-inst | BG | 8/8 | 8/8 | 8/8 | 8/8 | -0.01732/-0.00589/-0.00699/-0.00225 | 0.1676/0.2861/0.4302 | 8/8/18/1.1505 | 0.9999999986 | 4.44e-16 | ACTIVE_LOCAL_ROUTER_SUCCESS |
| llama3-8b-inst | RS | 8/8 | 8/8 | 8/8 | 8/8 | -0.02204/-0.00901/-0.01071/-0.00020 | 0.0744/0.3786/0.5023 | 8/8/17/1.2399 | 0.9999999997 | 2.22e-16 | ACTIVE_LOCAL_ROUTER_SUCCESS |
| qwen2.5-7b-inst | BG | 8/8 | 8/8 | 8/8 | 8/8 | -0.16469/-0.13131/-0.12769/-0.07218 | 1.2652/1.6364/2.4045 | 8/8/20/1.6015 | 0.7992474973 | 4.44e-16 | ACTIVE_LOCAL_ROUTER_SUCCESS |
| qwen2.5-7b-inst | RS | 8/8 | 8/8 | 8/8 | 8/8 | -0.21673/-0.16393/-0.16793/-0.13976 | 1.5858/1.8160/1.9666 | 8/8/22/1.5388 | 0.7523374200 | 4.44e-16 | ACTIVE_LOCAL_ROUTER_SUCCESS |

FACT:

- 모든 64 field에서 active layer=5, equality rank=1, feasible allocation dimension=4였다.
- Neutral은 정의대로 pi=a/q와 v=[1,1,1,1,1]을 수치적으로 재현했다.
- Soft의 v>1은 hidden cap 위반이 아니라 허용된 redistribution이다. per-layer cap/clipping influence는 0이고 neutral-relative global update-energy certificate가 hard technical gate다.
- H는 atomic empty history라 전 field에서 inactive였다. Soft objective에는 기존 structural P와 functional P만 실제 influence가 있었다.

## 7. Soft stepwise 핵심 telemetry

| model | alloc | k | q | coverage | risk Δ | coeff L1 | max v | v>1 | pi Neff/top | actual/pred/ratio | structural P | functional-P score | capacity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | BG | 1 | 2.86102 | 0.5562 | -0.00461 | 0.2742 | 1.1040 | 3 | 4.993/L7:0.216 | 1.69706/2.86102/0.593 | 0.000055 | 0.99539 | 0.018972 |
| llama3-8b-inst | BG | 2 | 2.77036 | 0.8522 | -0.01732 | 0.4302 | 1.1223 | 2 | 4.980/L6:0.227 | 2.77835/2.77036/1.003 | 0.000112 | 0.92715 | 0.040770 |
| llama3-8b-inst | BG | 3 | 1.93827 | 0.7824 | -0.00660 | 0.2653 | 1.1056 | 2 | 4.990/L5:0.216 | 1.95804/1.93827/1.010 | 0.000150 | 0.99340 | 0.053606 |
| llama3-8b-inst | BG | 4 | 1.62008 | 0.7882 | -0.00543 | 0.2981 | 1.1505 | 1 | 4.985/L6:0.230 | 0.99658/1.62008/0.615 | 0.000168 | 0.99457 | 0.059033 |
| llama3-8b-inst | BG | 5 | 0.74168 | 0.9049 | -0.00225 | 0.1676 | 1.0726 | 2 | 4.968/L7:0.226 | 0.48000/0.74168/0.647 | 0.000182 | 0.99775 | 0.063929 |
| llama3-8b-inst | BG | 6 | 0.45894 | 0.5201 | -0.00792 | 0.3197 | 1.1389 | 3 | 4.951/L5:0.226 | 0.23596/0.45894/0.514 | 0.000171 | 0.92558 | 0.061163 |
| llama3-8b-inst | BG | 7 | 0.44080 | 0.3272 | -0.00589 | 0.2296 | 1.1113 | 2 | 4.983/L5:0.222 | 0.12774/0.44080/0.290 | 0.000162 | 0.99411 | 0.058173 |
| llama3-8b-inst | BG | 8 | 0.15700 | 0.5153 | -0.00589 | 0.3161 | 1.0907 | 3 | 4.956/L7:0.227 | 0.05888/0.15700/0.375 | 0.000147 | 0.99411 | 0.054803 |
| llama3-8b-inst | RS | 1 | 2.71243 | 0.5616 | -0.01001 | 0.5023 | 1.1432 | 2 | 4.970/L5:0.225 | 1.74601/2.71243/0.644 | 0.000053 | 0.98999 | 0.018479 |
| llama3-8b-inst | RS | 2 | 3.01460 | 0.9078 | -0.02204 | 0.4763 | 1.1306 | 2 | 4.975/L6:0.228 | 2.91843/3.01460/0.968 | 0.000108 | 0.92889 | 0.039960 |
| llama3-8b-inst | RS | 3 | 2.04350 | 0.9282 | -0.00020 | 0.0744 | 1.0274 | 2 | 4.992/L8:0.217 | 1.90294/2.04350/0.931 | 0.000155 | 0.99980 | 0.054174 |
| llama3-8b-inst | RS | 4 | 1.51880 | 0.9608 | -0.00800 | 0.2956 | 1.1423 | 2 | 4.985/L6:0.229 | 1.00871/1.51880/0.664 | 0.000169 | 0.99200 | 0.059343 |
| llama3-8b-inst | RS | 5 | 0.53033 | 0.9478 | -0.00682 | 0.3307 | 1.1132 | 3 | 4.952/L8:0.225 | 0.33120/0.53033/0.625 | 0.000180 | 0.91680 | 0.063823 |
| llama3-8b-inst | RS | 6 | 0.36144 | 0.6943 | -0.00618 | 0.3135 | 1.0917 | 3 | 4.970/L8:0.216 | 0.22131/0.36144/0.612 | 0.000193 | 0.00000 | 0.067473 |
| llama3-8b-inst | RS | 7 | 0.19063 | 0.6250 | -0.01833 | 0.5017 | 1.2399 | 1 | 4.907/L5:0.259 | 0.11580/0.19063/0.607 | 0.000190 | 0.98167 | 0.068361 |
| llama3-8b-inst | RS | 8 | 0.19925 | 0.1557 | -0.01407 | 0.4265 | 1.1856 | 2 | 4.949/L5:0.244 | 0.06880/0.19925/0.345 | 0.000175 | 0.93062 | 0.062803 |
| qwen2.5-7b-inst | BG | 1 | 6.34548 | 0.4868 | -0.07218 | 2.4045 | 1.5364 | 2 | 3.982/L8:0.381 | 1.74578/6.34548/0.275 | 0.002587 | 0.92782 | 0.121158 |
| qwen2.5-7b-inst | BG | 2 | 2.99525 | 0.6033 | -0.11572 | 1.5038 | 1.5020 | 3 | 4.413/L6:0.298 | 2.71185/2.99525/0.905 | 0.003848 | 0.88428 | 0.190867 |
| qwen2.5-7b-inst | BG | 3 | 2.10516 | 0.4318 | -0.10598 | 1.6868 | 1.4671 | 2 | 4.334/L8:0.338 | 2.38526/2.10516/1.133 | 0.004577 | 0.89402 | 0.219428 |
| qwen2.5-7b-inst | BG | 4 | 1.56921 | 0.2831 | -0.14690 | 1.5993 | 1.2541 | 3 | 4.110/L8:0.362 | 1.33942/1.56921/0.854 | 0.004848 | 0.81299 | 0.230056 |
| qwen2.5-7b-inst | BG | 5 | 1.26881 | 0.3485 | -0.15061 | 1.8920 | 1.3647 | 2 | 3.975/L8:0.406 | 0.67713/1.26881/0.534 | 0.004526 | 0.84939 | 0.217310 |
| qwen2.5-7b-inst | BG | 6 | 1.79573 | 0.5647 | -0.10228 | 1.2652 | 1.2888 | 3 | 4.465/L8:0.317 | 0.30507/1.79573/0.170 | 0.004955 | 0.85941 | 0.245151 |
| qwen2.5-7b-inst | BG | 7 | 0.34555 | 0.4096 | -0.16314 | 1.6735 | 1.6015 | 2 | 4.190/L6:0.321 | 0.08399/0.34555/0.243 | 0.004605 | 0.83686 | 0.222103 |
| qwen2.5-7b-inst | BG | 8 | 0.12930 | 0.0389 | -0.16469 | 1.4402 | 1.2782 | 3 | 4.247/L8:0.353 | 0.03286/0.12930/0.254 | 0.004345 | 0.71130 | 0.211103 |
| qwen2.5-7b-inst | RS | 1 | 5.32034 | 0.4939 | -0.13976 | 1.5858 | 1.5388 | 3 | 4.402/L6:0.322 | 2.32200/5.32034/0.436 | 0.002107 | 0.80188 | 0.102203 |
| qwen2.5-7b-inst | RS | 2 | 2.95177 | 0.6182 | -0.14881 | 1.6623 | 1.4347 | 3 | 4.317/L8:0.287 | 2.69776/2.95177/0.914 | 0.003810 | 0.51304 | 0.183822 |
| qwen2.5-7b-inst | RS | 3 | 2.61301 | 0.5549 | -0.14744 | 1.6482 | 1.4007 | 3 | 4.281/L8:0.318 | 2.49805/2.61301/0.956 | 0.004560 | 0.00000 | 0.218682 |
| qwen2.5-7b-inst | RS | 4 | 1.68974 | 0.3715 | -0.16538 | 1.9544 | 1.5280 | 2 | 4.168/L8:0.345 | 1.08458/1.68974/0.642 | 0.004974 | 0.83462 | 0.236202 |
| qwen2.5-7b-inst | RS | 5 | 0.90326 | 0.2704 | -0.16247 | 1.9666 | 1.4590 | 2 | 4.104/L7:0.328 | 0.43887/0.90326/0.486 | 0.004960 | 0.83753 | 0.234497 |
| qwen2.5-7b-inst | RS | 6 | 0.35216 | 0.1488 | -0.21673 | 1.8692 | 1.2603 | 3 | 3.933/L8:0.371 | 0.18932/0.35216/0.538 | 0.004709 | 0.77451 | 0.220106 |
| qwen2.5-7b-inst | RS | 7 | 0.22338 | 0.4726 | -0.19430 | 1.8062 | 1.4451 | 3 | 4.119/L8:0.321 | 0.04762/0.22338/0.213 | 0.004508 | 0.47948 | 0.213509 |
| qwen2.5-7b-inst | RS | 8 | 0.21162 | 0.0668 | -0.16858 | 1.8258 | 1.4852 | 3 | 4.205/L6:0.303 | 0.01681/0.21162/0.079 | 0.004162 | 0.83142 | 0.197481 |

## 8. Actual realization

| model | alloc | Neutral ratio min/med/mean/max | Soft ratio min/med/mean/max | negative actual N/S |
|---|---|---:|---:|---:|
| llama3-8b-inst | BG | 0.278/0.606/0.650/1.008 | 0.290/0.604/0.631/1.010 | 0/0 |
| llama3-8b-inst | RS | 0.331/0.632/0.673/0.971 | 0.345/0.634/0.675/0.968 | 0/0 |
| qwen2.5-7b-inst | BG | 0.180/0.481/0.571/1.122 | 0.170/0.404/0.546/1.133 | 0/0 |
| qwen2.5-7b-inst | RS | 0.029/0.511/0.543/0.933 | 0.079/0.512/0.533/0.956 | 0/0 |

INFERENCE:

- predicted equality는 모두 통과했지만 BF16 actual realization은 step별로 달랐다. 그러므로 local proxy 개선과 terminal preservation은 분리해서 해석해야 한다.
- negative actual progress가 있으면 그 횟수는 위 표에 그대로 남겼으며 rescue/rollback은 수행하지 않았다.

## 9. Compute와 timing

| model | alloc | arm | edit-core | field+slope | functional basis | materialize | target grad | router | F/B/tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | BG | NEUTRAL | 97.430s | 30.561s | 35.909s | 10.918s | 4.970s | 0.0337s | 130/80/25642 |
| llama3-8b-inst | BG | SOFT | 101.023s | 37.598s | 35.040s | 11.277s | 4.838s | 0.0375s | 130/80/25642 |
| llama3-8b-inst | RS | NEUTRAL | 103.041s | 38.923s | 36.280s | 11.145s | 5.096s | 0.0416s | 130/80/25642 |
| llama3-8b-inst | RS | SOFT | 102.685s | 37.482s | 37.524s | 11.269s | 4.556s | 0.0404s | 130/80/25642 |
| qwen2.5-7b-inst | BG | NEUTRAL | 111.782s | 45.151s | 37.452s | 12.255s | 4.057s | 0.0791s | 130/80/23302 |
| qwen2.5-7b-inst | BG | SOFT | 114.369s | 46.705s | 37.940s | 12.420s | 3.879s | 0.0745s | 130/80/23302 |
| qwen2.5-7b-inst | RS | NEUTRAL | 127.135s | 56.697s | 40.727s | 12.726s | 4.059s | 0.0967s | 130/80/23302 |
| qwen2.5-7b-inst | RS | SOFT | 124.647s | 51.193s | 39.617s | 13.780s | 3.708s | 0.0808s | 130/80/23302 |

- router solve는 0.034–0.097초로 병목이 아니다. 주요 비용은 field/physical slope, functional preservation basis, materialization이다.
- 각 arm은 model forward 130, backward 80, target backward 40, slope backward 40, materialization 8을 기록했다. hot-hook dense assembly/hash는 0이다.
- authoritative non-overlapping whole-job forward/token scalar와 true GPU utilization/peak CUDA는 NOT_RECORDED다.

## 10. 세 가지 verdict

### A. Local Soft optimizer가 기회가 있을 때 P/H allocation objective를 달성했는가?

**PASS.** 32/32 Soft field에서 DOF=4, allocation changed, objective influence>0, same-state q equality, energy certificate를 유지하면서 normalized worst P/H risk를 Neutral shadow보다 낮췄다.

### B. Matched predicted strength에서 atomic actual preservation이 개선됐는가?

**MIXED / NOT UNIVERSAL.** Qwen BG Loc가 80→81로 개선됐지만 나머지 세 pair는 count tie이고 continuous locality와 Gen 변화는 혼합적이다. local proxy success를 universal endpoint preservation으로 승격하지 않는다.

### C. Historical H에 관해 무엇을 말할 수 있는가?

**NOT_RECORDED / NO CLAIM.** 이 실험은 atomic H=0이며 persistent history, replay H, sequential round가 모두 0이다. Historical 효과는 추론할 수 없다.

## 11. NOT_RECORDED와 주장 경계

- z-oracle endpoint metrics for this Progress-Simplex activation
- authoritative non-overlapping full-job forward/token scalar beyond the phase ledger
- true GPU utilization and peak CUDA memory
- Official Native phase-specific direct-z backward/iteration identity
- fresh-sample or B100 evidence
- Historical/sequential H effect

Claim boundary: Reused outcome-selected sealed B10; atomic mechanistic/descriptive evidence only; no promotion, B100, sequential, Historical, or universal preservation claim.

## 12. Immutable hashes

| pair | terminal | manifest | action-freeze |
|---|---|---|---|
| llama3-8b-inst-BG | `898a65a68cda84430a2d6aa8c14cc325de3d0c742a1a219c2e2b8dcf5fac8bb1` | `d4adceec47c8fcb3d2c6fe8ecd69065a71052142fb5608cab01bbc84ceede403` | `ebe8206ba9d59fa20270a4c247e8a7d7921ce94749be74248c05a28f92388d96` |
| llama3-8b-inst-RS | `ee28211a6ff39c52b525022394ddb60c45d3d1855eaa554e5c8ec216e6533eda` | `53515c92fcc2dec66036acae067a12cf18d99c1d972af37d529d42c7c9aaa4c5` | `2cfde39b789c4d08669da71fc68559d28ce3edea6c34b3dca26dc332d1cd0fa9` |
| qwen2.5-7b-inst-BG | `5743de9ffa42018379906ebeb10c3155a4080dc3f6e04c44c60cb320dbb07c45` | `39420b5e951dd9108d0c1c1454bd2bd8f7eeaf1728982b0fbeb0ee6db86110dc` | `00696caef225be6ba68e5378a6ea4744ef8bc785fd48f01fc8b18ea9913cd0c2` |
| qwen2.5-7b-inst-RS | `a2fbb5520f8bbc0d8e43d801c666e8d22e76237132a6ac64e423d4fe55b971f4` | `7ed427e49fb0c84c13389e65913d49f30a5198e728e1e4fec396b241042210f3` | `5eb61f0c62583c127d58d4b36d9e532675d02be7ff0455f8be2ca2f132eccf61` |

- stepwise analysis SHA256: `64fa5e0205d640d83afd4e448c75dbf24b71fb230a644fe98697950f50aed212`
- report SHA256 is bound in the sibling receipt after report creation.
