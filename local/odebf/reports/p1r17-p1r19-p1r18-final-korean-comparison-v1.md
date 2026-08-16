# P1R17·P1R19·P1R18 최종 한국어 교차 비교

## 결론

세 결과가 지지하는 결론은 다음과 같다.

1. **P1R17 Full-NoSoft**는 두 모델과 RS/BG 모두에서 W-only Eff/Gen `10/10, 20/20`에 도달했다. 반면 penalty 기반 Full-Soft는 nominal edit retention을 크게 줄였고, 특히 Qwen에서 endpoint Eff/Gen이 RS `9/10,12/20`, BG `8/10,13/20`으로 약해졌다. 이때 높아진 Loc 또는 낮아진 P/capacity를 보존 이득이라 부를 수 없다.
2. **P1R19 strength-preserving package**는 8개 셀 모두 K8/τ1을 완료했고, 모든 셀의 W-only Eff가 `10/10`, Gen은 Llama RS-Neutral의 `18/20`을 제외하면 `20/20`이었다. 즉 P1R17 Soft의 강도 손실과 기존 R13 under-edit를 크게 제거했다.
3. 그러나 P1R19의 matched-strength Soft는 **structural-P와 cumulative capacity가 네 쌍 모두 Neutral보다 높았고**, functional-P mean이 낮아진 쌍은 Llama/RS 하나뿐이었다. 따라서 “강도를 맞추면 Soft가 일반적으로 preservation을 개선한다”는 가설은 지지되지 않는다.
4. P1R19는 floor 제거, physical W-only slope, demand/equality routing을 함께 바꾼 **full method-package delta**다. Soft allocation 하나의 고립된 인과효과로 해석할 수 없다.
5. P1R18의 AlphaEdit/MEMIT은 같은 seal/order에서 양 모델 모두 Eff/Gen `10/10,20/20`을 달성했다. 이는 기술적 기준점이지만 one-shot과 K8의 계측 범위가 다르므로 시간 배수를 직접 주장하지 않는다.

최종 판정은 `MECHANISTIC_DESCRIPTIVE_ONLY / SCIENTIFIC_PROMOTION_FALSE / METHOD_HOLD`다.

## 입력 artifact와 독립 검증

### 고정 입력

- P1R17 보고서: SHA-256 `7860a9242248759cd62db33939b04906a030fb8ae7a7b5d0cbe0642aadb33168`, 23,508 bytes, 280 lines. 서버1에서 재해시가 일치했다.
- P1R19 보고서: SHA-256 `267fbf88eb4bdff5ad8c3202e7363f6ef56d2f6f3db6219c9e64f6176a98ab5f`, 10,439 bytes, 129 lines. 서버1에서 재해시가 일치했다.
- P1R19 manifest: SHA-256 `d064d8f4e17e9373689f98da6ccc37c6281e86492995c7b71b57de500deeb87c`.
- P1R19 handoff receipt: SHA-256 `73336136f1250d5cccda009671d4630a00806aca50053f2c5631b1a05e70a468`.
- P1R19 producer file-tree binding: `909873c7d9d0a2ea7dc8d9f821ada7e709a0699d22b20c83c0c50047eebab91b`.
- P1R18 업데이트 보고서: SHA-256 `768bfd29a326d503f629513d707a8560c0dbd002b4a26237520b8a27cdfe4ac6`, 13,476 bytes. 이 task에는 원본 bytes가 로컬 전달되지 않아 GH의 immutable hash-bound receipt를 사용했으며, 구 hash `e95224...`는 사용하지 않았다.
- R13 exact Dynamic reference packet: SHA-256 `386521002648d7599af530f382a5c1b81486923a052977f20bf78b4cac1d7f91`, 15,933 bytes.

### P1R19 handoff 검증 결과

- 전달 디렉터리는 mode `0700`, top-level 11개는 모두 regular non-symlink mode `0600`이었다.
- manifest에 선언된 9개 파일(8 cell JSON + terminal report)은 path/size/mode/SHA-256가 전부 일치했다.
- 나머지 두 파일은 `manifest.json`과 `handoff-receipt.json`이며, receipt가 report/manifest SHA와 manifest root를 정확히 연결한다.
- 8개 cell JSON의 endpoint/count/NLL/routing/P/capacity/timing 표를 terminal report와 재대조했고 불일치는 없었다.
- raw prompt/target/token tensor payload는 패키지에 없고, 결과 재계산이나 모델 실행은 하지 않았다.

## 비교 가능성과 경계

### 정확히 공통인 축

- seal root: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`
- request order: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- W0 공식 count: Llama `2/10,3/20,84/100`; Qwen `0/10,4/20,80/100`
- frozen CounterFact evaluator, action-freeze 이후 공식 평가, alias별 동일 W0
- P1R17/P1R19: Dynamic only, K=8, h=1/8, τ=1, first-hit observation-only

### 직접 등치하면 안 되는 축

- P1R17과 P1R19는 서로 다른 method package다. P1R17은 penalty 기반 Soft, P1R19는 strength equality/demand routing이다.
- P1R19 Neutral/Soft는 k0 field identity는 exact지만, 첫 write 이후 W,z 궤적이 달라진다. 이후 step의 값은 동일 상태 pointwise contrast가 아니라 trajectory-level contrast다.
- P1R18 AlphaEdit/MEMIT은 one-shot joint B10이고 P1R17/P1R19는 K8이다. wall-time 포함 범위와 내부 계측이 다르다.
- P/H/capacity의 절대값은 method-specific geometry와 누적 경로에 의존하므로 task 간 절대 비교보다 같은 task의 Neutral/Soft 쌍 비교를 우선한다.
- R13은 exact receipt가 있는 마지막 유효 prefix만 쓴다. 특히 **Qwen BG-Soft k8은 존재하지 않으며**, k7/τ=.875만 사용한다.

## Endpoint 공식 count: P1R17 대 P1R19

| 모델 | Allocation | Router | P1R17 W E/G/L | P1R19 W E/G/L | R19−R17 |
|---|---|---|---|---|---|
| Llama | RS | Neutral/NoSoft | 10/10, 20/20, 81/100 | 10/10, 18/20, 82/100 | 0, -2, +1 |
| Llama | RS | Soft | 10/10, 17/20, 82/100 | 10/10, 20/20, 82/100 | 0, +3, 0 |
| Llama | BG | Neutral/NoSoft | 10/10, 20/20, 79/100 | 10/10, 20/20, 81/100 | 0, 0, +2 |
| Llama | BG | Soft | 10/10, 18/20, 83/100 | 10/10, 20/20, 81/100 | 0, +2, -2 |
| Qwen | RS | Neutral/NoSoft | 10/10, 20/20, 78/100 | 10/10, 20/20, 79/100 | 0, 0, +1 |
| Qwen | RS | Soft | 9/10, 12/20, 81/100 | 10/10, 20/20, 79/100 | +1, +8, -2 |
| Qwen | BG | Neutral/NoSoft | 10/10, 20/20, 81/100 | 10/10, 20/20, 80/100 | 0, 0, -1 |
| Qwen | BG | Soft | 8/10, 13/20, 80/100 | 10/10, 20/20, 80/100 | +2, +7, 0 |

P1R17의 Soft−NoSoft는 Llama RS `0,-3,+1`, Llama BG `0,-2,+4`, Qwen RS `-1,-8,+3`, Qwen BG `-2,-7,-1`이었다. P1R19의 Soft−Neutral은 Llama RS에서만 `0,+2,0`이고 나머지 세 쌍은 `0,0,0`이었다. P1R19가 endpoint strength를 되찾았지만 Soft 자체의 일반적인 공식 metric 이득은 관측되지 않았다.

## 연속 NLL와 margin

표의 `N/O/M`은 mean target-new NLL / mean target-old NLL / receipt-defined margin이다. Eff/Gen margin은 `old-new`, Loc preservation margin은 `new-old`다.

### P1R17 W-only endpoint

| Cell | Eff N/O/M | Gen N/O/M | Loc N/O/M |
|---|---|---|---|
| L RS NoSoft | .016689 / 12.918750 / 12.902061 | .727423 / 9.992188 / 9.264765 | 8.371250 / 4.251235 / 4.120015 |
| L RS Soft | 1.006683 / 5.829688 / 4.823004 | 2.839014 / 5.668066 / 2.829053 | 8.418125 / 4.320745 / 4.097380 |
| L BG NoSoft | .058983 / 10.162500 / 10.103517 | 1.175311 / 8.679688 / 7.504376 | 8.223438 / 4.366880 / 3.856558 |
| L BG Soft | 1.136572 / 5.507812 / 4.371240 | 2.911475 / 6.111523 / 3.200049 | 8.391719 / 4.341609 / 4.050110 |
| Q RS NoSoft | .025308 / 17.137500 / 17.112192 | 2.191266 / 10.508984 / 8.317719 | 8.856250 / 5.170669 / 3.685581 |
| Q RS Soft | 2.728516 / 6.122656 / 3.394141 | 4.448340 / 6.060840 / 1.612500 | 8.853906 / 5.168325 / 3.685581 |
| Q BG NoSoft | .151111 / 11.303125 / 11.152014 | 2.494073 / 8.914844 / 6.420770 | 8.841875 / 5.177900 / 3.663975 |
| Q BG Soft | 3.521680 / 6.189844 / 2.668164 | 5.018115 / 6.295605 / 1.277490 | 8.867500 / 5.169077 / 3.698423 |

Count가 같은 Llama Soft도 NoSoft보다 target-new NLL이 높고 Gen margin이 작다. 따라서 count만으로 강도가 완전히 같다고 볼 수 없다.

### P1R19 W-only endpoint

| Cell | Eff N/O/M | Gen N/O/M | Loc N/O/M |
|---|---|---|---|
| L RS Neutral | .912091 / 7.114063 / 6.201971 | 2.361707 / 6.552148 / 4.190442 | 8.343594 / 4.338984 / 4.004609 |
| L RS Soft | .386723 / 9.143750 / 8.757027 | 1.396069 / 7.824219 / 6.428149 | 8.360156 / 4.326841 / 4.033315 |
| L BG Neutral | .067722 / 10.443750 / 10.376028 | .911218 / 9.675000 / 8.763782 | 8.231563 / 4.334651 / 3.896912 |
| L BG Soft | .031602 / 10.565625 / 10.534023 | .881331 / 9.585938 / 8.704607 | 8.265469 / 4.320354 / 3.945115 |
| Q RS Neutral | .037691 / 18.656250 / 18.618559 | 1.609158 / 11.059375 / 9.450217 | 8.848750 / 5.173379 / 3.675371 |
| Q RS Soft | .026397 / 19.187500 / 19.161103 | 1.606291 / 11.493750 / 9.887459 | 8.845156 / 5.179551 / 3.665605 |
| Q BG Neutral | .086884 / 12.428125 / 12.341241 | 2.101981 / 10.054688 / 7.952707 | 8.839375 / 5.175820 / 3.663555 |
| Q BG Soft | .215604 / 12.556250 / 12.340646 | 2.128979 / 9.977344 / 7.848364 | 8.840938 / 5.179531 / 3.661406 |

P1R19 controller endpoint target-new NLL은 L RS N/S `1.002913/.483299`, L BG `.119672/.053443`, Q RS `.012528/.011883`, Q BG `.061009/.104315`였다. Qwen BG에서는 Soft가 이 연속 지표를 오히려 악화했다.

### z-oracle

P1R17 endpoint z-oracle은 `NOT_RECORDED`다. P1R19은 모든 셀에서 z-oracle Eff/Gen `10/10,20/20`이며 Loc은 W-only와 동일했다.

| P1R19 Cell | z Eff N/O/M | z Gen N/O/M |
|---|---|---|
| L RS Neutral | .010175 / 17.762500 / 17.752325 | .346572 / 15.050000 / 14.703428 |
| L RS Soft | .002755 / 18.556250 / 18.553495 | .382780 / 15.134375 / 14.751595 |
| L BG Neutral | .014359 / 12.162500 / 12.148141 | .697687 / 10.423438 / 9.725751 |
| L BG Soft | .024580 / 11.275000 / 11.250420 | .662975 / 10.293750 / 9.630775 |
| Q RS Neutral | .001148 / 21.362500 / 21.361352 | .943304 / 12.226563 / 11.283258 |
| Q RS Soft | .005809 / 20.096875 / 20.091066 | 1.140240 / 11.856250 / 10.716010 |
| Q BG Neutral | .036656 / 13.421875 / 13.385219 | 1.808276 / 10.204688 / 8.396411 |
| Q BG Soft | .021849 / 14.362500 / 14.340651 | 2.059967 / 10.752344 / 8.692377 |

W-only도 이미 대부분 ceiling count이므로 z-oracle count gap은 거의 닫혔다. 다만 W/z 연속 NLL 차이는 남아 있어 writer realization이 완전하다는 뜻은 아니다.

## 실제 강도·coverage·realization

### P1R17: penalty Soft의 강도 감쇠

NoSoft의 endpoint `r_E`는 네 셀 모두 약 1이고 nominal distance는 0이다. Soft의 endpoint는 다음과 같다.

| Cell | λ_P | r_E | R_PH | distance-to-u | cumulative capacity |
|---|---:|---:|---:|---:|---:|
| L RS Soft | .749366 | .101464 | .125299 | .808212 | .386590 |
| L BG Soft | .749366 | .150508 | .178796 | .719549 | .371102 |
| Q RS Soft | .887570 | .073837 | .078672 | .887031 | 1.236405 |
| Q BG Soft | .887570 | .052344 | .056018 | .932554 | 1.017215 |

Soft가 P/capacity를 줄인 것은 사실이지만 nominal semantic edit의 약 5–15%만 유지한 말단 상태다. 따라서 Loc/P 개선을 preservation gain으로 해석하지 않는다.

### P1R19: equality와 demand coverage

각 trajectory에서 `physical_a_dot_v == alpha_apply`가 receipt 정밀도로 성립했다. 그러나 demand가 caps를 넘으면 coverage는 1 미만이며, Neutral/Soft가 서로 다른 후속 상태에 있기 때문에 같은 numeric alpha를 공유한다는 뜻은 아니다.

| Cell | endpoint α_req | α_apply=aᵀv | endpoint coverage | K8 min coverage | endpoint DOF / fixed bounds |
|---|---:|---:|---:|---:|---:|
| L RS Neutral | .136088 | .136088 | 1.000000 | .795359 | 4 / 0 |
| L RS Soft | 1.299793 | 1.299793 | 1.000000 | .795359 | 4 / 0 |
| L BG Neutral | 5.696158 | .364479 | .063987 | .063987 | 0 / 5 |
| L BG Soft | 2.894510 | .320618 | .110767 | .110767 | 0 / 5 |
| Q RS Neutral | 1.895083 | .237446 | .125296 | .125296 | 0 / 5 |
| Q RS Soft | 2.138174 | .062309 | .029141 | .029141 | 0 / 5 |
| Q BG Neutral | 1.674459 | .462803 | .276390 | .276390 | 0 / 5 |
| Q BG Soft | 2.065851 | 1.105379 | .535072 | .412256 | 0 / 5 |

30/64 transition에서 feasible allocation dimension이 0이었다. Qwen 네 셀의 endpoint velocity는 모두 다섯 cap에 붙었고, 이런 상태에서는 Soft가 share를 바꿀 자유도가 사실상 없다.

### Endpoint physical realization 관측

| Task/cell | Predicted aᵀv | Actual W progress | Actual/pred. | cosine | norm gain | residual ratio |
|---|---:|---:|---:|---:|---:|---:|
| R17 L RS NoSoft | .160623 | .003129 | .019481 | .028454 | .526078 | .797592 |
| R17 L RS Soft | .242844 | .003283 | .013521 | .077383 | .126159 | .622490 |
| R17 L BG NoSoft | .039930 | .036073 | .903400 | .191423 | .428251 | .684826 |
| R17 L BG Soft | .414740 | .086346 | .208194 | .117147 | 23.245477 | 23.193218 |
| R17 Q RS NoSoft | .086109 | -.055203 | -.641083 | .028214 | .618099 | .875232 |
| R17 Q RS Soft | .327453 | .002962 | .009045 | .034393 | .078159 | .625754 |
| R17 Q BG NoSoft | .156889 | -.154229 | -.983043 | .095764 | 10.721658 | 10.680564 |
| R17 Q BG Soft | .186330 | -.217335 | -1.166398 | .080960 | .504256 | .834370 |
| R19 L RS Neutral | .136088 | .114337 | .840173 | .111982 | .130795 | .993929 |
| R19 L RS Soft | 1.299793 | .798632 | .614430 | .100792 | 1.058989 | 1.385122 |
| R19 L BG Neutral | .364479 | .042007 | .115253 | .399310 | 44.956305 | 44.642203 |
| R19 L BG Soft | .320618 | .104018 | .324429 | .335366 | 26.383336 | 26.107573 |
| R19 Q RS Neutral | .237446 | .051243 | .215809 | -.066379 | .984021 | 1.450967 |
| R19 Q RS Soft | .062309 | .003938 | .063198 | -.094704 | .879064 | 1.393808 |
| R19 Q BG Neutral | .462803 | .073182 | .158127 | .267226 | 4.919389 | 4.829230 |
| R19 Q BG Soft | 1.105379 | .165057 | .149322 | .338870 | 9.965950 | 9.717192 |

큰 norm/residual ratio는 일부 intended displacement가 거의 0인 행의 분모 민감도를 포함한다. P1R19 Qwen BG-Soft k7은 predicted `+.6745`와 actual `-.0733`을 기록했지만 no-retry 계약에 따라 그대로 수용했다. Equality-constrained slope와 BF16 실제 실현은 구분해야 한다.

## Layer routing, q, entropy, HHI, effective layers

### P1R17 endpoint

| Cell | L4..L8 coefficient share | Neff | HHI |
|---|---|---:|---:|
| L RS NoSoft | .2,.2,.2,.2,.2 | 5.000 | .200 |
| L RS Soft | .175643,.244304,.227379,.187547,.165127 | 4.886 | .205 |
| L BG NoSoft | .2,.2,.2,.2,.2 | 5.000 | .200 |
| L BG Soft | .206459,.222540,.207616,.193174,.170211 | 4.962 | .202 |
| Q RS NoSoft | .2,.2,.2,.2,.2 | 5.000 | .200 |
| Q RS Soft | .114002,.099304,.262399,.315160,.209134 | 4.259 | .235 |
| Q BG NoSoft | .25,.25,.25,.25,0 | 4.000 | .250 |
| Q BG Soft | .084553,≈0,.246613,.297255,.371578 | 3.397 | .294 |

P1R17 terminal report에는 normalized entropy와 별도 명칭의 `q` scalar/vector가 없다. 이를 다른 필드로 소급 재명명하지 않고 `NOT_RECORDED`로 둔다.

### P1R19 endpoint

| Cell | L4..L8 coefficient share | entropy(norm.) | HHI | Neff |
|---|---|---:|---:|---:|
| L RS Neutral | .147059,.196356,.225467,.222354,.208764 | .993347 | .204041 | 4.900972 |
| L RS Soft | .238680,.249046,.203019,.176696,.132560 | .985429 | .209002 | 4.784643 |
| L BG Neutral | .2,.2,.2,.2,.2 | 1.000000 | .200000 | 5.000000 |
| L BG Soft | .2,.2,.2,.2,.2 | 1.000000 | .200000 | 5.000000 |
| Q RS Neutral | .2,.2,.2,.2,.2 | 1.000000 | .200000 | 5.000000 |
| Q RS Soft | .2,.2,.2,.2,.2 | 1.000000 | .200000 | 5.000000 |
| Q BG Neutral | .2,.2,.2,.2,.2 | 1.000000 | .200000 | 5.000000 |
| Q BG Soft | .2,.2,.2,.2,.2 | 1.000000 | .200000 | 5.000000 |

P1R19 cell schema에도 별도 `q` 명칭 필드는 없다. authoritative allocation은 velocity/applied-coefficient share와 `feasible_dimension`으로 기록돼 있으므로 `xi_star` 등을 임의로 q라고 부르지 않는다. Router는 기존 target gradient의 `alpha_req`와 기존 no-hook physical slope를 재사용했고 추가 model forward/backward는 0이었다.

## P/H와 cumulative capacity

### P1R17 내부 대비

P1R17 보고서가 제공하는 `P mean/raw max`는 기능적 P 관측값이며, 독립 structural-P endpoint scalar는 `NOT_RECORDED`다. H는 empty history로 0/inactive다.

| 모델/allocation | P mean NoSoft→Soft | P raw max NoSoft→Soft | capacity NoSoft→Soft |
|---|---:|---:|---:|
| Llama RS | .009552→.001168 | .025568→.005415 | 2.779155→.386590 |
| Llama BG | .005233→.001112 | .015475→.002776 | 2.404178→.371102 |
| Qwen RS | .000882→.000455 | .001962→.001031 | 18.992679→1.236405 |
| Qwen BG | .000912→.000575 | .002667→.001725 | 15.305625→1.017215 |

이 감소는 r_E의 대규모 하락과 함께 발생했으므로 preservation evidence가 아니다.

### P1R19 matched-strength 내부 대비

Structural-P는 pinned EasyEdit Wikipedia key-second-moment proxy이며 Soft allocation에만 사용됐고 threshold/veto가 아니다. Functional P/H는 observation-only이며 H는 inactive다.

| 모델/allocation | structural-P N→S | functional-P mean N→S | functional raw max N→S | capacity N→S |
|---|---:|---:|---:|---:|
| Llama RS | .000756→.001110 | .003140→.002637 | .022795→.018538 | .528367→.932499 |
| Llama BG | .003517→.003965 | .007087→.007408 | .059884→.057741 | 2.425964→3.113790 |
| Qwen RS | .141306→.164709 | .057567→.057796 | .564298→.564683 | 15.634784→18.338693 |
| Qwen BG | .152006→.152152 | .075607→.077028 | .746282→.760491 | 16.619826→16.715879 |

FACT: structural-P와 capacity는 네 쌍 모두 Soft가 높다. Functional-P mean이 개선된 것은 Llama/RS 하나뿐이다. Llama/BG의 raw max만 소폭 낮아졌지만 mean/structural-P/capacity는 높아 “일반 P 개선”으로 분류할 수 없다.

## R13 exact Dynamic reference

| Cell | R13 last valid W E/G/L | source point | P1R19 W E/G/L |
|---|---|---|---|
| L RS-N | 9/10,17/20,82/100 | k8 | 10/10,18/20,82/100 |
| L RS-S | 10/10,17/20,82/100 | k8 | 10/10,20/20,82/100 |
| L BG-N | 8/10,15/20,83/100 | k6 | 10/10,20/20,81/100 |
| L BG-S | 10/10,18/20,83/100 | k8 | 10/10,20/20,81/100 |
| Q RS-N | 5/10,10/20,80/100 | k4 | 10/10,20/20,79/100 |
| Q RS-S | 4/10,7/20,81/100 | k3 | 10/10,20/20,79/100 |
| Q BG-N | 9/10,17/20,81/100 | k8 | 10/10,20/20,80/100 |
| Q BG-S | 8/10,14/20,80/100 | **k7/τ=.875** | 10/10,20/20,80/100 |

R13의 중도 종료는 last-valid prefix와 비교한 것이다. R13 대비 P1R19의 개선은 full method-package contrast이며, Qwen BG-S k8을 합성·보간하지 않았다.

## P1R18 AlphaEdit/MEMIT descriptive baseline

| 모델 | W0 E/G/L | AlphaEdit E/G/L | edit wall | MEMIT E/G/L | edit wall |
|---|---|---|---:|---|---:|
| Llama | 2/10,3/20,84/100 | 10/10,20/20,79/100 | 51.90s | 10/10,20/20,82/100 | 68.85s |
| Qwen | 0/10,4/20,80/100 | 10/10,20/20,80/100 | 49.88s | 10/10,20/20,80/100 | 88.63s |

- AlphaEdit은 caller-scoped BF16 key→FP32 adapter를 사용했다.
- P1R18은 continuous NLL과 내부 forward/backward/token count를 기록하지 않았다.
- one-shot edit wall과 K8 end-to-end wall의 범위가 다르므로 precise pure-edit multiplier를 계산하지 않는다.
- 이 표는 동일 seal/order의 독립적인 표준-method 기준점일 뿐 P1R17/P1R19 설정을 선택하거나 변경하지 않았다.

## Compute, wall, memory

### P1R17

| Cell | partitioned F/B/tokens | rollout F/B | functional-basis wall | post-freeze wall | scheduler wall | MaxRSS KiB |
|---|---|---|---:|---:|---:|---:|
| L RS NoSoft | 4551/90/147767 | 4102/80 | 86.70s | 139.34s | 6959s | 8419112 |
| L RS Soft | 4551/90/147767 | 4102/80 | 100.75s | 139.85s | 7222s | 8413732 |
| L BG NoSoft | 4551/90/147767 | 4102/80 | 82.95s | 139.80s | 7002s | 8474860 |
| L BG Soft | 4551/90/147767 | 4102/80 | 96.65s | 140.91s | 7274s | 8433464 |
| Q RS NoSoft | 4555/90/134487 | 4102/80 | 90.68s | 157.32s | 7908s | 11870908 |
| Q RS Soft | 4555/90/134487 | 4102/80 | 92.64s | 156.28s | 7988s | 11865452 |
| Q BG NoSoft | 4555/90/134487 | 4102/80 | 96.55s | 158.76s | 7931s | 11895024 |
| Q BG Soft | 4555/90/134487 | 4102/80 | 92.89s | 157.27s | 7912s | 11905060 |

P1R17에는 authoritative `edit_core`, `artifact/model_load`, `stepwise_eval`의 P1R19 호환 분해가 없다. 위 partition을 재조합해 임의의 whole-job counter나 pure-edit multiplier를 만들지 않는다. 알려진 Stage3 SLSQP→certified trust-constr fallback 23회는 과학적 retry가 아니다.

### P1R19

단위는 초다. `accounted wall`이 세부 phase를 포함한 total receipt이고 scheduler는 Slurm elapsed다.

| Cell | edit_core | functional probe | stepwise eval | artifact/model load | accounted wall | scheduler | MaxRSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| L RS Neutral | 2769.2 | 87.2 | 255.0 | 9.1 | 4218.9 | 4245 | 8.44 GB |
| L RS Soft | 2741.9 | 87.2 | 245.8 | 10.3 | 4132.6 | 4158 | 8.42 GB |
| L BG Neutral | 2609.9 | 81.9 | 229.4 | 10.0 | 3913.4 | 3938 | 9.78 GB |
| L BG Soft | 2558.0 | 81.2 | 232.9 | 8.9 | 3863.0 | 3889 | 8.51 GB |
| Q RS Neutral | 2864.7 | 88.1 | 268.3 | 14.4 | 4370.1 | 4403 | 21.06 GB |
| Q RS Soft | 2840.5 | 87.1 | 255.1 | 10.2 | 4276.7 | 4306 | 13.96 GB |
| Q BG Neutral | 3131.5 | 95.0 | 293.5 | 9.2 | 4762.9 | 4795 | 12.06 GB |
| Q BG Soft | 2741.4 | 84.6 | 279.6 | 9.9 | 4298.0 | 4331 | 11.99 GB |

- cell당 rollout은 model forward 2194, backward 240, target backward 80, functional-P replay 64다.
- stepwise evaluation은 model forward 789, evaluator forward 780이다.
- router가 추가한 model forward/backward는 0이다.
- 비용의 주 구성은 K8 field/state refresh, functional probe, stepwise evaluation이다. QP 또는 covariance를 단일 병목이라고 결론낼 근거가 없다.
- Native reference는 edit-core 약 25.7–29.8s, end-to-end 27.0–31.1s 범위로 별도 계측됐다. 알고리즘과 범위가 달라 R13/P1R17/P1R19 대비 정확한 pure-edit 배수를 주장하지 않는다.

## FACT / INFERENCE / NOT_RECORDED

### FACT

- P1R17과 P1R19의 각 8개 trajectory는 K8/τ1, action-freeze, terminal/manifest, exact W0 restore를 완료했다.
- P1R17 NoSoft는 4/4 Eff/Gen ceiling count에 도달했다. P1R17 Soft는 4/4에서 말단 r_E가 크게 낮았고 Gen이 모두 NoSoft보다 낮았다.
- P1R19은 8/8 Eff `10/10`; 7/8 Gen `20/20`, Llama RS Neutral만 `18/20`이었다. 모든 z-oracle Eff/Gen은 `10/10,20/20`이다.
- P1R19 Soft의 structural-P와 cumulative capacity는 네 Neutral/Soft 쌍 모두 증가했다. Functional-P mean은 Llama RS에서만 감소했다.
- P1R19 Soft의 Loc은 모든 쌍에서 Neutral과 동일했다.
- P1R19의 equality slope는 통과했지만 coverage가 1보다 작은 cell과 DOF=0/all-cap step이 많았고, 실제 BF16 progress는 예측과 다를 수 있었다.
- P1R18 AlphaEdit/MEMIT은 양 alias에서 Eff/Gen ceiling count를 달성했다.

### INFERENCE

- P1R17의 낮은 P/capacity와 일부 높은 Loc는 edit-strength 감쇠에 의해 교란되므로 preservation gain이 아니다.
- P1R19는 P1R17 Soft strength loss와 R13 under-edit를 대부분 제거했지만, equality-matched allocation이 일반적으로 손상을 줄였다는 증거는 없다.
- Llama RS의 `Gen +2` 및 functional-P 감소는 국소 신호이나, structural-P/capacity 증가와 다른 세 쌍의 무이득/악화 때문에 보편적 Soft 정책을 지지하지 않는다.
- allocation RS/BG 효과는 모델과 method package에 따라 혼합되어 있으며, 독립적인 allocation 인과효과로 promotion할 수 없다.
- 표준 one-shot baseline과 비교하면 K8 method가 ceiling count에 도달할 수 있음을 보이지만, 계산 효율 또는 우월성의 직접 배수 비교는 성립하지 않는다.

### NOT_RECORDED / NOT_PROVEN

- P1R17 endpoint z-oracle, 독립 structural-P endpoint scalar, normalized entropy, 명시적 `q` 필드.
- P1R17의 P1R19 호환 edit_core/functional_probe/stepwise_eval/model-load phase 분해와 하나의 authoritative whole-job F/B/token scalar.
- P1R18 continuous NLL, 내부 forward/backward/token, end-to-end phase decomposition.
- P1R19의 별도 `q` 명칭 필드; authoritative allocation fields를 q로 소급 재명명하지 않았다.
- fresh unseen seal replication, 통계적 불확실성, alias-invariant 일반화, scientific promotion.
- R13 Qwen BG-Soft k8. 마지막 exact point는 k7/τ=.875이다.

## 최종 판정

P1R19는 P1R17 penalty Soft가 잃었던 semantic edit strength와 endpoint Eff/Gen을 실질적으로 복구했다. 그러나 strength/coverage와 physical realization을 먼저 맞춰 본 뒤에도, Soft의 structural-P와 cumulative capacity가 네 쌍 모두 높고 functional-P가 개선된 쌍은 Llama RS 하나뿐이었다. 그러므로 **일반적인 matched-strength preservation 효과는 지지되지 않는다**.

이 결론은 동일 outcome-selected R13 seal에서의 기계론적·기술적 비교다. P1R19는 full method-package delta이고 P1R18은 descriptive one-shot baseline이다. fresh-seal 검증과 promotion authority가 없으므로 `scientific_promotion=false`를 유지한다.
