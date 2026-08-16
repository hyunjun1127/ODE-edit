# P1R17 Full-Drive Soft Dynamic 2×2×2 터미널 보고서

## 요약

P1R17의 8개 Dynamic 궤적은 모두 스케줄러 `COMPLETED 0:0`, K=8, τ=1로 종료했다. 모든 셀에서 action-freeze 이후 평가, terminal/manifest 생성, 정확한 W0 복원, persistent commit/history 0이 확인됐다. 과학적 retry/rescue와 typed scientific termination은 없었다.

핵심 결과는 명확하다.

- Full-NoSoft는 양 모델·양 allocation에서 Eff/Gen `10/10, 20/20`을 달성했다.
- Full-Soft는 Llama에서 Eff를 유지했지만 Gen이 RS `17/20`, BG `18/20`으로 낮아졌다. Qwen에서는 RS `9/10, 12/20`, BG `8/10, 13/20`으로 더 크게 약해졌다.
- Soft의 locality 변화는 Llama RS `+1`, BG `+4`, Qwen RS `+3`, BG `-1`이지만, 같은 셀에서 실제 edit retention `r_E`가 크게 낮아졌다. 따라서 이를 matched-strength preservation gain으로 해석할 수 없다.
- Full-NoSoft는 각 셀의 nominal velocity를 그대로 사용해 `r_E≈1`, distance-to-nominal=0이었다. Full-Soft의 말단 `r_E`는 Llama RS `0.1015`, BG `0.1505`, Qwen RS `0.0738`, BG `0.0523`으로, penalty 기반 Soft의 edit-strength 손실이 관측됐다.
- 이 결과는 outcome-selected R13 seal을 재사용한 mechanistic regression이다. `scientific_promotion=false`이며 fresh-seal 일반화 주장을 하지 않는다.

## 고정 계약과 provenance

- 실행 checkpoint: `f705150920b407914edd211e543a447b8143cf9c`
- parent: `ac6dd1f6d32b6fad103175f9a7fa152daf2ca9c5`
- tree: `e41564fe67dd437df5b9a0dde340147e15e6569e`
- source manifest SHA-256: `b2fa2c2b4366fa6a3744e2c423a26d9763b982d607e04864fd263863d04ea897`
- numerical lock file SHA-256: `561679b4d72e3c0145424fa8341a0cbdd3966859723eb14d2a4b0b8ebf6355af`
- terminal-bound numerical lock SHA-256: `7e09a1e75542c32f240e8591cd58db15c735e3aeb68d3edb3c694ca06de848ea`
- lambda lock SHA-256: `7b17f444c19679a54bab298f067aa571481af3a9a07d50e931ee2401af7fc82f`
- Llama λ_P: `0.7493662147044176`
- Qwen λ_P: `0.8875695033966782`
- R13 seal root: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`
- request order: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- panel: `{Llama,Qwen} × {RS,BG} × {Full-NoSoft,Full-Soft}`, Dynamic only
- 공통 grid: K=8, h=1/8, τ=1; first-hit observation only

모델별 λ는 각 모델의 R13 RS-Neutral Dynamic k0 geometry에서 outcome-free solver-only replay로 고정했다. 같은 모델 안에서는 RS/BG와 전 K8 단계에 동일한 λ를 사용했다. alias-invariant hyperparameter 또는 universal Soft policy 주장은 하지 않는다.

## 실행 무결성

| Job | 모델 | Allocation | Routing | Scheduler | τ | W0 restore | Retry/rescue | Terminal artifacts |
|---:|---|---|---|---|---:|---|---:|---|
| 18310 | Llama | RS | Full-NoSoft | COMPLETED 0:0 | 1 | exact | 0 | PASS |
| 18311 | Llama | RS | Full-Soft | COMPLETED 0:0 | 1 | exact | 0 | PASS |
| 18312 | Qwen | RS | Full-NoSoft | COMPLETED 0:0 | 1 | exact | 0 | PASS |
| 18313 | Qwen | RS | Full-Soft | COMPLETED 0:0 | 1 | exact | 0 | PASS |
| 18314 | Llama | BG | Full-NoSoft | COMPLETED 0:0 | 1 | exact | 0 | PASS |
| 18315 | Llama | BG | Full-Soft | COMPLETED 0:0 | 1 | exact | 0 | PASS |
| 18316 | Qwen | BG | Full-NoSoft | COMPLETED 0:0 | 1 | exact | 0 | PASS |
| 18317 | Qwen | BG | Full-Soft | COMPLETED 0:0 | 1 | exact | 0 | PASS |

공통으로 persistent commit=0, history append=0, heldout controller access=0, Native/direct-z cold-controller access=0, hard H/P influence=0이다. 알려진 Stage3 `SLSQP stationarity → certified trust-constr` 수치 fallback은 총 23회였고, 모두 인증된 동일 목적 해법으로 종료했다. 과학적 retry나 routing 변경은 0이다.

## Endpoint W-only 평가

표의 `N/O/M`은 평균 target-new NLL / target-old NLL / receipt-defined margin이다. Eff/Gen margin은 `old-new`, Loc preservation margin은 `new-old`이다.

| Job/cell | Eff | Eff N/O/M | Gen | Gen N/O/M | Loc | Loc N/O/M |
|---|---:|---|---:|---|---:|---|
| 18310 L RS NoSoft | 10/10 | 0.016689 / 12.918750 / 12.902061 | 20/20 | 0.727423 / 9.992188 / 9.264765 | 81/100 | 8.371250 / 4.251235 / 4.120015 |
| 18311 L RS Soft | 10/10 | 1.006683 / 5.829688 / 4.823004 | 17/20 | 2.839014 / 5.668066 / 2.829053 | 82/100 | 8.418125 / 4.320745 / 4.097380 |
| 18312 Q RS NoSoft | 10/10 | 0.025308 / 17.137500 / 17.112192 | 20/20 | 2.191266 / 10.508984 / 8.317719 | 78/100 | 8.856250 / 5.170669 / 3.685581 |
| 18313 Q RS Soft | 9/10 | 2.728516 / 6.122656 / 3.394141 | 12/20 | 4.448340 / 6.060840 / 1.612500 | 81/100 | 8.853906 / 5.168325 / 3.685581 |
| 18314 L BG NoSoft | 10/10 | 0.058983 / 10.162500 / 10.103517 | 20/20 | 1.175311 / 8.679688 / 7.504376 | 79/100 | 8.223438 / 4.366880 / 3.856558 |
| 18315 L BG Soft | 10/10 | 1.136572 / 5.507812 / 4.371240 | 18/20 | 2.911475 / 6.111523 / 3.200049 | 83/100 | 8.391719 / 4.341609 / 4.050110 |
| 18316 Q BG NoSoft | 10/10 | 0.151111 / 11.303125 / 11.152014 | 20/20 | 2.494073 / 8.914844 / 6.420770 | 81/100 | 8.841875 / 5.177900 / 3.663975 |
| 18317 Q BG Soft | 8/10 | 3.521680 / 6.189844 / 2.668164 | 13/20 | 5.018115 / 6.295605 / 1.277490 | 80/100 | 8.867500 / 5.169077 / 3.698423 |

같은 실행의 action-freeze 기준 reference는 Llama W0 `2/10,3/20,84/100`, Native `10/10,20/20,79/100`; Qwen W0 `0/10,4/20,80/100`, Native `10/10,20/20,80/100`이다. Full-NoSoft 네 셀은 Native Eff/Gen count를 맞췄다. Soft는 Llama Eff만 유지했고, 모든 모델/allocation에서 Gen count가 Native보다 낮았다.

## Stepwise online efficacy와 first-hit

아래는 k=1..8의 online rewrite success count다. first-hit은 관측만 했고 rollout을 멈추지 않았다.

| Cell | k1..k8 success | 최초 10/10 |
|---|---|---|
| L RS NoSoft | 4, 5, 10, 10, 10, 10, 10, 10 | k3 |
| L RS Soft | 3, 4, 4, 5, 6, 9, 10, 10 | k7 |
| Q RS NoSoft | 4, 8, 9, 9, 10, 10, 10, 10 | k5 |
| Q RS Soft | 3, 4, 5, 5, 6, 8, 8, 9 | 없음 |
| L BG NoSoft | 3, 6, 9, 10, 10, 10, 10, 10 | k4 |
| L BG Soft | 2, 4, 4, 4, 7, 9, 10, 10 | k7 |
| Q BG NoSoft | 3, 6, 8, 9, 9, 10, 10, 10 | k6 |
| Q BG Soft | 3, 4, 4, 4, 6, 6, 8, 8 | 없음 |

## Stepwise routing receipts

각 배열은 k=0..7 source field 순서다. `r_E`는 nominal edit retention, `R_PH`는 normalized P/H risk score, `dist`는 nominal `u`로부터의 normalized distance, `cap`은 누적 BF16 capacity다.

### Llama RS

- NoSoft pmax: `[1.897011, 3.218748, 1.872205, 1.114824, 0.630453, 0.219652, 0.087795, 0.160623]`
- NoSoft r_E: `[1, 1, 1, 1, 1, 1, 1, 1]`
- NoSoft R_PH: `[1, 1.881434, 1.992710, 2.791261, 3.649798, 4.021539, 4.671866, 5.409198]`
- NoSoft dist: `[0, 0, 0, 0, 0, 0, 0, 0]`
- NoSoft cap: `[0.031333, 0.174525, 0.390759, 0.690064, 1.101736, 1.590313, 2.132643, 2.779155]`; fallback 0
- Soft pmax: `[1.897011, 3.334064, 4.228487, 4.400607, 4.213878, 3.992441, 2.757362, 2.393387]`
- Soft r_E: `[0.900001, 0.344663, 0.346949, 0.259989, 0.174911, 0.178271, 0.097911, 0.101464]`
- Soft R_PH: `[0.856695, 0.339671, 0.359399, 0.284639, 0.204029, 0.206986, 0.120994, 0.125299]`
- Soft dist: `[0.041470, 0.492471, 0.433025, 0.546499, 0.682426, 0.676652, 0.815853, 0.808212]`
- Soft cap: `[0.024069, 0.059297, 0.114214, 0.177976, 0.229441, 0.295874, 0.336727, 0.386590]`; fallback 5

### Qwen RS

- NoSoft pmax: `[3.912699, 3.741746, 2.429128, 1.098132, 0.924345, 0.196816, 0.235874, 0.086109]`
- NoSoft r_E: `[1, 1, 1, 1, 1, 1, 1, 1]`
- NoSoft R_PH: `[1, 1.806293, 1.643126, 2.479216, 3.331755, 3.356859, 4.566262, 4.732426]`
- NoSoft dist: `[0, 0, 0, 0, 0, 0, 0, 0]`
- NoSoft cap: `[0.260530, 1.179599, 2.835644, 4.727869, 7.716200, 10.764086, 14.727965, 18.992679]`; fallback 0
- Soft pmax: `[3.912699, 3.846120, 4.392470, 5.415081, 5.617457, 5.238610, 4.945690, 4.434804]`
- Soft r_E: `[0.900001, 0.190104, 0.251706, 0.175215, 0.144855, 0.119086, 0.101767, 0.073837]`
- Soft R_PH: `[0.885583, 0.180116, 0.230294, 0.159695, 0.143741, 0.120926, 0.105317, 0.078672]`
- Soft dist: `[0.020516, 0.722197, 0.620777, 0.763247, 0.769133, 0.827297, 0.847562, 0.887031]`
- Soft cap: `[0.231860, 0.314172, 0.491584, 0.636047, 0.798291, 0.953165, 1.109977, 1.236405]`; fallback 6

### Llama BG

- NoSoft pmax: `[2.008990, 3.024007, 1.963386, 1.050470, 0.681746, 0.189025, 0.401682, 0.039930]`
- NoSoft r_E: `[1, 1, 1, 1, 1, 1, 1, 1]`
- NoSoft R_PH: `[1, 1.870700, 2.001684, 2.751366, 3.288197, 4.399609, 4.791844, 5.239228]`
- NoSoft dist: `[0, 0, 0, 0, 0, 0, 0, 0]`
- NoSoft cap: `[0.032566, 0.175570, 0.405756, 0.706462, 1.065290, 1.464829, 1.937171, 2.404178]`; fallback 0
- Soft pmax: `[2.008990, 3.177161, 4.434372, 4.714266, 4.397920, 4.016367, 3.100357, 2.755596]`
- Soft r_E: `[0.825926, 0.374405, 0.322174, 0.261854, 0.144570, 0.125979, 0.096130, 0.150508]`
- Soft R_PH: `[0.722494, 0.359013, 0.342434, 0.287392, 0.172987, 0.152836, 0.119032, 0.178796]`
- Soft dist: `[0.110846, 0.435164, 0.464167, 0.545039, 0.731764, 0.763254, 0.819690, 0.719549]`
- Soft cap: `[0.025409, 0.064636, 0.118864, 0.185802, 0.227457, 0.269885, 0.304356, 0.371102]`; fallback 7

### Qwen BG

- NoSoft pmax: `[3.587398, 3.907598, 2.456162, 1.048240, 1.733600, 0.832023, 0.284266, 0.156889]`
- NoSoft r_E: `[1, 1, 1, 1, 1, 1, 1, 1]`
- NoSoft R_PH: `[1, 1.822257, 2.153013, 2.849815, 3.418160, 3.817100, 4.185595, 3.794060]`
- NoSoft dist: `[0, 0, 0, 0, 0, 0, 0, 0]`
- NoSoft cap: `[0.294543, 1.274755, 2.794553, 4.642024, 6.935919, 9.521461, 12.533262, 15.305625]`; fallback 0
- Soft pmax: `[3.587398, 3.905555, 4.747645, 5.568077, 5.622329, 4.757655, 4.091776, 3.559696]`
- Soft r_E: `[0.561979, 0.335881, 0.185467, 0.174620, 0.125707, 0.094069, 0.122987, 0.052344]`
- Soft R_PH: `[0.476389, 0.231030, 0.173935, 0.167562, 0.127308, 0.097516, 0.124869, 0.056018]`
- Soft dist: `[0.523888, 0.542483, 0.707442, 0.728223, 0.789716, 0.852830, 0.803597, 0.932554]`
- Soft cap: `[0.126474, 0.237699, 0.358833, 0.512958, 0.647188, 0.761278, 0.938875, 1.017215]`; fallback 5

## Routing 자유도, P/H, capacity

| Cell | Endpoint layer shares L4..L8 | Neff | HHI | P mean / raw max | H |
|---|---|---:|---:|---|---|
| L RS NoSoft | .2,.2,.2,.2,.2 | 5.000 | .200 | .009552 / .025568 | 0 inactive |
| L RS Soft | .175643,.244304,.227379,.187547,.165127 | 4.886 | .205 | .001168 / .005415 | 0 inactive |
| Q RS NoSoft | .2,.2,.2,.2,.2 | 5.000 | .200 | .000882 / .001962 | 0 inactive |
| Q RS Soft | .114002,.099304,.262399,.315160,.209134 | 4.259 | .235 | .000455 / .001031 | 0 inactive |
| L BG NoSoft | .2,.2,.2,.2,.2 | 5.000 | .200 | .005233 / .015475 | 0 inactive |
| L BG Soft | .206459,.222540,.207616,.193174,.170211 | 4.962 | .202 | .001112 / .002776 | 0 inactive |
| Q BG NoSoft | .25,.25,.25,.25,0 | 4.000 | .250 | .000912 / .002667 | 0 inactive |
| Q BG Soft | .084553,≈0,.246613,.297255,.371578 | 3.397 | .294 | .000575 / .001725 | 0 inactive |

모든 셀에서 severe-concentration flag는 false였다. Soft는 P score와 capacity를 낮췄지만 동시에 nominal edit retention을 크게 낮췄다. P/H는 관측 및 Soft 목적에 쓰였을 뿐 hard veto/threshold가 아니며, H는 atomic B10에서 비활성이다.

## Physical realization 관측

아래는 말단 source field의 평균 predicted progress, actual weight-only progress, ρ, request-wise cosine 평균, norm gain 평균, residual ratio 평균이다. 이 telemetry는 관측 전용이고 transition accept/reject에 사용되지 않았다.

| Cell | Pred. | Actual | ρ | Cosine | Norm gain | Residual ratio |
|---|---:|---:|---:|---:|---:|---:|
| L RS NoSoft | .160623 | .003129 | .019481 | .028454 | .526078 | .797592 |
| L RS Soft | .242844 | .003283 | .013521 | .077383 | .126159 | .622490 |
| Q RS NoSoft | .086109 | -.055203 | -.641083 | .028214 | .618099 | .875232 |
| Q RS Soft | .327453 | .002962 | .009045 | .034393 | .078159 | .625754 |
| L BG NoSoft | .039930 | .036073 | .903400 | .191423 | .428251 | .684826 |
| L BG Soft | .414740 | .086346 | .208194 | .117147 | 23.245477 | 23.193218 |
| Q BG NoSoft | .156889 | -.154229 | -.983043 | .095764 | 10.721658 | 10.680564 |
| Q BG Soft | .186330 | -.217335 | -1.166398 | .080960 | .504256 | .834370 |

Llama BG Soft와 Qwen BG NoSoft의 큰 norm/residual ratio는 일부 intended displacement가 거의 0인 행의 분모 민감도로 인해 불안정하다. 부호가 음수인 actual progress도 포함해 그대로 보고하며, 이를 성공 방향으로 재분류하지 않는다.

## 대비

### P1R17 내부 Soft − NoSoft count 차이

| 모델/allocation | ΔEff | ΔGen | ΔLoc | 해석 경계 |
|---|---:|---:|---:|---|
| Llama RS | 0 | -3 | +1 | weaker edit; preservation 주장 불가 |
| Llama BG | 0 | -2 | +4 | weaker edit; preservation 주장 불가 |
| Qwen RS | -1 | -8 | +3 | markedly weaker edit |
| Qwen BG | -2 | -7 | -1 | weaker edit, locality도 개선 없음 |

RS→BG allocation 효과는 NoSoft에서 Llama count가 동일하되 Loc `81→79`, Qwen은 Loc `78→81`이었다. Soft에서는 Llama Gen `17→18`, Loc `82→83`; Qwen Eff `9→8`, Gen `12→13`, Loc `81→80`으로 작고 혼합된 변화였다. allocation 효과보다 penalty 기반 Soft strength attenuation이 더 큰 공통 패턴이다.

### 동일 R13 Dynamic cell 대비 P1R17 count 변화

R13 matched Dynamic facts의 immutable report SHA-256은 `421fdedb03adb28aa056f01bb411f2337ca79c99f03628c1690345110d949c05`다.

| Cell | R13 E/G/L | P1R17 E/G/L | ΔE/ΔG/ΔL | target-new NLL Δ E/G/L |
|---|---|---|---|---|
| L RS NoSoft vs R13 RS-N | 9/17/82 | 10/20/81 | +1/+3/-1 | -.666311 / -1.636577 / -.061750 |
| L RS Soft vs R13 RS-S | 10/17/82 | 10/17/82 | 0/0/0 | +.259683 / +.474014 / -.011875 |
| Q RS NoSoft vs R13 RS-N | 5/10/80 | 10/20/78 | +5/+10/-2 | -5.050473 / -4.185297 / -.014531 |
| Q RS Soft vs R13 RS-S | 4/7/81 | 9/12/81 | +5/+5/0 | -3.779297 / -3.375879 / -.026563 |
| L BG NoSoft vs R13 BG-N | 8/15/83 | 10/20/79 | +2/+5/-4 | -1.896486 / -2.550959 / -.160626 |
| L BG Soft vs R13 BG-S | 10/18/83 | 10/18/83 | 0/0/0 | +.145019 / +.092408 / -.005625 |
| Q BG NoSoft vs R13 BG-N | 9/17/81 | 10/20/81 | +1/+3/0 | -1.588049 / -1.287201 / -.006875 |
| Q BG Soft vs R13 BG-S | 8/14/80 | 8/13/80 | 0/-1/0 | +.781836 / +.382568 / +.001562 |

FACT 수준에서 Full-NoSoft는 R13 Neutral 대비 모든 셀의 Eff/Gen을 개선했다. Full-Soft는 Qwen RS에서 R13 Soft보다 개선됐지만 Qwen BG에서는 Gen이 1 낮았고, Llama Soft는 count가 동일했다. 서로 다른 routing objective와 realized strength이므로 이 표는 matched-seal descriptive contrast이지 독립적인 promotion proof가 아니다.

### P1R18 AlphaEdit/MEMIT one-shot joint B10 baseline

P1R18의 업데이트된 immutable report만 사용한다: SHA-256 `768bfd29a326d503f629513d707a8560c0dbd002b4a26237520b8a27cdfe4ac6`, 13,476 bytes. 동일 seal/order와 frozen evaluator를 사용했다.

| 모델 | W0 E/G/L | AlphaEdit E/G/L | Alpha edit wall | MEMIT E/G/L | MEMIT edit wall |
|---|---|---|---:|---|---:|
| Llama | 2/10, 3/20, 84/100 | 10/10, 20/20, 79/100 | 51.90s | 10/10, 20/20, 82/100 | 68.85s |
| Qwen | 0/10, 4/20, 80/100 | 10/10, 20/20, 80/100 | 49.88s | 10/10, 20/20, 80/100 | 88.63s |

P1R17 Full-NoSoft도 양 모델에서 Eff/Gen count를 맞췄지만 K8 end-to-end 실험이고, P1R18은 표준 one-shot baseline이다. wall time의 범위와 포함 비용이 다르므로 직접적인 순수-edit 배수로 환산하지 않는다. P1R18은 내부 forward/backward/token count를 기록하지 않았고, AlphaEdit은 caller-scoped BF16 key→FP32 adapter를 사용했다. P1R18의 continuous NLL은 현재 handoff에 기록되지 않아 count만 기술적으로 연결한다.

## Compute, wall, memory

내부 partitioned counter는 bootstrap/rollout/post-freeze 구간의 합이며, 하나의 authoritative non-overlapping whole-job scalar로 재해석하지 않는다.

| Job/cell | F/B/tokens | Rollout F/B | Functional-basis wall | Post-freeze wall | Scheduler wall | MaxRSS KiB |
|---|---|---|---:|---:|---:|---:|
| 18310 L RS NoSoft | 4551/90/147767 | 4102/80 | 86.70s | 139.34s | 6959s | 8419112 |
| 18311 L RS Soft | 4551/90/147767 | 4102/80 | 100.75s | 139.85s | 7222s | 8413732 |
| 18312 Q RS NoSoft | 4555/90/134487 | 4102/80 | 90.68s | 157.32s | 7908s | 11870908 |
| 18313 Q RS Soft | 4555/90/134487 | 4102/80 | 92.64s | 156.28s | 7988s | 11865452 |
| 18314 L BG NoSoft | 4551/90/147767 | 4102/80 | 82.95s | 139.80s | 7002s | 8474860 |
| 18315 L BG Soft | 4551/90/147767 | 4102/80 | 96.65s | 140.91s | 7274s | 8433464 |
| 18316 Q BG NoSoft | 4555/90/134487 | 4102/80 | 96.55s | 158.76s | 7931s | 11895024 |
| 18317 Q BG Soft | 4555/90/134487 | 4102/80 | 92.89s | 157.27s | 7912s | 11905060 |

CUDA allocated/reserved는 Llama `21,011,203,072 / 22,890,414,080` bytes, Qwen `23,873,667,072 / 26,075,987,968` bytes였다. Soft의 추가 wall은 Llama에서 보였지만 Qwen에서는 셀별 변동 범위 안이었다. runtime bottleneck을 QP/covariance 단일 요소로 귀속하지 않는다.

## Immutable terminal hashes

순서는 terminal / manifest / action-freeze / W0 / panel SHA-256이다.

- 18310: `4e44ffb22f5a8c7d546235e6e5f8f3da7ee40ddcf3f8ba1937413db9f3ce4358` / `281e1a9885d2308db8a7e6bf54468a31c928d5a247f6a6753ac93c229a07e16d` / `a7946811a98d4e759dae1698ac7a03430b80c5c8dbe84137f6767680c36efc14` / `4bfd5cb5dd729f6e4891fc323819d8c0550b8167216e2401e23799652eed83df` / `3e26a56d0afab16702a4793de534711f86e9f5dac9d5ae841c7118da427427da`
- 18311: `f166c292ae13a35a719d70ede090120b88871d1a0691888b088d656868c1e689` / `7763082ab9c14a9b772219a95423502c78e115fa4d03e8bb30df2332047b2161` / `7a3f4678b7c27edf4d612a0fa53f1e956b3f38132f5f5309e847381ded19716f` / `6f046653072d47e37f1ecdb2dce5e76d0d6c2b40c84d59cc075b4666c01b2625` / `5a9ec62273960aee25541f7781beb0fa231d35b1c686a89ab173d2f797af97f4`
- 18312: `412ecff5ddb89b5e27a4a0312b2c9109f3c1a8f1e9e356fcbfa405c516c3c31a` / `bd4b01533391077c2bccfa0749f2aab7f7e92d592f73ce59ac2f46e23681c774` / `d24316f92cc3752b2df06e9ba092f42ef8c0710c319749cae013ac4427ac4943` / `644d132fdcb673bb0604412005cdb3bc872d9a60b44bed2a70cc7125ecea0a46` / `228fda38399384618e0ef12f5f487186ee533f18004a3b893221ec1a40702f6a`
- 18313: `670fba0816e6897cf2957d783187258b7ca8cebe0182b5cf2ebc2a43dc5685ef` / `f2b98588fd32245c64f19560f5afa13090f1b3dd1104fff855d41439f84445b5` / `880c60b7b4b2b0de5e0a93fa7366da63582f36d4370d49031627bc5a53c598ab` / `118cbc01731d71fc770f3b7c1a014c679657094cf685d4146dc962d638516adf` / `1d28ea5a8f05be0764a3d9ecd280f4d7e0365e9c65880ff744a62a89bfa508d9`
- 18314: `78dfee94c32cbbd98e4054cd259b5bc85243ef3579169cc7899304b31bc6a2eb` / `eb6bd964ae5901bd42ec54ed276a70b880b7c218083c17d90fded1316269f92b` / `0822b77053bcc29fef23db597183e0a173b9c5ad0fb23d2ae1397ef7dd44e6f4` / `4d94f3a17fb1c960448f71d0039cddb8ae117c5b88b6d919b8c96f2e98367351` / `8ad7bad5c130d58db51c4f42ea36bccf470e6ad33673fb4ec0b81c4ba9257aad`
- 18315: `7225e6ae0b915d193b2319e53f7065fb49a59103fc75ee85e58d2675fc27575e` / `f0f932168f08ae97a433364b772b4e75409b9012b74fc592abb39d39fead6ef9` / `9f3922e4ace3324f8dfc46b9d00bf408202e32cd66b4ed2fb627cb95c0ced408` / `04ede4f894f175d1a1a81587f01a7984815425bc87921c7ad63a83552bfc7bcd` / `8e704c856506851e1e27691958adee340ae32d0fd441005144cc8bd068d60b89`
- 18316: `f3a9cf4f4a2b2637ca4a1331ef45eaab7de3a9bf5e41ab02db1b0274e525d234` / `b4ad3d985ad4b8cab61cf932da784da91136d59f375020c57e576e2cab61d615` / `a07f4395398dc789b99ff860c92c360f13547acbac6c78b412e3499cc6948546` / `afb7c7650fe6813377c891f939703dfdf7e511b5808957dbbe9d3410dfa90506` / `d520a93469871615759d41088b2207e0471446220f002fab40f6b6fc2b992ff0`
- 18317: `278d2fcd04884e8cdcf5d5bf0b322b9d92ad6d28b68713991d95bd0910c8d38e` / `f7a9c44347a0a8195c0aebc1a9c63b35cd92ee652f075c1a9ca5517dc9e4c662` / `171e04711646538ae72f2fbb5d01677b638a6cf16179cec33f8c20805bd8a33a` / `bbeacd545c7f1936d5b153b1a6bd35c07059784eec2a3555c3c8a2200c5bdedf` / `9bccb6e7abd59090f363be30fcc2a42897c177e7151b1072d3f26c6549074422`

## FACT / INFERENCE / NOT_RECORDED

### FACT

- 8/8 셀은 K8/τ1 및 terminal/W0/manifest를 완료했다.
- Full-NoSoft는 4/4 셀에서 W-only Eff/Gen `10/10,20/20`이다.
- Full-Soft는 모든 셀에서 NoSoft보다 terminal `r_E`가 낮고 nominal distance가 크다.
- Soft의 P와 capacity는 대체로 낮지만 semantic edit strength도 약하다.
- 23회의 알려진 Stage3 numerical fallback은 모두 certified trust-constr로 종료됐고 과학적 retry는 없었다.
- P/H/realization telemetry는 hard veto 또는 rollback을 일으키지 않았다.

### INFERENCE

- P1R17의 penalty-based Soft는 layer allocation을 바꾸고 P/capacity를 줄이지만, 이 panel에서는 그 대가로 특히 Qwen의 semantic edit strength와 Gen을 크게 잃었다.
- Full-NoSoft의 R13 대비 개선은 overlay/soft preservation 제약 없이 physical W-only endpoint drive를 유지한 효과와 일관된다.
- Llama Soft가 Eff count를 유지하더라도 continuous new-NLL과 Gen 악화가 있어 Native-strength를 맞췄다고 볼 수 없다.
- allocation RS/BG 효과는 모델·routing에 따라 혼합돼 있으며, Soft strength 손실보다 작다.

### NOT_RECORDED

- P1R17의 endpoint z-oracle 평가는 기록되지 않았다.
- 하나의 authoritative non-overlapping whole-job internal forward/backward/token scalar는 기록되지 않았다. 위 수치는 partitioned ledger 합계다.
- rollout phase의 별도 authoritative peak-memory 값은 기록되지 않았고 scheduler MaxRSS/CUDA receipts만 있다.
- P1R18 handoff에는 continuous NLL과 내부 forward/backward/token count가 없다.
- fresh-seal replication, 통계적 불확실성, scientific promotion 근거는 기록되지 않았다.

## 향후 P1R19/P1R18 최종 비교에 고정할 해석 규칙

P1R19는 SH2 소유의 독립 실험이며 이 보고서 작성 과정에서 구현·실행·모니터링하지 않았다. immutable P1R19 report가 도착한 뒤에만 별도 최종 한국어 비교 보고서를 만든다. 그 보고서에는 다음 규칙을 적용한다.

- P1R19는 Dynamic only이며 Hold가 없다.
- P1R19 router는 추가 model forward/backward를 만들지 않는다. `alpha_req`는 기존 target gradient, `a_l`은 기존 `nohook_signed_progress`를 재사용한다.
- Structural-P는 pinned EasyEdit Wikipedia key-second-moment proxy이고 Soft allocation에만 사용하며 threshold/veto가 아니다.
- Loc/P/H 해석 전에 matched `a^T v`와 coverage를 먼저 비교한다. actual semantic edit strength가 맞지 않으면 preservation gain이라 부르지 않는다.
- 시간은 `edit_core_time`, `functional_probe_time`, `stepwise_eval_time`, `artifact/model_load_time`, `total_wall_time`으로 분리한다.
- Native overhead는 edit-core와 end-to-end로 나누며 R13 receipts에서 정확한 pure-edit multiplier를 추론하지 않는다.
- runtime 병목을 QP/covariance로 단정하지 않고 K8 field refresh, functional probes, stepwise evaluation을 구분한다.
- 최종 비교는 R17 penalty-based Soft strength loss, R19 strength-preserving allocation effect, P1R18 AlphaEdit/MEMIT descriptive baseline을 분리한다.

## 결론

P1R17은 Full-NoSoft physical drive가 동일 reused seal에서 양 모델 모두 Native 수준 Eff/Gen count에 도달할 수 있음을 보였다. 반면 현재 penalty-based Full-Soft는 특히 Qwen에서 edit retention과 Gen을 크게 낮췄다. 낮은 P/capacity 또는 일부 높은 Loc는 strength-matched preservation evidence가 아니다. 본 결과는 다음 routing 설계의 mechanistic 근거이지만 outcome-selected seal에 한정되며 scientific promotion을 허용하지 않는다.
