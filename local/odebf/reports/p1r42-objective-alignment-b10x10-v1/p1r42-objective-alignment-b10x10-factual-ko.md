# P1R42 Move/Accept/Hold Objective Alignment B10×10 사실 보고서

- instruction_id: `ODEEDIT-S05-P1R42-P1R39-MOVE-ACCEPT-HOLD-OBJECTIVE-ALIGNMENT-B10X10-V1`
- P1R42 source HEAD/tree: `ca68a4f459fd7303a4d5abbde2e1bf7aee0d805c` / `4af77417a52c70b99784d64dbfcabba0a91a1960`
- P1R39 parent HEAD/tree: `763457560f2efb177a56310dfd87526772cf8158` / `1f58423b53dfb7e84c8011ab08c2bf7a5ad4ca25`
- contract SHA256: `5ae7074d2280d8975a5735ceca108377d4f270292a293232f7a690efe25344b1`
- numerical lock SHA256/root: `b5212808d6493afb4b5c2d2bbf2508d7253fd29738882f6483c9a1381d40dc0c` / `976ec4039d815591ea312d3c3907920afa5465d85f0b907737baf345319118f8`
- source manifest SHA256/root: `ad0e492473d87f54b76375e0e532d13335a28514c679d298ff9e8ef0bf6c053f` / `e6ea592263672b377f6279d3659025ccfac4816dde9a253583eca2693ee42c3a`
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- 실행 범위: Llama/Qwen × Neutral/Soft, 각 10개 독립 B10 case, K=8, terminal evaluator 1회/case
- scientific_promotion: `false`

## 1. 실행 및 무결성

- Slurm production job: `19795`; array task 4/4 `COMPLETED`, ExitCode `0:0`.
- B1 job/technical gate: `19791` / `PASS` (4/4 cells, K8); receipt SHA/root: `8db68c1fa73e6040f84a2986ec054ad7b35e09a09fc81c6d727f7c0985486afb` / `8589684d9c2f325cd895683f22ce073d1365ea6a780f6d1c0f69fbfd76504bef`.
- attempts/endpoints/failures: `40/40/0`.
- accepted transitions/materializations: `320/320`.
- action-freeze hash matches: `40/40`; terminal hash matches: `40/40`; case manifest hash matches: `40/40`.
- W0 pointer/byte restore: `40/40` / `40/40`.
- retry/backtracking 합산 / inner heldout / history influence: `0/0/0`.
- result integrity file count/bytes/root: `768` / `9374882` / `627c5248db104cf92d8282af4f6d1bff7bd673eb12275cdab359343b4fb51fbb`.
- transport reconciliation SHA/root: `f7665744a019deb5fa64c43a1bf11cf903857f0b8711683ae574769a8bc1214d` / `f991b41083642298eeccccf06c5a4435c65397acb5e4b802b9fdfce63f96bcd3`; authoritative transferred member SHA/root: `389263abfacee87845cbd62c1fa34ca5f73b0b9ddfb3fea28b106ba6f4cd02fd` / `8d7e3116cf7446777ecbf92fe10d1eb6f570b8bc84c71823d41edfd448df2107`.

## 2. P1R42 terminal endpoint

| Model | Arm | Eff | Gen | Loc | Eff new NLL | Eff margin | Gen new NLL | Gen margin | Loc new NLL | Loc margin | z8 full6 | W8 full6 | z−W gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 100/100 | 179/200 | 875/1000 | 0.151236 | 11.104077 | 2.317305 | 5.848217 | 10.357687 | 5.557493 | 0.088340 | 0.156630 | 0.068290 |
| llama3-8b-inst | Soft | 100/100 | 180/200 | 873/1000 | 0.114023 | 11.231446 | 2.343338 | 5.754834 | 10.372960 | 5.574281 | 0.087947 | 0.134031 | 0.046084 |
| qwen2.5-7b-inst | Neutral | 96/100 | 161/200 | 844/1000 | 0.752705 | 10.422179 | 4.050829 | 3.719669 | 10.121647 | 4.879018 | 0.583211 | 0.632283 | 0.049072 |
| qwen2.5-7b-inst | Soft | 95/100 | 155/200 | 846/1000 | 0.811343 | 10.586488 | 4.085420 | 3.820024 | 10.127878 | 4.894415 | 0.655734 | 0.699345 | 0.043611 |

## 3. 동일 case P1R39 및 Official AlphaEdit 대비

P1R39은 동일 model·arm·case 키로 40/40 조인했다. Official 값은 P1R39 동일-stream receipt에서 model·case별로 10개씩 한 번만 집계했다.

| Model | Arm | ΔEff vs R39 | ΔGen vs R39 | ΔLoc vs R39 | ΔEff NLL vs R39 | Δz8 vs R39 | ΔW8 vs R39 | ΔP vs R39 | Δcapacity vs R39 | Δenergy vs R39 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | +0 | +0 | +0 | 0.003211 | -0.003942 | 0.012419 | 0.000830 | 0.194308 | 0.268791 |
| llama3-8b-inst | Soft | +0 | -1 | -2 | -0.011598 | 0.005766 | 0.005171 | 0.000031 | 0.000223 | 0.000517 |
| qwen2.5-7b-inst | Neutral | +0 | +6 | +0 | -0.076500 | -0.097300 | -0.101874 | -0.023065 | -0.643812 | -0.556769 |
| qwen2.5-7b-inst | Soft | -1 | -1 | +1 | 0.040913 | 0.038921 | 0.031042 | 0.000476 | 0.021059 | 0.023972 |

| Model | Arm | P1R42 Eff/Gen/Loc | Official Eff/Gen/Loc | Δ counts (P1R42−Official) | P1R42 Eff NLL | Official Eff NLL | Δ Eff NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 100/100/179/200/875/1000 | 100/100/185/200/859/1000 | +0/-6/+16 | 0.151236 | 0.001169 | 0.150067 |
| llama3-8b-inst | Soft | 100/100/180/200/873/1000 | 100/100/185/200/859/1000 | +0/-5/+14 | 0.114023 | 0.001169 | 0.112854 |
| qwen2.5-7b-inst | Neutral | 96/100/161/200/844/1000 | 100/100/192/200/828/1000 | -4/-31/+16 | 0.752705 | 0.032646 | 0.720059 |
| qwen2.5-7b-inst | Soft | 95/100/155/200/846/1000 | 100/100/192/200/828/1000 | -5/-37/+18 | 0.811343 | 0.032646 | 0.778697 |

Official continuous Gen/Loc NLL·margin과 Official z/W/P/capacity/energy는 전달된 동일-stream per-case 표에 `NOT_RECORDED`이다.

P1R39 parent raw endpoint:

| Model | Arm | Eff | Gen | Loc | Eff new NLL | z8 full6 | W8 full6 | z−W gap | P | capacity | energy | edit-core Σ(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 100/100 | 179/200 | 875/1000 | 0.148025 | 0.092282 | 0.144211 | 0.051929 | 0.006599 | 0.054000 | 0.087323 | 1025.133 |
| llama3-8b-inst | Soft | 100/100 | 181/200 | 875/1000 | 0.125622 | 0.082181 | 0.128860 | 0.046678 | 0.006274 | 0.035811 | 0.058079 | 1039.322 |
| qwen2.5-7b-inst | Neutral | 96/100 | 155/200 | 844/1000 | 0.829205 | 0.680512 | 0.734158 | 0.053646 | 0.213970 | 1.026097 | 0.862239 | 1437.037 |
| qwen2.5-7b-inst | Soft | 96/100 | 156/200 | 845/1000 | 0.770430 | 0.616813 | 0.668303 | 0.051490 | 0.121404 | 0.121591 | 0.109191 | 1359.389 |

## 4. Terminal target request 분포

각 값은 terminal z8-oracle per-request receipt에서 계산했다. median/p90은 P1R39 보고 규약과 동일하게 10개 case별 통계의 평균이다.

| Model | Arm | Run | mean | case-median mean | case-p90 mean | worst max | count NLL<.05 /100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | P1R42 | 0.088340 | 0.024176 | 0.132332 | 2.089943 | 87/100 |
| llama3-8b-inst | Neutral | P1R39 | 0.092282 | 0.006208 | 0.141485 | 3.707090 | 89/100 |
| llama3-8b-inst | Soft | P1R42 | 0.087947 | 0.023746 | 0.125157 | 2.003942 | 84/100 |
| llama3-8b-inst | Soft | P1R39 | 0.082181 | 0.006043 | 0.126323 | 2.588469 | 85/100 |
| qwen2.5-7b-inst | Neutral | P1R42 | 0.583211 | 0.037201 | 1.662040 | 8.164314 | 71/100 |
| qwen2.5-7b-inst | Neutral | P1R39 | 0.680512 | 0.026968 | 1.896820 | 8.140265 | 69/100 |
| qwen2.5-7b-inst | Soft | P1R42 | 0.655734 | 0.035792 | 1.857341 | 8.321791 | 73/100 |
| qwen2.5-7b-inst | Soft | P1R39 | 0.616813 | 0.027330 | 1.732374 | 8.319790 | 72/100 |

## 5. Move/Accept/Hold 및 projection telemetry

| Model | Arm | held req-step | first-hit req | reactivated | rejects | late k6–8 rejects | counterfactual old hold | old-active & V<.05 | raw conflict>0 | removed>0 | safe inner max | zero semantic grad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 220 | 83 | 2 | 34 | 25 | 33 | 187 | 604 | 604 | 2.462e-12 | 0 |
| llama3-8b-inst | Soft | 215 | 80 | 0 | 38 | 28 | 35 | 180 | 609 | 609 | 9.294e-12 | 0 |
| qwen2.5-7b-inst | Neutral | 214 | 69 | 2 | 85 | 60 | 103 | 111 | 474 | 474 | 5.738e-12 | 0 |
| qwen2.5-7b-inst | Soft | 214 | 67 | 1 | 84 | 61 | 96 | 118 | 445 | 445 | 3.058e-11 | 0 |

- target direction policy receipt: `SEMANTIC_PRIMARY_PROJECTED_PRESERVATION_FIXED_NORMALIZED` on 320/320 accepted transitions.
- hold policy receipt: `CURRENT_STATE_SEMANTIC_V_INSTANTANEOUS_NO_CARRY` on 320/320 transitions.
- target trial count: 320; added selection forward/backward/materialization: 0/0/0.
- gamma/trust/Adam/debt access counts: 0.

Held writer receipt:

| Model | Arm | held req-step | held residual recorded | held actual Σ | held actual mean | positive/negative/zero |
|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 220 | 220 | 5.483060 | 0.024923 | 200/20/0 |
| llama3-8b-inst | Soft | 215 | 215 | 6.469406 | 0.030090 | 200/15/0 |
| qwen2.5-7b-inst | Neutral | 214 | 214 | 7.227873 | 0.033775 | 194/20/0 |
| qwen2.5-7b-inst | Soft | 214 | 214 | 4.193466 | 0.019596 | 190/24/0 |

## 6. Writer, preservation, routing

| Model | Arm | predicted Σ | actual Σ | actual/predicted | mean step realization | negative steps | P endpoint mean | functional-P mean | capacity endpoint mean | energy endpoint mean | entropy | top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 131.562011 | 102.318664 | 0.777722 | 0.714167 | 0 | 0.007429 | 0.016006 | 0.248308 | 0.356115 | 1.443894 | 0.360118 |
| llama3-8b-inst | Soft | 130.720462 | 102.544655 | 0.784458 | 0.735578 | 0 | 0.006305 | 0.015361 | 0.036034 | 0.058596 | 1.058951 | 0.567487 |
| qwen2.5-7b-inst | Neutral | 132.694578 | 98.221955 | 0.740211 | 0.706498 | 0 | 0.190905 | 0.003267 | 0.382285 | 0.305470 | 1.388450 | 0.394742 |
| qwen2.5-7b-inst | Soft | 130.080819 | 97.551340 | 0.749929 | 0.734525 | 0 | 0.121880 | 0.002975 | 0.142650 | 0.133163 | 0.954850 | 0.577031 |

Neutral−Soft 동일-model 산술 차이:

| Model | Soft−Neutral Eff/Gen/Loc | Eff NLL | z8 | W8 | z−W gap | P | functional-P | capacity | energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | +0/+1/-2 | -0.037212 | -0.000393 | -0.022599 | -0.022206 | -0.001124 | -0.000645 | -0.212274 | -0.297519 |
| qwen2.5-7b-inst | -1/-6/+2 | 0.058638 | 0.072523 | 0.067062 | -0.005461 | -0.069025 | -0.000291 | -0.239635 | -0.172307 |

## 7. Compute 및 wall time

| Model | Arm | edit-core Σ / mean (s) | evaluator Σ / mean (s) | outer wall / scheduler elapsed (s) | F/B | target B / slope B | tokens / padded | materializations | peak GPU/host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 766.080 / 76.608 | 152.157 / 15.216 | 1059.267 / 1097 | 2200/1250 | 850/400 | 371778/412504 | 80 | NOT_RECORDED/NOT_RECORDED |
| llama3-8b-inst | Soft | 737.074 / 73.707 | 163.659 / 16.366 | 1049.485 / 1089 | 2200/1250 | 850/400 | 371778/412504 | 80 | NOT_RECORDED/NOT_RECORDED |
| qwen2.5-7b-inst | Neutral | 941.293 / 94.129 | 192.442 / 19.244 | 1298.703 / 1342 | 2200/1250 | 850/400 | 340727/412904 | 80 | NOT_RECORDED/NOT_RECORDED |
| qwen2.5-7b-inst | Soft | 883.583 / 88.358 | 184.531 / 18.453 | 1237.517 / 1285 | 2200/1250 | 850/400 | 340727/412904 | 80 | NOT_RECORDED/NOT_RECORDED |

P1R42 phase wall 합계는 `analysis-summary.json`의 `phase_wall_sums`에 기록했다. scheduler terminal MaxRSS와 result 내 peak GPU/host memory는 `NOT_RECORDED`이다.

P1R39 edit-core 대비 P1R42 산술 비율:

| Model | Arm | P1R42 edit-core Σ(s) | P1R39 edit-core Σ(s) | ratio | Δseconds |
|---|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 766.080 | 1025.133 | 0.747299 | -259.053 |
| llama3-8b-inst | Soft | 737.074 | 1039.322 | 0.709187 | -302.248 |
| qwen2.5-7b-inst | Neutral | 941.293 | 1437.037 | 0.655024 | -495.743 |
| qwen2.5-7b-inst | Soft | 883.583 | 1359.389 | 0.649986 | -475.805 |

## 8. Typed outcomes 및 기록 경계

- technical failures: 0/40 production cases.
- scientific typed failures: 0/40 production cases.
- incomplete prefixes/imputation: 0/0.
- W0 restore failures: 0/40.
- source/model/GPU/Slurm mutation after terminal analysis start: 0.
- P1R39 comparison fields absent from its raw-free handoff, Official continuous Gen/Loc fields, and peak memory are `NOT_RECORDED`.
- 이 보고서는 실행 identity, 수치, 산술 delta, contract-defined mechanical gate만 기록한다. scientific_promotion=`false`.

## 9. 산출물

- `p1r42-objective-alignment-per-case.json/.csv`: 40 rows.
- `p1r42-objective-alignment-per-step.json/.csv`: 320 rows.
- `p1r42-objective-alignment-per-request.json/.csv`: 3,200 rows.
- `p1r42-objective-alignment-paired-comparisons.json/.csv`: 40 rows.
- `analysis-summary.json`, `result-integrity.json`, `analysis-manifest.json`, `analysis-receipt.json`.
