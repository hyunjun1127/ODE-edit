# P1R27 Joint W-CLF / z-tracking / semantic-first BF — Atomic B10 종합 보고서

작성 시각: 2026-08-13 (Asia/Seoul)  
instruction: `ODEEDIT-S05-P1R27-JOINT-WCLF-ZTRACKING-SEMANTIC-PRESERVING-BF-V1`  
scientific checkpoint: `56b744449a5bdd44b9e0798ae3fa486962dbdc22`  
execution TECH-R1: `c78e7e63f54403b52be3dc619555b2844225852e`  
source tree: `07c2adfb9f3cc2323a6b8c665bd2ff238635fd51`  
seal/order: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628` / `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`

## 한 줄 결론

P1R27은 기술 계약과 semantic-first 우선순위는 지켰지만, **Atomic strength는 회복하지 못했다.** 첫 1회(오직 Llama-RS만 2회)의 non-stall write 뒤 대부분 상태가 `WRITER_CLF_SEMANTIC_STALL`로 totalize되어, 64 accepted grid transition 중 실질 write는 10회(`JOINT_WRITE` 9 + `SOFT_NO_ROUTING_DOF` 1)뿐이었다. 최종 Eff/Gen은 Llama `8/14–15`, Qwen `6–7/14–15`로 Official AlphaEdit와 P1R24보다 낮다. Soft는 Neutral과 예측 semantic strength를 맞췄지만 보존 신호는 혼재하며, Historical 진입은 `HOLD`다.

## 1. FACT — 실행과 무결성

- Phase-A focused/impacted gates: 25/25 PASS; compile/bash/dry-plan/source-manifest/session gate PASS.
- numerical lock root: `fd26df65b51faccd6e4cc06280398a9f0f5fc185cd6354c44c1dae7537e5b9c8`.
- source-manifest root: `5d033523af86eef58a344416fb6d64df440cb3c0cf2574af8dd42d529a799506`.
- 최초 B1 `19135`는 ignored local session-boundary 파일 부재로 4/4 pre-model exit4였다. result root/model/scientific/materialization action은 0이었고 실패 로그는 보존됐다.
- TECH-R1은 session binding/provenance만 고쳤다. scientific/numerical bytes는 불변이다.
- B1 replacement `19139`는 4/4 `COMPLETED0`; 8/8 arms K8, materialization8, W0 restore, terminal/manifest rehash PASS였다.
- Production `19145` array `0-3%4`: 4/4 `COMPLETED0`, 8/8 arms K8/tau1, scientific-invalid 0, retry/backtracking/history 0.
- 모든 manifest의 terminal SHA, terminal의 action-freeze SHA, arm별 W0 pointer/5-weight byte restore를 독립 재계산했다. 전부 PASS다.
- 각 arm은 official terminal evaluator를 action-freeze 뒤 정확히 1회 사용했다. inner-step heldout access는 0이다.

| task | cell | scheduler | elapsed | MaxRSS |
|---|---|---|---:|---:|
| `19145_0` (`19146`) | Llama RS N/S | COMPLETED0 | 09:35 | 6.84 GiB |
| `19145_1` (`19147`) | Llama BG N/S | COMPLETED0 | 09:49 | 7.02 GiB |
| `19145_2` (`19148`) | Qwen RS N/S | COMPLETED0 | 10:38 | 10.17 GiB |
| `19145_3` (`19145`) | Qwen BG N/S | COMPLETED0 | 10:21 | 10.19 GiB |

## 2. FACT — endpoint 품질

`E/G/L`은 W-only official evaluator count다. NLL과 margin은 각 metric의 frozen evaluator mean이다.

| model | alloc | arm | E/G/L | Eff new / margin | Gen new / margin | Loc new / margin |
|---|---|---|---:|---:|---:|---:|
| Llama | RS | Neutral | 8/14/79 | 3.693 / 2.195 | 4.308 / 2.016 | 8.035 / 3.523 |
| Llama | RS | Soft | 8/14/79 | 3.682 / 2.211 | 4.294 / 2.041 | 8.027 / 3.513 |
| Llama | BG | Neutral | 8/15/72 | 3.617 / 2.669 | 4.258 / 2.496 | 7.634 / 2.846 |
| Llama | BG | Soft | 8/15/73 | 3.643 / 2.634 | 4.261 / 2.483 | 7.636 / 2.850 |
| Qwen | RS | Neutral | 7/15/80 | 5.207 / 1.703 | 4.883 / 2.867 | 8.808 / 3.604 |
| Qwen | RS | Soft | 7/15/79 | 5.175 / 1.732 | 4.890 / 2.869 | 8.808 / 3.611 |
| Qwen | BG | Neutral | 6/15/79 | 5.802 / .636 | 5.674 / 1.688 | 8.840 / 3.608 |
| Qwen | BG | Soft | 6/14/80 | 5.786 / .653 | 5.690 / 1.677 | 8.846 / 3.619 |

같은 seal의 Official AlphaEdit reference는 Llama `10/20/79`, Qwen `10/20/80`이다. P1R24 completed reference는 Llama RS N/S `10/20/82`, `10/20/81`; Llama BG `10/20/81`, `10/20/82`; Qwen RS `10/19/80`, `10/20/80`이다. P1R24 Qwen BG endpoint는 typed boundary로 `NOT_RECORDED`다. 따라서 P1R27의 낮은 edit count를 locality 개선으로 해석할 수 없다.

## 3. FACT — z-CLF, W-CLF 및 tracking

초기 full-six W-only target-new NLL은 Llama `8.36549`, Qwen `9.30923`이다.

| model | alloc | arm | terminal W full-six NLL | z8 oracle NLL | W−z | initial gamma | 실질 write / 8 | stall / 8 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Llama | RS | Neutral | 3.98998 | 5.22187 | −1.23189 | .57052 | 2 | 6 |
| Llama | RS | Soft | 3.98571 | 5.20838 | −1.22267 | .57052 | 2 | 6 |
| Llama | BG | Neutral | 3.90028 | 6.07771 | −2.17743 | .56542 | 1 | 7 |
| Llama | BG | Soft | 3.90370 | 6.08761 | −2.18391 | .56542 | 1 | 7 |
| Qwen | RS | Neutral | 4.40401 | 7.12274 | −2.71873 | .57596 | 1 | 7 |
| Qwen | RS | Soft | 4.41461 | 7.13435 | −2.71974 | .57596 | 1 | 7 |
| Qwen | BG | Neutral | 5.07681 | 7.51299 | −2.43618 | .58393 | 1 | 7 |
| Qwen | BG | Soft | 5.07221 | 7.51008 | −2.43787 | .58393 | 1 | 7 |

- 총 `WRITER_CLF_SEMANTIC_STALL`은 54/64 transition이다. Stall 구간에도 target-side `r_W`는 대체로 0이 아니며, 현재 선택된 physical/tracking field에서 certified positive write를 만들지 못해 semantic slack으로 totalize된 scientific outcome이다. NaN/solver corruption이 아니다.
- Stall 상태에서는 semantic slack이 현재 `r_W`와 같고 z/W를 함께 hold했다. meaningless solve/retry는 없었다.
- initial action은 모두 유한했고 `gamma_z<1`; P/H가 gamma에 미친 영향은 0이다.
- k8 field의 pre-write mean residual norm은 Llama RS 약 `.0295`, Llama BG 약 `.0020`, Qwen RS 약 `.0020`, Qwen BG 약 `6.4e-5`다. 정확한 post-k8 terminal lag scalar는 별도 직렬화되지 않았다. 이 작은 k8 값도 약한 z trajectory를 따라간 관찰이며 edit strength 성공을 뜻하지 않는다.
- `W−z<0`가 전 arm에서 관찰됐다. 따라서 주된 실패는 “z oracle은 강하지만 writer가 못 따라간 경우”가 아니라, z target 자체가 약하고 writer도 조기 stall한 **target/writer field issue**다.

실질 write 구간의 누적 actual/predicted realization은 Llama BG 약 `1.535`, Llama RS 약 `1.585`, Qwen BG 약 `.556`, Qwen RS 약 `.790`이다. Llama RS의 두 번째 작은 write는 actual progress가 음수였다. candidate trial/rollback/retry는 0이다.

## 4. FACT — Soft preservation/BF

Neutral과 Soft는 각 동일 initial state에서 같은 `r_W`를 받았고 predicted semantic strength가 일치했다. Soft의 strength attenuation은 0이다. 다만 실제 routing DOF는 거의 첫 write에만 있었고, 그 뒤 stall이 지배했다.

| model | alloc | Soft−Neutral E/G/L | Δ Structural-P | Δ functional-P mean | Δ capacity | 판독 |
|---|---|---:|---:|---:|---:|---|
| Llama | RS | 0/0/0 | +3.52e-6 | +8.38e-4 | +5.94e-4 | P/fP/cap 모두 악화 |
| Llama | BG | 0/0/+1 | −2.39e-7 | −4.00e-4 | +1.80e-7 | 유일한 fP 개선, capacity는 증가 |
| Qwen | RS | 0/0/−1 | −4.41e-6 | +6.18e-4 | +8.34e-6 | raw P만 미세 개선 |
| Qwen | BG | 0/−1/+1 | −4.98e-6 | +1.67e-4 | +1.47e-5 | raw P 개선, fP/Gen 악화 |

- Soft decision influence는 네 Soft arm의 첫 active state에서만 명시적으로 관찰됐다. Llama-RS k2는 `SOFT_NO_ROUTING_DOF`; 이후에는 stall이다.
- 같은-state first allocation의 Neutral–Soft coefficient L1 거리는 `8.6e-5–9.5e-5`로 작았다.
- terminal Structural-P는 Llama `.00393–.00671`, Qwen `.18650–.21425`; terminal capacity 합은 Llama `2.49–3.80`, Qwen `17.57–19.58`이다.
- capacity는 Soft에서 네 pair 모두 미세 증가했다. Functional-P mean은 Llama-BG에서만 개선됐다.
- 그러므로 semantic priority 구현은 PASS지만, BF preservation 효과는 일반적으로 성립하지 않는다.

## 5. FACT — compute와 timing

모든 arm은 `180 model F / 125 B-autograd / materialization 8`이다. processed tokens는 Llama `27,207`, Qwen `24,687`이다.

| model | alloc | Neutral edit-core | Soft edit-core | Soft/Neutral | scheduler pair elapsed |
|---|---|---:|---:|---:|---:|
| Llama | RS | 220.61 s | 272.00 s | 1.233× | 575 s |
| Llama | BG | 240.45 s | 265.74 s | 1.105× | 589 s |
| Qwen | RS | 236.58 s | 307.62 s | 1.300× | 638 s |
| Qwen | BG | 267.09 s | 261.01 s | .977× | 621 s |

- field+physical-slope가 arm당 약 `177.68–261.38 s`로 지배적이다. routing solve는 `.004–.010 s` 수준이다.
- official evaluator wall은 arm당 약 `0.99–1.07 s`; model-load wall은 분리되어 기록되지 않았다.
- Official AlphaEdit pure-edit reference `31.908 s`(Llama), `27.273 s`(Qwen)에 대한 단순 timing-only 비는 약 `6.91–8.52×`, `8.68–11.28×`다. 알고리즘/phase가 달라 formal rho나 순수 controller overhead로 주장하지 않는다.
- authoritative GPU peak는 `NOT_RECORDED`; 위 MaxRSS는 scheduler host memory다.

## 6. 이전 실험과의 경계

- P1R24는 일부 arm에서 `10/20` strength를 회복했지만 Qwen BG가 typed boundary였다. P1R27은 그 missing arm을 보간하지 않는다.
- P1R25는 RS Paced-Soft negative ablation일 뿐 BG comparator가 아니다. P1R25 endpoint는 Llama `10/20/81`, Qwen `10/18/78`; P1R27의 edit count는 더 낮다.
- P1R26은 correction activation `0/40`의 `TECHNICAL_HOLD / SCIENTIFIC_UNTESTED`라 performance baseline이 아니다.
- 모든 비교는 동일 outcome-selected 재사용 B10의 mechanistic/descriptive 범위다.

## 7. INFERENCE

1. **z-CLF:** `PARTIAL/WEAK`. zero-seeking 및 backpressure 수치는 정상 작동했지만 z8 oracle NLL이 `5.21–7.51`로 높아 target field 자체가 충분히 강하지 않았다.
2. **W-CLF strength:** `NOT_RECOVERED`. 첫 action은 강했으나 이후 nonzero `r_W`를 충족할 certified positive action이 현재 physical/tracking field와 수치 domain에서 사라져 54/64 stall이 발생했다. totality는 옳았지만 과학적 edit strength는 부족했다.
3. **Tracking:** `MECHANICS_PASS / SCIENTIFIC_INSUFFICIENT`. gamma는 writer보다 z를 빠르게 하지 않았고 lag는 작았지만, 약한 z target을 추적한 결과다.
4. **BF preservation:** `PRIORITY_PASS / SIGNAL_ABSENT`. Soft가 strength를 낮추지는 않았지만 P/fP/capacity/Loc 개선이 일관되지 않았다.
5. **Historical readiness:** `HOLD`. Atomic strength와 일반적 preservation 신호가 모두 부족하므로 Historical 승격 근거가 없다.

## 8. NOT_RECORDED

- P1R24 Qwen-BG endpoint/Soft.
- P1R27 별도 model-load wall, GPU peak/utilization, Native와 겹치지 않는 formal pure-edit rho.
- 정확한 post-k8 terminal lag scalar.
- fresh-sample, sequential/Historical, lifelong, B100 결과.
- Soft가 장기 history에서 갖는 효과.

## 최종 판정 및 주장 경계

`P1R27_ATOMIC_TECHNICAL_PASS; W_CLF_TOTALITY_PASS; ATOMIC_STRENGTH_NOT_RECOVERED; BF_PRESERVATION_SIGNAL_ABSENT; HISTORICAL_HOLD; SCIENTIFIC_PROMOTION_FALSE`

주장 가능한 범위는 “state-dependent locally linearized W-CLF와 z backpressure, semantic-first lexicographic allocation이 계약대로 실행됐고 stall을 안전하게 totalize했다”까지다. CBF/safe-set, exact preservation, universal BF, Historical readiness 또는 promotion은 주장하지 않는다. 추가 model/GPU/Slurm 작업은 수행하지 않았다.
