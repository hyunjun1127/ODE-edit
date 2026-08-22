# P1R30 A2 Atomic B10 전체 행렬 진단 종료 보고서

instruction: `ODEEDIT-S05-P1R30-DEBT-PRIORITY-A0-RELATIVE-BARRIER-V1`  
amendment: `P1R30_A2_DIAGNOSTIC_FULL_MATRIX_OVERRIDE`  
scientific source: `4373697786b50f9a936ad726a5e43bc093f19ac1`, tree `fad4fd881f1406686a36d941852dcb8a9d390761`  
isolated Qwen-BG-Soft dispatch/launcher child: `8fcf9dea22a684edf6c02556238a5bdcd80e9f51`, tree `4fff8272bc6ac5e76de3794013cf82634a2d699f`  
seal/order: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628` / `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`

## 결론

사전 규정된 Neutral recovery gate는 바뀌지 않았다.

- `PREDECLARED_NEUTRAL_RECOVERY_GATE=FAIL`
- Llama RS Neutral: PASS
- Qwen RS Neutral: `SCIENTIFIC_FAIL_NEUTRAL_RECOVERY`
- `scientific_promotion=false`

A2 진단 override 아래 8개 cell을 모두 독립적으로 시도했다. 6개는 K8/tau1 endpoint, action-freeze, terminal evaluator 및 exact W0 restore까지 유효하다. Qwen BG Neutral과 Soft는 각각 k4/tau=.5까지 정상 accepted prefix를 만든 뒤, 다음 field에서 같은 typed `WRITER_NO_POSITIVE_DIRECTION` 경계로 종료했다. 두 prefix는 endpoint로 보간하거나 대체하지 않았다.

`FULL_MATRIX_DIAGNOSTIC=PARTIAL_6_OF_8_ENDPOINTS_PLUS_2_TYPED_K4_PREFIXES`

Soft의 결과는 모델·allocation에 따라 달랐다.

1. Llama BG에서는 Soft가 실제 allocation을 6/8 step에서 바꿨고, BG Neutral 대비 terminal Structural-P `−10.15%`, endpoint cumulative BF16 capacity `−12.32%`, Eff/Gen new NLL 및 realization이 함께 개선됐다. 이는 가장 분명한 긍정적 mechanistic signal이다.
2. Llama RS에서는 Soft가 5/8 step을 바꿨지만 별도 Neutral trajectory 대비 terminal Structural-P `+3.35%`, cumulative capacity `+4.29%`, Eff/Gen NLL이 소폭 악화됐다. same-state local objective는 감소했으나 누적 trajectory 보존으로 이어지지 않은 혼합 결과다.
3. Qwen RS는 8/8 `SOFT_FALLBACK_NEUTRAL`로 Soft decision influence가 0이고 endpoint가 Neutral과 동일하다.
4. Qwen BG도 k1..k4 모두 exact Neutral fallback 뒤 두 arm 모두 같은 과학적 writer 경계에 도달했다. endpoint Soft 효과는 식별할 수 없다.

따라서 `BARRIER/DEBT_PRIORITY_ROUTING_EFFECT=DESCRIPTIVE_MECHANISTIC_MIXED`이며 범용 보존 개선이나 strength recovery를 주장할 수 없다.

## 1. FACT — 제출, scheduler, 분모

유효한 기존 RS Neutral 2개는 재사용했고, A2 array `19299_[0-3]%4`로 누락 6개를 제출했다. Qwen BG paired task에서 Neutral 경계가 Soft 시작을 막았기 때문에, 과학 바이트를 바꾸지 않는 isolated dispatch/launcher child로 Soft만 별도 시도했다.

| 논리 task | cell | scheduler | elapsed | MaxRSS | 결과 |
|---|---|---|---:|---:|---|
| reused `19281` | Llama RS N | COMPLETED `0:0` | 00:02:10 | 7,163,016 KiB | K8 endpoint |
| reused `19280` | Qwen RS N | COMPLETED `0:0` | 00:02:43 | 10,264,364 KiB | K8 endpoint |
| `19299_0` / raw `19300` | Llama RS S | COMPLETED `0:0` | 00:02:00 | 7,197,436 KiB | K8 endpoint |
| `19299_1` / raw `19301` | Qwen RS S | COMPLETED `0:0` | 00:02:16 | 10,561,068 KiB | K8 endpoint |
| `19299_2` / raw `19302` | Llama BG N+S | COMPLETED `0:0` | 00:04:00 | 6,997,068 KiB | endpoint 2개 |
| `19299_3` / raw `19299` | Qwen BG N | FAILED `1:0` | 00:01:37 | 10,309,196 KiB | typed k4 prefix |
| `19316_0` | Qwen BG S | FAILED `1:0` | 00:01:41 | 10,153,864 KiB | typed k4 prefix |

현재 P1R30 active/pending job은 0이고 callback job은 생성하지 않았다.

## 2. FACT — terminal 및 result integrity

유효 endpoint 6개 모두 다음 chain을 독립 재검증했다.

- actual terminal SHA = manifest `terminal_sha256`
- actual action-freeze SHA = terminal `action_freeze_sha256`
- actual W0-endpoint SHA = terminal `W0_endpoint_sha256`
- terminal rollout identity = action-freeze rollout identity = endpoint rollout identity
- accepted k1..k8 identity sequence = terminal accepted-receipt identity sequence
- K=8, tau=1, dynamic field refresh 8, static/frozen field reuse 0
- materialization 8, one-h, incremental BF16 accumulation 0, exact pointer/parameter restore
- sequential/history/replay-H/inner evaluator influence 0
- inverse-slope, `rho/slopes`, absolute strength cap, retry/backtracking 0

| result root | terminal | manifest | action-freeze | W0 endpoint |
|---|---|---|---|---|
| Llama RS N | `f53a52fbe0aed31a7f4fe5eab07c1775807a3d9ceec01f6275baa212b4e0ff04` | `dc98dbfc09e3e4eb0ed7b7a046d972f4722cc2e14ef52a3763b90392421f8e44` | `e8c9f9e0364330f87b02988c3d0465d991baf688509841a79e474822ac7db303` | `f10ddb6347f7356d07b6f6085b80d2a3c79d29748cd88582bfafb553ebe8ef8f` |
| Llama RS S | `829fe7d073a32677ec792e9fe7bcc5bc38dc50d31867a178b2ab35065ae48669` | `2f4bef668a5685324b823e50f764e58185202f8dd7eeec69bd8e33ab415f1a0c` | `238d7d81bc4f534acebc789075f33a963c8debcdb8af4fe60ce04dbdf39abd6e` | `cd7a690770d685d920e6e8b3398a6da123f1c3185b13b424b7d304bb79ef2dac` |
| Llama BG pair | `8dd9519b2be686a7eac34ef753a5549fc01f83ba463935b25982d0c9736b58d1` | `81ac5beb02c1f8bccb92eb50e37ee435b5387de41e85f3e3f49e6a2a0a06367f` | `c5852d664cda01078dc1140a81a78b6138ba6e6441ffd38caff4d9a70df055c1` | `4272329cbc60cc8032edf54305e768120a1dc62847b6f05718264275e6079da5` |
| Qwen RS N | `0f169b9a6d5e8aab1a9d9b3c9bf0646becbfa5e102fbf18648b16292f01d6bff` | `024e6585a75209a3503fbfa9b3a8884ac943adbf2f0d0451306cb5abbc97855b` | `a3a4cc023a645de77c577c28c71afcc623a481a0fd08b8173809cc7841872b37` | `4b6c33d4c4a139b7dbe171dc2fbaf8343c9f7f1761645275361f77eeb323f63e` |
| Qwen RS S | `9ffb9a90b20c607ca8d0915dd5e687f27c796befaf3672679d3c897c56cc5664` | `5d1c928fb5e7b3c35c4d4fe3e291e4e533fc58b850c78f9ee4922965d40b612e` | `b99438492eaa1b70c64d2bd6a8f215b26b77a9a792b64f5356925cc05a7e90f4` | `e8dabdbfa4798d61088636b767764536517725fd4d2cdd99c275d6fdf8900a3b` |

Qwen BG Neutral/Soft failure SHA는 각각 `b28118678dab5ce74dff43ebe6154655b0370cf58537ba9fbbe5a570f6e2d7b5`와 `b999f7984b10265858fc1681efd88daf44ef877f9c3f0f09c8fb58485822d7a6`이다. accepted-prefix identity-sequence root는 각각 `de0960e8a1c5919cdf52f7318512de6f1bfb751263a595407414ff62aafcfaae`, `88ae11c1a6cfaf33cb5677db756e192e370ec460274f9ea855103018d6ea3dc0`이다.

## 3. FACT — terminal Eff/Gen/Loc 및 continuous metrics

N/O/M은 target-new NLL / target-old NLL / receipt margin이다.

| Cell | Eff/Gen/Loc | Eff N/O/M | Gen N/O/M | Loc N/O/M |
|---|---:|---|---|---|
| Llama RS N | 10/10 · 20/20 · 81/100 | .023298 / 11.5375 / 11.514202 | .985429 / 8.8250 / 7.839571 | 8.314531 / 4.317358 / 3.997173 |
| Llama RS S | 10/10 · 20/20 · 81/100 | .026079 / 11.4813 / 11.455171 | 1.082293 / 8.7078 / 7.625520 | 8.321406 / 4.314993 / 4.006414 |
| Llama BG N | 10/10 · 20/20 · 81/100 | .743001 / 9.8563 / 9.113249 | 1.448070 / 8.7547 / 7.306618 | 8.264844 / 4.277100 / 3.987744 |
| Llama BG S | 10/10 · 20/20 · 81/100 | .607318 / 10.3250 / 9.717682 | 1.224545 / 8.8922 / 7.667642 | 8.276875 / 4.280623 / 3.996252 |
| Qwen RS N | 10/10 · 19/20 · 79/100 | 1.399106 / 10.4875 / 9.088394 | 2.900348 / 9.1148 / 6.214496 | 8.824688 / 5.191187 / 3.633501 |
| Qwen RS S | 10/10 · 19/20 · 79/100 | 1.399106 / 10.4875 / 9.088394 | 2.900348 / 9.1148 / 6.214496 | 8.824688 / 5.191187 / 3.633501 |
| Qwen BG N/S | endpoint 없음 | `NOT_RECORDED` | `NOT_RECORDED` | `NOT_RECORDED` |

W0는 Llama `2/10, 3/20, 84/100`, Qwen `0/10, 4/20, 80/100`이었다.

## 4. FACT — predicted/actual progress, debt, Structural-P, capacity

`ΣP_k`는 accepted k1..k8의 cumulative Structural-P discrete sum이다. `C_BF16`은 endpoint cumulative BF16 capacity의 5-layer 합이다. `C8/E8`은 k8 router의 selected capacity/energy이고 actual BF16 step energy와 구분한다.

| Cell | Σ predicted / actual / realization | negative mean step | debt mean/max | ΣP / P8 | C8 / E8 | C_BF16 | z-oracle NLL / W Eff NLL |
|---|---|---:|---|---|---|---:|---|
| Llama RS N | 13.3830 / 8.3161 / .6214 | 0 | 5.0290 / 11.9914 | .011712 / .003427 | .1201 / .2049 | 2.5036 | .039342 / .023298 |
| Llama RS S | 13.4135 / 8.3147 / .6199 | 0 | 5.0668 / 11.9076 | .011774 / .003542 | .1435 / .2449 | 2.6110 | .039926 / .026079 |
| Llama BG N | 15.5190 / 7.5938 / .4893 | 1 | 7.9515 / 25.6628 | .015768 / .005989 | .6194 / 1.0172 | 4.3777 | .078068 / .743001 |
| Llama BG S | 15.5718 / 7.8136 / .5018 | 1 | 7.7720 / 26.8620 | .015146 / .005381 | .3985 / .6571 | 3.8384 | .087832 / .607318 |
| Qwen RS N | 34.7234 / 8.0273 / .2312 | 2 | 26.8230 / 65.5739 | 1.028464 / .529561 | 50.2470 / 29.1718 | 58.9364 | .056804 / 1.399106 |
| Qwen RS S | 34.7234 / 8.0273 / .2312 | 2 | 26.8230 / 65.5739 | 1.028464 / .529561 | 50.2470 / 29.1718 | 58.9364 | .056804 / 1.399106 |
| Qwen BG N* | 27.5727 / 8.4435 / .3062 | 0 | 19.3278 / 68.9647 | .153471 / .069329 | 1.5947 / 1.6395 | 7.8198 | endpoint 없음 |
| Qwen BG S* | N과 동일 | 0 | N과 동일 | N과 동일 | N과 동일 | N과 동일 | endpoint 없음 |

`*`는 k1..k4 prefix 합계다. terminal debt가 아니라 k4 직후 debt다.

모든 Neutral은 exact A0 `c=c0`, debt total-magnitude influence 0이다. 모든 Soft accepted transition은 mean/debt reference-relative progress와 energy parity certificate를 통과했으며, inverse-slope count와 magnitude influence count는 0이다. 남은 debt는 숨기거나 갚은 것으로 impute하지 않았다.

## 5. FACT / INFERENCE — Soft allocation과 barrier 효과

모든 관측 field의 direct-c feasible dimension은 3이었다. 따라서 Qwen 결과는 수학적 `NO_ROUTING_DOF`가 아니라, nontrivial Soft solution을 인증하지 못해 exact `c0`로 되돌아간 것이다.

| Cell | Soft selected / fallback | mean coefficient L1 / max | same-state ΣP 변화 | same-state Σcapacity 변화 | 별도 Neutral trajectory 대비 |
|---|---:|---:|---:|---:|---|
| Llama RS | 5 / 3 | .00784 / .01448 | −.186% | +.045% | P8 +3.35%, ΣP +.53%, C_BF16 +4.29% |
| Llama BG | 6 / 2 | .01023 / .02683 | −.298% | −3.93% | P8 −10.15%, ΣP −3.95%, C_BF16 −12.32% |
| Qwen RS | 0 / 8 | 0 / 0 | 0 | 0 | endpoint 완전 동일 |
| Qwen BG prefix | 0 / 4 | 0 / 0 | 0 | 0 | k4 prefix 완전 동일 |

해석:

- Llama BG는 local optimizer 성공이 누적 P/capacity 및 Eff/Gen continuous metric 방향까지 이어진 `DESCRIPTIVE_PRESERVATION_SIGNAL`이다. 하지만 한 allocation·한 모델의 reused B10 신호이므로 promotion 근거는 아니다.
- Llama RS는 Soft-state local P를 낮췄지만 trajectory drift 후 endpoint P/capacity와 edit NLL이 악화됐다. `LOCAL_ROUTER_SUCCESS_WITHOUT_PACKAGE_LEVEL_PRESERVATION`이다.
- Qwen RS/BG는 Soft 선택 영향이 0이어서 barrier 효과가 `NOT_IDENTIFIABLE`이다.
- Qwen RS realization `.2312`, 큰 debt tail, terminal 음의 progress를 포함한 기존 Neutral recovery 실패는 A2 결과로 번복되지 않는다.

## 6. FACT — compute

| Cell | edit-core | model F / B | tokens | materializations |
|---|---:|---:|---:|---:|
| Llama RS N | 73.479 s | 180 / 125 | 27,207 | 8 |
| Llama RS S | 67.173 s | 180 / 125 | 27,207 | 8 |
| Llama BG N | 72.017 s | 180 / 125 | 27,207 | 8 |
| Llama BG S | 106.138 s | 180 / 125 | 27,207 | 8 |
| Qwen RS N | 98.928 s | 180 / 125 | 24,687 | 8 |
| Qwen RS S | 76.562 s | 180 / 125 | 24,687 | 8 |

Soft router가 추가한 model F/B는 0이다. wall-time 차이는 paired ordering·GPU runtime 변동을 포함하므로 router 자체의 인과적 비용으로 해석하지 않는다. Qwen BG prefix는 완결 compute ledger가 없고 materialization 4회까지만 확인됐다. Online functional-P/fKL과 stepwise external evaluator는 0이며 terminal functional-P/fKL은 기록되지 않았다.

## 7. Baseline 경계

Official Native는 같은 sealed B10/order/W0/evaluator를 사용한 pinned EasyEdit AlphaEdit endpoint reference다. P1R30 A0-relative causal control은 아니다.

| 모델 | Official Native E/G/L | Eff/Gen/Loc new NLL | P1R30 관찰 |
|---|---:|---|---|
| Llama | 10/10 · 20/20 · 79/100 | .002545 / .750505 / 8.082656 | 모든 유효 cell count는 E/G 동일, Loc +2; continuous edit NLL은 Native보다 약함 |
| Qwen | 10/10 · 20/20 · 80/100 | .009536 / 1.287744 / 8.831719 | RS는 Gen/Loc −1 및 continuous edit NLL가 크게 약함; BG endpoint 없음 |

Exact matched P1R24 reproduction인 P1R31의 create-once raw-free package는 bounded 확인 시점에 없었다. 따라서 P1R24/P1R31과의 case/cell matched claim은 이 보고서에 포함하지 않았다. package가 이후 도착하면 별도 addendum만 작성한다.

## 8. TECHNICAL_FAIL — 보존된 isolation lineage

Qwen BG paired Neutral의 scientific boundary가 같은 process 안의 Soft 시작을 막은 것은 cell 독립성 측면의 orchestration 문제였다. 다음 최소 repair는 science/objective/tolerance/sample/evaluator를 바꾸지 않았다.

- commit `99f7627`: isolated BG Soft role/dispatch 추가
- commit `8fcf9de`: 해당 role의 Slurm launcher mapping 추가
- focused 20/20, compile/bash/session/source-manifest PASS
- job `19314`: held inspection에서 non-array 제출을 발견해 release 전 취소, model/GPU/result action 0
- job `19315`: `SLURM_ARRAY_TASK_ID` unbound로 1초 pre-model 실패, accepted transition 0
- job `19316_0`: 같은 source를 올바른 array invocation으로 실행, k4 후 scientific typed boundary

정상 cell 재실행은 0이다.

## 9. SCIENTIFIC_FAIL / claim boundary

1. `PREDECLARED_NEUTRAL_RECOVERY_GATE=FAIL`은 유지한다.
2. Full matrix는 8/8 attempted이나 유효 endpoint는 6/8이다. Qwen BG 2개는 typed prefix로만 분모에 남긴다.
3. Llama BG에서 preservation signal이 있으나 Llama RS는 혼합, Qwen은 비식별/경계다. universal barrier/debt-priority improvement를 주장할 수 없다.
4. debt는 request priority에만 관여했고 total magnitude는 바꾸지 않았다. 따라서 Qwen strength 실패를 debt 증폭으로 설명할 수 없다.
5. 결과는 reused outcome-selected Atomic B10의 descriptive/mechanistic evidence다. fresh-sample, sequential/Historical, lifelong preservation 또는 scientific promotion 증거가 아니다.

## 10. NOT_RECORDED

- Qwen BG terminal E/G/L, continuous metrics, action-freeze, W0 restore
- online/terminal functional-P 및 fKL
- authoritative GPU utilization/time
- exact same-sample P1R24/P1R31 matched package와 P1R28-C2 package
- sequential/Historical behavior

## 최종 판정

`TECHNICAL_INTEGRITY_VALID_ENDPOINTS=PASS`  
`PREDECLARED_NEUTRAL_RECOVERY_GATE=FAIL_UNCHANGED`  
`FULL_MATRIX_DIAGNOSTIC=PARTIAL_6_ENDPOINTS_2_TYPED_PREFIXES`  
`LLAMA_BG_BARRIER_SIGNAL=DESCRIPTIVE_POSITIVE`  
`LLAMA_RS_BARRIER_SIGNAL=MIXED`  
`QWEN_BARRIER_SIGNAL=NOT_IDENTIFIABLE_WITH_BG_TYPED_BOUNDARY`  
`SCIENTIFIC_PROMOTION=false`  
`FOLLOWUP_MODEL_GPU_SLURM_ACTION=0`
