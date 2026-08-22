# P1R30 RS Debt-Neutral B10 회복 gate 종료 보고서

instruction: `ODEEDIT-S05-P1R30-DEBT-PRIORITY-A0-RELATIVE-BARRIER-V1`  
estimand: reused outcome-selected sealed B10의 Atomic RS Debt-Neutral  
source head: `4373697786b50f9a936ad726a5e43bc093f19ac1`  
source tree: `fad4fd881f1406686a36d941852dcb8a9d390761`  

## 결론

P1R30 RS Debt-Neutral은 Llama와 Qwen 모두 기술적으로 정상 완주했다. 두 셀 모두 K=8, tau=1, dynamic field refresh 8회, accepted BF16 materialization 8회, terminal action-freeze, exact W0 restore를 통과했고 scientific-invalid는 0이었다.

그러나 사전 규정된 Neutral recovery gate는 양 alias 동시 PASS가 아니다.

- Llama: `PASS`
- Qwen: `SCIENTIFIC_FAIL_NEUTRAL_RECOVERY`
- combined: `BOTH_ALIASES_P1R24_A0_STRENGTH_REGION_PASS=false`

Qwen은 endpoint count만 보면 Official Native 대비 Eff 동일, Gen/Loc 각 −1로 lenient count 범위에 있다. 반면 continuous Eff/Gen target-new NLL은 `1.399106/2.900348`로 Official Native의 `.009536/1.287744`보다 명백히 약하고, 누적 predicted/actual progress는 `34.723361/8.027288`로 realization `.231178`에 그쳤다. 실제 mean progress가 음수인 step이 2회이고 terminal step은 predicted `2.308601`에 actual `−1.236690`이었다. terminal debt mean/max는 `26.823004/65.573888`, k8 update energy/capacity는 `29.171759/50.247021`로 후반 급증했다.

따라서 이는 기술 결함이나 solver certificate 실패가 아니라, Qwen에서 A0 writer strength region을 회복하지 못한 유한 scientific outcome이다. 계약상 Soft 및 A1 full RS/BG×Neutral/Soft 8-cell matrix는 제출하지 않았다.

## 1. FACT — 실행 및 무결성

TECH-R3 submission은 Slurm array `19280`, `0-1%2`, 최대 동시 GPU 2, server2 janghj cap 4로 held inspection 후 release됐다. accounting raw job은 Llama `19281`, Qwen `19280`이다.

| 모델 | scheduler | elapsed | MaxRSS | terminal SHA-256 | manifest SHA-256 | action-freeze SHA-256 |
|---|---|---:|---:|---|---|---|
| Llama3-8B-Instruct | COMPLETED `0:0` | 00:02:10 | 7,163,016 KiB | `f53a52fbe0aed31a7f4fe5eab07c1775807a3d9ceec01f6275baa212b4e0ff04` | `dc98dbfc09e3e4eb0ed7b7a046d972f4722cc2e14ef52a3763b90392421f8e44` | `e8c9f9e0364330f87b02988c3d0465d991baf688509841a79e474822ac7db303` |
| Qwen2.5-7B-Instruct | COMPLETED `0:0` | 00:02:43 | 10,264,364 KiB | `0f169b9a6d5e8aab1a9d9b3c9bf0646becbfa5e102fbf18648b16292f01d6bff` | `024e6585a75209a3503fbfa9b3a8884ac943adbf2f0d0451306cb5abbc97855b` | `a3a4cc023a645de77c577c28c71afcc623a481a0fd08b8173809cc7841872b37` |

독립 재검증에서 양 모델 모두 다음 chain이 PASS했다.

- actual terminal SHA = manifest `terminal_sha256`
- actual action-freeze SHA = terminal `action_freeze_sha256`
- actual W0-endpoint SHA = terminal `W0_endpoint_sha256`
- paired-initial gate file SHA = action-freeze binding
- action-freeze rollout identity = terminal rollout identity
- accepted k1..k8 file identity list = terminal accepted receipt identity list
- manifest `W0_restored=true`, scientific-invalid count 0

공통 contract facts:

- exact sealed order `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- RS, Atomic joint B10, persistent history/replay H/sequential influence 0
- dynamic refresh 8, frozen/static field reuse 0
- direct applied-`c` A0 Neutral: selected `c=c0`; debt magnitude decision influence 0
- inverse-slope, `rho/slopes`, legacy exact-strength helper, absolute strength cap, retry/backtracking 0
- one-h certificate, mean/debt progress ratio 1, energy ratio 1 at all steps
- requestwise slope reduction certificate residual max 0; FP64→model-facing FP32 representation rounding은 Llama `1.91e-7`, Qwen `4.29e-7`로 별도 telemetry이며 decision influence 0

## 2. FACT — terminal endpoint

| 모델/방법 | Eff | Gen | Loc | Eff new NLL | Gen new NLL | Loc new NLL | Eff margin | Gen margin | Loc margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama P1R30 RS Neutral | 10/10 | 20/20 | 81/100 | .023298 | .985429 | 8.314531 | 11.514202 | 7.839571 | 3.997173 |
| Llama Official Native | 10/10 | 20/20 | 79/100 | .002545 | .750505 | 8.082656 | 12.628705 | 9.858870 | 3.757390 |
| Qwen P1R30 RS Neutral | 10/10 | 19/20 | 79/100 | 1.399106 | 2.900348 | 8.824688 | 9.088394 | 6.214496 | 3.633501 |
| Qwen Official Native | 10/10 | 20/20 | 80/100 | .009536 | 1.287744 | 8.831719 | 15.415464 | 11.134131 | 3.693335 |

Official Native는 같은 sealed B10/order/W0/evaluator를 사용한 pinned Official EasyEdit AlphaEdit endpoint reference다. P1R30 Neutral과 알고리즘이 다르므로 A0-relative causal control로 해석하지 않는다.

## 3. FACT — predicted/actual progress, debt, P, capacity

| 모델 | Σ predicted | Σ actual | realization | negative mean steps | terminal debt mean/max | terminal Structural-P | k8 energy | k8 capacity | z-oracle NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | 13.383049 | 8.316071 | .621388 | 0 | 5.028974 / 11.991423 | .003427 | .204940 | .120073 | .039342 |
| Qwen | 34.723361 | 8.027288 | .231178 | 2 | 26.823004 / 65.573888 | .529561 | 29.171759 | 50.247021 | .056804 |

Llama의 per-step energy는 `.117,.101,.111,.152,.079,.163,.113,.205`, capacity는 `.059,.052,.058,.081,.044,.091,.065,.120`으로 후반 runaway가 없다. Qwen은 k1..k7 energy가 `1.114,1.212,1.405,1.422,2.221,1.683,.769`였으나 k8에 `29.172`로, capacity는 `1.254→50.247`로 급증했다. Qwen terminal actual progress는 `−1.236690`, request-wise negative actual은 9/10이었다.

## 4. FACT — compute

| 모델 | edit-core | model F | B/autograd | capture F | slope B | target B | processed tokens | materializations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | 73.479 s | 180 | 125 | 45 | 40 | 85 | 27,207 | 8 |
| Qwen | 98.928 s | 180 | 125 | 45 | 40 | 85 | 24,687 | 8 |

Online functional-P model F/B와 stepwise external evaluator는 0이다. terminal functional-P는 `NOT_RECORDED_P1R30_ONLINE_FUNCTIONAL_P_DISABLED`이다.

## 5. TECHNICAL_FAIL — 보존된 선행 시도

성공한 TECH-R3 이전 실패는 scientific action 전에 발생했고 모두 immutable하게 보존했다.

1. job `19251`: external physical GPU capacity collision, model load/accepted transition 0.
2. job `19260`: exact result namespace adapter omission, pre-model/accepted transition 0.
3. job `19276`: FP64 requestwise mean과 의도된 model-facing FP32 roundtrip 사이 certificate-coordinate mismatch, accepted transition 0. TECH-R3는 tolerance/physical slope/target/routing을 바꾸지 않고 model-facing coordinate에서 reduction identity를 인증하고 pre-cast rounding을 별도 receipt했다.

이 technical lineage는 최종 scientific outcome을 중복하거나 선택하지 않았다.

## 6. INFERENCE / SCIENTIFIC_FAIL

1. **Llama Neutral recovery는 PASS다.** Official Native 수준의 Eff/Gen count, Loc +2, continuous edit metric의 비붕괴, negative mean progress 0, 안정적인 energy/capacity가 함께 관찰됐다.
2. **Qwen Neutral recovery는 FAIL이다.** count만으로는 경계 내지만 continuous strength, realization, debt tail, terminal negative progress, late energy/capacity가 동시에 실패 방향이다.
3. **Debt priority의 total-scale 비증폭 contract는 지켜졌다.** Neutral이므로 모든 step에서 `c=c0`이고 debt의 allocation/magnitude 영향은 0이다. 즉 Qwen 실패를 debt-priority routing 효과나 Soft barrier 효과로 해석할 수 없다. 이것은 A0 reference writer의 해당 trajectory에서 물리적 realization이 무너진 결과다.
4. **Soft 효과는 아직 측정되지 않았다.** 양 alias Neutral 회복이라는 선행 gate가 false이므로 Soft 또는 BG까지 실행해 유리한 cell을 선택하는 것은 계약 위반이다.
5. **Scientific promotion은 불가하다.** P1R30은 reused outcome-selected sealed B10의 negative recovery boundary이며 fresh-sample/promoted method 증거가 아니다.

## 7. NOT_RECORDED / 비교 경계

- exact same-seal P1R24 A0 endpoint receipt는 현재 허용된 SH2 closure에 없다. 따라서 P1R30과 P1R24 A0의 exact paired continuous difference는 `NOT_RECORDED`다.
- GH가 제공한 broader P1R24 B10×10 aggregate(Llama 99/100,177/200,873/1000; Qwen 100/100,163/200,844/1000)는 다른 stream/denominator의 descriptive reference라서 이 matched gate 계산에 쓰지 않았다.
- P1R28 C2의 create-once matched raw-free endpoint package는 현재 closure에 없다.
- P1R31은 이후 matched comparison input일 뿐이며 본 제출/gate를 기다리게 하지 않았다. P1R31 결과가 도착하면 별도 addendum만 작성할 수 있다.
- functional-P, authoritative GPU utilization은 result receipt에 기록되지 않았다. MaxRSS는 Slurm batch accounting 값이다.

## 최종 판정

`RS_NEUTRAL_TECHNICAL_INTEGRITY=PASS`  
`LLAMA_NEUTRAL_RECOVERY=PASS`  
`QWEN_NEUTRAL_RECOVERY=FAIL_SCIENTIFIC`  
`BOTH_ALIASES_P1R24_A0_STRENGTH_REGION_PASS=false`  
`SOFT_AND_FULL_MATRIX_RELEASE=NOT_AUTHORIZED_BY_GATE`  
`FOLLOWUP_MODEL_GPU_SLURM_ACTION=0`
