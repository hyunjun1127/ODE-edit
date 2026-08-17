# P1R51 Request-wise Semantic Allocation A1 최종 Neutral+Soft 분석

## 결론

**최종 분류: `A1_PARTIAL`.** 두 모델에서 target velocity 총에너지는 P1R43 nominal과 보존되었고 terminal z/W continuous NLL은 same-arm P1R43보다 낮아졌다. 그러나 z-Gen correct는 same-arm P1R43 대비 Llama Soft −1/200, Qwen Neutral −3/200, Qwen Soft −3/200이며, Qwen Neutral은 z NLL 감소에도 W8 NLL이 +0.097908 증가했다. 따라서 `FAILURE_C_SIGNAL`과 Neutral의 `FAILURE_D_WRITER_REALIZATION_SIGNAL`을 함께 기록한다. Soft는 Qwen writer gap과 P/capacity/energy를 줄였지만 Gen count 감소가 남아 full positive로 분류하지 않는다.

이 과학 분류는 계약 §8–10의 gate/matrix에 한정한다. `scientific_promotion=false`이며 추가 실험 실행은 본 분석 범위 밖이다.

## 범위·동일성·실행

- P1R51 source HEAD/tree: `9b4a88b73bd0c0fdd0b1470d77ea4354bd0e3486` / `4a3b5e2dbd7e15bf8da32117c25fd88ec2eaafc7`; branch `main`.
- exact parent P1R43 HEAD/tree: `11508b6da11d606521b703037034e1814b70d8a8` / `a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b`.
- contract: `e19617806b9f60945291885d5fcc91383ee4f2c5198aaf40f77ab38bb9e1562a`, 15841 bytes, 520 lines.
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`.
- 모델/data/case/seed: Llama3-8B-Instruct 및 Qwen2.5-7B-Instruct, exact frozen independent B10 case01–10, request 10/case, `PYTHONHASHSEED=51`.
- Neutral job 20251, Soft job 20257; 각 array `0-1%2`, 1GPU/8CPU/65000MiB, endpoint 20.
- 실행 명령은 source-defined submitter의 `sbatch --hold --parsable --array 0-1%2 --chdir <repo> --nodelist devbox ... session05_ode_bf_p1r51_rsa_a1.sbatch <HEAD> <result-parent> <neutral-full|soft-full>`이며, 각 task는 `python -m project.run_scripts.session05_ode_bf_p1r51_rsa_a1`을 실행했다.
- 변경 파일: numerical/source lock 2개, P1 runtime/scalable/independent runtime 3개, P1R51 panel/runtime/allocation 3개, tests 2개, launcher/sbatch/dry-plan/submitter 4개(총 14개). 상세 경로는 integrity table에 수록했다.
- 40/40 endpoint가 alias/case/request order/capture plan/objective plan/evaluator/aggregator/target span/evaluation case identity exact matched. Official AlphaEdit는 model+case 공통 paired reference이며 latent-z/W8/P/capacity/energy는 `NOT_RECORDED`.

## 기술 무결성

| 모델·arm | endpoint | K/materialization | max energy rel.err | W0/action-freeze | fallback/retry/nonfinite |
|---|---:|---:|---:|---:|---:|
| llama3-8b-inst Neutral | 10/10 | 80/80 | 1.636e-11 | 10/10 PASS | 0/0/0 |
| llama3-8b-inst Soft | 10/10 | 80/80 | 1.626e-11 | 10/10 PASS | 0/0/0 |
| qwen2.5-7b-inst Neutral | 10/10 | 80/80 | 1.004e-11 | 10/10 PASS | 0/0/0 |
| qwen2.5-7b-inst Soft | 10/10 | 80/80 | 1.283e-11 | 10/10 PASS | 0/0/0 |

모든 320 accepted step에서 physical h=1, second-h/remaining/debt=0이다. target policy는 `REQUESTWISE_NLL_PROPORTIONAL_ENERGY_ALLOCATION`, parent operator는 `ENTRY_CALIBRATED_PURE_SEMANTIC_GRADIENT`로 공통이다. 동일 W0 pre-route인 k1에서 Neutral/Soft nominal velocity, RSA velocity, target-next 및 alpha_req bytes/value가 20/20 일치했다. k2 이후 trajectory hash 불일치는 arm별 W 상태 차이이므로 동일성을 요구하지 않았다.

## 모델×arm aggregate

| 모델 | arm | z8/W8 full6 NLL | zEff/zGen | WEff/WGen/Loc | gap | P/cap/energy | realization |
|---|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0.064860/0.098573 | 100/100, 183/200 | 100/100, 181/200, 871/1000 | 0.033712 | 0.009056/0.137741/0.214587 | 0.791139 |
| llama3-8b-inst | Soft | 0.061245/0.084344 | 100/100, 183/200 | 100/100, 182/200, 870/1000 | 0.023099 | 0.008112/0.057195/0.088573 | 0.795296 |
| qwen2.5-7b-inst | Neutral | 0.562288/0.992543 | 97/100, 158/200 | 96/100, 153/200, 842/1000 | 0.430254 | 6.631193/1631.054180/966.093660 | 0.711544 |
| qwen2.5-7b-inst | Soft | 0.468272/0.513687 | 97/100, 154/200 | 97/100, 150/200, 844/1000 | 0.045415 | 0.173870/0.287982/0.133448 | 0.755198 |

## paired 산술 차이

| 모델 | 비교 | Δz8/ΔW8/Δgap | ΔzGen/ΔWGen/ΔLoc | ΔP/Δcap/Δenergy |
|---|---|---:|---:|---:|
| llama3-8b-inst | Soft−Neutral | -0.003616/-0.014229/-0.010613 | +0/+1/-1 | -0.000944/-0.080547/-0.126014 |
| llama3-8b-inst | Neutral−P1R43 Neutral | -0.171557/-0.216671/-0.045114 | +1/+0/+3 | +0.000444/-0.011015/-0.011752 |
| llama3-8b-inst | Soft−P1R43 Soft | -0.113651/-0.185176/-0.071524 | -1/+1/+1 | -0.000236/-0.035570/-0.051881 |
| qwen2.5-7b-inst | Soft−Neutral | -0.094016/-0.478856/-0.384840 | -4/-3/+2 | -6.457323/-1630.766199/-965.960211 |
| qwen2.5-7b-inst | Neutral−P1R43 Neutral | -0.227471/+0.097908/+0.325379 | -3/-1/-3 | +6.364690/+1629.819434/+965.408371 |
| qwen2.5-7b-inst | Soft−P1R43 Soft | -0.285274/-0.307356/-0.022081 | -3/-2/-1 | -0.016052/-0.000464/-0.067860 |

## target/writer failure A/B/C/D

### llama3-8b-inst

- Failure A: `UNDER_ALLOCATION_COMPONENT_REDUCED_SIGNAL`. P1R43 Neutral 대비 terminal parent-p90 hard-tail mean Δ=-1.101578, 개선 10/10; Soft same-arm Δ=-0.666926.
- Failure B: `NO_REPEATED_SIGNAL`. extra-allocation+no-improvement request-step Neutral/Soft=0/0, flat-gradient=0/0.
- Failure C: `FAILURE_C_SIGNAL`. same-arm Δz-Gen correct Neutral/Soft=+1/-1 (각 /200).
- Failure D: `NOT_OBSERVED`. same-arm ΔW8 Neutral/Soft=-0.216671/-0.185176, Δgap=-0.045114/-0.071524.

### qwen2.5-7b-inst

- Failure A: `UNDER_ALLOCATION_COMPONENT_REDUCED_SIGNAL`. P1R43 Neutral 대비 terminal parent-p90 hard-tail mean Δ=-2.551976, 개선 7/10; Soft same-arm Δ=-2.668371.
- Failure B: `RESIDUAL_POOR_DIRECTION_CASES_RECORDED`. extra-allocation+no-improvement request-step Neutral/Soft=20/19, flat-gradient=0/0.
- Failure C: `FAILURE_C_SIGNAL`. same-arm Δz-Gen correct Neutral/Soft=-3/-3 (각 /200).
- Failure D: `FAILURE_D_WRITER_REALIZATION_SIGNAL_IN_NEUTRAL`. same-arm ΔW8 Neutral/Soft=+0.097908/-0.307356, Δgap=+0.325379/-0.022081.

기존 ‘Qwen z-Eff 실패 4 request’의 exact request ID는 contract/P1R43 raw-free aggregate에 `NOT_RECORDED`; 임의 ID를 만들지 않았다. per-request 표는 전 1600 paired request-step을 보존하고, parent terminal p90 기준 hard-tail flag/rank를 별도로 제공한다.

## Neutral/Soft Pareto와 preservation

Llama Soft−Neutral은 z8/W8 NLL −0.003616/−0.014229, Loc −1/1000, P/capacity/energy −0.000944/−0.080547/−0.126014이다. Qwen은 −0.094016/−0.478856, Loc +2/1000, P/capacity/energy −6.457323/−1630.766199/−965.960211이다. 다만 Qwen z-Gen/W-Gen은 Soft가 Neutral보다 −4/200/−3/200이므로 strict Pareto dominance로 확정하지 않는다. Loc/P/capacity 이점은 continuous strength와 correct count를 함께 제시했으며 absolute strength가 약한 경우의 단독 preservation claim은 하지 않았다.

## Official/Native reference

Official AlphaEdit paired reference는 exact stream/order/evaluator identity가 검증된 20 model-case이다. P1R51의 Eff/Gen/Loc 및 Eff NLL/margin 산술 차이는 combined table에 기록했다. Official/Native latent-z, W8 full-six NLL, Gen continuous NLL/margin, P/capacity/energy/realization은 `NOT_RECORDED`; 재실행·대체·imputation하지 않았다.

## 비용

| 모델 | arm | F/B | target F/B | tokens | key/field refresh | materialization | pure edit s | terminal eval s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 2230/1250 | 800/850 | 378204 | 80 | 80 | 1195.050 | 380.201 |
| llama3-8b-inst | Soft | 2235/1250 | 800/850 | 379325 | 80 | 80 | 1120.058 | 383.411 |
| qwen2.5-7b-inst | Neutral | 2385/1250 | 800/850 | 377015 | 80 | 80 | 1233.866 | 424.541 |
| qwen2.5-7b-inst | Soft | 2370/1250 | 800/850 | 373820 | 80 | 80 | 1425.425 | 426.273 |

## 최종 gate

- 기술 상태: `PASS` (40/40 endpoints, 320/320 K/materializations, typed failure 0).
- 과학 상태: `A1_PARTIAL`.
- Failure taxonomy: target hard-tail/continuous strength 개선 신호 + context-generalization 미해결(`FAILURE_C_SIGNAL`) + Qwen Neutral writer 병목(`FAILURE_D_WRITER_REALIZATION_SIGNAL`), Soft에서 writer 병목 산술 완화.
- 후속 경계: 계약 §10 분기상 train target 개선·z-Gen 불변/감소 및 Neutral writer 소실 사실까지만 기록한다. 새로운 실행·튜닝·promotion은 승인하지 않는다.
- `scientific_promotion=false`.

## 실패 request cohort amendment

cohort는 terminal z-Eff/z-Gen/W-Eff/W-Gen failure, CURRENT/RESCUE, same-arm P1R43 regression, high RSA share+weak/negative NLL progress, z-success/W-failure의 8종이다. 한 request는 여러 cohort/label에 중복될 수 있으며, count overlap은 raw-free association일 뿐 causal correlation claim이 아니다.

| 모델 | arm | zEff fail | zGen fail | WEff fail | WGen fail | C/R | regressed | high-share weak | z-ok/W-fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0/100 | 14/100 | 0/100 | 16/100 | 3/100 | 37/100 | 0/100 | 2/100 |
| llama3-8b-inst | Soft | 0/100 | 14/100 | 0/100 | 15/100 | 4/100 | 40/100 | 0/100 | 1/100 |
| qwen2.5-7b-inst | Neutral | 3/100 | 33/100 | 4/100 | 37/100 | 16/100 | 62/100 | 0/100 | 7/100 |
| qwen2.5-7b-inst | Soft | 3/100 | 35/100 | 3/100 | 38/100 | 18/100 | 55/100 | 0/100 | 3/100 |

허용 label은 `HARD_SAMPLE_SUPPORTED`, `NEW_RSA_FAILURE`, `DIRECTION_LIMITED`, `WRITER_LIMITED`, `CONTEXT_GENERALIZATION_LIMITED`, `INCONCLUSIVE`만 사용했다. 각 failed-request 행은 model/case/request/order, entry/terminal z/W NLL, terminal outcomes 및 P1R43/N→S deltas, gradient norm ratio/flat count, nominal/RSA share·rank·cumulative, accepted path, PRIMARY/RESCUE/CURRENT sequence/first step, z-W gap/residual/realization/negative progress, layer weights/entropy/P/capacity/energy를 포함한다.

기존 네 hard request ID는 `NOT_RECORDED`; imputation하지 않았다. 대신 parent Neutral terminal p90 기준 descriptive flag와 전체 100 request/model/arm denominator를 보존했다.

## A–F 질문

- A. hard samples에 RSA energy가 배정되고 terminal target NLL이 감소했는가? → `PARTIAL_SUPPORTED`.
- B. 높은 RSA share에도 progress가 약한 direction-limited request가 남았는가? → `RECORDED`.
- C. train target strength가 context generalization까지 이어졌는가? → `CONTEXT_GENERALIZATION_LIMITED`.
- D. z 성공을 W가 모두 실현했는가? → `WRITER_LIMITED_REQUESTS_RECORDED`.
- E. Soft는 same-strength 구간에서 P/capacity/energy 및 preservation을 개선했는가? → `MIXED_PARETO`.
- F. A1 최종 positive/partial/negative 분류는 무엇인가? → `A1_PARTIAL`.

세부 cohort/pattern denominator, 대표 request와 N→S transition은 별도 machine table에 수록했다.
