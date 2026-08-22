# P4-Euler ZA h=1 Llama B2–B10 최종 factual report

> **핵심 한계 — projection boundary saturation.** 이 실행은 `PROJECTION_DOMINATED_EXPLORATORY_RUN`이다. Z+는 286/450 (63.56%), Z±는 381/450 (84.67%) request×microstep에서 projection clamp가 발생했다. 5개 step 모두 clamp된 request도 각각 19/90, 54/90이다. clamp radius factor `.75`와 이 hit fraction은 서로 다른 양이다. 아래 NLL/margin과의 관계는 기술적 연관만 기록하며 인과로 해석하지 않는다.

## 판정과 범위

- 기술 판정: `TERMINAL_TECHNICAL_PASS_POST_ZA_PAUSED`.
- 과학 분류: `USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION`에 따른 `EXPLORATORY_CONFIRMATION_AFTER_CALIBRATION`; predeclared confirmation이 아니다.
- 설정: Llama3-8B-Instruct, sealed B2–B10 9 slices/90 requests, `h=1`, `M=5`, `T_z=5`, target-only W0 freeze.
- 비교: causal panel은 Z±−Z+뿐이다. Native-Z는 원본 EasyEdit compute-z의 observation-only 외부 reference다.
- 후속: ZB/Qwen/재튜닝/추가 calibration/repair/rerun/GPU/Slurm/model action 0. `scientific_promotion=false`, `IDLE_AWAITING_GH_CALL`.

## 1. Projection clamp

| arm | hit/denominator | hit rate | all-5-step saturated | zero-hit request |
|---|---:|---:|---:|---:|
| Z+ | 286/450 | 63.56% | 19/90 | 7/90 |
| Z± | 381/450 | 84.67% | 54/90 | 1/90 |

Slice/arm별 분자·분모·율과 all-5 request 수는 `slice-arm-clamp.csv`, request별 0–5 hit 분포는 `request-endpoint-train.csv`, step별 hit는 `request-microsteps.csv`에 있다.

## 2. Train target endpoint (90 requests/arm)

| arm | metric | mean | median | p90 | max | positive margin |
|---|---|---:|---:|---:|---:|---:|
| Z+ | new NLL | 0.977994 | 0.035024 | 4.088684 | 8.402637 | — |
| Z+ | target-true NLL | 10.793322 | 11.014881 | 15.607003 | 20.314486 | — |
| Z+ | new−true margin | 9.815328 | 10.108035 | 15.590626 | 20.310148 | 89/90 |
| Z± | new NLL | 1.382357 | 0.073352 | 5.902587 | 12.950777 | — |
| Z± | target-true NLL | 10.748158 | 11.168121 | 14.519860 | 20.040829 | — |
| Z± | new−true margin | 9.365802 | 10.127538 | 14.286766 | 20.035141 | 84/90 |

여기서 margin은 `NLL(target_true) − NLL(target_new)`와 동치인 new-minus-true log-prob margin이다. 양수면 target_new 우세다.

## 3. Accepted-z terminal rewrite/rephrase

이 값은 W를 쓰지 않은 terminal z-injection 관측이다. rewrite는 request당 1 prompt, rephrase는 request당 2 prompts다.

| arm | metric | new NLL mean/median/p90/max | margin mean/median/p90/max | prompt success | official accuracy | strict request success |
|---|---|---|---|---:|---:|---:|
| Z+ | rewrite | 0.916760/0.029762/3.845517/8.385878 | 9.948946/10.036323/15.553218/21.990140 | 89/90 | 89/90 | 89/90 |
| Z+ | rephrase | 2.001898/0.804904/4.677961/9.196800 | 7.709063/7.795613/11.901535/17.894484 | 172/180 | 172/180 | 84/90 |
| Z± | rewrite | 1.243781/0.066492/3.844919/13.036105 | 9.691266/10.277875/14.849351/21.455558 | 84/90 | 84/90 | 84/90 |
| Z± | rephrase | 2.340551/1.131983/6.439856/12.499746 | 7.502814/7.918476/11.953566/18.206622 | 167/180 | 167/180 | 81/90 |
| Native-Z | rewrite | 0.000850/0.000666/0.001706/0.006371 | 14.792931/14.529140/19.631737/27.224535 | 90/90 | 90/90 | 90/90 |
| Native-Z | rephrase | 1.246154/0.138178/3.714318/9.861938 | 9.916788/9.821234/14.477145/22.546459 | 174/180 | 174/180 | 86/90 |

Native-Z는 causal panel이 아니므로 Euler arm과 동일 schedule/optimizer effect로 해석하지 않는다.

## 4. Z±−Z+ paired delta

모든 delta는 동일 case/request에서 `Z± − Z+`다. 음의 new-NLL delta는 Z±가 더 낮은 new NLL임을 뜻하고, 양의 margin delta는 Z±가 더 큰 target-new 우세임을 뜻한다.

| panel | metric | n | mean | median | p90 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| train endpoint | delta_clamp_fraction_Zpm_minus_Zplus | 90 | 0.211111 | 0.200000 | 0.600000 | -0.400000 | 1.000000 |
| train endpoint | delta_train_new_nll_Zpm_minus_Zplus | 90 | 0.404362 | 0.002867 | 1.642276 | -7.287691 | 12.946825 |
| train endpoint | delta_train_true_nll_Zpm_minus_Zplus | 90 | -0.045164 | 0.287791 | 2.013293 | -10.602874 | 8.652051 |
| train endpoint | delta_train_margin_Zpm_minus_Zplus | 90 | -0.449526 | 0.206492 | 3.238757 | -17.885576 | 11.245689 |
| rewrite | delta_nll_new_Zpm_minus_Zplus | 90 | 0.327021 | 0.002027 | 1.616613 | -8.309130 | 12.483804 |
| rewrite | delta_nll_true_Zpm_minus_Zplus | 90 | 0.069341 | 0.149387 | 3.014498 | -8.185576 | 10.339700 |
| rewrite | delta_margin_Zpm_minus_Zplus | 90 | -0.257680 | 0.172259 | 3.557829 | -16.699494 | 11.983269 |
| rephrase | delta_nll_new_Zpm_minus_Zplus | 90 | 0.338653 | 0.013704 | 1.833579 | -8.029763 | 11.898974 |
| rephrase | delta_nll_true_Zpm_minus_Zplus | 90 | 0.132403 | 0.119515 | 2.140857 | -9.975743 | 10.101794 |
| rephrase | delta_margin_Zpm_minus_Zplus | 90 | -0.206250 | 0.081258 | 3.559107 | -14.420701 | 9.309006 |

Prompt/strict success의 paired 합 차이도 `analysis.json`과 `evaluator-paired-deltas.csv`에 기록했다. 이는 h=1 내부 arm 비교이며 h=.25와 합산하지 않는다.

## 5. Microstep trajectory

| arm | step | clamp | objective mean | train new NLL mean | train true NLL mean | margin mean | raw field norm mean | post-clamp displacement mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z+ | 1 | 32/90 (35.56%) | 10.256684 | 10.256684 | 5.062010 | -5.194675 | 4.671800 | 3.743975 |
| Z+ | 2 | 61/90 (67.78%) | 6.161895 | 6.073445 | 6.174495 | 0.101050 | 3.167823 | 4.265176 |
| Z+ | 3 | 71/90 (78.89%) | 2.983654 | 2.870970 | 7.877500 | 5.006531 | 3.225802 | 4.431040 |
| Z+ | 4 | 63/90 (70.00%) | 1.679579 | 1.553088 | 9.169692 | 7.616603 | 3.068012 | 4.445103 |
| Z+ | 5 | 59/90 (65.56%) | 1.619326 | 1.497691 | 9.707749 | 8.210058 | 2.189847 | 4.445161 |
| Z± | 1 | 73/90 (81.11%) | 15.845246 | 10.256684 | 5.062010 | -5.194675 | 9.103651 | 4.312417 |
| Z± | 2 | 82/90 (91.11%) | 8.683250 | 6.724486 | 6.933412 | 0.208926 | 4.133584 | 4.454880 |
| Z± | 3 | 82/90 (91.11%) | 3.851660 | 3.429384 | 8.928208 | 5.498823 | 4.006376 | 4.491138 |
| Z± | 4 | 74/90 (82.22%) | 2.440209 | 2.064185 | 9.454412 | 7.390227 | 4.096046 | 4.490046 |
| Z± | 5 | 70/90 (77.78%) | 1.994061 | 1.525447 | 10.176056 | 8.650608 | 3.147203 | 4.488412 |

`slice-microstep-summary.csv`에는 semantic/KL/decay, raw Euler movement, pre/post-clamp displacement, removed norm/energy, boundary outward/tangential component를 slice×arm×step으로 기록했다. `request-microsteps.csv`는 900개 request×microstep 행을 보존한다. post-projection incremental step norm은 `NOT_RECORDED`이며 추정하지 않았다.

## 6. Clamp–metric relation (descriptive only)

| arm | y | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| Z+ | endpoint_train_new_nll | 90 | 0.097979 | 0.113727 |
| Z+ | endpoint_train_true_nll | 90 | 0.012375 | 0.016362 |
| Z+ | endpoint_train_margin_new_minus_true | 90 | -0.038520 | -0.038929 |
| Z± | endpoint_train_new_nll | 90 | 0.028987 | 0.269783 |
| Z± | endpoint_train_true_nll | 90 | -0.082439 | -0.030992 |
| Z± | endpoint_train_margin_new_minus_true | 90 | -0.073124 | -0.072688 |

상관은 projection 포화와 endpoint 관측의 동시 변화를 요약할 뿐이다. request 난이도·radius·field 크기 등이 함께 달라지므로 인과효과로 주장하지 않는다.

### Clamp strata

| arm | hit steps | n | endpoint new NLL mean | endpoint margin mean | rewrite new NLL mean | rewrite strict | rephrase new NLL mean | rephrase strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z+ | 0/5 | 7 | 0.080246 | 9.780772 | 0.070228 | 7/7 | 1.424375 | 7/7 |
| Z+ | 1/5 | 5 | 0.050063 | 11.506917 | 0.041468 | 5/5 | 0.956280 | 5/5 |
| Z+ | 2/5 | 13 | 0.833759 | 10.720168 | 0.712879 | 13/13 | 1.546080 | 12/13 |
| Z+ | 3/5 | 24 | 1.174519 | 9.701679 | 1.083718 | 24/24 | 2.202705 | 21/24 |
| Z+ | 4/5 | 22 | 1.853106 | 7.950750 | 1.801326 | 21/22 | 3.003197 | 20/22 |
| Z+ | 5/5 | 19 | 0.390095 | 11.066343 | 0.363349 | 19/19 | 1.388656 | 19/19 |
| Z± | 0/5 | 1 | 0.047761 | 9.160267 | 0.029226 | 1/1 | 0.026942 | 1/1 |
| Z± | 1/5 | 1 | 0.016052 | 10.667019 | 0.009884 | 1/1 | 4.095864 | 1/1 |
| Z± | 2/5 | 7 | 0.153328 | 12.705737 | 0.148071 | 7/7 | 1.182009 | 7/7 |
| Z± | 3/5 | 12 | 1.106011 | 9.575324 | 1.180729 | 11/12 | 2.075221 | 11/12 |
| Z± | 4/5 | 15 | 3.763803 | 6.526128 | 3.552308 | 11/15 | 4.412699 | 10/15 |
| Z± | 5/5 | 54 | 0.991589 | 9.654795 | 0.803913 | 53/54 | 1.984436 | 51/54 |

각 stratum은 관측된 clamp hit step 수로 사후 분할한 기술 통계다. 표본 수가 불균형하고 배정이 무작위가 아니므로 clamp 효과 추정치가 아니다.

## 7. Compute와 empirical overhead

| arm | target solve mean/median/p90/max sec | evaluator mean sec | field eval total | autograd.grad total |
|---|---|---:|---:|---:|
| Z+ | 16.415654/16.243536/16.958014/17.224902 | 1.876175 | 45 | 45 |
| Z± | 21.553734/21.486559/21.969027/22.270392 | 1.872750 | 45 | 45 |

- Slurm: 9/9 COMPLETED exit0; cell elapsed mean 126.222222초, 범위 125–128초.
- 할당량 합: 1136 GPU-seconds, 9088 allocated CPU-seconds.
- case terminal peak GPU memory는 `compute.csv`에 있다. Slurm parent row MaxRSS는 `NOT_RECORDED_PARENT_ROW`이므로 GPU peak와 혼동하지 않았다.
- Native-Z compute는 canonical EasyEdit 내부 알고리즘이므로 Euler의 5 field eval과 동일 단위 overhead로 정규화하지 않았다.

## 8. 불변식과 receipt completeness

- 9 case terminals, 27 arm terminals, 18 target action-freeze receipts, 90 target microstep receipts 모두 rooted identity PASS.
- FULL-FP32, nonfinite 0, W pointer/version/bytes 변화 0, restore failure 0.
- optimizer/Adam/SGD/backward/parameter-grad/duplicate evaluation 0.
- writer/materialization/cache append/K8 repeat 0; heldout는 terminal-only.
- target autograd.grad 90회 = 9 slices × 2 arms × 5 steps. Native compute_z 90회 = 9 slices × 10 requests, selection influence 0.

## 9. h=.25 및 Stage2와의 경계

- h=.25, M5, T_z=1.25는 B1_CASE01 calibration-only에서 선택된 설정이다. 이번 h=1 B2–B10 denominator와 합치거나 동일 confirmatory claim으로 다루지 않는다.
- B1의 h=1 원 admissibility FAIL과 clamp Z+ 37/50, Z± 43/50은 그대로 남는다.
- Stage2 same-T latent endpoint refinement failure는 `FAILED_OBSERVED_NONDECISIONAL`로 보존된다. 이번 결과는 continuous-ODE convergence, step-size stability, unique trajectory를 입증하지 않는다.

## 10. 결론

h=1 target-only 실행은 기술적으로 완결되었고 terminal z-injection에서 낮은 new NLL과 높은 positive-margin 비율을 관측했다. 다만 Z±−Z+의 평균 delta는 train new NLL 0.404362, train margin -0.449526이고, rewrite/rephrase new NLL delta도 양수여서 aggregate상 Z± 개선으로 읽히지 않는다. 또한 업데이트의 상당 부분, 특히 Z±가 projection boundary에 포화되었다. 따라서 허용되는 결론은 봉인된 Llama B2–B10에서 이 discrete raw-projected Euler 설정의 factual target/z-injection 행동을 관측했다는 것뿐이다. writer/weight edit 성능, 연속 ODE 정당화, 자동 승격은 주장하지 않는다.

상태: `scientific_promotion=false`; `ZB=0`; `IDLE_AWAITING_GH_CALL`.
