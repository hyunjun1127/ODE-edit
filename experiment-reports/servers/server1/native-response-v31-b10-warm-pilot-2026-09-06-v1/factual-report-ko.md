# Native-response ODE v3.1 — B10 short-history mechanism pilot 최종 사실 보고서

상태: **REVIEW_READY / GH 독립 코드·산출물 검토 대기**. scientific_promotion=false.

본 실험은 Official D10A(B10) 한 번으로 만든 공통 warm entry에서 D10B(B10)를 독립 편집한 짧은 history pilot이다. B9 replay 또는 lifelong 재현이 아니다. Server4 진행 retention rerun의 source/process/cache/checkpoint/result를 변경하거나 그 결과를 사용하지 않았다.

## 1. 핵심 성능·action·비용

RS/PS는 target-new NLL < target-true NLL, NS는 neighborhood target-true NLL < target-new NLL이다. 모든 tie는 실패다. RS 분모10, PS20, NS100/arm/cell이며 old loss 분모는 D10A warm-entry 성공 요청만이다. NLL은 token 평균이다. `PRE_EDIT_WARM`·target-new/true 분포·strict coverage·request-paired 상세표는 [primary 보고서](primary/factual-report-ko.md)에 별도 기록했다.

| cell | arm | RS | PS | NS | NLL | old_loss | V_ratio | QN_net | core_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-MEMIT | O_NATIVE | 9/10 | 17/20 | 90/100 | 0.2828978858393384 | 0/10 | 0.1129406385880372 | 0.17086997891460193 | 41.92959468066692 |
| Llama-MEMIT | ORBFH_HIST | 10/10 | 18/20 | 90/100 | 0.10024637755705043 | 0/10 | 0.13455534377961262 | 0.16825032250172184 | 172.11152659915388 |
| Llama-MEMIT | JV_NATIVE | 10/10 | 18/20 | 90/100 | 0.007166834984673187 | 0/10 | 0.008644859078070584 | 0.5922725590461556 | 197.12174249999225 |
| Llama-MEMIT | ORB_RAY_N | 10/10 | 18/20 | 90/100 | 0.010499974258709698 | 0/10 | 0.05920142463147326 | 0.2780541495452221 | 208.10002791509032 |
| Llama-AlphaEdit | O_NATIVE | 10/10 | 18/20 | 86/100 | 0.0008299750235892134 | 0/10 | 0.0010871945557260078 | 0.1264624461847209 | 33.36494185589254 |
| Llama-AlphaEdit | ORBFH_HIST | 10/10 | 18/20 | 86/100 | 0.00717485586865223 | 0/10 | 0.045307662718103034 | 0.0696532427704089 | 228.3239911980927 |
| Llama-AlphaEdit | JV_NATIVE | 10/10 | 20/20 | 85/100 | 0.0017379082077241038 | 0/10 | 0.004608298341466849 | 0.16648726129096397 | 144.77473674714565 |
| Llama-AlphaEdit | ORB_RAY_N | 10/10 | 18/20 | 85/100 | 0.0019747731814277357 | 0/10 | 0.021263247980957516 | 0.09061267009462705 | 154.80512097850442 |
| Qwen-MEMIT | O_NATIVE | 10/10 | 17/20 | 88/100 | 0.02273912318632938 | 0/10 | 0.018687143931283 | 0.10998621453177265 | 71.51967075653374 |
| Qwen-MEMIT | ORBFH_HIST | 10/10 | 18/20 | 88/100 | 0.02207605162402615 | 0/10 | 0.02642050877884341 | 0.1155481778113835 | 253.10079197771847 |
| Qwen-MEMIT | JV_NATIVE | 10/10 | 17/20 | 88/100 | 0.019593577925115825 | 0/10 | 0.007006493670237749 | 0.3951363448710385 | 353.82796165905893 |
| Qwen-MEMIT | ORB_RAY_N | 10/10 | 18/20 | 88/100 | 0.02230460859136656 | 0/10 | 0.034250788721230496 | 0.15664770111525936 | 392.8418708052486 |
| Qwen-AlphaEdit | O_NATIVE | 10/10 | 17/20 | 89/100 | 0.024586987151997163 | 0/10 | 0.00010052582510455886 | 0.04567736767585738 | 38.40893394686282 |
| Qwen-AlphaEdit | ORBFH_HIST | 10/10 | 17/20 | 88/100 | 0.02518622565548867 | 0/10 | 0.01151337980747895 | 0.033988235026963416 | 270.8146279975772 |
| Qwen-AlphaEdit | JV_NATIVE | 10/10 | 17/20 | 90/100 | 0.027896269515622407 | 0/10 | 0.004370283359631236 | 0.1020799935849007 | 199.20371014997363 |
| Qwen-AlphaEdit | ORB_RAY_N | 10/10 | 17/20 | 87/100 | 0.026387392182368786 | 0/10 | 0.02039467567507225 | 0.04140477910901827 | 208.0587241537869 |

## 2. Mechanism 판정: 서로 다른 질문을 합치지 않음

1. 같은 state의 joint-vs-ray native Rturn은 32/32 finite comparison에서 양수다. 단순 ray 감속만으로 동일 physical velocity가 된다는 가설은 이 기록과 맞지 않는다.
2. 실제 JV/ray trajectory의 node readout, V/V0 및 materialized endpoint native action도 다르다. 그러나 cross-arm dense endpoint distance와 Frobenius angle은 저장되지 않았으므로 그 수치는 만들지 않았다.
3. JV는 ray보다 네 cell 모두 native endpoint action이 크다. 더 작은 latent residual을 더 적은 action/해로움과 혼동하지 않는다. Primary D10A 신규 망각은 모든 arm 0/10이다. 이 표만으로 retention 우월성을 입증할 수 없다.
4. Primary JV의 rephrase target-new 평균 NLL은 네 cell 모두 ray보다 높다. 일부 PS/NS 이득과 평균 NLL 결과를 함께 보고하며 일반적인 성능 우월성을 주장하지 않는다.

| cell | native_Rturn_mean | JV_ray_native_action_ratio | JV_ray_rewrite_nll_delta | JV_ray_rephrase_nll_delta | JV_Official_core_time_ratio |
| --- | --- | --- | --- | --- | --- |
| Llama-MEMIT | 0.5283784076839181 | 2.130061932234641 | -0.003333139274036511 | 0.11808245927095395 | 4.701255616737025 |
| Llama-AlphaEdit | 0.36912289565104417 | 1.837350793405612 | -0.00023686497370363187 | 0.05740957386151413 | 4.3391275001301155 |
| Qwen-MEMIT | 0.5314592937258023 | 2.522452241927906 | -0.002711030666250735 | 0.084872038732283 | 4.947281746633805 |
| Qwen-AlphaEdit | 0.476314322748041 | 2.465415726917063 | 0.0015088773332536214 | 0.08985934709198773 | 5.186389979622027 |

## 3. 수학·fidelity와 경계

Native metric/qN_ref/normalization은 entry에서 고정하고 매 node native full-residual direction과 JVP를 current joint state에서 재관측했다. NNLS는 nonnegative FP64 active-set reference, physical forward/overlay는 FP32이며 MEMIT stock ephemeral FP64 solve를 보존했다. G0 B1 두 fixture×4 cells=8/8 PASS; CPU algebra와 source/transaction 검증은 component receipts에 결속했다.

JV와 ORB_RAY_N은 T2/N4/h.5, historical ORBFH는 원래 ordered T1/N4/h.25이다. T 숫자를 equal exposure로 해석하지 않는다. Infinitesimal native-action dissipation 관계는 locality/retention/finite-step monotonicity 보장이 아니다.

## 4. Refinement·H10 audit

[후속 진단 보고서](diagnostics/diagnostic-factual-ko.md)는 D2 cold DEV B1 fixed-T N2/4/8의 실제 block endpoint/activation 거리와 H10의 independent warm 평가를 분리한다. 모든 미실행·실패 상태는 해당 run_registry/audit table에 기록한다. N 선택, interpolation, budget expansion 또는 H10 결과로 primary 변경은 없다.

## 5. 계산량·예산·한계

총 GPU 점유시간은 기술 시도/G0/model load/diagnostic/CPU 관측 중 GPU reservation까지 포함하여 **5.058889 GPUh**다. cell별 누적 seconds=[4064, 3491, 6325, 4332]; 각각7200s, 총28800s 이내다. 동시성 override는 cap4이며 총 예산 확대가 아니다. job별 상세는 gpu-hour-ledger.json이다.

Arm core wall은 evaluator callback을 제외하되 dictionary/solve/shadow/transaction을 포함한다. Setup/target/model load는 별도다. 실제 model.forward/JVP 수를 제공하며 FLOPs로 환산하지 않았다. Peak GPU memory는 arm마다 reset한 값이 아니라 process 누적 high-water다. N0 clipping·gain tuning·새 reference bank·controller old-prompt access=0. Intermediate matched-progress endpoint 평가 및 setup backward ledger 누락은 primary/missing_fields.csv에 명시했다. 추가 telemetry-only rerun은 하지 않았다.

## 6. Provenance와 GH 독립 검토

Baseline report ref=f2dcfd4ab6fcb2917ae0a29cbc384cf95a82eb3c이며 historical execution HEAD와 구분한다. LM-ORBFH B9/LM-JAC B6/LA-JAC B5의 exact historical state는 입력에 없어 HISTORICAL_STATE_UNAVAILABLE, replay0이다.

Execution source=29884f208bca5afc2367c67510779b4a674cbbb1 / tree=5c7310d74357461f3040cf22342282b69cda7baf. Raw source/fixture/node/endpoint 경로, 실행 대 분석 branch/worktree, code/test 파일과 locks, component SHA/member roots는 REVIEW_READY.json에 있다. 초기 local commits는 공유 clone의 GH author 설정을 상속했으며 이는 GH 코드 검토 서명이 아니다. 최종 SH1 worktree는 별도 server-head identity/access gate를 적용했고 공유 GH 설정·기존 commit identity는 변경하지 않았다.

Git 포함은 reusable code/tests와 raw-free tables/reports/locks/manifests/receipts뿐이다. Raw prompts/logs/model/weights/tensors/cache/checkpoint/dataset/credentials는 제외했다. GH 독립 검토 전 scientific_promotion=false를 유지한다.
