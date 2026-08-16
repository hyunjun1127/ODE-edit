# P1R28 Corrected-Coupling Kill Test — Stage-B TECH-R5 최종 보고서

작성 시각: 2026-08-13 (Asia/Seoul)  
지시: `ODEEDIT-S05-P1R28-P1R24-ANCHORED-CORRECTED-COUPLING-KILL-TEST-V1`  
판정: `C1_STRENGTH_GATE_SCIENTIFIC_FAIL; P1R24_ATOMIC_FALLBACK_FROZEN; SCIENTIFIC_PROMOTION_FALSE`

## 요약

TECH-R5 Stage-B는 좌표·상태·물질화 기술 게이트를 모두 통과했다. P1R27의 약 9배 first-step gain은 재현되지 않았고, C1/C2의 첫 step 에너지와 gain은 같은 sealed B10의 P1R24 A0와 정확히 일치했다. C2는 두 모델의 모든 step에서 `lambda=1`이 feasible였고, tracking 때문에 semantic write가 stall한 사례는 없었다.

그러나 C1의 Llama W-only efficacy target-new NLL은 P1R24 A0의 `0.021823`에서 `0.222251`로 증가했다(동일한 E/G count `10/20`이지만 약 10.2배). 이는 계약이 요구한 ``P1R24 strength region에서 material한 continuous target-new NLL 악화 없음``을 만족하지 못한다. 수치 margin을 새로 만들지 않고, 관측된 절대 NLL과 약 10배의 직접 대비만으로 C1 strength gate를 `SCIENTIFIC_FAIL`로 분류한다. 따라서 계약의 stop rule에 따라 P1R27 joint/coupling branch는 여기서 종료하고 P1R24 Atomic fallback을 동결한다. C2 RS-Soft는 **ELIGIBLE_NEXT가 아니며 제출하지 않았다**.

## 1. 범위·계보·비교 경계

| Role | 과학/실행 계보 | 상태 |
| --- | --- | --- |
| A0 | P1R24 RS-Neutral immutable exact reuse, scientific head `ce8c6c36348752f1407f7d713d30e6b5c727379b` | 같은 seal/order/W0/evaluator 확인 후 재사용; 새 A0 model run 0 |
| C0 | P1R27 negative reference, checkpoint `56b744449a5bdd44b9e0798ae3fa486962dbdc22`, report SHA `f1b057…30fa09` | 재실행하지 않음; 첫-step gain 9.28–9.54 음성 대조만 사용 |
| C1 | P1R28 semantic-only corrected coupling, `d=Δz`, lag scale 0 | 새 TECH-R5 B10 실행 |
| C2 | P1R28 feasibility-bounded lag coupling, 최대 feasible lambda | 새 TECH-R5 B10 실행 |

- P1R28 TECH-R5 source head: `8a8de0e7b4c1666a7bc22d02468410e97f14eeec`; tree `9b75c5a723acdece6bc9af3d9eae513c05be2e0f`.
- P1R28 base: `ce8c6c36348752f1407f7d713d30e6b5c727379b`; TECH-R5 numerical lock SHA: `b9a47600032ebeb6c24361be25e828d6dec0360134089bae44aa420e9dbfbdfd`.
- Exact sealed input: seal `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`, order `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`, B10, RS/Neutral, Atomic, `K=8`, `h=1/8`.
- Stage-B job: `19179`, array `0-1%2`; 2 GPU maximum for this task, 2 jobs × 2 corrected arms = 4 trajectories. Both tasks terminal exit `0:0`.
- C0의 stall 설명은 append-only correction SHA `ccde37…a07`을 적용한다. 즉 P1R27 raw behavior는 W만 hold하고 z는 계속 진행한 것이며, P1R28의 final stall은 W와 z를 함께 hold하도록 별도 검사했다.

이 보고서는 raw-free terminal/manifest/action-freeze/accepted-step/anchor receipts, P1R28 code와 immutable P1R24/P1R27 raw-free report만 읽었다. 프롬프트·타깃 텍스트·텐서·사설 로그를 읽지 않았고, 모델/evaluator/Slurm을 실행하지 않았다.

## 2. 무결성 및 실행 총체성

| Alias | terminal SHA-256 | manifest SHA-256 | action-freeze SHA-256 | manifest→terminal 재해시 | W0 pointer/bytes restore |
| --- | --- | --- | --- | --- | --- |
| Llama | `4d62fb24…5c01e1` | `ca0c7215…eac17e` | `4c91342d…099cd7` | PASS | PASS |
| Qwen | `dc616536…1bb468` | `3b5903b5…b7c96f` | `42ca18f7…3fbe3e` | PASS | PASS |

두 terminal은 source head, B10 count와 action-freeze SHA를 보유하고, 각 manifest의 `terminal_sha256`는 로컬 terminal 파일 SHA와 정확히 일치했다. 모든 corrected arm은 `tau=1`, accepted update 8, materialization 8, dynamic field refresh 8, scientific-invalid 0으로 종결했다.

- `actions_frozen_before_heldout=true`; inner-step held-out access 0.
- persistent commit/history append/replay-H/sequential-controller influence 모두 0.
- retry/backtracking/imputation/candidate materialization은 0이다.
- C2 feasibility solve가 추가한 model forward/backward/materialization은 0이다.

## 3. 좌표·강도·first-step scale 검증

모든 32 corrected transition(C1/C2 × 8 × 2 alias)은

\[
(h\bar a)^T v=\bar a^T(hv),\quad \theta=hv
\]

을 그대로 기록했다. ratio는 모두 1, coordinate residual은 모두 0, `h_application_count=1`, `second_h_division_count=0`, 금지 contraction count 0이다. P1R27의 per-layer full-residual 복제는 0이며, layer routing은 한 개의 matched demand만 배분했다.

| Alias | Arm | first gain | A0 first gain | first energy | A0 first energy | A0 energy ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama | C1 | 2.118537 | 2.118537 | .115367 | .115367 | 1.000000 |
| Llama | C2 | 2.118537 | 2.118537 | .115367 | .115367 | 1.000000 |
| Qwen | C1 | 1.927838 | 1.927838 | 1.150102 | 1.150102 | 1.000000 |
| Qwen | C2 | 1.927838 | 1.927838 | 1.150102 | 1.150102 | 1.000000 |

따라서 P1R27의 9.28–9.54 first-step gain failure는 0/4로 제거되었다. 이는 P1R28 C1/C2의 기술·기전 성공이며, endpoint efficacy 성공을 자동으로 뜻하지는 않는다.

## 4. C2 lambda, reachability 및 stall

| Alias | Arm | status | lambda trajectory | `r_max` range | `rho` range | min coverage | max equality residual |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Llama | C1 | JOINT_WRITE 8/8 | 0×8 | .755818–22.045170 | .290186–4.830061 | 1 | 4.44e-16 |
| Llama | C2 | JOINT_WRITE 8/8 | 1×8 | 1.612136–22.045170 | .301173–4.830061 | 1 | 2.22e-16 |
| Qwen | C1 | JOINT_WRITE 7/8; SEMANTIC_STALL 1/8 | 0×8 | .415328–49.630638 | .603081–10.771235 | 1 | 4.44e-16 |
| Qwen | C2 | JOINT_WRITE 8/8 | 1×8 | 4.442876–49.630638 | 1.243319–10.771235 | 1 | 4.44e-16 |

- C2의 모든 step에서 full-lag endpoint `lambda=1`이 feasible였다. `TRACKING_CONFLICT_RECTIFIED=0`; 따라서 lambda를 줄여야 하는 conflict branch는 `NOT_RECORDED`다.
- Qwen C1 k8은 `r_max=.415328 < rho=3.320305`로 `SEMANTIC_STALL`이었다. 이는 C1의 pure semantic path에서 발생한 boundary이고 tracking이 원인이 아니다.
- 해당 stall은 `W`, controller z 및 physical terminal z의 before/after hash가 각각 동일했다. W/z hold가 구현된 scientific totality이며 technical error는 아니다.
- positive pure-semantic slope가 있는데 tracking이 stall을 유발한 case는 0이다. C2는 Qwen에서 마지막 C1 stall이 있었던 이후의 서로 다른 current state에서도 끝까지 8 write를 유지했지만, trajectory가 이미 달라진 뒤의 관측이므로 이를 same-state causal proof로 해석하지 않는다.

## 5. Official endpoint 결과

`E/G/L`은 W-only frozen official evaluator의 correct count다. `new`는 각 metric의 mean target-new NLL, `margin`은 mean margin이다. A0는 exact immutable P1R24 RS-Neutral reference다.

| Model | Arm | E/G/L | Eff new / margin | Gen new / margin | Loc new / margin | z8 oracle new | terminal Structural-P |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama | A0 | 10/20/82 | .021823 / 11.521927 | .961012 / 7.837425 | 8.311719 / 4.001926 | .039689 | .003428 |
| Llama | C1 | 10/20/80 | .222251 / 10.968374 | 1.405054 / 7.776196 | 8.283906 / 3.910833 | .044447 | .001630 |
| Llama | C2 | 10/20/81 | .029996 / 11.332504 | 1.084636 / 7.645052 | 8.313281 / 3.995488 | .043892 | .003422 |
| Qwen | A0 | 10/19/80 | .538097 / 11.433778 | 2.404022 / 7.424884 | 8.829219 / 3.655415 | .048375 | .253839 |
| Qwen | C1 | 10/20/80 | .399302 / 12.328823 | 2.475327 / 8.541861 | 8.830625 / 3.661689 | .380461 | .108942 |
| Qwen | C2 | 10/20/80 | .072930 / 13.967695 | 2.014288 / 8.474774 | 8.845625 / 3.672261 | .080505 | .184933 |

### A0 대비 직접 관찰

- Llama C1: E/G count는 A0와 같지만 efficacy new NLL `+.200428` (약 10.2×), Gen new NLL `+.444041` (약 1.46×), Loc `-2`다.
- Llama C2: E/G count는 A0와 같고, efficacy/Gen new NLL은 각각 `+.008173`, `+.123624`; Loc `-1`이다.
- Qwen C1: E는 같고 G는 `+1`; efficacy new NLL은 `.138794` 낮지만 Gen new NLL은 `+.071304`; Loc은 같다.
- Qwen C2: E는 같고 G는 `+1`; efficacy/Gen new NLL은 각각 `.465167`, `.389734` 낮고 Loc은 같다.

Structural-P 차이는 각기 다른 C1/C2 trajectory의 terminal observation이다. Soft arm이나 same-state preservation objective가 없는 Neutral-only test이므로, P 값의 감소를 P1R28의 preservation gain으로 해석하지 않는다.

## 6. W-only realization과 비용

| Model | Arm | delayed actual / predicted sum | mean realization ratio | nonpositive actual | edit-core s | terminal evaluator s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama | C1 | 8.2550 / 13.0504 | .5392 | terminal k8 1 | 170.536 | 1.079 |
| Llama | C2 | 8.2631 / 13.0890 | .5219 | 0 | 220.634 | 1.081 |
| Qwen | C1 | 9.0127 / 30.8245 | .2725 | delayed k5 1; k8 semantic stall | 175.118 | .991 |
| Qwen | C2 | 9.1446 / 32.2295 | .2269 | delayed k5 1 | 253.927 | .991 |

각 corrected arm의 source-accounted compute는 `180 model forwards`, `125 autograd/backward`, `8 materializations`, `36 logical groups`다. Processed tokens는 Llama `27,207`, Qwen `24,687`이다. full-six field/physical slope가 wall time의 주 부분이고 routing solve는 수 ms 수준이다. C2 lambda certification은 추가 model F/B/materialization 0이다.

Terminal receipt에는 authoritative GPU peak 또는 분리된 model-load wall이 없으므로 `NOT_RECORDED`다. Native를 재실행하지 않았으므로 formal Native ratio도 산출하지 않는다.

## 7. 판정

### FACT

1. TECH-R5 Stage-B의 4 corrected trajectories는 모두 `K8/tau1`, action freeze, terminal evaluator 1회, W0 pointer/byte restore를 통과했다.
2. Coordinate/h identity, no-double-h, one total matched demand, P1R27 gain 제거, C2 no-extra-model-call gates가 모두 PASS다.
3. C2는 양 모델에서 모든 step `lambda=1`로 feasible였고, tracking conflict로 lambda가 축소되거나 semantic write가 stall한 사례는 0이다.
4. C1 Llama는 A0와 동일한 E/G count에도 efficacy target-new NLL이 `.021823→.222251`로 증가했다. 이 값은 raw-free terminal evaluator receipt의 직접 비교다.
5. Qwen C1 k8 `SEMANTIC_STALL`은 lambda=0 pure semantic path에서 `r_max<rho`여서 발생했으며, W와 z 둘 다 hold했다.

### SCIENTIFIC_FAIL

`C1_STRENGTH_GATE_SCIENTIFIC_FAIL`이다. Llama C1의 continuous efficacy target-new NLL이 P1R24 A0보다 약 10.2배 높아져, count가 같다는 사실만으로 P1R24 strength region 유지라고 볼 수 없다. 이 판정에는 새 threshold, tolerance, sample 또는 outcome-tuned rule을 추가하지 않았다.

계약의 stop rule을 적용한다.

1. P1R27 joint/coupling branch를 종료한다.
2. P1R24 Atomic fallback을 동결한다.
3. C2 결과는 mechanism/lag feasibility 관측으로만 보관한다. C1 gate가 실패했으므로 C2 RS-Soft는 `ELIGIBLE_NEXT`가 아니며 실행하지 않는다.
4. Historical, B100, BG, Soft 또는 후속 model/GPU/Slurm action은 이 보고서로 열리지 않는다.

### TECHNICAL_FAIL

최종 TECH-R5 Stage-B에 technical failure는 없다. 이전 19162/19164 및 TECH-R2–R4 failed roots/receipt는 immutable로 보존되어 있으며, TECH-R5 endpoint에 재사용되지 않았다.

### INFERENCE

- P1R28은 P1R27의 coordinate-scale pathology와 ``stall에서 z가 계속 진행``하는 문제를 해결했다. 이는 corrected coupling totality의 기술·기전 성공이다.
- 그렇지만 C1의 Llama continuous strength degradation 때문에, 이 한 번의 Neutral-only kill test는 coupling branch의 endpoint strength 회복을 지지하지 않는다.
- C2의 Qwen endpoint 개선과 full-lag feasibility는 흥미로운 trajectory observation이지만, 실패한 C1 gate를 override하거나 후속 arm을 정당화하지 않는다.

### NOT_RECORDED

- `TRACKING_CONFLICT_RECTIFIED`가 실제로 필요한 state의 behavior (이번 run에서는 lambda=1이 항상 feasible).
- C2 RS-Soft, BG, Historical/sequential/B100/fresh-sample 결과.
- formal Native ratio, authoritative GPU peak, 분리 model-load wall.
- P1R28의 general preservation 또는 long-horizon benefit.

## 8. 주장 경계와 artifact

이 결과는 outcome-selected 재사용 sealed B10의 Atomic, RS, Neutral-only one-shot kill test다. 결과는 mechanistic/descriptive이며 `scientific_promotion=false`다. CBF/safe-set, exact tracking, universal preservation, Soft 효과, Historical readiness 또는 일반화된 coupling claim을 하지 않는다.

주요 artifact:

- Llama terminal: `local/odebf/results/p1r28-tech-r5/s05-p1r28-llama3-8b-inst-b10-rs-c1-c2-pair-v1/terminal.json`
- Qwen terminal: `local/odebf/results/p1r28-tech-r5/s05-p1r28-qwen2.5-7b-inst-b10-rs-c1-c2-pair-v1/terminal.json`
- Stage-A TECH-R5 report: `local/odebf/reports/p1r28-corrected-coupling-v1/stage-a-tech-r5-analysis/p1r28-stage-a-tech-r5-mechanism-report-ko.md` (SHA `21af6d…99ebac`)
- P1R27 original report: SHA `f1b05789437ebf30e8c93ac07ea2c1bd6b51c56868f90f1ade96cc178f30fa09`
- append-only P1R27 correction: SHA `ccde37b5b5805484088ac61984613a8fc27530861d49583c4ea05f53a6309a07`
