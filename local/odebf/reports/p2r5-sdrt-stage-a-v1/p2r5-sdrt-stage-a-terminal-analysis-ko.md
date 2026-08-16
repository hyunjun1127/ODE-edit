# P2R5 SDRT Stage-A 8-endpoint 종단 분석 보고서

- 작성 시각(Asia/Seoul): `2026-08-15T22:27:29+09:00`
- 계약: `/mnt/raid5/janghj/.codex/attachments/409d3a84-d321-4d00-8479-9075f9a63ff2/pasted-text.txt` / SHA256 `55cc177ecee86750bde732c6f880ea480696269da856495829437c9b59cc669d` / 15003 bytes / 485 lines.
- Stage-A 분모: endpoint 8/8, writer step 64/64, request-step 640/640. 유효 endpoint만 사용했고 실패 prefix는 endpoint로 사용하지 않았다.
- Canonical: P2R4 Clamp ON=`CLAMP_ON_CANONICAL_LOCKED`; Clamp OFF=`SCIENTIFIC_FAILURE_PRESERVATION`이며 재실행/비교 rescue 0.
- 최종 gate: **SCIENTIFIC_FAIL**. Stage B는 `CLOSED_PENDING_GH_STAGE_A_REVIEW`이다.

## FACT — endpoint

| model | case | arm | W Eff/Gen/Loc | z Eff/Gen | z/W full6 NLL | capacity | P | Δcap vs P2R4 | ΔLoc vs P2R4 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 3 | SDRT-CAP | 10/19/87 | 10/19 | 0.000263902/0.0002752 | 38.401289 | 0.057000 | -1.488125 | +5 |
| llama3-8b-inst | 3 | SDRT-STRUCTP | 10/20/87 | 10/19 | 0.899222/0.882306 | 39.542795 | 0.059759 | -0.346619 | +5 |
| llama3-8b-inst | 5 | SDRT-CAP | 9/19/80 | 9/19 | 1.16193/1.11135 | 62.325160 | 0.079516 | -6.861066 | +8 |
| llama3-8b-inst | 5 | SDRT-STRUCTP | 10/19/79 | 10/19 | 0.453106/0.600193 | 70.666651 | 0.087857 | +6.404355 | +8 |
| qwen2.5-7b-inst | 1 | SDRT-CAP | 10/19/92 | 10/19 | 0.000691883/0.000730956 | 433.102459 | 4.021854 | +292.836251 | -1 |
| qwen2.5-7b-inst | 1 | SDRT-STRUCTP | 10/19/92 | 10/19 | 0.000652816/0.000641789 | 478.775640 | 4.458374 | +338.509432 | -1 |
| qwen2.5-7b-inst | 4 | SDRT-CAP | 10/18/96 | 10/18 | 0.000799663/0.000825595 | 352.069038 | 3.395444 | +89.588053 | +0 |
| qwen2.5-7b-inst | 4 | SDRT-STRUCTP | 10/18/95 | 10/18 | 0.00082001/0.000843948 | 356.312767 | 3.465873 | +93.831782 | -1 |

P2R4 비교는 동일 alias/case에서 SDRT-CAP→Clamp-ON NEUTRAL, SDRT-STRUCTP→Clamp-ON SOFTP를 사용했다. evaluator/stream/order identity는 transferred canonical identity와 일치한다.

## FACT — pilot gate

| gate | status | pass/denominator | evidence |
|---|---|---:|---|
| technical_integrity_W0_action_freeze | PASS | 8/8 | terminal/manifest/action-freeze/W0 hashes |
| coefficient_mass_in_0_1 | PASS | 640/640 | min=0.000644964922627;max=1.00000000003 |
| capacity_reduced_vs_matched_P2R4_clamp_on | FAIL | 3/8 | matched alias/case and CAP->NEUTRAL, STRUCTP->SOFTP |
| W_Eff_Gen_not_lower_than_own_z_counts | PASS | 8/8 | terminal correct-count arithmetic |
| Llama_case3_5_Loc_improved | PASS | 4/4 | matched clamp-ON arm deltas |
| Qwen_Loc_non_degradation | FAIL | 1/4 | matched clamp-ON arm deltas |
| STRUCTP_allocation_differs_from_CAP | PASS | 4/4 | at least one paired-step L2 distance >1e-8 per case |
| STRUCTP_predicted_P_decreases_on_same_semantic_state | FAIL | 0/4 | same-prestate comparisons exist only at k0; later paths diverge |
| silent_neutral_fallback_zero | PASS | 64/64 | fallback_total=0 |
| clamp_off_rescue_zero | PASS | 64/64 | clamp_off_access_total=0 |
| pilot_branch_A_capacity_not_reduced | ACTIVATED | 5/8 | cause among contract alternatives NOT_IDENTIFIED |
| pilot_branch_B_capacity_down_but_W_count_below_z | NOT_ACTIVATED | 0/8 | W Eff/Gen correct counts never below own z |
| pilot_branch_C_barrier_inert | NOT_ACTIVATED | 4/4 | all four cases have allocation movement >1e-8 |
| pilot_branch_D_P_down_without_Loc_capacity_gain | NOT_IDENTIFIABLE | 0/4 | same-state P reduction absent at k0; later paths are not same-state comparisons |
| P2R5_STAGE_A | SCIENTIFIC_FAIL | 7/10 | Stage B remains CLOSED_PENDING_GH_STAGE_A_REVIEW |

## INFERENCE — 계약 분기

- `A`: ACTIVATED. Matched P2R4 Clamp-ON 대비 capacity 감소는 3/8 endpoint였다. 증가 endpoint는 Llama case5 STRUCTP, Qwen case1 CAP/STRUCTP, Qwen case4 CAP/STRUCTP의 표 양수 Δcap 항목이다. coefficient/cumulative quadratic 결함과 full-mass 유사 동작 중 직접 원인은 이 raw-free 분석에서 `NOT_IDENTIFIED`이다.
- `B`: NOT_ACTIVATED by terminal correct-count gate. Capacity가 감소한 3개 endpoint에서도 W Eff/Gen correct count가 자기 z-inject count보다 낮은 경우는 0/3이었다.
- `C`: NOT_ACTIVATED. 네 case 모두 적어도 한 step에서 CAP−STRUCTP allocation L2가 1e-8을 초과했다.
- `D`: NOT_IDENTIFIABLE. 동일 pre-route state 비교는 k0에만 존재하고 그 지점의 predicted-P 감소는 0/4였다. k1 이후 arm state가 달라져 cross-arm P 차이는 same-state proxy 판정에 사용하지 않았다.
- Stage-A scientific gate는 capacity 3/8, Qwen Loc non-degradation 1/4, same-state predicted-P reduction 0/4로 `SCIENTIFIC_FAIL`; Stage B는 닫힌 상태를 유지한다.

## FACT — 무결성·compute·금지 영향

- 8/8 case manifest→terminal SHA, action-freeze SHA, enclosing job manifest→terminal SHA가 일치했다.
- W0 pointer/bytes 8/8, action-freeze 8/8, heldout controller access 0, target microstep 24/endpoint, writer transition/materialization 8/endpoint.
- coefficient mass range: `0.000644964922627`–`1.00000000003`; fallback total `0`; clamp-OFF access total `0`.
- compute totals: target F/B `1000/960`, KL F/B `1000/960`, response F/VJP `320/320`, materializations `64`.

## FACT — Official AlphaEdit reference

Official identity SHA256 `e12b19ff941f39a594d2ff0f8718849d102644fd2ec79f53ccb6d7efc9fd8d3c`; selected 4 model/case identities are matched. endpoint JSON/CSV에 Eff/Gen/Loc와 산술 delta를 기록했다. Official continuous Gen/Loc와 z/W/P/capacity/energy는 `NOT_RECORDED`이다.

## TECHNICAL_FAIL — 보존된 attempt 이력

- 19989: 0/8 valid; semantic-face PSD quadratic backend technical failure; W0 8/8.
- 19995: 0/8 valid; nonexistent telemetry field mapping technical failure; W0 8/8.
- 20000: 2/8 valid, 6 technical failures; valid 2개는 재실행하지 않았다; W0 8/8.
- 20006: 0 scientific/model actions; CLI `P2R5_ARMS` import first-false gate.
- 20015: missing-only 2/6 valid, 4 technical solver-coordinate failures.
- 20021: missing-only 3/4 valid, 1 convex P-tie certification failure.
- 20029: remaining one endpoint valid. 각 repair receipt는 기존 roots/logs를 immutable로 보존한다.

## NOT_RECORDED

- stepwise heldout Eff/Gen/Loc: NOT_RECORDED (계약상 terminal-only).
- same-state STRUCTP-vs-CAP P candidate delta at k1..k7: NOT_RECORDED; arm states diverged.
- Official continuous Gen/Loc, z/W gap, Structural-P/capacity/energy: NOT_RECORDED.
- peak GPU/host memory authoritative scalar: NOT_RECORDED in selected case terminals.

## 산출물 경계

모든 표는 raw-free scalar/hash/status만 포함한다. raw prompts/targets/generations/tensors/weights/context templates/private-target/runtime logs는 읽거나 복사하지 않았다. `scientific_promotion=false`.
