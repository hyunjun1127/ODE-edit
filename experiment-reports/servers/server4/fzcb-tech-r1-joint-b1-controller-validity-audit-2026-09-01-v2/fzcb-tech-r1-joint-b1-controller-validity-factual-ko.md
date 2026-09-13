# FzCB TECH-R1 Llama/Qwen 공동 B1 controller-validity 감사 — 사실 보고서

## 최상단 판정 경계

- 이전 `HOLD_AFTER_QWEN_B1_LOCAL_INFEASIBILITY`는 최종 과학 결론으로 재사용하지 않았다.
- `tau_range/tau_null/tau_z/tau_cbf/tau_budget/tau_grad`는 단위를 분리해 B1 전에 봉인했다.
- full exact sensitivity가 없는 상태의 2/8/32/128 sketch는 full infeasibility를 선언하지 못하며, 해당 경우 `SUBSPACE_INCONCLUSIVE`만 허용한다.
- B10 제출=0, main push=0, scientific promotion=false.

## 공동 핵심 표

| 모델 | A0 | 초기 psi0 | ||g_free||² | psi_min | min hcc/A0 | FZCB action/A0 | static action/A0 | typed status |
|---|---|---|---|---|---|---|---|---|
| llama3-8b-inst | 0.00017695863789413124 | 1.6141880557914785e-05 | NOT_RECORDED | NOT_RECORDED | 0.0 | NOT_RECORDED | NOT_RECORDED | NUMERICAL_CONTROLLER_FAILURE |
| qwen2.5-7b-inst | 0.16463251411914825 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NUMERICAL_CONTROLLER_FAILURE |

## fixed-z 및 authority 규모

| 모델 | ||a0|| | ||delta*|| | ||z*|| | ||delta*||/||a0|| | coef dim | output dim | null dim | sensitivity authority |
|---|---|---|---|---|---|---|---|---|
| llama3-8b-inst | 6.152595520019531 | 4.614446640014648 | 7.467119216918945 | 0.75 | 20480 | 4096 | 16384 | NOT_RECORDED |
| qwen2.5-7b-inst | 67.87109375 | 173.4636993408203 | 184.56028747558594 | 2.555781602859497 | 17920 | 3584 | 14336 | NOT_RECORDED |

## Arm terminal/실패 표

| 모델 | arm | 상태 | valid denom | closure | action | action/A0 | failure type |
|---|---|---|---|---|---|---|---|
| llama3-8b-inst | OFFICIAL_MEMIT | TERMINAL_VALID | 1 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NA |
| llama3-8b-inst | TRUE_FROZEN_C_SPLIT | SCIENTIFIC_CONTROLLER_HOLD | 0 | 0.15773199498653412 | NOT_RECORDED | NOT_RECORDED | ScientificBoundary |
| llama3-8b-inst | REFRESHED_EQUALITY_ONLY | TERMINAL_VALID | 1 | 0.0013482719659805298 | 0.00017077271479593037 | 0.9650431130584215 | NA |
| llama3-8b-inst | FZCB | NUMERICAL_CONTROLLER_FAILURE | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NumericalSensitivityFailure |
| llama3-8b-inst | STRONG_STATIC_SAME_OBJECTIVE | SCIENTIFIC_CONTROLLER_HOLD | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | CandidateFailure |
| qwen2.5-7b-inst | OFFICIAL_MEMIT | TERMINAL_VALID | 1 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NA |
| qwen2.5-7b-inst | TRUE_FROZEN_C_SPLIT | SCIENTIFIC_CONTROLLER_HOLD | 0 | 19.683279037475586 | NOT_RECORDED | NOT_RECORDED | ScientificBoundary |
| qwen2.5-7b-inst | REFRESHED_EQUALITY_ONLY | TERMINAL_VALID | 1 | 0.00011310038098599762 | 0.052342516486532986 | 0.31793547445100384 | NA |
| qwen2.5-7b-inst | FZCB | NUMERICAL_CONTROLLER_FAILURE | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NumericalSensitivityFailure |
| qwen2.5-7b-inst | STRONG_STATIC_SAME_OBJECTIVE | SCIENTIFIC_CONTROLLER_HOLD | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | CandidateFailure |

## FACT

- 각 모델의 stock MEMIT `compute_z`는 case당 한 번만 호출됐고 모든 arm이 같은 target root를 공유했다.
- Arm 결과는 완료/실패 즉시 atomic journal로 봉인돼 이후 FZCB 실패가 앞선 결과를 지우지 않았다.
- `predicted_suffix_after`와 실제 remaining equality-only rollout의 `realized_suffix_action`은 별도 필드다.
- 없는 telemetry는 모든 CSV에서 `NOT_RECORDED`로 남겼으며 추정·imputation은 0이다.

## INFERENCE

- full sensitivity 없이 sketch만 존재하는 경우에는 strict full-space feasible/infeasible 결론을 내릴 수 없다.
- frozen comparator의 actual full-model closure 실패는 frozen 선형 identity와 full-model realization이 다름을 뜻하며, 다른 arm의 결과로 덮지 않는다.

## DECISION

- 이 B1 감사의 모델별 typed status는 위 공동 표 그대로다.
- 두 모델 모두 controller-validity를 완결하지 못하면 B10은 열지 않는다.
- `scientific_promotion=false`; GH 검토 전 task 상태는 `HOLD_AWAITING_GH_FAILURE_REVIEW`다.

## 산출물

- joint table: `/data/janghj/ODE-edit/local/worktrees/fzcb-tech-r1-joint-b1-controller-validity-audit-tech-r5/experiment-reports/servers/server4/fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2/joint-controller-validity-summary.csv`
- arm table: `/data/janghj/ODE-edit/local/worktrees/fzcb-tech-r1-joint-b1-controller-validity-audit-tech-r5/experiment-reports/servers/server4/fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2/arm-status.csv`
- failure ledger: `/data/janghj/ODE-edit/local/worktrees/fzcb-tech-r1-joint-b1-controller-validity-audit-tech-r5/experiment-reports/servers/server4/fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2/failure-ledger.csv`
- waypoint: `/data/janghj/ODE-edit/local/worktrees/fzcb-tech-r1-joint-b1-controller-validity-audit-tech-r5/experiment-reports/servers/server4/fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2/waypoint-summary.csv`
- FD axes: `/data/janghj/ODE-edit/local/worktrees/fzcb-tech-r1-joint-b1-controller-validity-audit-tech-r5/experiment-reports/servers/server4/fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2/fd-axis-sweep.csv`
- sketch ladder: `/data/janghj/ODE-edit/local/worktrees/fzcb-tech-r1-joint-b1-controller-validity-audit-tech-r5/experiment-reports/servers/server4/fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2/sketch-ladder.csv`
