# P2R6 Llama closed-loop QP repair Phase-1 종료 기록

- 작성 범위: 기존 결과·소스에 대한 읽기 전용 raw-free 분석. 새 모델/GPU/Slurm/evaluator 실행 0.
- 계약: `ODEEDIT-S05-P2R6-LLAMA-OUTER1-QP-CAPTURE-REPAIR-PHASE1-V1`; red review SHA256 `b9494b606b98df184fa5ae7f08787f9a4ecbc77750fca8060ce6c90e883ee26a` (9851 bytes, 288 lines).
- 최종 scientific source: HEAD `f7b18905f5751c329c28d7e19bd6db7fb33d61dc`, tree `d2b9e5c41e7da4ded45a6e975974d1b341d996ad`.
- scheduler: job20114 task0 (`20115`) `FAILED 1:0`, 00:11:44; task1 (`20114`) `COMPLETED 0:0`, 00:11:12.
- 시도 8, 유효 endpoint 7, 기술적으로 무효인 incomplete prefix 1(Llama case5 AS-CAP), retry/tuning/rerun=0.
- `scientific_failure=false`; 완전한 2모델×4arm matrix가 아니므로 Phase-1 scientific gate=`NOT_RUN`; Phase2=`CLOSED`.

## 엄격 기술 게이트

| 항목 | 사실 |
|---|---|
| Llama AS-CAP 실패 위치 | outer step 1, `AS_CAPACITY_IN_SEMANTIC_REGION` QP certificate |
| solver 상태 | `ACTIVE_SET_CROSSOVER_CERTIFICATE_FAILED` |
| r_pri / r_stat / r_comp | 1.2305712004945235e-12 / 2.638772880758644e-12 / 1.2638838334226404e-08 |
| external tolerance | 1e-08 |
| false gate | `r_comp > external_tolerance` |
| Llama AS W0 복원 | pointer=true, bytes=true |
| 유효 endpoint K8 / action-freeze / W0 | 7 / 7 / 7 |
| 실패 prefix physical write | 1; 이후 action-freeze·terminal evaluator=`NOT_RECORDED` |

## endpoint 표

아래 8행 기계 표는 `p2r6-red-r2-phase1-llama-qp-repair-endpoints.jsonl`에 있다. z-intervention objective와 W-only full-six NLL은 별도 열이다.

| 모델 | case | arm | 상태 | K | W0 | z-objective NLL | W-only NLL | W Eff/Gen/Loc | P | BF16 capacity |
|---|---:|---|---|---:|---|---:|---:|---|---:|---:|
| Llama3-8B-Instruct | 5 | A0-CAP | VALID_ENDPOINT | 8 | exact | 1.16192889 | 1.11134699 | 9/10 / 19/20 / 80/100 | 0.0795164002 | 62.3251597 |
| Llama3-8B-Instruct | 5 | AETA-CAP | VALID_ENDPOINT | 8 | exact | 0.856819583 | 0.510136097 | 10/10 / 19/20 / 80/100 | 0.0546651402 | 43.4552391 |
| Llama3-8B-Instruct | 5 | AR-CAP | VALID_ENDPOINT | 8 | exact | 0.310920778 | 0.25436708 | 10/10 / 19/20 / 81/100 | 0.0626989887 | 48.5607383 |
| Llama3-8B-Instruct | 5 | AS-CAP | TECHNICAL_INVALID_INCOMPLETE_PREFIX | 1 | exact | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | 1 | A0-CAP | VALID_ENDPOINT | 8 | exact | 0.00069188335 | 0.000730955966 | 10/10 / 19/20 / 92/100 | 4.02185383 | 433.102459 |
| Qwen2.5-7B-Instruct | 1 | AETA-CAP | VALID_ENDPOINT | 8 | exact | 0.000724450948 | 0.0019331367 | 10/10 / 19/20 / 92/100 | 1.30119592 | 123.37248 |
| Qwen2.5-7B-Instruct | 1 | AR-CAP | VALID_ENDPOINT | 8 | exact | 0.000862581386 | 0.000799411759 | 10/10 / 19/20 / 92/100 | 1.67211404 | 164.263653 |
| Qwen2.5-7B-Instruct | 1 | AS-CAP | VALID_ENDPOINT | 8 | exact | 0.000680446088 | 0.0770679194 | 10/10 / 18/20 / 92/100 | 2.26812612 | 242.554423 |

## per-step 표

- 57행: 7 valid endpoint × K8 = 56행, Llama AS-CAP의 valid physical prefix step0 = 1행.
- `predicted_minus_deficit_mean`, `predicted_over_deficit_minus_one_mean`, `predicted_minus_actual_progress_mean`, `alpha_change_l2_from_prior_step`은 receipt 원시 배열에서 계산한 산술값이다. 첫 physical step의 alpha change는 비교할 이전 step이 없어 `NOT_RECORDED_INITIAL_STEP`이다.
- 모든 행은 request/layer의 원시 입력·prompt·target·tensor를 포함하지 않는다.
- 파일: `p2r6-red-r2-phase1-llama-qp-repair-steps.jsonl`.

## compute 및 금지 영향

- valid endpoint의 writer transition=8, materialization=8, response VJP=40, target microstep=24이다.
- 유효 endpoint terminal evaluator는 action-freeze 뒤에만 수행되었다. 각 terminal receipt의 heldout controller influence=0, inner-step heldout evaluation=0.
- valid terminal forbidden-influence receipts: retry=0, backtracking=0, remaining-horizon division=0, semantic debt=0, functional-P veto=0, historical/sequential=0, shadow model F/B/materialization=0.

## 기존 정정 부록

- append-only 정정 부록: `p2r6-red-r2-phase1-report-correction-addendum-ko.md`, SHA256 `2dcf03bf03038ebc548c23cc483bada90a7480018d384d1a905dc0e830a5904e`.
- 해당 부록은 기존 §5.2의 terminal z-intervention objective와 W-only full-six target-new NLL 표기 구분을 기록한다. 기존 보고서는 변경하지 않았다.

## 상태

- `PHASE1_TECHNICAL_INVALID_STRICT_STOP`
- `PHASE1_SCIENTIFIC_GATE=NOT_RUN`
- `PHASE2=CLOSED`
- `scientific_failure=false`
- 이 문서는 사실·기계적 계약 상태·수치만 기록하며, 과학적 해석 또는 후속 실행 권고를 포함하지 않는다.
