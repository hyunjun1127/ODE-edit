# SH4 — low-cost write/donor core 완료 상세 리뷰
Instruction ID: ODEEDIT-S06-LOW-COST-WRITE-DONOR-CORE-DETAILED-REVIEW-SH4-V1
Nonce: ODEEDIT-GH-SH4-LOWCOST-COMPLETED-DETAILED-REVIEW-20260913-R1
Target: SH4 server4 session01a04939-b5c7-7a03-ba2d-ef3343d62cfd, CWD /data/janghj/ODE-edit, repo hyunjun1127/ODE-edit.
사용자 최신 명령: "server1과 4 task 모두 완료되었으니 자세히 리뷰시켜라".
이는 해당 완료 작업의 명시적 recall이며 CPU 결과검산/상세보고/main 통합을 허용한다. 새 GPU 실험 권한은 아니다.
원문 지시·최신 PROTOCOL·실행 lock을 정독하고 새 analysis worktree/branch/output에서 진행한다. 실제 main/source/tree/dirty를 분리 기록하고 공유 dirty/기존 raw/실패 attempt/타 task를 보존한다.
정확한 지정 job만 단발 scheduler 확인한다. 사용자 완료 통지와 terminal-valid를 구분하고, 아직 실행 중이면 해당 부분을 NOT_TERMINAL로 남긴다. 반복 polling/대기/다른 task 모니터링0.
Slurm submit/requeue/cancel/hold/new GPU/model forward/evaluator/replay/native write 권한 NOT_ALLOWED. Resource cap2 유지; CPU tensor reload·재해시·독립 reducer·저장된 수치의 분석/그림 재생성은 ALLOWED.
미측정 endpoint/지표를 새 평가하거나 인접 state/분모로 채우지 않는다. technical failure는 정확 범위·원인·비용·필요 최소 보완을 보고하되 GPU rerun 자동제출0.
기존 완료한 유효 검산은 manifest/identity로 재사용하고 동기검증을 핑계로 전체 이전 작업을 다시 하지 않는다. 이번 신규 결과의 provenance/분모/상태/계측 coverage는 책임지고 검사한다.
복잡하고 독립적인 raw reducer/상태·계약 감사만 bounded subagent를 활용하고 단순 체크/수정/보고는 직접 수행한다. GH는 이를 중복 raw/GPU 감사하지 않는다.
Red preflight는 analysis source/input allowlist/일치하는 endpoint·평가 semantics, postrun은 independently reduced counts/NLL/transitions/누락/비용/원문 coverage를 확인한다. 허위 비교·수치 오류는 해당 비교 HOLD; 유한 저성능/비수렴/비용초과는 사실로 유지한다.
한국어 SH 보고에는 사실/수치/분모/paired 대비/오류/반대 근거/미검증만 기록한다. 최종 원인 해석·선택·허용 claim은 GH 종합 소유다.
Raw/checkpoint/prompt/full logs/tensors는 local-only. PNG는 직접 작성한 코드 실행으로 생성하고 CSV/코드/환경/hash 및 재현 명령을 기록한다.
이번 review raw broadcast는 NO_BROADCAST_NOT_REQUIRED. 기존 local/imported/봉인 Git 자료만 사용하며 원격 새 raw 전송은 필요 범위/path/SHA를 보고한다. 별도 기존 승인 없는 포괄 rsync나 삭제0.
완료된 본인 범위 source+raw-free 보고만 최신 main 포함 여부 확인 후 clean integration/nonforce main push한다. 다른 SH 커밋을 임의 중복merge하거나 미완료 runtime을 실행완료로 올리지 않는다.
첫 실제 결과표는 전체 분석 완료를 기다리지 않고 전달한다. 다음 상세 report/manifest/receipt와 최종 main HEAD/tree/report SHA를 보고하고 TASK_COMPLETE_STOP. 작업 중 단계별 재승인 질문은 불필요하다.

## 지정 결과와 governing contract
Job46451 odeedit_lowcost_core_s4; source7ece056c33fbb4246245c15f5f7c2a678315c05c/tree352cdd8ca3f3d7b6f2b15f3ed6a6f775fb478443.
Archive d6cf34ab416ee313a01c8491ac0ab9d15a0f39d88a5ea8f7ead43b439dbe4312, lock a672a782c20543b66add9ed83dc5cf04432e9f98f6cc2646e40b68e3627a9317.
Root /data/janghj/ODE-edit/local/low-cost-write-donor-pilot/20260913-v1/attempt-v1/output (control receipt 실제 경로 확인).
원 지시 ODEEDIT-S06-LOW-COST-WRITE-DONOR-PILOT-SH4-V1 및 원문 GH-LOW-COST-WRITE-DONOR-PILOT-20260913-R2, design/cells/schema-v2를 재확인한다.
마지막 관측은 PENDING/GPU initial gate NOT_YET_RUN이었다. 이번에 실제 저장된 gate/terminal을 검증하고, 과거에 initial PASS를 관측한 것으로 기록하지 않는다.
이번 recall은 core 상세 review만 허용한다. 후보선택/GH policy-lock/audit/suffix/추가 alpha·layer cell/신규 GPU 평가 자동 시작0.

## 상세 검산
1. scheduler 완료와 scientific terminal-valid를 분리한다. N4/S875/S75/FULL8/RES8/REFIT4 전6개 예정 대비 실제 endpoint/오류/미측정을 열거한다.
   같은 Server4 fresh-native, L4 W50/M50, B051 [5000,5100), model/source/context/order/P/target-mode/evaluation inventory를 대조한다.
   다른 host나 historical original-target replay를 matched 기준으로 섞지 않는다.
2. D4=actual WN4-We, alpha0/1 snapshot, .875/.75 FP32 materialization, branch entry W/M/RNG 복원과 actual endpoint 평가를 확인한다.
   FULL8/RES8/REFIT4 second z/K/R이 각각 실제 partial state에서 fresh인지, 기존 z를 잘못 공유하지 않았는지 확인한다.
   Native z25 loss evaluations/max24 Adamupdates, fit/finalize 원본 method parity와 logger 비개입을 저장된 증거 범위에서 기록한다.
3. M8이 공통 We에서 이전5000 event를 B100/FP32 chronological 순서로 한 번 재인코딩했는지 확인한다.
   0/historical BLUE M8/alpha별 재구성 대체0; FULL8/RES8 공통M8; physical L8→asset4→singleton0, L4→0.
   같은 batch 두 fitting 사이 current M append0, final selected layer별 append1, REFIT4도1.
   Native wrapper 부작용 제거와 history bytes/counters를 확인한다. Branch 간 history 유출이면 해당 비교를 명시하며 finite 값만으로 valid라 하지 않는다.
4. 같은 canonical MB16 Current R100/P200/N1000, Historical128 R/P/N, Wiki128, MMLUdev32를 확인한다.
   actual counts/분모/lost-gained/true-new NLL/desired signed margins/ties=failure/strict/active-superseded와 NLL tail을 전arm에 기록한다.
   Wiki token/mask 수, MMLU alternative correct/wrong/invalid 정수 주표와 generation parser/별도F1을 섞지 않는다.
   row-identity pairing/endpoint guards를 확인하고 실제 공통 문항만 비교한다. 분모 맞춤·추정0.
5. 전6row 표와 N4 대비 차이를 먼저 GH에 보낸다. 이후 matched 대비:
   S875/S75 vs N4 (scalar trade-off),
   RES8 vs S75 및 N4 (second fitting 회복),
   RES8 vs REFIT4 (같은 second-z 예산에서 cross-layer),
   RES8 vs FULL8 (shrink two-stage).
   개별 손실/반대 방향/Current-Historical-general의 불일치를 숨기지 않는다.
6. auditN1280/MMLU68 사전 seal/disjointness/선택 미사용은 확인만 한다.
   이미 평가·열람했다면 시점/범위를 공개하고 blind로 오인시키지 않는다.
   이번 audit 실행0, FutureN/online alpha/donor selection0; claim-decision=PENDING_GH_REVIEW.
7. 비용은 load/restore/P/stats/M8 과거5000 input 재인코딩, N4shared z/key/solve/history, second fit, 진단, 평가, I-O, allocated wall/GPU를 분리한다.
   신규N4 포함core400request-z 기대와 실제 counter, study shared 비용1회와 각 policy online 필요비용을 구분한다.
   warm setup 포함 상각/steady-state, same-host 비율, 실측memory/disk 및 실패/미기록 counter를 남긴다.
   기존2–8GPUh/20–40GiB는 estimate이지 실측이 아니다. 실측 근거 없는 low-cost 주장0.
8. 품질/비용 참고선은 WITHIN/EXCEEDS/NOT_RECORDED만 표시한다. AND gate나 CI가0 포함한다고 자동fail하지 않는다.
   SH는 실제 수치를 냉정하게 기재하고 후보를 임의 선정하지 않는다. 최종 claim은 GH 소유다.
   audit/5-batch suffix 미실행을 명시하고 core완료를 pilot전체 완료라 부르지 않는다.

## 허용 write/산출물
전용 codex/server4-lowcost-core-detailed-review-20260913-v1.
project/run_scripts/low_cost_write_donor_pilot/ analysis/reducer/tests/plots만 허용; 원runtime/equation 변경0.
기존 실행branch source를 게시할 때 actual execution commit을 별도pin하며 타task live/미완성 source를 합치지 않는다.
audits/servers/server4/2026-09-13-lowcost-core-detailed-review/
experiment-reports/servers/server4/low-cost-write-donor-pilot-2026-09-13-v1/core-completed-review-v1/
messages/acks/server4/2026-09-13-lowcost-core-detailed-review.md
messages/server-heads/server4/2026-09-13-lowcost-core-detailed-review.md
tasks/status/server4/2026-09-13-lowcost-core-detailed-review.json
runs/lowcost-core-review-20260913-v1/ (small metadata).
한국어 diagnostic-report-ko.md, evidence-reuse/input/analysis manifests, comparison-capsule, endpoint-metrics, paired-transitions, quality-frontier, history-provenance, compute-ledger, codePNG/rooted-receipt를 제출한다.
최종 handoff에 전arm 결과/반대 근거/참고선 초과/미검증/재구성가능 snapshot identity/main HEAD/tree/report SHA를 포함한다.
