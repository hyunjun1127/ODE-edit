# SH4 수신 및 구현 ACK

Nonce: `GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1`.

실제 server4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` / repository `hyunjun1127/ODE-edit`를 결속했다. 별도 clean worktree는 `/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/worktree`다. main73d76459의 실행 envelope를 적용하며 이전 E3 pause를 상속하지 않는다.

승인 소형 3파일·29 regular member / 5,238,704B를 receiver full SHA/size 검증했다. 압축의 정확 설계-root 접두사만 제거하고 traversal/link/device/중복/미승인 member를 거부했다. 원 bytes는 그대로 보존한다. Receipt SHA `1ff354e07cc2f76588a1b831e1469be12530bf1f03e01e13f0f86a15ff34961e`.

T0는 CP24/선택 tensor120/model shard4/W0 선택 weight/고정10k/토큰을 새로 검증했다. Runtime-binding SHA `717808cb2460b034d7f526bb14cf325e0ad817f678d6fa7cd98844649f051e41`; CPU188.255초·process peak2125.289MiB. GPU/model forward0. CP 신규전송0.

초기 상태 `IMPLEMENTING_NOT_SUBMITTED`, job_ids=[]였다. 실제 이후 제출은 별도 submission/release receipt로 구분한다. Whole U·FP64 subtraction·FP32 one-cast·MB16·기존 evaluator 그대로, project/task cap2/각1GPU·8CPU·60416MiB, 신규 checkpoint0. T0→T4 완료 또는 technical blockage까지 계속한다.

GH direct 수신 ACK도 같은 nonce로 회수했으며, GH 감사/ACK를 단계 선행조건으로 사용하지 않는다. NO_BROADCAST_NOT_REQUIRED.

## 최신 사용자 중지 ACK

“리소스가 없으니 코드 파이프라인 점검만 하고 모니터링은 중단하자” 수신·적용. 기존 CPU16검사 및 소스 반례만 확인했다. T1 routing KeyError가 재현되어 파이프라인 전체 PASS는 아니다. 점검 범위이므로 수리/새 제출/기존 job 변경 없이 53176/53177/53179를 보존하고 monitoring_active=false, automatic_resume=false로 사용자 호출을 기다린다. 이 override 이후 scheduler/log/result 조회0.

## 후속 사용자 정정 — 수리·재제출 승인 / job 모니터링만 중단

“아니 점검에서 오류 사항있으면 수리 재제출해 / job 모니터링만 하지 말라는거였어”를 적용했다. T1 routing을 최소수리하고 CPU22 tests를 WT/frozen에서 PASS했다. 정확 이전3개를 미시작 상태에서 취소·보존하고 새53182/53183/CPU53184(afterany:53182:53183)를 held 검사·release했다. 제출 snapshot PENDING(None), actual GPU gate 미관측. 이후 job 진행 모니터링·자동 재개 없이 사용자 호출을 기다린다. 실행source730a4a97, 상세기록은 routing-repair-r2 report/receipt다.

## 2026-09-25 rerun recall ACK 및 typed blockage

Nonce `ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RERUN-20260925-R1` 수신. 정확53182/53183 FAILED, CPU53184 COMPLETED/science TECHNICAL_BLOCKED를 확인했다. 최초 원인은 MEMIT T1 MB16↔MB1 NLL차0.0005242824554443359>고정0.00025다. Alpha 자체T1 PASS 뒤 peer failure를 받았다. 저장9개 fidelity를 독립CPU2592행검산, 원state/12diagonal/input/source를 재결속했다. 실패 W100 상세raw 부재로 하위 원인은 미확정이고, 정의를 유지하는 수리 근거가 없어 envelope대로 BLOCKED_NUMERICAL_CONTRACT를 인계한다. 새제출0/원runtime·threshold변경0/모니터링0. 원부모GPU비용1569초. 상세보고는 rerun-20260925-r1 package다.

## 2026-09-25 사용자 기록 전용 override 및 실제 제출

ACK nonce `ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RECORD-ONLY-20260925-R1`.
이전 수치계약 차단은 최신 사용자 권한으로 해제했다. 원 기록/기준/오차는 보존한다.
수치 근접성·flip만 기록 전용이며 identity·finite·exact restore·완결성은 계속 차단한다.
CPU32 PASS 후 source `b856babb`를 새 `attempt-r3-record-only`에 봉인했다.
Alpha53283/MEMIT53284/CPU53285(afterany:53283:53284) held 검사·release 완료.
INITIAL_NOT_OBSERVED, numerical_certification=NOT_ESTABLISHED. 신규 CP0, cap2.
Release 후 scheduler/log/result 조회0, monitoring_active=false/automatic_resume=false.
상세는 record-only-20260925-r1 report/submission/lock에 기록했다.
