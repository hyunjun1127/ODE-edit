# GH → SH4: 수치 재현 차이 기록 전용 전환·즉시 재제출

Instruction / nonce: ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RECORD-ONLY-20260925-R1.
Parent: GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1.
대상: SH4/session01a04939-b5c7-7a03-ba2d-ef3343d62cfd,
server4 / /data/janghj/ODE-edit / hyunjun1127/ODE-edit.
사용자: “저런 strict한 gate는 재현에 큰 차이가 없으니 기록만 하고 그냥 올리라고 해”.

## 우선 적용할 최신 실행 권한

이 사용자의 명시적 override는 이전 원 설계·rerun envelope의 수치계약 위반에 의한
BLOCKED_NUMERICAL_CONTRACT/재승인 대기 조건을 이 task에서 대체한다.
SH4가 기록 전용 전환을 구현하고 두 family 및 collector를 실제 제출하라.
추가 사용자/GH 승인이나 별도 원인 분리 GPU 실험을 선행조건으로 요구하지 않는다.
임계값을 올려 PASS를 만들거나 동일 검사를 통과할 때까지 반복하는 방식이 아니다.

정확 기존 실패53182/53183/53184와 실행730a4a9768e5650e01fd9afdc4e0f7895c86ea92,
lock54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699,
1569 allocated GPU초, MEMIT W100 MB16↔MB1 최대NLL차
0.0005242824554443359 > 0.00025, Alpha 자체 T1 PASS/peer failure 및
collector TECHNICAL_BLOCKED는 원본 그대로 보존한다.
이 차이의 low-level 원인은 NOT_ESTABLISHED이고 실패W100 case/category/margin은
NOT_RECORDED다. 사용자 허가를 ‘무해한 FP32 오차가 입증됨’으로 해석하지 않는다.

## RECORD_ONLY 정책의 구체적 범위

- MB16↔MB1, 동일 endpoint 반복, 과거 평가↔현재 평가, restore 전후 및 상태 sentinel의
  **NLL/margin 수치 근접성·success flip 기준**은 diagnostics-only로 전환한다.
  기존 허용치와 실제오차/초과량/위치/flip을 기록하되 해당 초과로 raise/exit 또는
  T1·peer barrier·후속 stage·collector 실패를 발생시키지 않는다.
- worker.py compare/check_raw의 NLL_FIDELITY/MARGIN_FIDELITY/NONBOUNDARY_FLIP과
  HISTORICAL_FIDELITY/ORIGINAL_NONBOUNDARY_FLIP 등 같은 의미의 생산 호출 경로를
  모두 확인한다. MB 비교 한 곳만 풀고 뒤 단계의 동일 근접성 검사로 다시 차단하지 않는다.
- numerical_fidelity_policy=RECORD_ONLY_USER_DIRECTED,
  numerical_certification=NOT_ESTABLISHED를 새 source/config/lock/각 receipt/최종보고에
  결속한다. COMPLETED_WITH_NUMERICAL_WARNINGS와 성공적 계산·완결성을 구분한다.
  DAG가 completion/PASS envelope를 필요로 하면 그 의미를 structural completion으로
  명시하고 warning payload/waiver hash를 결속하여 양 worker·collector가 동일 정책을
  읽게 하라. 숫자 기준이 실제로 PASS했다고 표기하거나 옛 PASS.json을 위조하지 않는다.
- 이전에 저장되지 못한 실패 evidence를 숨기지 않는다. 새 평가에서는 raw/per-case
  비교 근거와 오차를 diagnostic 판정 전에 atomic 저장한다. 새 NaN/손상/IO 예외로
  종료해도 먼저 생성된 근거와 최초 exception을 보존한다.

source/input/token/case/order/분모 identity, actual 선택weight bytes와 intended state,
whole-five-weight materialization, exact restore, finite/NaN/Inf, raw 완결·중복·I/O,
자원/소유/권한 검사는 그대로 유지한다. 잘못된 데이터·weight·복원·누락을
‘수치 경고’로 바꾸지 않는다. scalar M/B/C 정의·회계식·원 관측값도 변경하지 않는다.
모델/dtype/kernel/MB16/패널/문항/FP64제거→FP32한번/고정156셀+16셀을 그대로 둔다.
이번 수치검증 비차단화는 다른 task에 대한 전역 gate 삭제 명령이 아니다.

## 재사용·실행

공통T0/24CP/model/토큰·기존9개 saved fidelity와 Alpha12diagonal은 input/source/
evaluator/state identity와 evidence 수준에 맞춰 재사용한다. 같은 성공 검사를 전부
반복해서 시간·비용을 새로 쓰지 않는다. 원 receipt를 새source PASS로 조용히 옮기지
말고 explicit reuse bridge를 남긴다. 실패 MEMIT W100의 미저장 자료는 만들어내지
않고 필요한 평가를 새 source에서 수행·기록한 뒤 초과가 있어도 본 측정을 진행한다.
기존 과학 score task0이므로 실제 T2P/T2F/T3B와 collector T3A/T4를 수행한다.
T0→T1→T2P→T2F→T3A/T3B→T4 내부 자동 진행은 유효한 음성 결과나
diagnostic warning에 막히지 않아야 한다. 구조적 실패 수집은 유지한다.

새 immutable attempt/source/archive/실행lock을 별도로 만든다. 원 runtime/raw/source/
실패비용·설계manifest 변경0. 정의를 바꾸지 않는 wiring/직렬화 기술오류는 범위 내
최소수리 허용, broad GPU diagnosis sweep/new editor/z/nativefit/history0.

## 제출·자원·종료

server4 project/task cap2, 두family 각1GPU/8CPU/host60416MiB 이하,
exportNONE/Requeue0. collector8CPU/24576MiB/GPU0와 afterany 새GPUIDs.
다른 active/admitted project 점유와 disk/메모리 확인, exact held source/owner/
fullargv/resources/dependency 검사 후 release. 무관 job 변경0, 이미 종료된 원job
재취소0. 대기 자원이라면 cap-safe pending 등록하고 중복 submit하지 않는다.
save_checkpoints=false. 기존 원CP read-only, 새로운 CP/fullendpoint/delta bundle 저장0.

**정상 등록·held 검사·release 뒤 agent 모니터링 중단**. 실제 initial/terminal을
보려고 추가 scheduler/log/result 조회하거나 대기하지 않는다. INITIAL_NOT_OBSERVED,
monitoring_active=false/automatic_resume=false로 source/lock/job mapping을 보고하고
STOP한다. 자율 Slurm runner/reducer만 자연 진행한다. 완료 분석은 사용자 recall 때.

## 허용 파일·게시 및 검산

기존 SH4 envelope의 project/run_scripts/historical_update_timeaxis/와
/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/ 아래 새 attempt,
해당 server4 report/audit/plans-update/ACK/messages/status/runs 경로만 허용한다.
전용clean branch codex/server4-historical-timeaxis-record-only-20260925-v1 권장.
record-only source+bounded CPU회귀+소형 제출인계의 nonforce branch/main 게시 승인.
공유dirty/native/env/Gitidentity·원자료 보존, raw/tensor/prompt/fullstdout Git0.
NO_BROADCAST_NOT_REQUIRED 근거 기록. SH1 이관은 철회 유지.

CPU regression에서 같은 0.00052428 초과/큰 finite 초과/flip도 warning으로 기록되어
downstream에 전달되는지, identity/비finite/restore failure는 계속 차단되는지,
warning 때문에 peer join/collector가 실패하지 않는지 좁게 검사한다.
검증 완료를 broad actual GPU numerical PASS로 확대하지 않는다.
첫 ACK에 nonce/정책 적용/미제출 상태를 남기고 즉시 구현·제출을 이어가라.
