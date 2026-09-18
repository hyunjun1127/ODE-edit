# GH → SH4: 저장 reserve admission waiver / M 지금 제출

instruction_id: ODEEDIT-S06-SINGLE-LAYER-EDIT-PRESERVING-CORRECTION-M-SH4-V1
nonce: ODEEDIT-GH-SH4-ENFC-STORAGE-WAIVER-SUBMIT-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 최신 사용자 원문

> 일단 올리라고 해. 내가 공간 확보 해놓을게

이는 명시적인 same-task recall이자 **기존 보수적72GiB free-space reserve를 제출 차단 조건으로 삼지 않고 M을 지금 제출하라는 승인**이다. 사용자가 공간을 확보할 예정이라는 사실을 기록하되, 이미 확보 완료되었다고 주장하지 않는다. GH/SH4가 기존 자료를 삭제하라는 권한은 아니다.

## 즉시 실행

1. 준비 source fb9d2ac41455dc6e97fa78080b7307047f8395b2 및 기존 T-skip/M 계약을 이어받는다. 새 override와 source/config/input/archive를 immutable하게 봉인한 뒤 **M10/10 independent cold episode를 cap2로 upfront held inspection→release**한다. 다른 점유가 없으면 array0–9%2, 각1GPU/8CPU/60416MiB/exportNONE/Requeue0. 기존 planned walltime/source조건은 실제 lock에 명시한다.
2. 제출 직전 disk free/inode/writable 경로를 한 번 관측·기록하되, **free<72GiB라는 이유만으로 freeze·archive·submission·job bootstrap을 다시 차단하지 않는다**. 모든 관련 admission 위치에서 같은 명시적 사용자 waiver를 참조한다. 72GiB 원 estimate를 지우거나 더 작은값으로 위장하지 않는다. 상태는 USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP이며 reserved/exclusive allocation 또는 충분공간 PASS가 아니다.
3. 여유공간 확보 확인·새저장계획·대체storage 탐색·사용자 추가승인을 다시 제출 선행조건으로 넣지 않는다. 다른 완결성 검사와 cap2 actual admission은 유지한다. 당장 source/archive/lock의 실제 쓰기조차 불가능한 ENOSPC/권한/IO 오류는 숨기지 말고 실제 오류를 보고한다. 단순 보수적reserve 부족과 실제write실패를 구분한다.
4. **T 새실행/수리/continuation0**, T_READY/afterok/old failcancel 연동0. T=SKIPPED_USER_DIRECTED/full_numerical_validation=NOT_ESTABLISHED 유지. M 방법/geometry/optimizer/guard/threshold/공식observer분리 변경0. 추가진단·새smoke로 T를 재도입하지 않는다.
5. B1 nativefit/적합 observer REUSE, B2–10 independent W0/M0 최대9sharednativefit/900target, 원8arm/10episode/80finalL4endpoint 보존 그대로다. 저장 의무를 없애거나 평가를 축소하는 지시가 아니다. 원 T49928/M49973 및 모든실패/partial/source/teacher/native/cost3231GPU초 보존, 다른task변경0.
6. CPU는 필요한 좁은 waiver-routing/no-deletion/실제쓰기오류보존 검사를 적용한다. 기존 CPU82+4/문서FULL_READ는 exact근거 재사용 가능하며 전체중복검사로 제출을 지연시키지 않는다.

## 런타임 저장 안전 / 모니터링

실제 쓰기 실패·filesystem 오류는 감지하고 가능한 failure receipt/비용/완료범위를 보존한다. atomic/create-once/checkpoint완결성 검사는 유지한다. 불완전 endpoint를 저장완료/성공으로 표시하지 않는다. 실제 ENOSPC 등으로 필수 저장이 불가능해진 실행은 기술 실패로 처리하며 원자료 자동삭제·덮어쓰기·평가제외로 감추지 않는다. 이 지시는 원72GiB 추정치에 대한 admission waiver이지 모든 IO failure를 무시하라는 지시가 아니다.

기존자료 삭제/이동/cleanup job/과거 checkpoint 정리/다른server 전송0. 공간 확보는 사용자 소유 작업이다. SH4가 사용자의 확보 작업을 추적하는 새 모니터·daemon을 만들지 않는다.

**기존 M 초기 gate 후 pause 유지.** M 전체 정상release 후 가능한 대표 실제M 초기 실행을 확인하거나, 실행 가능한 main이 없고 GPU 부족이 확인된 경우 기존main pending예외로 인계한다. 72GiB 미확보를 다시 STORAGE_BLOCKED로 재인계하거나 공간확보때까지 자동대기하지 않는다. 이후 agent monitoring/heartbeat/terminalwait/추가submit/상세분석확장0, submitted 프로그램은 자연진행한다. S/R/L 제출0.

## 제출 보고와 범위

최초ACK에 waiver 적용·T생략·cap2 M전체제출·삭제0을 확인하고, 실제 jobID/arraymapping/source/tree/archive/lock/관측free+72GiB waived/heldinspection/release를 바로 보고한다. 아직 제출되지 않았으면 그렇게 표시하며 정상제출 완료까지 계속한다.

기존 허용 write/ownscope nonforce main 및 raw-free/PNG정책 상속:
project/run_scripts/single_layer_edit_preserving_correction/ 와
/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/
및 기존 audits/reports/acks/server-heads/runs/task-status 경로만.
원 global정본/sharednative/shared환경/타job 변경0. 기존 report는 불변이고 새 waiver/submission receipt로 연결한다. GH 중복 raw/GPU 감사나 재승인 대기 없이 진행하라.
