# ENFC M B1 — 재사용·제출 및 자원 대기 인계

상태: `RESOURCE_PENDING_INITIAL_NOT_YET`. 실험 완료 또는 수치 PASS가 아니다.

## 범위와 재사용

O0/b001 ordinal [0,100), unique100. 최종 비교는 N4/SCALE/CA/KL-P/EN-S/EN-F/EN-COV/EN-F4의 8arm×100이며 unique800이 아니다. CA-EXACT 및 RAND±는 별도 진단이다. Past는 비어 있고 Report256은 열지 않는다. B2 이후 및 S/R/L 실행은 없다.

정지된 S4 B1와 정확한 companion 269개, 16,733,125,876 bytes를 SH1 단일 writer가 create-once 수신했다. 양쪽 전체 SHA/size 검증 완료. 기존 native capsule WN SHA `3d33f44b13688fb3c18db8eb4d8be5cc6509aea161c96bed254e89240f265015`를 복원하며 새 native fit/target/history append/teacher 생성은 0으로 봉인했다.

| Arm | 최적화 | 보존된 상태 | S1 작업 |
|---|---|---|---|
| N4 | REUSE | NATIVE_ENDPOINT | observer |
| SCALE | REUSE | TRIAL_BUDGET_EXHAUSTED_NO_NEW_ACCEPT, N4와 같은 bytes | 동일 endpoint observer 재사용 |
| CA | REUSE | TRIAL_BUDGET_EXHAUSTED_NO_NEW_ACCEPT, N4와 같은 bytes | 동일 endpoint observer 재사용 |
| KL-P | REUSE | GRADIENT_BUDGET_EXHAUSTED | observer |
| EN-S | REUSE | GRADIENT_BUDGET_EXHAUSTED | observer |
| EN-F | RUN_MISSING | 취소 partial은 REFERENCE_ONLY | immutable WN에서 원 예산으로 시작 |
| EN-COV | RUN_MISSING | endpoint 없음 | immutable WN에서 원 예산으로 시작 |
| EN-F4 | RUN_MISSING | endpoint 없음 | immutable WN에서 원 4×6 예산으로 시작 |

선택 seal/ledger SHA, endpoint FP32 shape/finite/W0/WN/context/request identity를 CPU에서 검증했다. 유효 fallback은 제거하지 않았다. S4의 M50050 두 cell 합9768GPU-sec는 과거 비용이며 이번 새 계산량에 다시 합산하지 않는다. 취소 SIGTERM rollback은 NOT_VERIFIED다.

## Source와 플랫폼

- Frozen M: `87f65ea2abcbe7e77e04367f73a001d63443734b`, tree `980c04b3e998caab0284b855041f6ad3e2e30356`.
- S1 execution: `3f1941b21538d6a7ad0afd756774ad6a0b605750`, tree `72503847266864bf5e1b0c96d63ba99aa03d846d`.
- Worktree: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-enfc-single-batch-m-v1`.
- 실행 source는 그 위치에서 고정되며 main publication worktree와 분리된다.
- 수정은 S1 node/path binding, B1-only reuse orchestration이다. 기존 optimizer/geometry/guard/observer 수식과 threshold는 그대로다. S runtime은 가져오지 않았다. TorchVersion metadata의 builtin str 변환도 유지한다.
- S4 Blackwell과 S1 A6000의 bitwise parity는 주장하지 않는다. 같은 native/완료 endpoint를 보존하며, S1에서 호환성이 입증되지 않은 관측을 다시 측정한다. Captured-key identity가 다르면 그 입력에 의존하는 exact geometry만 같은 cutoff로 재계산한다. 초기 gradient는 GPU와 cache identity가 모두 같을 때만 재사용한다.
- Teacher192 payload는 원 SHA 그대로다. 새 manifest는 절대경로만 바꾼 derivative이며 원 manifest를 보존한다. W0 teacher 재생성0.

T는 `SKIPPED_USER_DIRECTED`, `full_numerical_validation=NOT_ESTABLISHED`. CPU40 tests PASS 및 compile/shell/task memory audit PASS는 실제 모델의 수치 검증 PASS가 아니다. 저장소 전체 memory audit의 기존 S4 legacy launcher 6개 위반은 이 task 밖으로 남겼으며 수정하지 않았다.

## 제출과 정확한 자원 대기 근거

`50098 / odeedit_enfc_m_b001_s1`를 held→owner/source/args/resource 검사→release 했다. Dependency=null, Requeue=0, exportNONE. 1GPU/8CPU/mem182272MiB, wall24h 요청. Hourcap=null이며 다른 실험 예산을 상속하지 않는다. 신규 데이터 뒤 추가32GiB reserve, 당시 가용 약814.9GB. Wall24h는 S1에서 아직 실측하지 않은 allocation 상한이지 실제 비용이나 예산 확대가 아니다.

제출 전 project active+admitted=0, 새1≤cap2. 단발 확인에서 `PENDING`, reason=`ReqNodeNotAvail,_May_be_reserved_for_other_job`, allocated TRES 없음, runtime0이었다. Node는 `MIXED+PLANNED`, GPU 총8/할당6이었다. 따라서 GPU가 전부 사용 중이라고 주장하지 않는다. 현재 요청한 연속24h 실행창에 대한 scheduler/node reservation 가용성 부족으로 대기 중이다. Scheduler 제시 시작시간은 2026-09-19 07:30 KST였으며 보장이 아니다. 다른 job/예약을 취소하거나 요청 시간을 임의 변경하지 않았다.

실제 model load/correction/observer/reset initial gate는 **NOT_YET_RUN**. 제출 승인이나 CPU 준비를 initial-valid로 대체하지 않는다. 정상 release 후 resource pending인 이 상태를 인계하며 시작/terminal 대기를 위한 polling/heartbeat/자동 깨우기는 설치하지 않는다. 제출된 원 프로그램은 자원 배정 시 계속한다. 사용자 recall 때 실제 gate/완료 범위를 확인한다.

## 정확한 경로·lock

Task root: 위 execution worktree의 `local/enfc-single-batch-m/20260918-v1/`.

- `submission-r1/execution.lock.json`: SHA `b463293b23131b1453d1d3521a58f8f57e1804b0ca18d0fd8fcd0bdc64999147`.
- `submission-r1/{held,inspection,released}.json`: owner/source/args/resource/release 증거.
- `imports/server4-b001-r1/{allowlist,receiver-seal}.json`: source/destination full SHA/size 및 sole-writer 증거.
- `locks/{reuse,source-input,resource,teacher-path-diff}.json`: 재사용/고정 자산/자원/path-only 변경.
- 예정 output: `output-r1/b001/attempt-v1/`.
- 최초 actual marker: `M_INITIAL_VALID_WITH_T_SKIPPED.json`. EN-F 저장/reload/selection seal/실제 observer와 Dev128/W0·M0 reset 이후 기록되며, 전체 observer나 모든 결과의 완료 표지가 아니다.

남은 범위는 3개 미완료 optimization, 필요한 endpoint observer/Dev128/greedy32/RAND± 및 종료 integrity다. 실제 결과의 상세 사실 분석은 완료 recall 이후다. Scientific promotion=false. Raw/teacher/weights/prompts/log Git0; `NO_BROADCAST_NOT_REQUIRED`. S4 live S 및 다른 paused task는 조회·변경하지 않았다.
