# GH → SH4: ENFC T/M 병행 제출 및 T 실패 시 M 동반 중단

instruction_id: ODEEDIT-S06-SINGLE-LAYER-EDIT-PRESERVING-CORRECTION-M-SH4-V1
nonce: ODEEDIT-GH-SH4-ENFC-T-M-PARALLEL-FAILCANCEL-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit
gpu_cap: 2

## 최신 사용자 원문 / 즉시 적용

> T 가 너무 오래걸리는데 M을 기다리기엔 시간낭비가 크므로 M 도 동시에 올려서 T에 오류가 발생했으면 M도 같이 내리는 것으로 하자

이는 기존 envelope의 **T_READY 이후에만 M 등록·실행** 순서를 변경하는 명시적 승인이다. 현재 T49928의 완료를 기다리지 않고, CPU/source/input/resource/재사용 계획 및 M 실행에 필요한 구현이 준비되는 즉시 M 누락 단위를 등록·검사·release하고 T와 병행 실행한다. T를 PASS로 꾸미거나 기술 검증을 없애는 지시가 아니다.

M-only / reuse-first / 모든 M final L4 endpoint 보존 / cap2 / S·R·L 제출0은 그대로다. 기존 제어문·lock은 역사 기록으로 보존하고 새 override receipt 및 M 실행 lock에 이번 변경을 결속한다. 같은 active task에 대한 related steer이며 별도 중복 task/기술 job 생성 지시가 아니다.

## 병행 제출과 자원

1. 정확 T49928 및 이미 존재하는 본task M job ID/state/owner/source를 한정 확인한다. GH의 이전 RUNNING 보고를 현재 상태로 추정하지 않는다. T가 이미 실패했으면 M 신규 제출0, 아래 실패 처리 후 인계한다. 이미 T가 성공했다면 실제 READY/최종 receipt와 source 적용범위를 검증한 뒤 원범위 M을 진행한다.
2. T가 정상 진행 중이면 T1GPU + M lane1GPU = cap2로 병행한다. 신규 M에 afterok:T_READY 실행 차단을 걸지 않는다. 원 M 재사용 판정의 B1 nativefit REUSE/B2–B10 최대9 cold fits 및 observer reuse를 그대로 유지한다. 준비된 동일 source의 필요한 M을 upfront 등록하되 array throttle/dependency로 T와 M 합계 동시가능 capacity가2를 넘지 않게 한다.
3. 기술 미완료 때문에만 기다리지 않지만, M runner 자체의 누락 구현/잘못된 import/source/data binding/실행 불가 자원을 무시하여 제출하지 않는다. 알려진 receipt helper 중복 status 오류가 M 경로에도 있으면 M frozen source를 최소 수정하고 CPU 회귀검사한다. 이미 실행 중 T의 source를 덮거나 T를 몰래 교체하지 않는다. M/T source 차이와 기술 검증 커버리지 차이를 기록한다.
4. T 성공·resource release가 확정되면 본task M의 자체 array throttle/lane만 안전하게2로 확장할 수 있다. 다른 job 변경0. T를 속도 목적으로 취소하거나 새과학arm을 추가하지 않는다. 기존 한 job 기본1GPU/8CPU/60416MiB/exportNONE/Requeue0와 actual preflight 유지.
5. T가 끝나기 전에 M이 낸 candidate/commit/평가는 **PROVISIONAL_T_UNRESOLVED**다. T/model validation PASS 또는 M scientific-valid 완료로 표기하지 않는다. T가 성공해도 M source에 적용 가능한 기술 항목만 PASS 결속하며 다른 implementation의 무조건 동일성 주장0.

## T 오류 → 연결된 M 전부 중단 (새 exact 취소 권한)

이번 사용자 원문은 **본task T 오류 시, 이번에 함께 제출한 M running 및 pending job을 취소하는 권한**을 명시적으로 부여한다.

- T의 비정상 scheduler 종료(FAILED/OOM/TIMEOUT/CANCELLED/NODE_FAIL 등) 또는 실패 receipt/필수 검사 fail을 관측하면 즉시 같은 attempt에 결속된 M 후속 release/submit을 금지하고, **그 M의 정확 allowlisted running+pending IDs만 scancel**한다. 이미 완료된 M은 취소 대상이 아니며 미검증 산출물로 보존한다. 사용자 전체 queue, 다른 task, 과거 job을 광범위 취소하지 않는다.
- receipt helper/저장 오류도 T 성공이 아니므로 이 동반중단 규칙을 적용한다. 과학 오류가 아니었다는 RCA와 중단 이유는 별도로 구분한다. 단순 낮은 성능, 합법적인0correction/공간없음/nativefallback은 T 오류로 바꾸지 않는다.
- live process cleanup/entry 복원이 실제 확인됐는지와 SIGTERM 등으로 미확인인지를 구별한다. 취소했다고 rollback 성공으로 주장하지 않는다. 기존/신규 native capsules, M partial endpoints, logs, partial evaluation, source/locks와 allocation비용을 보존하고 중복계상하지 않는다. 파일 삭제0.
- cancel 요청/terminal 상태/자원 해제를 정확 task 범위에서 확인하고 paired-stop receipt에 T 원인·관측시각·M exact ID/상태·취소결과·완료/미완료범위·비용을 기록한다. 과거 완료자료나 공통teacher/native를 다시 만들지 않는다.
- T 실패 후 새 M 재제출을 자동 반복하지 않는다. CPU RCA/최소수리 제안과 재사용 가능한 evidence를 인계하고 WAITING_USER_RESUME으로 중지한다. 수치 threshold/FD grid/과학식 변경0.

## 모니터링 종료의 최소 정합성

기존 **본실험 M 초기 gate 후 pause** 의도는 유지하되, 이번 T 실패 연동을 수행할 수 없게 방치하지 않는다.

가장 단순한 기본 운영은 **T의 성공/실패가 결정될 때까지 본task bounded 관찰을 계속하고, T 성공 + M actual initial 확인이면 pause**하는 것이다. 이 대기는 M 제출을 막는 선행조건이 아니며 두 job은 이미 병행 실행한다. M initial이 먼저 확인돼도 아직 T 오류에 대한 취소 책임이 해결되지 않았다면 즉시 agent STOP하지 않는다.

만약 T 미완료 상태에서 초기 인계가 필요하다면, 먼저 기존 task-owned launcher 안에 정확 T/linked M IDs를 감시하고 T 실패에 M 실행·대기열을 실제 중단하는 bounded fail-stop을 구현·검증하고 그 책임 경계를 receipt에 명시해야 한다. 단순 파일 marker 확인 한 번이나 보고 메시지만으로 연동이 된 것으로 간주하지 않는다. 별도 상주 daemon/제품 자동callback/새주기재개 예약은 만들지 않는다. 구현 복잡성이 늘면 위의 기본 bounded 관찰 방식을 사용한다.

M이 actual initial 전 자원부족으로 대기할 때도 T가 진행 중이고 실패연동이 미확정이면 방치하지 않는다. T 성공 후 기존의 확인된 main GPU resource 부족 예외는 그대로 적용할 수 있다. 정확하지 않은 PENDING/Reason=None/CPU메모리 원인으로 즉시 STOP0.
T 성공+M초기 또는 실패동반중단 인계 후에는 polling/terminalwait/추가submit/상세분석/main확장0, submitted M 자연진행(성공일 때만), explicit USER recall 대기. M 전체 결과나 S/R/L 확대는 이번권한 밖이다.

## 보고·수정 범위

원 envelope의 session/read/write/Slurm/source/자산보존/PNG/report 정책은 그대로 상속한다. 이번 추가 write는 기존 task-local control/launcher/tests/submission/paired-stop receipts와 compact status이다. 원 PROTOCOL/global 설계/native/science tolerance/shared환경 수정0. Ownscope source+compact제출/초기인계 nonforce main 게시권한 유지.

중간보고에 다음을 명시한다:
- override 적용, 현재 T 정확상태, T-ready 대기 없이 M 준비/등록 범위;
- T/M job mapping과 동시2GPU 계획/실제관측, source차이와 known helper 수리;
- M PROVISIONAL 상태, T실패 exact 취소 대상목록과 모니터링 책임;
- 이후 T 성공+M초기 pause 또는 T 오류+M동반중단/보존 인계.

GH 중복 scientific/raw/GPU 감사나 추가 단계별 승인 없이 위 변경을 적용하라.
