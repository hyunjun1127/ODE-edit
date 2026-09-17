# GH → SH4: v2 재개 — 초기 gate는 본실험 기준

instruction_id: ODEEDIT-S06-SEQUENTIAL-LOCAL-Z-ALLOCATION-V2-SH4-V1
nonce: ODEEDIT-GH-SH4-SLZV2-MAIN-GATE-RESUME-20260917-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 최신 사용자 원문 및 정정

> 내가 말한 초기 gate는 본실험의 초기 gate이다. 본실험의 정상 제출까지 task 이어서하라고 해. 단, 본실험이 pending중일 경우(gpu리소스 할당이 부족하여)에 한해 pending 중지 조건을 명령한 것이다.

이번 메시지는 명시적인 USER recall이다. 기존 모니터링 pause를 해제하고 이 v2 task를 지금 재개한다. 앞선 GH envelope §6의 **“기술 또는 science가 PENDING이면 중지” 해석은 사용자 의도보다 넓었으므로 정정한다.** 본 지시가 그 종료 조건과 기존 resume-manifest의 사용자 대기 상태보다 우선한다. 과거 당시 PENDING 관측/보고/receipt는 역사적 사실로 보존한다.

**기술 job PENDING, 기술 PASS, source 게시만으로 task를 종료하지 않는다.** 필수 기술 검증을 처리하고 기존 승인된 **여섯 본실험 모두 정상 등록·inspection·release**까지 진행한다. 이후 실제 실행 가능한 본실험의 초기 gate를 확인하거나, 아래의 한정된 main GPU-resource PENDING 예외를 적용한다.

## 재개 순서와 범위

1. 기존 technical49421 / odeedit_slz_v2_tech_s4의 실제 scheduler·완료 marker·필수 기술 결과를 정확 task 범위에서 확인한다. 마지막 PENDING 기록을 현재 상태로 추정하지 않는다. 이미 진행/완료한 단계를 중복 제출·재실행하지 않는다.
2. 기술 job이 자원 대기 중이어도 이번 main-PENDING 중지 예외를 적용하지 않는다. 필요한 구현/CPU 준비를 계속하고 본 task의 bounded 간격 관찰로 기술 결과를 확인한다. 별도 daemon/주기 callback/자동 재개 예약을 만들지 않는다.
3. 필수 actual Llama/native/teacher 반복/P-M mapping/C45678 coverage 등을 원 설계대로 검증한다. 기술 오류는 기존 승인 범위의 최소 수리·필요한 재검증으로 해결한다. gate 생략/허용치 완화/과학식 변경 승인 아님. 실제 설계 모순·필수 권한 부족 등 해결 불가 blocker는 정확히 보고하되 이를 GPU pending로 표시하지 않는다.
4. technical READY와 frozen source/config/input/dependency closure 후 N4/F48/G48/C4/C48/C45678 **6/6 본실험**을 upfront 정상 등록한다. 기존 등록이 있다면 exact receipt로 재사용하여 중복 제출0. 동일 W0/M0, first1000 원순서, B100×10 및 원 수치·평가·history5 계약은 변경0.
5. 모든 main의 owner/source/args/자원/dependency/held inspection과 release를 기록한다. 수동 hold·실수로 남은 dependency·기술 미완료를 정상 제출 완료나 자원부족으로 뭉뚱그리지 않는다.
6. 시작 가능한 대표 main(기존 계획대로 가능하면 C45678)의 실제 B1 candidate selection → final commit/M4..M8 각1회 append → B2 entry 연결을 확인하여 **MAIN_INITIAL_VALID**를 보고한다. N4 또는 기술 pilot만 확인하고 여섯 main 전체 검증으로 확대하지 않는다. 초기 gate는 기술적 실행 정상성이지 성능 개선 gate가 아니다.
7. 이 본실험 초기 gate가 통과하면 즉시 agent 모니터링을 중단한다. 모든 arm의 첫 batch나 W10 종료까지 기다리지 않는다. 제출된 프로그램/대기열은 변경 없이 자연 진행한다.

## 본실험 PENDING 예외 — GPU 할당 부족인 경우만

다음 조건을 함께 만족할 때에만 MAIN_GPU_RESOURCE_PENDING_HANDOFF 후 중지한다.

- 필요한 기술 READY가 확인되어 여섯 main을 정상 등록·검사·release했다.
- 초기 gate를 관찰할 실행 중 main이 없고, main이 실제 GPU 할당 부족으로 대기하고 있다.
- exact job ID/state/reason/time과 한정 resource 관측으로 GPU 자원 부족 근거를 기록했다. “Reason=None”, 새로 release 직후 일시적인 PENDING, 기술 Dependency, manual hold, source/설정 오류, CPU·host-memory 원인은 자동으로 이 조건을 만족하지 않는다.
- Resources/ReqNodeNotAvail 같은 reason도 GPU 부족임을 확인한 범위만 말한다. Queue 원인이 불분명하면 필요한 bounded 진단을 계속하며 추측으로 pause하지 않는다.

일부 main이 RUNNING이고 나머지가 cap2/array throttle 때문에 기다리는 정상 대기열이라면, 대기 arm이 있다는 이유로 일찍 멈추지 말고 **실행 중 main의 초기 gate**를 확인한다. 6개 main의 전체 완료나 모두의 시작을 기다리라는 뜻도 아니다.

## 자원·중간보고·종료

GPU cap2, 각 main 기본1GPU/8CPU/mem60416M/exportNONE/Requeue0. 기존 task와 admitted pending의 동시 가능한 용량까지 포함한다. 가용하면 두 GPU를 활용하되 기존 다른 job 변경·취소0. 이번 override는 monitoring 종료 경계만 바꾸며 신규 arm/추가 baseline/10k/repair 후속 scope를 추가하지 않는다.

direct 중간보고를 계속 보낸다: 현재 기술 상태와 actual gate/실측 비용, 6/6 main 제출 mapping 및 cap 확인, MAIN_INITIAL_VALID 또는 근거 있는 MAIN_GPU_RESOURCE_PENDING_HANDOFF. GH의 중복 raw/GPU 감사나 단계별 재승인은 선행조건이 아니다.
종료 시 source+compact 제출/초기인계 own-scope main 게시 권한을 유지하고, 이후 polling/sleep loop/heartbeat/terminal 대기/후속 submit/상세 분석/보고서 확장/자동 재개를 중지한다. 다음 USER recall을 기다린다. Main jobs 자체를 hold/cancel하지 않는다.

기존 frozen execution21297ec19e7f5aecec16d2fdb14cc79380a1df94 및 archive/lock, historical pause, disk W/M checkpoint0/정확 crash-resume 불가, source/raw/teacher/타 paused task는 그대로 보존한다. Monitoring override는 새 task-local receipt로 만들고 옛 lock을 덮어쓰지 않는다. 필요한 실행 수리는 새 immutable attempt/source로 분리한다.

## 실행 envelope 상속

원 envelope messages/head/2026-09-17-sh4-sequential-local-z-allocation-v2.md의 session/read/write/Slurm/red/resource/원 설계/데이터/평가 계약을 상속하며 **초기 종료 조건만 이 문서로 대체**한다.

허용 write는 기존과 동일:
- project/run_scripts/sequential_local_z_allocation/
- /data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/
- audits/servers/server4/2026-09-17-sequential-local-z-allocation-v2/
- experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2/
- messages/acks/server4/2026-09-17-sequential-local-z-allocation-v2.md
- messages/server-heads/server4/2026-09-17-sequential-local-z-allocation-v2.md
- runs/odeedit_sequential_local_z_v2_s4_20260917/
- tasks/status/odeedit_sequential_local_z_v2_s4_20260917/server4.json

Slurm ALLOWED는 기존 bounded 필수기술/필요한 최소 기술수리 및 6개 main뿐. Preflight 및 정확 source/resource 검증 유지, scientific block은 숨기지 않는다. NO_BROADCAST_NOT_REQUIRED; 새 대량 전송·삭제0. Own-scope nonforce publication만 허용하고 원 정본/global/shared native 수정0.
최초 응답에는 **정정 적용·재개**를 확인하고 기술 단계에서 다시 PENDING 중지하지 말 것.
