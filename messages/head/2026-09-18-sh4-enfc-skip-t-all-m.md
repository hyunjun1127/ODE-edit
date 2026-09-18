# GH → SH4: T 생략, M 전체 재제출 / cap2

instruction_id: ODEEDIT-S06-SINGLE-LAYER-EDIT-PRESERVING-CORRECTION-M-SH4-V1
nonce: ODEEDIT-GH-SH4-ENFC-SKIP-T-ALL-M-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit
gpu_cap: 2

## 최신 사용자 지시 / 즉시 재개

> 그냥 T는 건너 뛰고 M전부 올려

이는 explicit USER recall이다. 직전 paired-stop의 자동재개 금지는 이번 사용자 호출로 해제하며, **새 T 실행/복구/continuation 없이 M의 필요한 전체10 independent cold episode를 정상 등록·검사·release**한다. 전체 M 비교는8arm×10=80 final L4 endpoints와 원 설계 RAND±/CA-EXACT 진단이다. 기존 재사용 지침은 유지하므로 B1 native fit과 적합한 기존 평가를 다시 수행하라는 뜻이 아니다.

기존 T_READY 선행조건, T49928 성공 dependency, T 미완료 모니터링, T 실패→linked M 취소 연동은 **이 새 M attempt에는 적용하지 않는다**. 이미 실패한 T49928 때문에 새 M을 다시 취소하거나 T READY가 없어 시작을 막지 않는다. 과거 T49928/M49973_0..9/frozen lock/paired-stop을 고치지 말고 새 immutable attempt와 이번 waiver를 연결한다.

## T 생략의 정확한 의미

- T 재제출·repair continuation·cold8 native repeat·별도 actual derivative/FD/ULP/direct-gradient/noop/teacher self-KL/parity 기술 pilot은 새 M 제출의 선행조건으로 실행하지 않는다.
- 상태는 **T=SKIPPED_USER_DIRECTED; full_numerical_validation=NOT_ESTABLISHED**. 기존 T에서 실제 저장된 부분 PASS와 미실행 FD/미보존 projector 판정을 분리 보존한다. T_PASS/T_READY/GPU 기술 전체통과를 만들어내지 않는다.
- M 자체의 정확한 source/config/input/data/reuse identity, 필수 계산과 finite/state/storage/memory 계약은 유지한다. 이미 CPU 검증된 receipt helper status 중복 수리는 사용한다. 필요한 좁은 import/CLI/routing/실행 가능성 확인만 하며, 새 광범위 기술검증을 T의 다른 이름으로 추가하지 않는다.
- **방법 자체의 candidate acceptance 조건은 제거하지 않는다.** D K_E=0 교정 공간/원 rank cutoff/FP32 materialization/실제 response invariant/current 개별 NLL·ID guard/Armijo/예산·fallback/P-N observer 분리는 원 계약 그대로다. 이는 별도 T gate와 구별되는 M 방법의 정의다. T 생략을 과학식·threshold·method guard 변경으로 확대하지 않는다.
- 실제 M runtime의 OOM/NaN/shape/wrong-state/serialization 등 기술 오류는 실패로 보고·보존한다. 미검증을 성공으로 숨기거나 실패를 과학적 효능 결론으로 바꾸지 않는다.

## M 재사용·cold 실행·저장

1. 현재 endpoint 완료0/평가0인 기존 cancelled M을 성공 결과로 재사용하지 않는다. B1은 검증된 cold7 retained WN/native targets/K/zeroM/context capsule을 계속 REUSE한다. B2–B10은 매번 W0/M0인 독립 cold100이며 shared native 최대9회/900target이다. 과거 sequential B2+를 독립 cold로 바꾸지 않는다.
2. 기존330행 reuse matrix와10episode 계획을 새 source/attempt에 연결한다. REUSE fit/optimization/observer 중복0, EVAL_ONLY는 누락항목만, RUN_MISSING만 새 실행. 동일 endpoint의 canonical pair 등 기존 관측을 identity에 맞춰 재사용한다.
3. 각 episode의8arm이 같은 native z/Δ_N/W_N/A/K/P/M/teacher를 공유하도록 유지한다. 다른 episode 상태/history를 넘기지 않는다. 원 M cells의 cold independent history0와 sequential history 계약을 혼동하지 않는다. S/R/L 제출0.
4. cancelled b001의 부분geometry는 serialization/identity 완결성을 확인한 경우에만 재사용한다. 중단된 process rollback NOT_VERIFIED를 복구 성공으로 간주하지 않는다. 안전한 W0/검증된 native capsule에서 새 process를 시작하며 fullmodel 복제나 이전 native fit 재실행을 관성적으로 하지 않는다.
5. 새 M final80 L4 endpoint 보존, large raw/local-only, teacher 재생성0 기본. 기존 source/raw/teacher/로그/실패 자료 삭제·덮어쓰기0.

## 제출·자원·종료

- exact old T49928 및 M49973_[0-9]의 terminal/기존 paired-stop receipt를 결속하고, 제출 직전 현재 server4 본인 project active/admitted 동시가능 용량만 한정 확인한다. 다른 task 과학결과 점검/변경0.
- 다른 점유가 없으면 새 M array10개 **%2**로 upfront held inspection→release한다. cap2를 실제로 사용하며 예전 T lane 때문에 %1을 계속 유지하지 않는다. 각1GPU/8CPU/60416MiB/exportNONE/Requeue0, wall/storage reserve는 기존 실제 계획을 재확인하여 lock한다. 사용자 GPUh cap=null.
- 새 M 실행경로에는 T afterok/READY requirement/old failcancel watcher를 제거하는 명시적 waiver routing을 둔다. 과거 receipt를 수정하지 않는다. 원 method/controller/core 수치 변경0이며 source delta를 최소화한다. M source76bb903의 helper 수리 및 기존 CPU82 receipt는 적용 범위를 확인해 재사용하고 광범위 재검증을 반복하지 않는다.
- job mapping/실행 source/tree/archive/input/waiver lock/각 resource와 실제 submission 상태를 바로 GH에 보고한다. 제출 완료를 M 완료 또는 actual initial PASS로 부르지 않는다.
- 기존 최신 **실제 M 본실험 초기 gate 후 모니터링 중지**는 유지한다. 새 M 전체 정상release 뒤 대표 M의 실제 EN-F 경로(정상0correction/native fallback 포함), 실제 저장/복원·observer nonmutation 등 실행 정상성을 필요한 최소 범위에서 확인한다. 별도 T로 돌아가거나 모든80endpoint/전체10episode 완료를 기다리지 않는다. T 미완료는 이제 pause를 막는 조건도 아니다.
- main GPU 할당 부족으로 관찰할 running M이 없으면, exact job/reason/time 및 실제 GPU 부족이 확인된 경우만 기존 pending 인계 예외를 적용한다. Reason=None/수동hold/구Tdependency를 자원부족으로 쓰지 않는다.
- 초기 인계 뒤 agent polling/log/scheduler/result조회/terminalwait/heartbeat/callback/자동재개/후속submit/상세분석 확대0. 제출된 M 프로그램은 자연진행, S/R/L 자동확대0. 사용자 recall 후 상세분석한다.
- 실패하면 정확한 actual 경계·자료·비용과 미완료 범위를 보고한다. 별도 새 science 자동추가0. 이전3231 GPU-sec와 새attempt 비용은 분리하고 batch/extern/재사용teacher/native 중복가산0.

## 권한·게시

원 M envelope의 session/read/write/source보존·raw-free·PNG code-only·ownscope nonforce main 게시권한을 상속한다. 새 local namespace는 기존
/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/
아래 gate-skip/M 재실행 새 attempt 및 control/receipts로 분리한다. 원 global design/contract/cells/shared native/환경은 수정하지 않는다. source/config의 waiver만 versioned control로 봉인한다.
관련 문서:
- messages/head/2026-09-18-sh4-single-layer-edit-preserving-correction-m.md
- messages/head/2026-09-18-sh4-enfc-t-m-parallel-failcancel.md (이번 새 attempt의 T 연동은 superseded)
- experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/submission-and-paired-stop-ko.md (역사자료 불변)

최초ACK에서 **T생략/새M10episode cap2/재사용유지/수치검증미확립/초기M후pause**를 확인하고, GH의 중복raw감사나 재승인을 기다리지 말고 제출까지 진행하라.
