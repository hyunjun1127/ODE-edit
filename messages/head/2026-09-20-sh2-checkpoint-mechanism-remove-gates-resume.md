# GH → SH2: 검증 gate 제거 후 남은 기전 분석 제출·완료

Instruction: `ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1`
Override nonce: `ODEEDIT-GH-SH2-CHECKPOINT-MECHANISM-REMOVE-GATES-20260920-R1`

## 1. 최신 사용자 지시: 일회성 면제가 아니라 검증 gate 제거

사용자는 처음 “저런 gate는 pass 처리하는 걸로 하고 rerun올려”라고 지시한 뒤,
**“아니 저런 gate 자체를 아예 없애. 저런 것 때문에 실험이 너무 지연된다”**고 정정했다.
이 최신 지시는 원 contract/dispatch의 수치 동등성·재현성 gate 및 C01 의존 차단보다 우선한다.
본 task의 해당 검증 gate를 production 실행 경로에서 제거하고, 이미 구현된 나머지 분석을 바로 제출한다.
NS 한 행과 M1 네 원소만 일회성 예외로 통과시키거나 더 느슨한 threshold로 다시 검증하는 요청이 아니다.

이번 변경 범위는 **checkpoint mechanism audit task**다. 다른 task의 보호 정책·전역 helper는 변경하지 않는다.
원 report/contract/failed receipts/threshold/실행 source는 역사 자료로 불변 보존한다.
새 실행은 별도 source/override/config/lock에 `diagnostic_gates_enabled=false`,
`numerical_validation=NOT_ESTABLISHED`, `validation_policy=REMOVED_USER_DIRECTED`를 봉인한다.
실행 의존성은 허용되지만 original numerical PASS를 조작해 만들지 않는다.

## 2. 실제 제거할 범위

- C00/C01 prefix-vs-full/repeat 및 B1 archive NLL·margin/Gram/weight-response 수치 재현을
  신규 제출의 prerequisite/exit criterion에서 제거한다. 완료된 C00/C01 자료는 그대로 참조하고 재실행0.
- `key_bank.py`의 C01 PASS 요구와 capture parity로 C02/F00를 실패시키는 분기,
  `operator_lane.py`의 C01 PASS/pilot numerical PASS/physical-affine/dense-factor/actual-delta parity 차단,
  `activation_lane.py`의 C01 PASS 및 archive/physical/cached/kernel parity 차단을 포함해
  **모든 동등성·재현성 검사 때문에 후속 실제 분석이 막히는 실행 의존성을 제거**한다.
- FP32/FP64 solve residual, 재구성 오차 등 실제 분석 계산에서 저비용으로 얻는 값은 보고 가능하지만
  사전 고정 최대오차 초과만으로 job 종료/후속 cell BLOCKED/재승인 대기를 만들지 않는다.
  이를 위해 추가 reference forward/dense solve/FD/repeat를 실행하지 않는다.
- 검증 전용 duplicated forward/backward/dense solve/hash-loop 및 별도 GPU validation job을 추가하지 않는다.
  B91 dense-vs-factor 등 오직 parity 인증을 위한 중복 계산은 생략한다.
  실제 연구 질문에 필요한 key/h0, history LU/RHS, reconstruction/modes, actual suffix margin/gradient는 실행한다.
  불필요한 parity 계산을 빼면서 실제 과학 결과 자체를 생략하지 않는다.
- 기존 gate 결과 파일을 `PASS`로 덮거나 모든 exception/assert를 무차별 무시하지 않는다.
  과거 C01는 FAILED 그대로, 새 실행 readiness는 `READY_WITHOUT_DIAGNOSTIC_GATES`처럼
  artifact 가용성과 사용자 정책으로 판단한다. Report의 compute completion과 numerical certification을 분리한다.

**유지할 실행 안전·정의**: model/source/sample/target/shape/dtype/ID binding, 필수 입력 존재,
index/range, NaN/Inf·실제 solve 실패, CUDA/OOM, 실제 파일 I/O 실패, 원본 read-only,
분모·sample order·single-writer, 모델 상태 복원 및 job admission을 유지한다.
유한한 결과의 작은 수치 차이를 이런 safety failure로 재분류해 다시 차단하지 않는다.
수식/precision/evaluator/token packing/표본·대조·통계적 정의는 변경하지 않는다.
실제 불능 결과는 상태와 원인을 남기고 독립 작업을 계속한다. 숫자 대체/요청 제외0.

## 3. 재사용·실행 계획

기존 A00/A01/B00/C00, 저장된 C01 subchecks와 panel 선정·114 CPU tests·입력2304 member 결속을 재사용한다.
기존 source1eae5def/실패51071 및 완료51116/원 terminal/main4c7a513a는 덮지 않는다.
원 비용230+39=269 allocatedGPU-sec는 prior로 분리한다. 새 비용만 신규 계산한다.
기존 archive·256-vector W/M CPU 분석·입력 staging·원 B1 evaluation·native z fitting을 반복하지 않는다.

새 root: `local/checkpoint-mechanism-audit/20260920-v1/resume-no-gates-r1/**`.
원 `attempt-v1`은 입력 read-only, 새로운 source/archive/locks/output/control은 새 root에 보존한다.
완료된 key 부분은 identity가 일치하면 재사용하고 부족한 native700+geometry512를 준비한다.
준비된 key bank 뒤 13history/operator/7native-demand/control 및 activation/suffix 두 구간을 수행한다.
원 pilot4는 계산 순서·비용 산정에만 쓸 수 있고 numerical PASS 확대 gate로 유지하지 않는다.
원 31cell schema 중 검증 전용 상태는 SKIPPED_USER_DIRECTED 또는 역사 상태로 표시하며,
후속 과학 cell을 C01 실패의 자동 BLOCKED로 계속 보고하지 않는다.

**GPU cap2를 모두 활용**: 각 lane1GPU/8CPU/host≤60416MiB/exportNONE/Requeue0.
공통 key 등 필수 입력 의존성만 지키며, 준비된 independent operator/history와 activation/demand를
두 lane에 중복 없이 배치한다. Factor/output의 single writer를 지정한다.
다른 allocation+admitted pending과 동시 RAM/disk를 제출 직전 확인하고 project 총2를 넘지 않는다.
타 job 취소/hold/변경0. 지정 분석 job의 source freeze→held inspection→release까지 승인한다.
wall/storage는 실제 남은 작업과 기존 측정으로 산정하되 사용자 GPUh hardcap은 미지정이다.

변경 경로의 좁은 CPU routing/import/compile 검사를 수행하고 즉시 정상 제출한다.
광범위 CPU 재검산·독립 audit·검증용 GPU pilot을 새 선행조건으로 만들지 않는다.
기술 오류는 선보고·failure/cost 보존 후 의미 불변 최소수리/새 attempt 재제출 권한을 유지한다.
모든 분석 완료와 최종 보고/main까지 계속하며 initial gate/PENDING pause를 상속하지 않는다.
자동으로 새 편집 chain/EN/GSS/z fitting/history append/checkpoint 저장을 추가하지 않는다.

## 4. 세션·소유·보고

대상 SH2/server2/session `01a0493a-074c-7f91-9a13-769116326fef`,
CWD `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit`.
별도 clean worktree/branch에서 원 dirty/공유 환경/타 task PAUSE를 보존한다.
원 envelope `messages/head/2026-09-20-sh2-checkpoint-mechanism-audit.md`의 exact write 범위와
transfer/read 권한·H1–H4 판단 권한을 상속하되 이번 gate 제거가 충돌 시 우선한다.
새 source 변경은 `project/run_scripts/checkpoint_mechanism_audit/**`에 한정한다.
원 global 설계/contract를 조용히 수정하지 말고 task-local override로 봉인한다.

새 최종 package:
`experiment-reports/servers/server2/checkpoint-mechanism-audit-2026-09-20-v1/no-gates-resume-r1/**`.
최종 `report-ko.md`에 과거 실패/검증 생략, 원 수치와 이번 분석 결과/조건부 해석을 함께 기록한다.
“무해한 FP32 오차로 입증”, “원 gate numerical PASS”, “완전 재현”은 주장하지 않는다.
Raw NLL·Gram 초과값/개수·기존 실제 delta/response 성공은 원 receipt와 연결하고 삭제하지 않는다.
H1–H4는 실제 새 측정에 따라 갱신하되 검증 미확립이 해석에 주는 한계를 유지한다. H5는 후속 질문만.
15 outputs/실제 도달 cell·비용/peak·명령/source/input/provenance/CSV 기반4종 그림을 새 보고에 통합한다.
Final CPU 요약 검산은 보고서 산술·분모·data consistency 목적이며 새 실험 지연용 수치 gate가 아니다.

M0에서 제거하는 호출/분기와 유지하는 실행 안전 항목을 짧게 밝히고, 다음 중간보고에
실제 jobID/source/lock/두 lane·재사용·신규 범위를 전달한다. Source+compact 보고 own branch/main
nonforce 게시까지 승인, raw/tensor/checkpoint Git0, `NO_BROADCAST_NOT_REQUIRED`.
최종 완료 후 TASK_COMPLETE_STOP. 추가 GH 승인·중복 감사 대기0.
