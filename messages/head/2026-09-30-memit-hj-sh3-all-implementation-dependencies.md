# MEMIT HJ 전체 구현 점검과 의존성 선등록 추가 지시

Instruction / ACK nonce: `ODEEDIT-GH-SH3-MEMIT-HJ-ALL-IMPL-AUDIT-DAG-20260930-R1`.
Parent: `ODEEDIT-GH-SH3-MEMIT-HJ-V2-20260930-R1`.
GH → registered SH3 `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`, server3.
사용자 원문: “설계대로 구현을 모두 진행하게 하고, 코드 점검도 자세히 돌리게 시켜. 그리고 실험들 dependency 걸어서 pending으로 걸어놓으라고 해”.

## 이번 지시의 의미

기존 동일 task를 계속한다. 설계 v2의 전체 구현과 상세 코드 점검을 마친 뒤, 승인된 실험을 실제 dependency가 있는 Slurm jobs로 선등록하고 held inspection 후 release하라. T0 하나만 제출하거나 계획 JSON만 작성한 뒤 후속 실험 등록을 사용자 recall에 넘기지 않는다. GPU가 당장 없더라도 정상 release된 자원 또는 dependency 대기열로 등록한다. Pending을 만들기 위한 manual hold는 남기지 않으며, 자리가 있으면 정상 실행되어도 된다.

서버3 project/task cap은 **1**이다. 이번 지시는 arm·표본·수식·교정 threshold·checkpoint 예외를 바꾸지 않는다. 원 정본9개와 부모 envelope를 그대로 사용한다. 기존 준비·구현과 다른 task는 보존하고 같은 nonce 제출을 중복 생성하지 않는다.

## 전체 구현과 상세 코드 점검

본체8 logical arms, writer12와 history8의 28개 cell을 구현·실행·평가·수집 경로에 모두 연결한다. 설계 요구사항별 실제 함수/파일·test·runtime evidence 계획을 연결한 coverage 표를 작성한다. TODO·placeholder·예외를 정상 완료로 처리한 경로·누락된 CLI/실제 import·미구현 resume/reducer를 남긴 채 전체 구현 완료로 보고하지 않는다.

정적 코드 검토, production 함수 경계 fixture, 통합 dry run과 오류 주입을 상세히 수행한다. 최소한 다음을 포함한다.

- 고정10k order와 BS100 main / BS10×100 진단, anchor·past400·평가 시점·28cell 중복 및 누락, 실제 물리 실행과 공유 alias 집계.
- pinned BLUE native와 task-local adapter 차이, fulljoint 행렬 곱 순서, 비가환·특이 PSD Gram, 직접 solve 전환, L8 직접 residual, 실제 FP32 write 및 energy-matched shadow 복원.
- SPG Armijo·budget·zero step·반환점 재평가와 projected gradient, 6수락점 plateau window, 32요청 교정의 실패/미도달 포함 집계, 실제 FP64 oracle 경로와 dtype/cache/state 원상복구. Gradient cast만으로 FP64 검사 대체 금지.
- 모든 5층 write 뒤 history exactly-once, duplicate/superseded occurrence 유지, refresh drift/reference reset, 독립 RAM fork의 W/H/RNG/context/cache/ledger alias 차단.
- 임시 checkpoint 전체 상태 직렬화·재로드·실패 원자성·복구, source/config/data binding, 최신2개/refcount 보존과 삭제 가능 시점. TorchVersion 같은 metadata와 가상모듈 provenance 경계를 포함한다.
- R/P/N preference와 TF token-micro/prompt-macro/strict, true/new NLL, raw identity·ties·분모·observer 비변이, 계산 비용 중첩/추가 oracle 계측.
- 실제 launch argv/node/memory/source/lock, job dependency와 실패 전파, collector가 보고·manifest 쓰기 전에 COMPLETED를 기록하지 않는지, 중복 제출 방지.

발견한 구현 오류는 과학 정의를 바꾸지 않는 범위에서 수리하고 회귀검사를 추가한다. CPU PASS와 실제 모델 검증은 분리하며 설계 밖의 광범위 GPU 검증 캠페인을 추가하지 않는다. 코드 점검은 설계 구현의 누락·오류를 찾는 절차이며 성능이 낮다는 이유로 arm을 제외하거나 새 gate를 추가하는 절차가 아니다. owner audit와 별도 reviewer 수행 여부도 사실대로 표시한다.

## Dependency 선등록

설계의 state lifetime과 cap1을 만족하는 실제 physical job DAG를 봉인하고, 각 logical cell이 어떤 job 및 내부 단계에 포함되는지 전량 mapping하라.

1. 공통 T0a, W0 E0/T1, native000 첫1k 및 T0b 교정, non-Z 경로/anchor 진단, 교정 조건부 Z 경로, 최종 CPU collector를 빠짐없이 연결한다. CPU fixture만으로 모델 T0나 교정을 미리 PASS하지 않는다.
2. 실제 기술·자산 선행조건에는 afterok와 원자적 source/config-bound readiness receipt를 사용한다. 교정 결과가 필요한 Z job도 미리 등록할 수 있으나 실행 시 정확한 calibration lock/PASS를 소비해야 하며 아직 모르는 tol/cap을 임의 상수로 확정하지 않는다.
3. Z 교정 실패는 Z 의존 경로만 BLOCKED 처리한다. 이를 non-Z 000/100/001/101 및 독립 writer/history 실험까지 막는 단일 afterok chain으로 만들지 않는다. cap1의 자원 순서와 과학 readiness는 별개로 설계한다. 순서용 afterany를 쓸 경우 해당 job 자신의 과학 선행 receipt 확인을 생략하지 않는다.
4. 최종 CPU collector는 지정 GPU jobs의 afterany로 실패·차단·alias·완료를 모두 수집한다. 기술 실패를 성공으로 바꾸거나 dependency 실패를 science 음성 결과와 혼동하지 않는다.
5. cap1은 여러 pending job의 등록 금지가 아니라 최대 동시 GPU 상한이다. 모든 root/diagnostic/calibration job 사이에도 실제 최대 동시1이 유지되도록 직렬 dependency 또는 동등한 명시적 concurrency 제어를 구현·검사한다. 다른 task의 기존 점유도 admission에 반영하고 임의 취소하지 않는다.
6. RAM-only parent/child fork나 anchor 진단을 억지로 별도 프로세스로 쪼개어 상태를 잃거나 prefix를 중복 실행하지 않는다. 이런 부분은 같은 persistent job 안의 사전등록된 내부 DAG로 묶되, 독립 가능한 job groups는 실제 Slurm dependency로 연결한다. 28개 logical cell마다 별도 job 28개를 만들라는 뜻은 아니다.
7. Job 경계의 상태 전달은 설계가 허용한 임시 checkpoint와 exact resume에 한정한다. dependency 편의를 위해 단기 경로·진단의 새 checkpoint 예외를 추가하지 않는다. source를 변경할 때 기존 frozen 실행을 덮어쓰지 않는다.
8. Wall/storage/FP64 peak와 RAM fork 계획을 사전 점검한다. 교정 전 계획값과 실측값을 구분하며 실제 교정 결과에 의존하는 runtime lock은 후속 runner가 읽도록 한다. 사전등록 때문에 임계값·표본·round를 줄이거나 저장되지 않은 상태를 가정하지 않는다.

소유자·source·full argv·resource·dependency를 held 상태에서 확인한 뒤 정상 release한다. 후속 수동 사용자 승인이나 SH의 다음 turn을 실행 조건으로 두지 않는다. Job 간 dependency뿐 아니라 설계의 분기/교정 조건도 유지해야 한다.

## 인계와 모니터링

전체 구현 점검 결과와 실제 job ID → 단계/cell → dependency → resource → source/lock 표를 GH에 보고한다. PLANNED / IMPLEMENTED / REGISTERED / RELEASED / ACTUAL_PASS / BLOCKED를 구별한다. 미등록 항목을 등록 완료로 적지 않는다.

이번 지시는 **전체 구현·코드 점검·dependency 선등록을 우선 완료**하라는 추가 지시다. GPU가 없는 동안 준비 작업을 끝내고 후속 실험을 queue에 올리지 않은 채 T0 pending에서 멈추지 않는다. 전량 등록 후의 초기 gate 관찰/실제 자원 pending 인계 및 이후 polling 중지 경계는 부모 envelope를 유지한다. 등록 runner/collector는 자연 진행하며 초기 인계 뒤 agent heartbeat·자동 recall·중복 submission은 없다.

추가 USER/GH 재승인을 기다리지 말고 같은 task에서 진행하라. 이 메시지의 짧은 nonce ACK를 현재 turn에 남기고 작업을 계속한다. GH는 ACK만 짧게 회수하며 구현 완료·job 제출을 ACK로 대신 주장하지 않는다.
