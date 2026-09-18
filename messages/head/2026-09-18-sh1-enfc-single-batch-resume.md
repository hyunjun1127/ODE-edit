# GH → SH1: 중단 M의 B1 single-batch 대조만 server1에서 재개

instruction_id: ODEEDIT-S06-ENFC-SINGLE-BATCH-M-RESUME-SH1-V1
nonce: ODEEDIT-GH-SH1-ENFC-SINGLE-BATCH-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-f93a-7b50-bca0-65438eab2062
server: server1 / physical devbox
expected_cwd: /mnt/raid5/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 최신 사용자 지시와 범위

> single batch 대조 실험 중단ㄷ시킨건 server1에서 이어서 진행하자. batch 1개만 test 해도 될 것 같아

이는 SH4에서 취소한 ENFC **M single-batch 대조**의 server1 이관·재개 승인이다. 오래된 R-GD/R-Q repair나 새로운 sequential 실행이 아니다. 첫 고정 batch **O0/b001, ordinal [0,100), unique100**, cold W0/zeroM4 native endpoint 하나에서 원 M 비교를 수행한다. 성능을 보고 B1 대신 다른 batch를 고르지 않는다.

- 최종 비교8arm: N4 / SCALE / CA / KL-P / EN-S / EN-F / EN-COV / EN-F4.
- 기존 B1의 CA-EXACT geometry와 RAND± paired diagnostic만 포함한다. 새 optimization arm/ablation/sweep는 추가하지 않는다.
- B2–B10 및 S/R/L은 SH1 신규 범위 밖이다. server4의 S50071_[0–3] 실행·source·상태·모니터링은 변경하지 않는다.
- 최종8 L4 endpoint를 보존한다. 이미 valid 완료한 동일 endpoint는 재사용하여 신규 계산량을 줄인다. 비교 분모는8arm×100이며 unique800이라고 쓰지 않는다.

## 정본 및 source

다음 문서와 실제 사용 source를 전체 읽고 source/contract/sample/import identity를 lock한다.

- plans/global/2026-09-18-single-layer-edit-preserving-correction-design-v1.md
- 같은 prefix의 contract-v1.json, cells-v1.csv
- plans/global/2026-09-18-single-layer-edit-preserving-correction-sh4-dispatch/references/{06_blue_l4_detailed_audit,07_single_layer_method_positioning}.md
- messages/head/2026-09-18-sh4-enfc-skip-t-all-m.md 및 storage-waiver-submit.md: T 생략은 유지하되 S4 저장공간 waiver를 S1에 자동 적용하지 않는다.
- experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/metadata-repair-M-r1/ 및 sequential-four-r1/initial/의 compact provenance.
- repaired M execution **87f65ea2abcbe7e77e04367f73a001d63443734b**, tree **980c04b3e998caab0284b855041f6ad3e2e30356**를 M source 출발점으로 사용한다. S runtime을 M에 혼합하지 않는다. 기존 receipt status 중복 수리 및 TorchVersion→builtin str metadata 수리를 보존한다.
- project/run_scripts/single_layer_edit_preserving_correction/의 실제 M binding/runner/runtime/geometry/optimizer/observer/reuse 코드. 최신 main6dd01ac2에 게시된 경로라도 실행87f65ea2와의 차이는 구분한다.

이 문서의 **B1 단일범위 + reuse-first + T 생략**이 원 설계의 T 필수/10coldbatch 일괄 재실행/확대 규칙보다 우선한다. 나머지 method 수치/후보 guard/optimizer 예산/평가 정의는 유지한다.

## 재사용·수신·안전한 재개

SH1이 B1 항목별 REUSE / EVAL_ONLY / RUN_MISSING / REFERENCE_ONLY 표를 만들고 누락만 계산한다.

1. SH4 M50050_0/_1은 사용자 요청으로 취소됐고 각4884초·합9768 GPU-sec이며 SIGTERM rollback은 NOT_VERIFIED다. 취소 process 자체를 resume 성공으로 취급하지 않는다. 기존 T/M 실패·중단 비용은 과거 비용으로 분리하며 신규 비용에 중복 가산하지 않는다.
2. S4의 `/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/M/attempt-metadata-r1/episodes/b001/` 아래 **정지한 B1 자료**부터 확인한다. 기본 native는 기존 cold7 보존 WN/target/K/zeroM/context capsule로 REUSE한다. 해당 exact capsule/geometry/initial gradient와 완료된 N4/SCALE/기타 arm의 seal·endpoint·observer를 사용한다. S50071 live output/로그/Slurm 조회는 금지한다.
3. 실제 존재·size/SHA·schema·source-state·request/context·teacher·P/M identity가 맞는 완료 arm은 optimization 재실행0. endpoint만 있으면 누락 observer만 실행한다. partial trial/미봉인 endpoint를 완료 arm으로 승격하지 않는다. 완결 resume state가 없으면 해당 미완료 arm만 immutable WN에서 재시작하며 부분 비용을 보존한다.
4. SH1을 필요한 **정확 allowlist pull의 sole writer**로 지정한다. repo-local S4의 완료/취소 B1 자료 및 그 manifest가 참조하는 보존 native/context/P/teacher/reference subset만 새 S1 `local/enfc-single-batch-m/20260918-v1/imports/`로 nonoverwrite staging→양쪽 SHA/size 확인→seal한다. 필요한 prefix는 S4 `/data/janghj/ODE-edit/local/{single-layer-edit-preserving-correction,local-z-adaptive-allocation,bg1-c4-ours-first,ep-tw1-c4}/` 안으로 제한한다. 정확 파일 allowlist를 먼저 남기고 live S 디렉터리는 제외한다. 직접 rsync가 필요한 경우 custom imports/nonoverwrite 예외 receipt를 남긴다. --delete, 원본 이동/삭제, 전체 결과/전모델 무차별 복사0.
5. 로컬에 같은 model/tokenizer/P/stats/teacher가 있으면 재사용한다. SH4에 경로/기존 seal 질의가 필요하면 metadata-only peer 요청이며 S 모니터링 재개를 요구하지 않는다. 적합한 teacher를 재생성하지 않는다.
6. S1 새 process는 로컬 pretrained W0/zeroM4를 결속하고 그 위에 같은 B1 native WN bytes를 복원한다. runtime/platform 경로·메모리 적응만 별도 source로 기록한다. hardware/library 차이로 bitwise parity를 가정하지 않는다. 관측 재사용의 조건이 부족하면 누락 동일 endpoint 평가만 수행하고 다른 비교값과 출처를 분리한다.
7. B1 nativefit을 관성적으로 재실행하지 않는다. exact native capsule이 실제로 없거나 부적합하면 이유와 대안을 보고하고, 조용히 다른 endpoint/target을 생성하지 않는다.

## 수치·과학 경계

- T=SKIPPED_USER_DIRECTED / full_numerical_validation=NOT_ESTABLISHED 유지. 새 cold8/FD/ULP/direct-gradient/대규모 T 재검증을 선행조건으로 만들지 않는다.
- L4 단일 weight, W0 fixed teacher, D K_E=0 공간, P_star/rank cutoff, FP32 materialization, 실제 response invariant, 개별 NLL/ID guard, Armijo, EN-F4 4×6, EN-COV W0 cumulative 목적, fallback/예산과 selection→official observer 분리 그대로다. T 생략은 method guard 삭제가 아니다.
- M 한 cold batch이므로 Past64는 빈 집합, arm 간 state/history 누적0. sequential history append 규약을 M에 복사하지 않는다.
- Report256은 열지 않는다. Current R/P/N(분모100/200/1000), strict/joint/true-new NLL/paired loss-gain/W0-correct N, Dev128 및 원 M greedy32·geometry/CA-EXACT/RAND± 범위만 기록한다. 불리한 결과/유효 fallback도 보존한다.
- one-batch diagnostic이며 M10 batch-cluster bootstrap·long-horizon/일반화/통계적 동등성 claim은 불가하다. SH는 사실·수치만 보고하고 GH 해석과 분리한다.
- 실제 기술 오류는 먼저 source-backed 오류를 보고한 뒤, 이미 사용자가 승인한 최소 기술 수리·해당 미완료 범위 재제출을 SH1이 직접 수행할 수 있다. 과학식/threshold 변경은 이 권한에 포함하지 않는다.

## 실행 권한·경계·모니터링

Slurm submission **allowed**: 기존 server1 project cap **2**, 한 job1GPU/8CPU, 기본 host mem182272MiB(현 ceiling을 제출 직전 확인), exportNONE/Requeue0. 현재 다른 project active/admitted를 SH1이 확인하고 총 cap 안에서 제출한다. batch 수를 늘려 cap을 채우지 않는다. 단일 B1 runner를 우선하며 arm 분할은 동일 native/재사용/원 규약을 해치지 않고 중복 계산이 없을 때만 허용한다. GPU-hour cap=null, wall/storage는 실제 B1 남은 계산·endpoint 저장량으로 사전 명시한다.

기존 **실제 본실험 초기 gate만 확인 후 pause** 정책을 유지한다. 필요한 B1 계산 전체를 프로그램에 등록·held 검사·release한 뒤 대표 실제 미완료 correction의 저장/reload/selection seal/observer/reset 경계(가능하면 EN-F)를 확인하고 MONITORING_PAUSED_AWAITING_USER. 이미 sealed EN-F를 재사용한다면 gate를 만들기 위해 EN-F를 반복하지 말고 새 실행 arm의 실제 초기 정상성을 구분한다. 모든 B1 결과 terminal을 자동 대기하지 않는다. 정상 release 후 실제 GPU 자원 부족으로 running main이 없는 경우에만 exact reason/resource 근거로 pending 인계 가능하다. 단순 준비/기술 pending은 종료 사유가 아니다.

초기 인계 뒤 polling/log/result/terminalwait/heartbeat/자동재개/후속submit/상세분석0; 이미 제출한 프로그램은 자연 진행한다. 사용자가 완료 recall하면 B1의 기존+새 비교와 비용/누락을 합친 상세 사실 보고를 작성한다.

## 파일 소유·게시·완료경로

SH1은 현재 session/cwd/origin 확인 후 전용 `codex/server1-enfc-single-batch-m-v1` branch와 새 worktree를 사용한다. 현재 app-server resume에서 SH1 CWD `/mnt/raid5/janghj/ODE-edit`를 확인했다(과거 registry의29e4 worktree 표기는 역사값). 새 task worktree는 이 repo common Git directory 아래로 결속한다. user dirty와 타 agent 편집을 되돌리지 않는다.

허용 write:
- 전용 branch에서 `project/run_scripts/single_layer_edit_preserving_correction/server1_single_batch/` 및 필요한 최소 기존 package 플랫폼/reuse 수정·그 tests. 수정마다 frozen M과 수치 차이 없음을 기록하고 원 SH4 frozen/source/root는 변경하지 않는다.
- `local/enfc-single-batch-m/20260918-v1/` (S1 repo 또는 전용 WT 아래): inputs/imports/output/checkpoints/logs/locks.
- `experiment-reports/servers/server1/enfc-single-batch-m-2026-09-18-v1/`.
- `audits/servers/server1/2026-09-18-enfc-single-batch-m/`.
- `messages/acks/server1/2026-09-18-enfc-single-batch-m.md`, `messages/server-heads/server1/2026-09-18-enfc-single-batch-m.md`.
- `tasks/status/ODEEDIT-S06-ENFC-SINGLE-BATCH-M-RESUME-SH1-V1/server1.json`, `runs/odeedit_enfc_m_b001_s1_20260918/`.
- `transfers/verifications/2026-09-18-enfc-single-batch-server1/`.

raw/tensor/teacher/prompt/full log Git0. 자체 source/data/reuse/resource·소유경계 preflight 및 초기 증거/manifest/보고서 렌더 postcheck를 수행한다. 별도 subagent는 복잡한 독립 작업에만 사용한다. CPU check를 actual numerical PASS로 쓰지 않는다. narrow runtime technical failure는 수정하되 불리한 성능/작은 효과는 중단 gate가 아니다.

자기 범위 source와 compact 재사용/제출/초기 보고는 검산 후 own branch 및 clean integration에서 nonforce main 게시 허용. canonical 설계/contract/SH4 보고는 수정하지 않는다. 큰 산출물의 전체 서버 방송은 불필요(NO_BROADCAST_NOT_REQUIRED); 선택 수신 provenance와 S1 경로를 보고한다.

첫 회신에는 FULL_READ, B1 8arm별 완료/누락·재사용표, exact import 계획, source/runtime 차이, S1 cap2 자원 계획을 보고하고 이미 승인된 준비/누락 실행은 재승인 대기 없이 진행하라. GH는 raw/GPU 중복 감사하지 않는다. 최종 제출 인계에는 실제 job/source/lock/output/initial 또는 resource-pending 증거와 남은 범위를 명시한다.
