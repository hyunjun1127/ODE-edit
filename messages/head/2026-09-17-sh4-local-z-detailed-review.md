# GH → SH4: Local-z 7-arm 상세 리뷰 — CPU 분석·보고·main 통합

instruction_id: ODEEDIT-S06-LOCAL-Z-SEVENARM-DETAILED-REVIEW-SH4-V1
nonce: ODEEDIT-GH-SH4-LOCAL-Z-DETAILED-REVIEW-20260917-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 1. 사용자 요청과 이번 재개 경계

사용자 원문: “server4 실험 자세히 리뷰시켜”.
직전 local-z adaptive allocation의 준비 및 7개 정책 결과에 대한 명시적 review recall이다. 이전 CAKE/baseline·alpha-cap 보고서 분리와 SL-ZFlow는 완료/별도 scope로 보존한다. Server4 GPU cap은 최신 사용자 지시대로 **1**이며 이번 리뷰는 **새 GPU 사용0**이다.

본 지시는 CPU 분석·한국어 상세 보고서·본 scope main 통합을 완료할 때까지 진행할 권한이다. 과거 INITIAL_GATE_ONLY/PENDING pause는 이 한정 리뷰에 대해서만 해제한다. 새로운 과학 실행·기술 replay·평가·teacher 생성·모델 로드·재제출·cancel/requeue/hold/dependency/throttle 변경은 허용하지 않는다. 반복 scheduler/result polling, 다른 paused task 재개도 하지 않는다.
완료를 가정하지 말고 정확한 지정 job의 상태와 저장 산출물을 구분해 확인한다. 일부 미완료/실패라도 독립적으로 가능한 완료 결과의 CPU 분석과 부분 보고는 진행하며, 누락을 채우려고 GPU 실행하지 않는다. 불완전 결과를 full 7-arm 또는 전체70batch PASS로 승격하지 않는다.

## 2. 정본·실행·입력 재결속

최신 origin/main에서 별도 clean branch/worktree를 만든다. GH 확인 base는 e62824df52149d947ee4836e59863b1e12a55b7d / tree65a288f5d186cf4e60fa6fae4ec77635011d67b7. 사용자 dirty/기존 worktree/실행 source와 잠긴 원 raw를 보존한다.

다음을 전체 읽거나 exact SHA와 기존 FULL_READ receipt로 재결속하고, 새 envelope와 실제 아직 읽지 않은 분석 대상 코드는 직접 읽는다.
- plans/global/2026-09-16-local-z-adaptive-allocation-design-v1.md, contract-v1.json, cells-v1.csv 및 기존 CPU 참조.
- messages/head/2026-09-16-sh4-local-z-adaptive-allocation.md, 해당 dispatch/source-input-manifest와 cap1 정책.
- experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/preparation-submission-ko.md, 원 pending/자원/source/receipt, 최신 PROTOCOL.
- 실제 frozen 실행32a92ad6f3fff2f258d8778f3936d152e975ac1b / tree75a96b2e2122d2be6b096af12c22df8a4365a8e1. Archive6378df3a237d5a6c489b87c90f6d99c0221b409857632852916f2f5bffa7f901, execution.lock093bb13dd6b42b8f2f8b8478248b067add1d94533547afa8ca9af416e6f14629. 실행 source와 이번 analysis/publication source를 구분한다.
- 원 local root /data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/; resume-manifest.json SHA541012971022dd245df9aa64d085379d931116e7eb4dad2eba085a31e0920ebe.
- W0/zero M4/M8, model revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager, 양 TF32off, seed20260916, 공통 context/token/RNG/P4asset0/P8asset4.
- fixed10k datasetSHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1, wholeorder5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729, first1000 orderedroot40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
- C4 reference768 및 S64/Dev128 teacher의 기존/재생성 여부, actual common capsule/READY와 사용 teacher identity. seed20260915(C4)와 method seed를 분리한다.

GH는 위 문서·출판 상태만 확인했으며 원격 job/scientific raw를 재감사하지 않았다. GH 중복 감사나 단계별 재승인을 선행조건으로 요청하지 말고 SH4가 이 리뷰의 검산을 책임진다. 이 review에 새 global design 또는 미게시 파일을 조용히 추가하지 않는다.

## 3. 한정 상태 확인과 첫 표

정확한 48679(공통 준비)와 48680_[0-6](과학 array)만 한 번 bounded scheduler audit한다. child ID는 해당 array 관계로 해석하고 owner/node/resource/exit/allocation을 기록한다. Mapping은 0 LD, 1 N4, 2 REFIT4, 3 L75, 4 T75, 5 L4D, 6 TD이며 보고 표는 N4/REFIT4/L75/T75/L4D/LD/TD 순서로 정리한다.
기존 agent 마지막 관측은 2026-09-16T11:36:50.122707+00:00의 PENDING/actual gate NOT_OBSERVED였음을 유지한다. 지금 저장 증거로 확인하는 사실을 당시 관측으로 소급하지 않는다.

- COMPLETED와 과학 terminal-valid/commit/분모 검증을 분리한다.
- RUNNING/PENDING은 그 상태만 보고하고 live tensor/raw를 분석하거나 완료까지 대기하지 않는다.
- FAILED/TIMEOUT/CANCELLED/READY 미생성은 정확한 첫 원인·완료 경계·미실행 arm·계산 비용을 read-only RCA한다. Dependency 때문에 미시작한 arm을 과학 실패로 표기하지 않는다. 수정·재제출은 본 recall에 포함되지 않는다.
- 완료·고정된 raw의 first W10 full1000 표를 독립 NLL reducer로 먼저 전달한다. 미완료는 NA와 사유를 적는다. 분모 RS1000/PS2000/NS10000, count/percent/N4대비 pp와 요청·prompt·target identity를 확인한다. Missing을0으로 채우거나 aggregate만으로 paired row를 합성하지 않는다.
- 첫 표는 metric 수준의 검증과 전체 source/state 검증의 남은 범위를 표시한다. 이후 상세 분석을 별도 승인 없이 계속한다.

## 4. 핵심 성능·retention 비교

실제 저장된 true/new NLL로 canonical RS/PS/NS(엄격 부등식, tie=failure)를 독립 재집계한다. TF strict·two-P strict는 별도 지표로 둔다. Finite/cardinality/중복/누락/order와 actual evaluation endpoint를 확인한다.

1. 새 matched 7arm W10 전체1000, W5 first500, W10 동일first500/후반500을 분리한다. 각 batch Current100 및 pooled at-write를 fullseen final과 혼합하지 않는다.
2. Entry→at-write, at-write→final, W5→W10 same-first500의 loss/gain·NLL/margin 변화·p95/p99/tails를 exact case/prompt/target ID로 연결한다. W0 관측은 실제 있는 공통 평가만1회 사용한다.
3. ALL/ACTIVE/SUPERSEDED와 합법적 target overwrite를 구분한다. 저장 checkpoint 사이 실패/회복 시점을 보간하여 정확한 최초 실패라 부르지 않는다.
4. 주 비교: LD−L4D, LD−L75, LD−REFIT4, LD−TD, T75−L75 및 모든 정책−새 N4. RS/PS/NS/strict와 비용을 함께 표시하고 서로 반대 방향인 결과도 유지한다.
5. historical BLUE/L4/MEMIT/AlphaEdit/CAKE 등을 넣는 경우 기존 봉인 표의 비교 가능한 first1000만 별도 참고표로 둔다. Warm5000→6000/full10k와 이번 cold first1000을 같은 최종 분모로 비교하지 않는다. 원래 hparams/seed/context/kernel/history 차이와 검증 수준을 표시하고 이전 결과로 새 N4/REFIT4를 대체하지 않는다. 다른 보고서와 raw를 무제한 재감사하지 않는다.

## 5. “설계대로 실제 어떻게 동작했는가” 필수 장

문서 선언만 나열하지 말고 실제 frozen file/function/line 또는 source SHA, 저장 필드·수치·상태를 연결한 requirement→implementation→observed evidence→판정/한계 표를 만든다.

- LOCAL: own-entry z4/N4 뒤 각 a4 partial에서 fresh z8/h8/K8, a8끼리만 fit 공유. TERMINAL: entry Z8 고정, L4 residual Z8−h8(entry), L8 residual Z8−h8(partial), writer-layer key. Z8−h4 혼용 여부와 actual FP32 endpoint gate0/1/.5/.75, 모든 token에 physical weight 적용 경로를 확인한다.
- REFIT4는 동일 L4 두 번째 fresh fitting인지, write-refresh/과거warm 경로를 실행한 것이 아닌지 확인한다.
- Common cold capsule·zero memories·branch target cache identity·후보 생성 및 평가 순서 격리·teacher 재현/재생성·준비 READY를 실제 저장 근거로 점검한다.
- Dynamic own-entry N4는 독립 N4 chain과 다르다. E token→context→request, H/Past64 canonical, S64 fixed W0 full-vocab KL 평균 정의·strict ID집합·epsilon/plateau/tie 순서가 실제 적용됐는지 검산한다.
- E≤max(E_N4,.05)+1e-4, Current strict subset; Past 존재 시 H≤H_N4+1e-4 및 Past strict subset; Dmin+1e-6 tie의 N4→L8변화0→작은 실제 concat norm→ID 순서를 stored candidate scores/IDs로 CPU 재선택하고 실제 selected와 대조한다.
- 후보별 E/H/D/strict/tail, feasibility/각 조건의 탈락, raw와 selected, N4 선택률, a4/a8 분포, dedup/equivalence를 batch별 공개한다. TD의 common N4와 terminal 후보를 분리하고 fixed L75/T75에 dynamic screen을 적용했다고 쓰지 않는다.
- Received-event Past64의 active fact/현재 overwrite 제외/최신 target 반복event/SHA 우선순위와 B1empty, arm간 동일 입력 ID를 검산한다. P/N/Dev observer가 selector·target 선택에 들어가지 않았는지 source와 저장 근거를 나눈다.
- Candidate inner history0; N4/REFIT4/L4D는M4만1, L75/T75/LD/TD는M4/M8 각1. LD/TD의 a8=0/N4 선택에도M8 append가 있었는지 확인한다. 전체70batch 완료 시 계획 총history110이며 실제 완료 수로 검산한다.
- Stable checkpoint·commit·parent W/M/context/RNG/ledger와 selected final keys를 대조한다. 가능한 selected endpoint 재구성은 저장된 native fits/entry/gates를 CPU로 검산하되 FP32 delta만으로 exact reconstruction을 주장하지 않는다.
- 한 대표 LD 선택 batch, common N4 선택 batch가 있으면 그 예, LOCAL/TERMINAL 같은 gate의 예를 수치로 풀어 설명한다. 존재하지 않는 사례를 만들지 않는다.
- NaN/손상/OOM이 N4 fallback으로 숨겨졌는지, 기술오류를 품질탈락과 혼동했는지 확인한다.

선택 S64 D와 observer Dev128/공식NS의 관계, Current 평균 E와 개별 요청·PS·Past의 실제 손실을 따로 보여준다. no-plateau/strict-only shadow는 stored scores의 같은-state 재선택만; 새로운 forward/commit/반사실 lifelong 결과 합성0. 관측치 연관·산술차이만 보고하고 S64 개선=실제 locality 보장, a8>0=capacity 증명, N4 fallback=terminal allocation 성공이라 하지 않는다.

## 6. 검증 수준·계산 비용·누락

실제 output inventory/fullSHA·size와 source/config/sample/P/teacher/capsule 연결을 검산한다. 신규 실험의 source/runtime archive와 결과를 대상으로 하되 변경 없는 대형 model/teacher는 prior verified identity+현재 stat 등의 재사용 수준을 명시한다. 모델 재로드·GPU continuation·FP32 parity 재실행은 하지 않는다.
B1/B5/B10×7=21CP와63batch links는 전체 완료 시 예상값이다. 실제 존재하는 CP의 selected tensor shape/dtype/finite/hash/history/context/RNG/next-index를 CPU weights_only로 확인하고 미저장 필드는 NOT_RECORDED. Saved CPU parity, runtime guard, GPU continuation을 구분한다. 실패 raw/기존 pending 기록은 immutable로 보존한다.

계획 target12000/solves150/후보최대190와 actual target/loss/Adam/clamp/solve/copy/history/eval 수치를 대조한다. Dedup로 actual unique endpoints가 줄어든 사유와 선언 후보 수를 따로 기록한다. Clamp hit 미기록이면 final anchor/radius로 횟수를 추정하지 않는다.
준비48679·science7·teacher 재사용/재생성·실패 비용을 분리한다. Allocation GPU-sec와 program wall, component timer를 분리하고 nested 중복가산0. Load/context/prefix·target·solve·candidate Current/Past/S64·observer·teacher streaming·commit/CP/I-O·peak GPU/host·실제disk를 가능한 범위에서 보고한다. Purewriter나I/O 분리계측이 없으면 NOT_SEPARATED. 할당시간을 utilization으로 부르지 않는다.
시작 당시 cap1/dependency%1과 exact job intervals를 비교하되 불필요한 전체 사용자 queue 조회 없이 지정 job 기록으로만 검산한다.

검산 실패는 보고서 metric/state의 판정을 낮추고 첫 원인·영향 범위를 적는다. Analysis/reducer/plot/schema의 CPU-only 버그는 원 raw 불변으로 수정·재검산 가능하다. Runtime/과학식/임계값 수리·재실행은 별도 사용자 지시가 필요하다. 낮은 성능은 기술오류 또는 분석중단 gate가 아니다. 실제 독립 reducer/자체 검사/별도 reviewer의 수행 수준을 사실대로 적는다. 기존 CPU12가 real-model PASS인 것처럼 쓰지 않는다.

## 7. 산출물·main 게시·허용 경로

독립 branch codex/server4-local-z-detailed-review-20260917-v1 및 새 분석 결과 디렉터리를 사용한다. 원문·실행·pending·preparation 보고는 수정하지 않는다. 최종 canonical 한국어 보고서:
experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/diagnostic-report-ko.md
동 package에 first-final-table.csv, final/paired/retention/candidate/selection/constraint/compute/source-state-coverage 표, code-generated PNG, input/analysis-manifest.json, rooted-receipt.json, 재현명령을 둔다. 완료가 아니면 report heading/status와표에서 PARTIAL/FAILED/NOT_RUN을 정확히 드러낸다. Missing 결과 때문에 valid 결과를 숨기지 않는다.

허용 write paths:
- project/run_scripts/local_z_adaptive_allocation/analysis/ — 새 CPU reducer/analysis/plot/tests 전용. 기존 runtime module 수정0.
- /data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/review-20260917-v1/ — 새 local review receipts/reducers/temp output, 원attempts read-only.
- audits/servers/server4/2026-09-17-local-z-adaptive-allocation-review/
- experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/
- experiment-reports/servers/server4/README.md — 새 정본 링크만; CAKE/cap 독립 보고 링크 보존.
- messages/acks/server4/2026-09-17-local-z-detailed-review.md
- messages/server-heads/server4/2026-09-17-local-z-detailed-review.md
- runs/odeedit_local_z_adaptive_allocation_s4_20260916_v1/review-20260917-v1/
- tasks/status/odeedit_local_z_adaptive_allocation_s4_20260916_v1/server4.json — 기존 source/job/pause provenance 보존한 현재 review 상태.

Slurm submission NOT_ALLOWED; scheduler read-only exact targets once, GPUcap1 유지/리뷰GPU0. 모델/evaluator/replay/새teacher/추가arm/스윕/10k/Late/Report256/GPU gate/재제출/취소/rsync/삭제0. NO_BROADCAST_NOT_REQUIRED: 서버 내 기존 자료의 CPU 분석이고 raw 대형복제 필요없음.
PNG는 직접 코드 실행으로 생성하고 가능하면 byte 재현한다. GFM table의 내부 pipe·열 수·백틱·상대링크/이미지 및 한국어 렌더를 점검한다. 그래프·CSV·본문 수치/분모를 대조한다.
Postrun source/hash/metric/state/비용/권한/누락 검사를 마치고 본인 scope source+compact report만 최신 타SH/GH 변경을 보존해 non-force main에 게시한다. Raw/tensor/prompt/fullstdout은 Git에 넣지 않는다. 실패나 의미있는 비교불가를 감추고 PASS하지 않는다. 별도 remote raw/GPU audit를 GH에 중복 요청하지 않는다.
SH는 사실·수치·제약과 질문별 evidence를 정리한다. 인과해석·우월성·후속 정책 선택/실험은 GH 별도 global review 소유다. 기존 cap/CAKE 분리 보고에 이번 결과를 덮어쓰거나 합치지 않는다.
최초 FULL_READ/M0→한정 상태/첫 실제 표→상세 보고/main 인계를 진행하고, 완료/부분완료 경계와 남은 한계를 compact하게 전달한 뒤 TASK_COMPLETE_STOP. 자동 monitoring/후속submit/callback0.
