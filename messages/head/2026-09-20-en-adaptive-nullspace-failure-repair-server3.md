# GH → SH4 RCA / SH3 repair-submit: failed EN adaptive-nullspace B300

Instruction ID: ODEEDIT-S06-EN-ADAPT-B300-REPAIR-SH3-V1
Nonce SH3: ODEEDIT-GH-SH3-EN-ADAPT-REPAIR-RUN-20260920-R1
Nonce SH4: ODEEDIT-GH-SH4-EN-ADAPT-FAILURE-TO-S3-20260920-R1

## 最新 authority

사용자: “odeedit_en_adapt_B300_s4 이거 fail되었는데 확인해보고 server3에서 repair run 올리라고 해”.
이는 S3 실행중지/SH4실행 이관을 **이 task에 한해 역전**한다.
SH4는 지정 실패job의 read-only/CPU RCA와 인계만, **S4 재제출0**.
SH3는 RCA를 받아 최소 technical repair·새sourcefreeze·server3 repair run 제출의 sole owner다.
SH3 session01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3/ubuntu,
/data/janghj/ODE-edit/hyunjun1127/ODE-edit. S3 project/task cap1,
각1GPU/8CPU/host≤121856MiB/exportNONE/Requeue0, 기존 다른job변경0.
SH4 session01a04939-b5c7-7a03-ba2d-ef3343d62cfd/server4, same logical CWD/repo.
신규 실험/방법 지시가 아니라 승인된 동일 B300 실험의 실패수리·실행서버 변경이다.

## 1. 먼저 정확한 실패를 분리

Known job51260/nameodeedit_en_adapt_B300_s4, execution
b6e86234640ca127546aee094f2a67bbe684a490/tree5eacfc956214ffbd0cfccbeb56c068a888b88990.
Archive eee7e31a072fcf54d4390f5d6679ef97b75f848208b3e9996b452f1ec81e9867,
lock cfc4e2150cf57f71d0fea838d2302d84626402b8ad9e8941690da3220bd671ca.
S4 root /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server4-migration-r1/attempt-v1/.

SH4가 정확 owner/source/args/accounting/첫stderr/실패receipt를 한정 확인하고
OOM/timeout/IO/프로그래밍/수치/정상scientific fallback을 구분한다.
First exception과 cleanup secondary failure, 마지막 완료batch/arm/commit/history/teacher,
구체적 수정라인·회귀fixture·재사용단위를 보고한다. 추측으로 원인을 확정하지 않는다.
실패source/raw/partial/비용 불변, 과거 intermediate RUNNING 기록을 덮지 않는다.
실제로 여전히 실행중이라면 이를 먼저 알리고 중복run하지 않는다.
한정RCA와 작은 source/manifest handoff 이후 SH4는 STOP, GPU진단/새평가/재제출0.
SH3는 GH의 중복 raw감사나 단계별 재승인을 기다릴 필요가 없다.

## 2. SH3가 받을 source와 변경 경계

원SH3구현43904c13/runner2ff2393f를 무조건 되돌려 사용하지 않는다.
S4 실제 b6e86234640ca127546aee094f2a67bbe684a490의 source/config/input locks를 기준으로
S4 port 중 수리한 ideal-ray curvature 및 actual-Armijo 의미를 포함하여 정확히 보존한다.
S4 최신 branch codex/server4-en-adaptive-nullspace-b300-v1의 최근분석1bb93e1d는
실제execution과 다르다. 이후분석/metric/report 보완을 따로 재사용하고
원runtime가 수정되어 실행됐다는 식으로 provenance를 합치지 않는다.
SH4 RCA 최종commit/diff가 나오면 readonly확보 후 SH3 repair에 결속한다.

S3 readiness FP32/eager/TF32off/model/native/evaluator/context/P/C0 사용.
실제실행에 소비되는 source/config/import/teacher/metrics/runtime를 새로 봉인한다.
S4 absolute paths/server4/59GiB↔S3 ubuntu/runtime/119GiB 차이를 조정하되
공유env/EasyEdit/source/teacher 원본은 변경하지 않는다.
버그 최소수리와 scoped CPU회귀를 수행한다. 성능결과를 이유로 controller/threshold/
rank/epsilon/arm/budget/reference분모/평가를 바꾸는 수리는 승인하지 않았다.
실제 technical bug가 추가로 드러나면 선보고·원자료보존 후 승인범위 최소수리 가능.

## 3. 정확한 restart와 재사용

기본 noCP 때문에 selected W/M/optimizer/동등delta disk 저장이 없었다.
과거 checkpoint/중간RAM/hash/metric만으로 exact resume 가능하다고 주장하지 않는다.
재구성 가능한 완전상태가 확인되지 않으면 **fresh W0/zeroM4 B1부터 동일 B300을 실행**한다.
이는 crash 상태 부재로 필요한 restart이며, 완료 raw를 같은 trajectory로 이어붙이지 않는다.
R512/Dev128 generated256 teacher/inputs/Pstar/source/CPU증거는 identity가 맞으면 재사용.
S3에2560files101519959223B fullSHA/size/CPUshape 검산완료 자산이 이미 있다.
대용량 reference/model 재전송·재생성0; S4 실패자료와 작은 config/source/receipt만
필요시에 exactpull한다. Native/geometry/G는 현재 own-entry와 hardware·hook identity를
충족할 때만 계산재사용 가능하며, saved weight 없는 scalar receipts는 대체물이 아니다.

기존 T0의 cached/physical gradient 동일·FD 등 제한된 측정은 원서버 증거로 보존한다.
S4 precision NOT_ESTABLISHED를 S3 PASS로 바꾸지 않는다.
수정영향 없는 과거validation 전체를 관성적으로 반복하거나
이 task에 없던 엄격한 archive-NLL/M1 gates를 추가하지 않는다.
새경로에 필요한 bounded T0와 정본실험을 구분, 설계의 exploratory 규약을 유지한다.
정상 no-accepted/fallback/품질저하는 고장으로 rerun하지 않는다.

## 4. 불변 과학 범위와 추가 지표

정본9개와 original SH3 envelope의 수학·샘플·관측권한을 상속한다.
Same fixed10k first300 순서/B100, W0/zeroM4, L4단일/원nativeL2/hparams,
B1 N4/EN_EXACT/EN_NUM/EN_ADAPT 공유계산,
B2/B3 N4/EN_EXACT/EN_ADAPT own-trajectory, primaryepsilon.05,
공통2candidate/전체R512/activehistory/선택후officialobserver 유지.
S2공통 z-hook prefix/batching/필요위치full-vocab head와 FP32 조건,
문서별dense gradientD2H0 유지. 새policy/B1000/10k0.
RSPSNS와 TF rewrite/rephrase/neighborhood token-micro/prompt-macro/strict,
true/new/desired NLL, pairedlost/gained/current-activepast-allseen 분리 및
requestbootstrap10000seed20260920 유지. Metrics 결손은 가능한 기존raw후처리로 보완한다.
Checkpoint 미저장; no edited W/M/delta/resume disk, 기존자료삭제0.

## 5. 제출·중복방지·모니터링

S4 실패terminal/실행중복없음 및 SH3 currentownadmission을 확인 후 cap1내 단일repair등록.
적절한 실제메모리/disk/임시peak/후속출력여유·시간예상을 새로 계산한다.
S3 최근전송후 free약17.6GB는 과거비독점관측, 입력중복복제·불필요nativecache
재생성 없이 output/scratch까지 예산을 확보한다. 부족시 임의삭제 없이 명시blocker.
No storagewaiver/119GiB이하 무근거충분성주장0.
Held exact owner/source/fullargs/node/GPU/CPU/memory/dependency검사→release.
한정RCA·수리·정상제출까지 진행하며 준비/CPU PASS만으로 run올렸다고 보고하지 않는다.

최신 사용자는 기존 S4모니터링을 중지했고 이번 요청은 실패확인·repair run 제출이다.
최신 사용자 추가 지시 “모니터링은 하지 말라고 해”가 기존 초기경계 관찰 조건을 대체한다.
**수리·제출전 admission·held검사→release 후 monitoring0**.
초기gate/첫batch/실패지점 통과확인, scheduler/log/result polling,
pending사유 추가조회·resource대기·sleep/heartbeat/callback/완료대기를 하지 않는다.
제출 자체의 job/source/resource/release receipt만 보고하고
actualinitial/terminal=NOT_OBSERVED, MONITORING_PAUSED_AWAITING_USER로 종료한다.
Sealed 프로그램은 B300 자연진행; 이 지시로 job cancel/hold하지 않는다.
사용자가 별도 completion review를 호출하면 상세terminal분석/main을 진행한다.
지금scope는 RCA/repair/submission handoff이며 초기성공/B300완료를 미리 주장하지 않는다.

## 6. 선택 인계 권한 / write paths

SH3 sole destination pull/writer. S4 위 attempt-v1와 동일task의 RCA하위경로에서
frozen source archive/config/lock/error/timing/완료단계 manifest 및
새로생긴 재사용필수 작은 관측자료만 file별source→dest/size/SHA목록으로 exact수신.
S4 Git source branch는 Git수신 우선. 원본KEEP/--delete·이동·덮어쓰기0.
Live scientificraw/무관task자료 접근0, 모델/reference101.5GB 재전송0.
필요큰자료가 새로 발견되면 실제재사용필요와용량을 명시, 원래목적외 수집0.

SH3 신규 root /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server3-repair-r1/.
Clean codex/server3-en-adaptive-nullspace-repair-r1 worktree.
허용source project/run_scripts/en_adaptive_nullspace/** 및scopedtests/launcher,
audits/servers/server3/2026-09-20-en-adaptive-nullspace-repair/**,
experiment-reports/servers/server3/en-adaptive-nullspace-2026-09-20-v1/repair-r1/**,
transfers/verifications/2026-09-20-en-adaptive-nullspace-repair/**,
tasks/status/server3-en-adaptive-nullspace-repair-20260920-v1/**,
runs/odeedit_en_adaptive_nullspace_s3_repair_20260920/**,
messages/acks/server3/2026-09-20-en-adaptive-nullspace-repair.md,
messages/server-heads/server3/2026-09-20-en-adaptive-nullspace-repair*.md.
SH4는 기존task audit/report/status 아래 failure-handoff-r1을 새로만들어 기록가능.
Compact 원인/수리/비용/reuse/restart/source/job/미검증 handoff와 source를
ownscope nonforce branch/main게시까지 허용. 타SH변경보존, rawGit0.
NO_BROADCAST_NOT_REQUIRED. GH과학해석/후속선정은 별도다.
