# GH → SH4: 완료된 sequential EN-F 한 arm 상세 리뷰

instruction_id: ODEEDIT-S06-ENFC-S-ENF-COMPLETED-DETAILED-REVIEW-SH4-V1
nonce: ODEEDIT-GH-SH4-ENFC-S-ENF-REVIEW-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 사용자 recall / exact 대상

> odeedit_enfc_S4_s4 도 en_f 끝난거 자세히 리뷰시켜.

이번 승인 대상은 기존 array **50071_1 = EN-F**의 O0 first1000 / cold W0·zeroM4 → B100×10 완료 결과다. EN-F4(50071_3)와 혼동하지 않는다. sibling EN-S/EN-COV/EN-F4 및 타 task의 scheduler/raw/result/모니터링은 이번에 재개하지 않는다. array 전체 조회 대신 정확 child50071_1을 한 번 accounting으로 owner/name/state/exit/allocation에 대조하고, 실제 mapping이 다르면 이름만으로 대상을 넓히지 말고 보고한다.

초기 pause를 이 리뷰 범위에 한해 해제한다. 실제 terminal 확인→첫 W10 표→CPU 독립 검산·설계 동작 검토→상세 한국어 사실 보고/main 게시까지 재승인 없이 진행한 뒤 STOP한다. 이전 initial은 EN-S 대표 경계만 관측했다는 기록을 유지하고 EN-F initial을 당시 확인했다고 소급하지 않는다.

**신규 GPU/model/forward/evaluator/Slurmwrite·submit·rerun·추가 arm/순서·T/FD/ULP·R/L·자동 후속0.** 실패 또는 누락이 있으면 경계와 미실행 항목을 보고하고 새 실행으로 조용히 보완하지 않는다. GH는 중복 raw/GPU 감사를 하지 않는다.

## 정본·read scope

원 design/contract/cells/positioning/BLUE audit, M reuse-first/T skip 지시, latest 사용자 M중단→4-policy S override 및 S submission/initial 문서 전체와 실제 실행을 결속한다. 기존 exact FULL_READ는 byte identity 확인 후 재사용할 수 있다.

- 실행 **9e5884f5a2f8dcb7fc7406084f3fde94df450a55**, tree **aaacc6a0c55e19038a81516f40a9b4938c3fb38a**.
- archive SHA **d2d31418632d98f9056db20666a663e79dde4bcb3c6ddbdc90cf0a2b771496bf**.
- lock SHA **4d600d4b57df58203fb21f447116c0362b8a731b9b6d71c4485353d913224d56**.
- S root `/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/S/attempt-v1/`: lockから index1 EN-F의 정확 output/log 경로를 먼저 resolve한다. 해당 arm과 공통 immutable source/input/teacher/reference/native B1 seals만 읽는다. sibling scientific raw는 열지 않는다.
- 기존 source/input/reuse/initial receipts 및 `experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/sequential-four-r1/{submission,initial}/`.
- SH1 B1 완료 보고는 main `2d86d020`의 `experiment-reports/servers/server1/enfc-single-batch-m-2026-09-18-v1/completed-review-r1/`를 참고할 수 있으나 S4 S chain과 혼합 집계하지 않는다. 그 M은 cross-hardware single cold100 diagnostic이다.

source·raw·weight·teacher·history·기존 seal/보고는 read-only 보존하며 새 clean 분석 branch/WT를 사용한다. local에 있는 과거 완료 baseline raw 또는 게시 compact report만 조건 대조 후 재사용하고, 새 원격 raw 수집·rsync0.

## 선행 첫 표

정확 EN-F W10 fullseen1000의 R/P/N 실제 분자/분모(1000/2000/10000), strict/joint, W0-correct N retention, S64/Dev128, nonzero/fallback batch 수 및 actual allocation을 먼저 전달한다. W5 first500과 W10 same-first500, 각 batch Current는 별도 표다. 미완료/미측정은 NA·사유를 적고 예비 표의 검산 수준을 명시한다. 최종 성공률이 좋거나 나쁘다고 리뷰를 선별하지 않는다.

## 상세 검산과 필수 장

1. **실제 completeness/provenance**: terminal1000/10batch/unique1000, request order·case/prompt/target identity, 10 commit/history append, adjacent selected W/M/context/RNG/ledger→next entry links, 현재 checkpoint/native preview 존재와 size/SHA/CPU weights_only shape·finite를 확인한다. 전체 CP가 실제 없으면 보존 범위를 명시한다. Runtime hash/receipt와 독립 tensor reconstruction/GPU continuation을 구분한다. native B1 재사용과 B2+ 새 fit의 actual count·source를 결속한다.
2. **raw 독립 reducer**: RS/PS new<true, NS true<new, tie=failure의 canonical 정의로 count/분모를 재집계한다. Current entry→own native→selected, W5/W10 fullseen, W5→W10 같은 first500을 정확 ID로 pairing한다. 새500/old500, at-write→final, received/ACTIVE/SUPERSEDED, R+twoP joint·TF-strict·greedy32·censored, N true/new NLL·margin·p95/p99, success lost/gained 및 W0-correct neighborhood retention을 저장 범위에서 보고한다. 총점 동일과 동일 성공집합을 혼동하지 않는다. 없는 per-case를 aggregate로 복원하지 않는다.
3. **실제 method 동작**: 설계→실행 파일/함수/line→실물 evidence 대응표를 작성한다. L4 only/all-valid-token old/new K_E/허용 range P_star, q/rank/ambiguous band, GP/GQ/chi, fixed W0 teacher S64 KL, D_acc64와 실제 FP32 D, normalized DK residual/projection leakage/logit/NLL·ID invariant, 개별 current finite guard, actual Armijo, trial/dedup/accepted/rejected/stop reason/fallback을 batch별 대조한다. 평균 plateau나 다른 cap/threshold가 섞이지 않았는지 확인한다. source 주장과 실제 저장 근거를 구분한다.
4. **sequential 핵심**: B2+는 자기 selected entry에서 새 native z/update를 계산했는지, 다른 arm/M의 이후 batch 결과 공유0인지 확인한다. Past64는 이미 받은 latest-active/current-overwrite 제외·고정 SHA·성공여부 비선별인지, guard 기준이 own WN인지 검사한다. history는 실제 selected 또는 native fallback에서 정확히1회, 후보 append0인지 대조한다. native가 만든 망각과 correction의 추가 손상을 entry→native→selected로 분리한다. observer는 selection seal 뒤 실행하며 P/N·Dev가 policy로 돌아가지 않아야 한다.
5. **worked examples와 분포**: 저장된 nonzero accepted 사례와 fallback/reject 또는 최소/최대 correction 사례를 선택 기준을 명시해 설명한다(실제 없으면 없음). nominal/actual step·norm/native비·cumulative W−W0·ideal/actual 감소·guard 원인·반응 보존 수치를 표로 둔다. correction 공간 존재, preservation 감소 방향, protected response, 독립 locality, PS/과거 편집, 비용을 별도 사실 축으로 기록한다. S64 KL 감소를 locality 개선으로 합치지 않는다.
6. **비교 범위**: own-native preview는 EN-F 자신의 trajectory에 대한 same-state shadow이지 독립 N4 sequential chain이 아니다. 사용 가능한 과거 N4/BLUE-L4 first1000 baseline은 model/revision/context/token/order/hparams/precision/evaluator/start/history 조건을 비교해 direct-comparable 또는 REFERENCE_ONLY로 구분한다. 비교 가능 raw가 local에 없으면 공개 집계와 한계만 적는다. S4 다른 현재 sibling의 결과를 임의로 추가하지 않는다. SH1 M EN-F의100요청 결과를 S1000의 대조분모로 사용하지 않는다.
7. **actual 비용**: 정확50071_1 parent allocatedGPU-sec/program wall/peak를 기록하고 extern/batch 중복 합산0. native fit/z/Adam/loss/solve, geometry/cache/teacher I/O, neural gradient, accepted/rejected trials, guard/invariant/Past, official/Dev/generation/fullseen evaluation, checkpoint/I/O로 저장된 counter·timer를 분해한다. nested timers는 합산하지 않고 NOT_SEPARATED/NOT_RECORDED를 노출한다. 계획10batch 상한을 actual로 복사하지 않는다. shared B1 native/teacher·과거 M/T/metadata failure 비용은 기존 lineage 별도이며 이번 allocation에 중복 청구하지 않는다.

## 한계·판정 규칙

T=SKIPPED_USER_DIRECTED / full_numerical_validation=NOT_ESTABLISHED, M→S=USER_DIRECTED_NOT_ESTABLISHED 유지. CPU 검산·후보 invariant만으로 FD/direct-gradient/전체 수치 검증 PASS를 만들지 않는다. independent GPU continuation/bitwise model parity는 미실행이면 NOT_TESTED. 하나의 fixed-order chain은 독립10batch 반복이나 unseen-data/장기10k 증명이 아니다. case-level uncertainty는 이 trajectory 기술통계로 한정하고 통계적 동등성/인과 우열/과학적 promotion을 주장하지 않는다. Report256 미개방 유지. SH는 사실·수치·계약 판정만 보고하며 GH가 별도 종합한다.

## 운영·허용 write·게시

cap2 정책은 유지하지만 이번 리뷰 GPU 사용0. shared dirty/기존 WT 상태를 보존하고 branch `codex/server4-enfc-s-enf-completed-review-v1` 같은 별도 clean WT에서 분석한다. session/CWD/origin/common Git identity 확인을 생략하지 않는다. Subagent는 복잡한 독립 작업에만 필요시 사용하고 실제 별도 red 여부를 명시한다.

허용 write:
- `project/run_scripts/single_layer_edit_preserving_correction/analysis/` 안의 EN-F sequential CPU reducer/review 및 그 tests(실행 numeric/runtime는 수정0).
- `local/single-layer-edit-preserving-correction/20260918-v1/S/completed-en-f-review-r1/`.
- `experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/sequential-four-r1/en-f-completed-review-r1/`의 `diagnostic-report-ko.md`, first/final table, batch/trial/paired/retention/reuse/compute CSV, manifest/rooted receipt, 재현 가능한 필요한 code-only PNG.
- `audits/servers/server4/2026-09-18-enfc-sequential-enf-review/`.
- `messages/acks/server4/2026-09-18-enfc-sequential-enf-review.md`, `messages/server-heads/server4/2026-09-18-enfc-sequential-enf-review.md`.
- `tasks/status/ODEEDIT-S06-ENFC-S-ENF-COMPLETED-DETAILED-REVIEW-SH4-V1/server4.json`, `runs/odeedit_enfc_S4_s4_20260918/en-f-completed-review-r1/`.

자료hash/독립집계·선택계산/CPU tests·GFM열·링크·실제 renderer 가용여부·그림 코드재현·raw-free 검산 후 ownscope source+보고를 clean integration에서 최신main 보존/nonforce push까지 승인한다. 원 source/실패·초기 기록 불변, 큰 raw/tensor/prompt/log Git0, NO_BROADCAST_NOT_REQUIRED. helper 거부/미실행 검사는 숨기거나 PASS로 표시하지 않는다.

첫 M0에서 정확 target/read scope와 CPU-only를 확인하고 첫 표를 선행 전달한다. 완료 시 report 절대/상대경로·SHA/실행 및 분석source/main·핵심 수치/미검증·비용을 반환하고 TASK_COMPLETE_STOP한다. 새 실행이나 다른 arm으로 자동 확대하지 않는다.
