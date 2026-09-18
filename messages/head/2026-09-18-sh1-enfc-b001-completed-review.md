# GH → SH1: ENFC B1 완료 실험 상세 리뷰

instruction_id: ODEEDIT-S06-ENFC-B001-COMPLETED-DETAILED-REVIEW-SH1-V1
nonce: ODEEDIT-GH-SH1-ENFC-B001-COMPLETED-REVIEW-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-f93a-7b50-bca0-65438eab2062
server: server1 / devbox
expected_cwd: /mnt/raid5/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 사용자 recall과 종료 지점

> odeedit_enfc_m_b001_s1 실험 끝난거 자세히 리뷰시켜

이번 호출은 job **50098 / odeedit_enfc_m_b001_s1**의 완료 확인·상세 CPU 리뷰·보고서/main 게시까지 명시적으로 재개한다. 초기 gate에서 다시 멈추지 말고 이 리뷰 범위의 최종 전달까지 수행하라. 과거 pending 및 initial NOT_YET_RUN 기록은 당시 provenance로 그대로 보존한다. GH는 중복 raw/GPU 검산하지 않는다.

정확한 50098만 한 번 scheduler accounting으로 owner/state/exit/time/allocation을 확인한다. scheduler COMPLETED와 scientific 결과 완결성을 구분한다. 실제 실패·누락이면 그 경계와 원인을 보고하고, 성공/완료를 추정하지 않는다. **이번 recall은 리뷰 전용으로 신규 GPU/model/forward/evaluator/Slurm 제출·수정·재실행을 허용하지 않는다.** GPU가 필요한 누락은 NOT_MEASURED로 목록화해 GH에 반환한다. 과거 자동 기술 재제출 승인을 이번 리뷰의 새 실행 권한으로 사용하지 않는다.

## 읽기·입력 경계

- messages/head/2026-09-18-sh1-enfc-single-batch-resume.md 및 이 지시문 전체, 원 ENFC design/contract/cells/positioning/BLUE audit와 적용된 T-skip/reuse 지침을 다시 결속한다. 기존 FULL_READ의 exact bytes는 해시로 재사용 가능하다.
- 실행 source **3f1941b21538d6a7ad0afd756774ad6a0b605750**, tree **72503847266864bf5e1b0c96d63ba99aa03d846d**; lock SHA **b463293b23131b1453d1d3521a58f8f57e1804b0ca18d0fd8fcd0bdc64999147**.
- 실행 WT `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-enfc-single-batch-m-v1`.
- 해당 WT의 `local/enfc-single-batch-m/20260918-v1/{submission-r1,output-r1/b001/attempt-v1,imports/server4-b001-r1,locks}` 및 이 task의 exact logs만 read-only 접근한다.
- 기존 source 87f65ea2 및 수신269 member/16,733,125,876B seal, S4 완료 endpoint·selection ledger 원본을 local imports에서 재사용한다. S4 live S50071, 다른 task의 scheduler/result/raw/모니터링은 접근·재개하지 않는다. 신규 rsync/원격 raw 수집0.
- 새 분석은 clean 전용 branch/worktree에서 수행한다. 원 실행 source·raw·partial·선택 ledger·endpoint·teacher·과거 보고는 덮어쓰지 않는다.

## 먼저 받을 실제 결과표

N4 / SCALE / CA / KL-P / EN-S / EN-F / EN-COV / EN-F4 8개에 대해 실제 B1 final endpoint의 Current RS/PS/NS, strict/joint, S64 loss, Dev128, native 대비 변화, selected norm/fallback/stop reason을 한 표로 먼저 전달한다. 기본 R/P/N 분모100/200/1000, unique100. 값이 없으면 NA와 이유를 적고 추정·0 채움 금지. 첫 표는 부분검산 수준임을 표시하며 전체 integrity 완료 주장과 분리한다. RAND±는 별도 관측행, CA-EXACT는 geometry행이다.

## 필수 상세 리뷰

1. **완료·재사용 accounting:** 기존5개 N4/SCALE/CA/KL-P/EN-S 최적화 REUSE, 신규 EN-F/EN-COV/EN-F4 RUN_MISSING, observer EVAL_ONLY를 실제 완료자료와 대조한다. 8개 arm이 모두 같은 immutable WN/native z/Δ/A/K/P/zeroM/context/request order 및 teacher에 결속되는지 확인한다. SCALE/CA가 N4와 byte-identical이면 의미상 비교행은 유지하되 평가 계산/독립 endpoint/비용은 중복 집계하지 않는다. 종료·실패·미실행 상태를 arm별 공개한다.
2. **독립 metric 집계:** 저장 true/new NLL에서 원 RS/PS new<true, NS true<new, tie=failure 규약으로 재집계한다. exact case/prompt/target identity로 paired lost/gained, NLL/margin 평균·tail·p95/p99, W0-correct N retention, R+twoP joint, R/P TF-strict를 검산한다. 원래 없는 지표를 aggregate에서 복원하지 않는다. greedy32의 token-prefix/길이/EOS/censored를 TF-strict와 구분한다. 동일군 총점이 같아도 ID별 손실·회복은 공개한다.
3. **설계대로 동작했는가:** 설계→frozen 파일/함수/line→저장 evidence 표를 작성한다. L4 only, common native endpoint, all-valid-token old/new lock, P_star/rank/ambiguity/remaining dimension, actual FP32 materialization, W0 teacher, S64 KL 또는 EN-COV 누적 activation 목적, current finite NLL/ID guard, invariant residual/logit bounds, 실제 Armijo/accepted/rejected/dedup/fallback, EN-F4 4×6 fresh gradient와 ideal/actual D 구분, cold M history0/빈 Past, selection 후 P/N·Dev observer 및 reset을 확인한다. source가 그러하다는 근거와 실제 저장 runtime 증거를 구분한다. T 생략 때문에 미확인인 항목은 그대로 남긴다.
4. **교정 동작 상세:** arm별 native norm/correction norm/cumulative W−W0, χ와 projected-gradient 비율, 시도 step과 실제 displacement/예측·실제 감소, guard별 거절 건수, 마지막 수용 또는 N4 fallback을 기록한다. EN-F 대 CA/KL-P/EN-S/EN-COV/EN-F4의 수치 차이, CA exact-null 공간, RAND± norm match/두 부호를 모두 보고한다. 허용 correction과 실제 locality 개선은 별개 열로 둔다. 효과가 없거나 성능이 나쁜 결과도 포함한다.
5. **플랫폼·자료 범위:** S4 Blackwell의 재사용 endpoint와 S1 A6000의 신규 observation/correction을 구분한다. 이번 재계산 geometry/gradient의 source-state/captured-key identity와 실제 실행 이유를 기록한다. 교차 hardware byte parity는 측정하지 않았다면 NOT_TESTED. 동일 수식으로 계산했다는 사실을 수치적 동등성 증명으로 쓰지 않는다. native/teacher 신규 생성0 계획이 실제 counters와 맞는지도 확인한다.
6. **보존·복원·완결성:** final8 L4 checkpoint의 현재 존재/size/SHA, 안전한 CPU weights_only shape/dtype/finite/metadata 및 native/reference 결속을 확인한다. 원 final state 및 observer/reset guard를 audit하되 source/receipt만으로 fullmodel byte 복원/GPU continuation을 검증했다고 하지 않는다. random artifact/원 factors·trial ledger·선택 seal도 필요한 범위에 연결한다. 누락·미보존 상태는 명시한다.
7. **실측 비용:** 50098의 parent allocation과 program wall, load/restore, geometry, shared/new gradient, accepted/rejected trials, invariant/guard, S64/Dev/official/generation observer, storage/I/O 및 peak를 가능한 저장 counter로 분리한다. 미분리 항목은 NOT_SEPARATED. 기존 S4 M50050 두 cell9768초, 앞 T/M3231초·metadata-failure2301초는 기존 lineage로만 기재하고 새 비용에 재합산하지 않는다. B1/B2 구분 불가인 과거 비용을 임의로 반분하지 않는다. 공유 비용과 arm별 비용을 구분한다.

## 보고·해석 경계

T=SKIPPED_USER_DIRECTED / full_numerical_validation=NOT_ESTABLISHED를 유지한다. CPU·checkpoint 검사로 T_PASS/FD/GPU 기술 전체 PASS를 새로 만들지 않는다. 한 cold batch100요청은 M10개의 독립 batch나 sequential/장기 검증이 아니다. arm별800 관측은 unique800이 아니며, M10 batch bootstrap 또는 one-batch 통계적 동등성 주장을 하지 않는다. P/N observer를 선택 기준으로 역사용하지 않는다. Report256은 미개방 유지한다.

SH는 실행 사실·수치·설계상 기계적 판정과 source-backed 기술 오류만 보고한다. 정책 선택·과학적 우열/원인 단정·장기효과/가설 지지·후속 실험 결정은 GH가 별도로 한다. 새 성능 gate·추가 arm·T 재실행·S/R/L 확대0.

## 권한·산출물·게시

server1 GPU cap2는 기존 정책으로 유지하되 이번 리뷰 신규 GPU 허용0. 모델/teacher 전량 재해시를 반복할 필요는 없고 기존 exact seal 재사용 범위를 명시한다. raw Git0; broadcast NO_BROADCAST_NOT_REQUIRED. subagent는 복잡한 독립 하위 작업에만, 단순 검사·정리는 직접 수행한다. 실제 독립 red 사용 여부를 사실대로 적는다.

허용 write(전용 분석 branch/WT):
- `project/run_scripts/single_layer_edit_preserving_correction/server1_single_batch/analysis/` 및 해당 CPU tests. frozen runtime 변경0.
- `local/enfc-single-batch-m/20260918-v1/completed-review-r1/`.
- `experiment-reports/servers/server1/enfc-single-batch-m-2026-09-18-v1/completed-review-r1/`: `diagnostic-report-ko.md`, first/final table CSV, paired/request-tail·mechanism/reuse/compute tables, artifact/analysis manifest 및 rooted receipt.
- `audits/servers/server1/2026-09-18-enfc-single-batch-m/completed-review-r1/`.
- `messages/acks/server1/2026-09-18-enfc-b001-completed-review.md`, `messages/server-heads/server1/2026-09-18-enfc-b001-completed-review.md`.
- `tasks/status/ODEEDIT-S06-ENFC-B001-COMPLETED-DETAILED-REVIEW-SH1-V1/server1.json`, `runs/odeedit_enfc_m_b001_s1_20260918/completed-review-r1/`.

한국어 보고서, 비교표, 필요한 code-only 그림, 재현명령과 source/input/output hashes를 남긴다. GFM 표 열·상대링크·한국어·manifest·원 bytes 보존·raw-free를 검사하고 사용 가능한 renderer로 실제 렌더를 확인한다(미실행은 구분). 새 CPU reducer/test 결과를 실제 모델 검증으로 확대하지 않는다.

own-scope 분석 source+compact 보고를 clean integration에서 최신 main 변경을 보존해 **nonforce main push**까지 수행하도록 승인한다. 충돌은 타인 변경을 되돌리지 말고 보고한다. 완료 후 report 절대/상대경로·SHA/source/main·첫8arm표·누락/한계·비용을 GH에 전달하고 TASK_COMPLETE_STOP. 단계마다 재승인을 기다리지 말되 새 실행이 필요하면 리뷰 범위 밖이므로 누락을 보고한다.
