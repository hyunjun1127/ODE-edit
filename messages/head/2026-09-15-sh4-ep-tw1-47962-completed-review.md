# GH → SH4: EP-TW-1 job47962 완료 실험 상세 리뷰

Instruction ID: ODEEDIT-S06-EP-TW1-47962-COMPLETED-DETAILED-REVIEW-SH4-V1
Nonce: ODEEDIT-GH-SH4-EP-TW1-47962-DETAILED-REVIEW-20260915-R1
Parent: ODEEDIT-S06-EP-TW1-DIAGNOSTIC-GATES-SKIP-RUN-SH4-V1.
발신 GH session 01a04939-8873-7673-8dca-4c7fc5e31af0.
수신 server4 / session 01a04939-b5c7-7a03-ba2d-ef3343d62cfd / CWD /data/janghj/ODE-edit / repository hyunjun1127/ODE-edit.
분석 기준 origin/main 833e6396522924b8bbb26eedadf763e2d3ebd4d4 / tree 1284e7e68ffdeceee923a028edca14d9020ceb2b.

## 1. 최신 recall과 작업 경계

사용자: “server4의 실험 종료되었으니 자세히 리뷰시키자.”
대상은 단일47962 / odeedit_ep_tw1_nogate_s4이다. 이 task에 한해 사용자 호출 대기 상태를 해제하고 완료 실험의 CPU 검산·상세 사실 보고·본 scope main 통합까지 진행하라. GH 중복 remote raw/GPU 감사나 단계별 재승인은 요구하지 않는다. 실험 완료는 사용자의 알림이며 실제 scheduler 완료와 artifact 완전성을 구분해서 기록한다.

본 지시/진단 skip envelope와 waiver/원 EP-TW-1 v3 method·reference 계약/자신의 submission report와 resume·execution lock을 결속하고 전체 읽어라. 이미 읽은 동일 bytes는 exact identity로 재사용 가능하다. 기존 shared dirty/실행 source/raw/실패/다른 worktree를 보존한 새 clean codex/server4-ep-tw1-47962-completed-review-v1에서 작업한다. 실제 최신 main은 fetch 후 기록하되 runtime source를 main으로 혼동하지 않는다.

새 model load·forward/backward·GPU evaluator·native edit·teacher 생성·FD/ULP/gradient 재검증·smoke·replay·Slurm submit/requeue/cancel/자원변경은 허용하지 않는다. 생략한 수치 gate를 사후 GPU 재실행해서 메우지 않는다. 저장된 tensor의 CPU weights_only reload와 산술 검산/분석은 허용한다. 다른 paused task/BG/N4 calibration/이전 repair 작업의 재개는 금지한다.

## 2. 정확한 입력과 종료 확인

- execution 6d317bdb2660d7e9919bc3a9fb878564e9729e37 / tree 6db14423a207eaccad195ce291c40bb8ce93cc8d
- archive SHA 2dec619e12f72e1cf6fce843c61adc68726ddc025a6b09cff326bb4b6229124a
- execution.lock SHA 5a19c2be919362d08b2ea80e69a406d7b6de569aade8de639036f69db5d5d8f9
- local root /data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1/
- output scientific-v1/; resume-manifest.json SHA d66d4acf77bdc45f36945308f4d691ad2bff894e60d0f439f9f64e3802fbe637
- submission report experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/submission-factual-report-ko.md SHA 7aca507c75e7d69cb39fa0c8567b01a4682c6cd5e8e5dd2fecf4befcbe1398ae.

指定job만 한정 scheduler audit으로 state/exit/signal/elapsed/allocated GPU-sec/peak memory를 확인한다. 완료 후 반복 polling하지 않는다. 종료가 실패/partial이면 완전1000을 가정하지 말고 정확 committed prefix/첫 오류를 먼저 보고하라. 이번 지시로 자동 repair/rerun하지 않는다. 기존 실패47884/47942는 봉인 RCA·비용 재사용만, 재조회·재진단하지 않는다.

## 3. 먼저 전할 실제 첫 표

기존 raw NLL pairs를 원 canonical 부등식으로 독립 집계해 actual W10/full first1000의 RS/PS/NS를 먼저 전달한다. 기대 분모는1000/2000/10000이고 실제 cardinality로 확인한다. RS/PS new<true, NS true<new, ties=failure를 유지하며 TF exact/strict/token은 별도다.

최종값과 별도로 B1–B10 각 current100, entry/raw native/selected 세 상태, W5 full500/W10 first500·full1000, 선택된 후보 빈도와 RAW fallback 빈도의 첫 표를 만든다. 현재 output에 없는 값은 NOT_RECORDED이며 추가 forward나 추정으로 채우지 않는다. 첫 표의 검증 수준을 명시하고 이 표 전달 후 전체 상세 리뷰를 계속한다.

## 4. 상세 산출물 검토 — CPU only

1. Source/lock/archive 및 실제 native/helper/config/model revision/tokenizer/dtype/backend/contexts/P4 physical4→asset0→local0/sample/order를 결속한다. First1000은 fixed10k의 앞부분이며 shuffle0, W0/coldM0, 10×B100인지 확인한다. 기존 heavy model/teacher 전체 재해시를 불필요하게 반복하지 말고 과거 fullSHA와 이번 검토 증거 범위를 구분한다.

2. 이번 실험 raw 산출물 inventory와 size/SHA, 누락/중복/nonfinite/terminal/commit/선택 endpoint 결속을 확인한다. 10 commit, 9 인접 B→B+1 links, 10 selected W4/M4 checkpoint가 실제로 존재하는지 검산한다. 저장 tensor의 shape/dtype/finite/selected layer·M·context/RNG·ledger·next batch/hash를 CPU로 확인하고 기록된 history1/inner append0을 점검한다. 저장되지 않은 중간 state나 full-model/GPU continuation은 미검증으로 둔다. 파일 SHA/CPU reload를 model-level 수치 parity로 승격하지 않는다.

3. Candidate 전체 RAW/C1/C05/C025의 실제 호출/duplicate/비유한/부적격·feasible·selected 상태와 사유를 공개한다. 각 batch raw Ep/strict ID set/D64와 후보 E/D64/strict ID set/실제 correction norm/trust·anchor 관측을 정리한다. 저장값으로 E≤Ep 및 exact strict ID-set 포함 조건, deterministic minD64/RAW tie/fallback 규칙이 적용됐는지 산술 확인한다. 같은 성공 count를 같은 문항 보존으로 대신하지 않는다. 새 품질 기준이나 scalar 종합점수로 후보를 사후 선택하지 않는다.

4. 같은 batch native RAW→selected Current R/P/N의 loss/gain·NLL/desired-margin 분포·tail을 identity로 연결한다. Raw native는 해당 EP trajectory entry에서 생성된 one-step reference이지 독립 native sequential baseline이 아니다. Entry→RAW→selected와 이후 batch의 변화는 구분한다. Controller가 사용한 Current E/strict 및 S64 D와 관측 전용 RS/PS/NS/Dev128을 분리한다. 정확 NLL byte 비교가 warning으로 전환된 항목은 발생 횟수·최대차이·strict 불일치 여부를 공개하며 삭제하거나 소급 PASS하지 않는다.

5. Accepted ledger의 requested/accepted/distinct/active/overwrite 수, 각 batch 확정 endpoint, at-write success→W10 유지/소실/회복과 old accepted 결과를 결속한다. 전체1000 requested population과 accepted-only subset 분모를 혼합하지 않는다. Historical requested target의 합당한 교체와 forgetting을 별도 표기한다. 관측되지 않은 중간 실패/회복 시점을 추정하지 않는다. W5 first500→W10 first500 paired transition 및 age/cohort별 수치를 가능한 기존 rows로 정리한다.

6. S64는 controller/선택용, Dev128 W5/W10은 별도 관측용으로 표시한다. D64 감소와 실제 NS 향상을 동일하게 취급하지 않는다. C4 reference/teacher 재사용을 명시하고 미실행 Report256/Audit/MMLU/FutureN은 그대로 NOT_MEASURED로 둔다. 추가 holdout 평가를 이번 리뷰 권한으로 실행하지 않는다.

7. 기존 baseline 비교는 현재 접근 가능한 완료 publication·로컬 raw만 재사용한다. 같은 W0 first1000/order/actual final endpoint를 확인할 수 있는 W0, AlphaEdit_BLUE_L4_ONLY, AlphaEdit_BLUE(L4+L8), native AlphaEdit 및 MEMIT reference가 있으면 명칭·model/config/source/context/seed/GPU/evaluator 호환성 표와 함께 별도표에 넣는다. 없는 baseline은 NOT_AVAILABLE, 서로 다른 표본/구간·warm B51–B60 full6000/10k W100은 직접 비교하지 않는다. 동일 case/prompt/target identity가 검증된 범위에서만 paired 비교하고 aggregate publication을 per-case 데이터로 복원하지 않는다. 원격 대형 raw/CP 신규 전송과 baseline rerun은 하지 않는다.

8. 실제 compute는 model/load/teacher reuse/native target+solve/map/gE/gD/candidate scoring/finalize/evaluation/checkpoint/I-O로 구분한다. 기록된 target·Adam·solve·forward/backward·history counts와 각 단위를 공개한다. Pure writer 미분리면 NOT_SEPARATED. 신규47962 allocated GPU-sec와 이전47884 473s/47942 69s/teacher98s를 분리하고,286.5445s nativefit은473초 내부 component라 이중가산하지 않는다. Skip 때문에 빨라졌다는 통제 속도 비교는 하지 않는다.

## 5. 생략된 검증과 보고 한계

검증 status는 SKIPPED_USER_DIRECTED / numerical_validation=NOT_ESTABLISHED 그대로 유지한다. 구형 G0_PASS와 혼동하지 말고 실제 관측된 marker 및 terminal 상태를 적는다. 제출 당시 agent가 PENDING까지만 봤다는 provenance는 변경하지 않는다. 이번 사후 CPU 검산은 원 수치 FD/direct/self-KL 검증의 대체가 아니다. 기존47942 E direct PASS도 그 saved episode 범위뿐이다.

SH는 실행 조건·사실·수치·산술 비교·artifact·기술 오류·검증 범위만 보고한다. 과학적 효능·방법 우열·인과 설명·승격·후속 후보 선택은 GH 소유이며 이번에 수행하지 않는다. 값이 나쁘거나 RAW가 자주 선택돼도 결과로 보존한다. 누락이 있으면 coverage 표에 기록하고 기존 가능한 분석은 계속한다.

## 6. 산출물·허용 write·main

공식 report package:
experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1/
최종 diagnostic-report-ko.md, first-final-table.csv, batch/current/whole-prefix/paired/ledger/candidate/cost/coverage/compatibility CSV, 코드 생성 PNG, source/raw inventories, analysis-manifest.json, rooted-receipt.json 및 재현 명령을 포함한다. 표/그림은 필요한 비교를 충분히 보여주되 미측정 자료를 생성하지 않는다.

허용 분석 source는 새
project/run_scripts/bg_tw_reference/ep_tw/review_nogate.py,
plot_nogate_review.py, test_review_nogate.py 및 review_nogate/ 하위 보조 모듈·테스트와 분석 README뿐이다.
기존 runtime/policy/native/evaluator/실행 launcher와 원 raw는 read-only.
추가 기록은 audits/servers/server4/2026-09-15-ep-tw1-47962-completed-review/,
messages/acks/server4/2026-09-15-ep-tw1-47962-completed-review.md,
messages/server-heads/server4/2026-09-15-ep-tw1-47962-completed-review.md,
runs/odeedit_ep_tw1_47962_review_s4_v1/,
tasks/status/odeedit_ep_tw1_47962_review_s4_v1/server4.json.
Local 분석 scratch는 local/ep-tw1-c4/20260915-v1/gate-skip-r1/completed-review-v1/ 새 디렉터리만 사용한다.

새GPU/Slurm submit NOT_ALLOWED. cap2/mem60416M 및 다른 job 상태 불변. CPU 분석만으로 진행한다. 간단한 검산/보고는 SH4 직접, 복잡한 독립 분석에만 bounded subagent를 사용하고 서로의 파일 소유를 분리한다. PNG는 Python 등 직접 작성한 코드로 생성하고 입력/코드/출력 identity를 기록한다. Imagegen/수동 그림 편집0.

Raw-free source+report의 own-scope non-force branch/main 통합은 기존 사용자 권한에 따라 허용한다. 최신 main의 다른 SH/GH 작업을 보존하며 충돌이면 해당 통합만 멈춰 보고한다. Red/CPU 분석 검토는 source/metric·분모·증거/경로/해시 확인이며 생략된 model gate를 다시 선행조건으로 만들지 않는다. 기존 다른 report/실패/부분표를 덮어쓰지 않는다. Raw/tensor/prompt/gradient/teacher/log payload Git0, NO_BROADCAST_NOT_REQUIRED.

첫 표 이후 상세 CPU 리뷰와 본 scope main 통합까지 반복 승인 없이 완료하라. 최종 handoff에 main HEAD/tree, 실행/분석 source 구분, report/manifest/receipt 경로·SHA, 완료/누락 범위, 비용, skipped validation 상태를 간결히 보고한다. 이 리뷰 완료 후 TASK_COMPLETE_STOP / automatic_resume=false. 추가 실험·후속 tuning·타task monitoring은 하지 않는다.
