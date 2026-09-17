# GH → SH4: Sequential Local-z v2 완료 실험 상세 CPU 리뷰

instruction_id: ODEEDIT-S06-SLZV2-COMPLETED-DETAILED-REVIEW-SH4-V1
parent_instruction: ODEEDIT-S06-SEQUENTIAL-LOCAL-Z-ALLOCATION-V2-SH4-V1
nonce: ODEEDIT-GH-SH4-SLZV2-COMPLETED-REVIEW-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 사용자 요청·실행 경계

사용자 원문: **“server4실험 끝난거 자세히 리뷰시키자”**.
직전 인계의 Sequential Local-z Allocation v2 본실험49466_[0–5] 여섯 arm에 대한 명시적 recall이다. 이 scope의 CPU 검산·사실 보고·main 통합을 끝까지 진행하고 종료한다. 초기 gate에서 다시 중지하는 실행-stage 규칙은 이번 완료 리뷰에는 적용하지 않는다. 타 task 모니터링은 재개하지 않는다.

대상 mapping: 49466_0=C45678, _1=N4, _2=F48, _3=G48, _4=C4, _5=C48.
정확 여섯 scheduler 상태/exit/allocation만 한정 확인한다. technical49421의 완료·3379 allocated GPU-sec 및 저장9단계 READY는 기존 봉인 증거 재사용을 우선한다. 이전repair49238/cold7/CAKE/EP/ZFlow/ORBODE scheduler 및 live 자료는 조회하지 않는다.
사용자 완료 말씀과 scheduler COMPLETED, 실제 terminal/commit/분모/state 유효성을 구분한다. 실패/미완료가 있으면 실제 완료 부분·최초 오류·누락을 보고하고 **재제출·재평가·모델 로드 없이** 가능한 CPU 검산만 계속한다. 예상과 다른 RUNNING은 live 과학파일 접근/완료대기 loop 없이 NOT_TERMINAL로 분리한다.

GPU cap2는 정책 기록으로 유지하지만 이번 리뷰 신규 GPU는 **0**이다. Slurm은 지정job read-only audit만 허용, submit/cancel/requeue/hold/throttle/config 변경 **NOT_ALLOWED**. 모델 load/forward/backward/nativefit/teacher generation/evaluator/GPU continuation/new arm/sweep0. CPU torch weights_only/mmap으로 이미 저장된 필요한 tensor를 확인하는 것은 허용한다.

## 정본·lineage·작업 공간

최신 origin/main을 fetch하고 별도 clean codex/server4-slz-v2-completed-review-20260918-v1 worktree와 새 review local namespace를 사용한다. Shared root와 과거 dirty/user 상태를 보존한다.
설계/contract/cells/CPU reference·수렴 근거, 원 envelope, main-gate 정정, 최종인계/실행lock을 전체 읽거나 exact hash가 같은 기존 full-read receipt로 재결속한다.
- plans/global/2026-09-17-sequential-local-z-allocation-design-v2.md SHA fac114d403502c96cbcb33a58b0ff1ee2b0b65b017c44b22a2a22adcda749b73.
- contract-v2.json SHA a7aaac67ffe1f6fb47d372b1eaea62d33d4cf12fcb9e0874fbe92d2750239f8c; cells-v2.csv SHA92f51ceffa031a5373ef782e0d08587c5c4634590932ed646c1099b7630ac9ad.
- messages/head/2026-09-17-sh4-sequential-local-z-allocation-v2.md 및 2026-09-17-sh4-slz-v2-main-initial-gate-override.md.
- 실제 frozen execution21297ec19e7f5aecec16d2fdb14cc79380a1df94/tree26be0758ee75503161c7cafffdec8397e6cf8165.
- archive c97083a1e1039c059b082bcf0ad0de1143bbfbe41a26be1db5c5d7a1bf54a7a8, execution.lock a41cb76a25ab98b02c043397cd9cf4db0768e9ae312931b349008c3e9b303d18.
- 인계 main38fe0d39a768d7d0eecdf4f232f0e76355508187/tree169439a5cb81227014ed714022a14e36aa861cdb.
- local /data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/resume-r1/resume-manifest.json SHA dd5b95aa340dd12bcc371b1558592c632bc008494dcc98dcb600f32a0eafd52e.
- raw 각 arms/{C45678,N4,F48,G48,C4,C48}/attempt-v1/output.
- technical READY SHA d4ed2138d658ade5c5b565eb5df1ca8241e2fe0e4f054a2ad6c39977d3782666 및 기존 submission/resource/history 보존.

실행 source, submission control source, 이번 analysis source와 publication commit을 분리한다. 2026-09-17T07:16:03Z의 마지막 agent 관측은 main6개 PENDING/Resources와 GPU8/8였고 initial scientific gate 미관측이었다. 지금 저장 증거로 B1을 검산해도 과거 실시간 initial 관측을 PASS로 소급 변경하지 않는다.

## 첫 중간 산출물: 실제 W10 여섯 arm 표

완료별 actual W10 full first1000 raw true/new NLL pair를 독립 reducer로 계산한다. case/prompt/target/order/finite/tie/endpoint identity와 cardinality를 먼저 확인한다. RS/PS는 new<old, NS는 old<new, tie=failure라는 기존 canonical 정의를 유지하고 실제 schema에서 결속한다.
전량일 때 분모1000/2000/10000. 완주60batch/6000 arm-request/unique1000은 **기대값**이고 실제값을 검증해 별도 보고한다. No-update/ownN4 fallback도 요청분모에서 빼지 않는다. 실패 arm에 집계값을 추정하거나 부분 current pooling을 final1000으로 쓰지 않는다.

첫 final table은 보고 순서 N4/F48/G48/C4/C48/C45678로 제시하고 count/분모/%/ΔN4 pp/status와 first-table SHA를 중간보고한다. 이것은 전체 source/state/비용 검산 완료가 아니라는 경계를 유지한다.
같은 실행의 새 N4가 주 대조다. 과거 cold7 N4/LD/REFIT4·다른seed/10k/warm 실험은 필요할 때만 이미 게시된 aggregate 참고표로 완전히 분리하고 이번 paired 주표에 대신 넣지 않는다.

## 상세 수치·paired 분석

1. Current entry/at-write와 실제 W5 first500/W10 full1000, W10 same first500 및 뒤500을 분리한다. Fullseen과 current pooling의 분모/해석을 섞지 않는다.
2. R/P/N true/new NLL·desired margin·TF strict/token/two-P를 각각 정의하고 기록된 수준만 독립 집계한다. 평균·중앙값·분위수/악화 tail 및 개별 loss/gain ID를 함께 제시한다.
3. W0-success-conditioned NS, atwrite→later, W5→W10 first500 유지/소실/회복, age/cohort와 ALL/ACTIVE/SUPERSEDED 및 정당 overwrite를 분리한다. 미관측 batch 사이의 정확한 최초 실패 시점을 추정하지 않는다.
4. primary paired 수치: C4−N4, C48−C4, C48−F48, C48−G48, C45678−C48. 동일 prompt/target identity로 결속하고 success lost/gained 및 NLL 변화 분포를 제시한다. Case 단위 불확실성을 계산하면 동일case의 R/P/N 관련행을 묶고 방법/seed/반복수를 기록한다. Prompt를 독립 표본으로 부풀리지 않는다.
5. S64 online B와 Dev128 observer를 별도표로 한다. B1/5/10 후보 official P/N은 실제 저장된 completed/scored endpoint만 사용한다. Budget 미완료 후보 재생성0, 없는 paired 관측은 NOT_RECORDED다.
6. 최종 RS/PS 감소를 NS 증가와 합쳐 하나의 성공 판정으로 숨기지 않는다. 한 개발stream으로 보편적 비열화·우월성/인과효과를 주장하지 않는다. SH는 사실·산술차이·coverage를 보고하고 GH가 별도 해석한다.

## “설계대로 작동했는가”를 실제 코드·저장 증거로 점검

설계항목→실제file/function/line/source SHA→저장artifact/독립 CPU 검사→PASS/FAIL/NOT_RECORDED 검증수준의 conformance 표를 포함한다. 문서의 의도·코드 경로·실제 관측을 구분한다.

- 모든 arm W0/zeroM4..8/common context·tokenizer·modelrevision·FP32/eager/TF32off/seed20260916/fixed prefix가 같았는가. Effective teacher/S64/Dev128/Past64의 문항·토큰·teacher identity와 데이터분리를 검산한다.
- 매batch ownWe fullL4 fit 한 번, 후속 layer는 ascending current-prefix local-z/key/solve인가. Prefix/gate가 변할 때 실제 target/fit hash·receipt·cache hit/miss가 그 변경에 맞는지 확인한다. 같은 layer gate만 바꿀 때 동일prefix fit 공유와 정확0 skip/1 endpoint copy가 규약과 맞는지 검산한다.
- L5–L7을 포함한 native L2/lr/decay/clamp/KL/max25loss·24Adam/earlystop이 frozen source대로인지 확인한다. 24step cap을 수렴 또는 덜 학습된target 증명으로 부르지 않는다. 관측된 zero-step/cap-stop/native loss 구성·clamp를 분포로 제시하며 미저장 마지막gradient는 추정하지 않는다.
- F48은 raw(.75,.5) 항상 commit, 품질은 observer인가. G48은 .05 plateau 없는 v2 guard로 새 실행됐는가. C4/C48/C45678와 F48의 예외를 혼동하지 않는다.
- Current/Past E/H≤ownN4+1e-4, strict/pair 성공ID subset, epsilonB1e-6·global best tie 순위가 실제 기록과 맞는지 independent selection replay로 확인한다. 평균E/H·canonical NLL·PS를 별개로 표기한다. Pair target 없는 경우만 비활성, 임의 oldtarget 복원0.
- 실제 SciPy1.15.3/u=1−a/u0=0/rhobeg.25/tol.01/catol1e-8/maxiter64/정확한 scale과 raw-u bounds-before-transform, out-of-bounds GPU0/clip0, objective/constraint 동일score 공유를 대조한다. Solver 반환값과 실제 selected incumbent를 구분한다.
- budget search32+prune8 fits, score24+4, extraAdam9600/wholefit2400 reservation 및 releaseunused, mandatoryN4 별도, incomplete candidate score배제, typedBudgetExceeded를 ledger로 재계산한다. 거절·불완전작업 비용도 포함한다. 각stop 이유를 숨기지 않는다.
- support/gate 연속값 분포, 실제 nonzero weight layer와 단순 nonzero gate를 분리한다. Pruning8→7→6→5의 시도/완료/제거/미완료, L4 제거금지, 작은 gate snap0, global epsilon 손실 누적0을 검산한다. 삭제검사가 미완료이면 그 layer 필요성 검증으로 쓰지 않는다.
- 각 batch completed search vectors≥d+2 및 distincta4≥2의 count-proxy와 실제unique endpoints·INSUFFICIENT_SEARCH를 명시한다. Main에서 미달이어도 결과를 버리거나 예산변경하지 않았는지 확인한다. 기술 first100 coverage11을 모든main batch에 자동 복사하지 않는다.
- 최종 selected endpoint에서 M4..M8 정확히1회씩 append, inner append0/게이트가중0/zero-step키제외0/과거key refresh0. 완주 시300 append와 각arm9link,총54link는 계획 기준으로 실제증거를 확인한다. 이전 다른 layer 누적분을 유지한 ownN4와 독립 N4 chain을 구분한다.
- Official P/N·Dev128은 selection/next-state seal 후만 관측됐는가. Observer가 controller input에 섞이지 않았고 W/M/RNG 복원이 기록됐는가. Adaptive solver 호출순서 자체의 permutation invariance를 새 요구로 추가하지 않는다.
- 대표 실제 경로 예시: C48/C45678의 선택·추가층지원/zero prune/ownN4 선택/품질탈락/예산미완료 중 실제 존재하는 사례를 골라 배치entry→fit→search→prune→selected→history를 수치로 설명한다. 없는 사례는 없다고 기록한다.

**W/M disk checkpoint0**가 의도된 사용자 저장정책이다. CPU target/key/delta 등이 남아 있으면 가능한 해당 수준만 확인한다. Commit hash/receipt/실행guard와 사후독립 전체selected W/M 재구성은 같지 않다. 자료가 없으면 NOT_AVAILABLE/NOT_RECORDED, GPUoff-on/continuation NOT_TESTED를 유지한다. 삭제cold7CP 재생성·없는 fullmodel tensor 검증PASS0.

## 비용·자원·무결성

새 main6 allocation GPU-sec와 technical49421 3379sec, 재사용teacher/prior 준비 비용을 분리한다. job/step/extern 중복합산0. Interval로 실제최대동시 GPU와 cap2를 대조하고 allocation을 utilization이라 부르지 않는다.
계획 상한을 actual counter로 복사하지 말고 native targets/Adam/loss/solve/score를 receipt의 비중첩 호출 기준으로 재집계한다. 원설계상 max targets89000/Adam408000/loss497000/solve890/score960/history300과 실측을 구분한다.
Prefix fit·target·solve·teacherstream·objective/constraint/observer F/B/tokenwork·history·cache/restore·state I/O·totalwall/peak memory, 기록된 rejection/incomplete비용을 함께 제시한다. Nestedtimer중복가산0; 순수writer/IO가 분리되지 않으면 NOT_SEPARATED.
기술에서 uninstrumented800 reference target의 Adam/loss 미계측과 nativeinclusive2315.1324sec의 중첩을 유지한다. 新44GiB reserve와 최초64GiB reserve는 추정계획, 실제파일 bytes/공유volume 여유/저장누락을 구분한다. 공유disk 변화의 독점 원인을 추정하지 않는다.

완료범위 raw member allowlist/fileSHA/size/schema/caseorder/eval endpoint/terminal·commit·선택chain/source config를 CPU로 봉인한다. 큰 model/공통teacher 전량재해시는 exact기존 evidence+현재binding으로 재사용 우선, 이번 fullrehash와 이전 evidence재사용을 구분한다. Canonical output은 수정0, 분석중 schema/path 처리수리는 새 analysis code에서만 하고 회귀검사한다. Raw missing/corruption은 기술한계로 보고, 자동 raw 재계산0.

## 산출물·중간보고·완료

중간보고:
1. FULL_READ/M0 + exact6job terminal/실패/미완료 표. GH중복검사 요청0.
2. 첫 actualW10 여섯arm 독립NLL표 + 검증수준.
3. 중요한 source/state/denominator 불일치 또는 typed한계가 발견되면 근거와 즉시 보고; 자동재실행0.
4. 한국어 상세 factual report/package 및 own-scope main완료.

최종 report에는 실제scope/오류·누락, final/누적/cohort/paired표, source-conformance, gate/support/search/pruning/질적조건 결과, 실제cost/memory/storage, 증거수준/미측정, 재현명령을 포함한다. 후보 비교곡선은 저장된 실제 endpoint만 표시하고 미측정 성능을 interpolation으로 관측값처럼 표시하지 않는다.
CSV·manifest·rooted receipt/raw identity inventory/source-analysis lineage를 봉인한다. PNG는 repository 코드로 생성하고 재현·육안확인, GFM표/실제HTML렌더 가능범위/열수/링크/수치산술을 검산한다. 미설치도구 검사를 PASS로 쓰지 않는다.
SH factual-only 원칙 유지. 결과의 우열·기전적 인과·후속실험 선정은 하지 않는다. 품질조건·비용차이 및 추가층의 실측 지원범위는 그대로 공개한다.

## 허용 write·publication·종료

- project/run_scripts/sequential_local_z_allocation/analysis/completed_review_20260918/ — 새 CPU reducer/audit/tests/plot; 기존 실행runtime/locks변경0.
- /data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/completed-review-20260918-v1/ — 새analysis/artifact receipts/worktree scratch, 원 arms/technical/raw는 read-only.
- audits/servers/server4/2026-09-18-sequential-local-z-allocation-v2-review/
- experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2/completed-review-20260918-v1/
- messages/acks/server4/2026-09-18-sequential-local-z-allocation-v2-review.md
- messages/server-heads/server4/2026-09-18-sequential-local-z-allocation-v2-review.md
- runs/odeedit_slz_v2_review_s4_20260918_v1/
- tasks/status/odeedit_slz_v2_review_s4_20260918_v1/server4.json

위scope 구현/CPU검사/보고서 및 compact상태만 최신main에 **nonforce push ALLOWED**. 타SH/GH 동시 변경 보존, 충돌 시 exact범위 보고. 공유global설계/native/helper/환경/원실행source·원raw 수정0. NewGPU/model/eval/Slurm mutation/rsync/삭제0. NO_BROADCAST_NOT_REQUIRED, raw/tensor/prompt/fullstdout Git0.
Red 관점의 source/data-leakage/selector/cost/state/noCP 한계와 publication 검사를 수행하고 별도독립red 여부는 사실대로 기록한다. 과학block은 숨기거나 threshold완화하지 않는다. Generichelper가명시허용path를거부하면 제한과envelope근거를기록, 허위PASS/공유정책수정0.
이 완료리뷰는 각단계 재승인 없이 최종보고/main까지 계속한다. 후기권한은 **CPU 분석/보고**뿐이며 새science실행까지 확장하지 않는다. 최종compact수치·report/manifest/receiptSHA/mainHEAD·한계를 GH에 전달한 뒤 TASK_COMPLETE_STOP. 자동모니터/후속실험0.
