# GH → SH4: 종료된 Server4 실험 상세 사실 리뷰

instruction_id: ODEEDIT-S06-SERVER4-COMPLETED-CAKE-CAP-DETAILED-REVIEW-SH4-V1
nonce: ODEEDIT-GH-SH4-COMPLETED-CAKE-CAP-REVIEW-20260916-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit
사용자 최신 원문: “server4에서 끝난 실험들 자세히 리뷰시켜”

## 1. recall의 범위와 권한

현재 사용자 호출로 **완료된 실험의 CPU 분석·사실 리뷰·보고·main 통합**을 재개한다. 주 신규 대상은 CAKE48101과 cap sweep48148/48149/48150, CAP1=47962는 기존 완료 리뷰를 재사용한다. Server4의 기존 완료 실험들은 inventory에 포함하고 이미 상세 리뷰가 있으면 정본 결과·검증 수준·보고 경로를 링크한다. 종료 실험 목록과 미완료/실패를 먼저 구분하며 모든 역사 raw를 다시 감사하는 task가 아니다.

SH4의 최근 로컬 task에서 alpha sweep 모니터링을 사용자 지시로 중지했음을 GH가 확인했다. 이번 요청은 completed-review recall이지 새 GPU 실행이나 미완료 job의 계속 모니터링 허가가 아니다. 해당 정확 job들에 **한정 terminal scheduler 조회**를 하되 RUNNING/PENDING이면 NOT_COMPLETED로 inventory에만 기록하고 live scientific raw 접근/완료대기/반복 polling 없이 나머지 완료된 범위를 진행한다. 실패/timeout/cancel이면 보존된 terminal evidence와 최초 오류를 CPU로 진단하고 부분 결과를 terminal-valid로 승격하지 않는다. 새 submit/requeue/restart/cancel/hold/throttle/GPU/model-load/forward/evaluator 실행은 **not allowed**다.

원본 사용자 instruction/source/config/lock/기존 pause 및 제출 PENDING 관측은 역사 provenance로 그대로 둔다. 현재 terminal 상태를 과거 initial gate PASS로 바꾸지 않는다. 이 리뷰를 완료할 때까지 분석·보고는 진행하되 task 완료 뒤 STOP; daemon/callback/heartbeat/자동 후속 실험0. 기존 ORBODE 중지 감사, SH2 SL-ZFlow와 다른 paused task에는 접근·재개0. GH 중복 raw/GPU 감사나 단계별 재승인을 요구하지 말라.

## 2. 우선 대상 identity

### CAKE — fixed10k B100×100

- job48101 / odeedit_cake_native_lifelong_s4
- execution7884aeb6000f8343139172825ec6c4ca24357fc0 / tree4dabdb1e408726974ac0f91285ad35bac698b88e
- archive983589f0b6ca5dcca85508bdaa2d5a3a1d38128697ec89dedf54f5ede1f16ebf
- lockf2ade3ee9dbdabdc6a1ac00a9d36b0e902710a44cf5e2a6a49c6d861e18eb401
- root /data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1
- output/main, logs/48101.out/.err; submission publication experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/
- upstream CAKEc8243e1d7e43ca9cf64d552f96221fcb9561aac2 / tree4f59249bb23c7cacf0f6490bce8127c74ff7b111
- 사용자 override: **W/M tensor checkpoint 저장하지 않음**. 12시점 fullseen 평가와 memory history는 유지. 없는 CP를 추가 생성/복원하기 위한 forward0, hash-only를 tensor재해시/정확restart/continuation PASS로 오기0.

### EP-TW-1 alpha cap sweep — W0 first1000 B100×10

- CAP10=48148, CAP100=48149, NORM_ONLY=48150
- execution8c64366c2314f188e034e5f4403fe89f0eaad873 / tree76ef5072b70134bf4bc1f9d3659a056041922bde
- archive304be4f3de58ce2f9457f63324646525000191609ba64e1bb1e290253e59ea21
- CAP10 locke1ecd43a3b26162dc9cbecfa83ca9ac6f79fd76a29d5600c7273f1db7cab70b4
- CAP100 lock6a1c34a722849f73b2dd8203d2190cfbbb147374802a1f33e89118eb9c742eeb
- NORM_ONLY lockfda2657e39ba00af78b0ff554c1a2af9d75204601fcaa3984c7a6c4e0bee8b1b
- root /data/janghj/ODE-edit/local/ep-tw1-alpha-cap-sweep/20260915-v1/
- outputs {CAP10,CAP100,NORM_ONLY}/attempt-v1/scientific-v1
- CAP1 reuse47962 source6d317bdb2660d7e9919bc3a9fb878564e9729e37 / lock5a19c2be919362d08b2ea80e69a406d7b6de569aade8de639036f69db5d5d8f9
- CAP1 정본: experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1/; GH 표 렌더 수정c18f961 및 갱신 manifest를 유지한다.
- alpha sweep 기존 준비 분석/자체 검산/branch a7bfd2f 및 그 이후 실제 local tip을 확인해 재사용하되 execution8c64366과 분석/게시 source를 분리한다. 마지막 source branch push 이후 수정/dirty를 덮거나 되돌리지 않는다.
- CAP1 현재 CP부재와 과거 CPU검산은 분리. 신규3arm은 당시10CP/arm+W10 저장 계약이므로 현재 실물/기록을 실제 확인한다. CAKE 저장 waiver를 sweep에 적용하지 않는다.
- SKIPPED_USER_DIRECTED / numerical_validation=NOT_ESTABLISHED 유지; FD/ULP/direct/selfKL 진단을 이번 리뷰에서 재실행하지 않는다.

정본 alpha 원문/설계/contract/cells/GH dispatch·6SHA/input locks, CAKE source/native/hparams/배포 차이/최신 no-checkpoint override를 전체 읽거나 exact이전 FULL_READ receipt로 결속한다. 새 GH envelope와 마지막 pause는 반드시 읽고 현재 리뷰 source를 별도 seal한다.

## 3. inventory와 첫 결과표

첫 M0: 실제 source/문서·job→arm→scope 매핑, terminal 상태(현재 한정 조회), 결과가용성, 신규 대 기존재사용, 누락/기술실패, 분석 작업 경로를 전달하라. Scope 밖 전체 scheduler/타사용자 query는 하지 않는다.

그 뒤 independent raw NLL reducer로 다음 **서로 다른 표**를 먼저 보낸다.
1. CAKE actual W100/full10000 R/P/N counts 및 실제 분모10000/20000/100000. 중간 완료라면 해당 상태와 분모만 명시하고 final10k로 표기하지 않는다.
2. CAP1(reuse)/CAP10/CAP100/NORM_ONLY actual W10/full1000 counts, 분모1000/2000/10000. 미완료 arm은 명시 NA/NOT_COMPLETED이며 drop/imputation0.

GH가 원격 raw 중복집계를 하지 않는 소유 경계 유지. First table은 metric/order/pair/hash 확인 범위를 명시하며 전체 state/CP/설계 일치 감사 완료와 분리한다. 최종 completion 전에 모든 신규 결과의 audit를 마무리한다.

## 4. CAKE 상세 분석과 기존 baseline 비교

100 committed B100/unique10000, 각 target/key/solve/history count는 source 기대값과 actual 관측을 분리한다. 원 apply당 L8 target100→L4..8 native sequential residual/directsolve→최종5history 각1, wrapper 중복finalize0, P physical4..8→asset0..4를 확인한다. causal_scores의 원 key0..4, temperature.1, L2=10/decay.4/clamp.5, native weight allocation과 남은 residual 처리식을 actual source/function/line/hash 및 저장 telemetry와 연결한다.

“CAKE 원본이 설계대로 호출됐는가” 장을 두고, 미사용 notebooks.util import1줄/EOF LF 외 native source변경, 실제 import/runtime/backend 차이를 명확히 한다. 원README torch2.6/transformers4.51.3 대비 실제2.9.1cu128/4.44.2, old native decay.5/clamp.75 등 차이를 숨기지 않는다. source一致·runtime구조 확인·actual tensor재현/미측정 수준을 각각 표시한다. checkpoint 미저장으로 independent W/M tensor 검증은 불가능할 수 있음을 그대로 기록한다. 이를 검증하기 위한 새 모델 실행0.

각 batch Current와 all-seen rewrite, 12시점 actual fullseen R/P/N, 최종10k·동일 first1000을 **가용 원시 결과 안에서** 비교한다. RS/PS newNLL<trueNLL, NS trueNLL<newNLL, tiesfailure, strict/token 별도. current행을 실제 state/prompt identity로 재사용한 경우만 중복을 제거한다.

비교는 동일 fixed10k순서의 기존 W0/NATIVE_ALPHAEDIT/NATIVE_MEMIT/MEMIT_BLUE/AlphaEdit_BLUE 및 L4..8 singleton을 사용한다. 주표는 MEMIT/AlphaEdit 계열을 따로 두고 CAKE를 AlphaEdit 계열 native/BLUE/L4-only와 명확히 나란히 놓는다. Original은 단독 라벨로 쓰지 말고 BASE_ALPHAEDIT_NATIVE / BASE_MEMIT_NATIVE / AlphaEdit_BLUE(L4+L8)처럼 명시한다.
기존 정본: experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/, blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3/ 및 fivearm1k. 완료 sealed 결과를 재사용하고 새 비교의 source/model/seed/context/evaluator/평가시점/order/cardinality를 대응시킨다.

원시 case/prompt/target identity가 맞는 범위에서 atwrite→final, 최초 실패/후속 loss/recovery, cohort age/early-middle-late, active/superseded, margin/true-new NLL 분포·tail과 paired 전이를 분석한다. 현재 남아있는 local raw가 없으면 aggregate 비교만 표시하고 행별 joint분포/미관측 failure시점을 추정하지 않는다. 원시 재이관이나 GPU 보완은 요청 범위가 아니다.

## 5. Alpha sweep 상세 분석·설계 작동 감사

4arm 모두 공통W0/M0 first1000/order/model/tokenizer/context/teacher/P/원 floating 경로를 비교하고, 신규 arm 각각 자기 trajectory의 z/K/A/gE/gD를 썼는지 확인한다. CAP1 미래방향 replay와 다른arm state 재사용0. cap1/10/100 bounded와 disabled/null이 유일한 과학 설정 차이인지 executed frozen source diff를 남긴다.

방법 동작을 수식·실제 source·저장 수치의 세 층으로 상세히 설명하되 사실만 기록:
- alpha_norm=.25||actualVp-Wentry||/(||dA||+epsilon), bounded min(cap,alpha_norm), NORM_ONLY=alpha_norm.
- cap-bound와 norm-bound 횟수; alpha_norm/alpha_used, pre/post-ball norm, 25%trust, retraction/halfspace 활성·실제 <gE,C>, linear prediction과 actual E/D 분리.
- RAW/C1/C05/C025 모두 생성·finite·실제FP32 materialization·byte-dedup 여부.
- E<=ownRAW E (positive allowance0), strict exact-ID set 보존, minactualD64/RAWtie, 전 후보를 평가했는지 및 선택 재집계.
- 最大C1 크기, RAW=0 포함 selected 크기, nonzero selected만의 크기를 나눠 correction/native 비율·분포 및 RAW 빈도를 비교한다.
- ball/trust 후 방향과 scalar shrink의 관계, teacher와 gE/gD분리, accepted-ledger/history exactly1/비선택parameter 보호를 기록. 저장되지 않은 tensor·gradient는 source확인 수준 이상으로 확대하지 않는다.
- 각 batch ENTRY→RAW→selected는 same-state local counterfactual이며 **독립 native sequential baseline이 아님**을 명시한다.
- B1과 대표 corrected/RAW batch worked example을 저장 값으로 제시하고, cap증가 자체가 실제 선택 보정/성과 증가를 의미한다고 선기록하지 않는다.
- 모든40logicalbatch(10reuse+30new, 실제 가용성에 따름), 신규27 W/M links/30CP/1000fit perarm 등 기대 inventory를 actual과 대조. No raw를 삭제/덮어쓰지 않는다.

비교표: W10 all1000; W5first500→W10same500; last500/current/atwrite→final; R/P strict·two-P·true/new NLL/margin/tails; case/prompt별 loss/gain/active-superseded. 필요시 요청 단위 paired CI를 정확 sample clustering과 산술 정의로 표기하되 선별에 사용0. 모든 반대 방향 수치·misses·RAW 포함.
S64는 선택에 쓰인 값, Dev128은 별도 observer로 분리하며 미측정 Report/Audit/MMLU/FutureN은 NOT_MEASURED. Existing N4/BLUE/L4-only/가용 baseline의 **동일 first1000**만 비교. CAKE W100/full10k를 EP W10/1k와 같은 표의 직접 우열로 혼합하지 않는다. CAKE B10 actual first1000가 있으면 그 별도 시점만 1k 참고표에 포함한다.

## 6. 비용·검증·오류 경계

실제 Slurm allocated GPU-sec/GPUh·최대 allocation overlap과 실행 timer를 분리한다. CAKE 포함cap2 사용은 bounded scheduler evidence로, compute utilization은 미측정이면 미측정. 추정6.4117GPUh 등을 실측으로 사용하지 않는다.
native target/Adam早停/losssteps/key/solve/history, EP E/D/all후보forward(탈락포함)/선택/eval/I-O/peak를 분리. nested시간 중복합산0, purewriter 미계측이면 NOT_SEPARATED.
CAP1 7694GPUsec/teacher98/기술실패473+69은 재사용·prior 별도이고 신규3arm에 중복 청구0. CAKE 비용은 자기 실제job ledger 별도. 새 review GPU시간0.

신규 output manifest의 실제 가용 member SHA/size, chain terminal→commit→평가 state/order 결속, pair cardinality/finite/ties, epoch batch대분모, 저장CP CPU weights_only/schema/tensor hashes 및history exactlyonce를 검산하라. 기존 완료검증/대형 immutable asset 검사는 identity가 같으면 재사용하고 검증시각·수준을 밝힌다. CAKE W/M tensor 미저장을 error/backfill 대상으로 바꾸지 않는다.
No off/on parity·미분·GPUcontinuation 검증을 CPU hashPASS로 대체하지 않는다. CAP1 기존CP 부재/원래검증도 현상과 역사를 나누어 설명한다. Framework/hardware동일 여부와 수치동일성 별도.

분석 코드 오류는 별도 analysis attempt에서 수리 가능. 실행 알고리즘 오류를 발견하면 source-backed RCA/영향범위/partial validity만 기록하고 재실험·새수치gate 없이 GH에 보고한다. 완료도 안 된 run을 scientificPASS로 하지 않는다. Whole integrity 완료 여부와 metric 재집계 수준을 분리한다.

## 7. 구현·보고·publication envelope

새 clean branch codex/server4-completed-cake-cap-review-20260916-v1, 별도 worktree. Existing sweep analysis source/localtip를 선택적으로 재사용하고 old dirty/실행/실패/기존 완료보고 bytes를 보존한다.
허용 write:
- project/run_scripts/server4_completed_review/ 아래 CPU inventory/reducer/source-audit/report/plot/tests 신규 namespace.
- project/run_scripts/bg_tw_reference/ep_tw/의 기존 cap sweep **분석·보고 모듈만** 새 worktree에서 최소 보완. numerical policy/runner/fitter/model/evaluator실행 파일 수정0.
- /data/janghj/ODE-edit/local/server4-completed-review/20260916-v1/ 아래 analysis/firsttables/receipts/staging. 원 과학raw 경로는 read-only.
- audits/servers/server4/2026-09-16-completed-cake-cap-review/
- experiment-reports/servers/server4/completed-experiments-review-2026-09-16-v1/
- experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/
- experiment-reports/servers/server4/ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v1/
- messages/acks/server4/2026-09-16-completed-cake-cap-review.md
- messages/server-heads/server4/2026-09-16-completed-cake-cap-review.md
- runs/odeedit_server4_completed_cake_cap_review_20260916_v1/
- tasks/status/odeedit_server4_completed_cake_cap_review_20260916_v1/server4.json

상위 종합 report에 run-inventory.csv/evidence-reuse-manifest와 각 family 상세 diagnostic-report-ko.md 링크를 둔다. CAKE/sweep 각각 최종표·source대설계작동·누적/전이/비용/한계/미측정/재현 명령·raw inventory/manifest/rooted receipt가 필요하다. 이미 정본 상세 보고가 있는 역사적 실험은 result/identity/검증수준을 재사용하고 독립 새검산이라고 오기하지 않는다. 누락 정본 추가 리뷰가 필요한 **완료된 현재 관련 run**을 발견하면 exact대상/범위를inventory에 먼저 남기고 CPU 리뷰로 한정한다. 중지된 ORBODE science audit는 재개하지 않고 정본 링크만 둔다.

SH factual-only: 사실·산술·명시된 검증 수준·source-backed 기술 RCA/NOT_RECORDED. 우열·기여율·효능 원인·후속method 선택은 GH 별도 global review로 남긴다. 자료가 효과를 보였다고 scientific promotion하지 않는다.
PNG는 직접코드생성만, byte/hash/명령; Markdown표 pipe/행열/렌더 검사 및 manifest동기화. 기존 GH tablefix보존, 임의CRLF sealedbyte정규화0.

본scope 완료 code/test/report만 latestmain 포함 확인→clean integration/non-force push를 승인한다. CPU focused/reducer/산술/source/inventory/PNG·표렌더/raw-free/access pre/postrun검사와 실제 independent/자체감사 수준을 명시. Raw/tensor/prompt/fullstdout Git0. 이미 main인source 중복merge0; 분석에 필요한 미게시 own cap source는 실제 execution immutable identity·closure를 확인해 ownscope로 통합 가능하다. 역할helper가 exact 사용자승인분석경로를 거부하면 그 경로와 허용문맥을 기록하고 helperPASS로 오기하지 않는다; helper/sharedPROTOCOL변경0.
NO_BROADCAST_NOT_REQUIRED: server4 로컬 read-only 분석, 새로운 rsync/delete0. Git 충돌은 임의 ours/theirs/reset으로 해소0.

최종 mainHEAD/tree/보고경로/SHA/manifest/테스트/대상완료·실패·미완료/새GPU0/남은미측정 포함하여 GH에 direct handoff 후 TASK_COMPLETE_STOP. 과거 pause 중지 때문에 미확인인 부분을 먼저 사실로 정리하고, 사용자 호출에 해당하는 완료 보고를 충실히 끝내라.
