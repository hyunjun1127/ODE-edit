# GH → SH4: R512-G256 matched B1 상세 완료 리뷰

Instruction ID: `ODEEDIT-S06-EN-REUSE-R512-G256-B1-COMPLETED-REVIEW-SH4-V1`
Nonce: `ODEEDIT-GH-SH4-EN-REUSE-G256-B1-DETAILED-REVIEW-20260919-R1`

## 1. 최신 사용자 recall과 권한

사용자: “odeedit_en_reuse_g256_B1_s4 자세히 리뷰시켜”. 이번 지시는 정지된 해당 task의 **CPU-only 완료 상태 확인·상세 사실 리뷰·own-scope main 게시**를 재개한다. 기존 초기/준비/등록 시점의 PENDING 기록은 역사 그대로 보존한다.

대상 server4, session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, CWD `/data/janghj/ODE-edit`, repository `hyunjun1127/ODE-edit`. GH session `01a04939-8873-7673-8dca-4c7fc5e31af0`. 최신 origin/main을 확보하고 전용 clean `codex/server4-en-reuse-g256-b1-completed-review-v1` worktree에서 수행한다. 원 실행 branch/dirty/partial/teacher/raw/source를 보존한다.

확인 대상은 **B1 job50449 / odeedit_en_reuse_g256_B1_s4**, 준비 **50410 / odeedit_en_reuse_g256_prep_s4**다. 정확 owner/name/source/submission/execution lock을 먼저 대조하고 지정 job accounting을 한 번 읽는다. 원 afterok:50410 등록과 실제 실행을 구분한다. Step/extern과 parent allocation을 중복 합산하지 않는다. 다른 job의 scheduler/log/raw 조회는 없다.

이 요청 자체가 terminal 증거는 아니다. 예상과 달리 pending/running이면 확인한 상태와 남은 범위만 보고하고 반복 모니터링 없이 인계한다. 실패/부분완료이면 보존 자료의 CPU RCA 및 partial report까지 수행하되 자동 repair/rerun은 하지 않는다. Exact identity 불일치는 명시하여 다른 job으로 조용히 대체하지 않는다.

**신규 GPU/model load/forward/backward/generation/evaluator 실행·Slurm write·재제출·checkpoint 삭제/이동·새 원격 raw 전송은 승인하지 않는다.** Project cap2는 유지하지만 이번 리뷰 신규GPU0이다. `max_batches=1`, `sequential_authorized=false`, `auto_continue=false`; B2/chain/추가 arm 예약도 없다. 본 리뷰/게시가 끝나면 STOP한다.

## 2. FULL_READ 및 provenance

이번 envelope 전체와 원 `messages/head/2026-09-19-sh4-en-execution-reuse-r512-g256-b1.md`, 최신 사용자 pause/afterok 제출 기록 및 실제 resume/submission/execution lock을 읽는다. 아래 정본은 기존 FULL_READ exact SHA가 동일한 경우 해당 receipt를 결속해 재사용하고, 변경된 것은 전체 읽는다.

- `plans/global/2026-09-19-en-execution-reuse-r512-g256-design-v1.md` (SHA b671e1eb26d01068b561cf32144d66291f43c8f11505d550e6945229b43b16bd)
- 같은 이름 `...contract-v1.json` (SHA 84dc6d7b35afb21cb095606a490132ccfc3b72bdbd5ac6ef3de2d241ce1700a0), design-checks audit 및 연결된 EN 설계/contract/source/보고/PROTOCOL.

준비 execution a297039dc756a0e8953e4bab4695e66361a161ac와 이후 구현5133ae1 및 **실제50449 frozen source**를 구분한다. 마지막 구현 commit을 실행 identity로 추정하지 않는다. 실제 B1 archive/lock/config/import/model/tokenizer/hparams/FP32/P/context/request order/native capsule/teacher/geometry를 결속한다. Root `/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/` 안의 정확 제출 자료에서 B1 output을 찾는다. 타task live raw는 열지 않는다.

## 3. 첫 표와 독립 endpoint 검산

우선 실제 존재하는 완료 endpoint의 첫 표를 중간보고한다: shared N4, LEGACY_SCHEDULE_R512_G256, REUSE_SCHEDULE_R512_G256. 정본 R100/P200/N1000의 true/new NLL, strict 부등식과 ties=failure, case/prompt/target/token/order/finite/cardinality를 독립 CPU reducer로 검산한다. 같은 weight SHA라는 이유로 평가 결과를 대입하지 않는다. 재사용 관측은 원 endpoint·evaluator·model/input identity와 재사용 범위를 표시한다.

R/P/N 성공 ID lost/gained, strict rewrite/two-P/joint, NLL 변화·tail, 가능하면 W0-correct neighborhood retention을 남긴다. 없는 관측은 NOT_MEASURED이며 새 GPU로 보완하지 않는다. S64 과거 결과와 새 R512/G256을 같은 목적함수/분모로 합치지 않는다. R512 train KL과 generated Dev128 observer를 구분하고 Report256은 개봉하지 않는다.

## 4. 핵심: 실행 중복 제거가 설계대로 작동했는가

설계 조항→실제 frozen file/function/line→저장 evidence→판정/한계를 표로 연결한다.

- 두 schedule이 같은 WN/geometry/R512/G256/data adapter/head/reduction 경로에서 **독립 G/H와 trial**을 계산했는지 확인한다. Timed arm 간 gradient/candidate/선택 공유 여부를 명시한다. Shared native/teacher/불변 geometry와 method 계산은 구분한다.
- L과 문서별 loss, G/H/chi/eta0, trial별 실제 FP32 candidate bytes/SHA, Armijo·Current/Past·invariant, 첫 accepted index/stop/fallback 및 selected endpoint를 비교한다. Pure dedup exactness와 원 scientific protection tolerance 통과를 혼동하지 않는다. 불일치는 크기·위치·원인 미확정 여부를 그대로 기록한다.
- EndpointSession의 WN/current candidate 2slot, CPU immutable owner 및 alias/epoch/teacher/input/trial/coverage invalidation, H2D call/bytes, Current hidden/score와 native anchor 재사용, partial/stale graph 차단을 source+actual counter로 확인한다. CPU fixture만 있는 분기는 actual로 확대하지 않는다.
- Cached geometry가 동일 dtype/backend/block/reduction route인지 확인하고 actual DK/leakage/FP32/logit/NLL 검사를 cache hit만으로 생략하지 않았는지 확인한다. Physical 검증은 optimized cache와 독립인지, 실제 범위를 제한해 기록한다.
- 원 최대8trial/gradient sweep 최대1회/512문서 backward·반감·첫 통과 선택·guard 순서·finite/rollback이 유지됐는지 확인한다. Optional head chunk/FP64 누적/generation KV는 실제 적용 여부와 영향만 기록한다. `/data/janghj/tmp/dnm/hooking.py` 등 별도 점검 코드를 새로 적용하지 않는다.
- B1 Past 없음은 N/A다. Nonempty Past/B2 correctness는 주장하지 않는다. History append exactly-once, 후보/observer append0, post-observer restore와 final W4/M4/RNG/context/ledger CP3개(N4/legacy/reuse)의 존재·schema·SHA·parent/선택 결속을 검산한다. CPU reload를 GPU continuation PASS로 확대하지 않는다.

## 5. R512/generated256 전체 참여 확인

Input SHA507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb, R512=S64+Reserve320+train128과 별도 Dev128, 새 data ID/teacher manifest를 확인한다. W0 raw greedy/lowest-token-ID ties/no processor, configured EOS와 max256, prompt129, 실제 Ti, TF129+Ti−1와 score positions128..128+Ti−1를 대조한다. 전640 capsule 완결/partial 여부 및 TF argmax=generated IDs 확인 수준을 명시한다.

Ti 분포/EOS 종료/censor를 실제로 집계한다. 모든Ti256이면 EOS branch actual 검증을 주장하지 않는다. Old16/natural teacher 재명명·짧은 문서 제외·문서/token/vocab 축소·Dev 혼입 여부를 확인한다. 목적함수/gradient는 각 문서 실제Ti 평균→512문서 평균이며 모든512·전체 실제 위치·full vocabulary가 각 sweep에 참여했는지 actual coverage/counter로 검산한다. Resident chunk 수를 bank 크기로 쓰지 않는다.

## 6. 비용 및 효과의 분리

준비50410 allocation/program, generation/teacher/key/cache/I-O와 B150449 allocation/program을 별도 표기한다. 기존 native capsule의 fit 재사용·원비용·새fit0 여부를 실제 자료로 확인한다. 실패/중단 시도 비용이 있으면 원 lineage를 보존해 중복 없이 포함한다.

Matched schedule 각각의 objective/suffix/head/backward/teacher I-O, endpoint validation/H2D, Current/Past guard/invariant/cache hit/miss, geometry/hash/event/CP, observer를 분리한다. 중첩 timer는 합산하지 않고 미분리 값은 NOT_SEPARATED. 가능한 동일 경계의 correction wall 및 공통 setup 포함/제외 비교를 둘 다 제시한다. 4C→C 호출 감소를 전체4배속으로 표현하지 않는다. B1 1회로 안정적 p50/p90나 일반적 배수를 주장하지 않는다.

실제 peak memory/storage/teacher·key bytes를 계획 reserve111.8688GiB와 구분한다. Allocation은 GPU utilization이 아니다. 학습 KL 감소·같은 endpoint·속도 개선 각각의 근거와 미검증 범위를 분리한다.

## 7. 산출물·게시·종료

전용 review CPU reducer/tests만 `project/run_scripts/en_execution_reuse/` 아래 새 analysis 파일로 허용한다. Frozen runtime/원archive/raw/정본은 변경하지 않는다. 기존 승인 scope의 아직 미게시 구현은 실제 execution/analysis를 구분하여 own-scope source로 main에 통합할 수 있으나 review를 이유로 runtime 기능을 새로 수정하지 않는다.

허용 쓰기:
- `experiment-reports/servers/server4/en-execution-reuse-r512-g256-20260919-v1/completed-review-v1/**`
- `audits/servers/server4/en-execution-reuse-r512-g256-20260919-v1/completed-review-v1/**`
- 기존 task의 `plans/updates/server4/`, `tasks/status/`, `messages/acks/server4/`, `messages/server-heads/server4/` 전용 경로
- `runs/odeedit_en_reuse_r512_g256_b1_s4_20260919/completed-review-v1/**` compact-only 명시 허용
- ignored local task root의 새 `completed-review-v1/**` 및 전용 worktree

한국어 `diagnostic-report-ko.md`, 첫/최종 endpoint CSV, parity/coverage/source-conformance/trial/cost/CP inventory, analysis-manifest/rooted-receipt 및 재현명령을 남긴다. 숫자/표열/상대링크/렌더를 검사하고 그림은 코드로 생성·재현하며 실제 수행한 검사만 PASS로 표기한다. 별도 red를 쓰지 않았다면 독립 agent 감사라고 쓰지 않는다. Raw/tensor/teacher/prompt/fullstdout Git0. NO_BROADCAST_NOT_REQUIRED, 새로운 remote payload transfer0.

FULL_READ/M0→첫 실제 비교표→필요한 중요한 불일치→상세 완료 인계 순서로 보고한다. CPU 리뷰/own-scope nonforce main 통합까지 별도 단계 승인 없이 계속하되, 공유 dirty/타SH 변경 보존·공용 Git identity 변경0. SH는 사실·수치/기계적 계약 판정, GH는 별도 종합 해석을 맡는다. 완료 시 report/hash/source/main/cost/미측정 범위와 `TASK_COMPLETE_STOP`, monitoring_active=false, automatic_resume=false를 인계한다. 사용자 별도 허가 전 sequential은 계속 금지다.
