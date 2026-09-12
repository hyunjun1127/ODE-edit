# GH → SH1: Baseline mechanism-first E0/E1 구현·실행·사실 보고

instruction_id: ODEEDIT-S06-BASELINE-MECHANISM-FIRST-E01-SH1-V1
nonce: ODEEDIT-GH-SH1-BASELINE-MECHANISM-E01-20260912-R1
task_id: baseline-mechanism-first-e01-sh1-20260912-v1
from: GH / 01a04939-8873-7673-8dca-4c7fc5e31af0
to: SH1 / 01a04939-f93a-7b50-bca0-65438eab2062
server/CWD/repository: server1(devbox) / /mnt/raid5/janghj/ODE-edit / hyunjun1127/ODE-edit
유형: 신규 독립 baseline 진단. 기존 multilayer A/AOS task의 recall/변경/취소가 아니다.

## 0. 기준과 우선순위

사용자의 2026-09-12 최신 지시: 지정 설계문을 전체 읽고 기존 근거 재사용, E0, E1-A, E1-B의 구현·CPU 분석·native GPU replay·복원 검증까지 실행한다. SH는 조건/사실/수치/artifact/기술 실패를 기록하고 GH가 최종 원인 분석을 종합한다. 완료 baseline을 다시 10k 훈련하거나 Barrier/ODE의 필요성을 전제하지 않는다.
기준 원문:
project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md
GH 확인 원본 /mnt/raid5/janghj/ODE-edit/ 아래 같은 경로, 582 LF/53160 bytes, SHA256 e4dc0b1fc0666775deb6e43cbdf18269c8e8e6d41d70c192e2bd4a2661a6ccbe.
이 commit의 설계 사본은 원본 bytes exact다. 사용자의 좁힌 실행범위가 설계 §1의 E0–E2 표현과 §9–12의 후속 실행 제안보다 우선한다. E2–E5는 읽되 실행하지 않는다.

GH는 설계 전체와 §3/6/15 reference 파일 존재·소형 SHA, R2 보고, native AlphaEdit target/solve/history 경로 및 일부 primitive 경계를 확인했다. 이는 전체 source audit나 checkpoint GPU fidelity 완료가 아니다. audits/servers/server1/2026-09-12-baseline-mechanism-first-e01/gh-reference-availability.json의 FILE_BYTES_AVAILABLE 상태를 FULL_READ/실험완료와 혼동하지 않는다.
SH1은 실제 사용 코드/설계/참조를 직접 읽고 정확한 경로/commit/blob/SHA를 재결속한다. 새 source와 기존 executed-source/report provenance를 구분하며 여러 commit의 runtime 조각을 임의 혼합하지 않는다. Mutable root의 미커밋 설계를 몰래 수정하지 않는다.

## 1. 책임 분배와 독립 작업

SH1은 이 task 전체 구현/자원/원시계측/수치 검증/사실 보고 owner다. GH 중복 raw/GPU 감사나 단계별 재승인을 기다리지 않는다.
SH1 내부에서 다음처럼 나누어 병렬 작업하라. Worker는 혼자 작업하는 것이 아니며 다른 사람의 변경을 되돌리지 않는다. 부모 SH1만 source 통합/submit/push한다.
- evidence/CPU owner: evidence-reuse-manifest, E1-A case/prompt join, overwrite/age/margin matching, compact 수치표. 소유 modules evidence.py, case_analysis.py와 해당 tests.
- fixture/native owner: E0 W0/CP restore, mapping/source-exact native continuation, input/runtime/restore locks. 소유 modules fixtures.py, native_runner.py와 해당 tests.
- instrumentation owner: E1-B target/write/key/all-position response/geometry/signed derivative 및 비용. 소유 modules instrumentation.py, geometry.py, signed_response.py와 해당 tests.
- SH1 통합 owner: contracts/runner/ledger/CLI, source pin, 공통 source/test 검증, scheduler. 공통 fixture API 먼저 봉인.
- red reviewer: 실행 코드를 고치지 않고 source-equation/mapping/observer noninterference/정보 분리/분모/typed status/경로/Git 사실 검사.

자원이 제한되면 SH1이 역할을 순차 겸임할 수 있으나 서로 다른 worker가 같은 module을 수정하지 않는다. 전체 구조 구현이 선행 GPU 조건이 되지는 않으며 E0와 필요한 최소 계측으로 진입한다.
GH는 SH1의 최종 사실 package를 받아 새 관측과 기존 baseline을 종합하고, 대안 설명과 다음 최소 개입의 우선순위를 별도 global 보고서에 작성한다. SH에게 과학적 결론/새 방법 선택을 떠넘기지 않는다.
SH2는 checkpoint 보존 owner, SH4는 원시 평가/source 보존 owner로 경로/봉인 자료 지원만 직접 요청 가능하다. 두 서버의 GPU 실행·paused task·진행 중 downstream 보고를 넘겨받거나 재개하지 않는다.

## 2. 범위 고정

승인: 기존 evidence 재사용 + E0 + E1-A + E1-B.
E1-B canonical20 cells = AlphaEdit BLUE singleton layers {4,5,6,7,8} × entry n={0,1000,5000,9000}, 각 다음 B100 한 번.
이는 2000 cell-request executions이며 unique2000이 아니다. 동일한 네 B100을 5 layer에 사용한다. 실제 unique count는 fixed sample에서 확인한다.
E0 source-exact native continuation은 fidelity 비교 목적이다. warm15 branch에서 필요시 다음 저장 endpoint까지 n1000→2000,5000→6000,9000→10000의 각10 B100을 진행한다. 첫 B100은 E1-B와 동일 identity이면 재사용한다. 15 windows를 전부 진행하면 warm150 batches + cold5 B100=155 native batch executions이며 E1의20cells와 더해 중복계수하지 않는다. 필요한 continuation 범위와 원 checkpoint 대조 가능성을 먼저 run-index에 기록한다.
새 W0→10k chain5개, W0→n 전체 replay 자동확대, candidate suffix/full10k, MEMIT 신규 native GPU20cell 복제는 승인하지 않는다. MEMIT singleton은 E1-A 기존 raw 분석에 포함한다.

금지:
- 성능용 update α sweep/shrink 및 α 후보 선택
- P rank/ridge 변경, history rotation/M0 대체
- compute-z budget/clamp/target/loss/context 변경
- 위치별 partial activation intervention
- M2/M3/M4, Barrier/ODE/target-realization controller
- 추가 editable layer 보정, 신규 방법 continuation, E2 이후 실행
허용 α는 E1 signed derivative 검사용 작은 finite-difference probe뿐이다. 반복 forward 해상도에서 공통 probe를 사전 고정하고 성능 선택에 쓰지 않는다. Native α=1 endpoint는 원 write로 한 번만 관측하며 derivative probe와 performance sweep을 구분한다.

## 3. Evidence reuse와 입력 보존

먼저 evidence-reuse-manifest.json을 작성한다. 각 항목을 별도 상태로 둔다:
1. 완료된 기존 관측/집계
2. 실제 접근 가능한 raw/CP/source
3. 기존 per-case로 새 계산할 분석
4. 신규 GPU 계측 필요
5. 접근불가/NOT_RECORDED/아직 미검증
기존 final metric/at-write loss/cohort/paired NLL/overwrite 전수 결과를 동기 재감사 명목으로 다시 만들지 않는다. Source/index membership/필요 row join만 새 분석에 결속하고 기존 검산 receipt를 재사용한다. Aggregate에서 per-case joint distribution을 만들지 않는다.

고정 데이터:
 /mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json
dataset SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1
ordered root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
기존 load_prefix 및 모델 전 verify 정책 유지; 새 selection/order/실패 case 교체0.

183CP는 이미 server2로 이동했고 server4 원 CP는 삭제됐다. 원래 S4 경로를 CP 가용성으로 쓰지 않는다.
- S2 catalog: /mnt/raid5/janghj/ODE-edit/local/checkpoint-migration-server4/20260911-v1/
- S2 archive: /mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/
- 기존72CP: /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/
- Git mapping: transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv
Original source/contexts/native targets/current.json/seen-full.json 등 companion은 S4에 보존된 것과 GH local review에 이미 수신한 것을 먼저 inventory로 확인한다. SH1 자신의 ABC prepared는 BLUE-L4 source/entry/target identity가 정확히 같은 부분만 재사용한다. L5–L8을 L4 prepared로 대체하지 않는다.
선택전 검증/수신 record는 CPU hash/schema이며 GPU continuation PASS가 아니다.

이번 task에 필요한 파일의 create-once selective rsync를 승인한다:
- S2 위 catalog가 지정한 Alpha singleton L4–L8의 entry 및 E0 다음 stored endpoint CP, 필요한 B1 reference만.
- S4 완료 singleton Alpha/MEMIT의 current/seen-full 및 metadata/contexts/target/source companions 중 E0/E1에 필요한 allowlist만.
- 새 destination은 SH1 새 local/baseline-mechanism-first-e01/20260912-v1/imports/<source-id>/.
SH1을 단일 destination writer로 하여 source host/실제path/bytes/SHA/공간을 먼저 manifest로 봉인하고 SH2/SH4와 ownership 확인한다. Source 읽기전용, --delete/overwrite/source삭제0, 전체183CP/전체repo/fullmodel 무차별전송0. 기존 S1 exact 사본은 재사용한다. 비파괴 custom landing은 이 task에 한정된 broadcast-helper 경로 예외이며 transfers/verifications에 남긴다. 실제 파일 범위가 달라지거나 공간/identity 문제가 있으면 그 부분 HOLD, 가능한 CPU 구현은 계속한다. 새 raw 전체의 타서버 복제는 NO_BROADCAST_NOT_REQUIRED로 남기고 compact 결과만 공유한다.

## 4. E0: 실제 출발점·native 재개

먼저 L4/L8 공통 loader와 원래 tensor의 stack→local mapping 확인, 이어 L5/L6/L7 mapping 확인. Physical L4..8=P asset0..4, singleton local0를 tensor hash로 명시한다. 레이어 변경은 한 weight의 원본 singleton 설정이며 5-layer/MEMIT/JV cache를 섞지 않는다.
Llama revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32 parameter/기존 예외/eager/tokenizer/padding/position/microbatch/contexts/RNG/native hparams 그대로. Alpha BLUE singleton L2=1; base 5layer Alpha L2=10과 혼동하지 않는다.

W0 cold path와 저장 checkpoint restore를 구분하고 selected/nonselected weights, M, P/C0, module globals, contexts, Python/NumPy/Torch/CUDA RNG, next batch index를 bind한다. W0에 저장 contexts를 주입하면 원 context 생성의 RNG 소비 상태도 대응시킨다.
No-op restore bytes/history, hook/cache cleanup, panel repeated forward 수치 envelope를 먼저 기록한다. Solver 구현 오차를 forward noise로 합리화하거나 성능에 맞춰 tolerance를 완화하지 않는다.
각 다음 native B100의 원 target/order/context/hash/actual update/history/metric을 비교한다. B11/B51/B91 fullW 미저장 한계를 명시하고 필요시 다음 저장 W/M endpoint까지 native continuation하여 실제 tensor 차이를 확인한다. 기존 exact target cache 재사용 검증과 compute-z recomputation 검증을 별도 표기한다.
Tensor/hash mismatch는 위치/범위/원인을 기록한다. 불확실 branch를 원 trajectory와 동등한 것으로 비교하거나 인접 CP/재생성 context로 조용히 대체하지 않는다. Technical 원인 수정은 새 attempt/source와 영향범위만, 이전 실패bytes/cost 보존. 독립 E1-A는 계속한다.

## 5. E1-A: case별 관찰 분석

AlphaEdit 및 MEMIT singleton L4..L8의 current.json과 seen-full.json을 case_id/prompt_index/prompt+target identity로 join한다. Raw margin은 원부호 그대로 유지하고 desired-margin derived column을 추가한다.
Relation/subject/request order/at-write margin/final outcome/age/token length/overwrite version/active 상태를 붙인다. 전체 population 주표를 유지하고 common-at-write-success subset, common-support margin-matched/age/relation/token-length strata를 추가한다.
Matching/bin/지원구간 정의와 제외율·분모·빈 bin·subset size를 기록한다. At-write margin은 layer 선택 이후 변수이며 matching이 gap의 인과 기여율이 아니다. SH 수치표는 산술 차이만, 원인 해석은 GH 소유.
Observed failure checkpoint와 interval-censored 구간을 구분한다. 저장되지 않은 중간 실패/회복을 추정하지 않는다. 정당 overwrite와 실제 active history loss는 별도분모; 주 원분모 유지.
최종 표는 MEMIT/AlphaEdit family별로 정리하고 기존 BLUE(L4+L8)/원본5layer/W0 reference를 별도 source/config 조건과 함께 연결한다. 이번 신규 GPU는 singleton Alpha20cell뿐이다.

## 6. E1-B: 패널·native 계측 계약

같은 entry의 layer 비교는 동일 current100/historical128/general128을 사용한다. Historical: seen ordinal early/middle/recent 세구간43/43/42, outcome-independent fixed hash로 선정, active/superseded 분리. n0 historical 없음. General128은 설계가 허용한 기존 general text 자산/절단·tokenization·seed를 사전봉인; 공식 평가 P/N에서 선별하지 않는다. 외부 데이터 다운로드/새 보호 bank objective는 추가하지 않는다. 가용 general corpus 출처·선정이 정의되지 않으면 소스 기반 판단을 기록하고 과학적 변경이 필요할 때만 질의한다.
n>0 current+history 규격은 2964 prompt pairs/5928 true-new sequences + general별도이며 실제 prompt cardinality를 manifest로 확인한다. 동일 W0/entry/native endpoint 평가 identity가 같으면 재사용한다.

Backward 우선 4cells: L4/L8 × n5000/9000. 설계 예시 subpanel CurrentR32/HistoricalR32/HistoricalP32/CurrentN32/HistoricalN32/General16을 동일 hash로 봉인한다. R/P/N scalar와 General NLL은 별도. 실제 비용·신호에 따른 범위내 추가 backward는 선택 근거/coverage를 사전에 append하여 기록하되20cells 밖으로 확대0. Dense Jacobian/전체dense gradient의 일괄저장0.
- target: compute-z 최종 trainingNLL/실제iteration/stopreason/clamp, 없다면 NOT_OBSERVED. observer 출력 수집이 optimizer/context/RNG를 바꾸지 않는지 검사.
- write: 실제 native K/R columns와 context 평균/반복/위치, actual materialized ΔW, norm/relative norm, ΔWK−R, post-write activation.
- exposure: raw overlap/Fᵀq/writer-weighted overlap/actual ΔWq, true/new 전체 teacher-forced 위치. subject/다른prefix/continuation을 tagging하되 위치 intervention하지 않음.
- geometry: P symmetry/idempotence/rank, C0/projected C_U spectrum, Pk 비율, projected history/effective rank/overlap, native nonsymmetric system 및 reduced symmetric diagnostic conditioning/residual, 누적 S와 이번Δ의 covariance leakage/signed inner product.
- output: signed <grad(desired margin), actualΔ>, 작은α FD, native α1 실제 margin/NLL. R/P margin=NLLtrue−NLLnew, N margin=NLLnew−NLLtrue, tiefailure. General은 별도NLL.
공식 평가 P(paraphrase)/N(neighborhood) 및 historical/general 관측값은 진단에만 사용하고 write/target/step/후보선택에 주입0. Native projector P와 M/C0는 원 writer의 입력으로 그대로 사용한다. 고정 sequence/singleton이면 해당 module input key가 어느layer든 자기write로 바뀌지 않는다는 조건을 확인한다. “L4만 key안정/L8-only keydrift” 가정을 쓰지 않는다.
Exact spectrum 비용이 크면 선언한 수치법/rank/residual로 approximate를 명시하고 exact rank/condition으로 표기하지 않는다. Native P/M를 diagnostic용으로 대칭화한 tensor로 writer에 되돌리지 않는다.

## 7. primitive 재사용·최소 correctness

C1 alpha_backend는 B10/BF16 주변 wrapper/상수까지 상속하지 않는다. FP32 native solve·dtype/residual primitive만 B100 actual geometry에 맞게 별도 adapter한다.
C2 low-rank factor와 source의 dense-RHS solve는 대수 동치여도 materialized FP32 동일하지 않을 수 있다. Native dense update가 기준, factor/native 차이 및 derivative-direction 차이 기록. Official 경로를 factor로 교체0.
C3/C4 capture/snapshot/rollback/selected+RNG 검사 재사용, M/P/C0/context/next index 연결 추가.
C5 event_derivative를 저장하고 max(0,−derivative) progress_slope로 대체0. Full true/new 부호와 모든 관련 token mask 유지.
C6 covariance math만 재사용; routing utility/smooth event/stopping계승0.
C7 leakage0/zero response는 finite 결과이지 임의오류 아님.
C8 symmetric_history 처리와 native nonsymmetric 순서 차이를 기록; package의 argmax locality evaluator/barrier승계0.
C9 BLUE true/new NLL bit·margin·집계 원규칙 유지.

최소 CPU: layer/transpose/stack mapping; desiredmargin부호; native dense-v-factor 및 contraction oracle; selected/M/context/RNG restore; identity join/overwrite/분모; observer의 writer 입력 불변. Existing tests 우선, 대형 반복 audit0.
GPU: E0 actualweight/history/forward와 representative signed finite difference validation. NaN/Inf/wrongwrite/수식오류는 technical HOLD/수정. Finite 낮은성능/약한효과/zero leakage/큰condition/해상도미충족을 scientific 성공으로 미화하거나 성능 gate로 sample교체하지 않음. Numerical unresolved는 해당 derivative 주장만 제한하고 독립 자료 보존.

## 8. 실행·감시·예산

Slurm submission ALLOWED: 위 E0/native20cells와 해당 forward/backward만. Fresh session/CWD/repo/preflight/source/data/output/cap checks 후 held→inspect→release. server1 project GPUcap2, 기본1GPU/process, mem182272M/GPU 상한,CPU/Slurmwall은 실측계획으로 고정. 기존 AOS45633 등 프로젝트 active/예약capacity를 포함하여 recount; 종료/회수/동시탑재/변경0. 자원없으면 WAITING_FOR_ISOLATED_RESOURCE, CPU독립작업 계속.
GPU-hour cap은 사용자미배정(null). 기존 S/D48GPUh·옛1.42h·다른task 예산을 상속하지 않는다. 소형 native/계측 actual load/restore,z,key,solve,spectrum,F/B,eval,I/O와메모리/저장량을 먼저 보고하고 remaining estimate를 분리한다. Scope 실행 권한을 budgetnull만으로 무기한 대기시키지 않되 신규유료/자원확대0; 실제 자원불가면 보고.
E0 continuation/FD/technical attempts도 비용에 포함, 재사용20cell 중복비용0. Edit total과 subcomponent중복합산0.

기존 사용자 INITIAL_GATE_ONLY 정책 유지: 준비·CPU·필요한 resource/수신검증·실제 초기 gate까지 진행하고, 최초 actual E0/계측 검증 후 MONITORING_PAUSED_AWAITING_USER. INITIAL_VALID를 E0 전체continuation/20cell완료라 부르지 않는다. 이미 승인·봉인·제출한 program은 계속; agent polling/heartbeat/terminalwait/후속submit/최종분석/main통합은 사용자 recall 전0. 이 task에는 자동완료 감시/자동wake를 만들지 않는다.
최종까지의 scope 승인은 유지되며 recall 시 반복적인 과학승인을 요구하지 않는다. 운영 pause와 science/resource blocker를 구분한다. 기존 A/B와 다른실험의 paused 상태는 이번 명령으로 해제하지 않는다.

## 9. 저장·보고·종료

권장 새 branch codex/server1-baseline-mechanism-first-e01-v1와 고유 worktree.
SH1 승인 write:
- project/run_scripts/baseline_mechanism_first/ (새 adapter/CPU분석/계측/tests/plot; 기존baseline source불변)
- local/baseline-mechanism-first-e01/20260912-v1/ (고유 attempt/inputs/raw/private rows)
- audits/servers/server1/2026-09-12-baseline-mechanism-first-e01/
- experiment-reports/servers/server1/baseline-mechanism-first-e01-2026-09-12-v1/
- messages/server-heads/server1/2026-09-12-baseline-mechanism-first-e01.md
- messages/acks/server1/2026-09-12-baseline-mechanism-first-e01.md
- tasks/status/baseline-mechanism-first-e01-sh1-20260912-v1/server1.json
- runs/baseline-mechanism-first-e01-sh1-20260912-v1/
- plans/updates/server1/2026-09-12-baseline-mechanism-first-e01.md
- transfers/verifications/2026-09-12-baseline-mechanism-first-e01/
Envelope 밖 primitive 수정이 필요하면 범위와이유 보고, 원본실험source수정으로 우회0.

필수: source/runtime/science/resource/sample-panel locks, evidence-reuse-manifest.json, run-index.csv, resume_fidelity.json, margin_age_matched_retention.csv, paired_case_ledger_manifest.json, native_target_write.csv, writer_interference.csv, projected_geometry.csv, signed_response.csv, instrumentation_cost.csv, failure/coverage registry, reconstructable selected snapshots 및 source/asset/raw manifests.
대형raw/CP/keys/gradient/prompts는 local-only, Git은 code/tests/compact표/manifest/checksum/사실보고서. PNG는 직접 code생성/재현, imagegen/visualize0.
SH 사실 보고서 factual-report-ko.md: 재사용/새계측 구분,E0fidelity와 미검증,E1-A paired/matched수치,E1-B target→write→token→output수치,실측비용/coverage/typedfailures. 추정원인/우열권고는 GH로넘긴다.
전용branch sourcepush와 최종완료 ownscope non-force mainpush 승인; pause동안 자동main금지. Unrelated A/B/dirtyuserdiff를 함께 merge0. Red preflight/postrun/내용분모/기술fidelity/정보분리/Gitrawfree검사, warn기록후진행가능/block부분중지. Final factual package rehash/보고서경로SHA/source분리/cost/미측정목록을 GH에게 전달.

GH 소유 최종 종합:
experiment-reports/global/baseline-mechanism-first-e01-2026-09-12-v1/diagnostic-report-ko.md
질문별 관측과 일치하는 설명·배제못한대안·evidence수준·E2이후 최소개입 우선순위(제안만)를 정리한다. 아직 없는 E0/E1 결과를 기존 baseline 수치로 채우지 않는다.
E0/E1 package + GH 종합보고 제출 시 이번 task 완료. E2–E5 자동시작0.
