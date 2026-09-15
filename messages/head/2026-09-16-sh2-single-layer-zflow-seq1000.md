# GH → SH2: SL-ZFlow Llama 구현·기술 검증·W0 SEQ1000·사실 보고

instruction_id: ODEEDIT-S06-SINGLE-LAYER-ZFLOW-SEQ1000-SH2-V1
user_instruction_id: GH-SL-ZFLOW-IMPLEMENT-SEQ1000-20260916-V1
nonce: ODEEDIT-GH-SH2-SL-ZFLOW-SEQ1000-20260916-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a0493a-074c-7f91-9a13-769116326fef
target_server: server2
expected_cwd: /mnt/raid5/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 1. 최신 권한과 완료 경계

사용자는 명시적으로 SH2/server2, GPU cap2를 지정했다. 원 상세 지시문의 Server4 우선 검토 문구보다 이 지정이 우선한다. 이 문서와 함께 보낸 사용자 본문, 원 상세 지시문 및 reference를 전체 읽고 별도 task로 수행하라. 다른 기존 SH2/SH1/SH4 작업을 재개·변경하지 않는다.

구현, 필요한 CPU/실제 Llama 기술 검증, MAIN 한 W0/M0 B100×10, 조건 충족 시에만 N4 한 chain, 독립 결과 검산·한국어 사실 보고·own-scope main 통합까지 승인한다. 이번 task의 “끝까지 진행”은 기존 INITIAL_GATE_ONLY/PENDING 인계 후 정지 정책에 대한 task-specific override다. 실제 초기 gate 보고 뒤에도 본 task를 계속한다. 성능이 낮음, no-update, RESOURCE_STOP은 선별 gate가 아니다. GH의 중복 raw/GPU 점검이나 단계별 재승인을 선행조건으로 삼지 않는다.

과학적 모순·새 방법/범위 필요·해결 불가능한 자원 문제는 정확히 보고하되 독립적으로 가능한 본scope 구현·CPU 분석은 계속한다. 새 daemon/callback/다른 task 자동 재개는 만들지 않는다. 과거 EP-TW diagnostic waiver와 CAKE no-checkpoint waiver는 **이번 지시로 상속되지 않는다**. 이번에 사용자가 요구한 actual Llama parity·resume은 수행한다.

## 2. 원문과 CPU package 확보

GH가 아래 16개 원본 문서/source/test를 기존 dirty root에서 새 clean publication worktree로 byte-exact 복사했다. 원 source manifest의 12개 member SHA/size 대조 완료이며 실제 Llama 검증은 아니다. publication commit은 live 전달에 붙인다. 새 runtime이 이미 존재한다고 가정하지 말라.

- project/proposals/2026-09-16-single-layer-zflow-gh-instruction.md
- plans/global/2026-09-16-single-layer-zflow-pipeline-v1.md
- plans/global/2026-09-16-single-layer-zflow-contract-v1.json
- audits/global/2026-09-16-single-layer-zflow-design-review-ko.md
- audits/global/2026-09-16-single-layer-zflow-pipeline-checks.json
- audits/global/2026-09-16-single-layer-zflow-pipeline-cpu-demo.json
- project/run_scripts/single_layer_zflow/{README.md,__init__.py,demo.py,flow_core.py,oracle.py,transaction.py,tests/__init__.py,tests/test_flow_core.py,tests/test_oracle_transaction.py,tests/test_pipeline.py}

정확 SHA/bytes는 plans/global/2026-09-16-single-layer-zflow-sh2-dispatch/source-input-manifest.json. 사용자 현재 본문은 같은 디렉터리 authoritative-user-message.md. Reference status/llama_adapter_implemented=false/durable_commit_implemented=false 및 CPU30/demo 사실은 원문 그대로 보존한다. 새 source/import/config/실행 archive/lock에 actual 구현 identity를 별도 기록한다. Source SHA와 report·publication SHA를 혼동하지 않는다.

flow_core/oracle/transaction/demo는 GH가 직접 읽었다. CPU oracle의 full-logits callback과 in-memory receipt는 production selected-position full-vocabulary head/디스크 crash recovery의 구현 완료가 아니다. Demo.run_demo의 explicit off/null·on/positive-budget 검사가 lower-level integrate 호출에서 빠지지 않도록 production config 경로에 유지하라. 이 API 검사는 barrier arm 추가 권한이 아니다.

## 3. 구현 소유와 경로

SH2가 구현·native binding·기술 검증·source freeze·GPU 제출·운영·factual analysis·통합을 소유한다. 새 clean branch codex/server2-single-layer-zflow-seq1000-v1, 새 worktree 사용. 원 공유 dirty 및 다른 branch/출력은 보존한다. 필요한 복잡하고 독립적인 구현/검사만 제한된 blue/red 업무 분리 가능하며 단순 점검은 직접 수행한다. Worker는 원격 전달/commit/push 최종 소유자가 아니다.

허용 write:
- project/run_scripts/single_layer_zflow/ 하위 실제 native adapter/prefix-suffix/head/oracle/transaction/resume/runner/submission/instrumentation/analysis/tests/README.
- /mnt/raid5/janghj/ODE-edit/local/single-layer-zflow/20260916-v1/ 하위 authoritative/inputs/technical/main/conditional-n4/attempt-rN/raw/checkpoints/locks/receipts/logs.
- audits/servers/server2/2026-09-16-single-layer-zflow-seq1000/
- experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/
- messages/acks/server2/2026-09-16-single-layer-zflow-seq1000.md
- messages/server-heads/server2/2026-09-16-single-layer-zflow-seq1000.md
- plans/updates/server2/2026-09-16-single-layer-zflow-seq1000.md
- runs/odeedit_sl_zflow_seq1000_s2_v1/
- tasks/status/odeedit_sl_zflow_seq1000_s2_v1/server2.json
- transfers/verifications/2026-09-16-single-layer-zflow-server2-inputs/

원 BLUE/AlphaEdit native fitter 및 다른 experiment namespace는 read-only 재사용. 기존 NativeSingletonFitter.fit/native compute_z 완료를 MAIN 준비로 호출하지 않는다. 기존 finalize만 호출해서 exactly-once라고 하지 않는다. 공유 PROTOCOL/global plan/다른 서버 source는 수정하지 않는다. envelope 밖 source 변경이 꼭 필요하면 이유와 정확 경로를 먼저 보고한다.

## 4. 방법·수치 계약

물리 parameter=model.layers.4.mlp.down_proj.weight 하나. 매 batch 자기 entry W/M에서 X0=0, W(X)=W_entry+XB. N=P(KKᵀ+M_entry)+lambda_write I; B=E solve(N,PK)ᵀ. N은 비대칭 그대로; q=m=100/E=I는 actual native key binding을 확인한다. S=(B M_entry Bᵀ+lambda_write B Bᵀ)/m 및 spectral metric은 batch당 한 번 준비, 반복 factorization/기존 native endpoint normalization 없음.

Main lambda_write=1, lambda_flow=1(미튜닝 초기값), beta=.0625, barrier=off/budget=null, eta_initial=max=1, shrink=.5, clean2회 후growth1.5, max_oracle_calls=25. 나머지는 reference JSON exact 상속(relative damping.001, relative/absolute stationarity1e-5/1e-9, active1e-8, feasibility1e-10, complementarity1e-7, Armijo.0001, minimum_step1e-12). Toy default128/seed31을 실행 default로 가져오지 않는다. 수치 dtype/경로/tolerance는 실제 기술 측정 후 성능 확인 전에 봉인하며 실패 후보마다 확대하지 않는다.

F=L_edit+beta KL(p_current||p_entry)+lambda_flow C, C=tr(XSXᵀ)/2. Entry teacher는 native request-derived essence prompt/lookup으로 batch당 한번 고정; W0 C4 teacher나 외부 protection/PS/NS controller input으로 대체하지 않는다. Native key는 group 안 평균 후 group 간 평균(clean1/2, generated각1/10), edit loss는 6contexts각1/6 및 request/token 전역 weight1/(m*c_i*T_i)로 서로 구분.

모든 training token의 H_entry/A_p=BK_p, attention/position/label-shift/RoPE/native target-ID decode/join 검증. L5–31 suffix를 후보마다 fresh 계산; downstream KV 재사용0, suffix parameter densegrad0, X 입력 gradient 유지. 필요한 prediction/KL hidden만 gather 후 **full vocabulary** head. Raw K_p는 A 준비 후 해제 가능. Accepted L/g carry, rejected F+B도 기록. N_oracle=1+accepted+rejected; teacher/prefix/terminal parity는 별도.

작은-budget relative KKT·roundoff guard·step 회복 유지. FIRST_ORDER_STATIONARY/RESOURCE_STOP/NUMERICAL_STOP 구분, optimality/성과 인증 금지. Accepted0이면 W/M 불변 no-update receipt와 다음 batch index/평가 분모 보존; history append0. Accepted 있으면 마지막 상태만 actual FP32 W로 만들고 stored W.double()-entry W.double() cost·logits/NLL parity 확인. COMMIT_PARITY_FAIL/COMMIT_COST_FAIL은 entry 복원·해당 batch 기술중단이며 native fallback/요청 제외/no-update 결과로 덮지 않는다.

완전 checkpoint에 W/M/config/context/RNG/ledger/parent/batch_id/next index/cache-resume fingerprint 등을 함께 atomic publish. 준비 bundle+hash manifest 완료 전 resume 불가. 같은 commit 재시도 no-op, 다른 payload conflict. Native CPU FP32 K@Kᵀ 후 M_entry+Gram 순서, inner append0/확정 append1. 실제 resume/중단 경계와 next batch 연결을 검증하고 기술 실행과 main 비용·분모를 분리한다. Single GPU correctness 후 main은 fresh pretrained W0/coldM0에서 시작하며 기술 batch를 carry하지 않는다.

## 5. 데이터·runtime와 GH N4 재사용 판정 규칙

Server2 fixed dataset 우선: /mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json.
Dataset SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
whole ordered root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729,
first1000 root40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
공식 prefix loader로 B1[0:100]…B10[900:1000]를 model load 전에 검산·seal. 새 shuffle/filter/샘플 선택0; 1000 unique, scientific10batches. Conditional N4 포함시2000 arm-request이지만 unique1000이다.

기존 N4 우선 후보는 job38997 BLUE_L4_ONLY 또는 동일 first1000이 검증된 lifelong singleton의 B10이다. GH는 현재 Git publication 기준으로 N4 재사용을 **우선 승인**하며 최종 source/eval/state 입력 결속을 SH2에게 맡긴다. 정해진 판정 규칙을 아래에 승인했으므로 별도 재승인 대기 없이 manifest를 제출하고 진행한다.
- 기준 publication: experiment-reports/servers/server4/blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1/{blue-source-config-compatibility.csv,source-asset-inventory.csv,raw-member-inventory.csv,final_metrics.csv,factual-report-ko.md}.
- 기존 W0 Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, native BLUE311b076a92e4ed0f14f5c8b4909732da781bc5f7/helper1075540b45c29269e690ac63aae44758d8d63174, P physicalL4→stackindex0→local0. Seed20260907, contexts semanticSHAcf14b8571136b0413ded1ea8aa283a004ecbddbd818a6a8744b0296b63f72191/fileSHA33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e.
- N4 configSHA2392ab8392476ed019985e4292a8c0aa2a3957f36a06a704eb7590e8e8c99c5d, lock615e59ff8382113686da5359e56dc684112f2a70ab36913f0fe2c2ac6fb9e844. L4 observer index4는 기존 metadata오류이며 실제 P0와 구분.
- N4 기록 FP32/eager/right, TF32 matmulFalse/cudnnTrue, torch2.9.1+cu128/transformers4.44.2. 새 reference eager_tf32_off와 두 backend flag의 실제 적용·차이를 명시하고 임의 동등성 선언0. Model/tokenizer/context/order/평가/시작 state를 가능한 exact reuse; host변경만으로 bitwise parity는 주장하지 않는다.
- 기존 N4 W10 counts998/1000,1943/2000,8072/10000은 새 결과가 아니라 비교 후보 기존 수치다. 동일 case/prompt/target identity로 raw paired/NLL tail을 확인한다. Aggregate만으로 per-case pair를 추정하지 않는다.
- S2에 이전 이관된 B1/B5/B10는 transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv, /mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/blue-alphaedit-l4-oneshot-sequential/attempt-v1/main-llama/ 아래. 이 warm CP는 비교/소유 확인용이지 새 MAIN 시작점이 아니다. GH가 현재 원격 실물 재검산했다고 주장하지 않는다.
- 시작 W0/tokenizer/context/sample/runtime/evaluator의 비교 가능성이 확인되고 필요한 기존 metrics/raw가 확보되면 REUSE. 일부 기존 raw만 부족하면 exact endpoint가 있을 때 필요한 observation만 우선 보완하고 비용 별도 기록; 누락 CP 자체를 자동 full-chain rerun 사유로 하지 않는다.
- 필수 비교 자료나 동일 조건 연결을 실제로 재사용할 수 없는 경우에만 사유를 봉인하고 **조건부 N4 W0/M0 B100×10 한 chain**을 허용한다. 후자도 main과 같은 sample/runtime/evaluator/seed/context를 사용, N4 native source optimizer/early-stop/25steps/lr.1/decay.5/clamp.75/L2=1 유지. 기존보고 개선 여부로 재실행을 선택하지 않는다. 최대 신규 과학20batches 초과0.
- W50 REFIT4 suffix/EP RAW branch를 W0 N4 대조로 쓰지 않는다. 단일 새 method의 성과를 integrator 기여로 단정하지 않는다.

## 6. 자산 확보·자원·실제 실행

SH2 기존 model/P/native source/fixed10k/context/이관 아카이브부터 사용한다. Main에 새 CPU package가 없는 문제가 이번 GH source publication으로 해결되지만 서버2 실제 fetch/import fullSHA와 actual Python imports는 별도 확인한다. 원 BLUE/local helper가 없으면 지정 immutable source closure를 task-local inputs 아래로 확보하고 source 원본은 수정하지 않는다. Main용 pretrained W0를 old edited CP로 대체하지 않는다.

Peer에게 필요한 **완료된 본 비교 자료**의 exact path/identity 요청은 허용한다. Ordinary repo-local local/ 자산을 받을 필요가 있으면 source/receiver와 compact allowlist/SHA/size/destination을 먼저 기록하고 SH2를 sole receiving transfer owner로 하여 위 inputs/ 아래 non-overwrite staging→수신 검산→seal한다. --delete/기존 파일 overwrite/다른 task live output 접근/third-server broadcast/모델 중복 다운로드0. Repo 외부·민감 자산이거나 범위 불명확하면 승인 요청. 나머지 결과는 NO_BROADCAST_NOT_REQUIRED: server2 local 보존, raw-free 코드/보고 Git 공유.

Server2 cap=2는 기존 active+admitted pending 포함. cap이 비어 있다고 가정하지 말라. 기본 한 MAIN chain은1GPU/8CPU/60416M/exportNONE/Requeue0; baseline 조건 충족 시 총cap2 안에서 독립 병행 또는 pending dependency로 등록. 두 슬롯 채우려고 중복 arm을 만들지 않는다. 기존 다른 작업을 취소·hold·throttle·재개하지 않는다. Submission은 allowed: preflight source/config/data/import+actual correctness+resource 계획 뒤 held-inspect-release, jobname odeedit_sl_zflow_seq1000_s2 (조건부 N4는 별도 명칭).
GPU-hour hard cap=null. CPUtoy/native25시간으로 새 총비용 단정0. 실제 technical batch의 prefix/teacher/suffix F+B/peak/cost로 main 예상시간·wall/storage를 계산하고 제출전 lock한다. 기본 wall 상한48h 안에서 합리적으로 정하되 실제 제약상 불가능하면 자원 문제를 보고한다. 임의 성능·speedup 탈락선0. 안전한 microbatching/cache관리로 메모리 한도 준수, 새 과학조건 변경 금지.

기술 검증의 실행 수·시간은 별도 ledger, 재사용된 과거 비용은 신규 청구하지 않는다. 실제 numerical failure는 저장된 entry/trace/완전checkpoint에서 최소 수정 후 affected attempt만 재실행하고 원 실패 bytes/cost 보존. Main 정책 의미가 바뀌면 새 attempt/source로 명시, 다른 과학 arm이나 sweep는 자동 추가0.

## 7. 보고·감사·main 통합

CPU 회귀, actual full-write↔suffix logits/NLL/X-gradient/all-token/global weights/teacher, no-mutation/FP32cost, durable checkpoint/resume에 집중한 preflight와 postrun red 확인을 남긴다. Warn은 정해진 과학계약을 바꾸지 않는 명시된 미검증 설명과 함께 진행 가능, 실제 identity/parity/state 오류 block은 해결 후 진행한다. 숫자 낮음/효과작음은 block 아님. GH가 같은 raw/GPU를 중복 검사하지 않는다.

Node/terminal trace와 원문 필수 모든 관측을 실제 분모로 저장한다. 매 batch Current R/P/N(100/200/1000), R/P TFstrict/true-new NLL, W5 first500(500/1000/5000), W10 전체1000(1000/2000/10000)+같은 first500을 분리. No-update도 원분모 포함; atwrite→final/first500 retention/active-superseded/paired N4/NLL tail. C감소를 NS보장, RESOURCE_STOP을최적해로 쓰지 않는다.

Source/import/config/sample/model/P/context/teacher-entry identity, X/B/S/실제weight/history/resume inventory, accepted/rejected work 및 terminal cost/parity, commit/I-O/peak 분리. 기본 매 batch 일관된 checkpoint bundle과 W10L4/inference용base identity 보존. CAKE의 weight 저장 생략 적용0.

최종 experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/diagnostic-report-ko.md 및 node/batch/paired/compute CSV, manifest/rooted receipt/실행 명령을 작성하라. PNG가 필요하면 코드 생성만 사용하고 SHA/재현 명령과 표 렌더 검사를 남긴다. SH는 사실·산술·기술 RCA·미측정만 보고하며 효과·보존·비용/claim과 후속 Adam·barrier 비교의 과학적 해석은 GH가 별도 global review로 수행한다. 아직 실행하지 않은 Adam/barrier는 NOT_RUN이고 후속 실행0.

본scope source/tests/raw-free report를 latest main과 충돌 없는 clean integration에서 non-force push까지 승인한다. 기존 main/다른 scope 보존, source/header/table/manifest 일치 검사, conflicts는 destructive resolution0. Raw/teacher/model/checkpoint/fullstdout/prompt Git0. 모든 제출/완료 통지는 exact source/lock/job/출력/검증수준/남은 범위를 구분한다.

첫 M0에 source 확보·미구현목록·N4 reuse manifest 판단·10/조건부20batch·필요자산·cap/예상자원 및 실제source/output계획을 보고하고 끝까지 진행하라. GH는 이번 전달 단계에서 실험 완료를 기다리거나 중복 감사하지 않는다.
