# SH4 — low-cost 6-arm B100×10 sequential 비교 실행
Instruction ID: ODEEDIT-S06-LOWCOST-SIXARM-SEQUENTIAL-B100X10-SH4-V1
Nonce: ODEEDIT-GH-SH4-LOWCOST-SIXARM-SEQ10-20260913-R1

사용자 최신 지시: "server4의 실험에선, batch 1개로 진행해서 차이가 크게 보이지 않으니 batch 10개 sequential로 하는거 비교하는 것으로 실험올리자. task 전달하고 sample 순서도 잘 맞춰서 진행하라고 해"

## 1. 변경 범위와 책임
SH4/server4/session01a04939-b5c7-7a03-ba2d-ef3343d62cfd.
Expected CWD /data/janghj/ODE-edit, repo hyunjun1127/ODE-edit.
최신 사용자 요청에 따라 기존 core6(N4/S875/S75/FULL8/RES8/REFIT4)를 모두 같은 Middle entry에서 B51–B60까지 비교한다.
기존 "core후 최대1후보만 선정→audit→5batch suffix" 제한은 이번 고정6arm/10batch에 한해 override한다.
이는 전체60batch executions/6000 arm-request observations, unique suffix1000이다.
W0에서 새1000 sample을 고르거나 전체10k chain을6개 시작하는 실험이 아니다.
이번 확장은 차이가 반드시 커진다는 전제/우월성 판정이 아니며, 순차 누적 시 변화 측정이다.
소수 성능손실/비용 참고선 초과/유한 poor endpoint는 자동 탈락0. 모든6arm/원분모 보존.
Audit128/MMLU68는 그대로 미사용; 추가candidate선택/L5/alpha.5/다른entry/full10k/controller실험0.
GH 중복 raw/GPU 점검 대기0; SH4가 구현·검증·제출·초기gate를 맡는다. 복잡한 독립부분만 bounded subagents, 단순작업 직접.

## 2. 먼저 읽고 source 고정
원 GH-LOW-COST-WRITE-DONOR-PILOT-20260913-R2 및 design/cells/schema-v2, 기존 실행 envelope와 최신 PROTOCOL 전체 읽음.
실행 core7ece056c33fbb4246245c15f5f7c2a678315c05c/tree352cdd8ca3f3d7b6f2b15f3ed6a6f775fb478443.
완료 core보고서 experiment-reports/servers/server4/low-cost-write-donor-pilot-2026-09-13-v1/core-completed-review-v1/diagnostic-report-ko.md
SHA8ce7a2b0bbf6ae634f7f58b1035dcb79abbefc1700cb21c22fc5bf5f08c5988b.
baseline sourceBLUE311b076a92e4ed0f14f5c8b4909732da781bc5f7 / adapterb51dcf5 / schema58f50a closure 유지.
새 시작 main21d5240b7ab1497d1e3f0fb57ff23d03f4132ec8와 실제 runtime/source/tree/dirty/환경을 각각 기록한다.
기존 static runtime은 B051 current slicing/branch reset/hparams와 metadata를 hardcode하므로 단순 for-loop로 반복 호출하지 않는다.
기존 fitting.py의 fit/finalize 및 canonical evaluation은 재사용하고 신규 sequential orchestration에서 실제 batch index/entry/state/clock을 명시한다.
원본 BLUE repo·완료source/raw/report·SH1/SH2/중지 ORBODE 작업 변경0.

## 3. sample/order/entry lock — 반드시 동일
dataset /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json
SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1
orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
공식 load_prefix로 전체10000 verify 후, 이미 편집된prefix [0,5000)와 신규suffix [5000,6000)를 명시한다.
B051=[5000,5100), B052=[5100,5200), …, B060=[5900,6000).
0-based ordinal/1-based batch 혼동0, shuffle/resample/skip/replacement0.
모든6arm의 case_id/request/target/prompt/order hash를 batch별 봉인하고 동일 비교.
1000 신규 unique는 6개 arm 사이 공유이지 pooled unique6000이 아니다.
현재 available original L4 W50/M50와 core prepared.pt의 W4/W8/M4/M8/contexts/RNG를 CPU exact검산하여 재사용한다.
공통 L4-only We/W50, L8 pretrained W, originalM4와 공통We에서5000event로 재구성한M8가 출발점이다.
각 arm은 독립 process/model/module state에서 이 동일 entry로 시작. 전체model backbone/modelrevision/P/stats identity도 결속.
모델Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager, tokenizer/padding/TF32와 기존 canonicalMB16 동일.
Singleton BLUE L2=1 및 원래 z optimizer25loss/max24Adam·clamp·contexts 불변.
기존 W50 input은 S4 이미 selective수신한 copy 재사용; 불필요 checkpoint서버간 재전송/모델P복사0.

## 4. 매 batch 고정 정책을 반복한다
각 arm의 batch-entry Wb/Mb/RNG에서 fresh L4 native fit을 실행해 actual D4_b=WN4_b-Wentry4_b를 얻는다.
D4_b는 full native candidate의 실제 저장 FP32 weight 차이. 이전batchD4나 다른armD4를 쓰지 않는다.
first fit은 currenthistory를 append하지 않는다. native temporary weight를 해당 batch entry로 복원해 actual FP32 alpha materialization을 한다.
- N4: alpha1, secondfit 없음.
- S875: alpha.875, secondfit 없음.
- S75: alpha.75, secondfit 없음.
- FULL8: alpha1 뒤 현재 actual state에서 fresh L8 z/K/R/native solve.
- RES8: alpha.75 뒤 현재 actual state에서 fresh L8 z/K/R/native solve.
- REFIT4: alpha.75 뒤 현재 actual state에서 fresh L4 z/K/R/native solve.
alpha0/1 snapshot exact, 중간alpha는 기존 FP32 rounding 순서 유지. 품질에 따른alpha/endpoint 변경0.
두 fitting 사이 currentbatch history append0. 모든fit은 같은 batch-entry history를 사용한다.
최종 actual endpoint에서 선택layer별 해당batch history 정확히1:
N4/S875/S75/REFIT4는 M4 1회; FULL8/RES8는 M4/M8 각1회. REFIT4도 M4 두 번 아님.
P mapping L4 asset0/local0, L8 asset4/local0 불변.
M8 공통초기5000-event 준비는 검증된 prepared 재사용. 반복/alpha별5000재인코딩0.
donor arm M8는 이후 각자 최종endpoint keys를 FP32 append-only한다. scalar/refit arm의 미사용M8는 그대로 보관, donorhistory로 교환0.
다음 batch는 자기 전batch committed W/M/context/RNG에서 계속한다. 매batch 공통W50 reset/다른armendpoint import0.
평가/저장 시 임시복원하더라도 continuation state를 exact 되돌려야 한다. 모델객체/module globals mutable공유0.
한 번 보정 후 native정책으로 돌아가는 one-off intervention으로 바꾸지 않는다.
B51–B60 60 executions를 이번 fresh campaign에서 수행한다. 이전 static6 endpoint는 검증 reference로만 쓰며 부분 결과를 조용히 새chain에 이어 붙이지 않는다.
B51 independent execution은 source-equivalent 기존 core비교 가능. 새B51이 다르면 actual 차이를 기록하고 numerical근거 없이 성능 맞추기/tolerance tuning0.
기본 요청z=6×10×100+3×10×100=9000, fit/solve90 (실제reuse허용되지 않은 독립chain 기준).
검증된 common prepared reuse는 compute절약으로 별도 표시하고 과거setup 비용은 현재실행allocated에 중복가산하지 않는다.

## 5. 평가와 checkpoint를 프로그램에 포함
모든 batch actual committed endpoint에서 Current100 canonical R/P/N와 고정 Historical128 R/P/N, Wiki128, MMLUdev32를 동일하게 평가한다.
고정panel은 core의 identity를 재사용하며 성능을 보고 바꾸지 않는다. MMLU alternative integercorrect/invalid; audit68은 사용0.
Historical active/superseded는 해당시점까지의 동일 eventprefix로 annotate하되 ALL 원분모 유지.
매batch 새Current aggregate를 cumulative retention인 것처럼 합산하지 않는다.
각 arm B55/B60에서 그 시점까지 신규suffix seen500/1000을 actual W55/W60로 재평가한다.
B60은 전체seen6000 canonical R6000/P12000/N60000 평가, entry-old5000/new-suffix1000과 cohort/overwrite를 분리한다.
terminal suffix1000/Current100은 fullseen6000의 같은 identity rows로 재사용해 중복forward·분모 중복0.
B55도 Current포함 중복부분 재사용 가능하나 actual동일W/hash/평가계약을 검사한다.
W50 entry 및 필요한 W0 reference는 identity일치 기존 sealed평가를 먼저 재사용; 없다면 비교에 필요한 평가범위를 lock하고 비용분리. 전10k W0 새평가0.
true/new NLL, signeddesiredmargin, ties=failure, strict/token secondary, pairedlost/gained/atwrite→laterloss를 원case/prompt/target으로 남긴다.
futurebatch N이나 general/audit가 controller/온라인정책에 들어가지 않도록 분리한다.
미관측batch간 실패/회복시점은 추정0. Current/lifelong/fullseen분모를 분리한다.

B51/B55/B60 selected-state checkpoint 최소저장 (18개).
L4만바꾸는4arm은 selectedW4/M4+공통W8/M8provenance; donor2arm은 W4/W8/M4/M8 저장.
base model/source/P/stats/context/RNG/seenids/다음batchindex/실제historycounts/immutablepolicy/sampleorder도 결속.
추가 batch는 chronological actual low-rank increments 또는 필요한delta/journal+entry/commit hashes를 보존하되 exactreplay를 실검증하지 않으면 복원가능이라 주장0.
fullpretrained model 반복저장0. 저장공간 실제수량/bytes를 cap검사와 함께 추정하고 기존CP 삭제0.
최종fullseen evaluator가 n6000지원 및 schema normalize/actualendpoint guards를 사용하는지 CPUtest. 반환후reset모델을final로평가0.

## 6. 최소 검증과 초기 monitoring
기존 CPU/수학검증은 재사용하고 신규 sequentialstate/ordinals/history/평가복원/저장schema 연결만 집중검사한다.
필수 preflight: 동일6 sampleorder, preparedrestore/Pmap, batch1commit→batch2entry W/M exact, finalizationcounts, no branchleak, sameendpoint evalnonmutation, terminal분모/schema.
최소 actual GPU 초기gate는 첫 운영chain의 B51→B52 전달 및 realfit/materialization/history append/평가복원 finite검사.
가능하면 초기 cap2 slot은 N4와 RES8로 배정해 native와twofit 실제조건을 확인하되 별도대형smoke/반복FDgrid를 선행조건으로 만들지 않는다.
모든chain 각자의 상태검사는 제출된 프로그램 안에서 계속 수행한다. 모든arm gate나 전체B2를 agent가 기다리는 정책은 아니다.
전체6개 job/array를 upfront 봉인 제출하고 cap2 throttle/dependency를 걸어 자동스케줄되게 한다. 이후agent후속submit이 필요없도록 한다.
초기 actual gate 확인 후 MONITORING_PAUSED_AWAITING_USER; 프로그램은10batch+평가+저장을 계속 수행한다.
모두 PENDING이면 현재 사용자정책대로 정확한 heldinspection/release/순서와 dependency를 기록하고 PENDING_GATE_NOT_RUN으로 바로 pause 가능.
그 뒤 시작/firstbatch/terminal polling/heartbeat/callback/자동analysis/main integration0. 사용자recall만 재개.
NaN/Inf/잘못된 W/M/order 적용은 typedtechnicalfailure와 완성prefix/cost보존; 유한 nearstall/성능저하로 멈춤/rescue/replacement0.
Technical retry는 해당범위만 새attempt이며 science변경/오래된attemptoverwrite0.

## 7. 운영 envelope
전용codex/server4-lowcost-sixarm-seq10-v1 / 새clean worktree/source/process/output.
허용 source write: project/run_scripts/low_cost_write_donor_pilot/의 sequential runner/config/launch/tests/analysis.
project/run_scripts/baseline_mechanism_first/는 필수 evaluator/schema adapter만 전용source에서 변경, 원native math/SH1실행경로 변경0.
Raw /data/janghj/ODE-edit/local/low-cost-write-donor-seq10/20260913-v1/<attempt>/ create-once.
SLURM ALLOWED:6 independent chains B100×10, array%2 또는 동등cap-safe queue.
Jobname odeedit_lowcost_seq10_s4 (arm/runid metadata); server4 1GPU/8CPU/60416M per process/exportNONE.
fresh admission에 기존projectactive+admittedpending까지 계산, aggregatecap2; unrelatedjobs변경0.
GPU-hour budget=null, walltime은 실제측정기반 추정+합리적여유로lock (기존12h를자동hardbudget으로상속0).
한-batch fit timing만10배하여 total이라하지 않는다.6chain独립firstfit비용/sharedM8setup/매batcheval/fullseen/CP/I-O를 각각예상·실측.
CPU/source/config/asset/session/red preflight/memory audit 후held→inspect→release. submit권한추가승인대기0.
후속보고/codePNG는 직접코드생성;rawfreeGit/source+한국어report완료후recall시본scope nonforce mainintegration.
현재초기gate뒤자동mainpush0. NO_BROADCAST_NOT_REQUIRED: local재사용/보존+compactprovenance; 새원격raw전송은별도 exactscope요청.
서버 소유 경로:
messages/acks/server4/2026-09-13-lowcost-sixarm-seq10.md
messages/server-heads/server4/2026-09-13-lowcost-sixarm-seq10.md
tasks/status/server4/2026-09-13-lowcost-sixarm-seq10.json
runs/lowcost-sixarm-seq10-s4-20260913-v1/
audits/servers/server4/2026-09-13-lowcost-sixarm-seq10/
experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/
factual report에는batchcurve/current-suffix-fullseen/pairedtransitions/statehistory/cost/실패미측정을전부포함.
GH해석/claim선택은별도global. 이번10batch끝에서FULL10k안정성/다른entry효과를주장하지않는다.
최초ACK는 FULL_READ/source/commonentry/sampleB51..60hash/6arm등록/자원·저장예상, 이후6jobid/mapping/cap-safequeue/initialgate또는pending상태를보고하라.
