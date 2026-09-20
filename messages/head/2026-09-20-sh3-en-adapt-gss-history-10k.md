# GH → SH3: EN_ADAPT GSS history v1 / R2 / fixed10k

> Latest user override: [RES/GSS_REC only, cap2](2026-09-20-sh3-en-adapt-gss-history-twoarm-cap2.md).
> Below three-arm/cap1/shared-prefix planning is historical where superseded.

Instruction ID: ODEEDIT-S06-EN-ADAPT-GSS-HISTORY-10K-SH3-V1
Nonce: ODEEDIT-GH-SH3-EN-ADAPT-GSS-HISTORY-R2-20260920-R1
Owner: head-server3 / ubuntu / session01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3
CWD /data/janghj/ODE-edit; origin hyunjun1127/ODE-edit.

## 1. 최신 권한 / 이전 lifelong 요청 대체

사용자가 plans/global/2026-09-20-en-adapt-gss-history-design-v1.md를 지정해 SH3 task 전달을 요청했다.
**R2_B2_CHECK_THEN_DIRECT_LIFELONG** 설계대로 실제 구현·봉인·실행 제출을 승인한다.
문서 첫머리의 GPU미구현/미제출은 설계작성 당시 상태이며 이번 구현 권한을 막는 문구가 아니다.
기존 단일 EN_ADAPT all-active history 10k 준비 요청은 이번 **세 arm bounded bank**로 대체한다.
이전 all-active 별도10k를 추가 제출하지 않는다. 이미 제출했다면 exact mapping을 먼저 보고하고,
다른job/51290을 임의cancel하지 않는다. 완료 B300은 증거/reference일 뿐 새10k의 state가 아니다.

사전범위는 B1–B2 실제연결 확인뿐이며 **같은 B2 state에서 세 arm 모두 B3–B100을 연속 실행**한다.
별도 B300/B1000 pilot·B7 gate·성능에 따른 arm/기간선별·추가승인 대기0.
다만 직전 사용자 명시 **checkpoint 미저장**과 **제출 뒤 모니터링 중단**은 유지한다.
설계 §9의 checkpoint 보존 지시는 이번 실행에서 override한다:
save_checkpoints=false, edited W/M/optimizer/delta/동등resume bundle disk0.
B2→B100 및 공유prefix 분기는 **동일 persistent process RAM 상태**로 이어간다.
관측시점의 metrics/ledger/teacher/cost는 저장하되 checkpoint라고 위장하지 않는다.
Exact crash-resume=NOT_AVAILABLE. B2를 별도job으로 끝내 상태를 잃은 뒤 다시 시작하지 않는다.

## 2. 정본 / 기존 source를 재사용

다음 6개 FULL_READ 및 authority-manifest SHA/size 일치:
- plans/global/2026-09-20-en-adapt-gss-history-design-v1.md
- plans/global/2026-09-20-en-adapt-gss-history-contract-v1.json
- plans/global/2026-09-20-en-adapt-gss-history-cells-v1.csv (원 CRLF bytes 보존)
- audits/global/2026-09-20-en-adapt-gss-history-design-v1/history_reference.py
- audits/global/2026-09-20-en-adapt-gss-history-design-v1/verify_design.py
- audits/global/2026-09-20-en-adapt-gss-history-design-v1/design-checks.json
부모 EN adaptive-nullspace design/contract/selector, 최근 B300 completion report도 읽거나
정확 기존 FULL_READ receipt로 결속한다. 원 CPU45checks는 neural/sketch fidelity PASS가 아니다.
verify_design.py는 cells와 receipt를 덮어쓰므로 원 정본에서 재실행하지 말고 scratch로 분리한다.

부모 실제 execution5d452221288f3b924e1737578f11aaa654594422와
완료 분석 mainabd08ee1dde24ca975aace49c925a4ba00211004를 구분한다.
NumPy scalar JSON/atomic create-once 수리, ideal-ray curvature/actual Armijo,
fixed-order loader/expanded metrics/optimized z hook를 재사용한다.
새 history/sketch/selection/runtime는 project/run_scripts/en_adapt_gss_history/에 구현하거나
명시적 새config branch로 격리하여 기존 frozen runtime를 바꾸지 않는다.
기존 공유env/EasyEdit/user dirty/raw/teacher/model 유지.

S3 READY model/tokenizer/native/P/C0/context/runtime를 그대로 결속:
Llama-3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2,
FP32/eager/TF32off/L4 downproj/nativeL2=1 및 부모target/clamp/Adam/종료정책.
z-hook batching/prefix cache/full-vocabulary selected-position head와 요청별 종료고정 유지;
추가 z최적화/precision변경/gradclipping0. 실제S2사용provenance미확인을 소급PASS로 쓰지 않는다.

기존 fixed10k 동일 records[0:10000], B100×100, 모든 arm 같은 순서.
datasetSHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
S3 local R512/Dev128 2560files/101519959223B fullverified 입력을 REUSE.
새수신·재생성은 없거나 손상된 정확 구성원만 기존선택전송승인으로, sourceKEEP.

## 3. 세 arm과 history identity

승인 arm만:
EN_ADAPT_H_RES = whole active-ledger bottom-hash512 / uniform KL;
EN_ADAPT_H_GSS = fresh signed-gradient GSS backward pruning512 / uniform KL;
EN_ADAPT_H_GSS_REC = 동일 GSS / bounded recency KL (primary).
새 N4/R-only/all-active arm 제출0, 기존baseline은 비교조건 확인 후 reference로만 사용.

Native M / 전체 semantic ledger / steady replay H≤512 / pending A≤100를 분리한다.
GSS hot pool≤612; eviction은 nativeM·ledger·cold teacher 삭제가 아니다.
RES는 전체active versions 중 bottomhash512를 보충하고,
GSS는 evicted fact 자동resurrection하지 않는다(같은target 재등장의 재제안은 계약대로).
subject/relation_id fact-version identity와 same-batch latest-order/overwrite를 봉인한다.
같은fact/같은target 반복은 slot/teacher/birth/age를 reset하지 않으며 occurrence만 ledger추가.
Current fact는 해당batch pastloss에서 제외, 동일target 기존version은 이후돌아올수있다.
다른target은 oldversion retire후commit에서newversion활성. 실패edit도ledger/평가분모유지.
Teacher는 canonical supplied target 전체 TF의 최종 at-write full-vocab 분포,
새생성문/256target truncation/top-k 대체0; nativefallback teacher도보존.
원version teacher는 coldartifact로보존하며 bankeviction후최신entry로재생성하지 않는다.
Terminal B100에 후속소비없는 teacher추가forward0. B2 teacher/pending은 다음batch에 필요하다.
Commit단위 nativehistory/ledger/bank/pending/teacher 적용 exactly-once, 후보계산중M수정0.

## 4. 선택feature / 실제loss / 계산효율

GSS feature는 동일 **현재 own W_N**의 target NLL L4gradient·fixed Pstar,
objective replay는 **원 at-write full-vocab KL**. 서로혼용/추가NLLloss0.
R512 모든문서/validgeneratedposition W0teacherKL 유지; GSS로reference줄이지않는다.
각history targettoken평균, reference문서평균, L_R+L_H coefficient1.
REC age=t-1-version_created_batch, h=5.12, w=1+2^(-age/h), history내정규화.
Loss recency와 signed-cosine selection분리, absolutecosine/최근quota0.

NLL factor A_i K_i^T는 prefix포함 모든 validinputposition.
고정Gaussian output/inputdim32×32, 독립replica2/2048feature/mapseed20260920,
Pstar Vrandom 사전계산·projectedkey캐시. 과거state의A/G/sketch재사용0.
pool≤512는 selection NLLbackward/sketch/GSS0, 모두선택하여KL만계산.
pool>512는 동일microbatch graph에서 NLL VJP+KL VJP 각각1회, sum per-fact tokenmean;
NLL A→sketch→discard, KL A hostcache→선택후 weightedgradient합성.
Pool전체graph GPU유지0, per-fact denseL4gradient CPU/GPU저장0.
두VJP를공짜라고하거나 microbatchsize로추가평균하지말것.
선택후H/teacher/weights를최대2candidate사이에고정하고 pool612가아닌선택≤512평가.
Reference/history gradient 교차항을 포함한 G_R+G_H를 부모selector에 넣는다.
Primaryepsilon.05/current-onlyweightedQ/추가current downstreamguard0/기존Armijo·수식불변.

GSS signedcosine surrogate ||sum u||²; max u_j·sum u 제거를cap까지반복,
fixedhashpriority seed20260920/fact-version tie. zero≤1e-12는계약fallback, nonfinitetechnical.
한번제거surrogate 최적과globaloptimality 구분, originalGSS-Greedy재현이라고부르지 않는다.
B7은 예상첫overflow이지중간gate아님; overwrite에따른실제overflow시점기록.
Hash4 weightedKLfactor-vs-direct 및 hash64 factor exactinnerproduct-vs-sketch 진단은
새경로의 일회 bounded계측, 성능gate/수치오차숨김/차원튜닝으로쓰지않는다.
불안정은 SKETCH_SELECTION_UNRESOLVED, 기존precision NOT_ESTABLISHED를 새PASS로승격0.
명백한runtime/identity/finite/IO오류는선보고·원자료보존후 최소수리 가능;
methodthreshold/horizon/bank/recency수치변경은별도지시없이는금지.

## 5. 연속실행과 같은 prefix 공유

세논리arm은모두W0/M0에서출발. B1/B2 사전확인은10000stream의일부로중복계산0.
동일source/input/W/M/RNG/teacher/ledger/bank목적이확인되는prefix만계산공유.
Overwrite없는설계상 B1–B2 ALL, B3–B6 uniform RES/GSS와REC,
B7이후3개state로갈라지며 실제identity가다르면임의공유0.
GSS/RES 구분없이B6공유라고하드코딩하지말고actualbank/teacher/weights도결속.
B2확인은sealed프로그램내부에서하고통과후B3즉시계속; agent관찰/승인은기다리지않는다.
기존B300에는새GSS/REC의 B2 teacher/version정책과checkpoint가없으므로
과거B2state를이어서10k라고포장하지않는다.
과학무개선/nativefallback은결과로기록하며3arm모두계속한다.

설계 no-overwrite prefix-sharing수치292native/Rgradientstate·candidate≤584,
GSS_NLL115032fact-VJP/188sweep는 예상/상한이지실측값이아니다.
Prefix공유안된경우의추가재계산은명시하고조용히비용숫자를맞추지않는다.
Standalone각method100batch, 공동연구비용을3으로나누어개별속도로쓰지않는다.

## 6. 평가·산출물

매batch currentR/P/N, first100cohort, fixedhashhistorypanel.
Full latest-valid historyRS/PS+누적N 관측batch는
[2,5,10,20,30,40,50,60,70,80,90,100]. 이들은관측시점이며再gate도checkpoint저장시점도아니다.
패널크기·hash 규칙은미정이면기존부모/실행규약에서근거를골라성능보기전봉인,
모든arm동일패널·전체history관측과분리.
RS/PS/NS NLLpreference 외 사용자가요청한 rewrite/rephrase/neighborhood
TF tokenmicro/promptmacro/full-targetstrict, true/new/desiredNLL, margin/pairedlost-gained.
OfficialP/N은선택후observer전용, 동일forward에서통계·필요teacher재사용.
Bank안/밖·age·relation·active/superseded·atwrite성공의forget/recovery,
native손상vs보정신규손상, rank/response/KL_R/H/fallback/sketch오차/teacher실패율.
평균동일과ID동일구분; requestclusterbootstrap10000seed20260920 유지.
진단·observer·teacherI/O·실패후보·준비공유비용을방법throughput과총allocation에서구분.
원raw/compactsource/config/input/map/teacher/ledger/provenance는local,
Git은한국어compactreport/CSV/코드그림/manifest만.
최종report생성은프로그램/CPU코드에준비하되실험완료후agent review는사용자recall시수행.

## 7. 자원·저장·제출 후 monitoring0

Server3 project/taskcap1,1GPU8CPU/host≤121856MiB/ubuntu/gpu/exportNONE/Requeue0.
기존job/다른실험cancel/hold/throttle변경0. 제출전한정admission만실시.
한persistentlane에서세RAMbranch를유지하여중복B1/B2·checkpoint없이끝까지간다.
설계상수십시간이상가능하므로기존24h를관성상속하지말고,
실제B300구성요소/새pool·teacher·observer비용/세arm·schedulermaxwall로계획을산정한다.
공유cache/microbatch/streaming만으로resource최적화,분모·historycap·3arm축소0.
자원상불가능하면명확한필요량을보고하고임의로설계를바꾸지않는다.

S3 reference전송후free약17GB관측은과거비독점값이다.
Coldversionteacher/keys/ledger/activationcache/observer/atomictemp peak를새로산정.
Coldteacher는immutable replay입력으로보존가능하나editedweight/resume bundle이아니다.
FP32fullvocab teacher손실압축/양자화0; lossless압축은decode동일성·CPU/I/O비용명시.
기존reference복제/자료삭제0. 공간부족시정확부족분·대체저장필요를보고.
저장시간·walltime문제를몰래checkpoint/중복prefixrun으로우회하지않는다.

구현·CPU·source/config/input/sequence/map/resource봉인→held exact검사→release까지진행.
**Release뒤scheduler/log/result/initialgate/B2확인/첫overflow/완료polling0,
sleep/heartbeat/callback/자동재개0**. 제출job과sealedprogram은자연진행.
actualinitial/terminal NOT_OBSERVED로제출인계 후 MONITORING_PAUSED_AWAITING_USER.
Preparation단계의필수한정검사와원인확인은모니터링으로바꿔반복하지않는다.
이전사용자directreview/51290monitoring은새lifelong자동관찰권한이아니다.

## 8. 구현·인계 소유권

별도 codex/server3-en-adapt-gss-history-10k-v1 worktree, source namespace
project/run_scripts/en_adapt_gss_history/** 및scopedtests/기존read-onlyadapter.
local/en-adapt-gss-history/20260920-v1/**, 기존inputread-onlyreuse.
허용 compactwrite:
audits/servers/server3/2026-09-20-en-adapt-gss-history/**,
experiment-reports/servers/server3/en-adapt-gss-history-10k-2026-09-20-v1/**,
tasks/status/server3-en-adapt-gss-history-20260920-v1/**,
runs/odeedit_en_adapt_gss_history_10k_s3_20260920/**,
messages/acks/server3/2026-09-20-en-adapt-gss-history.md,
messages/server-heads/server3/2026-09-20-en-adapt-gss-history*.md,
plans/updates/server3/2026-09-20-en-adapt-gss-history.md.
Resource+source+submissioncompactownscope nonforce branch/main게시까지허용.
Sharedidentity/env/model/oldsource변경0; complex독립부분만boundedworker,
단순검사/보고는SH직접. GH중복raw/GPU감사대기0.
M0는 FULL_READ/supersededrequest상태/source·입력재사용/미구현/GSS수식·metrics/
RAMstate·coldteacher·wall·disk계획. 이후제출job/source/lock/미관측범위인계.
