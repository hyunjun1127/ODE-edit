# GH → SH4: L4-preserving repair — R-GD / R-QP 두 arm만 실행

instruction_id: ODEEDIT-S06-L4-PRESERVING-REPAIR-TWOARM-SH4-V1
nonce: ODEEDIT-GH-SH4-L4-PRESERVING-REPAIR-TWOARM-20260917-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 1. 최신 사용자 지시와 우선순위

최초 사용자: “/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-17-l4-preserving-batch-repair-v1.md 이거 SH2에게 task 진행시켜라”.
준비 중 최신 override: **“R-GD와 R-Q 만 실험 진행시키자. SH4에게 전달하는 것으로 해”**.
R-Q는 지정 설계의 R-QP를 뜻하며 별도 새 방법이 아니다. 최신 override가 설계의 five-arm 실행표·최초 SH2 지정보다 우선한다. SH2에는 실시간 dispatch/실험 지시를 하지 않았고 신규실행0이다.

실행 대상은 **R-GD와 R-QP 두 신규 W0→first1000/B100×10 chain**만이다. 총20과학 batches, unique1000, arm-request2000. N4/REFIT4/LD 신규 chain은 제외한다. 이들 기존 cold7 결과는 identity/조건을 확인한 재사용 비교 근거이지 이번 신규 실행이 아니다. 재사용 불가/누락도 신규 baseline 제출 사유로 자동 전환하지 않는다.
필수 bounded numerical pilot은 구현 확인용 first100 한 batch다. 그 안의 native full L4 anchor는 방법 준비이지 별도 N4 sequential arm이 아니다. Pilot 비용·관측은 과학20batch와 분리하고 main에 warm carry하지 않는다.

최신 추가 사용자 override: **“cap 2로 늘리자.”** 현재 문맥의 Server4 GPU cap을 **2**로 갱신한다. 이전 cap1 지시보다 우선하며 다른 서버 cap은 변경하지 않는다. 기술/prep/teacher/두 과학 arm과 다른 project admission을 합쳐 동시에 최대2GPU다. 공통 기술 검증이 끝나고 다른 점유가 없으면 R-GD/R-QP를 각각1GPU로 동시에 실행하도록 두 slot을 활용한다. 이를 위해 불필요한 다른 arm을 추가하지 않는다. 이전 cold7의 당시 cap 초과 기록은 역사적 사실로 보존하며 이번 cap2를 소급 적용하지 않는다. 기존 다른 job의 취소·설정 변경 권한은 없다.

## 2. 정본 및 준비

새 clean codex/server4-l4-preserving-repair-twoarm-v1 branch/worktree, 새 /data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1/ namespace에서 구현한다. latest origin/main을 fetch하고 기존 c6b9937a29e663da305464cd7a04448b24ea6361 및 사용자/타SH 변경을 보존한다.

정본을 전체 읽고 exact bytes를 local authoritative에 보존한다.
- plans/global/2026-09-17-l4-preserving-batch-repair-v1.md — 189행/23142B, SHA a01588b8d64f9abe05625c7e3d611576b7b940a93fb446aff309deaa573fa158.
- plans/global/2026-09-17-l4-preserving-batch-repair-contract-v1.json — design_revision2, SHA37fefce4c1131d31446ea15805b25700ca8d74537bcacd5d0fd934ed7378d02e.
- 원 cells-v1.csv SHA8e5c6144f6c48cd5e6f863d83a915d064cf362ede41304f9208de3b8cdea323d: 역사적5arm 설계 bytes 보존, 실행 allowlist는 이번 dispatch-approved-cells.csv의2arm.
- audits/global/2026-09-17-l4-preserving-repair/{validate_math.py,math-checks.json,validate_contract.py,contract-checks.json}. 기존8 toy math+28 contract checks는 실제 Llama 검증이 아니다. Toy solver는 최대2차원/소수 제약용이므로 3D 실제 guard용 solver로 그대로 승격하지 않는다.
- companion source-input-manifest, 최신 PROTOCOL/session/cap 정책 및 본 envelope.
- 이전 cold7 source32a92ad6f3fff2f258d8778f3936d152e975ac1b/보고 c6b9937a, native BLUE/fitting/finalizer, snapshot/restore/observer를 읽고 재사용 경계를 기록한다.
- 설계가 참조한 GH cold7 독립리뷰의 정본 prose는 companion references/cold7-independent-review-ko.md에 exact-copy로 동봉했다. 원본 SHA5c9a50f177f24a91239c51f671f9e39e7102ccbd4f6c72008f14f53b54cadbdd. 이는 이전 검토를 재사용하는 것이며 이번 GH가 raw를 다시 검산한 것이 아니다. 내부 과거 제안(별도 paraphrase set 등)은 최신 설계의 금지를 override하지 못한다. 원 source 경로·참고 링크 맵은 manifest에 남긴다.
- 기존 B-OS 설계/부분 보고, 실제 selected functional adapter와 최근 SH2 SL-ZFlow의 all-token/materialization/transaction 검증 코드는 참고 가능하다. 기존 PCG/controller/target-space correction·ODE·stopping policy를 이번 방법에 끼워 넣지 않는다.

GH는 design/contract/cells/두 CPU source·결과와 cold7/B-OS 참고 보고를 읽었다. 실제 Llama 연결·실행은 아직 미검증이다. SH4는 source import/config/environment identity를 별도로 봉인한다. 공유 EasyEdit/BLUE/의존환경/기존 runner를 직접 수정하지 않는다.

## 3. 같은 cold 시작점·same sample

두 arm 모두 pretrained W0/zero M4에서 시작한다. Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager, matmul/cudnn TF32off, method seed20260916, cold7 context text+actual token IDs/RNG binding을 유지한다.
원 projector P4는 그대로 사용한다. 원 context가 필요한 경우 공통 W0에서 한 번만 준비한다. Arm별 다른 context나 warm W/M을 가져오지 않는다.

S4 fixed10k /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1, official load_prefix를 모델 load 전에 적용.
datasetSHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
wholeorder5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729,
first1000root40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
각arm ordinal[0,1000),100씩10batch. Shuffle/filter/성공 기반 제외0.

S64/Dev128은 cold7이 실제 사용한 common capsule와 teacher를 local에서 재사용 우선한다. 기존 Teacher192/47592와 cold7 effective teacher의 관계를 확인하고 model/token/scoring position을 결속한다. 동일 W0 full-vocab teacher, vocab합→scored128position평균→문서평균, S64online/Dev128observer 분리. C4 seed20260915와 method seed 혼동0. 재현 연결 실패 시만 동일 IDs/tokens/W0에서 필요한192를 공통설정으로 한번 재생성하고 그 비용·identity를 분리한다. Report256 신규생성/타서버 broad transfer0.

## 4. 반드시 유지할 방법

We는 이전 accepted 상태이며 과거 L8 repair를 포함한다. 매batch native local-z4/full L4 write로 WN을 만든 뒤 W4 및 모든 비L8 parameter를 고정한다. 원 native target/solve/hparams(L2=1, 기존 BLUE L4 설정)는 불변이다.
Repair는 W8=WN8+Σc_jQ_j의 실제 모든-token weight write다. L8 target_new compute_z/fitted activation target/native P8/key-basis/M8/edit-z-ball은 **R-GD/R-QP에서 사용하지 않는다**. 다음 batch z4는 자신의 실제 We에서 새로 계산한다. 독립 N4 chain과 W4가 항상 같다는 주장은 하지 않는다.
Repair off이면 이번 Δ8=0으로 exact WN을 선택한다. 이전 누적 L8 보정을 W0로 되돌리지 않는다.

온라인 input은 S64, Current canonical R100, received active-fact Past canonical R64뿐이다. Raw(subject,relation)의 최신 유효target/event, current overwrite 제외, 기존 고정hash priority와 인코딩을 봉인하고 성공 filtering0. **별도 paraphrase 생성·저장·학습·guard 및 official P/N online feedback 전부0**. 원 native training context 절차를 새 paraphrase guard로 확장하지 않는다. P/N은 선택 봉인 후 observer만이다.

WN에서 Current/Past 평균 target-new NLL 및 TF-strict 정확한 성공ID집합, old target이 제공되는 canonical rewrite의 new<old NLL 선호 성공ID집합을 봉인한다. Old target이 없으면 NOT_AVAILABLE이며 임의 복원/argmax 치환0.
Actual candidate는 각 패널 L≤L(WN)+1e-4, 각 strict/preference ID subset, finite/W4고정, 실제 S64 KL 개선>1e-6을 만족해야 한다. .05 plateau는 없다. Current/Past 평균은 합치지 않고 요청별 악화·tail도 남긴다. Past anchor는 We가 아니라 WN이다.

R-QP: gB/gR/gH(비어있으면없음)의 full W8 gradient span을 FP64 Gram/QR로 정규직교화하여 최대3방향, B1최대2. Numerical zero/dependency 제거 판정과 residual을 잠근다. Q는 full W8 shape이며 별도 P8/LoRA/random/key support를 넣지 않는다.
R-GD: 같은 WN/S64의 Base descent **한 방향**에서 guard 없는 scalar quadratic proposal. 같은 GN curvature normalization, 실제 m=1의 trust scale, 반경축소·actual rewrite acceptance를 적용한다. Guard를 proposal에서 쓰지 않는 것과 실제 품질검사를 생략하는 것을 혼동하지 않는다. Dummy gradient로 비용을 맞추지 않는다.

Guard model: Current/Past mean NLL의 slack0, WN strict 성공 요청의 target-token margin과 제공된 preference 성공margin을 A c≤s로 둔다. Token competitor는 WN 최고 non-target에 고정, argmax-tie strict는margin0 가능. 새 경쟁token은 actual full-vocab 검사로 잡는다. 모든 logit equality/KL-anchor zero-gradient equality로 대체0.
Base GN은 p_N Fisher이며 gradient는p_N−p0. 문서/작은 microbatch별 full-vocab JVP를 누적·폐기하고 작은 H_B만 유지한다. 전체S64×vocab×directions 상주 또는 거대 PCG0.

Htilde=H_B+lambda_num I, condition cap1e6의 최소 shift와 whitening R=Htilde^-1/2:
lambda_num=max(0,(hmax−1e6*hmin)/(1e6−1)).
유의한 음의 eigenvalue, H≈0/유의gB는 기술문제로 분리한다. 임의 scientific ridge floor0.
QP는 FP64 min (Rᵀb)ᵀu+||u||²/2, ARu≤s 및 각|u_j|≤r. Raw/scaled primal·dual·stationarity·complementarity, row units/scaling·active set/dual을 저장한다. 작은3D라도 guard 수는 클 수 있으므로 2D toy brute-force가 production 규모에서 가능하다고 가정하지 않는다. Solver는 방법을 바꾸지 않는 구현 선택으로 정하고 CPU negative/large-scale cases를 검증한다. Matrix-norm-scaled tolerance로 음의dual/slack을 숨기지 않는다.

r0=sqrt(2*B(WN)/actual_m), B≤1e-6 또는m0이면 off. 최대6endpoint/축소5회, 실패시r/2, 같은 Q/b/A/H를 재사용하며 재선형화0. Rejected candidate마다 WN으로 완전 복원한다.
predicted=-[(Rᵀb)ᵀu+||u||²/2], actual=B(WN)−B(candidate); predicted>1e-6,actual>1e-6,agreement≥.1와actual품질 모두 만족하는 **첫** 후보만commit. 최대1repair/batch. 전체후보최적해·비선형최적성을 주장0.
정상 no-descent/conflict/guard실패/smallgain/FP32zero와 NaN/PSD/JVP/KKT 오류를 구분한다. 기술오류를 정상zero fallback 성공으로 숨기지 않는다. 낮은 NS/PS·off횟수는 run탈락/threshold tuning 사유가 아니다.

M4는 최종 whole-batch commit에 native Current keys로 정확히1회 append, candidate중0. Operational M8/P8 및 repair key history append0. 받은 요청 ledger는 off여도 전량 갱신한다. W4/W8/M4/ledger/context/RNG/next-index와 commit을 함께 durable하게 보존하고 double-append를 막는다.

## 5. 단계0 기술·비용 pilot와 단계1 두 main

필수 first100 numerical/mechanism pilot을 구현한다. 같은 saved native WN/inputs를 가능한 범위에서 공유하여 불필요 native fitting을 반복하지 않는다. Pilot은 scientific main의 accepted prefix가 아니며 두 main은 별도freshW0/M0다.
실제 모델에서 signed derivative 두 FD scale·JVP response/단위·fixed teacher·GN PSD/whitening·KKT·FP32 all-token functional↔physical materialization·W4고정·rollback/history0/최종historyonce·반복관측 오차를 확인한다. E/H repeat≤5e-5,Base≤5e-7,strict/preference 성공집합exact. 설계의 epsilon_L/B와scaledKKT1e-8은 불변이다.
FD step/scaling·abs/relative numerical tolerances·basis dependency/PSD 판정은 실제 quality 결과를 보기 전에 source/config/기술계약에 봉인한다. 국소성/roundoff를 확인하지 않은 단일큰FD를 미분오류로 단정하지 않으며, technical probe 수리와 방법변경을 구분한다. 첫예외/입력/gradient/probe/perturbation/복원증거를 실패 전 저장한다.
과거 EP diagnostic-skip을 자동 상속하지 않는다. 반대로 설계에 없는 heavy gate·motivation재증명·전과거checkpointGPUreplay를 추가하지 않는다. Technical failure는 영향받은 bounded 단계만 최소수리하고, 과학식/허용치/arm변경이 필요한 모순은 typed HOLD한다. 유효한zero도 정상 pilot 결과이며 nonzerorepair를 강제하지 않는다.

실제 first100 wall/peak/target/gradient/JVP/probe/observer/I-O를 얻어 두chain 비용과 disk·wall 계획을 봉인한다. Sweeps를 singleforward로 계수하지 않는다. Hour hardcap=null이며 과거 실측을 이번 총예산으로 강제하지 않는다. 자원부족을 품질허용치 완화로 해결하지 않는다.
기술 PASS/공통 source·sample·teacher closure 후 두 main을 모두 upfront 등록한다. Science 목록은 R-GD/R-QP만, 각 B100×10이다. 다른 점유가 없으면 array%2 또는 두 독립 job으로 동시 실행하고, 한 arm의 완료·점수에 다른 arm을 종속시키지 않는다. 공통 기술은 afterok+READY로 연결하여 기술 준비와 main의 중복 점유를 막는다. 다른 project admission이 있으면 합계 cap2를 지키는 scheduling을 봉인한다.
계획 과학20batch의 nativeL4 target2000/기본native solve20, L8target0, M4append20, repair검사최대120은 실측과 구분한다. Technical fitting/repeatedprobes/teacher는 별도비용이다. No-update도요청분모유지.

## 6. 자원·모니터링 경계

S4 total GPU cap2, 각1GPU/8CPU/mem60416M/exportNONE/Requeue0. Actual active+admitted pending의 동시실행 가능 용량을 합산해 submit 직전 재계수하고 held→owner/source/args/dependency/resource inspection→release한다. 기존 job mutation0. 필요한 teacher/pilot/main을 합쳐 최대2GPU이며 공통 준비 완료 뒤 가용 두 slot을 R-GD/R-QP에 쓴다. 다른 사용자의 자원 변경0.
본 요청은 기존 **INITIAL_GATE_ONLY_OR_PENDING_HANDOFF**를 override하지 않는다.
- source/필수기술준비 및 이번scope 등록을 진행하고, 시작가능하면 최초실제repair candidate/zero사유→commit/history→다음entry initialgate를 한정확인한뒤 MONITORING_PAUSED_AWAITING_USER.
- 기술pilot만PASS한경우science성과PASS라고하지않는다. 파일/receipt로공통기술READY확인후가능하면두main모두등록한다. 첫점수로선별0.
- Pending/외부대기이면 actualNOT_RUN과 등록/미등록scope를구분해즉시인계한다. 기다리는polling/sleep0. 조건부등록은 실제science시작/PASS와다르다.
- 등록프로그램은모든10batch/평가/저장을자연진행한다. Pause뒤agent polling/heartbeat/callback/terminal대기/followupsubmit/분석확장/main통합0.
- source+compactsubmit/gate/pause본scope main게시만pause확정전허용. 최종상세리뷰는userrecall후진행한다.
타SH/과거B/A/SL-ZFlow/CAKE/cap/ORBODE/cold7리뷰는재개하지않는다. SH4동료사용은복잡한독립부분만소유권분리/최종pause시정지,단순점검직접수행. GH중복raw/GPU감사·단계별승인을선행조건으로요청하지않는다.

## 7. 관측·baseline재사용·후속보고 준비

매batch We/WN/selected Current canonicalR/P/N, strict/token/two-P,true/newNLL; W5actualfirst500, W10samefirst500/후반500/전체1000, atwrite→W10 및 commonW0-success조건부N유지를저장한다. Tie=failure·모든requested분모유지,ALL/ACTIVE/SUPERSEDED와정당overwrite분리.
B1/B5/B10에서는 선택을먼저봉인한뒤 검사했던실제후보들의공식P/N 및 고정earlycohort를observer로계산한다. Earlycohort는미래/성능으로고르지않고firstB100처럼도착한고정prefix를실행전에명시한다. 선택전P/Naccess0;후보가상chain합성0. 동일rows재사용으로이중분모/forward계수0. Dev128은기존규정observer.
Q/b/A/H/lambda/radius/solverKKT/dual/anchorquality/guard성공ID/probe예측·실제gain/사유/accepted를연결한다. Gamma_free/Gamma_safe는동일trustbox,free는editresponse제약만제거;basis/box도다르게놓고ratio를계산하지않는다. Γfree≈0이면NA. 단위가다른dual직접비교0.
각batch W4/W8/M4/ledger/contextRNG/nextbatch와smallmodel/selectedprobe를재구성할checkpoint를남긴다. Q와fullgradient중복저장을피하고필요directions/response를CPU에보존;전체pretrained/Jacobiancopy0. CPUreload/hash와실제GPUresume검증범위를구분한다. No-checkpoint CAKE지시는상속하지않는다.
Target/loss/Adam·Base/Current/Pastgrad·directionJVP·QP·accepted/rejectedF/B/tokenwork·teacherstreaming·commitIO·peak와allocation을분리한다. 중첩timer중복가산0,순수writer미계측은NOT_SEPARATED.

N4/REFIT4/LD는이전cold7 W0/zeroM/seed20260916/sameprefix결과를read-onlyreuse하고 비교호환표를작성한다. Source/runtime/context/teacher/kernel/evaluator/sampling 차이는그대로공개하며exact미검증을PASS로바꾸지않는다. 비교불가항목은NA/HISTORICAL_REFERENCE,신규baseline0. 기존전체7armraw를동기검증명목으로재평가0. WarmB51–60/과거10k/singlelayer다른seed는별도참고일뿐이번paireddenominator와섞지않는다.
최종science목표RS/PS손실허용0pp는효과판정용이지B1실행선별gate가아니다. PS하락은NS이득으로상쇄하지않으며request-cluster불확실성·pairedlostgained/tails를보고한다. 한seed개발stream의동점/양의차이를보편적비열화증명으로확대0. SH는수치/사실/한계를정리하고GH가별도해석한다.

## 8. 허용 경로·인계

원설계/contract/원5cells/기존CPUevidence는byteexact보존하며새2armoverride와실행lock을분리한다. 기존runner/native/global파일수정0.
허용write:
- project/run_scripts/l4_preserving_repair/ — 새runtime/gradient/JVP/GN/QP/transaction/observer/analysis/tests/plot.
- /data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1/ — authoritative/cold/teacher(ifneeded)/technical/twoarms/receipts/raw/checkpoints.
- audits/servers/server4/2026-09-17-l4-preserving-repair/
- experiment-reports/servers/server4/l4-preserving-repair-seq1000-2026-09-17-v1/
- messages/acks/server4/2026-09-17-l4-preserving-repair.md
- messages/server-heads/server4/2026-09-17-l4-preserving-repair.md
- runs/odeedit_l4_preserving_repair_s4_20260917_v1/
- tasks/status/odeedit_l4_preserving_repair_s4_20260917_v1/server4.json

Slurm ALLOWED는 지정된 bounded technical/prep와 R-GD/R-QP 두 science chain뿐이다. 5arm/10k/main N4·REFIT4·LD/고정(.75,.5)/P8 ablation/m≤8/새 paraphrase/추가 barrier·ODE/sweep는 실행하지 않는다. 기존결과삭제/타서버전송0;NO_BROADCAST_NOT_REQUIRED(기존S4자산localreuse). 새로운외부전송필요시exacttargets/owner/size를먼저보고하되이미가능한독립CPU작업계속.
본scope구현검증·compactreceipt/report는최신타SH/GH변경을보존하여nonforcemain게시. Raw/tensor/prompt/fullstdoutGit0. 그림은코드로생성하고GFM표·링크·분모검산한다. Helper가명시허용path를거부하면근거/한계를기록하며거짓PASS나sharedpolicy수정0.
최초FULL_READ/M0에 latest two-arm override/cap2/source/미구현·재사용/기술계약/actual아닌자원예상·등록계획/모니터링경계를회신한다. 끝에는job/source/lock/output/resume및actualvalidation수준을인계하고사용자호출을기다린다.
