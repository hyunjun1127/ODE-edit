# v11 구현·qualification 계약

2026-10-04. SH4 소유 새 namespace `project/run_scripts/jlz_native_increment/`를 사용한다. 기존 v9/v10와 native baseline은 read-only reference이며 그 실행 코드를 덮어쓰지 않는다. 공통 helper 수정이 필요하면 이 namespace 안의 adapter로 해결하거나 GH에 정확한 추가 write scope를 요청한다.

## 1. 함수 책임

| 모듈/함수 | 계약 |
|---|---|
| profile.py / resolve_profile | B·layer/dim·native prompt/lookup/readout·dtype·coefficient·paths를 runtime에서 결속. V11 고정 method 선택은 변경하지 않음 |
| entry.py / prepare_entry | W/H identity, canonical block anchor, native KL teacher, rewrite entry keys/context-group mean을 생성. 각 batch fit 동안 고정 |
| geometry.py / freeze_entry_geometry | A=λC C0+H, frozen Khat, full-B SPD solves/cost factor. B×B off-diagonal 유지. Candidate마다 변경 금지 |
| subject.py / native_terms | Full-block subject δ hook; 요청 모든nativecontext에 동일δ. NLL token/context/native평균, KL current||entry, sumlayer norm |
| allocation.py / entry_ridge_cost | Stable G+E/root-sum 비용; 모든microbatch후 후보당딱1회. MAIN.1/NOALLOC0 |
| optimize.py / fit_joint | qAdam, Bsumbridge1회, clamp, Jmean<.05wholebatch 또는25후보, terminalevaluatedcandidate반환 |
| writer.py / apply_increment | L오름차순actualkey갱신→native ridge(D_l)→weightcast/add. z−h tracking/divisor 없음 |
| history.py 또는 writer.py / commit | FinalmodelmeanK→CPUFP32Gram 1회, nonselectedweight/rollback/W/H/RNG검증 |
| observe.py / evaluate | Native평가기동일RPN NLL·ACC·preference; 학습/stop/candidate선택과분리 |
| run.py / driver | 독립coldarm→20batches→지정current/cumulative평가→compactterminal. 파일identity/코드·자원lock |
| collect.py | CPU raw행재집계, paired/cohort/versionedclaim,분모/hash정합. SH보고는사실과수치 |

## 2. 계산 그래프와 메모리

Virtual forward에는 optimizer q→D→subject hidden→NLL/KL/norm gradient만 모델을 통과한다. Geometry는 entry에 고정되어 key/P/H로 gradient를 보내지 않는다. Frozen geometry 비용의 D gradient는 유지한다. 매 후보 actualbuilder 또는 actualweightgradient는 없다.

모델 parameters는 requires_grad=false지만 주입 이후 activation graph는 유지한다. Microbatch native loss를 나누어 backward하되 동일 q와 request-mean reduction을 사용한다. Entry allocation은 마지막에 한 번 backward한다. 전체 gradient를 B배하고 Adam에 전달하는 convention은 dense B*J와 일치해야 한다. q scale은 autograd parameterization에 내재시키거나 bridge에서 한 번만 반영한다.

모든 rewrite/KL 문장의 δ index를 case ID/request index로 확인한다. 평균 key의 가중치와 NLLcontext 가중치는 다르다. Canonical anchor는 entry no-hook block 출력이다. 상층 candidate virtual hidden을 anchor로 갱신하지 않는다.

Ridge residual D와 weight U를 구분한다. D의norm, actual U Kmean 및 U kcontext, whole hidden drift를 분리한다. 실제 FP32 materialized model이 평가·commit되는 상태와 동일해야 한다.

허용 효율화: 동일token/position을보존하는microbatchpaddingtrim, selected-positionfull-vocabhead, lowerprefixreuse, entrygeometryreuse, nativekeycaptureearlyexit. Nativebaseline은 기존실행순서와요청별Adam/stop/clamp를유지하며 새 batch최적화를이번에추가하지않는다. Evaluator/로더 공유의 최적화만 nativeparity를통과한 경우 허용한다.

## 3. 수치 및 native qualification

아래는 기술 정합을 위한 사전 허용오차다. 단순히 threshold를 통과시키기 위해 jitter, symmetrization, dtype저하, objective수정,실패요청제거를하지않는다. 실제conditioning/runtime때문에허용오차조정이필요하면 GH에원오차·재현·새수치근거를보고하여versioned계약으로남긴다. 새 사용자 성능승인을 요청하는 절차가 아니다.

| 검증 | 사전 기준 |
|---|---|
| CPU FP64 ridge identity·stablefactor·B/mshape | abs≤1e−10 (고정작은양정치검산) |
| CPU joint FD gradient | abs≤1e−7+rel1e−5 |
| Actual FP64 solve residual | ||AP+KKᵀP−K||F/max(||K||F,1)≤1e−8 |
| Stable cost 대 native solve의 G+E | abs≤1e−8+rel1e−6 |
| 동일모델/입력의native NLL/KL scalar parity | abs≤2e−5+rel2e−4 |
| FP32 microbatch dense/split D/q gradient | norm relative≤2e−3, denominator max(reference norm,1e−8); reference gradient norm≤1e−8이면 차이의 max-abs≤1e−6 |
| FP32 actual direct U/key delta/localadd parity | abs≤2e−5+rel2e−4, tensor residualmax 및 RMS동시기록 |
| Rewrite/keymean/target/token/context/lookup ID | 정확일치 |
| Accepted weight copy/hash, nonselected weights, history append count | 정확일치; append는batch/layer당1회 |
| 신규 teacher/reference/eval문장 fit유입 | 0 |

Preflight의 native parity는 m=1, allocation0, explicit 같은D를 주입한 loss/operator와 native compute_z/writer를 비교한다. qAdamtrajectory를 nativeδAdam과같다고검증하지않는다. Native writer는 같은inputK/D/A, 동일cast/add조건의 operator parity를검증한다. MEMIT-H wholepipeline은본래 L8residual/divisor를유지하는별도baseline이다.

B1,m1,actualpartialB3와wholeB논리/마이크로배치불변성을검증한다. T′/absolutefeedback 미포함을 sourcelevel및작은2layercounterexample로확인한다. 全層D=0이면 U=0,개별열D_r=0의crossmix는허용됨을검산한다. 과거v11검토의absolutezexample를학습코드로반영하지않는다.

## 4. Pilot과 본실험 진입

CPU math/source검사→실제nativeprofile작은qualification→main밖4case로각MAIN/NOALLOC/nativeBS2×2pilot. Pilot의평가지표는사실로만기록하며GOOD/BAD에따라계수·arm·layer·후보budget을바꾸지않는다. Toy/실제qualification의수치·state검사를통과하면별도사용자확인없이main을진행한다.

Whole-B100검증은MAIN B1본실행에포함한다. 입력ID/shape/frozengeometryhash/첫candidategraph/terminalcommit정합을확인하되mainB1을추가fit·재선택하지않는다. 메모리/속도microbatch선택은동일목적parity를통과한구현결정이며과학계수튜닝과구분한다.

실험순서는MAIN2000→NOALLOC2000→nativeMEMIT-H2000이다. 모든 arm별 W0/H0 cold재시작. CPU조건으로판정되는동일2k archivednative결과재사용요건을GH가충족했음을서명하면baselineGPU rerun만생략가능. 반대로과거결과가좋거나나쁘다는이유로재사용/재실행을정하지않는다.

## 5. 저장할최소telemetry

- 요청·배치·arm·candidate ID, logicalcandidate수/optimizerupdates/stopreason/thresholdstatistic, 각lossraw·weightedmean.
- 각층계획D/a의분포/요청별normshare, Qhat/Ehat/Chat/rootcost, entryK/A/P/hash와entrycapacitydiag/fullmatrixspectralsummary.
- 각층실제currentK drift, actualQ/E/C, plan-to-realizationmean/context변화/노름/투영, coefficientcrossmix. Actual C는실제K에서같은D로계산하며fitproxy와차이기록.
- 候補2/9/terminal의 NLL/KL/norm/allocation별D/q gradient norm·radial·totalcosine. Diagnosticgrad는optimizergrad에중복누적금지. Forward재계산필요시같은candidate/RNGstate를쓰고호출수를별도기록.
- Native KL teacher/input/lookup identity, entry/outputweight/historyhash, terminalonlycommit1회, rollbackstatus.
- 시계시간/최대GPU·hostmemory/유효token수/실제forward/backward/solve/head/microbatch수. 가속배율은실측전선언금지.
- R/P/N raw pair ID와targettokenhash, new/true NLL, desiredtoken correct/count, strict, tie,activeversion flag. Text/largearrays는ignoredlocal만.

Mean share는원인기여율이아니다. 현재costcapacity와실제손상상관은관측값이며실행gate로쓰지않는다. Layer집중/균등,NS하락,fitactualgap,RnewNLL강도차이자체로자동추가손실·scale·clamp를넣지않는다.

## 6. 장애·재현·상태

NoCP를유지한다. W/H/RNG transaction은현재프로세스RAM의rollback용이며완료후checkpoint복구를약속하지않는다. `checkpoint_saved=false`, `exact_resume=NOT_AVAILABLE`을명시한다. Slurmwalltime은SH4가작은pilot속도와currentqueue한도를고려해충분히배정한다. 끝까지가지못하면incomplete로기록하며새coldretry는같은scope/identity의기술재실행으로GH가조정한다. NS결과때문에재시작하거나일부좋은batch를이어붙이지않는다.

실행source snapshot,dependencyversions,model/tokenizer/C0/contextcache/datasetSHA,seed와W0/H0를lock한다. 허용된task외existingjob중단/재개,checkpoint삭제,데이터재추출,sharedsource수정은없다. SH4는factualreport를작성하고GH가정본·해석을관리한다.
