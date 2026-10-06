# Interference-priced group-L1 budget: 실제 순차 실험 계획

상태: **DESIGN_READY_NOT_IMPLEMENTED_NOT_SUBMITTED**. 이 문서는 실험 설계다. 새 production 구현, 실제 모델 검사, CPU 수치 검사, job 제출, GH/SH 전달을 수행했다는 보고가 아니다. 수식과 경계 조건은 [method-spec.md](method-spec.md), 구현 계약은 [implementation-contract.json](implementation-contract.json), 실험 설정의 정본은 [experiment-contract.json](experiment-contract.json)을 따른다.

## 1. 출발 증거와 검증할 주장

Source base는 `782c4c7a`다. v12-R의 실제 실행 source `635798ba276957312ec1aceda686ad563c906957`과 중간 결과를 게시한 source base를 구분한다. [검산된 W15 보고](../../../experiment-reports/servers/server4/jlz-v12r-realized-response-2k/intermediate-main-W15/report-ko.md)에서 MAIN job59262는 B1–B15,1,500 edits를 완료했다. B16 후보 로그 쓰기에서 ENOSPC로 종료했으며 W20은 없다. 공간 소모 주체는 `NOT_IDENTIFIED`다.

| scope / 지표 (%) | W5 | W10 | W15 |
|---|---:|---:|---:|
| allseen PS | 96.20 | 94.55 | 90.033 |
| allseen NS | 85.08 | 77.15 | 69.607 |
| 동일 first500 PS | 96.20 | 92.50 | 83.00 |
| 동일 first500 NS | 85.08 | 77.18 | 69.64 |
| 동일 first500 P TF strict | 72.90 | 64.70 | 51.10 |
| 동일 first500 N TF strict | 18.96 | 17.32 | 14.70 |

출처는 같은 디렉터리의 [metrics.csv](../../../experiment-reports/servers/server4/jlz-v12r-realized-response-2k/intermediate-main-W15/metrics.csv)다. 위 first500은 같은 요청 집합을 재평가한 값이며, 분모가 늘어나는 allseen과 분리한다. 완료된 prefix의 성능 변화와 ENOSPC 종료는 별개의 사실이다. 이번 게시에는 다른 네 대조군의 결과가 없으므로 L2, full-M pullback, expansion 중 하나의 독립 원인이 밝혀졌다고 쓰지 않는다. 기존 층별 Q/update norm 표와 사용자가 붙여 넣은 수치·toy PASS를 새 가격의 의미적 간섭 예측력에 대한 실제 검증으로 상속하지 않는다.

이번 질문은 **같은 realized-response 학습·ridge writer를 유지하면서, group-L1 제약의 층별 가격 배치가 획득과 누적 보존의 관계를 개선하는가**다. 새 ENERGY/L2 arm은 실행하지 않는다. 따라서 L2→L1 geometry 자체의 독립 인과효과를 주장하지 않는다. 가격은 입력 response의 간섭 proxy이며 실제 출력 손상이나 lifelong 안정성의 보장은 아니다.

## 2. 고정된 세 arm

모든 arm은 L4–L8 전체를 eligible로 두고, 동일 source/runtime/input/profile과 cold W0/H0에서 독립적으로 first2000을 BS100×20으로 편집한다. 기존 edited checkpoint, 다른 arm의 W/H, optimizer, teacher, response map 또는 가격을 가져오지 않는다.

| arm | candidate0 가격 처리 | 비교 역할 | 순서 |
|---|---|---|---|
| PRICE | method의 정규화된 층별 π 그대로 | 주 방법 | 우선 |
| FLAT | 같은 raw score를 계측하되 effective π는 모두1 | 균일 가격 group-L1 대조 | PRICE 다음 자원 허용 순서 |
| REVERSE | own-entry 정규화 π의 순위 배치를 역전 | 가격 multiset을 보존한 배치 방향 대조 | PRICE 다음 자원 허용 순서 |

가격은 각 arm·batch의 정상 candidate0 BUILD에 이미 필요한 native mean response `M=P.T@Kbar`에서 계산한다. `raw_score=κ_raw`를 저장하고, `κ_floor=max(κ_raw,1e-6*max_l κ_raw,1e-12)`, `π=κ_floor/min_l κ_floor`를 사용한다. raw score가 모두0이면 π는 모두1이다. 이 문단의 score 식과 축의 정의는 method 정본에 따른다. 새로운 W0-only key 연구나 가격 보정 fit를 만들지 않는다.

가격은 해당 batch의 25후보 안에서는 고정하고 다음 batch에서 own W/H로 다시 계산한다. 매 후보 실제 BUILD의 upper K/P는 기존처럼 새로 구하며, 가격 고정이 actual builder의 upper geometry 고정을 뜻하지 않는다. 기존 source의 raw A를 임의 대칭화하거나 jitter를 넣지 않는다. full causal K/P 역미분을 새로 추가하거나, 현재 v12-R의 선언된 pullback 근사를 바꾸지 않는다.

REVERSE는 요청 r마다 eligible layer의 index를 `(π_lr, layer_index)` 오름차순으로 stable sort하여 `l_0,...,l_(m-1)`을 얻고 `π_reverse[l_i,r]=π[l_(m-1-i),r]`로 정의한다. 난수나 결과 기반 재배열을 사용하지 않는다. exact tie는 그대로 허용한다. 각 요청의 원 π, effective π, permutation index, exact tie 수, 값이 실제로 바뀐 layer 수와 unchanged-request 수를 기록한다. 변화가 없는 tie 요청을 버리거나 다시 섞지 않는다. 같은 entry에서는 multiset/min/max/합이 정확히 보존되는지 검사한다.

이 invariance는 **동일한 entry에서 만든 원 가격과 그 역전본** 사이에 적용한다. 세 순차 arm의 W/H가 갈라진 다음에는 서로 다른 own-entry 가격을 갖는다. PRICE와 REVERSE가 모든 batch에서 같은 숫자 가격·최종 강도·상태를 공유한다고 주장하지 않는다.

## 3. 공통 fit·budget·controller

모델은 Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, native anchor L8, FP32 model/activation/requested residual, FP64 geometry/pullback/projection, eager, autocast off, TF32 off다. Native rewrite/KL 문장, context/token/subject lookup, NLL/KL reduction, teacher current||entry, H의 최종 rewrite-only mean key append를 유지한다.

`c=.75`, native KL `.0625`, norm `.5`, covariance `15000`, EfficiencyAdam의 lr `.1`, betas `(.9,.999)`, eps `1e-8`, maximum25 evaluations/24 updates를 고정한다. Optimizer 변수는 absolute R이며 group budget은 `sum_l π_lr*||R_lr||/a_lr <= β_r`, local cap은 `||R_lr||<=.75*a_lr`다. Native norm loss만 `.5/(a_star_r²)*sum_l||R_lr||`로 L8 anchor를 사용한다. Main과 대조군은 같은 좌표·norm·projection 및 마지막 평가 후보 commit을 사용한다. weighted group-L1의 exact absolute-R Euclidean projection은 method/implementation 정본을 그대로 구현하며 별도 strength sweep를 넣지 않는다.

새 budget은 dimensionless `β_base=.75`, `β_max=.75*max_l π_effective,lr`다. `e=0,...,4`에서 `β_e=.75*exp((e/4)*log(max_l π_effective,lr))`를 쓴다. 기존 own-update grace12 뒤 현재 native `F_r>=.05`이면 update 전에 한 단계 확장하며, 최종 평가 후보에서는 backward/update/expansion을 하지 않는다. β가 같은 경우에는 불필요한 expansion을 만들어 기록하지 않는다. 이 식은 이전 absolute `rho_base/rho_max`의 재사용이 아니다.

FLAT은 `max π=1`이므로 β_base=β_max=.75이고 확장이 없다. 따라서 PRICE–FLAT은 가격과 가격에 연결된 budget controller를 합친 대조다. PRICE–REVERSE는 동일 entry의 price multiset과 β 상한을 보존하지만, 이후의 F·own update·expansion·trajectory는 달라질 수 있다. 이 차이를 숨기지 않는다.

같은 β, 같은 requested norm 또는 π 정규화를 **동일한 의미적 편집 강도**라고 부르지 않는다. 공식 P/N 성능, W0 locality, 이전 실패 결과를 보고 계수·radius·중간 후보 선택을 바꾸지 않는다. PRICE의 B1 성능으로 FLAT/REVERSE 실행 여부를 정하지 않는다.

## 4. 별도 pilot 없이 실제 B1에서 검사

실행은 입력/source/runtime 결속 뒤 각 arm의 cold B1 candidate0부터 시작한다. Synthetic/toy 실험, W0 standalone key study, 별도 small-B/full-B fit, B1 성능 selection gate 또는 신규 baseline fit를 단계로 추가하지 않는다. 수식 유도는 method 정본을 따른다. 이 문서 작성 단계에서 수치·GPU 검사가 통과했다고 주장하지 않는다.

다음 항목은 같은 B1·같은 trajectory의 이미 계산한 tensor를 대상으로 검사한다. 통과 여부는 기술 정합 판정이며 RS/PS/NS 기준이 아니다.

- 전체 logical B의 mean-key/owner/layer/shape identity, raw A와 response M의 finite 여부, declared source/dtype 및 cold W/H 결속.
- score/floor/정규화의 finite·positive 범위와 all-zero 규칙, arm별 π 적용, REVERSE multiset/assignment 검산. 요청·층·가격을 제외해 통과시키지 않는다.
- weighted group-L1/local-cap projection의 KKT 식과 저장 FP32 residual의 실제 feasibility를 method 정본의 사전 고정 허용치로 검사한다. 허용치를 결과에 맞춰 늘리지 않는다.
- 정상 가격 계산은 c0의 existing M을 재사용하여 추가 model forward/solve가 없다. 실제 B1의 모든 eligible layer에서 고정 source/recipient pair `(r,j)=(0,1),(1,0)`을 사용하고, 기존 c0 P/Kbar/A로 `q=p_r+p_j*M[r,j]/(1-M[j,j])`를 구성한다. `A*q+Kminus*(Kminus.T*q)-k_r`의 선형계 잔차와 `q.T*k_j` 대비 `M[r,j]/(1-M[j,j])`를 검산한다. 신규 leave-one-out solve/factorization/forward/fit/update/commit은 모두0이며 기존 tensor의 matvec/reduction만 계상한다. B=1이면 pair 검사는 NOT_APPLICABLE이다. Nonzero k_r의 상대 residual은1e-6 이하, zero k_r의 절대 residual은1e-8 이하, coefficient 절대오차는 `1e-8+1e-6*abs(coefficient)` 이하다. 대상·tolerance는 결과에 따라 바꾸지 않는다.
- 매 batch exact terminal payload copy, no-resolve/no-double-add, final native history once, own next-entry join, observer 비변이를 확인한다.

비유한값, 잘못된 identity, solver/projection 검산 실패는 typed technical failure다. 새 coefficient, exact writer, fallback writer, hidden shrink 또는 추가 fit로 조용히 대체하지 않는다. 유한한 budget 소진·낮은 성능·가격 tie·집중 배분은 정해진 status와 함께 기록하고 정상 commit 정책을 따른다.

## 5. 평가와 attribution

입력은 기존 fixed CounterFact10k의 동일 first2000 schedule, seed20261002다. [기존 schedule](../../../plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv)의 SHA256은 `dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2`다. 재추출·shuffle·중복 제거·성능 기반 filtering은 하지 않는다.

| 관측 | 문항·분모 | 사용 |
|---|---|---|
| W0 | first2000 R2000/P4000/N20000 | 정상 초기 평가. runtime/token/state identity가 같은 raw만 재사용 가능 |
| 각 batch pre/post current | R100/P200/N1000 | 해당 batch의 획득과 즉시 locality 변화 |
| W5/W10/W15/W20 allseen | 해당 prefix 전체 | 누적 성능. W20은 R2000/P4000/N20000 |
| first500 | 같은500을 W5/10/15/20 | 분모 고정 유지 곡선 |
| birth cohorts | at-write→각 후속 milestone | edit age와 작성 직후 성능을 분리 |

같은 state/같은 문항의 평가는 재사용한다. Milestone current는 allseen raw에서 추출하여 중복 forward하지 않는다. W0는 정상 평가이며 별도 실행 admission/품질 gate가 아니다. 기존 evaluator와 CPU row reducer를 재사용하되 새 budget/price 기록 검산을 추가한다. 기반 코드는 [run.py](../../run_scripts/jlz_v12r/run.py), [collect.py](../../run_scripts/jlz_v12r/collect.py), [observe.py](../../run_scripts/jlz_realization/observe.py)다.

모든 endpoint에서 RS/PS/NS의 정수 분자·분모, TF strict/token-micro/prompt-macro, desired/true/new NLL과 양방향 margin을 보고한다. R/P desired=new, N desired=true이며 ties는 preference 실패다. Teacher forcing을 free generation으로 표시하지 않는다. W0-correct 및 at-write-correct의 lost/gained/retained를 보존한다. occurrence primary와 active/superseded 보조 분석은 같은 seen-prefix 정보만 쓴다.

Primary comparison은 PRICE–REVERSE의 W20 및 same-first500 trajectory, PRICE–FLAT의 보조 대조다. Paired raw row identity를 연결하여 모든 지표를 보고한다. 단일 seed/order이므로 보편적 우위나 통계적 동등성을 확정하지 않는다. 낮은 지표를 숨기기 위한 종합점수·endpoint 선택·신규 significance gate를 만들지 않는다.

획득을 줄여 NS만 높인 결과를 배분 개선으로 해석하지 않는다. 전체 요청의 R/P/TF/NLL, 작성 직후 성공 건수, requested/realized 총량·share·가격, Q_C0/Q_H·update norm·active group 수·β/expansion을 함께 공개한다. 동일 획득 조건의 사후 strata를 보조로 보고한다면 strata 정의와 원분모를 명시하고 원 cohort를 대체하지 않는다. P/N으로 strength를 맞춘 추가 fit, scalar sweep 또는 사후 budget 보정은 없다. 본 실험만으로 정확한 strength-matched 효과나 L1 geometry의 독립 효과를 주장하지 않는다.

V12R W15와 기존 CD/MEMIT-H 등은 일치하는 endpoint/문항만 historical reference로 표시한다. W15를 W20으로 연장하거나 기존 run의 runtime/seed/층 차이를 숨기지 않는다.

## 6. 실행량·비용·저장

세 arm 전체의 최대치는 distinct cohort2000,physical edit applications6000,60 commits,57 interbatch joins,300 layer-history appends,1500 BUILD/subject-forward,1440 logical subject-backward,144000 request updates다. 정상 main 최대25/24 규칙 안에서 종료가 빠르면 실측 수가 줄어든다. 관측용 actual forward와 실제 B1의 LOO residual matvec는 main candidate 수에 숨기지 않고 별도 계상한다. 이 matvec 검사 때문에 추가 solve/factorization 또는 model call을 만들지 않는다.

PRICE를 먼저 배치하고 FLAT/REVERSE는 실행 당시 GH가 실제 node/QoS/현재 작업의 cap에 맞춰 병렬 또는 순차 결속한다. 기존 job을 취소·변경하거나 자원 cap을 늘리는 권한은 이 문서에서 만들지 않는다. 두 대조군은 PRICE의 품질 조건에 의존하지 않는다. 실행 host/GPU/CPU/RAM/wall/ETA는 현재 **NOT_MEASURED/NOT_BOUND**이며 임의 수치를 추정해 넣지 않는다. Runtime 결속은 실행 준비 업무이며 별도 사용자 성능 승인 단계가 아니다.

Fit 안의 BUILD/subject/pullback/price/projection/telemetry와 observer·history·I/O를 exclusive timer로 구분한다. Inclusive fit/batch와 내부 timer를 더하지 않는다. Allocated GPU-seconds는 parent job당 한 번만 센다. GPU/host peak는 실제 측정한 값만 보고한다.

ENOSPC 재발을 막기 위해 candidate 로그는 고정 schema의 compact scalar/요청별 소형 요약으로 제한하고 반복 tensor·모델 weight·전체 vocabulary distribution·동일 payload 중복 저장을 금지한다. 모델/텐서/checkpoint/복원 bundle의 새 대규모 allocation은 없다. 평가 raw와 source/hash/commit receipt를 우선 보존한다. 기존 raw는 삭제·덮어쓰기하지 않는다.

실행 전 serializer의 최대 row 수·허용 scalar 배열 길이·record byte 상한으로 세 arm 저장 상계를 계산하고 기존 파일 크기와 실제 여유 공간을 결속한다. 구체 byte 상한은 implementation receipt에 저장하며 추정값을 실측이라 부르지 않는다. Batch 경계에서 다음 batch의 봉인된 최대 write 크기와 오류/terminal receipt reserve를 만족하는지 확인한다. 부족하면 새 batch fit 전에 `RESOURCE_BLOCKED_STORAGE`로 중단하며 데이터 삭제, 평가 축소, 로그 항목 임의 누락으로 이어 가지 않는다. 쓰기 실패가 rollback/terminal 증거도 막으면 복원 확인은 `NOT_VERIFIED`로 남긴다.

각 arm은20 commits와 W20/compact report 뒤 종료한다. noB21/noCP/exact_resume=NOT_AVAILABLE다. Partial/기술 실패는 완료 prefix와 분모를 그대로 보고하며 미관측 지표를0점으로 채우지 않는다. 이 계획은 설계 완료 상태이고 구현·검사·실험의 완료 상태와 별개다.
