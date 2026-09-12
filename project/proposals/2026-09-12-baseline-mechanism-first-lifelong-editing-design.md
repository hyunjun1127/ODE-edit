# Baseline 메커니즘 분석에서 출발하는 lifelong knowledge editing 실험 방향

작성일: 2026-09-12  
상태: **연구 방향 및 실험 설계 제안. 이 문서 작성으로 신규 GPU 실험을 실행하거나 방법의 효과를 확인한 것은 아니다.**  
주장 상태: **baseline 성능 차이와 일부 실패 조건은 기존 자료로 확인됨. 원인 분리와 새 방법의 유효성은 미확인.**

## 1. 연구 방향과 첫 산출물

이번 연구는 baseline이 높은 edit retention과 낮은 locality를 동시에 보이는 이유를 먼저 분석한다. 그 분석에서 확인한 문제를 가장 작은 변경으로 고친 뒤, 필요한 경우 목적함수와 수정 layer를 확장한다.

사용자의 방향은 다음과 같다.

1. 저장소에 이미 있는 결과·진단·구현을 최대한 활용한다.
2. Baseline의 실패를 Barrier + ODE의 필요성으로 미리 해석하지 않는다.
3. Target 생성, writer geometry, history 보호, 실제 출력 반응 중 어떤 부분이 문제인지 확인한 뒤 method를 선택한다.

따라서 E0–E2가 본 실험이다. E3는 진단 결과에 따른 방법 선택이며, E4는 추가 layer가 필요한 경우의 조건부 확장이다. C0 누적 정규화도 처음부터 채택할 방법으로 정하지 않는다.

**첫 산출물은 baseline별 차이, 그 차이를 바꾸는 최소 개입, 아직 구분되지 않은 원인을 담은 mechanism report다.** 보고서에서 다음 방법의 구성요소와 정보 조건을 결정한다.

Ordered Response-Barrier, ODE, target-realization controller는 이번 실험군에 포함하지 않는다. 기존 패키지의 capture·미분·복원·행렬 연산은 재사용할 수 있다. 해당 패키지의 objective, 방향 선택, clipping, stopping rule이 새 진단에 자동으로 들어가지 않게 한다.

이 문서는 연구 방향과 비교 설계를 정의한다. 서버별 실행 명령·자원 배정·job 제출은 구현과 소형 계측 비용을 확인한 뒤 별도 실행 계약으로 작성한다.

## 2. 연구 질문을 세 가지 현상으로 분리한다

주 기준은 AlphaEdit BLUE-style single-layer, L2=1, B=100이다.

| 질문 | 주 비교 | 구분할 현상 |
|---|---|---|
| RQ1. 왜 일부 layer에서 edit retention이 높은가? | AlphaEdit L4/L5 대 L6–L8 | 처음 성립한 rewrite가 후속 편집에서 유지되는 차이 |
| RQ2. Retention이 비슷할 때 왜 일반화·locality가 다른가? | AlphaEdit L4 대 L5 | Rephrase와 neighborhood의 실제 출력 차이 |
| RQ3. 왜 유리한 layer가 editor에 따라 달라지는가? | 기존 MEMIT singleton 대 AlphaEdit singleton | Layer 위치와 writer의 결합 효과 |
| RQ4. 확인된 손상을 가장 작은 변경으로 줄일 수 있는가? | 같은 entry에서 native와 최소 개입 | 현재 edit·과거 edit를 유지한 locality 개선 |
| RQ5. 단일 layer 수정으로 남는 한계가 있는가? | 개선된 L4와 예산을 맞춘 추가 layer | 추가 보존 방향의 존재와 실제 유용성 |

AlphaEdit L4와 L5의 final RS 차이는 0.05pp, 10,000 requests 중 5개다. L4의 retention을 고유한 현상으로 과장하지 않고, L4/L5의 R 유지와 L4의 P/N 우위를 구분한다. MEMIT에서는 L7의 final RS가 가장 높고 L4의 NS가 가장 높다. 모든 editor에 공통인 “L4 우위”를 전제하지 않는다. [R1, R2]

## 3. 기존 근거와 재실험하지 않을 부분

### 3.1 기준 자료

[R1]의 보고서 blob SHA는 0caaa201ae7f0ed27757c26c5ebee24aed50034a다. 현재 작업 디렉터리에서는 원래 report 경로 대신 이 SHA와 일치하는 봉인 복사본을 확인했다. [R2]는 원시 평가 재집계와 추가 W0 pairing을 수행한 독립 감사다.

AlphaEdit L4-only의 실제 final 10k 결과는 RS 99.390%, PS 95.680%, NS 65.348%다. W0 NS는 89.212%다. At-write RS 성공 9,993개 중 final 손실은 55개이고, at-write NS 성공 72,505개 중 final 손실은 11,547개다. 이 값은 새 run의 결과로 다시 기재하지 않고 기존 동기 근거로 사용한다.

기존의 최초 실패/후속 손실, cohort, NLL, paired transition, overwrite 분해를 동기 검증 명목으로 반복하지 않는다. 새 분석에서는 기존 row와 새 telemetry를 identity로 연결한다.

### 3.2 이미 있는 결과의 재사용 범위

| 자료 | 이미 관측한 내용 | 이번에 가져올 근거 | 아직 측정되지 않은 것 |
|---|---|---|---|
| R1/R2: BLUE·native lifelong | AlphaEdit L4/L5의 높은 R 유지, L4의 상대적 N 우위, editor별 layer 순위 차이 | 문제의 크기·위치·기존 비교 기준 | 각 원인의 개별 효과 |
| R3: Fixed-z nonuniqueness | 두 모델·두 editor의 소형 실험에서 같은 local 제약/action을 만족해도 기능적 결과가 다름 | Local 제약과 action만으로 보존 결과가 유일하게 정해지지 않음 | Lifelong L4에서 유용한 해를 찾을 수 있는지 |
| R4: L4 two-memory v2 | 구조적 mapping energy가 거의 0이어도 실제 KL/NLL은 악화될 수 있음 | Proxy와 실제 출력의 연결, 전체 token 위치 관측 필요성 | Native L4–L8의 signed response와 원인 위치 |
| R5: Multilayer A/B 완료분 | Joint edit와 L8 보정의 finite endpoint, layer별 실제 write, 일부 solver 미수렴 | 추가 layer의 자동 이득을 가정하지 않기, 기여도·수치 오차 기록 재사용 | 수렴한 solver와 공정한 예산 아래의 추가 자유도 |
| R6: Alpha-JV mechanism | History 사용 여부, L8 집중, response alignment와 operating-point 진단 | 계측 방법과 특정 실패 조건 | BLUE singleton의 history 방향 효과 |
| R7: Alpha-JV DEV sweep | 기존 controller의 실제 λ/N/T sweep과 성능 trade-off | 같은 knob sweep을 새 동기 검증으로 반복하지 않기 | 이번 L4 fixed10k 정책의 성능 |

R4의 OS에는 W0 기준 누적 cross term이 없었고 history 구성도 E3 후보와 다르다. 따라서 R4는 M3 실패의 근거가 아니다. R5의 미수렴 결과도 보존 최적해나 intrinsic capacity 부재를 입증하지 않는다.

기존 제안 D1의 global/local·부호·random·covariance 방향 대조는 설계 자산으로 재사용한다. 제안문에 적힌 절차를 완료 실험으로 승격하지 않는다. 자료가 다른 worktree나 로컬 봉인 package에 있으면 원래 repository path, commit/blob/file SHA와 로컬 위치를 함께 기록한다.

## 4. Baseline을 읽는 공통 수학과 주장 범위

### 4.1 표기

W^(n)은 n개 request를 편집한 전체 모델이다. 기존 보고서 W100은 W^(10000)에 해당한다. 아래 W_l은 선택한 FFN down/output projection weight이고, Δ_l은 그 weight의 이번 batch 변화다. S_l=W_l^(n)−W_l^0는 누적 변화다.

K는 writer가 실제 사용한 key matrix, R은 실제 residual matrix다. Context 평균·반복이 들어갈 수 있으므로 K의 column 수를 request 수와 자동으로 같다고 놓지 않는다. C0는 기존 uncentered key second moment, P는 저장된 projector, M은 누적 historical key statistic이다. Batch size B와 혼동하지 않도록 writer의 key-side factor는 F로 표기한다.

### 4.2 실제 writer 식

봉인된 BLUE AlphaEdit source의 구조는 다음과 같다.

\[
G=KK^\top+M,\qquad A=PG+\lambda I,\qquad
A\Delta^\top=PKR^\top,\qquad \lambda=1.
\]

수학적으로 F=A^(-1)PK라 두면 Δ=RFᵀ다. 실제 native 경로는 RHS를 먼저 만든 dense solve를 수행하므로, 따로 계산한 RFᵀ와 FP32 materialized update가 bit 단위로 같다고 가정하지 않는다.

MEMIT의 해당 writer는 기존 source의 covariance-regularized solve를 사용한다.

\[
\Delta_{\mathrm{MEMIT}}
=RK^\top(\lambda_C C_0+KK^\top)^{-1}.
\]

같은 이름의 인자가 패키지에 따라 C0 또는 M을 뜻할 수 있으므로 실제 식·tensor identity로 구분한다. Editor 간 target·regularization·history 정책이 다르며, MEMIT 대 AlphaEdit 차이를 M 하나의 인과 효과로 해석하지 않는다.

Native multi-layer AlphaEdit의 L2=10과 BLUE/singleton의 L2=1 차이는 그대로 보고한다. BLUE L4+L8은 L4 write 후 바뀐 모델에서 L8 local z를 다시 계산하고 전체 residual을 쓰므로, 하나의 edit 부담을 절반씩 나눈 대조가 아니다.

### 4.3 관측할 연결

\[
\text{target 생성}\ \longrightarrow\ R,K,P,M
\ \longrightarrow\ \Delta
\ \longrightarrow\ \{\Delta k_t\}_{t}
\ \longrightarrow\ \Delta\text{margin}
\ \longrightarrow\ \text{후속 손실}.
\]

단일 FFN down projection만 바꾸면 고정된 teacher-forced sequence에 대한 해당 module 입력 key는 어느 layer에서도 고정된다. L4만 안정적이고 L8-only는 key가 drift한다는 설명은 제외한다. Context 재생성·입력 변경·추가 parameter 변경은 별도 조건이다.

같은 weight가 모든 위치에 적용되므로 subject token의 Δk만으로 최종 N 반응을 설명할 수 없다. 다음 layer의 attention은 앞선 위치의 변화도 전달한다. 원하는 margin m에 대해

\[
g(x)=\langle\nabla_{W_l}m(x),\Delta_l\rangle
\]

를 측정하되, 이는 true/new 각 teacher-forced sequence의 관련 전체 위치를 포함한 일차 반응이다. 실제 m(W+Δ)−m(W)와 별도로 기록한다.

Layer별 W^(n)은 서로 다른 과거 trajectory를 포함한다. 같은 n에서의 차이는 그 trajectory에 조건부인 차이다. 서로 다른 n은 다음 request batch도 다르므로 n 효과를 age만의 효과로 읽지 않는다.

## 5. 환경·정보·평가 계약

### 5.1 재현 환경

- Llama3-8B-Instruct revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2.
- 기존 FP32 parameter, tokenizer, eager 경로, evaluator microbatch/padding/position 처리, contexts 정책을 유지한다.
- 주 singleton layer는 {4,5,6,7,8}, entry n은 {0,1000,5000,9000}, B=100이다.
- 실제 source/config/asset identity는 기존 manifest에 결속한다. FP16/BF16 전환, solver 교체, projector 재생성을 같은 비교에 조용히 섞지 않는다.
- 각 branch의 target cache는 entry weight·layer·request·context identity에 결속한다. Branch 사이에 case ID만 같은 z cache를 재사용하지 않는다.

### 5.2 방법이 읽는 정보와 진단이 읽는 정보를 분리한다

| 조건 | 방법이 사용하는 정보 | 비용·해석 |
|---|---|---|
| I0: native 정보 | 현재 request/compute-z contexts, native P/C0/M, W0 및 현재 weight로 계산한 S | 주 방법 후보의 기본 조건 |
| I1: historical replay 추가 | I0 + 과거 request·현재 유효 target·선택한 historical responses | Replay 저장량·forward/backward 비용 추가 |
| I2: preservation calibration 추가 | I0 또는 I1 + 독립 factual/general bank·reference | 추가 데이터와 reference 사용을 명시 |

Historical R/P/N와 general panel을 진단 목적으로 읽는 것은 가능하다. 이를 runtime update 선택·거부·step 결정에 쓰면 해당 정보 조건을 방법 표에 반영한다. 진단 gradient를 저장했다고 I0 방법이 그 gradient를 사용할 수 있는 것은 아니다.

공식 평가 P/N prompt와 성공 여부는 target 최적화, calibration-bank 구성, runtime 후보 선택에 사용하지 않는다. General diagnostic panel을 objective나 hyperparameter 선택에 쓰면 I2로 표시하고 별도 평가 panel을 둔다.

이미 본 fixed10k는 탐색 자료다. 이 결과를 보고 연구 방향이나 후보를 선택할 수 있지만, 이후 동일 문항을 다른 order로 평가해 blind test라고 부르지 않는다. 최종 후보·hyperparameter는 별도 개발 자료에서 봉인한다.

### 5.3 Metric

R/P 및 historical edited R/P의 desired margin은 NLL_true−NLL_new, N의 desired margin은 NLL_new−NLL_true다. 성공은 desired margin>0이며 tie는 실패다. 기존 raw margin의 부호를 바꾸지 않고 derived column을 추가한다.

주 집계는 기존 current, cumulative, at-write→final, cohort loss/recovery, paired NLL을 재사용한다. 추가 기록은 실제 ΔW·ΔWK·signed response, 정보 조건, 추가 메모리와 연산 비용이다.

주표는 전체 요청의 원래 분모를 유지한다. NS 개선은 true NLL과 competing-new NLL의 변화를 따로 설명하고 TF strict와 별도 생성/일반능력 평가를 보조로 사용한다. NLL-pair 성공, TF strict, 자유 생성 정확도를 같은 지표로 부르지 않는다.

### 5.4 Overwrite·중복·시간축

Fact identity는 가능한 canonical subject–relation과 request version으로 정의한다. Prefix마다 active/superseded 상태를 기록한다. 현재 유효 target의 retention과 폐기된 target의 교체를 분리하며, 모든 과거 target을 동시에 만족시키는 제약을 만들지 않는다.

기존 원분모 점수와 overwrite strata는 그대로 재사용한다. Active-fact 점수는 별도 보조 표다. 식별 가능한 N/edit 중복도 기존 규칙으로 표시하며 완전한 semantic overlap 검출을 선행 조건으로 요구하지 않는다.

Calibration 보호 대상이 현재 stream에서 정당하게 편집되면 그 시점에 reference를 제외/갱신한다. 미래 stream 전체를 미리 알아야 가능한 bank 갱신을 online I0/I1 정책으로 표현하지 않는다.

기존 BLUE/singleton historical R/P/N는 12개 checkpoint에서 관측됐다. Checkpoint 사이 실패→회복은 보이지 않을 수 있다. 기존 자료의 survival은 “최초 관측 실패 checkpoint”로 정의하거나 checkpoint별 retained/lost/recovered 곡선으로 보고한다. 정확한 최초 실패 시간을 소급해 만들지 않는다.

## 6. 기존 코드 재사용 지도

| 자산 | 재사용할 부분 | 분리하거나 추가할 부분 |
|---|---|---|
| C1: alpha_backend | Canonical FP32 solve·residual·dtype 검사 | Covariance라는 이름의 M 인자 식별, 기존 실행 wrapper 분리 |
| C2: alphaedit_factors | Dense/factor 대조·projection leakage·기존 CPU oracle | 전체 history columns를 요구하는 경로의 메모리 비용 |
| C3: motivation hooks | ForwardCapture, tensor hash, snapshot·clone/copy rollback | Alpha native 적용 순서 유지; MEMIT 전용 materialization과 구분 |
| C4: method TorchCheckpoint | Selected weight와 Python/NumPy/Torch/CUDA RNG 복원 검사 | M·P binding·contexts·next index를 추가로 복원 |
| C5: derivatives | Activation×output-gradient contraction, scalar-gate oracle | Desired margin을 scalar로 연결; signed derivative 보존 |
| C6: diagnostic_math | Covariance 내적·norm·cosine, finite difference | 기존 routing utility·smooth event를 metric으로 승계하지 않기 |
| C7: fixed_z_nonuniqueness algebra | Basis 구성·tangent projection·deterministic seed | Rank-one 가정의 B100 확장; 0 leakage를 오류로 처리하는 검사 수정 |
| C8: strength_neutral geometry | Factor/native 차이와 backward error 계측 | History symmetrization·연산 순서 차이 기록; barrier projection 제외 |
| C9: 봉인된 BLUE metrics | Preferred bit, 원본 margin 검증, 집계 | 새 telemetry와 identity join |

C5의 progress_slope=max(0,−derivative)는 ODE용 값이므로 E1에서는 signed event_derivative를 저장한다. C8의 별도 evaluator는 전후 argmax 보존을 locality로 쓰므로 이번 NS evaluator로 대체하지 않는다.

기존 dense oracle, hook cleanup, rollback, factorization 테스트를 우선 사용하고 새 adapter와 정보 분리에 필요한 검사만 추가한다. 구현의 연결은 “native singleton 실행 → 실제 tensor/activation 관측 → signed response → 기존 BLUE 집계”다.

## 7. E0 — 기존 state에서 정확히 분기할 수 있는가

목적은 기존 성능의 재감사가 아니라 새 비교의 출발점 검증이다. 복원 검증은 한 묶음으로 처리한다.

1. Source/model/tokenizer/sample/P/C0와 checkpoint의 selected weights·M·contexts·RNG·다음 batch index를 결속한다.
2. W0 초기화와 저장 checkpoint 복원 경로를 구분한다. 저장 contexts를 W0에 주입할 때 원래 context 생성이 소비한 RNG 상태도 맞춘다.
3. No-op roundtrip의 weight/history bytes와 hook/cache cleanup을 확인한다.
4. 고정 prompt panel의 반복 forward를 통해 수치 변동을 측정하고 비교 허용오차를 먼저 정한다. 결정적인 solver 오차를 forward noise로 정당화하지 않는다.
5. 다음 B100을 original singleton으로 진행해 기존 target, endpoint/history hash, update norm, current NLL/metric과 비교한다.
6. 다음 저장 endpoint까지 이어가 weight/history tensor 오차를 비교한다. 1k→2k, 5k→6k, 9k→10k가 해당한다.

먼저 L4/L8에서 공통 loader를 확인하고 L5–L7의 asset mapping을 확인한다. 기존 CPU reload 기록을 GPU continuation 검증으로 대체하지 않는다.

B11/B51/B91에는 full weight checkpoint가 없다. 한 batch의 exact hash가 일치하지 않을 때 norm만으로 fidelity를 선언하지 않는다. 다음 저장 endpoint의 tensor 비교를 사용한다. C0는 별도 native asset이며 checkpoint만으로 복원됐다고 가정하지 않는다.

입력 identity나 실제 update 적용이 잘못되면 해당 비교를 고친다. 원래 trajectory 재현이 확인되지 않은 branch를 그 trajectory의 continuation 결과로 보고하지 않는다. 독립적인 CPU 분석은 계속할 수 있다. 다른 출발점이 필요하면 별도 fresh-trajectory 실험으로 정의하며 원래 n의 대체 자료로 섞지 않는다.

산출물: resume_fidelity.json. 복원한 상태, 비교 가능한 수준, 수치 차이, 미측정 항목을 함께 기록한다.

## 8. E1 — Baseline 차이의 관측과 최소 계측

### 8.1 E1-A: 기존 per-case 자료를 연결한다

기존 current.json과 checkpoint seen-full.json을 case_id·prompt_index·prompt/target identity로 join한다. Relation, subject, request order, true/new token length, overwrite 상태를 기존 sample metadata에서 붙인다. Aggregate CSV만으로 per-case 결합분포를 복원하지 않는다.

분석용 ledger는 원시 자료가 있는 환경에서 만들고 큰 row 파일은 local 영역에 둔다. 원문 prompt를 포함하지 않는 compact summary와 identity manifest를 공유한다.

AlphaEdit와 MEMIT의 기존 singleton 자료를 함께 사용한다. 새 10k run 없이 다음을 비교한다.

- 같은 case의 at-write/final margin·NLL과 손실·회복.
- 공통 at-write margin 구간 안에서 age·relation·token length를 고려한 retention.
- L4/L5 대 L6–L8의 R 유지와 L4 대 L5의 P/N 차이.
- 전체 population과 common-success subset, active/overwrite strata.

Matching은 관찰 분석이다. At-write margin은 layer 선택 이후의 변수이므로 gap 감소를 headroom의 인과 기여율로 읽지 않는다. 공통 지지 구간·제외율·bin 표본 수를 공개한다.

산출물: margin_age_matched_retention.csv, paired_case_ledger_manifest.json.

### 8.2 E1-B: 20개 native replay cell

기본 범위는 5 layers × 4 entries × 다음 B100 한 번 = 20 cells다. E0에서 같은 조건으로 수집한 자료는 재사용한다. 새로운 singleton 10k chain 5개를 시작하지 않는다.

각 cell의 forward panel은 다음과 같다.

| Panel | 구성 | 용도 |
|---|---|---|
| Current | 다음 100 requests의 R/P/N | 현재 효능·일반화·추가 locality 손상 |
| Historical | Seen requests에서 고정 층화한 128 requests의 R/P/N | 과거 유지와 간섭 |
| General | 사전에 고정한 별도 128 text sequences | 일반 text의 key/출력 변화 |

Historical은 seen ordinal의 early/middle/recent 세 구간에서 43/43/42개를 고정 hash 순서로 뽑는다. 성능에 따라 뽑지 않으며 동일 entry의 layer 비교는 동일 case를 사용한다. n=0에는 historical panel이 없다. Active/superseded 상태를 보존하고 보고 시 분리한다.

R/P/N가 request당 1/2/10 prompts인 기존 규격에서 n>0의 full panel은 2,964 prompt pairs, true/new 합계 5,928 candidate sequences다. General은 별도다. W0·entry·native endpoint 평가와 α=1 결과는 identity가 같을 때 재사용한다.

Backward 계측은 우선 L4/L8 × n={5000,9000}의 4개 대표 cell에서 한다. 예시 subpanel은 current R 32, historical R 32, historical P 32, current N 32, historical N 32, general 16이다. 같은 hash 규칙으로 봉인한다. 신호를 확인할 필요가 있는 layer/entry에만 확대하고 최종 coverage를 보고한다. General의 scalar는 별도 NLL이며 R/P/N의 desired margin과 합쳐 집계하지 않는다.

128개 sequence나 작은 backward subpanel의 p99를 안정적인 population tail로 해석하지 않는다. 분위수에는 prompt·request cluster 수를 함께 붙인다.

### 8.3 함께 수집할 네 가지 연결

**① Target 상태와 write 실현**

- Native compute-z의 최종 training NLL, 실제 반복 수, 종료 이유, clamp 도달 여부.
- Writer가 사용하는 R/K의 context 구성·평균·반복 규칙.
- 실제 materialized ΔW의 norm·relative norm, intended solve와의 재구성 오차.
- \(\|\Delta WK-R\|\) 및 context별 실제 post-write activation 차이.
- 실제 at-write R/P margin과 NLL. Absolute z를 다른 layer에 교환하지 않는다.

Compute-z가 끝난 상태, writer의 local residual fitting, 최종 답변의 효능을 구분한다. 작은 local 오차가 높은 최종 효능을 보장한다고 가정하지 않는다.

**② 어떤 입력이 write에 노출되는가**

Native factor F에 대해 query q의 Fᵀq, 실제 ΔWq, 이후 margin 변화를 분리한다. Current/history/N/general별 raw key overlap과 writer-weighted overlap을 기록한다. Factor 정의와 scaling은 native 식으로 고정한다.

Teacher-forced true/new sequence의 전체 위치를 계측한다. Subject 위치, prefix의 다른 위치, continuation 관련 위치를 분리하되 전체 효과와 함께 보고한다. 고정 입력·singleton 조건의 key는 W0에서 재사용할 수 있다. 재사용 전 token IDs·position IDs·mask·contexts·module identity를 확인한다.

**③ Projector와 history의 geometry**

- P의 rank, symmetry/idempotence 오차와 저장 covariance의 결속.
- C0의 전체 spectrum 요약과 허용 공간 C_U=UᵀC0U의 spectrum·비등방성.
- \(\|Pk\|^2/(\|k\|^2+\epsilon)\).
- Projected history의 spectrum, effective rank, current/history/N overlap.
- Reduced symmetric system과 실제 native system의 conditioning·solve residual.
- 이번 Δ와 누적 S의 covariance leakage 및 두 변화 사이의 signed 내적.

큰 행렬의 정확한 분해 비용을 별도 기록한다. 근사 spectrum을 사용하면 알고리즘·rank·residual을 표시하고 exact rank/condition number로 이름 붙이지 않는다.

**④ 같은 hidden 변화가 출력에서 얼마나 위험한가**

Signed \(g(x)=\langle\nabla m,\Delta\rangle\), 작은 α의 finite difference, α=1의 실제 margin 변화를 비교한다. 작은 α는 반복 forward의 수치 해상도를 확인해 공통 값으로 정한다.

Hook contraction으로 dense weight gradient 저장을 피할 수 있지만 true/new의 부호와 전체 위치는 유지한다. Native factor 근사에 의한 derivative와 실제 materialized Δ에 대한 차분의 차이도 기록한다.

Hidden perturbation이 작고 margin 손실이 큰 경우 sensitivity/direction을 조사한다. 실제 native 방향 하나의 derivative만으로 layer의 전체 민감도나 intrinsic capacity를 확정하지 않는다.

산출물: native_target_write.csv, writer_interference.csv, projected_geometry.csv, signed_response.csv, instrumentation_cost.csv.

## 9. E2 — 관측된 원인에 필요한 최소 개입

기본 screening은 L4/L6/L8, n={1000,5000,9000}의 9개 entry다. E1에서 L5가 필요한 경계 대조이면 L5로 확대한다. 모든 개입의 전 조합을 실행하지 않는다. 같은 entry의 비교는 W·request·contexts·z·변경하지 않는 native 설정을 공유한다.

### 9.1 E2-A: Update 크기 — 기본 실행

Native Δ에 대해 α={0.25,0.5,0.75,1.0}의 endpoint를 평가한다. Z는 entry/layer별 한 번 계산한 값을 공유한다. α=1은 E1에서 재사용하므로 9 cells의 새 축소 endpoint는 최대 27개다.

예시 online 선택 규칙은 가장 작은 feasible α를 고르는 것이다. Feasible 조건의 초기 제안은 다음과 같다.

- 현재 canonical R 성공 수가 native보다 1/100을 초과해 줄지 않는다.
- Native에서 성공한 현재 R의 새 실패도 1/100 이내다.
- Request별·context별 평균을 맞춘 current training target-new NLL 증가가 0.1 nats/token 이내다.

이 수치는 개발 단계 제안이다. 별도 개발 자료에서 봉인하고 P/N/historical 결과로 α를 선택하지 않는다. Feasible한 축소가 없으면 α=1을 사용한 비율도 보고한다. Margin 분포와 NLL tail은 별도로 남긴다.

단발 α sweep은 규모 효과를 관측한다. 이후 continuation은 두 질문을 구분한다.

| 비교 | 후속 정책 | 질문 |
|---|---|---|
| One-off intervention | 첫 batch만 축소, 이후 branch state에서 native compute-z/write | 초기 write 축소의 효과가 오래 남는가? |
| Online shrink method | 매 batch 같은 current-only 선택 규칙 적용 | 축소 정책이 lifelong에서도 유용한가? |

One-off continuation은 신호가 있는 대표 cell부터 시작한다. Online shrink는 E3의 방법 후보로 검증한다. 두 결과를 같은 실험으로 합치지 않는다.

### 9.2 E2-B: Projector rank와 ridge — geometry 신호가 있을 때

각 layer 고유 covariance의 작은 eigenvalue 방향을 선택해 공통 rank r*를 만든다. 예시는 기본 rank의 최솟값이다. Rank 0 또는 퇴화 조건은 별도로 보고한다. 다른 layer의 projector tensor를 복사하지 않는다.

대조 순서는 native P/native ridge → common-rank P/native ridge → 필요할 때 projected scale을 맞춘 ridge다. P 재구성 자체의 수치 차이를 보려면 native rank를 같은 절차로 재구성한 sham도 둔다.

Rank·ridge 변경이 current 효능도 바꾸므로 가능한 공통 efficacy 영역의 frontier를 비교한다. 같은 rank가 같은 표현 공간이나 capability를 뜻하지는 않는다.

### 9.3 E2-C: History의 크기와 방향 — historical interference 신호가 있을 때

같은 W,z,P에서 projected history의 spectrum을 보존한 방향 개입을 한다. P의 orthonormal range U와 seeded orthogonal O를 사용한다.

\[
T_O=UOU^\top+(I-UU^\top),\qquad
M_{\mathrm{rot}}=T_OMT_O^\top.
\]

이 정의는 O=I에서 원래 M을 보존하며, 정확한 산술에서 full M의 spectrum/PSD와 projected history spectrum을 보존한다. O=I sham과 최소 3개의 사전 고정 rotation seed를 사용한다. Transform 생성 방식·계산비용을 기록한다.

기존 제안의 projected-only \(UO(U^\top MU)O^\top U^\top\)도 정확한 projector 아래에서는 native 해를 비교할 수 있지만, O=I에서 off-block가 제거된다. 이 형태를 쓰면 native M 대 projected M의 별도 sham을 먼저 비교한다.

각 arm의 actual update 크기·현재 효능·historical perturbation을 함께 보고한다. 회전이 훨씬 큰 update를 만든 결과를 방향만의 효과라고 부르지 않는다. Optional M=0은 추가 대조다.

이 회전은 mechanism 개입이다. History rotation을 최종 방법 후보로 올리지 않는다.

### 9.4 E2-D: Target 생성과 write 실현 — E1에서 잔차 신호가 있을 때

같은 layer·entry·writer에서 native compute-z의 한 설정만 바꾼다. 첫 소형 대조는 native 최대 25 steps 대 50 steps이며 다른 loss·clamp·contexts는 고정한다. Native가 이미 같은 지점에서 종료하면 추가 효과가 없다는 관측으로 남긴다.

Clamp 포화가 주요 신호일 때는 step budget 대조와 분리한 clamp 한 항목의 대조를 개발 자료에서 정의한다. P/N를 보고 request별로 z를 선택하지 않는다.

비교할 것은 target-side training loss, R, 실제 local fitting error, actual Δ, 최종 R/P/N이다. Target-side loss 개선이 실제 write 개선으로 이어지는지 확인한다. Layer 간 absolute z 교환이나 새 target controller는 사용하지 않는다.

### 9.5 E2-E: 전체 token 전달 — 구조 proxy와 출력이 어긋날 때

E1의 full-position response에서 신호가 있으면 소형 diagnostic panel에서 위치별 activation intervention을 한다. 같은 native Δ의 perturbation을 전체 위치, subject/선택 위치, 나머지 prefix 위치에 각각 적용해 실제 margin 반응을 비교한다.

전체 위치 intervention은 실제 dense update의 forward와 허용오차 내에서 일치하는지 먼저 확인한다. Teacher-forced true/new의 mask·position을 명시한다. 부분 위치 intervention은 deployment method가 아닌 원인 진단이며 I0 후보로 보고하지 않는다.

개별 위치군 효과의 합과 전체 효과가 다르면 비선형 interaction을 함께 기록한다. Subject 위치 보호만으로 충분한지, 어떤 위치의 전달이 빠졌는지를 묻는다.

### 9.6 E2 결과의 해석

| 관측과 개입 결과 | 다음 방법 분기 |
|---|---|
| 축소만으로 효능·retention을 유지하며 N 개선 | Current-only scalar shrink |
| Rank/ridge 통제가 writer 비용과 layer gap을 바꿈 | Projector 또는 solve regularization 개선 |
| Spectrum을 맞춘 history 방향 개입이 과거 간섭을 바꿈 | History weighting·보호 공간 개선 |
| Target 개선 또는 local fitting 변화가 최종 효능/손상을 바꿈 | Target 생성 또는 native writer 실현 개선 |
| Key perturbation과 구조 penalty가 출력 손상을 설명하지 못함 | Functional signal·추가 정보 필요성 검토 |
| 과거 누적 변화와의 signed 정렬이 축소 이상의 효과를 보임 | 누적 목적함수 M2/M3/M4 비교 |

모든 원인을 설명해야 E3로 갈 수 있는 것은 아니다. 재현되는 문제와 이를 바꾸는 개입을 확보하면 해당 방법 분기로 진행한다. 신호가 없는 항목을 결과 후 주 mechanism으로 승격하지 않는다.

산출물: amplitude_frontier.csv, geometry_interventions.csv, target_write_interventions.csv, position_response.csv, mechanism_decision.md. 실행하지 않은 조건부 항목은 미실행 이유를 남긴다.

## 10. E3 — 확인된 원인에서 최소 방법을 선택한다

### 10.1 방법 선택 원칙

후보는 E2 결과에서 고른다. 처음부터 M2–M4 또는 multi-layer 보정으로 고정하지 않는다. 효과가 비슷하면 추가 정보·메모리·연산과 변경 항목이 적은 후보를 우선한다.

방법 계약에는 바꾸는 한 구성요소, 읽는 정보, 계수 선택 규칙, native와 공유하는 항목, 실패 시 동작을 명시한다. 고정된 방법에 맞도록 diagnosis panel이나 설명 가설을 바꾸지 않는다.

E3의 첫 screen은 L4의 네 entry에서 한다. 같은 entry의 writer 변형은 같은 local z를 사용한다. Target 생성 자체를 바꾸는 후보는 별도 target-change arm으로 표시한다.

최대 두 후보를 선택해 n={1000,5000,9000}에서 다음 1,000 requests를 각자의 state로 진행한다. 이후 batch의 z는 해당 branch에서 계산한다. 동일 upstream·입력·contexts의 L4-only 후보들은 history append statistic의 일치도 검증한다.

Native + 2 후보 × 3 entries × 10 batches = 90 batch-equivalent다. 신규 후보 continuation은 60 batches, 6,000 arm-request executions이며 고유 6,000 requests를 뜻하지 않는다. E0에서 충실한 재개가 확인된 native evidence는 재사용한다.

### 10.2 조건부 방법 분기: 누적 locality 목적함수

누적 정렬/상쇄가 유용하다는 신호가 있을 때 아래 family를 연다.

| 후보 | 목적함수 | 비교 질문 |
|---|---|---|
| M0 | 실제 native L4 | 기준 |
| M1 | Current-only scalar shrink | 크기 변화로 충분한가? |
| M2 | J_native(Δ)+γ q_C(Δ) | 이번 leakage 억제의 효과 |
| M3 | J_native(Δ)+γ q_C(S+Δ) | 과거 변화와의 정렬/상쇄가 추가로 유용한가? |
| M4 | J_native(Δ)+γ‖S+Δ‖²_F | Covariance 방향 정보가 필요한가? |

\[
q_C(A)=\operatorname{tr}(AC_0A^\top),\qquad
q_C(S+\Delta)=q_C(S)+q_C(\Delta)
+2\operatorname{tr}(\Delta C_0S^\top).
\]

q_C는 layer-output surrogate다. Old edits를 지우는 것을 보존 개선으로 채택하지 않는다. 이 family에서도 native history 보호를 유지한다.

정확한 P=UUᵀ 아래에서 Δ=TUᵀ, K_U=UᵀK, M_U=UᵀMU라 두면

\[
J_{\mathrm{native}}(T)=\|TK_U-R\|_F^2+
\operatorname{tr}(TM_UT^\top)+\lambda\|T\|_F^2,
\quad H=K_UK_U^\top+M_U+\lambda I.
\]

C_U=UᵀC0U에 대한 해는 다음과 같다. 아래 γ는 normalization을 적용한 실제 계수로 읽는다.

\[
T_2=RK_U^\top(H+\gamma C_U)^{-1},
\]
\[
T_3=(RK_U^\top-\gamma SC_0U)(H+\gamma C_U)^{-1},
\]
\[
T_4=(RK_U^\top-\gamma SU)(H+\gamma I)^{-1}.
\]

구현 조건은 다음과 같다.

- C0는 uncentered moment로 유지한다. M2/M3는 동일 coefficient normalization을 사용한다.
- 전체 C0 trace 대신 C_U와 H의 상대 scale을 기록한다. γ 후보 예시는 dimensionless {0.01,0.1,1.0}이며 실제 normalization은 개발 자료에서 봉인한다.
- C_U=0이면 M2/M3가 무효인 조건을 보고한다. 고정 P 안에서 C_U가 상수배 identity에 가까우면 ridge/M4 대조와의 중복을 확인한다.
- \(\|\gamma H^{-1/2}C_UH^{-1/2}\|\)와 \(\gamma\|SC_0U\|_F/(\|RK_U^\top\|_F+\epsilon)\) 등 실제 벌점 세기와 cross term 크기를 기록한다.
- n=0에서 같은 γ의 M2=M3, γ=0에서 native와 일치하는지 확인한다. γ=0에서만 기존 함수로 우회해 새 solver의 동치를 주장하지 않는다.
- P의 symmetry/idempotence, 새 U로 재구성한 P 차이, solve residual, actual endpoint·ΔWK 차이를 확인한다.
- M3는 과거 S의 출력 방향을 추가할 수 있다. Current R의 span에 강제로 제한하지 않고 actual/effective rank·메모리를 보고한다.

동일한 full T 공간에서 M2/M3를 정확히 풀면 M3의 rank 증가는 cross term의 효과에 포함된다. 같은 current-output span에서 signed cancellation만의 효과를 주장하려면 SC0U를 col(R)에 투영한 별도 대조가 필요하다.

M1≈M3이면 누적 목적함수의 필요성은 약하다. M4≈M3이면 covariance 방향의 고유한 기여를 주장하지 않는다. q_C는 줄지만 NS가 개선되지 않으면 surrogate mismatch를 기록하고 같은 penalty의 무한 retuning으로 넘어가지 않는다.

M2의 leakage/history 구성은 BetaEdit과 겹친다. M3의 식이 다르다는 이유만으로 novelty를 확정하지 않는다. [P2]

## 11. E4 — 추가 layer의 보존 자유도는 필요할 때만 확인한다

### 11.1 실행할 이유와 정보 조건

개선된 L4의 한계가 남거나, 같은 edit 성능 아래 locality에 유용한 다른 방향의 존재를 물어야 할 때만 연다. E3의 어느 후보가 좋았다는 이유만으로 자동 진행하지 않는다.

Track A는 I0의 native moments만 사용하는 surrogate 실험이다. Historical semantic/NLL을 solver 제약으로 사용하면 I1으로 바뀐다. M에는 과거 prompt·target·response gradient가 없으므로 M-only 조건에서 그 제약을 계산할 수 없다.

Track B는 historical edit replay와 독립 preservation calibration을 사용하는 I1+I2 실험으로 정의한다. 예시 bank는 calibration 512, 독립 validation 512 prompts이며 fact/subject 분리 규칙을 사전 고정한다. 공식 P/N는 bank에 쓰지 않는다.

Reference는 미편집 지식에는 W0 또는 검증된 정답, 과거 edit에는 현재 유효 target이다. 매 batch의 손상된 W^(n)을 유일한 teacher로 삼아 기존 손상을 정상 상태로 고정하지 않는다.

### 11.2 목적함수의 관측 가능성

Track A에서 층별 q_C의 합만 최소화하면, L4가 고정된 L8 correction T8의 목적은

\[
q_{C_4}(S_4)+q_{C_8}(T_8)
\]

이고 T8=0이 최소다. 이 목적은 L8이 L4의 최종 출력 손상을 상쇄하는 이득을 볼 수 없다. 따라서 cross-layer preservation objective와 그 계산에 필요한 정보를 명시하기 전에는 P2의 compensation 실험을 실행 정의가 끝난 것으로 취급하지 않는다.

현재 기본 경로는 Track B에서 실제 preservation response를 최적화하고 current/historical edit response를 제약·검증하는 것이다. Track A에서는 local surrogate에서 관측 가능한 질문만 다루며, 그 실패를 추가 layer 자유도 부재로 읽지 않는다.

### 11.3 공정한 비교

| 후보 | Correction 위치 | 공통 L4 native edit | 총 input dictionary rank 예시 |
|---|---|---|---|
| P0 | 없음 | 고정 | 0 |
| P1 | L4 | 같은 endpoint에서 correction | 64 |
| P2 | L8 | L4 weight 고정 | 64 |
| P3 | L4+L8 | 같은 endpoint에서 joint correction | 32+32 |

Output dimension이 같을 때 free coefficient 수를 맞춘다. Dictionary 정보원·정규화·projection·optimizer·step budget·seed 정책을 맞추고 실제 coefficient 수·effective rank·wall-time을 보고한다. 필요하면 P1 rank128 대조로 단일 layer의 좁은 탐색 예산을 구분한다.

G_E의 null directions와 preservation response의 변화를 측정할 때는 edit-response map의 단위와 row 구성, effective constraint rank, direction norm을 함께 기록한다. 큰 parameter 공간에 null direction이 존재한다는 사실만으로 유용한 보존 여력을 주장하지 않는다. Finite update에서 실제 응답을 검증한다.

n=5000/9000의 static correction부터 시작한다. 이후 “한 번 correction 후 L4-only 10 batches”와 “매 batch correction policy”를 구분한다. 전자는 복구 효과의 지속성, 후자는 반복 정책이다.

L4-only trajectory의 native M4를 다른 trajectory의 M8와 혼합하지 않는다. L8 correction을 반복하면 후속 L4 write가 K8를 바꾸므로 M8의 출처, 재계측 또는 stale-history 근사와 비용을 명시한다.

P1의 성공은 L4 안의 더 나은 해를 지지한다. P2/P3만 성공하면 해당 정보·dictionary·예산 아래의 추가 자유도를 지지한다. 모든 후보의 실패는 현재 search/solver/reference의 실패이며 intrinsic capacity 부재의 증명이 아니다.

기존 R5의 대칭 signed layer attribution과 수치 residual 기록을 재사용한다. Static/functional projection과 sequential preservation은 CrispEdit 등과도 겹치므로 layer별 원인 분리와 실제 lifelong 결과에서 기여를 검토한다. [P3]

## 12. E5 — 선택한 방법의 lifelong 검증

### 12.1 실행과 비교 범위

1. E3/E4에서 선택한 최대 두 후보의 policy·hyperparameter·정보 조건을 봉인한다. 확증의 primary 후보는 test 전에 하나로 정하거나 다중 비교 처리를 미리 명시한다.
2. W0부터 B100×100을 실행한다. Suffix 시작점의 이득만으로 full lifelong 성능을 주장하지 않는다.
3. 최소 세 request orders에서 native와 paired 비교한다. 같은 fact 내부의 수정 순서를 보존하는 permutation을 주 order 실험으로 사용한다.
4. 별도 subject/fact-disjoint edit set이 있으면 새로운 사실로의 일반화를 확인한다. 해당 자료가 없으면 결과 범위를 fixed-corpus 재현과 order robustness로 한정한다.
5. 두 번째 모델에서 확인할 경우 layer index와 projector 설정을 독립 개발 자료에서 정한다. Llama의 L4를 자동으로 옮기지 않는다.

기본 baseline은 AlphaEdit BLUE-style L4-only와 BLUE L4+L8이다. 선택된 방법의 구성과 정보 조건에 따라 가장 가까운 선행방법을 추가한다. Native multi-layer와 MEMIT은 기존 진단 대조이며 모든 방법·모델·layer 조합을 처음부터 10k로 실행하지 않는다.

### 12.2 제안된 engineering target

별도 개발 자료에서 아래 값을 검토·봉인한다. Test 결과를 보고 완화하지 않는다. 이는 자연법칙이나 통계적으로 도출된 최적 허용폭이 아니다.

| 지표 | Paired 판정 제안 |
|---|---|
| Final RS | 후보−native의 95% confidence interval 하한 > −0.5pp |
| Final PS | 같은 하한 > −1.0pp |
| Final NS | 점추정 개선 ≥3.0pp이면서 95% confidence interval 하한 >0 |

“NS의 신뢰구간 하한 자체가 +3pp 이상”인 더 강한 주장은 별도로 구분한다. 기존 L4 수치에 적용한 descriptive target은 RS≥98.890%, PS≥94.680%, NS≥68.348%이지만, 새 order의 판정은 그 order의 paired native를 기준으로 한다.

현재 99.390% RS에서 0.5pp 허용은 final 실패가 61개에서 111개까지 늘어날 수 있음을 뜻한다. 따라서 성공률만 제시하지 않고 실제 실패 수와 active historical loss를 함께 보고한다.

추가 필수 보고:

- Historical R/P의 active-fact 손실·회복과 overwrite strata.
- Historical N retention, current N의 entry→post 추가 손상, W0→final 변화.
- Target-new NLL과 margin의 평균·하위/상위 tail, NS true/new NLL 두 항.
- TF strict 및 별도 생성/일반능력 평가, 정보 조건, wall-time, 추가 메모리.

Tail이나 개별 cohort의 “큰 악화 없음”을 추가 hard 채택 조건으로 사용할 때는 해당 metric·분모·허용폭을 개발 단계 방법 계약에 수치로 적는다. 결과 후 주관적인 조건을 추가하거나 좋은 집단만 보고하지 않는다.

### 12.3 불확실성의 단위

Request/subject cluster bootstrap은 고정된 edited trajectory에서 평가항목 변동을 추정한다. 이를 새 order의 학습 trajectory 변동까지 추정하는 confidence interval로 표현하지 않는다.

각 order의 paired Δ와 조건부 confidence interval을 별도로 공개한다. Order 세 개의 평균·범위도 보고하되 order 일반화의 정밀도를 과장하지 않는다. N prompt 100,000개는 독립 run 100,000개가 아니다.

## 13. 실행 묶음과 비용

| 묶음 | 기본 범위 | 기존 자료 활용 | 추가 비용 |
|---|---|---|---|
| A: 근거·입력 준비 | 기존 14 arms와 필요한 per-case join, E0 | 기존 감사·metric·CP manifest | CPU join, 복원·continuation 확인 |
| B: Baseline 계측 | E1 20 native cells, backward 우선 4 cells | 기존 endpoint·z·key의 identity 검증 후 재사용 | Capture·panel·gradient·spectrum |
| C: 최소 개입 | E2-A 9 cells, 나머지는 신호별 선택 | α=1·같은 z·entry 결과 | 최대 27개 축소 endpoints, 선택한 개입 |
| D: 방법 검증 | 최대 2 후보, 3 entries의 1k suffix | Fidelity가 확인된 native suffix | 신규 60 batches + candidate screen |
| E: 확장·확증 | 필요시 E4, 선택 후보 E5 | 관련 baseline·완료 진단 | 정보 bank·추가 layer·full chains |

기존 batch-cost.csv의 해당 20 cells 원래 edit 시간 합은 약 5,100초, 1.42시간이다. L4 세 suffix의 원래 edit 합은 약 8,507초, 2.36시간이고 evaluator 합은 약 3,321초, 0.92시간이다. 이는 저장된 실행의 참고값이며 새 계측·load·복원·solver 비용을 포함하지 않는다. [R8]

이 합산은 arm ID AlphaEdit_L4_ONLY부터 AlphaEdit_L8_ONLY까지의 batch {1,11,51,91}을 사용한다. 일부 신규 arm의 display label이 비어 있으므로 표시 문자열만으로 포함 여부를 판단하지 않는다.

새 후보 두 개가 native와 같은 edit 시간을 쓴다고 가정하면 60 candidate batches의 edit만 약 4.73시간이다. 후보의 compute-z 난도와 추가 objective 비용은 달라질 수 있으므로 GPU 예약 시간이나 속도 보장으로 쓰지 않는다.

별도 계수할 항목은 model load/restore, compute-z, key capture, solve/factorization, spectrum, forward, backward, evaluation, checkpoint I/O다. Edit 전체와 그 하위 compute-z/solve 시간을 중복 합산하지 않는다. 입력 길이에 따른 token 수·peak memory·총 arm-request executions도 기록한다.

모든 cell에서 전체 Jacobian이나 dense gradient를 저장하지 않는다. 기존 hook contraction과 저차원 factor 연산을 활용하고, 큰 tensor는 local artifact로 보존한다. CPU/GPU 비용이 커지는 항목은 대표 cell 실측 뒤 확대 범위를 정한다.

## 14. 산출물과 연구 결정

| 단계 | 주 산출물 | 답해야 할 질문 |
|---|---|---|
| 근거 재사용 | evidence-reuse-manifest.json | 어떤 결과·source·tensor를 재사용했고 어떤 범위까지 확인했는가? |
| E0 | resume_fidelity.json | 의도한 native state에서 비교하고 있는가? |
| E1 | Per-case 연결, native_target_write, writer_interference, projected_geometry, signed_response | 차이가 target/write/exposure/response 중 어디서 보이는가? |
| E2 | 개입별 paired 결과와 mechanism_decision.md | 어느 변경이 해당 차이를 실제로 바꾸는가? |
| E3/E4 | 방법·정보 계약, suffix/repair 결과 | 그 변경을 허용 정보와 비용으로 구현할 수 있는가? |
| E5 | Order별 final·retention·cost 및 claim boundary | 전체 lifelong과 정해진 일반화 범위에서 이득이 남는가? |

Evidence manifest는 실제 소스 경로·commit 또는 SHA·자료 상태를 기록한다. 기존 보고서가 있다는 것, 원시 artifact가 현재 접근 가능하다는 것, 독립적으로 tensor를 검증했다는 것을 서로 다른 항목으로 둔다.

Mechanism report에는 관측, 최소 개입, 가능한 설명, 배제되지 않은 대안, 선택한 방법 분기를 함께 적는다. 상관·matching·local derivative·finite intervention·full policy 결과를 같은 증거 수준으로 합치지 않는다.

다음 사항은 이번 계획의 완료 조건이 아니다: L4 내부 회로의 완전한 설명, 모든 가설의 기여율 합산, Barrier/ODE의 재도입, 모든 editor/model의 10k 전수 실행. 본 단계의 완료는 재사용 근거와 새 관측을 연결하고, 확인된 문제에 맞는 다음 방법 또는 미해결 질문을 구체적으로 정하는 것이다.

## 15. 근거·코드 링크

아래 링크는 현재 공유 작업공간에서 확인한 위치다. 다른 환경으로 실행을 옮길 때는 같은 문서를 새로 추정해 고르지 않고 manifest의 repository-relative path와 commit/blob/file identity로 재결속한다. 다른 worktree의 중간 보고서는 그 완료 범위와 제한을 유지한다.

### 15.1 기존 결과

- **R1:** [Server4 BLUE·native lifelong 종합 보고서 봉인 복사본](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md). 원래 repository path는 experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md, blob SHA는 0caaa201ae7f0ed27757c26c5ebee24aed50034a.
- **R2:** [BLUE·native lifelong 독립 감사](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md). Final, at-write, W0 transition, overwrite, 비용과 주장 범위.
- **R3:** [Fixed-z nonuniqueness 소형 실험](/mnt/raid5/janghj/ODE-edit/experiment-reports/servers/server4/fixed-z-nonuniqueness-screen-2026-08-31-v1/fixed-z-nonuniqueness-screen-factual-ko.md). Local equality/action과 기능적 비유일성의 완료 근거.
- **R4:** [L4 two-memory v2 독립 감사](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-review-ko.md). Proxy/output 불일치, token coverage, BF/Frozen 결과.
- **R5:** [Multilayer A 중간 보고서](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-multilayer-joint-edit-a-v1/experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/diagnostic-report-ko.md), [N4/A0/B-OS 완료분 비교](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-multilayer-joint-edit-a-v1/experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/peer-comparison-r2/comparison-ko.md). Partial endpoints와 solver 제한.
- **R6:** [Alpha-JV sequential mechanism 감사](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v1/gh-mechanism-review-ko.md).
- **R7:** [Alpha-JV 실제 DEV sweep terminal 보고](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-downstream-report-recall-20260912-v1/experiment-reports/servers/server1/alpha-jv-llama-diagnosis-sweep-2026-09-07-v1/sweep-terminal-v1/factual-report-ko.md). 이전 준비 보고서의 W5/context 부재 상태는 이후 자료 확보로 정정되었으며, exact historical intervention의 미실행 상태와 구분.
- **R8:** [기존 batch별 비용](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/batch-cost.csv), [checkpoint tensor manifest](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/checkpoint-tensors.csv).

### 15.2 재사용할 설계와 구현

- **D1:** [Single-layer cumulative-risk 진단 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-single-layer-cumulative-risk-diagnostic-design.md). 특히 global/local·부호·random 방향 대조와 평가 panel 구성. 제안 자료.
- **D2:** [Multilayer joint edit·compensation 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-11-multilayer-joint-edit-and-compensation-design.md). 조건부 E4와 비교할 이전 설계이며 R5의 실제 완료 범위와 구분.
- **C0-source:** [실제 실행된 BLUE AlphaEdit source](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/runtime-source/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit/AlphaEdit_main.py). Local target·solve·materialization·history 정책의 기준.
- **C1:** [Canonical AlphaEdit FP32 backend](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_bf/alpha_backend.py).
- **C2:** [AlphaEdit factors](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/alphaedit_factors.py), [기존 factor tests](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/tests/test_alphaedit_factors.py).
- **C3:** [Capture·snapshot·rollback hooks](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/hooks.py), [기존 hook tests](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/tests/test_hooks_artifacts.py).
- **C4:** [TorchCheckpoint](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_method/hooks.py).
- **C5:** [Directional derivatives](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_method/derivatives.py), [기존 derivative oracle tests](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_method/tests/test_derivatives_functional.py).
- **C6:** [Covariance·finite-difference diagnostic math](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/diagnostic_math.py).
- **C7:** [Fixed-z nonuniqueness algebra](/mnt/raid5/janghj/ODE-edit/project/run_scripts/fixed_z_nonuniqueness/algebra.py).
- **C8:** [Native factor/metric 계측이 있는 geometry](/mnt/raid5/janghj/ODE-edit/project/run_scripts/alphaedit_strength_neutral_barrier/geometry.py). 이 패키지의 barrier와 별도 locality evaluator는 이번 runtime/metric에 승계하지 않는다.
- **C9:** [봉인된 BLUE metric reducer](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/analysis-source/project/run_scripts/blue_lifelong_analysis/metrics.py), [기존 aggregation source](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/analysis-source/project/run_scripts/server4_experiments_review/aggregate.py).

### 15.3 선행연구의 버전과 비교 범위

- **P1:** Fang et al., [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models, v3](https://arxiv.org/html/2410.02355v3). Null-space/history objective의 배경. 실제 실행식과 수치 정책은 봉인 source를 기준으로 한다.
- **P2:** Liu et al., [BetaEdit: Null-Space Constrained Sequential Model Editing, v1](https://arxiv.org/html/2605.09285v1). Approximate null-space leakage, history-aware update 및 projector refresh와 비교한다.
- **P3:** Ikram et al., [CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing, v2](https://arxiv.org/html/2602.15823v2). Functional preservation과 sequential variant까지 포함해 비교한다.
- **P4:** Li et al., [Rethinking Residual Distribution in Locate-then-Edit Model Editing, v3](https://arxiv.org/html/2502.03748v3). BLUE의 local target·boundary-layer 전략 배경. 실제 baseline의 L2·target·write 정책을 source와 함께 읽는다.
