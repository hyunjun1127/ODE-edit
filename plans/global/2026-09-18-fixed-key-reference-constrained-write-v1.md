# Fixed-key Reference-Constrained Write — 방법과 빠른 실험 v1

**SUPERSEDED / 실행용 아님.** 사용자의 W0 greedy 답변 선택 보호 지시에 따라 [v2 방법](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-base-choice-constrained-write-v2.md)과 v2 contract로 교체한다. 중단 당시 이 문서의 전체-reference 개정과 v1 JSON의 소수-활성 설정이 일치하지 않았으므로 v1 산출물로 실행하지 않는다.

작성: 2026-09-18. 약칭 FRCW는 작업용 이름이다. 상태: **수식·실행 규약 설계 완료, 모델 구현·factual bank 구축·GPU 실험 미실행**. 사용자 요청은 방법 정의와 cold single-batch → sequential 설계다. 기존 EN 계약을 덮어쓰지 않는다.

현재 revision: **all-reference v1.1**. 사용자의 “factual reference를 모두 이용” 지시에 따라 32개 후보/16개 활성 제약 표본선택을 제거했다. 매 batch 모든 유효 factual reference의 두 보존 함수를 제약에 포함하고, 실제 endpoint에서도 모두 확인한다.

동반 파일: [contract](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-fixed-key-reference-constrained-write-contract-v1.json), [cells](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-fixed-key-reference-constrained-write-cells-v1.csv). 근거: [EN B001 재분석](/mnt/raid5/janghj/layer_allocation/09_enfc_b001_direct_reference_reassessment.md), [EN 수식](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-en-pipeline-mathematical-explanation-v1.md).

## 1. 고정하는 방법의 정체성

**L4 native local-z 편집이 실제로 실현한 response를 정확히 유지하는 weight 집합 안에서, 비편집 사실의 기능적 보존 제약을 만족하는 최소 변경 write를 구한다.**

- 유일한 변경 parameter: zero-based L4 `down_proj.weight`, 4096×14336. Gate/up, attention, 이전·이후 layer는 고정한다.
- Native local-z와 native write는 배치별 현재 상태에서 기존 규약대로 한 번 계산한다. Reference 또는 후보별 z 최적화는 0회다. 이는 batch당 z 한 vector라는 뜻이 아니라 기존 요청별 z 계산 묶음 1회다.
- 평균 reference KL을 최적화하지 않는다. Native update에 .75 등을 곱하지 않는다.
- Reference는 현재 write의 제약이며, reference gradient의 내적은 제약의 중복·충돌을 표현하는 통계다.
- 별도 paraphrase 학습·생성·replay target은 없다. 공식 P/N은 봉인된 endpoint의 observer다.
- v1.1은 **현재 edit의 exact response lock + 전체 유효 R512의 factual 제약**이다. 실제 endpoint에서 모든 bank 제약을 통과해야 한다. 이는 bank 밖의 unseen locality 보장을 뜻하지 않는다.

## 2. 기호와 native 편집

|기호|정의|
|---|---|
|W0|pre-edit L4 weight|
|W−_t|batch t 진입 weight; W−_1=W0|
|E_t|현재 B100 edit requests|
|Z_t|기존 L4 local-z optimization 결과|
|ΔN_t|현재 상태에서 native writer가 실제로 만든 update|
|WN_t=W−_t+ΔN_t|commit 전 native preview|
|D_t|편집 response를 유지하는 추가 weight 변화|
|W_t=WN_t+D_t|최종 commit; native fallback은 D_t=0|
|K_E,t|현재 rewrite의 보호 sequence에 포함된 모든 유효 token의 down_proj input keys|
|P*=VVᵀ|native preservation 허용 range의 직교 projector|
|R512|독립 외부 사실 reference 512개|
|H_t|과거 최신 유효 edit registry 및 제한된 screen 집합|

`Z_t = NativeZOpt_4(W−_t,E_t)`

`ΔN_t = NativeWrite_4(W−_t,Z_t,K_write,t,P_raw,M_{t−1})`.

Native write key K_write와 full-token 보호 key K_E는 다르다. `WN K_E=Z_t`라고 가정하지 않는다. 원 AlphaEdit/BLUE hparams의 native clamp=.75 등은 그대로 유지하지만, 새로운 allocation scale=.75를 추가하는 것이 아니다.

## 3. Single-layer fixed-key 구조와 정확한 보호 공간

고정 입력 x, position/mask 아래 L4 down_proj input K(x)는 W와 무관하다. 고정 residual을 B(x), frozen downstream을 F_x라 쓰면

`p_W(·|x) = F_x(B(x)+W K(x))`.

따라서 모듈 출력 변화 `ΔY=ΔW K(x)`는 유한 변화에서도 정확하다. 다만 F_x는 비선형이다. 같은 입력의 K와 B는 lifelong 동안 cache할 수 있지만, downstream Jacobian은 달라진다.

K_E는 native rewrite contexts 및 canonical의 new/old teacher-forcing 경로 전체 유효 token을 포함한다. 동일 causal prefix만 deduplicate한다. Subject 평균만 잠그지 않는다. Old target이 없으면 해당 pairwise exact claim에서 제외하고 결측을 기록한다.

허용 correction 공간은

`D_t-space = {D : D=DP*, D K_E,t=0}`.

`J_t=VᵀK_E,t`, `Q_t=V(I−J_t J_t†)Vᵀ`이면

`D∈D_t-space ⇔ D=DQ_t`.

이는 native 허용 공간과 edit-null 공간의 교집합이다. 두 projector를 임의로 순차 곱하지 않는다. 정확 산술에서 보호 sequence logits는 WN과 동일하다. FP32 physical endpoint는 별도 검사한다. 보지 않은 P나 다른 생성 prefix는 보장 범위 밖이다.

Reference key K_j에 대한 correction은 `D K_j=D U_j`, `U_j=Q_t K_j`다. U_j=0이면 current edit response를 유지한 채 해당 reference의 layer response를 바꿀 수 없다. 남은 key 차원 수가 크다는 것만으로 factual repair 가능성을 주장하지 않는다.

## 4. Reference 데이터와 보호할 함수

### 4.1 Factual bank의 정확한 역할

주 bank는 Wikipedia 근거가 있는 **512개의 독립적인 사실**, 개발 observer는 source/subject가 분리된 128개다. Bank의 고정 hash 순서 앞 256개를 크기 ablation용 subset으로 보관하되 첫 빠른 실행에는 별도 arm으로 넣지 않는다. 기존 일반 C4 S64/Dev128을 factual bank라고 재명명하지 않는다.

각 record는 `(id, source revision/URL/문장 위치, subject ID/aliases, relation, accepted answers/aliases, one canonical prompt x, answer a, fixed incorrect challenger c0, validity/dependency metadata, token hashes)`를 갖는다. 한 사실당 canonical prompt 하나이며 paraphrase set이 아니다.

준비 규약:

1. 실제로 확보한 Wikipedia snapshot과 지원 relation metadata의 revision/hash를 봉인한다. 관계·시간 범위와 정답을 출처에서 확인한다. 현재 workspace에서 이 구조의 bank가 준비되었다는 근거는 아직 없다. Exact source revision과 최종 IDs는 실행 전 manifest로 확정해야 한다.
2. 단일 정답 관계/시점으로 해석이 명확한 record를 사용한다. 정답 문자열만 같은 여러 entity, 복수 정답·불명확한 granularity 등은 제거한다. 지원되지 않는 relation은 임의 생성하지 않는다.
3. 동일 relation/type의 다른 record에서 확인 가능한 오답 후보를 만들고, 정답 alias 및 유효한 다른 답을 제외한다. W0에서 가장 높은 answer likelihood를 갖는 검증된 오답을 c0로 저장한다. W0가 a를 c0보다 선호하는 record를 주 보존 bank에 사용하고 이 필터를 보고한다.
4. Canonical prompt와 각 답변을 합친 teacher-forcing 입력은 64 tokens 이하, 답변은 16 tokens 이하인 record만 사용한다. 답변을 중간에서 자르지 않는다. Raw 문장에 답이 이미 노출되는 copying prompt는 제외한다.
5. Source/subject group으로 train512/dev128을 분리하고, group 내 선택은 `SHA256('FRCW-v1|20260918|record_id')` 순이다. 공식 CounterFact N/P를 조회해 좋은 reference를 선택하지 않는다. 평가 문장·fact 중복 배제는 별도 evaluator가 manifest로 검사하며 model selection에 쓰지 않는다.
6. 현재/과거 edit의 동일 subject alias, overwrite 대상, metadata상 함께 바뀌어야 하는 사실은 해당 batch의 보존 대상에서 제외한다. Future requests로 bank를 구성하거나 교체하지 않는다. 제외분은 그 batch에서 대체 sampling하지 않고 유효 분모를 보고한다. 이 metadata 필터를 완전한 의미적 충돌 해결로 주장하지 않는다.

이 record들을 확보하기 전에는 모델 실험을 시작할 수 없다. 일반 문서의 W0 argmax를 검증된 factual label로 조용히 대체하지 않는다. C4는 기존 Dev128 일반 분포 observer로 유지한다. Wikipedia 대신 C4에서 동일한 provenance·fact 조건을 충족한 bank를 만들려면 별도 data ID로 취급한다.

### 4.2 이번 batch에서 실제로 보호할 challenger

잘못된 새 target의 확산을 직접 다루되, 512×100개 답변을 모두 teacher-force하지 않는다.

Reference j의 후보는 c0와 현재 batch의 새 target 중 **해당 사실에서 오답임을 확인할 수 있는** 것들이다. Relation/type·answer aliases·dependency 검사를 통과한 후보만 허용한다. 각 reference prompt의 WN first-answer-position logits에서 후보 첫 token 확률이 가장 높은 것을 c_j,t로 선택한다. 같은 첫 token이면 stable target hash 순으로 결정한다. c0도 후보에 포함한다.

이는 저렴한 shortlist 규칙이며 full-answer likelihood의 최적 challenger 선택이 아니다. 같은 첫 token을 공유하는 답변에서는 구분력이 약함을 기록한다. Prompt-target tokenizer 경계가 달라 공통 prefix가 성립하지 않으면 해당 후보는 이 shortlist에 넣지 않는다. 정답·오답 구분이 검증되지 않은 edit target은 후보로 넣지 않는다.

선택한 c_j,t는 **그 batch의 모든 candidate/SQP round에서 고정**한다. 같은 c_j,t를 W−, WN, candidate 모두에 사용한다. Entry의 c0 margin과 candidate의 새 challenger margin을 비교하면 잘못된 damage가 생기므로 금지한다. W0에 대해 미리 저장한 c0 값만으로 새 c_j,t의 entry 값을 대신하지 않는다.

### 4.3 두 보존 함수와 entry bound

정답 token 평균 NLL을 `ell(W;x,a)=−(1/|a|)Σ_s log p_W(a_s|x,a_<s)`로 정의한다. 평가와 tokenizer/target-prefix 규약을 맞추고 EOS를 임의 추가하지 않는다.

`q_j^A(W)=ell(W;x_j,a_j)`

`q_j^M(W)=ell(W;x_j,a_j)−ell(W;x_j,c_j,t)`.

둘 다 낮을수록 좋다. A는 정답 자체의 약화, M은 정답 대비 오답의 우위 침범을 다룬다. 서로 다른 길이의 답변이면 M은 평균 NLL 차이며 sequence log-odds라고 부르지 않는다.

`b_j,t^c=q_j^c(W−_t), c∈{A,M}`를 batch entry bound로 정한다. 목표는 `q_j^c(W_t)≤b_j,t^c`다. Numerical acceptance epsilon=1e−4 NLL은 성능을 위해 학습하는 slack이 아니다. 누적 epsilon drift를 포함해 W0 대비 지표를 별도로 보고한다.

고정 c0를 이용한 W0 대비 drift는 B1/B10에 별도 관측한다. Dynamic challenger가 바뀐 batch들의 margin을 그대로 누적해 같은 지표라고 부르지 않는다.

## 5. 과거 edit의 처리

전체 latest-version registry에는 지금까지 실제 받은 요청만 넣는다. Replay 후보 memory는 최신 active record의 stable hash 우선 512개이고, 매 batch hash order의 순환 창 64개를 screen한다. 아직 64개 미만이면 전부 사용한다. B1은 빈 집합이다. 이 memory 유지 정책 자체를 원형 GSS라고 부르지 않는다.

과거 record는 최신 target을 a, 제공된 이전 target을 challenger로 사용한다. Overwrite 현재 대상은 이전 version을 controller에서 제외한 뒤 commit 시 registry·memory·cache를 원자적으로 갱신한다. Subject–relation canonicalization은 metadata proxy이며 모든 semantic alias를 완벽히 해결한다고 가정하지 않는다. Paraphrase target은 추가하지 않는다.

History screen의 entry→native damage도 reference와 같은 q로 측정하고, screen된 과거 요청은 모두 entry-bound 제약에 포함한다. Native가 이미 만든 past damage를 correction no-harm으로 감추지 않는다. 현재 edit exact lock, 전체 factual R512, 과거 screen64는 서로 다른 보호 범위다. History memory512 중 그 batch에 screen하지 않은 과거 요청의 보장은 주장하지 않는다. 사용자의 전체 factual reference 지시는 R512에 적용하며, 과거 요청 memory의 별도 규모·범위는 명시적으로 유지한다.

## 6. 핵심 constrained write와 실제 solver

전체 제약 index 집합을 `I_t={(j,c): j∈유효 R512∪HistoryScreen64, c∈{A,M}}`로 정의한다. Reference가512개이고 history가64개이면 최대1152개 scalar 제약이다. B1은 history가 없어 최대1024개다. Reference 목표를 가중합 loss로 최적화하지 않고 다음 nonlinear 제약 문제를 근사한다.

`min_{D=DQ_t} 0.5 ||D||_F²`

`subject to q_i(WN_t+D)≤b_i, 모든 i∈I_t`.

Reference 위반을 허용하는 성능 slack은 v1에 두지 않는다. Infeasible이면 native fallback이며 constraint-success가 아니다. 현재 edit response를 위해 native update를 약화시키거나 z를 다시 최적화하지 않는다.

### 6.1 선형화는 native preview에서 시작한다

이전 개념 분석의 entry-gradient 전체-update 근사보다 **실제 native 피해를 측정한 뒤 WN에서 선형화**하는 쪽을 v1로 고정한다. Entry는 보호 bound이고, native는 계산 중심이다. Native는 아직 commit하지 않은 상태이므로 reference가 최종 write를 제약한다는 해석은 유지된다.

Round k의 D_k와 W_k=WN+D_k에서

`g_i,k=∇_W q_i(W_k)`, `h_i,k=g_i,k Q_t`

`d_i,k=q_i(W_k)−b_i−<h_i,k,D_k>_F`.

최종 D를 변수로 하는 선형화 QP는

`min_{D=DQ_t} 0.5 ||D||_F²`

`subject to <h_i,k,D>_F≤−d_i,k`.

첫 round는 D_0=0이므로 d_i,0은 실제 native damage다. 두 번째 round에서 RHS의 `<h,D_k>` 항을 빠뜨리면 increment와 total correction을 혼동하는 오류다. 두 round 모두 native로부터의 최종 D norm을 최소화한다.

위 표기는 D_k=D_kQ인 정확 산술 기준이다. 구현의 W_k가 FP32 materialization이고 `D_k,actual=W_k−WN`이면 미세한 공간 밖 성분이 있을 수 있다. 따라서 실제 RHS는 **`d_i,k=q_i(W_k)−b_i−<g_i,k,D_k,actual>`**로 계산한다. 후보 변수 D가 허용 공간에 있으므로 좌변만 h_i,k로 투영한다. Actual center에 h를 대신 적용해 rounding 성분을 조용히 버리지 않는다.

### 6.2 작은 dual 문제

`G_ij=<h_i,h_j>_F`라 하면 dual은

`min_{alpha≥0} 0.5 alphaᵀ G alpha − dᵀ alpha`,

`D*=−Σ_i alpha_i h_i`.

KKT 조건은 `G alpha−d≥0`, `alpha_i(G alpha−d)_i=0`다. 한 위반 제약만 있고 h≠0이면 `D*=−d h/||h||²`다. 필요한 강도는 실제 피해량과 허용 gradient에서 정해진다. Alpha는 constraint dual variable이며 layer allocation weight가 아니다.

QP 전에 nonzero row와 RHS를 같은 positive gradient norm으로 나누어 conditioning을 개선할 수 있다. 이는 feasible set을 바꾸지 않는다. h≈0인데 d>0인 제약은 FIRST_ORDER_UNREACHABLE로 기록한다. 제약을 몰래 제거하고 전체 보호에 성공했다고 하지 않는다. 선형 QP infeasibility와 solver numerical failure를 구별한다. Primal feasibility, dual residual, complementarity를 **전체 I_t**에서 FP64로 검증한다. Arbitrary ridge로 Gram을 바꾸고 exact QP라고 부르지 않는다.

모든 reference를 제약에 사용한다고 모든 alpha가 양수여야 하는 것은 아니다. Nonbinding 제약의 alpha=0은 전체 문제를 푼 결과다. 처음부터 해당 reference를 표본선택으로 제외하는 것과 다르다. Working-set QP를 사용하더라도 제외된 row를 포함한 전체 primal/dual 검사가 완료되어야 해를 수용한다. Working-set에 16개 같은 hard cap을 두지 않는다.

## 7. Fixed key로 gradient·통계 계산량을 줄이는 방법

Reference의 teacher-forcing 경로 전체 key를 K_i, module-output derivative를 A_i=∂q_i/∂Y_i라 하면

`g_i=A_i K_iᵀ`, `h_i=A_i (Q_t K_i)ᵀ=A_i U_iᵀ`.

Contrast의 두 sequence는 column으로 연결하고 각 경로의 부호·NLL normalization을 A_i에 반영한다. Subject token만 사용하거나 답변 token 앞의 prefix gradient를 detach하지 않는다.

이 표현으로 full weight gradient를 row마다 저장할 필요 없이 A_i,U_i를 저장할 수 있다. Inner product는

`G_ij=tr((A_iᵀ A_j)(U_jᵀ U_i))`

로 계산한다. 최종 D materialization은 마지막에 수행한다. Factor 보관이 downstream backward 자체를 없애는 것은 아니다.

서로 다른 입력이 batch 축에서 독립이므로, sum(loss_j)를 **module-output activation에 대해** 역전파하면 sample별 A_j가 batch 축에 남는다. Full parameter에 대한 summed gradient만 얻으면 sample별 분해가 사라진다. 이를 이용해 per-example backward 호출을 microbatch로 묶는다. Cached/physical forward·AD 검증을 먼저 통과해야 한다.

두 함수의 gradient는 중복 계산하지 않는다. 정답과 challenger 각각의 NLL derivative를 구하면 `g^A=g_answer`, `g^M=g_answer−g_challenger`다. 따라서 한 round에서 factual512×두 답변 경로를 역전파하고 두 제약 channel을 조립할 수 있다. QP를 위해 A와 M을 각각 별도 전체 모델 backward로 재실행하지 않는다.

고정 reference K/B는 CPU cache, U/A는 전체 bank를 microbatch/block으로 계산한다. Q_t와 A_i는 batch/round마다 변한다. Dense 14336² Q를 매번 만들지 않고 EN의 verified basis 방식으로 적용한다. 단지 fixed key라는 이유로 downstream derivative를 오래 재사용하지 않는다. Challenger가 바뀌면 새 branch key가 필요하므로 동일 prefix와 새 suffix를 구별해 cache하며, cache 준비 비용도 계측한다.

Full per-row gradient1024개를 dense로 저장하지 않는다. Gram을 tiled factor contraction으로 구성하거나 `alpha → D(alpha)=−Σ alpha_i h_i → {<h_j,D(alpha)>}_j` 연산자로 푼다. Full Gram 자체는1024² FP64 약8.39MB(history 포함1152² 약10.62MB)이지만, 이를 만드는 tensor 연산·factor I/O는 훨씬 클 수 있다. Small dual matrix를 이유로 solve 전체가 싸다고 주장하지 않는다.

## 8. 전체 reference와 고정 계산 예산

Reference512와 History64의 A/M 값을 entry와 native에서 모두 계산한다. 첫 batch에는 history가 없고, 같은 fixed sequence의 entry scalar를 지난 endpoint cache에서 재사용할 수 있다. 새 challenger의 entry score는 별도로 계산한다.

1. `v_i=[q_i(WN)−b_i]_+`를 전체 row에서 측정한다. 전부 numerical epsilon 이하이고 보호 성공 ID도 유지되면 D=0으로 종료한다.
2. Correction이 필요하면 **위반 여부와 관계없이 전체 유효 R512의 두 gradient channel**을 계산해 QP에 넣는다. 현재 위반하지 않은 사실도 correction으로 손상될 수 있으므로 제외하지 않는다. HistoryScreen64도 같은 규칙이다.
3. Full-constraint QP를 한 번 풀고 전체 factual/history finite 제약을 검사한다. 통과하지 못하면 같은 전체 집합에서 한 번만 재선형화한다. QP2회, full-bank candidate 검사2회가 상한이다.
4. GSS sample reduction, gradient 후보32, 활성16 제한을 제거한다. Solver 내부의 계산 순서·working-set은 사용할 수 있으나 모든 row의 최종 feasibility/KKT와 실제 endpoint 검사를 생략하지 않는다.

R512만 있을 때 한 round의 최대1024 answer-path gradient 평가, 두 round2048이다. History64까지 있으면 round당 최대1152, 두 round2304다. 이는 개별 GPU backward 호출 횟수가 아니며 microbatch 묶음과 shared-answer gradient 재사용을 반영해 실제 호출 수를 따로 기록한다. Native screen과 첫 gradient, candidate screen과 다음 gradient를 공유할 수 있으면 실제 비용을 줄이되 중복 절감으로 계산하지 않는다.

Factual 입력 길이 상한64 기준 한 round의 answer-path input token 상한은65,536, 두 round131,072다. History, entry/native no-grad 검사, geometry, Current, official observer는 이 수에 포함되지 않는다. 512개 모두 쓰는 변경은 이전 소수-활성 제안보다 계산량을 늘린다. 속도는 같은 하드웨어에서 측정해야 한다.

## 9. Candidate 수용과 fallback

각 proposed D는 immutable WN에서 실제 FP32 weight로 materialize한다. FP32 actual delta와 nominal D를 구별한다. 두 번째 round도 실제 첫 후보의 q/gradient를 쓰되 총 D 좌표는 actual W−WN으로 일관되게 둔다. 다음 solve 결과는 다시 D=DQ인 공간에서 생성한다.

Cheap finite/geometry 검사를 통과한 후보에 **전체 factual/history row**의 suffix forward를 수행한다. `q_i(candidate)≤b_i+1e−4`가 전부 만족되고 entry-correct factual margin 성공 ID를 잃지 않으면 Current 최종 검증으로 간다. 아니면 한 번만 재선형화하고, 두 번 모두 실패하면 NATIVE_FALLBACK_FULL_REFERENCE_CONSTRAINT를 기록한다. Reference를 통과한 첫 후보가 Current guard에서 실패하면 추가 탐색하지 않는다.

최종 검증은 최대 한 후보에만 수행한다.

- **Current:** EN의 actual Δkey, logits/NLL/strict invariant·quality guard. Current new NLL은 own native+1e−4 이하, native canonical preference/strict 성공 ID 손실0. Full-token logits max1e−3/RMS1e−4, proposed normalized DK_E≤1e−10, actual max-token response≤1e−5. 이전 EN의 T skipped를 새 방법의 PASS로 물려받지 않는다.
- **전체 factual/history:** entry bound+1e−4를 모두 만족한다. 수치 허용량 이내라도 entry에서 성공한 factual margin ID를 실패로 바꾸면 거절한다. 평균 개선으로 개별 위반을 상쇄할 수 없다.
- **전체 유효 R512:** `V_R(W)=mean_i [q_i(W)−b_i]_+`와 최대 위반·성공 ID 변화를 보고한다. 이들은 전체 제약 검사 요약이며 평균 개선만으로 수용하지 않는다.
- **History screen64:** entry bound 조건과 함께 native의 canonical preference/strict 성공 ID 추가 손실0을 요구한다. Screen 밖의 과거 edit는 보장하지 않는다.

Reference violation V는 진단값이며, optimizer가 KL 복원 loss를 내려가는 구조는 아니다. 제약형이라는 표기만으로 목적함수형보다 본질적으로 새롭다고 주장하지 않는다. 의도적으로 수정하는 사실과 metadata상 충돌한 reference는 scope conflict로 명시하고 유효 분모에서 구별한다. 계산량을 줄이거나 실패를 숨기기 위해 reference를 제외하지 않는다.

통과하면 W_t=WN+D, 아니면 W_t=WN이다. Reference bound를 위반한 fallback은 실패/미해결로 집계한다. 수치 오류·잘못된 state·NaN을 과학적 fallback으로 숨기지 않고 기술 실패로 중단한다. Weight, native M append(1회), version registry, reference score cache, cursor/RNG, receipt를 원자적으로 commit한다.

## 10. 한 batch의 파이프라인

1. Entry/metadata 봉인: current overwrite를 반영한 유효 R/H 목록, 고정 입력 key cache, entry weight/hash 준비.
2. Own native local-z/write 1회: Z, ΔN, WN, native history의 commit 예정 상태 저장.
3. Current K_E와 Q 구성: full-token provenance, rank, 남은 차원, 실제 null 조건 확인.
4. R512/H64 screen: challenger를 결정·고정하고 동일 challenger의 entry/native q를 확보.
5. 전체 reference/history gradient factors와 projected 통계 계산 → full-constraint min-norm QP.
6. Actual candidate의 전체 reference/history 제약 검사. 필요 시 전체 집합에서 한 번만 재선형화.
7. 최초 reference-feasible 후보 하나에 Current 최종 guard. Accept 또는 native fallback.
8. Transaction commit 후 official observer 실행. Observer 결과가 같은 batch의 후보 선택으로 돌아가지 않는다.

## 11. 빠른 실행 순서: cold B1 → 같은 chain의 B2–B10

### 11.1 실행 준비와 B1 내부 기술 확인

Factual bank manifest가 준비되면 server1 같은 GPU/runtime에서 두 arm을 실행한다. Model은 기존 Llama-3-8B-Instruct revision과 FP32/eval/eager/TF32-off를 유지한다. 예전 mixed-hardware endpoint를 신규 matched baseline으로 자동 인정하지 않는다.

별도 대규모 T arm 대신 **B1 안에서** 고정 hash reference4개와 current prefix로 다음을 확인한다: key stationarity, cached/physical forward, factor-vs-direct projected gradient, 두 합법 방향의 centered finite difference, dual sign/KKT, FP32 restore/transaction. FD는 두 step 크기에서 방향 오차가 안정되는지 보고, AD 대비 상대오차1e−3 또는 noise-floor 절대 기준을 충족해야 한다. Cached/physical gradient relative1e−4 및 EN forward tolerances를 따른다. 이 검사는 모델 학습·추가 z 최적화를 하지 않는다. 실패하면 실험 결과를 과학적 비교로 진행하지 않는다.

### 11.2 P1: W0 → B100 한 batch, 두 endpoint

|arm|시작|동작|
|---|---|---|
|N4|W0|native L4 local-z/write|
|FRCW512|동일 W0 및 동일 native preview|위 reference-constrained write|

O0 first100을 사용한다. 이 B1은 이미 관측한 개발 batch이며 unseen test가 아니다. Native z/write는 두 endpoint가 공유하지만 standalone 비용은 각각 native 비용을 포함해 계산한다. 새 8-arm sweep, 여러 cold batch, layer/scale search는 하지 않는다.

Endpoint 봉인 뒤 R100/P200/N1000, paired 성공 ID, W0-correct N 손실/회복, true/false NLL 분해, 고정 F-dev128, C4-Dev128을 평가한다. Dev/P/N은 gradient·QP·candidate 선택에 사용하지 않는다.

**B1 → sequential 진입 기준:**

1. 기술 검사 PASS, 현재 exact/finite edit guard PASS.
2. Nonzero correction이 실제 채택되고 전체 reference/history bound와 Current guard 통과. 전부 fallback이면 바로 긴 chain을 돌리지 않는다.
3. N4 대비 canonical RS와 PS의 성공 ID 추가 손실0. NS net 성공수 감소0, W0-correct N 손실수 증가0.
4. F-dev128의 factual 성공수 감소0. 개발 지표는 확정된 두 endpoint에서 한 번만 읽는다.
5. 같은 hardware의 setup 제외 standalone editing 시간 비율이 N4의2배 이하. 초과는 효과가 없다는 판정이 아니라 현재 빠른 방법의 계산 예산 실패다. Official evaluation 비용은 별도로 보이고 total도 합산한다.

NS가 그대로여도 나머지를 통과하면 S10으로 간다. 이는 누적 이득을 확인할 가치가 있다는 개발 판단이며 B1 locality 개선 입증이 아니다. NS/P가 악화되거나 모든 후보가 거절되면 원인을 한 번 점검하고 version을 수정해야 한다. Official P/N으로 같은 version의 파라미터를 재선택하지 않는다.

### 11.3 S10: 각자의 B1 checkpoint에서 B100×10까지 즉시 연장

P1 이전에 policy/data/source를 봉인한다. 통과하면 N4는 N4 B1, FRCW는 FRCW B1에서 **동일 episode의 B2–B10**을 이어간다. 두 chain 모두 W0에서 시작한 것이다. 임의의 기존 5000-edit checkpoint를 쓰지 않는다.

B2부터는 서로 다른 W entry 때문에 각자 own z/native preview/history를 계산한다. N4의 native 방향을 ours에 이식하거나 공통 native shadow 위에서 correction만 비교하지 않는다. Fixed text의 upstream key/cache만 공유 가능하며 gradient·target·native endpoint는 공유하지 않는다. 설정 변경이나 source/data hash 변경이 있으면 B1부터 새 W0 chain이다.

빠른 sequential 평가:

- 새 batch의 N을 entry와 post-write에서 측정해 **arrival damage**를 기록한다. P/N reader는 controller 밖에 두고, 필요하면 봉인된 entry state를 별도 observer로 평가한다.
- 각 batch의 current R/P/N을 at-write로 저장한다.
- B1의 100개 요청을 고정 old cohort로 삼아 B2–B10마다 R/P/N을 재평가한다. 직전 endpoint 값은 다음 entry 값으로 재사용해 중복 평가를 줄인다.
- B10에서 seen1000 전체 R1000/P2000/N10000을 한 번 평가한다. Current/old-cohort와 같은 state의 이미 계산한 결과는 identity로 재사용한다. 매 batch 전체 prefix를 평가하는 고비용 방식은 쓰지 않는다.
- F-dev128 및 C4-Dev128은 B1/B5/B10, 고정 c0의 train-bank W0 drift는 B1/B10에서 관측한다. Online controller에 전달하지 않는다. 별도의 최종 Report256은 빠른 개발 동안 열지 않는다.

S10은 최종 RS/PS·old cohort 망각·arrival N 손실·최종 NS와 계산량을 함께 판단한다. 한 order/1000개 결과로 10k generalization을 선언하지 않는다. 이후 같은 policy로 10k까지 이어가는 것은 후속 확장이다. 첫 quick pass의 자동 범위는 P1+S10 두 chain이다.

## 12. 비용 예산과 필수 산출물

|항목|배치별 상한/규칙|
|---|---|
|추가 z|0|
|기본 reference|512 facts, 각 selected challenger 포함 최대2 answer paths|
|history screen|최대64 active edits; memory512의 순환 창|
|gradient 대상|전체 유효 R512 및 HistoryScreen64, 두 channel 모두|
|QP 제약 수|B1 최대1024; history64 포함 최대1152|
|재선형화|최대1회; 전체 집합 재계산|
|QP / 전체 reference candidate 검사|최대2회|
|Current 최종 guard|최대1 candidate|
|공식 P/N online 선택|0|

Reference 준비 시 c0 선정과 W0 filtering 비용, cached upstream 준비, entry/challenger 재평가, projected key/Gram, rejected candidate, native fallback 검사도 계측한다. 512개 forward 비용이 존재하며 per-arm이 EN-F보다 빠르다고 사전 단정하지 않는다. Full-vocab teacher 분포 저장은 필요 없지만 target NLL의 정확한 softmax 계산은 여전히 필요하다. 첫 answer logits만 계산한다고 multi-token likelihood로 바뀌는 것도 아니다.

필수 receipt는 source/data/model/tokenizer/weight hashes, current/reference token identity, bounds/entry/native/selected q, 전체 gradient row와 projection norm, reference coverage512/512 및 scope-conflict 분모, QP 전체 primal/dual residual과 nonzero multipliers, accepted/fallback 이유, 실제 D norm 및 ΔN norm, remaining dimension, finite guards, native/history transaction, 비용 breakdown이다. 공식 observer에는 paired IDs, numerator/denominator, at-write/entry/final 상태를 구별한다.

추가 ablation은 S10 이후다: 같은 factual reference를 쓰는 EN-KL/soft factual loss, fixed c0 대 request challenger, R256, original AlphaEdit/BLUE 비교. GSS로 reference를 줄이는 실험은 현재 주 방법 범위 밖이다. P1에서 이를 모두 실행하지 않는다. 따라서 P1/S10 두 arm만으로 데이터 변경 효과와 제약 solver의 기여를 분리하거나 novelty를 확정할 수는 없다.

## 13. 주장 가능한 범위

방법의 검증 대상은 **실현된 편집 response를 고정하는 single-layer affine 공간에서, 직접적인 factual 제약이 locality 비용을 줄일 수 있는가**다. Margin 보호는 APP, gradient 제약/선택은 GEM/GSS, functional curvature는 CrispEdit 등과 비교해야 한다. 데이터 supervision 증가와 compute를 맞춘 대조가 필요하다. 관련 경계는 09번 분석의 원문 링크를 따른다.

Fixed key, small QP, nonzero correction, reference training 개선만으로 NS 개선을 선언하지 않는다. Actual current response 보호는 수치 검증 범위에서, reference 보존은 전체 유효 R512 및 명시된 history screen 범위에서, PS와 unseen N은 observer 결과로 각각 보고한다.
