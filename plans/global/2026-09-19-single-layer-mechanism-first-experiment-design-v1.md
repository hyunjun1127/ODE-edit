# Single-layer 기전 검증과 판단 보존 보정: cold B100 → SEQ300 → SEQ1000

작성: 2026-09-19. 상태: **실험 설계**, runner 구현·새 모델 실행·GPU 제출 없음. 연산 재사용 최적화는 별도 작업으로 진행 중이며 이 문서는 방법·실증 조건을 정의한다. 검증된 최종 방법이나 novelty 확보 선언이 아니다.

관련 분석: [조건부 정리와 방법 검토](/mnt/raid5/janghj/layer_allocation/single_layer_mechanism_and_method_review_20260919.md), [EN R512/G256 감사](/mnt/raid5/janghj/layer_allocation/en_r512_g256_capacity_scope_audit_20260919.md). [실행 규약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-19-single-layer-mechanism-first-contract-v1.json), [단계별 cells](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-19-single-layer-mechanism-first-cells-v1.csv).

## 1. 이번에 판별할 질문

|ID|가설|확인에 필요한 결과|그 결과만으로 주장하지 않을 것|
|---|---|---|---|
|H1|Key 구별성과 target demand의 결합이 실제 write 비용을 설명한다|실제 writer metric의 spectrum/loading과 actual Δ의 대응|작은 singular value만으로 write가 반드시 증폭|
|H2|그 write의 일부 반응이 실제 보존 margin을 낮춘다|동일 입력의 ΔK와 finite margin 변화, component 개입|norm/activation energy가 곧 locality 손상|
|H3|같은 native edit response를 유지하며 제거 가능한 functional 손상이 있다|DK_E≈0와 actual Current 유지 아래 독립 보존 개선|모든 가능한 single-layer edit의 안전성|
|H4|EN의 평균 KL 목적 또는 한 방향 탐색이 제약이었다|동일 entry의 KL-Q, Decision-Line, Decision-Modes 비교|제한된 탐색의 실패가 전체 공간의 불가능성|
|H5|누적 변위를 이용하면 신규 write만 보는 것보다 유리하다|cold부터 시작한 STEP/CUM 비교와 동일 entry paired 진단|B1만으로 누적 상쇄 이득이 입증됨|

우선순위는 H1–H4의 B1 확인, 그다음 H5 및 과거 편집·이전 neighborhood 망각의 짧은 sequential 확인이다. 처음부터 10k 전체나 layer allocation grid를 실행하지 않는다.

## 2. 고정 조건과 데이터 역할

- Model: 기존 EN의 pinned Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`를 기존 실제 실행 manifest와 재대조한다.
- 편집 위치: zero-based `model.layers.4.mlp.down_proj.weight`, 4096×14336. 다른 weight는 고정.
- 시작: **W0와 zero native history**. 5000-edit 등 warm checkpoint 시작 금지.
- 편집 stream: 기존 fixed10k의 봉인된 순서 첫1000, B100. 여러 번 분석한 개발 데이터임을 명시한다. 새 confirmatory stream이라고 부르지 않는다.
- Native local-z: 기존 local-L4 설정·context weighting·projector·write algebra 그대로. request당 native target 계산 1회, correction마다 z 추가 계산 없음.
- Precision/attention/tokenizer/decoding: 기존 검증 실행의 FP32/eager/TF32-off 규약 계승. 수정된 runtime은 동일 수치 의미를 기술 검증한 뒤 source hash를 봉인한다.
- Train reference: **기존 C4512 전체**, S64+Reserve320+AdditionalTrain128. Reference 교체·축소·공식 N 편입 없음.
- W0 생성: 최대256, EOS이면 실제 길이 사용. 기존 run의 실제130,235 train 위치를 재사용하려면 capsule hash가 같아야 한다. 최대256을 모든 문서의 실제 길이256으로 간주하지 않는다.
- Dev128: 별도 generated reference. Selector/gradient/line search에 접근하지 않음.
- PS/N: 공식 observer 전용. 별도 paraphrase target set을 만들지 않음.
- History: 지금까지 받은 요청 중 latest-valid canonical target 전체. 첫1000에서는 최대900 Past이므로 sampling/GSS/recency 가중을 우선 사용하지 않는다.

Reference의 y0는 W0 행동 기준이다. C4 continuation을 factual QA 정답이라고 부르지 않는다. 현재 입력·reference·Past 사이의 scope conflict는 선언한 identity 규칙으로 기록하며, 의미적으로 모든 충돌을 탐지했다고 주장하지 않는다.

## 3. 최소 실행 흐름과 arm

### T0 — 재사용·수치·분석 경로 검증

모델 실험을 시작할 때 기존 native B1/teacher를 재사용할 수 있는지 확인한다. W0, WN, dataset/order, context, P/M, tokenizer, teacher, runtime numerical contract가 일치해야 한다. 새 source와 과거 source가 byte-identical할 필요는 없지만 변경된 경로의 parity 증거와 의미적 equivalence가 있어야 한다. 불일치하면 새 W0/B1 native를 한 번 만든다. 동일-state 증거 없이 과거 gradient/endpoint를 혼용하지 않는다.

기술 검사:

1. 실제 dense writer 및 FP32 Δ 재현.
2. 고정 full-token key로 계산한 H0+ΔK와 actual weight forward의 parity.
3. Q_E의 공간 결속·Current response/logit invariant.
4. 문서별 scalar margin의 activation gradient와 계수 gradient: 직접 AD 및 작은 central FD 대조.
5. 누적 교차항 계산, STEP=CUM at B1, basis 합으로 원 방향 복원.
6. 전체512 coverage, EOS/censor/argmax tie와 prefix position 정렬.

최소 기술 panel은 source hash 기반으로 고정한4reference+4Current이며, 큰 손상을 보이는 사례를 골라 numerical tolerance를 넓히지 않는다. 실제 full512 acceptance는 panel 검사로 대체하지 않는다.

### B1 — W0에서 B100, 4개의 고유 endpoint arm

|Arm|목적/방법|편집 보호|역할|
|---|---|---|---|
|N4|Native local-L4|native 자체|주 baseline|
|EN-KL-Q|기존 mean full-vocab KL, 한 projected gradient, 최대4후보|Q_E+Current guard|이전 EN 기준점|
|DEC-LINE|아래 choice-risk gradient 한 방향, 같은 후보 상한|Q_E+Current guard|목적 변경만으로 충분한지|
|DEC-MODES-CUM|functional gradient의 여러 성분+누적 activation 방향, joint coefficient solve|Q_E+Current guard|주 방법 후보|

DEC-MODES-STEP은 B1에서 CUM과 **동일한 문제**다. 별도 gradient·candidate·endpoint 실행 없이 alias로 기록한다. B1에서 이 둘이 다르면 누적 기준·입력·정규화·solver 결속 오류를 먼저 해결한다.

N4 preview, Q_E, reference capsule, reference marginal-gradient factors는 공유한다. EN-KL은 loss가 달라 별도 derivative가 필요하다. 기존 동일-state EN gradient/후보를 검증해 재사용할 수 있으면 EN을 다시 실행하지 않는다. 과거46.6분과 새 runtime 시간은 unmatched임을 명시하며 paired speed claim에 사용하지 않는다.

DEC-LINE과 MODES는 같은 위험·최종 acceptance 규칙을 사용한다. EN-KL은 기존 알고리즘의 목적·수용 규칙을 유지하므로 KL↔DEC 차이는 **목적과 그 목적에 따른 후보 수용 정책의 결합 비교**다. 순수 loss 한 요인 ablation이라고 과장하지 않는다.

### S3 — 같은 W0에서 시작한 B100×3

3개 chain: **N4 / DEC-MODES-STEP / DEC-MODES-CUM**. B1의 STEP/CUM 공통 commit은 원본을 복제해 독립 state로 분기한다. 이후 각 arm의 실제 entry에서 native z/write를 계산한다. 다른 arm의 z/gradient/history를 재사용하지 않는다.

B1의 source·policy·threshold가 바뀌지 않았으면 B1 checkpoint에서 B2를 이어서 실행한다. 이는 전체 chain이 W0에서 시작한 것이다. B1 결과를 보고 방법을 바꿨으면 새 version은 W0/B1부터 시작한다.

B2에는 N4의 own-native endpoint에서 STEP/CUM을 함께 계산하는 **same-entry shadow probe 1개**를 추가한다. 같은 reference factors와 functional4개 방향을 공유하되 다섯 번째 covariance 방향은 STEP/CUM별로 다르게 만든다. 두 후보를 실제 평가하되 N4 chain에 commit하지 않는다. 이는 누적 기준 차이와 서로 다른 trajectory 효과를 구분하기 위한 유일한 추가 paired probe다. 이 비용은 method throughput에서 분리한다.

### S10 — S3를 B100×10으로 연장

S3에서 기술적으로 유효하고 기능적 보존 신호가 관측된 경우에만 동일한3chain을 B10까지 이어간다. EN/DEC-LINE은 기본 sequential arm에서 제외한다. B1~B3를 반복 실행하지 않는다. 실패한 arm도 완료한 모든 batch와 fallback을 보고하며 좋은 arm만 남긴 final 평균을 만들지 않는다.

## 4. 실제 writer 기전 계측

입력 key K, 동일 module 좌표의 residual R, raw P, C_hist, λ를 저장한다. 실제 source의 방향/연산 순서에 맞춰

\[
M=\lambda I+P_{raw}(KK^T+C_{hist}),\quad
B=M^{-1}P_{raw}K,\quad \Delta_{alg}=RB^T
\]

를 분석한다. 실제 FP32 dense RHS는 solve(M,(PK)R^T)ᵀ처럼 연산 순서가 달라질 수 있으므로 actual Δ=FP64(WN)−FP64(Wentry)를 기준으로 차이를 기록한다. [기존 dense 재현 코드](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/alphaedit_factors.py:372).

P=UUᵀ인 ideal 분석에서는 X=UᵀK, C=λI+UᵀC_histU, C^(−1/2)X=LΣVᵀ로 두고 gain σ/(1+σ²), realized gain σ²/(1+σ²)를 사용한다. Exact-interpolation의1/σ 비용은 **조건부 비교량**으로만 보고한다.

Mode별 기록:

- σ_j, ||Rv_j||², actual/ideal write mode, realized residual/error.
- 실제 Δ norm, Δ/W0 norm, allowed leakage, solve residual.
- K는 request-mean native key와 full-token correction-lock key를 별도 이름으로 보관.
- current-current/현재-past/현재-reference geometry를 분리. B1 current-past는 `NOT_APPLICABLE`.
- W0 기준 E, Δ, 각 reference full-token의 B=EK와 V=ΔK, 정확한2〈B,V〉+||V||².

저비용 algebra control은 K/solver 고정, R의 request 열 permutation이다. Residual 전체 norm을 보존하면서 loading 변화가 write에 미치는 영향을 측정한다. 이것은 의미적으로 올바른 편집 arm이 아니며 RS/NS 개선 비교에 포함하지 않는다.

## 5. 판단 위험의 정의

Reference i의 고정 W0 continuation prefix에서 모든 generated position s의 full vocabulary를 평가한다.

\[
m_{is}(W)=z_{y^0_{is}}(W)-\max_{v\ne y^0_{is}}z_v(W),
\quad \mu_i(W)=\min_s m_{is}(W),
\]

\[
\boxed{\Phi(W)=\frac1{512}\sum_i[-\mu_i(W)]_+^2.}
\]

각 문서는 동일 가중치다. 최악 위치가 이미 정상 선택이면 추가 margin 회복 압력을 주지 않는다. 문서별 최악 위치만 미분하지만 **모든 위치·모든512문서를 forward 검사**하며 최종 개별 선택도 검사한다. 이 손실은 전체 token별 제약을 모두 선형화한 것과 다르다.

Base top-1과 다른 tie 선택은 Φ=0이어도 불일치다. 문서 tie는 작은 position, competitor tie는 봉인된 token ID 규칙으로 처리하고 실제 argmax 불일치 수를 별도로 센다. Tie-only 손상에서 gradient rule이 개선을 만들지 못하면 `TIE_ONLY_NO_DIRECTION`으로 기록한다. 성능을 보고 margin floor를 새로 추가하지 않는다.

Φ_R=Φ라 하고 아래 history 위험 Φ_H와 합친 Ψ=Φ_R+Φ_H를 방향 생성에 사용한다. 두 block은 각각 표본 평균을 낸 뒤 합한다. 이는 사전 고정한 equal-block convention이며 최적 가중치라는 주장이 아니다. Φ_R=Φ_H=0이고 tie 불일치도 없으면 correction을 생략한다. Reference가 안전해도 history가 손상됐으면 history gradient로 보정한다. Reference/Past가 모두 안전한데 N 손상이 존재하면 **현재 보호 신호의 부재**라는 결과다. 그 자리에서 KL·확률 floor·새 factual reference로 목적을 바꾸지 않는다.

## 6. 문서별 gradient factor와 제한된 다방향 후보

각 문서에서 현재 최악 위치/competitor의 gap을 고정해

\[
A_i=\nabla_{H_i}\mu_i(WN),\quad G_i=A_iK_i^T,
\quad G_\Phi=-\frac2{512}\sum_i[-\mu_i]_+G_i
\]

를 얻는다. 모든512문서의 A_i를 확보한다. A_i는 예측 위치만의 gradient가 아니라 downstream이 의존하는 **전체 valid input-token module-output gradient**다. Graph는 문서별로 폐기한다. Dense G_i512개를 보관하거나 매번 CPU로 전송하지 않는다.

History가 있으면 G_Ψ=G_Φ+G_H, 없으면 G_Ψ=G_Φ다. H=−G_Ψ Q_E의 상위3 singular component와 **H−그3성분의 합**을 최대4개 방향으로 사용한다. 마지막 residual을 포함하므로 H 전체가 후보 span에 포함된다. Dense full SVD는 사용하지 않고 randomized/operator top3를 사용한다. Seed20260919를 봉인하고 재투영·재정규화 후 원 gradient의 span 복원 오차와 잘린/중복 rank를 기록한다. 작은 residual을 버렸다면 exact 포함 대신 해당 tolerance 내 포함이라고 보고한다.

단순 DEC-LINE은 H 한 방향만 사용한다. 다방향이 실제로 rank1로 퇴화하면 넓은 탐색을 했다고 부르지 않는다.

C_R는 모든512문서의 전체 valid input keys로 만든 document-normalized covariance다.

\[
C_R=\frac1{512}\sum_i\frac1{L_i}K_iK_i^T.
\]

추가 방향은

\[
H_{step}=-\Delta_N C_RQ_E,\qquad
H_{cum}=-(W_N-W_0)C_RQ_E.
\]

STEP/CUM은 같은 기능 gradient4성분에 각각 이 방향1개를 추가한다. 따라서 최대5방향이며 차이는 **누적 E C_RQ_E가 제공하는 방향 정보**다. 계수의 부호가 자유이므로 이 방향의 부호나 정규화 전 norm을 복원 강도로 해석하지 않는다. 추가 전후의 residual norm, basis angle, effective rank를 기록하며 추가 방향이 기존 span에 있으면 실질적으로 동일 arm이다. 이 방향은 후보를 제공하는 통계일 뿐 최종 acceptance 목적이 아니다. C_R는 dense로 만들지 않고 문서별 `(E K_i)(K_i^T Q_E)/L_i` 등의 streaming action으로 계산하며 matrix 연산/I/O 비용을 계측한다.

기존 native metric에서 Δ_NQ_E≈0일 수 있다. H_step/H_cum까지0이면 해당 방향을 제외하고 원인을 기록한다. 이것을 모든 functional correction이 불가능하다는 증거로 해석하지 않는다.

방향은 Frobenius orthonormalize하고 실제 rank q≤5를 기록한다. 아래 후보는

\[
D(a)=\sum_{k=1}^q a_kD_k,\qquad W(a)=WN+D(a)
\]

이다. Native 자체를0.75배 하는 강도 메뉴가 아니라, 서로 다른 보정 방향의 계수를 공동으로 정한다. 전체 weight space의 최적해를 찾는 방법은 아니다.

## 7. 작은 계수 문제와 실제 후보 수용

저장 A_i/K_i에서

\[
J_{ik}=\langle A_i,D_kK_i\rangle_F
\]

를 계산한다. **512×q Jacobian을 얻기 위해 token별 backward나 두 번째 model backward가 필요하지 않다.** 같은 endpoint의 같은 scalar margin에 대한 factor contraction이다. Directional AD로 T0에서 검증한다.

초기 반경은 b=Ψ(WN)/||G_ΨQ_E||_F로 정의한다. 이는 first-order zero-loss scale에서 유래한 **proposal 반경**이며 유효한 보존 budget이나 안전 보장이 아니다. Ψ>0인데 G_ΨQ_E=0이면 b를 계산하지 않고 `NO_PROJECTED_DIRECTION`으로 종료한다. h가 매우 작아 b가 커지는 경우를 그대로 기록하며 backtracking의 필요성을 측정한다.

공동 제안은 다음 작은 convex QCQP로 만든다.

1. ξ_i≥0, μ_i+J_i a≥−ξ_i, ||a||₂≤b 아래 Σξ_i²/512를 최소화한다.
2. 모든 reference의 노출 row에 `μ_i+J_i a≥min(μ_i,0)`를 적용한다. 안전한 문서는0을 지키고, 이미 unsafe한 문서도 현재 worst deficit이 더 커지지 않게 한다. 이것만으로 같은 문서의 다른 token을 보호한 것은 아니다.
3. 아래 Current/Past의 선형화 가능한 조건을 추가한다. Past는 entry 기준의 hard condition이며 Exact-Q Current의 Jacobian은 이론적으로0이다. History 손상은 방향/radius 생성에도 포함되지만 최종 목표는 허용 범위 내 Φ_H=0이다.
4. 최적 risk 수준을 solver tolerance 내 유지하는 해 중 ||a||²가 가장 작은 해를 고른다. 목적의 우선순위를 가중치로 섞지 않는 lexicographic solve다. Float64 solver의 초기 objective-gap 허용은 `1e-12+1e-8*abs(f_star)`(mean squared-margin 단위), row feasibility는 `1e-8*max(1,abs(mu),b*norm(J_i))`(margin 단위)로 봉인한다. Solver precision과 actual neural acceptance tolerance는 구분하고, 최적값·primal residual·두 단계 gap을 저장한다.

이는 문서별 노출된 최악 pair에 대한 **local soft repair 문제**다. Slack이 남았는데 전체 base-choice constraint를 만족했다고 부르지 않는다. 안전했던 모든 position을 선형화한 것도 아니므로 실제 finite 검사에서 새로운 violation이 나올 수 있다.

Joint solution a*에서 최대4개의 actual 후보만 검사한다: a*, a*/2, a*/4, a*/8. Native에서 이미 손상된 history row의 margin h_j<0, 예측 증가 d_j=J_ja*>0이면 `alpha≥−h_j/d_j`가 필요하다. 이 선형 하한보다 작은 halve는 실제 평가 전 제외한다. 이는 nonlinear feasibility 증명이 아니며 skipped 후보 수를 기록한다. 각 후보는 immutable WN에서 `FP32(WN+D(a))`로 materialize한다. 첫 유효 후보를 선택하며 거절 후보를 N으로 비교해 바꾸지 않는다.

수용은 다음을 모두 요구한다.

- Actual Current guard와 Q_E invariant 통과.
- 아래 active-history guard 통과.
- Native에서 W0 token을 올바르게 선택하던 **모든 reference position**에서 신규 불일치0.
- 각 reference 문서의 worst deficit이 native보다 numerical tolerance 이상 커지지 않음. Φ_R 비증가, Ψ의 수치적으로 의미 있는 감소, Φ_H의 entry 허용조건 통과를 요구. Argmax 불일치 전체 수 증가0도 확인.
- 전체512/실제 valid 위치/full vocabulary 검사 완료.

현재 unsafe 위치의 margin이 개선돼도 flip 개수가 같을 수 있다. 이런 수용은 `MARGIN_ONLY`로 표기하고 choice 복구와 분리한다. Native safe position no-new-flip 조건이 수용을 막으면 `REFERENCE_NEW_FLIP`으로 기록한다. 이후 허용 손실을 임의로 넓히지 않는다.

Decision 수치 기준은 별도로 봉인한다. 문서 worst-deficit 비증가의 허용량은1e−4nats, 위험의 의미 있는 감소량은 `tau_risk=1e-10+1e-6*max(Psi_native,Phi_R_native)` nats²로 둔다. 실제 token-ID no-new-flip에는 이 허용량을 적용하지 않는다. 두 위험이 정확히0이면 no-op, 양수지만 tau_risk 이하이면 `BELOW_RISK_RESOLUTION`으로 구분한다. Projected-gradient zero 기준은 `||GQ_E|| <= 1e-12*||G||`(G=0 별도)다. 이 값은 방향 생성 규칙의 수치 분기이며 보편적 capacity 판정이 아니다. T0의 repeated identical-endpoint margin/risk 변동이 위 허용량의1/10을 넘으면 원인을 해결하고 새 기술 version을 봉인한다. Observer 성능을 보고 허용량을 늘리지 않는다.

최대4후보가 모두 실패하면 native fallback이다. 한 선형화와 유한 후보 탐색의 실패이며 Q_E 전체의 불가능성이나 수렴 판정이 아니다. Main에서 자동으로8후보/추가 gradient를 붙이지 않는다. 재선형화가 필요한지 판단할 자료로 worst-pair 전환·linear prediction error를 남긴다.

## 8. Current와 history의 정확한 보호 기준

Current 기준은 동일 arm/entry에서 만든 own-native다. Canonical 및 native rewrite contexts의 제공된 new/old teacher-forcing 경로 전체를 K_E에 포함한다. 공식 paraphrase는 포함하지 않는다.

- Native 성공 canonical preference ID lost0.
- Native 성공 target TF-strict ID lost0.
- 요청별 target mean NLL 증가≤사전 봉인된 numerical tolerance.
- Actual DK_E, allowed leakage, full-logit invariant가 기존 EN의 검증된 기술 기준 내.

Initial proposed NLL tolerance는1e−4nats다. 이는 semantic 허용량의 최적값이 아니라 기술 계약이며, 반복 동일 forward의 오차가 이 값을 넘으면 tolerance를 키워 성능을 통과시키지 않고 runtime 원인을 해결한다. 다른 invariant threshold는 기존 실행 contract 값을 명시적으로 계승하고 run receipt에 복사한다.

History registry는 성공 여부와 무관하게 모든 도착을 기록한다. Subject/relation identity의 최신 version을 active로 하고, 이번 batch가 overwrite하는 과거 version은 보호 대상에서 제외한다. Native C_hist는 registry와 별개이며 이 설계가 superseded contribution을 자동으로 빼지는 않는다.

History j의 scalar safety slack h_j(W)는 `(entry desired NLL + epsilon − current desired NLL)`, entry에서 preference 성공했으면 그 pairwise margin, entry에서 TF-strict 성공했으면 현재 desired-target token들의 최저 choice margin의 최솟값이다. 현재 없는 branch는 만들지 않고 그 보호 범위를 명시한다. Φ_H는 `mean_j [-h_j(W)]_+^2`이며 Past가 없으면0이다. 이는 baseline W0의 과거 답변을 복원하는 loss가 아니다. 각 Past의 worst slack derivative를 확보해 G_H와 coefficient constraint에 사용한다. Tie의 실제 ID 검사는 별도다.

한 history request가 pairwise NLL을 요구하면 new/old의 서로 다른 teacher-forcing 경로가 필요하다. 따라서 history request900개를 모델 sequence900개 또는 backward900회로 바꾸어 쓰지 않는다. 경로 p별 factor를 유지하고 `J_jk=sum_p inner(A_jp,D_k K_jp)`로 합산한다. Request 수, unique TF path 수, valid token 수, actual forward/backward 호출을 각각 기록한다. 한 scalar를 두 branch의 합으로 만든 graph가 두 경로의 실제 연산을 없애는 것은 아니다.

최종 guard는 **전체 active Past canonical**에 대해 실행한다. 기준은 batch entry이며 성공 preference/TF-strict ID lost0, request별 desired NLL 증가≤tolerance다. 이미 entry에서 실패한 Past도 기록하며 그 실패를 성공 집합에서 숨기지 않는다. Native→candidate뿐 아니라 entry→native와 entry→selected를 분리한다.

Past의 일부가 native에서 이미 손상되면 a=0이 history 조건을 만족하지 않을 수 있다. 초기 제안에서는 해당 Past의 coefficient-space NLL/margin row를 추가한다. 필요한 derivative는 prefix-cache와 고정 방향으로 계산하고 Current와 달리 추가 neural cost로 센다. 과거 전체를 exact key-lock하지 않는다.

History rows와 reference 문제의 **local QCQP** 교집합에 infeasibility certificate가 있으면 `LOCAL_HISTORY_CONSTRAINT_INFEASIBLE`이다. 제한된 finite 후보가 모두 실패한 경우는 `FINITE_SEARCH_UNRESOLVED`이며 원문제 infeasible이라고 부르지 않는다. Native fallback이 entry-history 조건을 위반하면 **`FALLBACK_WITH_PAST_VIOLATION`**으로 표시한다. Fallback을 보호 성공으로 집계하지 않는다. Silent skip, current batch 원상복구 후 성공으로 표시하는 정책은 없다.

## 9. 기전 개입: observer 전용, 후보 선택과 분리

B1 candidate selection과 weight hash를 봉인한 뒤 수행한다. Mandatory full512 평가를 일부 panel로 대신하는 절차가 아니다.

- 모든 reference에 대해 실제 response norm·교차항·margin·signed native derivative를 기록한다.
- 추가 frozen-suffix 개입은 고정32reference panel과32N panel로 제한한다. Reference panel은 reference-only 위험/변위 strata와 stable hash로 정하고, N panel은16lost+16stable을 사후 선택할 수 있으나 **posthoc mechanism-only**로 표기한다. 부족하면 deterministic rule로 남은 그룹을 채운다.
- Native actual/ideal parity가 확인된 mode에서 target-loading/write 비용 상위2개와 norm/rank를 맞춘 deterministic random control2개, 총4개의 global component intervention을 만든다.
- 모든 panel 입력에 동일 Δ_j를 빼는 H_N−Δ_jK_x를 적용한다. Input별로 다른 위치만 복원하는 patch와 구별한다.
- 이 개입의 Current 품질도 평가한다. 편집 실패를 동반한 N 회복은 method 성공이 아니다.
- Panel에서 유리했던 component를 최종 선택 arm에 소급 적용하지 않는다. 이를 이용해 만든 다음 version은 별도 cold 시작과 검증이 필요하다.

기하학↔write만 연결되면 writer mechanism, panel patch만 좋아지면 activation possibility, actual common weight에서 Current와 독립 N/PS를 함께 개선하면 method evidence로 등급을 구분한다.

## 10. 누적 손상·망각의 평가 cadence

모든 수용 후보의 controller는 reference512/Current/active Past만 본다. Official observer는 선택 봉인 후 실행한다.

|시점|필수 관측|
|---|---|
|각 batch entry|해당 batch current N, active Past canonical; 이미 저장된 이전 패널 상태|
|own-native, 선택 후 observer|current R/P/N; reference512; entry→native의 손상|
|selected endpoint 매 batch|current R/P/N, reference512, 전체 active Past canonical; entry→selected 및 native→selected|
|B1|Dev128 관측; 이미 측정한 current R/P/N은 all-seen과 동일|
|B3/B5/B10|all-seen R/P/N와 Dev128, 반복된 N panel|
|W0|해당 실험 전체1000요청의 R/P/N base 관측은 observer가 보관하고 selector에는 미전달|

핵심 지표:

1. Current RS/PS/NS, teacher-forced strict, request-level R+두P joint.
2. 동일 N의 entry→native/selected 즉시 손상, W0-correct gross lost/recovered.
3. At-write→checkpoint의 이전 R/P/N 유지와 최신 active edit 유지.
4. Reference token choice mismatch 수, 문서 전체 일치율, Φ, margin erosion, p95/p99/max와 신규손실/복구.
5. Dev128의 동일 choice/Φ/KL. Full-vocab KL은 selected endpoints B1/B3/B5/B10에서 observer로만 계산해 candidate마다 full teacher를 읽지 않음.
6. E, Δ_N, D, 최종Δ의 norm과 reference action. Path-length 합을 cumulative net displacement로 대체하지 않음.

B3/B5/B10 사이 모든 과거 N의 망각 시점은 관측하지 않는다. 이를 보완하기 위해 첫 batch의 고정1000N panel은 모든 commit에서 추적한다(현재 batch/all-seen과 중복이면 캐시 재사용). 이는 반복 패널이며 전체 lifelong N의 완전한 시점별 추적은 아니다.

Full-vocab KL/official observer 비용은 method selection 시간과 분리하고 총 program 비용에도 포함한다. Generic C4 선택 보존과 factual N, 일반 능력을 서로 대체하지 않는다. 일반 능력의 광범위 보장은 이번 pilot 범위 밖이며, 후속 주장을 하려면 task benchmark가 추가로 필요하다.

## 11. 진행·중단 판정

세 가지 판정을 별도로 기록한다: technical validity, controller 효과, 독립 functional 효과.

**B1→S3:** primary MODES의 비영 actual correction, Current 조건, 모든512 coverage, 적어도 하나의 실제 choice 복구 또는 Φ 감소가 수치 오차를 명확히 넘는지 확인한다. 기본 투자 기준으로 Φ 상대감소≥5%를 사용한다. 5%는 이론적 threshold가 아니라 사전 고정한 자원 배분 기준이며, margin-only이면 약한 signal로 표시한다. Official PS/strict/joint가 N4보다 낮으면 현 version은 편집 무손실 성공으로 부르지 않고 S3 확장을 중단한다. N의 B1 회복0만으로 즉시 불가능성을 선언하지는 않지만 이를 명시한다. Official observer에 따른 중단은 development gate이며 online candidate 선택이 아니다.

**S3→S10:** 판정 대상은 primary **CUM vs N4의 B3 all-seen**이다. CUM의 reference/Current/history 조건을 충족하고 PS/P-strict/R+두P joint의 점추정 손실이 없어야 한다. W0-correct N gross loss가 N4보다 줄거나 actual recovery가 관측되어야 한다. 통과하면 STEP은 성능과 무관하게 대조군으로 함께 연장한다. STEP만 좋아진 경우 CUM 성공으로 대신하지 않는다. 단, 어느 arm의 수치·실행 무결성 실패도 먼저 해결해야 한다. Gate 실패 시10batch로 늘리지 않으며, CUM과 STEP이 구별되지 않더라도 기록된3batch 결과를 숨기지 않는다.

**S10 연구 판정:** primary는 final W0-correct N retention 및 all-seen NS 개선이다. RS·PS·P-strict·joint·active-history는 별도 무손실 점추정 조건으로 보고한다. 한 순서1000개의 점추정 조건은 통계적 non-inferiority 증명이 아니다. 개선이 있어도 개발 증거이며 새 order/stream 확인 전 일반화된 방법 우위를 주장하지 않는다.

Uncertainty는 request를 cluster로 두고 보고한다. 한 요청의10neighbor를 독립 episode로 세지 않는다. Native/arm paired ID를 유지하고 paired difference의 interval을 계산한다. Reference bootstrap과 edit-stream variation도 구분한다.

## 12. 계산 상한과 실질적 단순화

- Native z: arm 자신의 batch entry에서 request당1회. Correction z0회.
- Decision reference backward: **한 번의512문서 sweep**, 최악 pair scalar당1회. 결과 A_i를 저장해 LINE/MODES/STEP/CUM의 같은-entry 계수 Jacobian에 재사용. History derivative는 최대900개의 active request에 추가로 필요하며, new/old canonical branch가 각각 필요하면 최대1800 TF 경로가 된다. 동일 경로는 중복 제거하되 경로와 token count를 기록한다.
- Main coefficient dimension≤5, nonlinear center1개, arm당 actual reference candidate 최대4회. **각 실제 candidate의 전체 active-history branch 검사 비용은 이 reference sweep 수와 별도**다. 네 후보 외 재선형화·무제한line search 없음.
- EN-KL은 별도1gradient sweep, 최대4후보. 동일-state 과거 산출물을 검증해 재사용 가능하면 생략.
- Initial A_i factor 저장 상한은 Σ L_i×4096×4bytes로 계산(현재 길이면 약3.2GB 규모). Exact capsule 길이로 사전 계산하고 RAM/disk ledger에 포함한다. 512×전체W gradient를 저장하지 않는다.
- Decision 목적은 W0 token IDs만 필요하므로 candidate loop에서 full-vocab teacher 확률 파일을 읽지 않는다. Full-vocab **model logits**는 계속 계산한다.
- 최악 pair를 full-vocab forward에서 정한 뒤 해당 hidden state와 LM-head의 두 row로 gap을 다시 구성하여 backward할 수 있다. T0에서 원래 full-logit scalar AD와 일치함을 확인한다. Full-vocab competitor 검색의 forward 비용은 남는다.
- C_R action, low-rank factorization, Q_E, factor contraction, physical validation도 시간·메모리에 포함한다. 작은 계수 수를 근거로 모델 backward가 없다고 말하지 않는다.
- Per-stage soft wall review는 correction/native 비율로 공개한다. 실용성 투자 기준은 S3에서 median correction≤2×own-native time. 이는 실측 전 예측이 아니며 초과하면 우수한 품질이 있더라도 `QUALITY_SIGNAL_COST_UNRESOLVED`로 구분한다. 실행 중 작업을 임의 중단하는 timeout과 혼동하지 않는다.

GSS/recency는 이번 primary에서 off다. 최대900 Past에서 먼저 전체 보호 효과를 확인한다. 이후 필요할 때 frozen directions/constraints의 working-set 순서와 full constraint 검사로 별도 비교한다. Reference512를64개로 줄여 얻은 속도 개선으로 보고하지 않는다.

B1의 최악 계산량은 EN1회와 Decision1회의 reference derivative sweep, EN/LINE/MODES 각최대4회의 candidate sweep이다. EN을 검증해 재사용하면 새 계산은 Decision1 derivative sweep과 LINE/MODES 최대8 candidate sweeps다. STEP alias는0회다. 이는 pilot 전체 비용이며 한 method의 per-batch 비용으로 나누어 표시하지 않는다. S3/S10의 STEP와CUM은 분기 뒤 각각 derivative/후보 비용을 갖는다. 모델 backbone, geometry, native, observer가 같지 않은 상태의 결과는 재사용하지 않는다.

두 가지 비용 회계를 함께 남긴다. **실제 연구 총비용**에서는 공통 sweep을1회만 센다. **각 method의 standalone 비용**에는 그 method에 필요한 공통512 derivative, geometry, factor cache/Jacobian/basis 비용을 전액 포함한다. 2×own-native 실용성 기준은 후자를 사용하며 공유 비용을 arm 수로 나눠 낮추지 않는다. Standalone accounting은 별도 독립 run wall-time 실측과도 구별한다.

## 13. 조건부 후속 진단과 결과별 해석

|관측|해석/다음 작업|
|---|---|
|Projected target demand와 actual write가 연결되지 않음|H1 수정. Condition number 기반 방법을 진행하지 않음|
|Reference activation action 감소, decision/N 악화|Proxy mismatch. Covariance 축소를 주 방법으로 채택하지 않음|
|DEC-LINE만으로 개선|목적/수용 정책의 변화가 주효. 다방향 배분 novelty를 주장하지 않음|
|MODES가 LINE보다 개선|같은 정보에서 한 방향 제약의 비용을 지지. 추가basis/solver비용까지 비교|
|Reference 개선, Dev/N 개선 없음|Reference relevance/generalization 문제. 더 긴stream을 해결책으로 삼지 않음|
|Reference 새 flip 때문에 모두 거절|제약과 후보공간의 tradeoff. 허용 손실을 사후 변경하지 않음|
|Q_E에서 효과없고 projected functional gradient가 작음|별도 DEC-FUNCTIONAL 진단 고려. Hard response lock의 비용과 다른 원인을 구분|
|CUM만 STEP보다 개선|누적 방향 정보의 추가 가치. Same-entry probe와 own-trajectory를 함께 확인|
|STEP=CUM 또는 cumulative direction0|이번조건에서누적정보의추가이득없음. Equality/degenerate basis를확인|
|Current는 유지하지만 PS 하락|선언한 입력 밖 일반화 문제. No-paraphrase-target 정책 유지, 성능교환으로 보고|
|Past entry 조건이 infeasible|Current/active-history 공동 제한으로 기록. Nativefallback을성공으로표기하지 않음|

DEC-FUNCTIONAL은 **기본 arm 아님**. 필요하면 같은 raw reference gradient를 P 허용 공간에서 사용하고 current NLL/margin의 coefficient Jacobian과 실제 finite guard를 넣는 cold 진단1개를 만든다. 공식PS를constraints에넣지않으며, 실험본을v1과분리한다. Tangent 보존이 finite 보존이라는주장도하지않는다.

## 14. 저장할 산출물

- Immutable run/source/model/data/reference manifests, RNG, native config and P/M bindings.
- W0/current/native/selected L4 weights 또는 정확한복원가능delta; B1/B3/B5/B10과각commit을복원할state ledger. Full8B checkpoint 복제는필수아님.
- Native K/R/solver factors; allowed basis; full-token lock geometry; residual loading spectrum; actual realization errors.
- Per-reference position/token/competitor/margin/choice/NLL, Φ와문서별gradient factor provenance.
- Candidate basis rank, singular decomposition residual, coefficients, QCQP/KKT/feasibility receipts, predicted/actualrisk, everyrejectreason.
- E/Δ/D의정확한cross-term표와 response statistics; mode intervention은별도observer폴더.
- Current/Past registry, overwrite/superseded counts, entry/native/selected comparison.
- Sealed selection ledger와그이후의R/P/N/Dev observer,pairedgrosslost/recovered표.
- Timers:prepare/native/geometry/referenceforward/backward/factorcontraction/solver/candidate/current/past/observer/I/O및peakmemory.

이번 설계의 성공은 **동일한 강한 편집을 유지하면서 제거 가능한 functional 손상이 실제로 존재하고, 그 성분을 정량적인 방향 선택으로 줄였다는 결과**다. 누적 quadratic, projected optimization, spectral basis 자체를 새 원리라고 주장하지 않는다.
