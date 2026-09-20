# EN approximate null-space: cold B100 → sequential B300

2026-09-20. 상태: **실험 규약 및 CPU selector reference 구현**. 모델 runner 구현·GPU 제출·새 모델 성능 측정은 아직 수행하지 않았다.

본 실험은 margin controller를 사용하지 않는다. EN의 reference KL 목적을 유지하고, current edit-response를 정확히 고정하는 조건을 **정량적인 activation-response 예산**으로 완화한다. 첫 목표는 새로운 방법의 성공 선언이 아니라, exact EN의 보호 공간이 실제 locality 개선을 제한했는지 구분하는 것이다.

선행 근거: [null-space 상세 재계산](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-en-nullspace-threshold-review/review-ko.md), [B1 독립 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/review-ko.md), [write 강도 분석](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-single-layer-write-strength-analysis/analysis-ko.md).

기계 판독 규약: [contract](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-en-adaptive-nullspace-contract-v1.json). 실행 cell: [CSV](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-en-adaptive-nullspace-cells-v1.csv).

## 1. 무엇을 비교할 것인가

|Arm|공간|목적|B1|B2–B3|
|---|---|---|---|---|
|N4|native L4 write|강한 baseline|공유 native|독립 chain|
|EN_EXACT|기존 exact edit-null Q0|기존 편집 반응 고정|실행|독립 chain|
|EN_NUM|동일-prefix 변형이 만드는 numerical tail까지만 해제|수치적 과보호 제거|실행|실행하지 않음|
|EN_ADAPT|gradient·편집 반응 예산으로 cutoff 선택|실제 edit-response 완화|실행|독립 chain|

모든 EN arm은 **같은 KL objective, 같은 native center, 같은 step 제안/수용 controller**를 사용한다. EN_EXACT는 이전 4-trial EN의 역사적 실행과 동일한 알고리즘이라고 부르지 않는다. 이번 공통 2-evaluation controller 아래의 exact-space 대조군이다. 과거 EN 결과는 참고치로만 사용한다.

EN_ADAPT는 한 layer 내 보호 공간의 threshold를 선택한다. Layer allocation, softmax layer weight, 별도 local-z 재최적화, native scalar 0.75, output-margin floor, DEC의 reference별 무손실 제약은 넣지 않는다.

KL-P는 모델 arm을 추가하지 않고 **geometry/gradient 상한 진단**으로 포함한다. ADAPT가 P 전체를 선택하면 그 사실을 명시한다. 그런 결과를 정교한 spectral 분리가 필요했다는 증거로 쓰지 않는다.

## 2. 고정할 native·데이터·시작점

- Llama-3-8B-Instruct, 기존 pinned revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`. 실행 전 기존 실제 manifest와 재대조.
- 수정 weight: zero-based `model.layers.4.mlp.down_proj.weight`, shape4096×14336 하나.
- 시작은 **pretrained W0, native history M4=0**. 5000-edit 상태에서 시작하지 않는다.
- Native의 local-L4 z·context·raw P·L2=1·loss stopping·FP32 write 순서는 기존 cold baseline 그대로 유지한다. 기존 `clamp_norm_factor=0.75`는 native target 최적화의 설정으로 유지하며, native write를 추가로0.75배 한다는 뜻이 아니다.
- 편집은 기존 fixed10k 봉인 순서의 첫300, batch100. 여러 번 사용한 개발 stream임을 명시하며 새 confirmatory evidence라고 부르지 않는다.
- Reference는 **기존 C4 512개 전체**, S64+Reserve320+AdditionalTrain128. 이번 비교에서 factual QA bank로 바꾸지 않는다. C4 continuation을 factual 정답이라고 부르지 않는다.
- W0 teacher: `max_new_tokens=256`, EOS 조기 종료. 같은 teacher prefix에서 모든 valid generated position의 full-vocab KL을 사용한다. 길이16/64 제한으로 돌아가지 않는다.
- 동일 capsule의 예상 train 위치130,235는 검증용 정보이며 manifest가 다르면 이를 강제하지 않는다.
- Dev128, 공식 PS/N은 selector·gradient·threshold·line search에 접근하지 않는다. 별도 paraphrase 학습 set을 만들지 않는다.
- Native raw projector의 threshold와 허용 공간 P*는 변경하지 않는다.

고정 key를 재사용하려면 token IDs, mask, position IDs, padding, tokenizer/BOS/EOS, context, upstream weights와 실행 정책의 identity가 같아야 한다. 다른 chain이라도 이 입력과 L4 이전 parameter는 고정이므로 prefix key cache는 공유할 수 있다. L4 이후 activation, z, gradient, candidate loss는 서로 다른 W에서 공유할 수 없다.

## 3. 저장된 B1을 사용하는 정확한 방식

Source reference는 Server4의 다음 완료 run이다.

`/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4/output/B1`

저장 keys, provenance, P* basis, native target captures, writer factor는 identity 대조 후 재사용한다. 원 run에는 full selected W/M와 EN-KL raw gradient가 저장되어 있지 않다.

- Full native WN을 재현하려면 원 raw P·M·dense RHS 연산 순서와 FP32 rounding을 사용하여 original native hash까지 대조한다.
- 저장 `R B^T` 근사만으로 original WN이라고 선언하지 않는다. 기존 재구성 상대차 약6.07e-6은 threshold 비교에서 혼용할 이유가 아니다.
- 원 native endpoint를 정확히 재구성하지 못하면 W0에서 native B100을 **한 번만** 새로 실행하고 네 arm이 공유한다.
- Reference KL gradient는 동일한 새 native에서 한 번 계산해야 한다. DEC activation-gradient factor는 KL gradient가 아니므로 대체하지 않는다.
- 오래된 raw를 덮어쓰지 않고 새 run directory에 동일성·재사용 여부를 기록한다.

## 4. Current covariance: 반복 수와 key byte 변화에 의존하지 않게 정의

Exact null space는 양의 column weight에 불변이지만 approximate 공간은 그렇지 않다. 따라서 첫 실험부터 모든 arm이 같은 명시적 가중을 사용한다.

요청 i의 기존 current-protection 입력들 중 동일한 전체 입력 identity를 합친 집합을 S_i라 하자. Canonical/native context와 supplied old/new teacher-forced branch는 기존 oracle 범위대로 포함하고, 공식 PS/N은 제외한다. 각 sequence s의 padding을 제외한 input-token 수를 L_is라 한다.

\[
C_E=\frac1B\sum_{i=1}^{B}\frac1{|S_i|}
\sum_{s\in S_i}\frac1{L_{is}}K_{is}K_{is}^T.
\tag{1}
\]

요청별 총 weight가1/B이므로 긴 요청이나 중복 context가 임의로 우세해지지 않는다. 기존 captured byte-column K_E에 alias weight를 누적하면

\[
C_E=K_E\Omega K_E^T=\bar K_E\bar K_E^T,\qquad
\bar K_E=K_E\Omega^{1/2},\quad\sum_a\omega_a=1.
\tag{2}
\]

Full input identity에는 token IDs·mask·position IDs·padding/BOS 정책을 포함한다. 같은 token prefix라도 실제 captured key bytes가 다르면 geometry 단계에서 임의로 병합하지 않는다. 모든 원 column을 보존하고 weight와 numerical-tail receipt를 별도로 만든다.

`current-protection-manifest.json`은 case→sequence→cache→position→actual_key_column과 각각의 weight를 담아야 한다. Mapping을 복원하지 못하면 균등 column weighting으로 조용히 대체하지 않는다. 올바른 oracle manifest를 먼저 생성한다.

**기존 raw matrix의 281개 tail·9,730차원 숫자는 사전 근거다.** 이번 request-normalized covariance의 singular value/cutoff는 다시 계산한다. 양의 column weighting은 exact span을 보존하지만 approximate eigenspace는 바꿀 수 있으므로, historical threshold 숫자를 새 공간에 그대로 사용하지 않는다. 모든 신규 arm에 같은 covariance를 사용하므로 이번 arm 간 차이는 공간 완화 정책이다.

## 5. Numerical tail과 method threshold를 분리

P*=VV^T, V^T V=I에서

\[
X=V^T\bar K_E=U\Sigma Z^T,\qquad\lambda_j=\sigma_j^2.
\tag{3}
\]

K K^T를 만들어 극소 eigenvalue의 numerical rank를 추정하지 않는다. 기존 TSQR/SVD와 FP64 geometry를 사용하고 SVD는 batch/geometry당 한 번 수행한다.

**EN_EXACT:** 기존 `c_rank=max(shape(X))*eps64*sigma_max` 및 ambiguity diagnostics를 사용한다. 이전 exact-space projector와 weighted exact projector의 차이를 검사한다.

**EN_NUM:** 동일 logical prefix/mask/position group의 deterministic 첫 captured key를 대표로 택하여 K_rep를 만든다. 데이터를 치환해서 실행하지 않고 차이의 증거만 사용한다.

\[
E_{dup}=V^T(K_E-K_{rep})\Omega^{1/2},\qquad
c_{num}=\max(c_{rank},\|E_{dup}\|_F).
\tag{4}
\]

Sigma≤c_num인 방향을 해제한다. 이는 동일-prefix 변형을 제거한 낮은-rank matrix까지의 관측된 차이를 이용한 보수적인 cutoff다. 모든 forward 오차에 대한 보편적인 noise bound는 아니다. 대표를 첫 key로 정한 규칙과 group 수·norm·gap을 기록한다. 중복 변형이 없으면 EN_NUM은 exact와 alias가 될 수 있다. 모든 batch에서281개를 자르거나 largest-gap만 무조건 선택하지 않는다.

**EN_ADAPT:** exact 공간부터 P 전체까지의 중첩된 spectral subspace를 고려한다. 동일 singular-value group을 임의로 반으로 자르지 않는다. Boundary는 숫자 tau뿐 아니라 blocked rank와 basis identity로 저장한다. c_num보다 큰 비영 방향까지 해제했는지 별도 표시한다.

## 6. Method: reference 감소 방향과 실제 response 예산으로 cutoff 선택

Native endpoint와 actual native write를

\[
W_N,\qquad \Delta_N=\operatorname{FP64}(W_N)-\operatorname{FP64}(W_{entry})
\]

로 고정한다. B1 목적 J는 reference 전체의 document-normalized forward KL이다.

\[
J(W)=\frac1{512}\sum_i\frac1{T_i}
\sum_s KL(p_{W_0}(\cdot|c_{is})\Vert p_W(\cdot|c_{is})).
\tag{5}
\]

Native에서 J_N=J(W_N), G=grad J(W_N)를 한 번 얻는다. 보정 norm 상한과 편집 반응 기준은

\[
\rho=\|\Delta_N\|_F,\qquad A_N=\|\Delta_N\bar K_E\|_F.
\tag{6}
\]

Correction norm≤native norm은 이번 pilot의 공통 탐색 상한이다. 최적성이나 semantic 안전을 보장하는 값이 아니다. 이 상한이 실제로 active했는지 반드시 기록한다.

Response 예산의 primary 값은 **epsilon=0.05**로 사전 고정한다.

\[
\boxed{\|D\bar K_E\|_F\le\epsilon A_N.}
\tag{7}
\]

이는 native가 만든 L4 편집 반응 RMS에 비해 보정의 RMS를 최대5% 허용한다는 뜻이다. Weight를5% 또는95%로 자르는 규칙도, RS5% 손실 허용도 아니다. Epsilon은 해석 가능한 실험 hyperparameter이며5%가 최적이라는 주장을 하지 않는다. Epsilon0.01/0.10은 먼저 같은 G로 **algebra-only sensitivity**를 계산한다.

각 후보 projector Q_j에 대해

\[
H_j=GQ_j,\quad g_j=\|H_j\|_F,\quad a_j=\|H_j\bar K_E\|_F
\]

를 얻고, g_j>0이면

\[
\boxed{
\eta_j=\min\left(\frac{J_N}{g_j^2},\frac\rho{g_j},\frac{\epsilon A_N}{a_j}\right),
\qquad D_j=-\eta_j H_j.
}
\tag{8}
\]

a_j=0이면 마지막 상한은 infinity다. A_N=0이면 비영 current-response 방향의 step은0이며, zero-response 방향은 허용한다. J_N이 numerical floor 이하이거나 g_j=0인 경우는 명시적 no-signal outcome이다.

첫 항은 기존 EN의 nonnegative KL을 이용한 first-order zero-loss step이다. 둘째와 셋째는 norm 및 response 예산이다. **Native gradient에서 예측한 KL 감소량**

\[
p_j=\eta_j g_j^2
\]

을 최대화하는 j를 선택한다. 동률이면 correction norm eta_j g_j가 작은 것을, 그다음 response eta_j a_j가 작은 것을, 그다음 더 엄격한 cutoff를 택한다. 부동소수점 tie band는 `64*eps64*max(1,max(abs(values)))`다. 이 규칙은 사후 NS나 PS를 보지 않는다.

이는 **중첩된 projected-gradient 후보들에서의 일차 선택**이다. 전체 weight space나 실제 nonlinear KL의 최적해라고 주장하지 않는다. P 전체가 선택되거나 exact/NUM과 동일해지는 경우도 결과로 보존한다.

### 모든 cutoff를 추가 backward 없이 계산하는 방법

\[
e_0=\|GQ_0\|_F^2,\qquad e_j=\|GVu_j\|_2^2.
\]

허용할 spectral component를 작은 lambda부터 누적하면

\[
g_k^2=e_0+\sum_{j\le k}e_j,\qquad
a_k^2=\sum_{j\le k}\lambda_j e_j
\tag{9}
\]

이다. 따라서 모든 cutoff·세 epsilon의 eta/p/norm/response를 작은 배열의 누적합으로 계산한다. Near-null을 수치상0으로 취급한 항의 residual은 실제 K action으로 별도 확인한다. Exact residual을 표현할 때 큰 두 norm의 차감으로 e0를 계산하지 않고 직접 residual norm을 사용한다.

보관할 결과는 lambda, e_j, e0, native norms, 각 boundary의 objective prediction·cap activation·선정 이유다. 본래 EN-KL raw gradient가 없어서 생겼던 audit 공백을 이런 작은 통계로 메운다. Full G는 B1 arm 평가 동안 RAM에서 한 번 공유하며 per-reference dense gradient512개를 저장하지 않는다.

### 정확한 보장의 범위

Fixed single-layer key에서 finite D의 반응 변화는 정확히 D K다. Q_tau에 제한된 D는

\[
\|D\bar K_E\|_F^2\le\tau\|D\|_F^2
\]

도 만족하지만, controller는 그보다 직접적인 **실제 projected-gradient action a_j**를 사용해 식(7)을 지킨다. RS/PS 보존은 이 부등식에서 따라오지 않는다. Request별 변화의 p50/p95/max를 함께 보고하여 평균 예산이 특정 요청의 큰 변화를 숨기는지 확인한다.

## 7. 두 번의 실제 후보 평가로 step 결정

모든 EN arm에 공통이다. B1 arm마다 z 재계산·gradient refresh는0회다.

1. 선정 D1을 immutable WN에 한 번 더해 FP32 candidate를 만든다. `actual_D=FP64(Wcand)-FP64(WN)`로 실제 norm·response·P leakage를 검사한다.
2. 모든512 reference의 실제 J1을 평가한다. 같은 forward에서 base-choice disagreement·target log-prob·문서별 KL도 저장하되 selector에는 J만 사용한다.
3. D1을 고정한 ray `W(t)=WN+t D1`, t∈[0,1]에서 d=<G,D1><0라 두고 `h=2*(J1-JN-d)`를 계산한다.
4. h>0이면 quadratic prediction의 optimum `t2=clip(-d/h,0,1)`을 두 번째 후보로 사용한다. h≤0이거나 t2가0/1과 같으면 새 endpoint를 만들지 않는다. T2가 매우 작아 FP32 endpoint가 native/첫 후보와 같으면 cache hit로 끝낸다.
5. Native와 최대 두 후보 중 geometry/finite 값이 유효하고 **실제 KL이 감소한** 후보를 선택한다. Armijo는 `Jcand≤JN+1e-4*<G,actual_D>`이며 실제 감소도 별도 요구한다. 통과 후보 중 J가 작은 것, 동률이면 norm이 작은 것을 택한다. 모두 실패하면 native로 복귀한다.

Finite quadratic curvature는 근사이고 두 점으로 global optimum을 보장하지 않는다. 두 후보 모두 실패하면 `SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE`로 기록하며 공간 전체가 무효라고 해석하지 않는다. 이전 실제 EN이 scale0.125에서 처음 성공했으므로 단순히 처음 두 scale1/.5만 실행하는 방식은 사용하지 않는다.

Ideal correction은 식(7)을 만족한다. FP32 rounding error E_round=actual_D−D_ideal를 따로 계산하고, 실제 response는

`||actual_D Kbar|| ≤ epsilon A_N + ||E_round Kbar||`

로 수치 오차와 의미적 예산을 분리한다. 두 항을 모두 보고하며 rounding을 추가 semantic budget으로 숨기지 않는다. Native allowed-range 누설도 같은 방식으로 ideal/rounding/actual을 구분한다.

EN_EXACT의 response/logit parity는 technical diagnostic이다. EN_NUM/ADAPT에 `DK=0`, native logit 동일성, `NLL≤native+1e-4`를 적용하지 않는다. RS/PS, reference 개별 token 성공이나 margin은 candidate 수용에 사용하지 않는다. 따라서 R/PS 저하 후보도 과학적으로 유효한 결과로 보고할 수 있으며 실험 목적에 맞지 않으면 이후 방법을 개선한다.

## 8. 순차 실험과 history

기본 범위는 **B1 후 바로 B3까지**, N4/EN_EXACT/EN_ADAPT 3chain이다. B1의 기술적으로 유효한 결과가 있으면 locality 개선 여부로 arm을 골라서 탈락시키지 않고 사전 지정한 세 chain을 진행한다. B1에서 정책을 바꾸면 새 version으로 W0부터 시작한다.

- B2/B3에서 각 chain은 own entry에서 native z/write를 한 번 계산한다. 다른 chain의 z/gradient를 공유하지 않는다.
- Current/reference fixed-key cache는 identity가 맞으면 chain 간 공유한다. Downstream state는 chain별이다.
- Stage B3까지 과거 요청은 최대200개이므로 전부 사용한다. History buffer cap512보다 작아서 **GSS에 의한 제거가 발생할 이유가 없다**.
- History는 latest-valid subject/relation target만 활성화하고 current overwrite에 해당하는 이전 target은 batch 전에 제거한다. Source request/target/arrival/overwrite ledger를 모두 보존한다.

History를 일반 reference와 같은 KL teacher 방식으로 사용한다. 요청이 도착한 batch의 **arm별 최종 선택 W**에서 canonical supplied new-target sequence를 teacher forcing하고 full-vocab teacher distribution을 한 번 저장한다. 생성한 paraphrase는 추가하지 않는다. 같은 요청의 다음 batch들에서는 이 at-write teacher를 고정한다. Teacher forcing 입력은 full target prefix이며 보호 label은 supplied desired target이다.

\[
J_t(W)=L_R(W)+\mathbf1_{|H_t|>0} L_H(W),
\]

\[
L_H(W)=\frac1{|H_t|}\sum_{i\in H_t}\frac1{|y_i^*|}
\sum_s KL(p_{W_{i,write}}(\cdot|x_i,y^*_{i,<s})\Vert p_W(\cdot|x_i,y^*_{i,<s})).
\tag{10}
\]

Reference/history 각각을 평균한 뒤 같은 block weight로 합한다. 이 선택 역시 사전 고정 convention이며 최적 가중치라고 주장하지 않는다. N4도 history teacher를 같은 정책으로 기록하되 보정 목적에는 사용하지 않는다.

At-write teacher는 최종 rewrite observer 또는 이미 수행한 current forward와 입력·prefix·head precision이 완전히 같으면 그 결과에서 추출한다. 다른 입력의 logit을 재사용하지 않는다. B3에서 끝나는 이번 실행의 마지막 batch teacher는 후속 소비가 없으므로 새 생성 forward를 요구하지 않는다. 별도 teacher forward가 필요했던 경우와 실제 teacher 저장 bytes는 반드시 비용에 포함한다.

Current Q_E는 current batch만으로 정의한다. History는 L_H를 통해 보정 방향과 실제 KL selection에 들어간다. 따라서 current response 예산을 history의 hard bound라고 주장하지 않는다. 각 후보의 L_R/L_H를 별도로 저장하여 합계 개선이 어느 block의 악화를 가렸는지 확인한다. At-write teacher는 그때의 모델 행동을 보호하는 기준이며, 애초 실패한 edit을 성공시켜 주지 않는다. 그런 요청도 평가 분모에서 제외하지 않는다.

GSS는512개를 넘는 history를 사용하는 다음 확장의 요소다. 원 논문은 gradient diversity를 이용한 memory selection을 제안한다. [GSS](https://arxiv.org/abs/1903.08671). 이번 B300에서 GSS/recency sampling arm을 추가하면 threshold 가설과 sampling 효과가 섞인다. B300에서는 history 전체를 사용하여 이를 피한다. 향후 선택 대상은 history이며 reference512개는 계속 전부 사용한다. 오래된 batch의 서로 다른 coefficient basis에서 얻은 gradient vector를 직접 비교하는 설계는 사용하지 않는다.

B1000 확대는 이번 필수 실행 범위 밖이다. B300 결과 후 history512 초과 시점의 gradient refresh 비용과 GSS/recency 규약을 별도 version으로 봉인한 뒤 확대한다. 이번 문서를 그대로 B1000/10k 제출 권한이나 검증 완료로 간주하지 않는다.

## 9. 실행 순서와 계산량 상한

### T0: 공통 입력과 새 계산만 검증

1. Native·teacher·P·stream identity 및 reuse eligibility.
2. Request/sequence/token weight의 합, alias coverage, weighted exact-space 동일성.
3. Numerical-tail witness, FP64 projector symmetry/idempotence/allowed-range.
4. 4개 고정 reference와4개 current 입력에서 cached actual H0+DK와 physical weight forward 대조. 선택은 identity hash 순서이며 실패 사례를 골라 tolerance를 바꾸지 않는다.
5. 같은 panel에서 native KL gradient 직접 AD 및 representative direction의 directional derivative 확인. 기존 FD waiver를 새로운 gradient 경로의 PASS로 쓰지 않는다. Precision 미확립이면 별도 exploratory 상태로 남긴다.
6. CPU selector의 scalar energy와 실제 projection/action, FP32 materialization/rounding의 일치.

### B1: 공유 계산

|항목|상한/정책|
|---|---|
|Native target-fitting batch pass|1; request당1회, 총100 local-z 최적화|
|새 z pass / correction|0|
|R512 gradient sweep|**1회, 세 EN arm 공유**|
|Weighted current SVD|1회|
|Threshold/epsilon 비교|저장 mode energy의 algebra-only sweep|
|Candidate R512 forward sweep|세 EN arm ×최대2 = **최대6회**|
|Official R/P/N observer|최종4endpoint; 동일 endpoint는1회만 평가|
|Dev128 observer|N4와 EN_ADAPT 최종 endpoint만|

S3까지 모두 진행했을 때, 새 계산 상한은 native batch pass7회(B1공유1+B2/B3각3), R512 gradient sweep5회(B1공유1+B2/B3각2), candidate R512 sweep14회(B1최대6+B2/B3각4)다. Native target 계산은 최대700 request-optimization이며700개의 단일 forward라는 뜻이 아니다. History teacher/gradient/forward, T0 및 official observer 비용은 이 숫자에 포함되지 않으므로 별도 카운터를 둔다.

같은 projector·같은 D·같은 materialized endpoint가 여러 arm에서 나오면 objective/observer를 캐시하고 alias로 기록한다. 다른 endpoint를 '비슷하다'는 이유로 공유하지 않는다. 중심 G, full-vocab teacher, prefix keys는 재사용하되 candidate suffix forward는 필요한 만큼 실행한다. 단순히8step을2step으로 이름만 바꾸거나 reference를64개로 축소하지 않는다.

**후보마다 current 전체의 downstream NLL/logit forward를 실행하지 않는다.** 기존 exact-quality guard를 대신하는 current-response 검사와 rounding 검사는 cached Kbar의 행렬 작용으로 처리한다. Current R/PS/N의 functional forward는 최종 선택 후 observer에서 수행한다. 이 변경은 후보 수 감소와 별개의 계산 절감이며, 그 대가로 기능적 성공을 candidate 수용 단계에서 보장하지 않는다는 점을 명시한다.

No candidate별 model reload/full tensor checksum, no per-document dense gradient D2H, no candidate별 z/current SVD. Hash는 불변 artifact 입구와 최종 endpoint당 한 번 계산하고 내부 재사용은 소유권/version guard로 확인한다. Chunking으로 full-vocab KL을 정확히 누적하며 top-k KL로 바꾸지 않는다.

연구 전체의 공유 비용과 한 method를 단독 실행하는 비용을 분리한다. Standalone EN에는 native+current geometry+reference gradient 전체 비용을 각각 배정하며 세 arm으로 나누지 않는다. History, 후보 forward, teacher 준비, observer, hashing을 분리하여 기록한다. 이전 실행의 EN correction/native 비율4.56은 참고치이며 새 runtime과 matched 측정 없이 속도 향상을 선언하지 않는다.

## 10. 평가와 결과 해석

모든 후보 선택과 arm policy가 봉인된 다음 official observer를 실행한다. 성능이 좋았던 arm만 보고하지 않는다.

**B1 전체:** R100/P200/N1000. Native 대비 gained/lost case ID를 함께 보고한다. R/PS 평균이 같다는 것과 성공 집합이 같다는 것을 구별한다. 개발 stream에서 `delta R>=0, delta P>=0, delta N>0`이면 aggregate Pareto 개선 신호로 부른다. 개별 R/P 손실0은 더 강한 별도 사실이다. B1 하나로 통계적 일반화를 선언하지 않는다.

**B2/B3:** current batch와 모든 active past의 R/P/N을 따로 평가한다. History 보호에는 canonical만 쓰지만 past PS도 observer로 측정한다. At-write 성공→later failure, W0-correct N 손실, retained/recovered/newly-lost N, current/past 분모를 보고한다. Overwrite로 무효화된 과거 target은 active 평가와 official unfiltered 결과를 구별하고 ledger에 이유를 남긴다.

**기전 표:** rank, tau/sigma/units, numerical tail 수, 추가 gradient energy, predicted KL decrease, norm cap/response cap 활성 여부, actual D/native norm, actual current response/native response, 요청별 response p95/max, mode0/native 방향 표현 가능성, actual KL 감소, R/P/N gained/lost.

**Reference:** train512/Dev128 KL·base token-choice flip·log-prob weakening·길이별 결과. Margin은 필요하면 관측치로만 저장하며 method에 사용하지 않는다. 이미 같은 candidate forward에서 얻는 관측치를 위해 추가 forward를 하지 않는다.

**통계:** case별 paired 변화 및 denominator를 먼저 보고하고, bootstrap은 request를 cluster로10,000회 resample한다. 같은 request의10개 neighbor를 독립1000표본으로 취급하지 않는다. Seed20260920. 개발 셋 사용과 작은 B300 범위를 명시한다.

|관측|가능한 해석|
|---|---|
|NUM≈EXACT, ADAPT가 non-numerical 방향을 열고 NS 개선·R/PS 유지|수치적 tail 정리 이상의 response 완화가 유효하다는 신호|
|NUM만으로 개선|동일-prefix numerical 과보호를 우선 수정할 근거|
|ADAPT가 exact/NUM과 같은 mask 또는 거의 같은 D|실질적으로 다른 intervention이 생성되지 않음|
|추가 descent·KL 개선은 있지만 NS 개선 없음|reference 목적/coverage 문제가 여전히 남음|
|NS 개선과 R/PS 손실 동반|이번 response 예산은 기능적 편집 보호에 불충분|
|큰 proposal은 KL 악화, 축소만 수용|finite curvature/탐색 범위 병목 가능성; null-space만의 실패로 단정 불가|
|B1 개선, past R/PS 또는 N 망각 악화|current-only geometry와 history objective의 lifelong 한계|
|P 전체가 선택되어 개선|해당 예산에서는 spectral 차단 필요성이 미확립|

Epsilon1%/10% sensitivity는 우선 algebra 결과다. 모델 품질 차이인 것처럼 표기하지 않는다. 이후 별도 모델 sensitivity를 수행하면 두 값을 모두 보고하고 primary5%를 사후 best로 바꾸지 않는다.

## 11. 산출물 및 구현 경계

새 구현 위치는 `project/run_scripts/en_adaptive_nullspace/`를 권장한다. 기존 EN/DEC module을 in-place 수정하여 과거 동작을 바꾸지 않는다. Frozen source의 reusable oracle/geometry를 adapter로 사용하되 actual integration path를 receipt에 기록한다.

- `run-manifest.json`: 모델/stream/teacher/native/P/source/precision, reuse status, current/history/reference 역할.
- `current-protection-manifest.json`, `weighted-geometry.json`: weights, SVD, rank/cutoff/duplicate witness.
- `gradient-spectrum.json`: e0/e_j/lambda, norm/action 검증, raw-gradient identity와 공유 횟수.
- `threshold-frontier.csv`: exact/NUM/모든 spectral boundary 및 epsilon0.01/.05/.10의 p/norm/response/cap, selector 선택.
- `candidate-ledger.jsonl`: 모든 trial, curvature, rounding, KL_R/KL_H, actual inner product, 예산, acceptance/fallback.
- `history-ledger.jsonl`, `history-teacher-manifest.json`: 유효 target/overwrite/at-write teacher identity.
- `official-paired.csv`, `reference-paired.csv`, `sequential-retention.csv`.
- `timing.json`: native, key/cache, SVD, R-gradient, H-gradient, projection, candidate suffix/head, teacher I/O, official observer, hashing.
- `report-ko.md`: 결과·미측정·실패·fallback·비용을 함께 기록.

이번 설계에 딸린 [CPU reference](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-en-adaptive-nullspace-design-v1/selector_reference.py)는 작은 energy-array selector와2-evaluation scalar interpolation 규약을 검증하기 위한 것이다. GPU runner나 neural performance 증거가 아니다. Runtime import만으로 모델을 로드하거나 실험을 제출하지 않는다.
