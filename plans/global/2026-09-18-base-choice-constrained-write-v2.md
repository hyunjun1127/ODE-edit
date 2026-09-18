# Base-choice constrained L4 write — v2 방법·GSS·빠른 검증

작성: 2026-09-18. 작업용 약칭 **BPCW**. 상태: 설계 및 CPU 수학 검증용; 모델 runner·W0 answer capsule·GPU 실험은 아직 없다. 이 문서는 기존 factual-label/entry-bound FRCW v1을 대체한다.

실행 전달문: [GH 지시문 — 구현·cold B100·조건부 SEQ1000](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-18-base-choice-constrained-write-gh-instruction-v2.md). 지시문 작성은 실제 dispatch나 GPU 제출을 뜻하지 않는다.

산출물: [실행 규약 JSON](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-base-choice-constrained-write-contract-v2.json), [빠른 실행 cells](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-base-choice-constrained-write-cells-v2.csv), [CPU 수학 검증12개 PASS](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-base-choice-constrained-write/math-checks.json). CPU 검증은 dual 부호, 전체 제약 확인, competitor 전환 시 이전 경계 유지, EOS/censor 및 동률 반례 등이며 실제 LLM의 수치·성능 검증은 아니다.

핵심은 **W0가 선택한 답변을 유지하는 전체 reference 제약 아래, native L4 편집 response를 바꾸지 않는 최소 weight 변경을 구하는 것**이다. Primary에는 확률 복원·원래 margin 복원·KL 최소화가 없다. Reference512 전부를 매 batch 검사한다. GSS는 bank 축소가 아니라 전체 local QP의 working-set 처리 순서에 사용한다.

## 1. 무엇을 보호하는가

Reference 입력 x_i에 대해 W0의 deterministic greedy sequence를 y_i^0=(y_i1,…,y_iTi)라 한다. 모델이 EOS를 출력하면 EOS도 포함한다. 최대 생성 길이에 도달하면 censor flag를 붙이며 저장된 prefix만 보호한다.

각 위치의 비교 문맥은 항상 c_is=(x_i,y_i,<s^0)다. Native/candidate가 새로 생성한 prefix로 대체하지 않는다. Base의 답변은 행동 기준이며 외부 검증 없이 factual ground truth로 부르지 않는다.

저장할 capsule은 prompt/token/mask/position, y0, EOS 또는 censor, base token log-probability, base top8 token IDs/logits, base margin, model/tokenizer/decoding hashes다. Top8은 기록용이며 candidate competitor를 top8로 제한하지 않는다. Prompt당 답변 하나이며 별도 paraphrase target set은 만들지 않는다.

원칙은 다음 세 가지를 구별하는 것이다.

|양|정의|v2 역할|
|---|---|---|
|선택 margin|m_is(W)=log p_W(y0_is\|c_is)−max_{v≠y0_is}log p_W(v\|c_is)|주 제약: m≥0 및 동일 greedy 선택|
|확률 약화|d_is(W)=log p_W0(y0_is\|c_is)−log p_W(y0_is\|c_is)|진단; 주 제약 아님|
|원래 우위 감소|m_is(W0)−m_is(W)|진단; 원래 margin 복원 요구 없음|

d≤δ를 추가하면 p_W(y0)≥exp(−δ)p_W0(y0)다. δ=0이나 다른 ratio를 주 방법에 조용히 넣지 않는다. 향후 probability floor는 명시적인 별도 ablation이다. 같은 prefix의 base-margin 보존 m≥m0는 choice 보존 m≥0보다 강하다. 확률 보존과 선택 보존은 서로 대체하지 않는다.

## 2. 빠른 pilot의 reference 입력

수식은 질문·cloze·문서 prefix 모두에 적용된다. **빠른 1차 pilot의 명시적 기본값은 기존 C4 자산을 활용한 behavior-preservation pilot**이다. Wikipedia factual question bank가 이미 확보되어 있다면 별도 dataset ID로 교체할 수 있지만, 문서 prefix 실험을 factual QA 보존 실험이라고 부르지 않는다.

- Train512: 기존 학습 가능 S64+Reserve320의384문서와 같은 pinned C4 train shard 규칙의 추가128문서. 기존 Dev128·Report256은 편입하지 않는다. Document IDs·새128 선정·중복 제거 manifest를 먼저 봉인한다. 현 상태에서 새 R512가 구축되었다고 가정하지 않는다.
- 입력 x_i: 각 봉인된256-token window의 처음128 tokens에 기존 BOS/tokenization 규약 적용. 별도 instruction wrapper를 붙이지 않는 raw continuation 조건이다.
- W0 출력: do_sample=false, num_beams=1, raw logits argmax, max_new_tokens=16. Repetition/min-length/forced-token processor는 사용하지 않는다. 모델의 EOS IDs와 동일 token-ID tie 규칙을 봉인한다. 이 설정은 질문 응답과 다르며 EOS가 없으면 16-token continuation-prefix 보존이다.
- Base token 수 T_i≤16이므로 최대8192개 위치를 보호한다. 512문서라는 이유로 512개 token 제약만 있는 것이 아니다.
- 고정 hash 앞256개는 후속 크기 ablation용으로만 보관한다. 주 실행은512개 전부다.
- 독립 Dev128에서도 동일한 W0 capsule을 준비하되 optimizer/GSS에는 전달하지 않는다. General C4 KL observer와 behavioral choice observer를 구별한다.

입력 distribution이 NS와 연결되지 않으면 margin 제약이나 GSS만 바꿔도 해결되지 않는다. C4 pilot에서 full-bank 보존은 성공하지만 N 개선이 없으면 reference relevance를 별도 원인으로 보고한다. 단순히 gradient budget을 늘리거나 “GSS가 부족했다”고 결론내리지 않는다.

## 3. Base greedy 경로와 teacher forcing

하나의 입력 `[x_i,y_i^0[:-1]]`을 causal forward하여 모든 생성 위치의 logits를 얻는다. 각 위치의 token-ID와 score shift를 명시적으로 검증한다. 전체 reference의 원래 입력 prefix와 generated prefix token이 동일해야 한다.

모든 보호 위치에서 실제 decoding argmax가 y0_is이고, EOS 조건도 같다면 deterministic greedy sequence는 induction으로 유지된다. 이 논리는 같은 함수·decoding 규칙에 대한 것이다. KV-cache 생성과 full-prefix teacher forcing의 numerical 차이는 별도로 점검한다. Pilot에서 기록하는 전체-bank 주 지표는 **teacher-forced greedy choice 일치**이며, direct generation은 고정8개 reference에서 B1/B10에 추가 확인한다. 16-token censor에 대해 무한 continuation 보존을 주장하지 않는다.

Base와 candidate의 top competitor는 달라도 된다. Candidate에서는 매번 full vocabulary에서

`v*_is(W)=argmax_{v≠y0_is} z_W(v|c_is)`

를 다시 찾는다. Base top8에 없던 token도 경쟁자가 된다.

## 4. Single-layer 구조와 exact current-edit lock

변경 parameter는 zero-based L4 `down_proj.weight` W∈R^(4096×14336) 하나다. 다른 weight가 고정이므로 고정 입력의 key K_i와 residual B_i는 변하지 않는다.

`Y_i(W)=B_i+W K_i`, `Y_i(W+D)−Y_i(W)=D K_i`.

매 batch t에서 기존 native L4 local-z/write를 한 번 수행해 `WN_t=Wentry_t+ΔN_t`를 얻는다. 후보마다 z를 재최적화하지 않는다. Native response를 보존할 현재 canonical/native rewrite contexts의 new/old teacher-forcing 경로 전체 token key를 K_E,t로 모은다.

`D K_E,t=0`, `D=D P*`를 동시에 요구한다. P*=VVᵀ, J=VᵀK_E이면

`Q_t=V(I−J J†)Vᵀ`, `D=D Q_t`.

그 결과 보호된 current sequence에서 native가 실제로 실현한 response를 유지한다. z와 WN K_E가 같다고 가정하지 않는다. PS나 보지 않은 생성 경로까지 exact하게 보장하지 않는다. Native allowed-space와 edit-null projector를 임의 순서로 곱하지 않는다.

## 5. 전체 reference에 대한 이상적 방법

동일 prefix에서는 log-softmax의 정규화항이 상쇄된다.

`m_is(W)=z_W(y0_is|c_is)−max_{v≠y0_is}z_W(v|c_is)`.

따라서 주 문제는

`min_{D=DQ_t} 0.5 ||D||_F²`

`s.t. z_(WN+D)(y0_is|c_is)−z_(WN+D)(v|c_is)≥0`

`for every reference i=1,…,512, every protected s, every v≠y0_is`.

이는 full-distribution matching이 아니라 선택 경계의 보존이다. 모든 reference가 안전하면 D=0이 최적이므로 backward/QP를 생략한다. 이미 안전한 reference도 candidate가 손상시키려 할 때는 binding 제약이 될 수 있다. “안전했던 reference는 항상 alpha=0”이라고 가정하지 않는다.

문제의 목적은 native로부터의 최소 변경이며 전체 nonlinear 문제의 전역 최소해를 실제 알고리즘이 보장하지는 않는다. 정확한 layer-response 선형성과 downstream margin의 비선형성을 구별한다.

### 5.1 동률과 수치 여유

과학적 목표는 m≥0와 원래 token 선택이다. 실제 accept는 전체 위치의 argmax token ID가 y0와 같아야 한다. m=0이면 봉인된 tie 규칙으로 판정하고 다른 token을 고른 경우 작은 음수 tolerance로 통과시키지 않는다.

QP의 boundary rounding을 줄이기 위해 수치 reserve만 사용할 수 있다. `tau_gap=max(1e−4,10×반복 gap 변동의 최대값)`를 성능을 보기 전 정하고 `kappa_is=min(m_is(W0),tau_gap)`로 둔다. Base가 이 reserve 때문에 infeasible해지지 않는다. tau_gap>1e−3이면 runtime numerical issue로 보고 원인을 해결한다. Tie에서 kappa=0이다. Kappa는 학습된 preservation budget이나 원래 margin 복원 비율이 아니다. Actual accept의 핵심은 전체 token-ID 선택 보존이며 kappa를 별도 성능 이득으로 주장하지 않는다.

## 6. Reference별 최악 margin으로 local QP 구성

각 reference의 모든 위치와 vocabulary를 forward로 검사해

`M_i(W)=min_s [m_is(W)−kappa_is]`

를 정의한다. `M_i≥0`이면 그 reference의 모든 token reserve를 만족한다. 현재 선형화점 W_k=WN+D_k에서 최악 위치 s_i*와 competitor v_i*를 선택한다. Tie는 position/token ID의 고정 순서다.

`mu_i,k=z_y(W_k)−z_v*(W_k)−kappa_i,s*`

`g_i,k=∇_W(z_y−z_v*)(W_k)`, `h_i,k=g_i,k Q_t`.

처음에는512개 reference 모두 하나의 local row를 갖는다. 이것은 모든 token-vocabulary pair의 선형화가 동시에 들어갔다는 뜻이 아니다. **Batch 안에서 발견한 pair `(reference,position,competitor)`는 유지한다.** 다음 round에는 전체 검사에서 찾은 새 worst pair를 추가하고, 이전 pair와 새 pair의 gradient를 모두 같은 현재 endpoint에서 다시 계산한다. 같은 pair는 deduplicate한다. 이전 competitor를 새 competitor로 단순 교체하면 알려진 경계를 다시 깨뜨리며 oscillation할 수 있다.

따라서 round1 row 수는512, round2는 최대1024다. 행 집합 J_t,k는 모든 reference를 포함하는 누적 exposed-pair 집합이며, 다음 식의 i는 이 pair row를 뜻한다. 각 pair의 mu는 해당 위치/competitor의 logit gap minus kappa다. Full vocabulary와 모든 위치의 실제 검사는 별도로 계속한다.

최종 D를 변수로 하는 local 제약은

`<h_i,k,D>_F ≥ b_i,k`,

`b_i,k=−mu_i,k+<g_i,k,D_k,actual>_F`.

Actual center `D_k,actual=FP64(W_k)−FP64(WN)`의 공간 밖 rounding 성분도 RHS에 반영한다. 첫 round는 D0=0이므로 b=−mu(WN)다. D를 increment로 잘못 해석하여 두 번째 RHS 항을 누락하지 않는다.

### 6.1 해와 강도

Local primal은 `min_D 0.5||D||² subject to <h_i,D>≥b_i`다. G_ij=<h_i,h_j>이면 dual과 reconstruction은

`min_{alpha≥0} 0.5 alphaᵀG alpha−bᵀalpha`,

`D*=Σ_i alpha_i h_i`.

KKT는 `G alpha−b≥0`, `alpha_i(G alpha−b)_i=0`이다. Margin을 올리는 방향이므로 v1 loss 감소식과 reconstruction 부호가 반대다.

한 제약만 있을 때 `D*=[b]_+ h/||h||²`다. Native margin−0.134, kappa=0이면 b=0.134다. 임의의 layer scale 없이 경계까지 필요한 양과 허용 민감도로 이동량이 정해진다.

h=0인데 b>0이면 현재 first-order 모델에서 해당 reference를 보호할 수 없다. Nonlinear 원문제 전체의 불가능성을 증명한 것은 아니다. QP infeasible, solver numerical failure, nonlinear finite-check 실패를 각각 기록한다. Gram ridge로 제약을 바꾸고 같은 해라고 부르지 않는다.

## 7. GSS를 넣는 위치: bank 축소 없이 working-set 순서 결정

[GSS 원문](https://arxiv.org/html/1903.08671)은 gradient 방향 다양성으로 보존 제약의 대표성을 다룬다. 여기서는 **512개 reference 중 일부를 버리는 buffer 정책으로 적용하지 않는다.** 모든 reference의 local gradient/offset을 확보한 뒤, QP에서 먼저 처리할 row를 정하는 데 GSS의 방향 정보를 쓴다.

Feature는 raw gradient가 아니라 `h_i=g_i Q_t`다. Offset까지 비교하기 위해 nonzero row는 `a_i=h_i/||h_i||`, `beta_i=b_i/||h_i||`로 normalize한다. 두 항을 함께 나누므로 feasible set은 같다. 같은 방향이어도 beta가 더 큰 제약이 더 강하다.

Working-set 알고리즘:

1. D=0에서 가장 큰 positive normalized violation beta를 가진 row를 seed로 둔다. 나머지는 cosine distance가 기존 집합과 가장 다른 row 순으로 초기32개까지 넣는다. Direction tie는 violation, pair ID 순이다. 32는 초기 solver block 크기이며 최종 reference 수 제한이 아니다.
2. Working-set QP를 푼다. **모든512 reference에서 누적된 전체 exposed-pair local inequality** `<h_i,D>≥b_i`를 검사한다.
3. 누락 row 중 위반된 것들을 찾는다. 가장 큰 위반은 반드시 포함하고, 나머지는 projected-gradient diversity 순으로 최대32개씩 추가한다. 이미 확보한 row를 cosine이 비슷하다는 이유로 영구 폐기하지 않는다. Working set은 필요한 만큼 증가하며 전체 exposed rows까지 들어갈 수 있다(round2 최대1024).
4. 전체 local row가 만족될 때만 local QP 해를 반환한다. Working-set solve의 optimality와 전체 feasibility가 확인되면 동일 local full-QP의 해다: subset 문제의 optimum은 full 문제의 lower bound이고, 그 해가 full feasible이면 두 objective가 같다.
5. 이후 실제 모델에서 모든 reference/위치/vocabulary를 검사한다. Local QP 통과는 전체 nonlinear 조건 통과와 다르다.

이 방법은 GSS-inspired working-set ordering이지 원 GSS replay algorithm의 재현이 아니다. 동일 gradient와 local QP에서 처리 순서만 다르므로 GSS 자체가 더 좋은 endpoint를 만든다고 기대하지 않는다. 차이는 solver 시간/working-set 크기다. Pure most-violation ordering과의 비교는 저장된 gradient/Gram으로 CPU에서 수행할 수 있어 새 GPU arm이 필요 없다.

**첫 버전에서는 모든 reference의 exposed-pair gradient를 구하므로 GSS가 neural backward 수를 줄인다고 주장하지 않는다.** 일부 gradient만 먼저 구하는 lazy oracle은 별도 후속 최적화다. 이를 섞어 “모두 사용”이라는 계약을 약화시키지 않는다. Exposed rows의 local full-QP 해와 원래 모든 token/vocabulary를 포함한 nonlinear 전역 최소해도 구별한다.

## 8. Fixed key와 token margin으로 줄일 수 있는 비용

Margin backward는 두 logits의 차이다. Competitor를 full-vocab forward로 찾은 뒤 그 index를 고정하고, backward는 해당 위치의 두 LM-head row만 사용해도 된다. 전체 vocabulary의 KL gradient나 dense teacher distribution은 필요 없다. 다만 competitor를 찾는 full-vocab forward 자체는 남는다.

Reference별 선택한 worst margin의 module-output derivative를 A_i라 하면 `g_i=A_i K_iᵀ`, `h_i=A_i(Q_tK_i)ᵀ`다. 모든 prefix token에 대한 downstream gradient를 포함하며 subject 또는 답변 위치의 key만 사용하지 않는다.

서로 독립인 example batch에서 pair-margin들의 합을 module-output activation에 대해 역전파하면 example별 A_i가 batch 축에 남는다. Full parameter의 summed gradient만 얻으면 sample별 정보가 사라진다. 한 reference의 여러 pair gradient를 summed loss에서 각각 분리할 수 있다고 주장하지 않는다. Round2에 같은 reference의 두 pair가 남으면 별도 example 복제 또는 명시적인 batched VJP로 분리한다. Shared forward의 추가 이득은 실제 구현에서 검증한다.

Factor Gram은 `G_ij=tr((A_iᵀA_j)(U_jᵀU_i))`, U_i=QK_i로 계산하거나 Gram-vector product로 푼다. Dense per-reference 4096×14336 gradient를512개 저장하지 않는다. Projector/basis, factor I/O, Gram 계산 비용도 계측한다.

R512의 K/B와 base capsule은 입력이 고정인 동안 재사용한다. Prefix와 teacher label은 W0 기준으로 고정하지만, Q_t·A_i·worst competitor는 batch/round마다 달라진다. 이번 방식은 v1의 별도 challenger teacher-forcing 경로와 매 batch 새 challenger에 대한 entry scoring이 필요 없다. 모든 경쟁 token을 같은 base prefix에서 비교한다.

빠른 pilot 기본 예산은 **최대2회 nonlinear linearization**이다. Round1의512개 pair gradient, round2의 누적 최대1024개 pair gradient, round당 QP 한 개와 실제 full-bank candidate 검사 한 번이다. 총 최대1536 pair-scalar gradient 평가이며 microbatch backward 호출 수와는 다르다. 이미 본 pair를 재사용하더라도 gradient 값은 현재 center에서 갱신한다. Native가 이미 전체 선택을 보존하면 gradient0회다. Base generation/cache는1회 준비 비용으로 별도 공개한다. EN보다 실제로 빠르다는 판정은 계측 뒤에 한다.

## 9. Accept, 재선형화, fallback

1. 전체 native reference scan에서 실제 token ID가 모두 유지되면 D=0으로 NATIVE_ALREADY_FEASIBLE을 반환한다. 작은 numerical reserve만 부족하다는 이유로 이미 안전한 native를 강화하지 않는다. 확률 d가 커져도 주 방법은 추가 correction을 강제하지 않는다.
2. 그렇지 않으면 전체512 reference의 initial pair gradient를 구해 GSS-ordered local full-QP를 푼다. Immutable WN에서 actual FP32 candidate를 만든다.
3. Cheap key-response/allowed-space 검사 뒤 모든 reference token의 full-vocabulary argmax/margin을 다시 계산한다. 하나라도 base token 선택을 잃으면 한 번만 전체 재선형화한다. 이전 pair를 보존하고 바뀐 worst position/competitor를 추가하며, 누적 pair 전부를 새 center에서 갱신한다.
4. 전체 reference 선택을 통과한 첫 후보에만 current-edit EN invariant/quality guard와 Past64 guard를 수행한다. 이 guard가 실패하면 추가 candidate 탐색 없이 native fallback이다.
5. 두 번의 nonlinear 시도 안에 통과하지 못하면 NATIVE_FALLBACK_REFERENCE로 끝낸다. 이미 reference 선택이 깨진 native fallback을 reference-preserved로 집계하지 않는다. Budget 종료는 원문제 infeasible 또는 수렴 판정이 아니다. Exact tie의 m=0 조건만으로 원하는 token-ID tie 결과를 만들 수 없는 경우 TIE_UNRESOLVED로 구별하고, reference를 제거하거나 tie 규칙을 바꿔 통과시키지 않는다.

Current guard는 기존 EN의 actual DKE, logits/NLL/strict 기준을 유지한다. Canonical preference/strict 성공 ID 추가 손실0, own-native 대비 new NLL+1e−4, protected logits max1e−3/RMS1e−4, nominal DKE normalized1e−10, actual max-token response1e−5 등이다. T-skipped였던 과거 EN 상태를 새 검증 PASS로 재사용하지 않는다.

Past64는 이미 관측한 최신 active 요청만 사용하고, overwrite 이전 target은 제외한다. 이번 v2의 주 최적화는 external R512의 base-choice 제약이다. Past64는 native 대비 correction의 추가 NLL/preference/strict 손상을 막는 별도 guard이며, native가 만든 forgetting까지 없앤다는 조건이 아니다. Native history M은 commit당1회 갱신한다. 과거 edit를 W0의 옛 답변으로 되돌리는 제약은 만들지 않는다.

Reference가 실제로 변경하려는 같은 사실/질문을 요구하면 fixed W0 choice와 current edit가 충돌할 수 있다. 이런 scope conflict는 별도로 식별·기록한다. 계산량 또는 불리한 결과 때문에 reference를 빼지 않는다. Pilot은 current 요청과의 직접 충돌이 없는512 입력을 사전에 봉인한다. Sequential에서 의도적 overwrite 충돌이 발견되면 동일512 완전보존 주장과 구별하고 명시적 version/scope 처리가 필요하다. 이를 gradient 실패와 혼동하지 않는다.

## 10. W0 → B1 → B2–B10의 빠른 실험

### 준비

R512 입력과 W0 capsule을 준비한다. 외부 factual 정답·오답 label 구축은 더 이상 필수 조건이 아니다. Train와 분리된 Dev128 capsule을 준비한다. Source/model/tokenizer/decoding/teacher-forcing shift manifest가 준비되기 전에는 실행-ready라고 하지 않는다.

B1 안에서 reference4개/current prefix를 써 key stationarity, cached/physical forward, two-logit margin gradient vs direct AD, 선택 competitor가 안정적인 방향의 AD–FD, actual FP32 null, state restore를 확인한다. Non-differentiable competitor tie에서 centered FD를 일반 smooth-gradient 검사처럼 해석하지 않는다. Working-set/full-QP 동등성은 CPU check로 함께 검증한다.

### P1: 동일 cold B100, 두 arm

|arm|설정|
|---|---|
|N4|W0에서 native L4 local-z/write|
|BPCW512|같은 W0와 native preview, 전체512 base-choice constraints|

기존 O0 first100은 개발 batch다. Native target/write는 공유할 수 있지만 standalone method 비용에는 각각 포함한다. Server1 동일 GPU/runtime에서 비교한다. 기존 mixed-hardware EN endpoint는 역사적 맥락이며 matched 신규 대조를 대신하지 않는다.

Endpoint를 봉인한 뒤에만 공식 R100/P200/N1000을 평가한다. Reference는 token flip 수/분모, 완전히 유지된 sequence 수/512, changed-EOS/censor, m/d 분포, max violation, required correction norm을 보고한다. Base top8에 없던 competitor가 얼마나 등장했는지도 보고한다. Dev128의 동일 지표와 기존 C4 KL은 observer다.

**S10 진입:** 기술검사 PASS; 전체 reference 검사와 current/past guard PASS; N4 대비 RS/PS 성공 ID 추가 손실0; NS net 감소0 및 W0-correct N 손실 증가0. Dev128 choice flips가 native보다 늘지 않을 것. Nonzero correction은 요구하지 않는다: NATIVE_ALREADY_FEASIBLE이면 sparse boundary activation이 뒤 batch에서 생기는지 S10으로 확인할 수 있다. Reference 위반 native fallback이면 바로 확대하지 않는다.

실행 가능성 기준은 setup 제외 standalone editing≤2×matched N4로 둔다. 이는 계산 예산이며 통계적 효과 기준이 아니다. 기술/효과 기준 통과 후 비용만 초과하면 “실용 예산 미충족”으로 분리한다. Pilot의 작은 표본에서 유의미한 성능 우위를 입증했다고 하지 않는다.

### S10: 정책 변경 없이 각자 B1에서 B100×10까지 연장

N4는 자신의 B1, BPCW는 자신의 B1 checkpoint에서 B2–B10을 이어간다. 둘 다 W0에서 출발한 chain이다. 다른 실험의5000-edit checkpoint를 쓰지 않는다. B2부터 target/native preview/history는 각자 own entry에서 계산한다. Fixed upstream key cache만 동일 입력에서 공유할 수 있다.

- 매 batch: current R/P/N at-write, current N entry→post arrival damage. Observer는 controller와 분리한다.
- 고정 B1 cohort: 이후 batch마다 R/P/N 및 지난 endpoint 대비 망각. 이미 같은 state에서 측정한 값은 재사용한다.
- B10: seen1000 전체 R1000/P2000/N10000 final 및 at-write 대비 retention.
- Reference512: 매 native/candidate/selected endpoint의 전체 choice 검사; Dev128은 B1/B5/B10만 별도 관측.
- Mean KL이 아니라 final NS, arrival N 손실, old N forgetting, RS/PS, reference choice/d drift, accepted/fallback/skip 비율과 비용을 함께 판단한다.

공식 P/N은 같은 batch의 solver·step·GSS·stopping에 사용하지 않는다. P1→S10 gate는 endpoint 봉인 후 개발 판단임을 기록한다. 한 sequence/order에서 나온1k 결과로10k/generalization을 확정하지 않는다. R256, probability floor, factual-query 입력 분포, same-reference KL와 원형 AlphaEdit/BLUE 비교는 S10 이후다.

## 11. 핵심 산출물과 해석

필수 기록: 전체512 input/capsule hashes, predicted token IDs, all protected-position m/d, full-vocab competitor, worst position, projector rank, local gradients/factors의 state ID, normalized offset, GSS working-set 추가 순서, 전체 local primal/dual/KKT, actual D와 guards, native/selected weights, registry/history/cache transaction, 시간·메모리·모든 forward/backward token 수.

GSS working-set과 full local QP가 같으면 이는 solver 검증이다. Reference 선택 안정성이 올라가도 unseen NS가 안 오르면 transfer failure다. D=0 skip이 많으면 기준이 너무 약한지 또는 실제 bank가 안전한지 m/d와 N 손상을 함께 분석한다. 확률 붕괴가 큰데 token choice만 유지된다면 probability floor의 필요성에 대한 근거가 되지만, 그 결과를 보고 같은 v2 안에 δ를 몰래 추가하지 않는다.

이 방법의 연구 가설은 single-layer fixed-key 구조에서 현재 편집 response를 유지하면서 **선택 경계 위반에 필요한 만큼만 write를 바꾸는 것**이다. Margin constraints, QP, GSS 자체는 기존 아이디어다. EN의 KL 감소 실패를 근거로 이 변경이 실제 locality를 개선하는지 별도 검증해야 한다.
