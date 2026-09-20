# EN_ADAPT + GSS history replay bank v1

2026-09-20, R2 실행 일정 수정. 상태: **방법·실험 계약 및 CPU reference 설계**. 사용자 지시에 따라 B2까지만 사전 확인한 뒤 바로 10k lifelong으로 이어간다. Neural runner 구현, GPU 제출, 성능 측정은 이번 작업에 포함하지 않는다.

부모 규약: [EN_ADAPT cold B300](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-en-adaptive-nullspace-experiment-v1.md). 부모의 completed/design 상태를 바꾸지 않고, history512 초과 이후의 별도 확장으로 정의한다.

## 1. 권고 구성

**L4 local-z native write → reference/history KL gradient → 기존 adaptive null-space selector → 최대 두 번의 실제 KL 후보 평가**를 유지한다. History replay bank를 GSS의 gradient diversity 원리로 관리한다.

- 고정 reference: 기존 C4 **512개 전부**, W0 teacher, 최대 생성256/EOS. 문서나 token을 GSS로 줄이지 않는다.
- History: **latest-valid 편집 사실 최대512개**, canonical supplied target, arm별 최종 at-write teacher. 공식 PS/N은 학습·선택에 접근하지 않는다.
- GSS 선택 feature: 현재 native endpoint에서 계산한 **desired-target NLL의 L4 gradient**.
- 실제 replay loss: 기존 **at-write teacher full-vocabulary KL**. Selection NLL을 목적함수에 몰래 더하지 않는다.
- 최근 편집 우대: replay loss의 bounded recency weight. GSS cosine에 gradient 크기 가중을 넣는 방식은 정규화에서 상쇄되므로 사용하지 않는다.
- Current covariance/Q_tau는 current edit만 사용한다. History를 hard null-space에 추가해 exact EN으로 돌아가지 않는다.
- 추가 local-z 최적화, margin guard, reference별 무손실 조건은 없다.

이 구성은 GSS 원본의 재현이 아니다. 원 논문의 signed gradient diversity surrogate를 사용하고, single-layer factor sketch·batch pruning·별도 KL replay·recency weighting을 결합한 **GSS 기반 adaptation**이다. GSS 자체를 ours의 novelty라고 주장하지 않는다.

## 2. 세 종류의 history를 분리한다

|구조|크기|역할|
|Native solver history M|기존 native 누적 정책|AlphaEdit write의 history penalty; replay eviction과 무관|
|Semantic ledger|모든 요청·target version·overwrite 이력|유효 사실과 과거 평가 anchor를 추적; gradient sweep하지 않음|
|Replay bank H|최대512 active fact versions|이번 EN 목적함수에 들어가는 사실|
|Pending queue A|직전 batch의 신규 active versions, 최대100|다음 batch에서 H와 함께 admission 후보가 됨|

Hot selection pool은 최대612개다. **512는 steady replay bank cap**이며 admission 순간의612와 혼동하지 않는다. Eviction은 semantic ledger 삭제나 native M의 key 제거를 의미하지 않는다. Evicted 사실은 기본 v1에서 자동 재검색하지 않는다. 추후 재등장/overwrite는 ledger 규칙을 따른다.

Cold text/teacher artifact와 hot replay tensor의 bytes를 따로 기록한다. Full ledger를 저장한다고 모든 과거 사실이 매 batch 보호되는 것은 아니다.

Version별 원 teacher는 cold artifact store에 유지한다. RES의 slot 보충과 같은-target 재등장에 필요한 teacher를 최신 state로 재생성하지 않는다. 따라서 hot cap512는 전체 디스크 저장량 cap이 아니며 cold teacher 누적 bytes도 보고한다.

## 3. Target와 teacher 규칙

History fact i의 canonical prompt x_i와 supplied target y_i*를 그대로 사용한다. 입력은 `(x_i, y*_{i,<s})`다. 정답 길이 T_i는 supplied target 전체이며 **256-token 생성 길이를 history 정답에 적용하지 않는다**. EOS는 원 supervised target에 있을 때만 label에 포함한다. History를16/128/256 token으로 생성해서 대체하지 않는다.

해당 fact version이 들어온 batch의 arm별 최종 accepted endpoint W_i^a에서

\[
q_{is}=p_{W_i^a}(\cdot\mid x_i,y^*_{i,<s})
\]

를 저장한다. Native fallback도 실제 최종 state이므로 그 teacher를 저장한다. 다음 entry 분포로 q를 매번 갱신하지 않는다. Teacher는 당시 모델 행동이며, 당시 실패한 rewrite를 정답으로 만들어 주지 않는다. At-write paired/strict 성공 여부와 target NLL을 함께 기록하고 실패 요청도 ledger/평가에서 빼지 않는다.

Teacher 추출은 최종 rewrite observer와 입력·prefix·precision이 동일하면 한 forward를 공유한다. Full-vocab teacher 저장/읽기/압축 비용을 계측하고 top-k KL로 바꾸지 않는다. 최종 B100 teacher는 후속 replay 소비가 없으므로 평가에 필요한 scalar 외 새 teacher forward를 요구하지 않는다. B2는 lifelong에 바로 이어지는 중간 state이므로 teacher와 pending queue를 반드시 보존한다.

Fact identity는 dataset의 `(subject, relation_id)`를 우선 사용한다. 이를 사용할 수 없는 데이터는 별도 봉인된 identity mapping이 필요하며 임의 semantic matching을 만들지 않는다.

- 다른 target의 재지시: old version은 이번 batch replay에서 제외하고, commit 이후 새 version만 active.
- 같은 fact/같은 target 반복: memory slot 중복·teacher 재설정·age 초기화를 하지 않는다. 기존 canonical teacher를 유지하고 occurrence만 ledger에 추가한다. 이미 evicted라면 새 occurrence를 admission 후보로 재제안할 수 있지만 teacher/age는 기존 version을 유지한다.
- Current에 포함된 fact는 그 batch의 past loss에서 제외한다. 같은 target 반복으로 제외된 기존 version은 이후 active pool로 복귀한다.
- 같은 batch 내부 상충 target은 원 native 처리 결과를 그대로 기록하고, ledger의 latest order로 active version을 정한다. 이를 동시에 만족시킨 편집으로 주장하지 않는다.
- Overwrite와 admission은 commit ID에 묶어 원자적으로 적용한다. 실패/재시작 시 teacher append나 native M append를 중복하지 않는다.

## 4. EN 보정 목적함수

기존 document-normalized reference loss를 L_R라 한다. History별 replay loss는

\[
\ell_i^{KL}(W)=\frac1{T_i}\sum_{s=1}^{T_i}KL(q_{is}\Vert p_W(\cdot\mid x_i,y^*_{i,<s})).
\tag{1}
\]

현재 batch t에서 age는 `a_i=t-1-version_created_batch_i≥0`이다. 새로 직전 batch에서 들어온 version의 age는0이다. Primary recency 규칙은

\[
h=\frac{C_H}{B}=\frac{512}{100}=5.12\;\text{batches},\qquad
w_i=1+2^{-a_i/h},\qquad\omega_i=\frac{w_i}{\sum_{j\in H_t}w_j}.
\tag{2}
\]

\[
L_H(W)=\sum_{i\in H_t}\omega_i\ell_i^{KL}(W),\qquad
J_t(W)=L_R(W)+\mathbf1_{H_t\ne\emptyset}L_H(W).
\tag{3}
\]

최근 fact의 원 weight는2, 오래된 fact는1에 접근한다. 오래된 fact의 loss weight를0으로 만들지 않으며 최근/오래된 개별 weight 비율은2 이하다. h를 nominal bank turnover horizon에 연결했지만, **2배 및 h=5.12가 최적이라는 정리는 없다**. Uniform w=1 대조군으로 recency 효과를 분리한다. Selection에 최근100 강제 quota나 별도 risk threshold까지 추가하지 않는다.

Reference/history를 각각 평균하므로 history 크기가 커져도 history block의 총 coefficient는1이다. 이 역시 고정 convention이다. 두 block의 gradient norm과 cosine을 기록하여 같은 coefficient가 같은 영향력인 것처럼 해석하지 않는다.

Native endpoint W_N에서

\[
G_R=\nabla L_R(W_N),\quad G_H=\sum_i\omega_i\nabla\ell_i^{KL}(W_N),\quad G=G_R+G_H.
\tag{4}
\]

부모 selector에는 **이 G와 J_N**을 넣는다. 각 Q_j에 대해

\[
H_j=GQ_j,\qquad
\eta_j=\min\left(\frac{J_N}{\|H_j\|^2},\frac{\|\Delta_N\|}{\|H_j\|},
\frac{0.05\|\Delta_N\bar K_E\|}{\|H_j\bar K_E\|}\right).
\tag{5}
\]

Predicted decrease 최대 boundary와 최대2개의 실제 candidate 평가 규칙은 그대로다. Candidate 사이에 bank·teacher·weights를 바꾸지 않는다. 모든 candidate에서 R512와 **선택된 history 전체≤512**로 같은 J를 계산한다. Candidate-specific GSS selection은 금지한다.

`||GQ||²`에는 reference/history 교차항이 있으므로 두 gradient의 spectral energy를 단순 합산하지 않는다. Mode별/전체 `2⟨G_RQ,G_HQ⟩`를 기록한다. L_R 감소가 L_H 증가를 가리거나 그 반대가 가능하므로 두 loss를 따로 보고한다. Sum-KL 개선이 과거 RS/PS의 무손실 보장은 아니다.

## 5. GSS에는 어떤 gradient를 사용하는가

At-write teacher를 저장한 같은 state에서 `∇KL(q||p)=0`이다. 따라서 admission 시 teacher KL gradient를 저장해 장기간 비교하는 방식은 적절하지 않다. 현재 W_N에서 primitive desired-target loss를 정의한다.

\[
\ell_i^{NLL}(W)=-\frac1{T_i}\sum_s\log p_W(y^*_{is}\mid x_i,y^*_{i,<s}),\qquad
F_i=\nabla_W\ell_i^{NLL}(W_N)P_*.
\tag{6}
\]

P_*는 부모의 고정 allowed space다. 매번 바뀌는 Q_tau 좌표에서 과거 sketch를 비교하지 않는다. **동일 native state·동일 고정 sketch map에서 pool 전체를 refresh**한다. 고정 key라고 activation gradient까지 고정되는 것은 아니다.

GSS는 signed cosine을 사용한다. 반대 방향은 중복으로 취급하지 않으므로 absolute cosine은 사용하지 않는다. NLL의 양의 recency weight를 곱하고 cosine normalize하면 weight가 사라진다. Recency는 식(2)의 실제 replay loss에 적용한다.

History pool≤512이면 전부 선택하므로 **GSS용 NLL backward 자체를 실행하지 않는다**. B1은 history가 없고 B2–B6도 overflow가 없으면 selection 계산이 없다. 원 stream의 overwrite/중복을 반영한 active count가 기준이다.

## 6. Single-layer 이점: per-fact dense gradient를 만들지 않는 feature

Teacher-forcing 입력의 모든 유효 input position에 대한 L4 key를 K_i, 해당 L4 linear output의 activation gradient를 A_i^NLL라 하면

\[
\nabla_W\ell_i^{NLL}=A_i^{NLL}K_i^T.
\tag{7}
\]

NLL/KL loss는 target 위치에서만 계산하지만 **A_i에는 prefix 위치도 포함**한다. Downstream attention을 통해 prompt activation도 target loss에 영향을 주므로 target-token key만으로 식(7)을 대체하지 않는다. Causal packing을 쓰면 example 사이 attention이 완전히 차단되어야 한다.

고정 Gaussian random maps O∈R^(o×32), V∈R^(d×32)를 원소 N(0,1/32)로 두고 두 독립 replica를 사용한다. P_*V를 한 번 계산하고, 고정 input마다 `C_i=VᵀP_*K_i`를 캐시한다.

\[
S_i=O^T F_i V=(O^TA_i^{NLL})C_i^T\in\mathbb R^{32\times32}.
\tag{8}
\]

두 replica의 vec(S)를 연결해 √2로 나눈 2048차원 sketch z_i를 사용한다. Map이 독립·등방성이면 `E⟨S_i,S_j⟩=⟨F_i,F_j⟩`이지만 **정규화된 cosine이 unbiased이거나 2048차원으로 정확하다는 보장은 없다**. 이 크기는 첫 실험의 비용 설정이다.

- 612개 sketch FP32는 약5.01MB(decimal); dense L4 per-fact gradient612개는 약143.75GB다.
- Native-state NLL sketch는 다음 batch에서 다시 계산한다. 재사용하는 것은 K_i와 projected-key map이며, activation gradient나 state-dependent score가 아니다.
- Full A_i^NLL는 sketch 후 버린다. Replay용 A_i^KL만 admission 완료까지 CPU pinned/host cache에 보관할 수 있다.
- Corrected W가 달라도 같은 입력의 upstream key와 sketch input map은 공유 가능하다. 서로 다른 W_N의 A·G·loss·teacher는 공유하지 않는다.

실제 모델의 고정 hash64 fact에서 factorized exact F inner product와 sketch cosine을 비교한다. Exact 비교는 `(A_iᵀA_j)`와 `(K_iᵀP_*K_j)`의 contraction으로 수행하고 dense per-fact gradient는 만들지 않는다. Pair rank/sign error와 sketch 두 seed의 bank overlap을 보고한다. 이것은 품질을 측정하기 위한 한 번의 진단이며, 과거 numeric waiver를 새 PASS로 바꾸지 않는다. 불안정하면 `SKETCH_SELECTION_UNRESOLVED`로 명시하고 차원 확대는 새 configuration으로 구분한다.

## 7. 명시적인 GSS selection: signed-cosine backward pruning

원 GSS의 surrogate는 정규화 gradient의 pairwise signed cosine 합이다. [논문 §3.3–3.4](https://arxiv.org/html/1903.08671). 여기서는 612개 이하의 refresh된 sketch에서 다음을 사용한다.

\[
u_i=z_i/\|z_i\|,\qquad
\Phi(S)=\sum_{i,j\in S}u_i^Tu_j=\left\|\sum_{i\in S}u_i\right\|^2.
\tag{9}
\]

Pool U에서 시작해 크기가512가 될 때까지

\[
j^*=\arg\max_{j\in S}u_j^T\sum_{i\in S}u_i
\tag{10}
\]

를 제거한다. `Φ(S\{j})=Φ(S)-2u_jᵀΣu_i+1`이므로 **한 번의 제거마다 해당 surrogate를 가장 작게 만드는 선택**이다. 매 단계 모든 후보 크기는 같아서 비교 가능하지만 전체 subset의 global optimum이나 모든 gradient 방향의 coverage를 보장하지 않는다. 반대 두 cluster가 상쇄되는 한계도 있으므로 bank의 relation/age/gradient coverage를 함께 보고한다.

모든 pair의 작은 Gram을 한 번 만든 뒤 row-sum을 갱신한다. Tie는 고정 hash(priority seed, fact-version ID)로 정한다. 최근성 tie-break는 넣지 않아 uniform/recency 비교를 혼동하지 않는다. 원 GSS-Greedy Algorithm2의 random-subset·확률적 replacement·c<1 규칙을 재현했다고 부르지 않는다.

Nonfinite gradient는 기술 실패다. Exact-zero 또는 sketch가 수치상 zero인 row는 cosine을 정의하지 않는다. Nonzero가 cap 이상이면 이 중 GSS 선택, 부족하면 모두 보존하고 zero pool에서 hash priority로 남은 slot을 채운다. Zero를 보호 성공으로 해석하지 않고 제외/선택 건수를 기록한다. Toy 진단과 실제 sketch 오차 진단에서 projected-norm과 sketch cancellation을 구분한다.

Bank에서 빠진 사실도 전체 semantic ledger와 공식 평가에 남는다. GSS가 작은 loss objective를 만들기 위해 teacher를 삭제하거나 실패 사례를 denominator에서 빼는 것을 허용하지 않는다.

## 8. 계산 재사용: 비용을 숨기지 않는 실행 순서

### Pool≤512

전부 선택 → history KL forward/backward 1회 sweep → reference gradient와 합 → selector → 최대2 candidate joint-KL forward. NLL selection·sketch·Gram 계산은0이다.

### Pool>512

1. W_N에서 pool≤612를 microbatch로 forward한다.
2. **동일 microbatch graph에서 NLL VJP와 KL VJP 두 번**을 계산한다. NLL A는 sketch로 압축하고, KL A는 fact별로 보관한다. Teacher KL scalar도 fact별 저장한다.
3. 해당 microbatch graph를 해제하고 다음으로 넘어간다. Pool 전체의 autograd graph를 GPU에 유지하지 않는다.
4. 모든 sketch가 모이면 GSS를 한 번 수행한다. 선택된 fact의 KL A와 fixed K로 G_H를 합성하고, 선택된 scalar/weights로 L_H(W_N)를 구한다.
5. G_R+G_H로 기존 threshold/step을 선택한다. 최대2 candidate에서 selected history≤512만 forward한다. Selection pool612 전체를 candidate마다 평가하지 않는다.

따라서 **overflow batch의 GSS 추가비용은 NLL backward 1 pool sweep, 탈락 후보에 대한 KL backward, sketch/selection 계산**이다. “KL backward에서 NLL gradient도 공짜로 얻는다”고 쓰지 않는다. 두 cotangent가 다르기 때문이다. 동일 microbatch의 logits/graph 재사용으로 **두 번째 backbone forward**를 제거하는 것이 실제 절감이다.

Per-fact KL activation cache FP32 크기는 `4*4096*Σ_i L_i` bytes다. K cache와 full-vocab teacher bytes는 별도다. History target이 짧아도 backward는 prompt 길이에도 의존하므로 label-token 수만으로 비용을 추정하지 않는다. Native/reference130k 위치와의 runtime 비율은 실측한다.

각 VJP는 microbatch 내 **fact별 target-token 평균 loss의 합**을 사용한다. Example 간 attention이 독립이면 한 번의 합-loss VJP로 각 row의 A_i를 얻는다. Microbatch size로 한 번 더 평균해도 된다고 가정하지 않는다. 최종 ω_i와 bank normalization은 selection 후 G_H 합성 단계에서 적용한다. NLL VJP는 graph를 유지하고 KL VJP 후 해제하며, 두 VJP 사이 parameter `.grad` 누적을 correction G로 혼용하지 않는다.

Neural 연결 시 hash로 고른4 history fact에서 factor로 합성한 selected KL gradient와 직접 weighted-KL autograd를 한 번 대조한다. 이 진단의 backward는 method throughput 외 별도 계측한다. CPU reference PASS는 이 actual-model 검증이나 sketch fidelity PASS를 대신하지 않는다.

같은 final endpoint observer에서 teacher를 추출하는 비용 재사용, candidate별 current downstream guard 제거, single native gradient, current SVD1회, 추가 z0, candidate≤2의 부모 규칙을 유지한다. 실제 teacher I/O와 실패 후보 비용도 포함한다.

## 9. 실험 arm과 최소 길이

모든 arm은 W0/M0에서 시작하는 동일 fixed10k 전체10,000개, batch100이다. **사전 확인은 B1–B2의 총200 edit로 끝내고, 그 B2 state에서 B3–B100 lifelong을 연속 실행한다.** 별도 B300/B1000 pilot이나 그 결과를 기다리는 중간 의사결정 단계는 두지 않는다. 외부 5k-edit checkpoint에서 시작하거나 B2 뒤 W0부터 중복 실행하지 않는다.

|Arm|Bank policy|History KL weight|분리하는 효과|
|EN_ADAPT_H_RES|전체 ledger의 active versions 중 hash-priority reservoir512|uniform|bounded replay baseline|
|EN_ADAPT_H_GSS|fresh NLL sketch의 GSS pruning512|uniform|random selection 대비 GSS|
|EN_ADAPT_H_GSS_REC|동일 GSS policy512|식(2)의 bounded recency|추가 최근 편집 우대|

RES는 균등 random priority의 bottom512를 active ledger 전체에서 유지한다. Overwrite로 slot이 비면 다음 active priority를 ledger에서 보충한다. 이것은 GSS의 evicted fact를 자동 재평가하는 기능과 다르며, RES는 metadata만으로 보충 가능하다. Teacher/key artifact의 cold access 비용도 계측한다.

별도 R-only EN_ADAPT나 N4와의 lifelong 우위는 이 세 arm만으로 주장하지 않는다. 기존 동일-source baseline이 있으면 별도 참조하고, 없으면 다음 효능 비교의 baseline으로 명시한다. 이번 질문의 primary는 **기존 history KL 위의 GSS 선택 가치**다.

### 두 batch 확인 → 즉시 lifelong

1. **B1:** history가 비어 있으므로 세 arm은 같은 EN_ADAPT endpoint다. Native write, R512 objective, correction과 final teacher 저장을 확인한다. GSS가 B1 NS를 개선한다는 주장을 하지 않는다.
2. **B2:** past100 전부 replay. Teacher/target/prefix 연결, same-state gradient 합성, ledger/overwrite, joint KL controller, 실제 runtime·메모리·후보 수를 확인한다. Current 및 past R/P/N을 관측하되 locality 향상·RS/PS 우위를 다음 단계의 통과 조건으로 삼지 않는다. 모든 past의 age가 같으면 REC도 동일 endpoint다.
3. **B2 확인 직후:** 동일 source/config를 유지한 채 사전 지정한 세 arm 모두 B3–B100으로 진행한다. 별도 성능 승인·B3 pilot·B10 pilot을 기다리지 않는다. B2 W, native M, active ledger, bank, pending queue, immutable teacher, input/map identity, RNG 및 commit 상태를 보존하여 이어간다. Runtime 무결성 오류는 수정하되 단순 native fallback이나 성능 무개선을 기술 실패로 바꾸지 않는다.
4. **Lifelong 내부 B3 이후:** REC와 uniform chain이 달라질 수 있다. 동일 endpoint임이 확인되는 prefix만 공유한다. GSS/RES의 차이는 실제 pool이512를 초과할 때부터 발생하며, 중복/overwrite가 없으면 B7이다. **B7은 중간 gate가 아니라 bank overflow 이벤트**다. 최초 overflow의 sketch/selection 진단은 해당 실행에 기록하고 별도 pilot으로 분리하지 않는다.
5. **B100 종료:** 세 arm의10k 결과를 모두 보고한다. Pool이 끝까지512를 넘지 않았다면 `GSS_NOT_EXERCISED`로 표시하며 인위적으로 bank를 줄이지 않는다. 성능에 따라 horizon·bank size·weight를 중간 변경하지 않는다.

모든 batch에서 current R/P/N, first100 고정 cohort 및 고정 hash history panel을 관측한다. B2, B5, B10, 이후 매10 batch와 B100에서 **전체 latest-valid history**의 RS/PS 및 누적 N을 평가하고 checkpoint를 보존한다. 이 시점들은 lifelong 내부의 관측/저장 지점이며 진행 여부를 다시 판단하는 gate가 아니다. 공식 paraphrase는 replay에 넣지 않는다. Observer 비용은 method 비용과 별도이되 총 연구비용에는 포함한다.

보고할 것: at-write 성공의 신규 forgetting/회복, native 전후 vs correction 전후 변화, bank 안/밖, age별, relation별, exact overwrite 제외/포함, paired와 strict R/PS, NS/NLL, selected-rank/epsilon/actual response, KL_R/KL_H, fallback, sketch 오차, teacher 실패율.

Native가 만든 history 손상을 보정이 복구했는지와 correction 자체가 새 손상을 만들었는지 구분한다. KL 감소와 RS/PS 유지가 다르면 그대로 실패 범위를 보고한다. Canonical history replay만으로 unseen PS 보존은 보장되지 않는다.

## 10. 비용 계약과 해석

Overwrite가 없고 입력/state가 정확히 공유되는 경우 uniform RES/GSS는 B6까지 같고 REC도 B2까지 같다. 따라서 B1–B100 세 cold chain의 실행을 prefix 공유하면 **native/R512 gradient state292개**, reference candidate sweep≤584다. B1–B2 사전 확인은 이 합계에 포함하며 별도 반복하지 않는다. 이것은 세 arm 합산 연구비용이며 standalone method를292/3으로 나누지 않는다. 단독 method는100 batch의 native 및 reference gradient를 사용한다.

두 GSS arm의 B7–B100 selection pool은 각각 첫600개와 이후93회 최대612개로 최대합57,516이다. 두 arm의 NLL selection VJP는 **115,032 fact-evaluations**, sweep188회다. 실제 overwrite와 중복에 따라 감소할 수 있다. 같은 pool의 KL VJP도 실행하지만 두 VJP는 forward를 공유한다. Candidate history는 각 batch의 selected bank를 최대2회 평가한다. 전체 method cost ledger에는 아래를 각각 남긴다.

- native z/solve; current key/SVD; R512 backward/candidate forward.
- history pool forward; NLL VJP; KL VJP; selected-gradient 합성.
- fixed key map cache; sketch; Gram/pruning; activation cache D2H/bytes.
- final teacher 추출/I/O; history candidate forward; official observer.

두 GSS arm의 memory선택용 CPU sketch 크기·matmul과 모든 cap은 **설계 수치**다. 속도향상이나 EN 대비 overhead 감소는 아직 미측정이다. KL 대신 NLL을 실제 replay loss로 쓰면 두 VJP를 하나로 합칠 수 있지만 loss의 의미가 바뀌므로 이 v1의 조용한 최적화로 넣지 않는다. 필요하면 별도 `H_NLL` ablation으로 검증한다.

## 11. 연구 포지션과 실패 판정

Ours의 중심은 **single-layer에서 finite edit-response 비용을 직접 계산하고, 그 예산 안에서 reference와 과거 편집의 기능 loss를 줄이는 adaptive write correction**이다. GSS는 제한된 factual history가 서로 다른 보호 방향을 담도록 하는 memory 구성 요소다.

- RES≈GSS이면 memory diversity의 추가 가치는 미확립.
- GSS의 bank KL만 좋아지고 bank 밖 R/PS가 나빠지면 대표성 실패.
- REC에서 최근 R/PS 개선·오래된 R/PS 악화면 recency tradeoff.
- Joint KL 개선에도 NS 불변이면 기존 reference와 locality의 proxy 문제를 해결하지 못한 것.
- GSS가 비용을 크게 늘리면 compact sketch만으로 병목을 해결했다고 주장하지 않음.
- P 전체가 선택되거나 history/reference gradient가 강하게 상쇄되면 해당 geometry 결과를 보고하고 성공한 cutoff만 선택적으로 해석하지 않음.

출처: [GSS 논문](https://arxiv.org/abs/1903.08671), [저자 코드](https://github.com/rahafaljundi/Gradient-based-Sample-Selection). 원 논문의 diversity surrogate와 ours의 replay/recency/sketch/controller 변형을 구분한다.

기계 계약: [JSON](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-en-adapt-gss-history-contract-v1.json).
실행 cell: [CSV](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-en-adapt-gss-history-cells-v1.csv).
CPU reference와 검산: [directory](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-en-adapt-gss-history-design-v1).
