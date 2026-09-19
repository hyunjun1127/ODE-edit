# Server2 L4-only checkpoint 재사용과 GSS history 보호 검토

작성: 2026-09-20 KST. 상태: **자산 직접 확인·수식 검토·후속 실험 범위 제안**. GPU 실행, checkpoint 복원 forward, 새 editing 실험은 하지 않았다. 2026-09-19 설계/contract를 덮어쓰지 않는 후속 검토다.

직접 확인 receipt: [checkpoint inventory](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-geometry-gss-history-review/server2-checkpoint-inventory.json). 기존 설계: [mechanism-first v1](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-19-single-layer-mechanism-first-experiment-design-v1.md).

## 1. 판단

1. **Server2 checkpoint를 먼저 사용하는 것이 타당하다.** L4-only 12개 checkpoint가 실제 존재한다. W/history는 Server2, native target/평가는 Server4에 나뉘어 남아 있다. 원 Server4 weight 경로 부재를 전체 분석 자산 부재로 확대한 판단은 정정해야 한다.
2. 분석 대상은 같은 입력 key의 시간적 변화가 아니라 **고정 key와 커지는 history metric의 관계, target demand, 실제 write, 최종 판단의 연결**이다.
3. W0 reference 512개·최대256 생성 길이는 그대로 유지하고, **latest-valid edit history에 별도 기능적 보호 조건**을 둔다. Reference와 history는 같은 correction에 들어가지만 기준 target은 다르다.
4. GSS는 history의 중복된 보호 방향을 줄이는 선택 장치다. **누적 activation 변위를 줄이는 목적이나 과거 정답 보장을 대신하지 않는다.** B1에는 history가 없으므로 GSS 추가로 기존 EN의 B1 locality 무개선이 설명되거나 해결되지는 않는다.

## 2. 실제 남아 있는 자산

최초 확인 시간은 2026-09-20 02:15–02:19 KST이며 이후 소형 companion 결속을 추가 확인했다. 읽기 전용 SSH, CPU mmap, metadata와 작은 diagonal 및 소형 B1 target/평가 검사를 수행했다.

|항목|확인 결과|검증 수준|
|---|---|---|
|Arm|AlphaEdit_L4_ONLY, 표시명 AlphaEdit_BLUE_L4_ONLY|B1/B100 metadata 및 source 설정|
|저장 경계|B001, B005, B010, B020, B030, B040, B050, B060, B070, B080, B090, B100|12개 현재 stat 및 이전 index 크기 일치|
|편집 수|100, 500, 1000, 2000, …, 10000|B100 설정·seen_ids|
|수정 weight|model.layers.4.mlp.down_proj.weight 하나|B1/B100 실제 weights_only load|
|Weight|FP32, 4096×14336|B1/B100 schema|
|History cache_c|FP32, 1×14336×14336; cache_c_is_history=true|B1/B100 schema|
|개별 key/residual/z|W-method-state.pt 내부에는 없음|B1/B100 schema|
|Source/config|commit/runtime/terminal/hparams/lock 보존|소형 source 확인; lock SHA 결속|
|Server4 companion|B1/B100 native-targets.pt, seen-full.json 존재|정확한 원경로 stat|

추가로 B1 소형 companion을 실제 읽었다. `native-targets.pt`의 values100개는 모두 FP32[4096], identities의 layer는 모두4다. Target ID 순서가 S2 B1 seen_ids와 일치한다. 평가 R100/P200/N1000의 ID 집합과 평가에 기록된 W/history hash도 해당 checkpoint metadata와 일치한다. 이는 기록 간 결속 검증이며 전체 W payload SHA 재계산은 아니다.

이 archive의 B1 성적은 **R100/100, P190/200, N867/1000**이다. 최근 EN cold B1의100/194/865와 다른 실행이므로 같은 endpoint처럼 합치지 않는다. 또한 평가 raw margin은 공통으로 `true_nll−new_nll`이어서 N 성공은 음수다. Locality 분석의 보호 margin에는 부호를 반대로 정규화해야 한다.

S4 `B002/native-targets.pt`도 현재 존재한다. 따라서 **S2 B1 entry + S4 B2 z**를 이용한 다음 batch residual 분석의 출발 자산이 있다. B2 target 내용·입력/source 결속과 실제 key/activation 추출은 후속 검증 사항이다. B100의 66MB 평가 JSON은 이번에 읽지 않았다.

Server2 checkpoint 공통 경로:

```text
/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/payload/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/Bxxx/W-method-state.pt
```

Server4 companion 공통 경로:

```text
/data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/Bxxx/
```

Pinned model revision은 `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, source commit은 `1075540b45c29269e690ac63aae44758d8d63174`다. Context group 크기는 [1,5]다. 설정의 clamp=.75는 **z target clamp**이며 과거 별도 실험의 L4 write 75% 적용과 다르다.

|History diagonal sanity|B1|B100|
|---|---:|---:|
|Trace|779.80191845|79,876.31351596|
|최소 diagonal|0.00197455|0.23265624|
|최대 diagonal|17.93844986|1,726.38183594|
|Finite/nonnegative|14,336/14,336|14,336/14,336|

이는 **PSD, rank, conditioning 또는 capacity의 측정 결과가 아니다.** Request 수 증가에 따른 trace 증가만으로 충돌 증가를 주장할 수 없다. 전체 tensor hash/finite scan, 큰 행렬 연산, eigendecomposition은 수행하지 않았다. 이관 manifest의 SHA는 과거 증거이며 이번에 재계산한 값이 아니다.

## 3. Checkpoint로 무엇을 정확히 알 수 있는가

### 3.1 Key는 고정되고 history가 변한다

L4 down projection 하나만 변경하고, token/prefix/mask/position/실행 모드를 고정하면

\[
K_x(W_t)=K_x(W_0),\qquad H_x(W_t)=H_x(W_0)+(W_t-W_0)K_x.
\]

이는 module/block 반응에 대한 항등식이다. 고정 suffix 함수가 비선형이므로 최종 logit/margin이 weight에 선형이라는 뜻은 아니다. 생성 prefix가 바뀌는 free decoding에도 동일 key를 그대로 적용할 수 없다.

따라서 W0 prefix를 한 번 실행하여 고정 probe와 선택한 다음 batch의 keys를 확보하면 모든 checkpoint에서 재사용할 수 있다. Checkpoint마다 local-z를 다시 최적화하거나 10k lifelong을 재실행할 필요는 없다.

Native writer K는 **context type 안의 평균 후 type 사이 평균**이다. [compute_ks](/mnt/raid5/janghj/layer_allocation/empirical/code/alpha_compute_ks.py:17). Subject mean key와 모든 token의 기능적 보호용 key는 별도로 관리한다.

### 3.2 개별 K 없이 가능한 history mapping 변화

저장 history가 \(M_a=\sum_{s\le a}K_sK_s^T\), 구간 update가 \(D_{a:b}=W_b-W_a\)이면

\[
\boxed{\operatorname{tr}(D_{a:b}M_aD_{a:b}^T)=\sum_{s\le a}\|D_{a:b}K_s\|_F^2.}
\]

이는 **과거 writer mean-key 반응의 변화량**이다. 과거 사실의 RS/PS 손실과 동일하지 않다. 수식상 개별 key 복원 없이 가능하지만, dense 14336 차원의 실제 계산은 비용이 있으므로 metadata 검사와 구별한다. 필요하면 고정 random sketch로 추정하고 seed·오차·정확 계산 여부를 표시한다.

위 항등식은 정확연산의 M에 대한 것이다. 실제 FP32 저장값이 \(M_a^{stored}=\sum K_sK_s^T+E_a^{round}\)이면 계산값에는 \(\operatorname{tr}(D_{a:b}E_a^{round}D_{a:b}^T)\)가 더해진다. 따라서 실제 receipt에는 **저장 history quadratic form**으로 표기하고, 개별 key-response energy와의 동일성에는 누적 오차 한계를 명시한다. 작은 음수 eigenvalue를 실제 covariance의 음수 방향으로 해석하지 않는다.

\(M_b-M_a\)에서는 구간의 aggregate covariance를 얻는다. M만으로 개별 key, prompt, target, gradient를 역복원할 수는 없다. Accepted response \(Y_s\)로부터의 누적 drift에는 \(\sum_sY_sK_s^T\) 같은 추가 target 통계가 필요하다.

### 3.3 Sparse boundary와 post-write history를 혼동하지 않는다

- B5−B1은 4batch, B10−B5는 5batch, 이후 저장 경계 차이는 보통 10batch의 **합산 update**다.
- B100−B1을 마지막 batch의 write로 해석할 수 없다.
- 저장 M_t는 B_t **종료 후** history다. B_t의 native solve에 entry history인 것처럼 넣으면 current key를 중복 반영한다.
- 원래 한 batch의 demand 분석은 W0→B1 또는 **저장 W_t와 다음 batch t+1의 원래 z**가 짝을 이루는 경계를 우선한다.
- B100 z가 남아 있어도 W99가 없으면 원래 B100 entry residual은 정확하게 복원되지 않는다.

### 3.4 Conditioning만으로 write 증폭을 결론 내리지 않는다

이상적 allowed projector를 \(P=UU^T\), \(X=U^TK\),

\[
C_t=\lambda I+U^TM_tU,\qquad G_t=X^TC_t^{-1}X
\]

로 두면 ideal ridge writer의 batch fitting은

\[
\Delta K=R\,G_t(I+G_t)^{-1}
\]

이다. G_t eigenvalue \(\theta\)의 fitting gain은 \(\theta/(1+\theta)\)다. 작은 singular value가 항상 큰 actual write로 이어지는 exact-interpolation 해석을 native AlphaEdit에 그대로 적용하면 안 된다. 작은 mode를 ridge가 억제하여 target fitting이 약해졌을 수도 있다.

같은 probe \(x=U^Tk\ne0\)에는

\[
\nu_t(k)=\frac{\lambda x^TC_t^{-1}x}{\|x\|^2}
\]

를 측정한다. PSD 누적이라는 ideal 조건에서 0<ν≤1이고 history 추가에 따라 비증가다. 이는 **soft history overlap/regularization**이지 key 자체의 drift나 사실 수용량이 아니다. 매번 다른 신규 batch만 비교하면 sample composition이 섞인다. 고정 독립 probe와 실제 다음 batch를 함께 평가하며, stream probe의 self-inclusion도 분리한다.

실제 P_raw, solve residual, precision 차이는 ideal projector와 별도 검증한다. 큰 inverse/square-root를 만들기보다 C_tY=X를 풀어 작은 X^TY를 분석하되 solve tolerance와 실행 비용을 기록한다.

### 3.5 저장 z와 residual의 정확한 연결

실제 residual은 \(R_i=z_i-h_i^{block}(W_{entry})\)다. 여기의 h는 **bare prompt subject 위치 block output**이며 writer mean key와 다르다. [native residual](/mnt/raid5/janghj/layer_allocation/empirical/code/AlphaEdit_main.py:118).

\[
h_i^{block}(W_{entry})=h_i^{block}(W_0)+(W_{entry}-W_0)k_i^{bare}.
\]

따라서 저장 z의 identity/source를 확인하고 W0 bare key/output을 한 번 얻으면 해당 entry의 R을 재구성할 수 있다. 이후 spectrum뿐 아니라 target loading, realized fitting, actual write를 함께 분석한다.

## 4. 누적 교차항의 적절한 역할

\(E=W_{entry}-W_0\), 신규 write Δ, 고정 reference K_R에 대해

\[
B=EK_R,\quad V=\Delta K_R,\quad
\|B+V\|_F^2-\|B\|_F^2=2\langle B,V\rangle_F+\|V\|_F^2.
\]

이는 상쇄/증폭의 정확한 **진단량**이다. 최적화 목표로 동일시하지 않는다. 필요한 edit 역시 W0에서 벗어나며, 반응 norm을 줄여도 실제 margin이 회복된다는 보장은 없다.

세 가지를 결합해 보고한다: (a) 교차항/반응 변화, (b) reference의 실제 W0-choice 변화, (c) history의 latest-target 성공 변화. Activation 감소인데 functional 개선이 없으면 그 방향을 preservation 성과로 인정하지 않는다.

History에는 W0를 목표로 쓰지 않는다. 또한 at-write activation을 완전히 복원하는 hard target도 기본값으로 삼지 않는다. 과거 편집의 **유효한 판단이 유지되는가**가 목적이다.

## 5. Reference와 history를 함께 쓰는 권고 구성

### 5.1 두 집합의 역할

|항목|고정 reference|Edit history|
|---|---|---|
|Coverage|기존 512개 모두, 최대256/EOS|과거 요청의 latest-valid 전체 ledger|
|기준|W0의 고정 continuation과 prefix|현재 유효한 desired target|
|보호 조건|기존 base-choice guide 계승|실제 rewrite 판단의 성공 조건|
|GSS|적용하지 않음; 512를 줄이지 않음|제약/gradient working set 선택에 적용|
|주된 평가|보호 reference + 미사용 dev/N|Past RS, observer Past PS, age별 forgetting|

기존 reference가 C4 continuation이면 이를 factual QA 정답이라고 부르지 않는다. Factual history는 유효 target을 가진 편집 요청에서 온다. 이 검토는 reference source/개수/길이를 바꾸지 않는다.

Full history의 text·ID·target·acceptance ledger는 유지한다. GSS에서 빠졌다는 이유로 과거 사실 기록을 삭제하지 않는다. Overwrite된 old target은 비활성화한다. Native cache_c의 모든 과거 key 누적은 **solver history**이며 latest-valid semantic ledger와 구별한다.

Reference가 명시적 overwrite의 old answer를 요구하면 두 hard 조건이 충돌할 수 있다. 확인된 동일 사실 충돌의 정책·제외/재정의를 사전에 명시하고 별도 보고한다. 모든512를 계속 검사하되 충돌하는 두 target을 동시에 만족했다고 주장하지 않는다. 단순 entity overlap이나 편집 후 손상을 이유로 임의 제외하지 않는다.

### 5.2 Current 편집을 유지하는 correction 공간

Native endpoint \(W_N=W_{entry}+\Delta_N\)에서

\[
W_{final}=W_N+D,\qquad D=DQ_t,\quad Q_tK_{E,t}=0.
\]

K_E,t에 **보호하려는 current 입력의 필요한 모든 token keys**가 포함되고 그 외 weight가 고정되어야 해당 입력의 native 반응/출력이 유지된다. Mean subject key만 고정하면 전체 rewrite/paraphrase 보장으로 확장할 수 없다. 자유 생성은 고정 TF prefix와 별도 검증한다.

Allowed space U 안에서는 \(Z=U^TK_E\), \(Q_t=U(I-ZZ^\dagger)U^T\)로 정의할 수 있다. 수치 rank/tolerance와 실제 finite endpoint parity는 따로 기록한다.

### 5.3 History는 confidence 복원보다 판단 보호를 먼저 둔다

실제 rewrite가 desired/old target의 평균 NLL 비교라면

\[
m_j(W)=L_j^{old}(W)-L_j^{desired}(W)
\]

를 사용한다. 해당 benchmark의 token averaging, prefix, tie 규칙을 그대로 재현한다. 더 큰 m이 desired target에 유리하다.

- **Entry에서 성공한 active history:** 실제 endpoint에서도 성공해야 한다. 이번 batch의 추가 forgetting을 막는 guard다.
- **이전에 수용됐지만 이미 실패한 active history:** 복구 대상으로 구분한다. 매 batch entry를 새로운 acceptance로 덮어쓰지 않는다.
- **처음부터 실패한 요청:** 초기 acquisition failure로 구분한다. 원래 성공했다가 잊힌 경우와 합치지 않는다.

Accepted NLL, 현재 NLL과의 차이, margin 약화는 진단량으로 저장한다. \(L_j(W)\le L_j^{accept}\)를 ε=0으로 hard 지정하면 과거 확신까지 복원하도록 강제하므로 기본 조건으로 권하지 않는다.

Native에서 선형화한 history 조건은

\[
m_j(W_N)+\langle H_j,D\rangle_F\ge0,\qquad H_j=\nabla_Wm_j(W_N)Q_t.
\]

이는 제안용 근사다. 가장 강한 competitor나 비선형 suffix 변화는 최종 actual 검사로 확인한다. Current lock 아래 양의 deficit을 가진 row의 H_j가 0이면 이번 correction 공간에서 **일차 복구 방향이 없는 경우**다. GSS로 해결할 수 있다고 주장하지 않는다.

Hard guard 실패와 native fallback은 보호 성공이 아니다. 이미 손상된 history를 복구하지 못한 결과, 새 forgetting, fallback을 각각 기록한다.

### 5.4 별도 paraphrase target set은 만들지 않는다

Canonical request와 이미 native editing에 사용된 contexts를 같은 fact bundle로 관리할 수 있다. 후자를 새 검증 paraphrase 데이터라고 부르지 않는다. Context별 primitive gradient를 무작정 평균하면 상쇄될 수 있으므로 bundle 내 최악 margin/복수 row를 유지한다.

공식 PS prompt, PS 결과, PS gradient는 GSS admission·선택·threshold·candidate 수용에 사용하지 않는다. Past PS는 observer로 평가한다. **Rewrite 보존으로 unseen paraphrase 보존을 보장할 수는 없다.** Rewrite만 유지되고 PS가 무너지면 해당 보호 범위가 부족하다는 결과다.

## 6. GSS를 어떻게 적용할 것인가

원본 GSS의 gradient diversity 선택, GEM의 gradient projection, MIR의 update 후 손실 증가 검색은 별개다. 여기의 제안은 GSS형 방향 다양성과 history 기능 제약을 결합하는 adaptation이다. 최근 edit 가중치도 원본 GSS의 보장이 아니다. [GSS §3](https://arxiv.org/html/1903.08671#S3), [GEM §3](https://arxiv.org/html/1706.08840#S3), [MIR §3.1](https://arxiv.org/html/1908.04742#S3.SS1).

### 6.1 두 gradient 공간을 구분한다

**장기 admission/후보 index:** 고정 allowed space P_*에서 primitive target margin/NLL gradient를 사용한다. **현재 batch correction 선택:** 같은 W_N에서 재계산한 H_j=G_jQ_t를 사용한다.

현재 batch의 full TF keys가 Q_t lock에 들어 있으면 새 요청의 G_new Q_t=0이다. 이를 장기 memory 중요도에 사용하면 새 edit을 모두 불필요하다고 판단한다. 이 영벡터는 학습 완료나 낮은 중요도의 뜻이 아니다.

또한 안전한 요청의 hinge-risk gradient는 0이므로 GSS feature로 쓰지 않는다. Primitive margin 또는 NLL gradient를 사용한다. Cached admission gradient는 오래될 수 있어 **후보 탐색 힌트**이며 현재 endpoint의 정확한 gradient와 구별한다.

### 6.2 Slack과 방향 다양성을 함께 본다

위 history halfspace에는 m_j(W_N)라는 offset이 있다. 같은 정규화 gradient 방향의 제약은 **정규화된 offset m_j/||H_j||_F**로 비교해야 한다. 더 작은 값이 더 강한 제약이다. Raw margin만 비교하면 gradient norm의 차이를 놓친다. H_j=0은 별도 처리하며 cosine만으로 row를 삭제하지 않는다.

권고 우선순위는 (1) native가 새로 손상시킨 entry-safe 요청, (2) 이미 잊힌 active 요청과 안전 여유가 작은 요청, (3) 나머지 보호 방향의 다양성이다. 실제 손상 우선 처리는 MIR형 요소이며 GSS 자체의 성질이라고 부르지 않는다. 같은 우선순위 안에서 recency를 tie-break로 사용하면 최근 편집을 약하게 우대할 수 있다.

더 강한 recency 가중이나 별도 최근-batch quota는 추가 정책이다. 최초 실험에서는 risk-only와 risk+GSS의 차이를 먼저 분리하고, 최근 항목 우대를 더할 경우 같은 예산의 별도 ablation으로 비교한다. Age별 성능을 보고 최근 이득과 오래된 edit 손실을 함께 판단한다.

### 6.3 단일 레이어 gradient factor 활용

각 TF 경로에서 고정 K_j와 현재 activation gradient A_j를 이용하면

\[
G_j=A_jK_j^T,\qquad
\langle G_iQ_t,G_jQ_t\rangle_F
=\operatorname{tr}(A_i^TA_jK_j^TQ_tK_i).
\]

여러 TF 경로는 합한다. Fact마다 dense weight gradient를 CPU에 복사할 필요가 없다. 하지만 A_j는 현재 endpoint마다 달라지며 fixed key라고 backward가 사라지지 않는다.

소수 correction 방향 D_k만 사용한다면 \(v_{jk}=\langle A_j,D_kK_j\rangle_F\)로 작은 선택 feature를 만들 수 있다. 이는 **해당 후보 공간의 다양성**이며 전체 weight-space GSS와 동일하지 않다. 방향 norm/metric이 다르면 whitening 후 비교한다.

## 7. 계산량과 최소 검증 순서

### 7.1 Checkpoint audit을 먼저 수행

1. S2 W/M과 S4 target/eval을 source·batch·request ID로 결속한다. 현재 stat 확인과 payload 내용 검증을 구별한다.
2. 기존 seen-full 평가에서 같은 cohort의 RS/PS/N 전이를 먼저 추출한다. 모델 forward 없이 가능한 부분부터 한다.
3. W0 prefix 1회로 고정 probe와 저장 경계 다음 batch의 native/bare keys를 추출한다. 평균 weighting과 h0+EK parity를 검증한다.
4. 같은 probe의 ν_t, target loading을 포함한 ridge fitting, 구간 Δ의 실제 반응을 비교한다.
5. 소수의 success→failure/유지 대조쌍에서 frozen suffix의 signed margin을 확인한다. Geometry norm만으로 최종 손상의 인과성을 선언하지 않는다.

역사적 checkpoint는 **baseline 기전 진단**에 사용한다. 새 방법의 성능 비교 chain은 기존 사용자 조건대로 W0에서 시작한다. Warm state의 보정 probe를 lifelong 방법의 최종 성능과 혼용하지 않는다.

### 7.2 History 선택의 효용과 비용을 분리

전체 latest-valid ledger → forward 위험 검사 → 제한된 후보의 fresh gradient → 다양성/제약 선택 → correction → endpoint history 검사 순서가 타당하다. Forward 전체 검사는 여전히 history 길이에 비례한다. GSS가 이 비용을 제거한다고 주장하지 않는다.

처음에는 active history budget M=256 fact bundle, fresh-gradient 후보 상한 C=512를 **측정용 예산**으로 둘 수 있다. 최적값이나 이론적 capacity가 아니다. Reference 512 전체 처리와는 별개다. 모든 10k gradient를 계산한 뒤 256개만 고르는 방식은 backward 절약이 아니다.

GSS selection에 C개 gradient가 필요했다면 비용은 M이 아니라 C로 센다. 보유 cached sketch, 새 gradient, refresh, fact별 TF 경로 수를 각각 보고한다. 후보 밖 위험 row와 constraint 불충족을 기록하며, 검사하지 않은 history의 보호를 보장하지 않는다.

첫 비교는 같은 entry와 같은 fresh-gradient 후보 C를 공유하여 **random M / risk-only M / risk+GSS M**의 차이를 본다. 위험 우선순위, 후보 coverage, gradient 예산을 고정하여 GSS가 실제로 제거한 중복과 남긴 functional 제약을 확인한다. 이후 선택된 정책의 cold sequential을 수행한다.

B1에는 history가 없어 GSS 성능 차이가 없어야 한다. B100×3에서는 Past≤200이므로 M=256이면 아직 선택 예산이 차지 않는다. **GSS 압축 효과는 실제 latest-valid Past가256을 넘는 첫 batch부터, 중복/overwrite가 없으면 B4부터** 검증 가능하다. 후보 cap512에 의한 backward 절약은 Past가512를 넘는 별도 구간까지 확인해야 한다. B3 기술 검증만으로 GSS 성공을 주장하지 않는다.

### 7.3 통과 기준

- Current: native와 current lock 반응/실제 rewrite가 tolerance 내 동일.
- Reference: 기존 full512 보호 규칙 유지; dev/N은 observer.
- History: entry-safe 신규 forgetting, accepted-but-failed 회복, 초기 실패를 분리; 미선택 요청도 포함한 coverage 명시.
- Paraphrase: 과거 edit의 PS를 age/cohort별 보고. Selection에는 사용하지 않음.
- GSS 가치: 동일 비용의 risk-only/random보다 history 보호 또는 constraint coverage가 좋아져야 한다.
- 비용: standalone native+reference+history forward/backward+선택+solver+검증 전체. 공유 audit 비용을 임의로 나누어 method가 빨라졌다고 보고하지 않는다.

## 8. 연구적 해석

GSS replay 추가만으로 novelty가 확보되지는 않는다. 더 검토할 가치가 있는 주장은 **single-layer의 고정 feature를 이용하여 현재 write와 동일한 편집 반응을 갖는 update 중에서, 기존 지식과 과거 편집의 기능적 손상이 작은 update를 선택한다**는 것이다.

이를 지지하려면 (a) 실제 history/reference 손상 방향이 존재하고, (b) current lock 뒤에도 해당 손상을 줄일 자유도가 남고, (c) 계산 가능한 선택이 그 자유도를 활용하며, (d) 공식 미사용 PS/N에서도 이득이 있어야 한다. GSS는 (c)의 계산 가능한 history 선택을 돕는다. 누적 교차항은 (a)/(b)의 진단을 돕는다. 어느 쪽도 단독으로 (d)를 보장하지 않는다.

현재의 직접 evidence는 **분석 자산의 존재와 구조**다. Key 구별성 저하→target demand→write 비용→neighbor margin 손실의 연결 및 GSS의 개선 효과는 아직 이 검토에서 측정한 결과가 아니다.
