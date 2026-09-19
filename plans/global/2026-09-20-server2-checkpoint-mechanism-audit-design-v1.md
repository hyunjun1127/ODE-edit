# Server2: 회수한 L4-only checkpoint의 기전 분석 실험 v1

작성: 2026-09-20 KST. **설계와 read-only preflight 완료용 문서이며, 분석 runner 구현·GPU 실행·새 편집 실험 결과가 아니다.** 실행 서버는 Server2다. 기존 checkpoint를 읽고 필요한 z/평가 자료만 Server4에서 staging한다. 새로운 local-z 최적화, 10k replay, 방법별 sequential 비교, GSS 실험은 이번 범위에 포함하지 않는다.

관련 근거: [이전 자산·수학 검토](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-geometry-gss-history-review/review-ko.md), [직접 inventory](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-geometry-gss-history-review/server2-checkpoint-inventory.json), [Server2 preflight](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-checkpoint-mechanism-design/server2-preflight.json), [평가 source 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-checkpoint-mechanism-design/functional-source-audit.json).

실행 명세: [contract](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-server2-checkpoint-mechanism-audit-contract-v1.json), [분석 단위와 의존성](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-server2-checkpoint-mechanism-audit-cells-v1.csv). CSV의31개 row는13개 history operator와7개 target batch 등을 구분한 분석 단위이며,31개 편집 arm이 아니다. 아직 호출 가능한 새 runner CLI는 없다.

## 1. 이번 실험이 답해야 하는 질문

|ID|질문|핵심 증거|그 결과만으로 주장하지 않을 것|
|---|---|---|---|
|H1|L4-only에서 rewrite/paraphrase와 neighborhood의 시간적 양상이 다른가?|At-write와 같은 cohort의 RS/PS/NS 전이, strict 지표|RS가 높다는 이유만으로 전체 모델의 preservation capacity가 충분함|
|H2|같은 key에 대한 누적 history의 제약이 커지는가?|고정 probe의 history operator response와 실제 다음 batch의 별도 분석|다른 batch 구성의 차이를 key drift로 해석|
|H3|Geometry와 target demand의 결합이 실제 write 비용을 설명하는가?|Raw adjusted-key factor의 singular gain × target loading; actual write 재현|작은 key singular value가 항상 native write를 증폭|
|H4|그 update가 영향을 주는 입력에서 실제 보존 margin이 감소하는가?|동일 입력의 ΔK, signed margin 전이, 소수 공통 weight 개입|Activation energy가 곧 locality 손상|
|H5|손상의 일부를 current edit 반응을 보존하면서 바꿀 여지가 있는가?|후속 단계에서 검사할 free-space sensitivity의 필요성 도출|이번 audit만으로 개선 방법이나 GSS 효용이 입증됨|

최소 완료 기준은 **기존 평가 종단 감사 + B1 실제 writer 재현 + 고정 probe의 history 효과 + actual interval response와 margin 연결**이다. H5는 결과에 따른 다음 연구 질문이며 이번 주 실험에 optimizer를 추가하지 않는다.

## 2. 자산과 실행 환경을 먼저 결속한다

### 2.1 상태와 원본

- Method: AlphaEdit_L4_ONLY / 표시명 AlphaEdit_BLUE_L4_ONLY.
- Weight: `model.layers.4.mlp.down_proj.weight`, FP32[4096,14336].
- W checkpoint: B001, B005, B010, B020, B030, B040, B050, B060, B070, B080, B090, B100.
- Native history: `cache_c` [1,14336,14336], **post-write Gram**. Static Wikipedia mom2 또는 forward weight와 혼동하지 않는다.
- Model revision: `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
- Native source head: `1075540b45c29269e690ac63aae44758d8d63174`; BLUE source head: `311b076a92e4ed0f14f5c8b4909732da781bc5f7`.
- Sealed sample root: `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`.
- W0 L4 expected FP32 tensor hash: `9421d3f6dfae10b4663c07a41696f5c47e299e1f15d1728e162482f0245eb851`.

S2 checkpoint root:

```text
/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/payload/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/
```

S4 companion source root:

```text
/data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/
```

S2 output root, 새 실행마다 아래에 새 attempt 디렉터리를 만든다:

```text
/mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/
```

### 2.2 이번 preflight로 확인한 실행상 주의점

Pinned snapshot·safetensor shards·tokenizer·index가 S2에 있다. W0 L4는 shard1의 BF16 tensor이며 **원 실행처럼 FP32로 cast한 뒤 hash를 비교**한다. BF16 hash와 저장 FP32 weight hash를 직접 비교하지 않는다.

저장 projector는 FP32[5,14336,14336]이며 source layers[4,5,6,7,8]의 **slot0가 L4**다. 저장 P4를 사용하므로 full14336 eigen/SVD를 새로 계산할 필요가 없다. Payload hash와 원 binding의 selected P4 fingerprint는 실제 load 단계에서 한 번 검증한다.

S2 기본 환경의 Transformers4.57.1은 원 실행4.44.2와 다르다. 원 runtime의 Torch/tokenizers 등 dependency를 결속한 별도 환경을 사용한다. 기존 환경을 덮어쓰지 않는다. 원 수치 설정은 FP32/eager, TF32 matmul=false, **TF32 cuDNN=true**다. 최신 EN 규약을 자동 적용하지 않는다.

Writer tokenizer와 evaluator tokenizer의 BOS 설정도 다르다. 원 evaluator는 microbatch16에서 수동 left-padding하고 position_ids를 명시하지 않는다. tokenizer.padding_side 문자열만 보고 구현을 바꾸지 않는다.

이관 manifest의 job39283_3과 source closure runtime의 job40426에는 표기 차이가 있다. Job명만으로 lineage 일치/불일치를 판정하지 않는다. Model/source/sample/context/P/W/M hash와 batch entry/endpoint 결속을 우선하고, job 불일치 자체도 receipt에 남긴다. 이 필수 identity 결속이 실패하면 writer 재현 단계는 진행하지 않는다.

### 2.3 S2에 staging할 최소 companion

|자료|범위|용도|
|---|---|---|
|seen-full.json|저장된12경계|같은 cohort의 종단 평가|
|current.json|B1–B100 전체|각 요청의 실제 최초 post-write anchor|
|commit/entry/native-observation/contexts|B1–B100 소형 기록|순서·state·actual per-batch norm·context 결속|
|native-targets.pt|B1,B2,B6,B11,B21,B51,B91|선택700요청의 저장 z|
|sample lock/dataset/source/runtime/config|봉인된 원본|Identity·tokenization·overwrite 검증|

현재 current100개와 seen-full12개는 실제 존재한다. Copy는 실행 준비 단계에서 S2의 `inputs/`로 하고 source→destination SHA를 확인한다. S2에 있는 W/M12개를 다시 전송하지 않는다. 원본 수정·원 run 경로 재사용은 금지한다.

직접 stat한 크기는 seen-full370,456,132bytes, current66,737,255bytes, target7개11,733,953bytes다. 합계 약448.9MB이며 sample lock 포함 약457.7MB다. Source/dependencies와 추가 소형 receipt의 크기는 별도 계측한다.

Hash/finite 검사는 최초 입력 검증 때 한 번 수행하고 파일 크기·mtime·hash manifest로 재사용한다. 함수 호출마다 큰 teacher/checkpoint를 재해싱하지 않는다. 상태 tensor 검증과 full matrix eigendecomposition은 별개다.

## 3. 단계 A — 모델 없이 실제 forgetting부터 복원

### 3.1 평가 row의 결속과 의미

Join key는 `(arm, metric_tag, identity)`다. Identity는 원 evaluator의 `[case_id,prompt_index,prompt,target_new,target_true]` digest이며 category가 들어 있지 않으므로 metric tag를 반드시 붙인다. 관측 key에는 checkpoint batch도 포함한다.

|지표|저장 raw margin|양수가 성공인 safety margin|대상 strict|
|---|---|---|---|
|RS/PS|true_nll−new_nll|raw margin|new_strict|
|NS|true_nll−new_nll|−raw margin|true_strict|

Success는 strict inequality다. Tie는 실패다. Target 평균 NLL의 두 후보 비교이며 전체 vocabulary margin이나 free-generation 성공률이 아니다.

`current.weight_state/cache_sha256`, `seen-full.state`, `commit.endpoint`, checkpoint metadata를 결속한다. 같은 endpoint의 current/seen-full 중복 rows는 scalar 일치 확인 후 한 관측으로 센다. 이 archive의 B1은 R100/P190/N867이다. 최근 EN의100/194/865를 기대값으로 사용하지 않는다.

### 3.2 비교 cohort

1. **At-write→현재:** request의 도착 batch current.json을 anchor로 한다. 처음 실패, 이후 forgetting, 처음 실패 후 회복을 분리한다.
2. **고정 초기 cohort:** 첫100요청은 모든12경계, 첫1000요청은 B10 이후 경계에서 평가한다.
3. **인접 checkpoint 구간:** a에서 이미 등장한 동일 rows만 a→b에 비교한다. 새 요청을 분모에 넣지 않는다.
4. **Age별 cohort:** arrival batch를 고정한 채 관측 edit age에 따라 비교한다. Sparse 경계의 최초 실패 시점은 interval-censored로 표시한다.

각각 1→1,1→0,0→1,0→0, loss/gain/net, margin 변화, 대상 strict 변화, request별 RS∧PS0∧PS1 joint를 출력한다. Loss 분모는 시작 시점 성공 수, recovery 분모는 시작 시점 실패 수다. 모든 비율에 분자/분모를 남긴다.

CI는 request cluster bootstrap2000회, seed20260920으로 계산한다. 같은 요청의 P2/N10/전체 시점을 함께 재표집한다. Subject-relation cluster를 사용한 sensitivity 결과도 병기한다. 단일 edit 순서에 대한 조건부 불확실성이며 여러 stream에 일반화한 CI는 아니다.

### 3.3 실제 overwrite 교란을 분리

원 sample은 **모든 ordered canonical case를 유지하는 정책**이며, 기록된 `no replacement`는 앞선 case를 삭제·대체하지 않는다는 뜻이다. 10,000요청/9,783subject-relation groups, 반복125groups, 앞선version217개, target이 달라지는120groups, 동일batch 내부 상충 group-batch5개가 확인됐다.

- 원래 all-case benchmark는 그대로 재현한다.
- 같은 구간에서 target 변경 충돌이 없는 cohort를 별도로 분석한다.
- Active-version 결과는 별도 진단으로 명명하며 원 benchmark를 대체하지 않는다.
- 같은 batch의 서로 다른 target은 `BATCH_INTERNAL_CONFLICT`; 배열 마지막이 실제 최신 정답이었다고 단정하지 않는다.
- RS가 original true target보다 new target을 선호하는지만 검사하므로 서로 다른 new target 두 개가 모두 RS 성공일 수 있다. 이를 두 상충 사실의 완전한 동시 저장으로 해석하지 않는다.

Entry.json은 entry 평가가 아니다. 별도로 W0 평가를 수행하지 않은 row에서 W0→post-write 손상을 계산했다고 하지 않는다. 기존 파일만으로 얻는 NS anchor는 **first-write-N**이다.

## 4. 단계 B — 12개 actual W/M의 구간 변화

W0와12개 checkpoint에서 W/M hash·finite·shape를 확인하고 W0 L4를 FP32로 결속한다. W 차이는 FP64로 cast한 뒤 빼며 reduction도 FP64로 수행한다.

\[
E_t=W_t-W_0,\qquad D_{a:b}=W_b-W_a.
\]

출력: ||E_t||/||W0||, ||D||, 누적/구간 Frobenius angle, M trace·diagonal 통계·request당 trace, 구간 ΔM의 trace·대칭 오차와 임의 방향 quadratic checks. Random checks는 PSD/rank 인증이 아니다. M0=0이며 저장 M_t는 post-write다.

과거 key에 대한 구간 작용은

\[
J^{stored}_{a:b}=\operatorname{tr}(D_{a:b}M_aD_{a:b}^T)
\]

로 측정한다. 개별 K가 없으므로 처음에는 Hutchinson estimator를 사용한다:

\[
z_l\in\{-1,+1\}^{4096},\quad q_l=D^Tz_l,\quad
\widehat J=r^{-1}\sum_lq_l^TM_aq_l.
\]

11개 인접 nonzero-history 구간에 같은 seed/vector를 쓴다. **최종 표본 수는 r=256으로 고정**하고64/128의 결과는 진행 진단으로만 사용한다. 1.96×MC SE/|mean|≤.1을 정밀도 지표로 보고하되 이 기준으로 조기 종료하지 않는다. Near-zero/음수에서는 절대 SE를 보고한다. 근사적인 MC 정밀도 지표이며 statistical preservation threshold나 엄밀한 유한표본 confidence 보장이 아니다.

J/n_a, J/tr(M_a), `d*J/(||D||²*tr(M_a))`를 함께 보고한다. 마지막 값은 update 크기를 제거한 history 방향 정렬 지표이며 capacity 소진율이 아니다. J/(b−a)를 batch당 실제 손상으로 부르지 않는다.

FP32 history 누적 때문에 Mstored=ΣKKᵀ+E_round다. Stored quadratic form과 실제 key 반응 에너지의 차이는 tr(D E_round Dᵀ)다. B1의 K1K1ᵀ 재구성으로 초기 오차를 측정하되 B100 오차 인증으로 확대하지 않는다.

Commit의 per-batch actual squared norm을 결합해

\[
\|D_{a:b}\|_F^2-\sum_{s=a+1}^{b}\|\Delta_s\|_F^2
\]

도 출력한다. 구간 내부 update들의 총 교차항이며 개별 pair별 상쇄나 single-batch 효과를 복원한 값은 아니다.

## 5. 단계 C — W0 prefix 한 번과 B1 재현

### 5.1 고정 input panel

|Panel|선정|크기|용도|
|---|---|---:|---|
|Native batches|B1,B2,B6,B11,B21,B51,B91|700requests|저장 entry 다음 batch의 원래 z/demand|
|Fixed geometry probe|동일 dataset에서 sealed10k case ID를 제외한 뒤 hash(seed,case_id) 정렬|512requests|모든 history에 동일 key를 대입|
|기능 진단 panel|단계G의 규칙|최대64neighbor rows|Full-token activation과 margin 연결|

Fixed geometry probe512는 **방법의 train reference512를 교체하는 자료가 아니다.** Geometry 관측용이다. Case ID disjoint를 확인하고 subject/relation overlap도 별도 기록한다. Semantic 독립성을 보장한다고 하지 않는다. 부족하면 몰래 stream에서 보충하지 않는다.

기존 reference512/G256은 변경하지 않는다. 이번 core audit에서는 이를 새로 생성하거나 EN guide를 다시 실행할 필요가 없다. 해당 reference에 결과를 연결하는 후속 분석에서는 기존512 전체를 사용한다.

### 5.2 Capture와 캐시

처음에는 B1/B2와 probe의 hash-first32로 기술 검사를 수행한다. 통과하면 나머지를 같은 model load에서 추출한다. L4까지 실행하고 필요한 key/block output을 capture한 뒤 suffix를 중단한다. 작은 panel에서 full forward와 parity를 확인한 후 이 shortcut을 사용한다.

저장할 것:

- Native writer mean key K: group[1,5] 내부 평균 후 group 간 평균, 즉 bare group .5와 나머지 각 .1.
- Bare prompt subject key k_bare와 W0 block output h0: residual용.
- Token IDs, mask/position, tokenizer/context/subject lookup hash, dtype, capture source.
- Functional probe는 필요한 **모든 valid token**의 K와 고정 prefix/residual 정보. Mean key로 대체하지 않음.

목표 평균/bare keys와 h0는 약0.13GB 수준의 payload다. 모든 raw context/full-token hidden을 영구 저장할 필요는 없다. 정확 bytes는 생성 후 기록한다. Native z는 한 번도 다시 최적화하지 않는다.

### 5.3 재현 gates

1. W0/P4/source/sample/tokenizer/context identity 및 입력 순서 결속.
2. B1의 actual M1과 새 K1K1ᵀ의 차이 측정.
3. Bare h_entry=h0+(W_entry−W0)k_bare와 physical entry forward 비교.
4. B1의 원래 dense-RHS native solve를 같은 FP32 연산 순서로 1회 실행하고 actual W1−W0 및 current-key response 비교.
5. W0에서 first100의 전체 R100/P200/N1000을 한 번 평가해 별도 W0 anchor를 만든다. 이어 archive B1의 같은 rows를 원 평가 배치 규칙으로 재현한다. B1의 R100/P190/N867 및 row별 scalar/bit 차이를 기록한다. 이렇게 새로 결속한 first100에만 W0→B1의 직접 변화라고 부른다.

초기 engineering tolerance: key/block rtol1e−5, atol1e−6; normalized solve residual≤1e−5; update/response relative discrepancy≤1e−3; pairwise mean-NLL/margin absolute discrepancy≤1e−4 nats. Bit 차이는 원/재현 margin이 모두 ±1e−4 band 안인 경우만 `NUMERIC_BOUNDARY`로 분리하고 robust bit mismatch는 gate 실패다. 수치 기준은 과학적 effect threshold가 아니다.

집계 규약도 고정한다. Key/block은 모든 원소가 `|new−ref|≤atol+rtol*|ref|`를 만족해야 한다. Solve는 **각 RHS column의 ||Ax−b||₂/||b||₂의 최대값**을 사용하고 backward-error 분모와 혼용하지 않는다. Update는 actual Δ를 분모로 한 전체 Frobenius 비율, response는 **request별 ||(Δrecon−Δactual)k||₂/||Δactual k||₂의 최대값**이다. 분모≤1e−12이면 상대값 대신 절대norm≤1e−7을 요구하고 zero-reference flag를 남긴다. NLL/margin은 row별 절대오차의 최대값이다. Norm/reduction 및 residual 검산 precision과 비용을 기록한다.

동일 경로를 반복한 spread가 위 기준의10%를 넘으면 numerical contract를 먼저 해결한다. Analysis 성과를 보고 tolerance를 넓히지 않는다. Failure 시 원인·최대 오차·분포를 보고하며 이미 검증된 저장평가/actual W 분석만 진행한다. Hash 차이가 있으면 approximate parity를 exact identity라고 부르지 않는다.

기존 `solve_isolated_alphaedit_factor*` helper는 M=0 전용이므로 lifelong M≠0에 그대로 재사용하지 않는다. 파일에 함수가 있다는 이유로 이번 runner가 구현됐다고 간주하지 않는다.

## 6. 단계 D/E — 고정 history operator와 실제 다음 batch

### 6.1 Raw operator: full projector decomposition 불필요

저장 P=P_raw와 λ=원 hparams.L2를 사용한다:

\[
H_t=\lambda I+PM_t,\quad Y_t=H_t^{-1}PK,\quad S_t=K^TY_t,\quad B_t=Y_t(I+S_t)^{-1}.
\]

정확연산에서 B_t는 `[λI+P(M_t+KKᵀ)]^{-1}PK`와 같다. H_t는 일반적으로 비대칭이므로 LU/일반 solve를 사용한다. CG/Cholesky를 그대로 적용하지 않는다. Inverse를 명시적으로 만들지 않는다.

같은 history에서 RHS bank를 합쳐 한 factorization을 재사용한다. Bank는 계산 공유용이며 **512probe+700requests를 하나의 native editing batch로 넣지 않는다.** 각 native100개마다 자기 S와 B를 만들고, probe는 개별 key별 self-gain을 산출한다.

Pilot histories는 M0,M1,M10,M100, 통과 후 M5,M20,M30,M40,M50,M60,M70,M80,M90로 확장한다. 따라서 최종13상태(W0+12checkpoint)에 같은 probe를 비교한다. W0에서는 M0=0을 이용한다.

Raw probe score s_t(k)=kᵀH_t^{-1}Pk, 정규화 ν=λs_t/||Pk||²를 보고한다. P가 ideal orthogonal projector이면 0<ν≤1이고 동일 probe에 history가 추가될 때 비증가한다. Raw P에서는 projector action·solve residual·skew·음수/out-of-range를 함께 보고하며 범위 밖 값을 clipping하여 정리와 맞추지 않는다. PK=0은 별도 분류한다.

실제 fitting transfer는 `Δfactor K=R(I+S)^{-T}S^T`다. S를 대칭으로 가정하여 식을 바꾸지 않는다. Symmetrized S eigenmode는 numerical ideal approximation인 선택 분석이며 raw operator 결과를 대체하지 않는다.

### 6.2 저장 target으로 native residual 재구성

|Target batch|Entry W/M|자산 성격|
|---|---|---|
|B1|W0/M0|Actual W1 endpoint가 있어 완전 tensor 대조 가능|
|B2|B1|원 z와 entry를 결합한 reconstructed native write|
|B6|B5|동일|
|B11|B10|동일|
|B21|B20|동일|
|B51|B50|동일|
|B91|B90|동일; late dense/factor parity anchor|

\[
R_i=z_i-\{h_i^{block}(W_0)+(W_{entry}-W_0)k_i^{bare}\}.
\]

Mean-context K를 bare residual에 사용하지 않는다. Z identity/순서/토큰 normalization을 결속한다. B1 외에는 원 next-weight tensor가 없으므로 **reconstructed write**로 명명한다. Commit norm과 원 current 평가를 대조하더라도 실제 endpoint tensor를 확보한 것은 아니다.

B1과 B91에서 원 dense-RHS `solve(λI+P(KKᵀ+M), (PK)Rᵀ)`와 factor 경로를 비교한다. Native FP32 rounding은 algebraic factorization과 다를 수 있다. Late gate가 실패하면 late factor mode attribution을 중단하거나 원 direct solve 결과까지만 보고한다.

### 6.3 실제 write 부담의 분해

Native adjusted-key B는14336×100이므로 thin SVD만 필요하다:

\[
B=L\Sigma V^T,\quad\Delta_{factor}=RV\Sigma L^T.
\]

\[
\boxed{\|\Delta_{factor}\|_F^2=\sum_j\sigma_j(B)^2\|Rv_j\|_2^2.}
\]

이 분해는 L의 직교성 때문에 mode 간 Frobenius 교차항이0이다. **Actual writer gain σ(B), target loading ||Rv||², 그 곱**을 각각 출력한다. Native dense/actual weight와의 rounding residual은 별도로 붙인다. 일반 G-eigenmode의 에너지를 plain weight norm으로 잘못 합산하지 않는다.

정확히는 E=Δactual−Δfactor에 대한 `2〈Δfactor,E〉+||E||²`도 actual energy에 포함된다. 입력 재구성 검증 전에는 E를 단순 rounding이라 하지 않고 **factor/native 재현 잔차**로 기록한다. Weight mode가 직교해도 neighbor에서의 Δ_jK는 직교하지 않으므로 preservation 비용에는 교차항이 남는다. BᵀB eigen 대신 B의 직접 thin SVD를 사용한다.

Raw key K/PK의 singular spectrum, S의 fitting transfer, B의 gain은 서로 다른 양이다. 작은 key mode와 actual gain의 관계를 측정하며 “작은 singular value=증폭”을 전제하지 않는다. 근접 singular values에서는 개별 vector가 불안정하므로 묶인 subspace의 총 loading/energy도 보고한다.

### 6.4 대수적 대조군

- **같은 K,R, 다른 M:** B1/B2의 K,R을 고정하고 pilot histories0,1,10,100에서 writer를 계산한다. 바뀐 상태의 target를 재최적화한 실제 editing 성과가 아니다.
- **같은 K/operator, R 열 permutation:** seed20260920,20개 permutation으로 target loading 정렬 효과를 본다. R Frobenius norm은 보존되지만 의미적으로 올바른 edit이 아니다.
- **원래 다음 batch:** 위7batch의 실제 entry와 원 z 분석. 서로 다른 request composition이 섞이므로 고정-K 대조군과 분리한다.

출력은 realized target error ||ΔK−R||/||R||, writer norm/relative W0 norm, gain/loading/energy, solver residual, native/source parity다. Counterfactual로 개선 방법의 RS/PS/NS를 주장하지 않는다.

## 7. 단계 F/G — 실제 구간 ΔK와 margin 연결

### 7.1 Exact activation bookkeeping

고정 input x에 대해 E=W_a−W0, D=W_b−W_a를 사용한다:

\[
B_x=EK_x,\quad V_x=DK_x,\quad
\|B_x+V_x\|_F^2-\|B_x\|_F^2=2\langle B_x,V_x\rangle_F+\|V_x\|_F^2.
\]

고정 geometry probe에서는 mean-key 반응, 실제 평가 TF 경로에서는 all-token 반응을 각각 계산한다. 두 값을 같은 지표로 합치지 않는다. Target new/true의 prefix가 다르면 별도 경로로 다룬다. 교차항은 진단량이며 최적화 목표가 아니다.

### 7.2 Functional panel과 실행량

Primary longitudinal population은 기존 JSON의 first100 cohort NS1000 및 R/P 전체다. 새 모델 probe는 원자료를 대체하지 않는다.

우선 구간 B1→B10에서 NS lost16 + retained16을 선택한다. Lost는 hash-seed 순서로 선택하고 retained는 같은 출발 cohort·entry margin·target 길이·가능한 relation으로 matching한다. 부족하면 실제 수를 기록하며 다른 구간으로 몰래 보충하지 않는다. 후기 확인은 B50→B100에서 같은32row 규칙을 한 번 반복한다. 동일군/구간의 target conflict는 제외하고 exclusion을 기록한다.

**두 구간 모두 first100 cohort의 동일 NS1000 rows를 후보 모집단으로 한다.** B50까지 도착한 전체5000요청으로 후기 모집단을 넓히지 않는다. Matching은 target 길이 일치, relation 일치, entry safety-margin의 절대 차이, seed hash의 사전 고정된 사전식 순서로 retained를 중복 없이 고른다. 실제 match 차이를 출력한다.

이는 outcome으로 고른 **사후 기전 panel**이다. 대표 성능·방법의 threshold·효과 크기 검정에 사용하지 않는다. PS/N을 observer로만 쓰는 향후 방법 설계와 구분한다. RS/PS는 해당 parent requests의 동반 변화로 보고한다.

각 구간에서 W(s)=W_a+sD, s=0,.5,1의 실제 margin과 strict를 측정한다. 같은 D가 모든 입력에 적용된다. W(s)는 진단용 상태이며 기존 lifelong checkpoint로 commit하지 않는다.

최대64개NS row의 entry에서 margin activation gradient A를 한 번씩 얻어

\[
\widehat{\Delta m}=\sum_{p\in\{true,new\}}\langle A_p,DK_p\rangle_F
\]

와 finite Δm을 비교한다. 부호 정확도, 비선형 remainder, ||DK||만의 설명력과 signed sensitivity를 결합한 설명력을 비교한다. Gradient는 필요한 모든 input positions를 포함한다. 세 interpolation 점과 entry gradient는 선분 전체의 증명이 아니다.

공통 weight component를 제거하는 개입은 **선택 후속 diagnostic**으로 둔다. 사례마다 다른 token patch를 넣고 구현 가능한 weight update라고 부르지 않는다. 제거 mode와 norm/rank-matched 대조군, 해당 current R/P disruption을 함께 측정해야 한다. 기본 실행에는 mode 삭제 최적화를 넣지 않는다.

## 8. 산출물과 중단 기준

|파일/그림|핵심 내용|
|---|---|
|source_bindings.csv / numerical-parity.json|원본↔S2 binding, deps, W/P/M/key/writer/evaluator gate|
|request_registry.csv|도착 batch, raw/request hash, group/version/conflict|
|functional_long.parquet|current+checkpoint 평가 row, strict, signed margin, state|
|paired_transitions.csv / at_write_outcomes.csv|같은 cohort의 loss/gain/net, acquisition·forgetting 분리|
|checkpoint_geometry.csv|actual W/M norm, trace, stored-action sketch와 MC SE|
|fixed_probe_history.csv|probe별 모든13history의 raw score/gain, numerical flags|
|native_write_modes.csv / reconstruction.csv|선택7batch의 gain×demand 분해, actual/reconstructed 구분|
|counterfactuals.csv|고정K/R history 대조와20permutation 대조|
|mechanism_panel.csv / activation_margin.csv|선정 이유, ΔK/교차항, 실제·예측 margin 변화|
|timing.json / report-ko.md|각 단계 F/B/solve/I/O/hash 시간, peak RAM/VRAM, 가설별 supported/mixed/not-supported/unresolved|

필수 그림: (1)고정cohort R/P/N과 acquisition/forgetting, (2)fixed-probe history 곡선, (3)B gain×target demand→write energy, (4)actual interval response/교차항과 signed margin 변화. 과학적 figure는 CSV로 재생성 가능한 PDF/PNG로 저장한다.

Identity 실패는 해당 source 결속 중단, numerical gate 실패는 그 이후 model/factor claim 중단이다. Failure 때문에 이미 유효한 CPU 종단 결과를 버리지 않는다. 신호가 없다는 이유로 threshold·panel·horizon을 사후 변경하지 않는다. 단순 상관은 인과 증명으로 승격하지 않는다.

입력 receipt는 평가/order, W/M, W0, model environment, P4, source, 각 batch target를 **독립 component status**로 기록한다. Late z 한 개가 없다고 저장평가나 W/M 분석을 막지 않는다. CSV dependency는 해당 component의 성공을 요구하고, 최종 report는 모든 작업의 **성공**이 아니라 PASS/FAILED/BLOCKED/SKIPPED 상태를 수집한 뒤 생성한다. 미실행 결과를 PASS로 만들지 않는다.

## 9. 계산 예산과 실행 순서

기본 자원은 S2 단일48GB GPU, CPU8threads, CPU RAM64GB 안에서 chunking한다. 실제 scheduler/사용가능 GPU는 실행 시 확인하며 이번 nvidia-smi snapshot을 예약으로 간주하지 않는다. CPU 단계는 GPU 없이 시작한다.

순서: **입력 결속 → 전체 저장평가 분석 → actual W/M sketch → 작은 prefix/B1 재현 → 700+512 key bank → 4history pilot → 나머지9history와7native batch → 최대64row suffix 기전 검사**.

모델 load는 가능한 한 한 번, keys는 한 번, history별 LU는 한 번, z optimization은0회다. RHS chunk는 메모리에 맞추되 원 native/evaluator token packing을 바꾸지 않는다. LU cache는 GPU 동시 상주 대신 CPU/disk 또는 즉시 소비를 사용하고 cache bytes/I/O를 계측한다.

먼저 pilot에서 prefix100요청, LU1개, RHS block, suffix32row의 시간을 측정하여 확장 비용을 산출한다. 측정 전 wall-clock 속도 향상을 약속하지 않는다. 선택 budget과 근사 sketch가 줄인 비용 및 남아 있는 suffix/backward 비용을 따로 보고한다.

이번 retrospective audit에서 warm checkpoint를 사용하는 것은 baseline 기전 분석이다. 이후 새 방법의 비교 chain은 사용자 조건대로 W0에서 시작한다. **이번에는 GSS memory 구성이나 EN correction을 실행하지 않는다.**
