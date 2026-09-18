# BG-TW v2：W0부터 시작하는 단계적 방법 설계

> 2026-09-15 후속 재검토: 아래는 이전 검토·설계의 기록이다. N4 기반 한도와 fixed-budget BG-1을 현재 첫 방법으로 사용하지 않는다. 현행 권고는 [파이프라인 재검토](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pipeline-reset-review-ko.md)와 [사전 보존 한도 없는 EP-TW-1 v3](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-edit-quality-preserving-tw-design-v3.md)를 따른다. 이전 실험 수치·수식 검토는 역사적 근거로 보존한다. 이 표지는 원격 V1 dispatch 변경을 뜻하지 않는다.

작성: 2026-09-15. 상태: 설계 확정안. 방법 구현·GPU 실행·원격 제출·성능 검증은 아직 수행하지 않았다.

기준 문서: [사용자 PDF](/mnt/raid5/janghj/ODE-edit/plans/global/ICLR_2027.pdf), [상세 검토](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pdf-method-review-ko.md).

## 1. 이번에 정하는 것

1. 첫 방법은 **BG-1: 단일 L4 native target, 실제 post-write original-KL feedback 한 번, 최종 write 한 번**이다.
2. 다음 확장은 accepted-old retention, 그다음 fresh target을 포함한 두 stage다. Response medoid·vulnerability mining·CBF-flow는 뒤에 둔다.
3. 모든 method 성능 비교는 **초기 pre-edit W0에서 B100×10, 요청 1,000개**다. 한 batch 결과로 정책을 선별하지 않는다.
4. 주 baseline은 AlphaEdit / MEMIT / AlphaEdit-BLUE / MEMIT-BLUE / AlphaEdit-L4_only다. REFIT4를 W0부터 실행한 가까운 대조도 둔다.
5. 본 lifelong 결과는 정책을 lock한 뒤 W0부터 B100×100의 10,000개로 측정한다.
6. ODE는 첫 결과의 효능 설명으로 미리 확정하지 않는다. 최소 방법의 가치가 있으면 계산과 old retention을 보완하고, ODE 고유 주장에 필요한 비교를 추가한다.

이는 사용자 v1의 Middle SEQ1000 주 실험을 대체한다. 기존 W50 suffix는 동기 자료로 보존한다.

## 2. 단계별 방법

| 버전 | 실제 target/write schedule | 추가 gradient / B100 | 보호 자료 | 답할 질문 |
| --- | --- | ---: | --- | --- |
| BG-1 | native full 1회, nominal η=1 | 1 | 고정 generic64 | 실제 write의 출력 보존 피드백이 단일층 endpoint를 개선하는가 |
| BG-1R | BG-1과 동일 | 1, objective에 old term 추가 | generic64 + selected old64 | 동일 write 수로 과거 accepted edit 손실을 줄이는가 |
| BG-2R | fresh full [.75,1] 두 stage | 최대2 | BG-1R과 동일 | fresh target 재계산이 추가 비용을 정당화하는가 |
| EP-2G-R | fixed-parent endpoint 최적화, gradient2, 최종 write1 | 2 | BG-2R과 동일 | 두 stage의 효과가 단순 추가 endpoint 최적화로 설명되는가 |
| CBF-flow | 별도 fixed-horizon velocity controller | 수치 step 수에 따라 증가 | 먼저 generic64 | 정의한 연속 field·integration이 도움이 되는가 |
| Core 확장 | 선택한 방법 유지 | 동일 quota부터 비교 | 512 bank, 선택/위험/refresh 모듈 | 같은 품질에서 전체 비용을 줄이는가 |

BG-1의 old 손실이 늘면 BG-1R이 직접적인 다음 수정이다. Generic 개선과 함께 under-edit가 드러나면 BG-2 계열의 fresh refinement를 살펴본다. 작은 NS 차이나 1.5× 비용 목표 하나만으로 후속 연구를 일괄 차단하지 않는다.

기전 비교군도 각각 1,000개를 끝까지 처리한다. I2/I4·NM4·E01의 추가 완료를 기다리지 않는다.

## 3. 공통 시작 상태와 baseline 정체성

### 3.1 W0

- 같은 original model revision, tokenizer, prompt normalization, precision과 evaluator를 사용한다.
- 기존 확인 revision은 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2인 Meta-Llama-3-8B-Instruct다. 실행 시 실제 file/config hash를 다시 결속한다.
- 추가 편집 이력이 없는 model weights에서 시작한다. Accepted-label ledger는 빈 집합이다.
- 각 방법의 edit-history 초기화는 native 규약을 따른다. 과거 edit Gram은 비어 있으며 original covariance/projector 등 사전 통계와 혼동하지 않는다.
- Base-derived P·covariance·tokenized contexts는 source와 생성 경로가 맞으면 재사용할 수 있다. W50 weight·M50·warm-entry acceptance ceiling은 주 method chain에 반입하지 않는다.
- 각 정책은 자기 W/history로 다음 batch의 target을 계산한다. 다른 정책의 미래 target·update를 공유하지 않는다.

### 3.2 주 baseline

아래는 보존된 configured-policy 비교와 actual capsule에서 확인한 이름 대응이다. “같은 설정으로 공정하게 만든다”는 이유로 baseline 고유 layer·regularizer를 바꾸지 않는다.

| 표시 이름 | 기존 source 이름 | 실제 layer | BLUE | 핵심 정책 |
| --- | --- | --- | --- | --- |
| AlphaEdit | BASE_ALPHAEDIT | [4,5,6,7,8] | false | entry L8 target, residual divisor 5/4/3/2/1, L2=10 |
| MEMIT | BASE_MEMIT | [4,5,6,7,8] | false | native covariance solve, entry L8 target, divisor 5/4/3/2/1 |
| AlphaEdit-BLUE | AlphaEdit_ORIGINAL | [4,8] | true | 각 layer의 현재 모델에서 fresh local target, divisor1, L2=1 |
| MEMIT-BLUE | MEMIT_ORIGINAL | [4,8] | true | 각 layer fresh local target, divisor1 |
| AlphaEdit-L4_only | AlphaEdit_L4_ONLY / N4 | [4] | true | singleton native L4, divisor1, L2=1 |
| REFIT4 | W0에서 새 실행 | [4] | true | fresh target 두 번, write [.75,1], inner history 고정 |

확인한 target 설정은 v_lr=.1, v_weight_decay=.5, v_num_grad_steps=25, v_loss_layer=31, clamp=.75다. 25는 loop loss 평가 상한이며 Adam update 상한은 24다. Warm 결과에서의 실제 최대 update 사용을 W0에서도 보장한다고 쓰지 않는다.

MEMIT의 native solve FP64와 AlphaEdit native FP32를 보존한다. 모든 모델 weight가 FP32라는 사실과 solve dtype은 다르다. 기존 주 seed는 20260907이다. 새로운 order seed와 numerical seed는 각각 기록한다.

다층 baseline과 BG-1의 차이는 layer·target·추가 control 정보·비용까지 포함한 정책 비교다. Barrier 효과의 직접적인 대조는 아래 same-data ablation이다.

## 4. 최소 방법 BG-1

### 4.1 Controller 데이터

- Reference 정의는 [C4·Pile와 CL reference 재조사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-reference-set-survey-ko.md)와 [C4-WebRef-v2 생성 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-reference-data-contract.json)을 따른다. Source뿐 아니라 문서 token·고정 prefix·scored position·W0 분포·manifest를 묶어 reference를 정의한다.
- 첫 source는 `allenai/c4`의 cleaned English `en`, revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`다. GPTQ·SparseGPT·Wanda의 반복 calibration 사용을 선택 근거로 삼되 CL의 공인 공통 set이라고 부르지 않는다. Pile는 후속 corpus 대조다. 기존 Wikipedia mom2/projector는 변경하지 않는다.
- 고정 텍스트768 = train의 control-development bank512 + 별도 validation의 Report256. Bank512 = S64 + Dev128 + Reserve320이다. 각 고정 첫 shard 전체를 읽어 hash로 후보를 선정하고 문서·URL·명시한 near-duplicate를 분리한다. Domain 겹침과 Wikipedia 일부는 허용하고 구성·알려진 평가 중복을 기록한다. 첫 controller는 S64만 사용하며 각 weight는1/64다.
- 각 입력은 자연 텍스트256 tokens에 BOS1개를 더한 길이257이다. 고정 teacher-forced prefix에서 logits index128…255의 전체 다음-token 분포를 저장하여128 scored positions를 만든다. Chat template·서로 다른 문서 연결·자유 생성 prefix는 첫 버전에 없다.
- S64와 Dev128의 W0 teacher를 먼저 만든다. Dev128은 W5/W10 development 확인용이며 candidate screen에는 쓰지 않는다. Reserve320 teacher는 연기하고 Report256 teacher·평가는 정책 lock 이후에 연다.
- 보호 objective 이름은 **D64**다. 512개 전체의 추정 보증이라고 부르지 않는다.
- Vulnerability16, dynamic core, medoids, witness-trigger는 BG-1에 없다.
- Native target을 위한 입력과 controller current loss는 제공된 canonical desired request에서만 얻는다. 공식 P/N·Historical 정답·Audit128·MMLU·FutureN은 controller 입력이 아니다.
- Generic control은 reporting panel과 sample/document identity를 분리한다. B512를 향후 선택에 쓰면 해당 평가도 controller-development 자료다.
- Teacher p0는 W0의 고정 full-vocabulary distribution이다. Generation trajectory 대신 고정 teacher-forced prefix와 position을 사용한다.
- Corpus가 바뀌었으므로 N4의 새 C4 D64로 ceiling을 다시 측정한다. FineWeb의 절대 b를 가져오거나 chain 안에서 entry마다 갱신하지 않는다. S128 확장은 S64+Reserve의 고정64개로 구성하며 Dev/Report를 학습에 흡수하지 않는다.

이 규약은 데이터 생성 설계다. 정식 reference768·exact tokenization·teacher cache는 아직 생성되지 않았다. 해당 준비와 좁은 기술 검사가 끝난 뒤 각 방법의 W0 B100×10 성능 비교를 진행한다.

\[
D_{64}(W)=\sum_{i\in S_{64}}w_i\,\frac1{128}
\sum_{\ell\in\text{score}(i)}
\operatorname{KL}(p_0(\cdot\mid x_{i,<\ell})\|p_W(\cdot\mid x_{i,<\ell})),
\quad \sum_iw_i=1 .
\]

실제 prefix 전체 token 수도 별도로 계수한다.

### 4.2 Current loss

\[
E_{\rm cur}(W)=\frac1{100}\sum_{r=1}^{100}
\frac1{|y_r|}\sum_{\ell=1}^{|y_r|}
-\log p_W(y_{r,\ell}\mid x_r,y_{r,<\ell}).
\]

요청별 desired-token 평균 후 request 평균이다. Native target optimizer의 여러 rewrite context·local KL는 변경하지 않는다. Route current loss의 canonical100과 native target objective를 같은 것이라고 표기하지 않는다.

평균 current loss는 후보 ranking에 쓴다. 개별 R/PS/strict 성공과 tail은 별도로 평가한다. 하나의 batch에서 모든 요청의 성공을 요구하여 batch 전체를 자동 탈락시키지는 않는다.

### 4.3 Native proposal 및 executable map

Wparent에서 native L4 target Zprop와 canonical Hparent를 계산한다. Native solve의 P, K, history와 context replication을 고정하여 S(R)=RA를 정의한다. Rprop=Zprop−Hparent다.

\[
\Phi(R,\eta)=W_{\rm parent}+\eta RA,\qquad \eta_{\rm nominal}=1 .
\]

Phi는 모든 token에 작용하는 실제 L4 down-projection 수정이다. Subject 위치 한 곳의 activation hook으로 대신하지 않는다.

### 4.4 Barrier objective

W0에서 D64=0이므로 b>0인 고정 ceiling을 둔다. 실행 중 매 batch마다 allowance를 다시 더하지 않는다.

\[
h(R)=\frac{b-D_{64}(\Phi(R,1))}{b},\quad
F(R)=E_{\rm cur}(\Phi(R,1))+\mu b_\tau(h(R)).
\]

Relaxed barrier는 사용자 v1의 C2 식을 그대로 쓴다.

\[
b_\tau(h)=
\begin{cases}
-\log h,&h\ge\tau,\\
-\log\tau-(h-\tau)/\tau+(h-\tau)^2/(2\tau^2),&h<\tau.
\end{cases}
\]

초기 trust ζ=.25, τ=.1을 제안한다. 측정된 최적값이 아니다. μ와 b의 calibration은 §8에서 별도 기록한다.

### 4.5 한 번의 correction

Nominal native preview에서 current/control microbatch의 정확한 weighting으로 gradient를 누적한다.

\[
G=\nabla_R F(R_{\rm prop}),\quad
\alpha=\zeta\frac{\|S(R_{\rm prop})\|_F}{\|S(G)\|_F+\epsilon_{\rm num}},
\quad R_{\rm tmp}=R_{\rm prop}-\alpha G.
\]

Fixed A이므로 inverse에 대한 gradient나 HVP/PCG는 필요 없다. G=0·zero native norm이면 correction을 생략하고 원 후보만 평가한다. Nonfinite gradient는 유효한 correction으로 사용하지 않고 typed 상태를 남긴다. Epsilon·alpha cap·numeric dtype은 실행 manifest에 고정한다.

Ztmp=Hparent+Rtmp를 native가 사용한 요청별 anchor/radius ball에 투영한다. Zprop 자체의 ball 만족을 먼저 확인한다. C=Zproj−Zprop에 [0,1] scale을 적용하여

\[
\|S(C)\|_F\le\zeta\|S(R_{\rm prop})\|_F
\]

를 만족시킨다. 최종 residual은 Rcorr=Rprop+C다. Convex interpolation이 ball 안에 남는다는 주장은 두 endpoint가 모두 ball 안에 있을 때만 사용한다.

Gradient용 mathematical map, native solve·FP32 materialization의 차이를 기록한다. 임의 weight-gradient의 추가나 clamp 제거는 이 버전에 없다.

### 4.6 네 개 이하의 endpoint

모든 후보는 같은 parent에서 평가한다.

| 후보 | 방향 | η |
| --- | --- | ---: |
| RAW | Rprop | 1 |
| CORR1 | Rcorr | 1 |
| CORR_HALF | Rcorr | .5 |
| CORR_QUARTER | Rcorr | .25 |

실제 materialized forward의 D64≤b 조건을 통과한 후보 중 Ecur가 최소인 것을 선택한다. 동률은 더 작은 actual delta norm, 그다음 고정 candidate order로 푼다. Numeric screen tolerance를 쓰면 보고하는 실제 ceiling도 b+tolerance로 표시한다.

G 계산 시의 forward와 RAW screen을 재사용할 수 있는지는 동일 actual weight·dtype·mask인지 확인한다. 그렇지 않으면 별도 forward로 계수한다. Candidate table 밖에 몰래 추가 probe를 하지 않는다.

No valid candidate면 Wparent를 유지한다. RAW가 screen을 통과하여 선택되는 것은 명시적인 후보 선택이며, 제약을 무시한 fallback이 아니다. Rejected 후보도 비용에 포함한다.

### 4.7 B100 transaction과 ledger

1. Entry W/history/context/RNG를 저장한다.
2. 100개 요청의 target 계산이 완료된 후 batch write 후보를 만든다. B1 immediate write로 바꾸지 않는다.
3. 후보 평가는 history·cache·RNG 등 다음 batch의 상태를 바꾸지 않도록 복원한다.
4. 최종 endpoint 하나를 선택하거나 parent를 유지한다.
5. **처리한 B100의 key를 endpoint에서 native finalizer로 한 번 등록한다.** Zero-write batch도 observed-key history를 한 번 등록한다. Inner append는 0이다.
6. Canonical desired response가 final 상태에서 TF-strict인 요청은 fulfilled/accepted-label ledger에 최초 loss와 시각을 남긴다. Zero-write에서도 이미 충족된 요청은 “fulfilled without write”로 분리한다.
7. Weight-write acceptance와 desired-request fulfillment를 별도 column으로 기록한다. Failed·partial·zero-write 요청을 all-request denominator에서 제외하지 않는다.

BG-1은 ledger를 저장만 하고 old loss를 feedback에 쓰지 않는다. 이를 통해 후속 retention 분석의 acceptance-time 기준을 처음부터 확보한다.

## 5. Accepted-old 확장 BG-1R

동일한 단일 proposal와 gradient quota에 selected old request64를 추가한다. 처음에는 age/relation-stratified reservoir를 사용한다. 동적 vulnerable32는 아직 넣지 않는다.

과거 accepted request i의 ceiling은

\[
c_i=L_i(W_{\rm accept,i})+\epsilon_{\rm old},\quad
h_i(W)=(c_i-L_i(W))/\epsilon_{\rm old}.
\]

로 고정한다. Old NLL의 token reduction은 Ecur와 같다. Tolerance는 control-development에서 한 값으로 고정하며 재수락·매 batch entry loss로 상승시키지 않는다.

\[
F_R(R)=E_{\rm cur}(\Phi(R))
+\mu_{\rm base}b_\tau(h_{\rm base})
+\frac{\mu_{\rm old}}{|J_t|}\sum_{i\in J_t}b_\tau(h_i).
\]

Jt가 비어 있으면 old term은 0이다. Per-request constraints는 평균 objective와 별개로 screen한다.

Entry에서 이미 ceiling을 넘은 selected old request는 원 ceiling을 유지한 채 L_i(candidate)≤L_i(parent)+numeric tolerance를 별도로 요구한다. 해당 endpoint는 debt-nonincreasing으로 표시하며 original-ceiling feasible과 합치지 않는다. 처음 feasible한 request는 original ceiling을 계속 적용한다.

Generic64와 selected old64 각각의 feasible/debt/violation 수를 보고한다. 64개 replay가 전체 old facts를 보호한다는 주장은 하지 않는다.

Conflict는 같은 subject/relation의 최신 **requested** desired target을 active task로 두는 공통 규칙을 적용한다. 새로운 overwrite 요청 시 이전 desired label은 superseded로 분리하고, 새 요청의 실패는 원분모에 남긴다. 동일 target의 반복은 기존 ceiling을 올리지 않는다. 같은 B100 안의 상충 요청도 요청 목록에서 삭제하지 않고 마지막 intended target과 superseded 상태를 따로 표시한다. 이는 exact-key task rule이며 semantic conflict 해결이 아니다.

## 6. 두 stage 확장 BG-2R과 강한 endpoint 대조

### 6.1 BG-2R

- Stage0: fresh native target, max24 Adam, η=.75, route correction 최대1.
- Stage1: 선택된 현재 provisional 모델에서 fresh native target, max24 Adam, η=1, route correction 최대1.
- 각 stage 후보는 RAW nominal과 CORR nominal/half/quarter의 최대4개다. 전체 최대8개 endpoint이며 preview의 추가 materialization forward는 실제 발생량을 별도 센다.
- Native optimizer state·local teacher·anchor는 원 REFIT4 fresh semantics를 따른다. I2/I4의 entry-offset·Adam carry를 혼합하지 않는다.
- Inner K, P, M, core, selected old, ceilings는 고정하고 batch finalizer는 한 번이다.
- 첫 비교에서는 stage 생략을 하지 않는다. Adaptive stopping은 별도 비교에서 모든 해당 대조군에 동일 규칙을 적용한다.

각 stage의 gradient는 해당 nominal η의 Phi에서 구한다. Trust normalization과 bound에도 동일 η를 사용한다:
α=ζ‖ηS(Rprop)‖/(‖ηS(G)‖+ε), ‖ηS(C)‖≤ζ‖ηS(Rprop)‖.
두 stage의 ceiling과 loss reduction은 동일하며 stage마다 새 damage allowance를 추가하지 않는다.

BG-1R과 비교하여 추가 target 재계산 비용을 포함한 frontier를 본다. REFIT4의 warm NS 신호 때문에 두 stage가 항상 더 좋다고 가정하지 않는다.

### 6.2 EP-2G-R: 실제 한 번의 write보다 endpoint 최적화를 비교

Fixed A에서 두 stage의 최종 변화는 Beff A다. 따라서 strong comparator는 Wparent+B A를 직접 두 번의 route gradient로 최적화하고 마지막에 materialize한다.

첫 native target ball 하나로 Beff를 제한하면 BG-2R보다 좁은 공간이 될 수 있다. 첫 비교에서는 direct endpoint residual의 추가 target clamp를 제거한 강한 대조로 허용하고, 양쪽에 동일한 batch-end write-norm bound와 output/old screen을 적용한다. Bound는 각 branch entry의 native delta norm에 대한 고정 배수로 control-development에서 선언한다. 실행 결과를 보고 BG-2R의 norm을 그대로 가져온 대조는 oracle 진단으로 표시한다.

초기 common endpoint bound 제안은 2×native norm이다. 이는 자명한 성능 동등성이나 최적값이 아니다. BG-2R의 실제 정상 후보를 과도하게 잘라내는지는 bound-hit 빈도로 공개한다. Bind되면 같은 개발 예산에서 공통 bound를 바꾸고 두 정책을 W0부터 다시 비교한다.

EP-2G-R과 BG-2R의 native target 호출 수는 다를 수 있다. Gradient quota2 비교와 실제 wall 비교를 함께 보고하며 이를 동일 compute라고 쓰지 않는다. Wall 범위에서 추가 endpoint optimization을 허용한 강한 대조는 별도 운영점으로 둔다.

BG-2R과 똑같은 virtual stage0 모델에서 똑같은 z를 다시 계산하고 똑같은 stage1을 수행한 뒤 마지막에 한 번 materialize하는 구현은 이 비교의 독립적인 알고리즘이 아니다. 정확한 동일 endpoint는 예상되는 등가다.

## 7. 실험 라운드

### A. 첫 W0 SEQ1000: 7 chains, 70 logical batches

| 정책 | 시작 | 길이 | 역할 |
| --- | --- | --- | --- |
| AlphaEdit | W0 | B1–B10 | 원 다층 baseline |
| MEMIT | W0 | B1–B10 | 원 다층 baseline |
| AlphaEdit-BLUE | W0 | B1–B10 | BLUE baseline |
| MEMIT-BLUE | W0 | B1–B10 | BLUE baseline |
| AlphaEdit-L4_only | W0 | B1–B10 | BG-1의 직접 native baseline |
| REFIT4 | W0 | B1–B10 | 강한 무보존-guide 두 stage |
| BG-1 | W0 | B1–B10 | 최소 방법 |

동일한 1,000개 요청을 일곱 독립 경로가 처리한다. 7,000개 unique edit나 full10k 실행이 아니다.

기존 baseline trajectory가 이미 W0에서 동일 source/model/order/config로 실행되었고 필요한 raw evaluation을 갖추면 재사용할 수 있다. “중간 checkpoint에서 method 시작”과 “W0부터 실행된 baseline 결과 재사용”은 다르다. 새 evaluator로 재평가하거나 누락된 시간·acceptance 기록을 보완한 비용은 별도 표시한다. Paired 비용/수치 주장은 같은 backend의 native 재실행 대조를 우선한다.

### B. BG-1의 기전 비교: 3 chains, 30 batches를 다음 라운드에 추가

| 정책 | BG-1과 동일하게 유지 | 바꾸는 것 |
| --- | --- | --- |
| Scalar guard | generic64, ceiling, canonical current ranking, 최종 screen | 방향은 raw로 고정; scale {1,.75,.5,.25} 네 개 |
| Fixed penalty | native proposal, gradient1, ball/trust, 네 후보·screen | μbτ(h)를 λD64로 교체 |
| Edit-gradient + guard | 위와 동일 | route objective에서 KL gradient 계수만 0; current gradient와 KL screen 유지 |

모두 W0 B1–B10이다. RAW direction에도 같은 finite screen이 주는 효과, current executable optimization 자체의 효과, slack weighting의 효과를 나눈다.

Fixed penalty는 동일한 development 예산으로 강도를 정한다. Barrier 첫 gradient와 같은 크기를 주는 slope-matched coefficient를 초기 대조로 포함할 수 있다. 부적절한 λ 한 개와의 비교만으로 nonlinear barrier의 필요성을 결론 내리지 않는다. Slope를 매 gradient마다 정확히 맞춘 adaptive penalty는 이론적으로 같은 방향을 주므로 추가 expensive run의 선행조건으로 삼지 않는다.

### C. 필요한 확장만 W0 SEQ1000으로 실행

BG-1R, BG-2R, EP-2G-R을 차례로 추가한다. 각 하나당 10 batches다. BG-1R의 정보·objective가 고정되기 전에 두 stage·dynamic core까지 한꺼번에 바꾸지 않는다.

한 batch 기술 검사는 허용한다. 이것은 native parity·VJP·transaction·memory 오류를 찾기 위한 검사이며 scientific ranking의 근거로 쓰지 않는다.

### D. Lock 후 full10k

선택된 하나의 방법과 다섯 baseline을 W0→B100으로 비교한다. 6 chains, 600 logical batches다. 동일 조건의 완결된 baseline 재사용은 provenance를 만족할 때만 허용한다.

Pilot에서 정책을 바꿨으면 이전 BG prefix endpoint에서 이어 붙이지 않고 W0부터 locked policy를 적용한다. 이미 개발에 사용한 첫1,000개가 새 reset만으로 독립 test가 되지는 않는다. 개발 사용 범위와 나머지 suffix를 명시하고, 이후 사전 고정한 추가 order에서 장기 재현을 평가한다.

Order robustness를 주장하려면 해당 claim의 비교군을 같은 추가 order에서도 실행한다. 문항 bootstrap은 order 반복의 대체물이 아니다.

Audit128/MMLU68은 최종 정책 lock 후 독립적인 일반능력 확인으로 둔다. “비슷할 것”이라는 예측은 완료 사실이 아니며 이 확인을 생략했다면 미측정이라고 남긴다. Method 구현이나 첫 SEQ1000 실행을 기다리게 하지 않는다.

## 8. Hyperparameter와 calibration

현재 empirical control KL의 scale을 새 S64에서 측정하지 않았으므로 b나 μ가 적정하다고 발표할 근거는 없다. 다음은 넓은 sweep을 피하기 위한 초기 calibration 규약 제안이다.

- ζ=.25, τ=.1은 한 개의 초기 설정이다.
- Development의 W0 N4 B1–B10에서 같은 S64의 D64를 수집한다.
- Pilot의 첫 budget은 b0=.9×max_{t=1,…,10}D64(Wt,N4)로 둔다. 수치상 0이면 positive floor를 명시한다.
- 초기 μ0=.01을 사용한다. 이는 Ecur와 barrier의 자연적인 동일 scale을 뜻하지 않는 시작값이다. Gradient component norm·clamp·rejection을 보고 필요한 변경을 같은 제한된 개발 예산에서 한다.
- Old 확장 첫 tolerance는 .1 nats/desired token, μold=.01을 제안한다. Measurement에 의해 적정성이 검증된 값이 아니다.
- Fixed penalty에는 동일 calibration 정보를 주고 λ의 개발 예산을 barrier보다 불리하게 하지 않는다.
- b0의 .9는 임시 operating point이며 논문의 performance gate가 아니다. Budget을 바꾸면 별도 정책 설정으로 W0 전체 1,000개를 다시 실행한다.

이 calibration은 개발 자료를 사용하므로 첫 SEQ1000 표는 development 결과다. N4가 같은 1,000개를 먼저 처리하여 얻은 D64를 이용한 것을 독립 test나 온라인 무료 정보로 부르지 않는다. Calibration forward·teacher·재실행을 development ledger에 포함한다.

확증 stream에서는 μ,b,tolerance,ceiling scale, core, candidate menu를 시작 전에 lock하고 미래 batch의 결과로 바꾸지 않는다. Development의 b0를 10k에 그대로 사용해야 한다는 뜻은 아니지만, 10k 실행 도중 ceiling을 올리며 같은 fixed-budget 정책이라고 보고해서는 안 된다.

## 9. 비용 계약

### 9.1 BG-1

- Native target full 상한1/request; actual Adam update·loss evaluation을 기록.
- 추가 route gradient1/B100: generic 8,192 backward-scored tokens + current target tokens. Full prefix forward/backward token은 별도.
- Materialized candidate 최대4: generic만 최대32,768 forward-scored tokens. Raw preview의 별도 actual forward가 필요하면 그 비용을 추가 기록.
- Original teacher S64 FP32 full-vocab 저장량: vocab128,256일 때 약3.914 GiB. CPU streaming을 기본으로 하고 GPU 상주를 전제하지 않는다.
- Solver/factorization, key/readout, rejected candidates, restore, finalizer, teacher read, logging/I/O를 모두 계수한다.

### 9.2 후속 확장

Old64의 실제 target/context token, BG-2R의 두 번째 target와 gradient, full-bank 확인, core refresh를 추가 비용으로 기록한다. Native z의 optimizer 한 step과 route gradient 한 pass를 동등한 단위로 쓰지 않는다.

Warm N4 약304.89초/B100, REFIT4 약372.75초/B100은 과거 참고치다. W0 실행의 measured N4가 새 비교 기준이다. 목표1.5×와 첫 자원 추정2×는 조정 가능한 계획 수치이며 quality·CI·cost를 묶은 scientific AND gate가 아니다.

Scheduler allocation은 1,000개를 완료할 수 있도록 실제 token/memory profile과 평가·저장 여유를 포함해 정한다. 할당 만료로 끝나지 않은 경로를 완료 성능표에 넣지 않는다. Runtime 초과 시 비용 항목·호출 빈도를 근거로 단순화하고, unseen 결과가 좋다는 이유만으로 quota를 숨겨 늘리지 않는다.

Online/development/evaluation/setup를 분리한다. 추가 original teacher·control 텍스트·old labels 접근량도 baseline 대비 증가한 자원이다.

## 10. 평가와 기록

W0 SEQ1000의 terminal full-seen은 1,000개다. 기존 5,000개 + 신규1,000개라는 warm 실험 분모를 재사용하지 않는다.

- 주표: all-request RS/PS/NS, canonical/P TF-strict, 두 paraphrase 모두 strict, true/new NLL 평균·paired tail.
- Coverage: write accepted/rejected, RAW/CORR 선택, fulfilled with/without write, partial success, 원분모.
- 시계열: current batch의 entry→stage endpoint→terminal. 공식 N은 평가에만 사용한다.
- Retention: 각 요청 at-write→W10, 첫500의 W5→W10, active/superseded. W0 pilot에는 “진입 전 과거5000”이 없다.
- Old ledger: acceptance-time loss, 고정 ceiling, selected coverage, entry debt, recovered/new violation.
- Generalization: 별도 reporting data의 original KL, Wiki/general NLL, 최종 lock 이후 Audit/MMLU. Control D64와 NS를 같은 지표로 해석하지 않는다.
- Geometry: raw/corrected residual, actual delta norm·cosine·비평행 성분, gradient component norm, target clamp·trust hit, acceptance scale.
- State: W/P/K/M·contexts·RNG hash, native source/config, history append count, cache changes.
- Cost: actual forward/backward calls·전체 tokens·scored tokens·wall·GPU/host memory·teacher bytes·rejected probes.

결과의 uncertainty는 paired request-cluster 분석으로 보여줄 수 있다. 0을 포함하는 CI 하나로 다음 연구를 자동 금지하지 않고, 효과 크기·quality와 cost의 교환·추가 order 결과를 해석한다.

## 11. 구현 시 필요한 좁은 검증

과학적 사전 선별이 아니라 잘못된 구현을 막기 위한 검사다.

1. Residual repeat/transpose를 포함한 native map shape와 finite-difference VJP.
2. Route-disabled 상태의 native proposal·endpoint·history parity.
3. Full down-projection의 모든 token에 update가 작용하는지 확인. Detached weight copy 때문에 gradient가 끊기지 않는지 검사.
4. Gradient accumulation의 token/request/mass weighting, teacher KL 방향, fixed-original reference.
5. C2 join과 h<0 gradient, raw/corrected ball·norm bound.
6. Candidate restore, finalizer exactly-once, rejected batch의 history, ledger conflict/ceiling 동작.
7. Native FP32 solve·cached map materialization 차이와 actual-forward screen.

현재 수행한 것은 작은 CPU 수식·산술 점검이다. LLM VJP, native FP32 parity와 성능은 미검증이다.

## 12. 결과에 따른 허용 주장

| 결과 | 유지할 주장 | 보류할 주장 |
| --- | --- | --- |
| BG-1이 native보다 좋은 endpoint/frontier | executable preservation feedback의 유용성 | multistep/ODE 필요성 |
| Scalar guard가 같거나 우세 | scale selection의 실용성 | 방향 제어의 추가 가치 |
| Edit-gradient + guard가 같음 | executable current optimization 또는 screen의 효과 | KL gradient의 필수성 |
| Fixed penalty가 같음 | preservation penalty의 유용성 | log-barrier의 독자적 필요성 |
| BG-1R이 old-active 손실을 줄임 | accepted-label feedback의 효과 | 전체 old knowledge 보증 |
| BG-2R이 wall-matched EP보다 좋음 | 해당 resource 범위의 fresh-target schedule 이점 | 새 affine space 또는 물리적 write 횟수의 필수성 |
| Control KL만 좋아짐 | 보호 surrogate 안의 변화 | 일반 지식·NS·old retention 개선 |
| Rejection으로 NS가 좋아짐 | coverage–quality–preservation frontier | 무손실 locality |
| Endpoint 같고 sampled peak만 낮음 | 제한된 경로 진단 | deployment 성능·continuous bypass |

ODE를 논문의 핵심으로 삼으려면 상세 검토의 fixed-horizon flow/CBF 조건과 같은 정의를 별도로 구현하고, 수치 step과 field가 섞이지 않는 비교를 해야 한다. 첫 BG-1을 곧바로 ODE 방법의 성공으로 명명하지 않는다.
