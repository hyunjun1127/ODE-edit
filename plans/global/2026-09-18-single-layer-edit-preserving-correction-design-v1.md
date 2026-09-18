# Single-layer edit-preserving correction: W0 기반 실험 설계 v1

작성: 2026-09-18. 상태: 상세 설계와 CPU 수학 검증. **실제 모델용 runner는 미구현이며 GPU job을 제출하지 않았다.** 설계의 수치값은 사전 설정이며 성능상 최적값이라는 뜻이 아니다.

관련 분석: [방법 포지셔닝](/mnt/raid5/janghj/layer_allocation/07_single_layer_method_positioning.md), [BLUE/L4 상세 감사](/mnt/raid5/janghj/layer_allocation/06_blue_l4_detailed_audit.md). 실행 조건은 [contract](2026-09-18-single-layer-edit-preserving-correction-contract-v1.json), 실행 단위는 [cells](2026-09-18-single-layer-edit-preserving-correction-cells-v1.csv)에 함께 고정한다.

## 1. 이번 실험에서 답할 질문

**같은 L4 native 편집이 만든 실제 반응을 유지하면서, 기존 target-side writer 밖의 방향으로 원지식 손상을 줄일 수 있는가?**

주 방법은 EN-F(full-token edit-null correction)다. L4 local-z와 native write를 한 번 수행한 후, 현재 편집 입력의 전체 token 반응을 유지하는 공간에서 W0 분포 KL을 낮춘다. 추가 layer, 재계산한 z, 새 paraphrase 학습 데이터, inference router를 도입하지 않는다.

검증할 가설을 분리한다.

|가설|필요한 증거|증거로 충분하지 않은 것|
|---|---|---|
|H1: 기존 writer 밖에 유효 자유도가 남음|동일 endpoint에서 CA exact 공간과 EN-F 공간의 rank·functional gradient 비교|weight parameter 수가 많음|
|H2: 그 방향으로 실제 locality 개선 가능|수치적 full-token response 보존, 개발 observer와 N 개선|S64 학습 KL 감소만 관측|
|H3: 단순 under-edit 효과와 구별됨|canonical 반응 유지, PS/strict/joint 관측상 무손실, SCALE 대비 비교|RS 평균만 같음|
|H4: 장기 적용에 유효|각자 W0부터 진행한 chain에서 arrival damage와 old forgetting 분리|공통 native 경로에서 얻은 shadow 결과를 이어붙임|
|H5: 비용이 실용적|추가 z=0, 실제 backward·forward·factorization·저장 비용과 이득|변수가 적거나 solve가 closed-form이라는 설명|

H1–H3가 먼저다. layer allocation sweep이나 MSSE 새 optimizer 구현을 동시에 진행하지 않는다. 모든 과학 결과는 성공하지 못한 native 요청과 correction fallback을 포함한 전체 분모로 보고한다.

## 2. 고정 실행 조건

- 모델: Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
- 편집 parameter: zero-based `model.layers.4.mlp.down_proj.weight`, shape `[4096,14336]` 하나.
- 모든 독립 batch와 모든 sequential chain은 pretrained W0에서 시작한다. warm 5k checkpoint는 사용하지 않는다. sequential chain 내부에서는 자기 직전 상태를 이어받는다.
- FP32, eval mode, eager attention, matmul/cuDNN TF32 모두 off. tokenizer·padding·microbatch·kernel/library identity를 runtime lock에 저장한다.
- Native hparams: L2=1, lr=.1, decay=.5, clamp=.75, essence KL=.0625, loss evaluation 최대25/Adam 최대24, native total-loss early-stop .05. native 계산 순서와 projector/history 연산은 변경하지 않는다.
- 공통 method seed=20260916. cold7에서 생성한 native context text/token bytes를 그대로 봉인해 사용한다. 해당 capsule을 확인할 수 없으면 W0에서 한 번 생성해 새 context ID로 모든 arm에 공통 사용하고, 과거 결과와 byte parity를 주장하지 않는다.
- P4는 W0 기반 원본을 유지한다. M4는 시작 시 exact zero이며 commit마다 native 규약대로 한 번 append한다. 교정 후보·실패 후보에서는 append하지 않는다.
- 다른 weight의 full manifest hash를 시작/종료에 검사하고 batch 경계에서는 mutation guard를 검사한다. pointer/version 검사만으로 전체 byte 동일성을 주장하지 않는다.

본 설계의 N4는 새 runtime에서 다시 실행한다. 기존 N4/BLUE 결과는 역사적 참고이며 신규 결과의 대체재가 아니다. B1 parity가 이후 trajectory 전체의 동일성을 증명하지 않는다.

## 3. 요청과 데이터 역할

### 3.1 편집 요청

기존 fixed10k dataset SHA는 `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, whole-order SHA는 `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`, first1000 root는 `40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd`다. 이는 기존 보고의 identity이며 실행 전에 실제 파일을 재검증한다.

M/S는 기존 순서 O0의 첫1,000개를 사용한다. M에서는 100개씩 10개 독립 batch로 나눠 매번 W0/M0로 복구한다. S에서는 B100×10의 실제 sequential chain으로 진행한다.

O1/O2는 동일 요청 집합을 `SHA256('ENFC-v1|order|20260918|case_id')`, `SHA256('ENFC-v1|order|20260919|case_id')` 순으로 정렬한다. tie는 case_id 문자열 순이다. 1k에서는 first1k 집합, 10k에서는 fixed10k 전체 집합에 각각 적용하므로 서로의 prefix라고 가정하지 않는다. duplicate case_id는 preflight 실패다. order manifest를 첫 loss 이전에 저장한다.

이미 본 fixed10k와 first1k는 개발·재현 데이터다. W0 재시작이나 새 순서가 unseen-request test를 만들어 주지는 않는다. O1/O2의 역할은 순서 강건성이다. 새 dataset/model 일반화는 별도 확장 단계다.

### 3.2 입력별 접근 권한

|집합|구성|optimizer 접근|역할|
|---|---|---|---|
|Current|현재 canonical 및 native rewrite contexts, 제공된 old/new target|허용|lock·finite guard|
|Native essence|native z에 원래 쓰이는 별도 essence prompt|native z에서 허용, correction에는 평가만|기존 절차 유지; full rewrite lock과 구별|
|S64|기존 C4 train 64문서|허용|원지식 KL gradient·line search|
|Past64|이미 받은 최신 active 요청의 canonical old/new 경로|허용|correction의 추가 history 손상 guard|
|Dev128|기존 C4 train의 분리된128문서|금지|완료 후 개발 observer|
|Report256|기존 C4 validation256; teacher 확인/생성 필요|금지|10k 모든 정책 봉인 후 마지막 holdout 보고|
|Official P/N|현재/이미 받은 요청의 공식 평가 prompt|금지|선택 봉인 후 observer|

새 paraphrase target·생성·replay set은 없다. P/N은 gradient, step, candidate selection, rank cutoff, online stop에 사용할 수 없다. 완료된 단계의 official PS로 다음 연구 단계의 확대 여부를 판단할 수 있으나 이는 개발 의사결정으로 기록한다. 후보별 PS를 보고 좋은 step을 골라 재보고하지 않는다.

S64/Dev128 C4 reference manifest SHA=`f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0`, Teacher192 manifest SHA=`f81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a`. 기존 완료 teacher를 재검증해 사용하며 오래된 reference contract의 PENDING 문자열을 최신 상태로 복사하지 않는다.

C4 입력은 BOS+256 tokens=257, scored logits indices `[128,256)`의128위치다. 전체 입력 token에 실제 write가 작용한다. S64 loss는 vocab sum→128 position mean→64 document mean의 `KL(p_W0 || p_W)`다. last-token KL, top-k KL, target-excluded KL로 조용히 바꾸지 않는다. teacher는 W0·full vocabulary FP32 logp로 고정한다. S64 한 sweep은64문서/8,192 scored positions/16,448 input tokens이며 backward token 비용을8,192로만 보고하지 않는다.

미래 편집 subject를 조회해 calibration bank를 바꾸지 않는다. 현재/이미 받은 subject 문자열·target과 reference의 overlap flag는 기록하되 v1 안에서 loss를 보고 문서를 교체하지 않는다. 기존 overlap audit의 범위를 표시하며 의미적 scope 분리가 완전하다고 주장하지 않는다.

Report256은 M/S/R 결과를 보고 설정을 고르는 동안 열지 않는다. teacher 준비는 가능하지만 method loss 출력은 reader 분리로 막는다. L 단계 전체 arm/source/threshold 봉인 후 모든 L chain 종료 때 한 번 보고한다. 일찍 열었다면 개발 observer로 재분류하고 독립 holdout이라고 부르지 않는다.

## 4. Current lock의 정확한 정의

현재 batch native endpoint를 W_N=W_entry+Δ_N으로 둔다. K_E는 다음 실제 token sequence들의 모든 유효 입력 위치에서 얻은 down_proj input key다.

1. native가 target fitting에 사용한 각 rewrite context+new target의 teacher-forcing 입력.
2. 같은 contexts+제공된 old target의 teacher-forcing 입력. new/old가 같은 입력 prefix면 token bytes로 deduplicate한다.
3. canonical prompt가 위 목록에 없으면 명시적으로 추가한다. 기존 native context에 포함됐는지 로그로 확인한다.

EOS를 새로 덧붙이지 않고 원래 tokenizer/teacher-forcing 규약을 그대로 쓴다. padding 위치는 제외한다. 서로 다른 context의 동일한 문자열 token도 prefix가 다르면 다른 key다. 첫 token만 보호하거나 subject key 평균으로 대체하지 않는다. old target 누락은 count하며 해당 요청은 new 경로만 보호하고 pairwise RS exact claim 분모에서 제외한다.

W0 fixed key 함수의 stationarity를 실제 W_entry/W_N/candidate에서 검사한다. 이 성질은 고정 token 입력에만 적용되며 생성 결과가 바뀐 이후의 새로운 경로까지 같은 key라는 뜻은 아니다.

수학적 조건은 `D K_E=0`. 실제 적용은 `W_cand=FP32(W_N+D_acc64)`, `D_actual=float64(W_cand)-float64(W_N)`로 정의해 실물 weight의 반응을 검사한다. D_acc64가 null이라고 D_actual도 정확히 null이라고 가정하지 않는다.

## 5. 교정 공간과 진단

P_raw는 native writer에서 그대로 사용한다. correction용 P_star는 생성 provenance에서 확인한 **native P_raw의 허용 range**, 즉 원 preservation covariance의 작은-eigenvalue 공간에 대한 직교 projector다. provenance가 없으면 sym(P_raw)의 eigenvalue를 .5에서 분리하되, `max min(|lambda|,|lambda−1|)≤.1`이고 [.1,.9] 구간에 값이 없는 경우만 복원한다. 큰 음수나1을 크게 넘는 값도 기술 실패다. 이 복원이 native P_raw의 숫자를 바꾸는 것은 아니다. P_raw 대 P_star 차이를 보고한다.

V가 range(P_star)의 orthonormal basis, J=VᵀK_E이면 `Q_E=V(I-JJ†)Vᵀ`. 두 projector를 단순 순차 곱하지 않는다. 실제로는 basis를 사용해 matrix-free로 GQ_E를 계산할 수 있다.

rank는 포착한 FP32 key를 FP64로 올린 선형대수에서 `tau=max(shape)*eps64*sigma_max`를 기본 threshold로 계산한다. `[tau/10,10*tau]`에 singular value가 있으면 RANK_UNRESOLVED로 기록하며 성능에 따라 cutoff를 바꾸지 않는다. sigma_max=0은 zero matrix로 처리한다. near-rank 문제의 조사·수정은 새 기술 버전으로 분리한다. FP32 epsilon으로 큰 cutoff를 만들어 near-null 방향을 exact free 방향으로 풀지 않는다.

필수 진단:

- rank(K_E), rank(VᵀK_E), q=rank(P_star)−rank(VᵀK_E), condition과 singular spectrum.
- native factor A의 numerical row rank, rank(AK_E), `row(A) ∩ ker(K_Eᵀ)` 차원. CA exact-null은 geometry 대조이며 거의0이면 optimization arm을 억지로 만들지 않는다.
- 동일 G에서 `||GP_star||²`, `||GQ_E||²`, 그 비율. q>0과 functional gradient가 남는다는 것은 다르다.
- native 출력 span U에 대해 `||UUᵀGQ_E||²/||GQ_E||²`를 사후 기록한다. v1 주 방법을 이 span에 제한하지 않는다.
- K_N의 비subject token 작용 및 subject 평균/개별 context 간 반응 차이. 이것만으로 손상의 인과 기여율을 계산하지 않는다.

## 6. 비교군

M의 모든 arm은 **같은 native target·Δ_N·W_N·A·K·P·M·S64 teacher**를 공유한다. 최초 G도 동일 목적이면 공유할 수 있지만, endpoint가 달라진 이후 gradient는 새로 계산한다. 공유 비용과 각 arm을 단독 실행한 환산 비용을 함께 보고한다.

|Arm|보정 공간/목적|Current 보호|핵심 대조|
|---|---|---|---|
|N4|native endpoint 그대로|native 성능|기준|
|SCALE|W_entry+sΔ_N, s∈[0,1]; S64 KL 최소화|finite quality guard|전체 편집 약화로 설명되는가|
|CA|row(A) 안에서 S64 KL 최소화|finite quality guard|기존 target-side family 밖 자유도가 필요한가|
|KL-P|range(P_star)에서 S64 KL 최소화|finite quality guard|일반 KL 보정으로 같은 효과가 나는가|
|EN-S|native subject 평균 key를 잠근 공간; S64 KL|추가 full Current finite guard|전체 token lock이 필요한가|
|EN-F|전체 K_E를 잠근 공간; S64 KL|response invariant+finite quality guard|주 방법|
|EN-COV|전체 K_E lock; C4 입력 전체 token의 W0 activation drift 최소화|EN-F와 동일|proxy 최소화와 functional 목적의 차이|
|EN-F4|EN-F와 같은 공간·목적, gradient 최대4회|EN-F와 동일|M에서만 계산 예산 대조; 주 방법 대체/online 선택 금지|

CA는 coefficient C의 임의 좌표 steepest descent 대신 orth(row(A))로 weight-space Euclidean projection을 사용한다. 원래 EP-TW의 optimizer·cap 재현이라고 부르지 않는다. raw A의 P_star 밖 수치 leakage도 기록한다.

EN-COV objective는 문서별 valid-token 평균 `0.5*||(W−W0)K_x||²`의 S64 문서 평균이다. batch increment만의 비용이 아니며 mean key로 대체하지 않는다. S64 output KL은 이 arm의 step 선택에 사용하지 않고 endpoint observer로만 기록한다.

M에 두 진단을 추가한다. **CA-EXACT**는 공간/rank/gradient만 계산한다. **RAND±**는 선택된 EN-F correction의 Frobenius norm과 맞춘 `±Gaussian Q_E` 한 쌍이다. seed=`SHA256(ENFC-v1|random|episode_id)`; 두 부호를 모두 보고하며 성능으로 선택하지 않는다. EN-F가0이면 paired random도0이다. 실제 FP32 norm match 상대오차1e−3 이내를 확인하고, 아니면 mismatch를 기록해 matched라고 부르지 않는다. Random은 standalone controller·sequential arm이 아니다.

## 7. Optimizer를 한 가지로 고정

모든 functional optimization arm은 실제 배포 weight의 full-token forward로 loss를 계산한다. pre-subject detach, subject-only straight-through gradient를 사용하지 않는다.

EN-F/EN-S/KL-P/CA는 respective orthogonal weight-space projector Π에 대해 `V=Π(G)`, `chi=<G,V>=||V||²`를 쓴다. initial proposed step은 **Polyak 형태 `eta0=L/chi`**다(KL 또는 activation loss의 알려진 하한0 사용). 고정 .75/.5 메뉴나 native norm의 일정 비율을 목표로 삼지 않는다. 0이 edit-lock 아래 실제 도달 가능한 최솟값이라는 뜻은 아니며, χ가 작으면 제안이 지나치게 클 수 있다. 이 step이 최적이라는 보장은 없으므로 실제 Armijo 감소로 backtrack한다. 유한 L/chi가 FP64에서도 표현되지 않으면 STEP_SCALE_UNRESOLVED로 종료하며 infinity를 반복 halving하지 않는다.

- **주 방법과 공통 대조는 gradient1회, trial 최대8회**다. trial=`eta0*0.5^j`, j=0..7, Armijo c=1e−4.
- EN-F4만 gradient 최대4회, 각 round 최대6 trial(j=0..5), 총24 trial이다. 각 round는 현재 accepted endpoint에서 fresh gradient를 계산한다. 이 대조는 M에서만 실행한다.
- initial loss는 첫 gradient sweep과 공유한다. 동일 시작점에서 EN-F와 EN-F4의 초기 gradient는 공유 가능하다.
- accepted 조건: 실제 이동의 예측 `p_actual=<G,W_trial−W_current>`가 음수이고 `L_trial <= L_current+c*p_actual`이며, 감소가 loss 수치허용오차보다 크고 아래 finite guards를 모두 통과할 것. nominal `−eta*chi`도 별도 기록한다. FP32 반올림이나 SCALE 경계 clipping 후에도 실제 이동을 기준으로 판정한다.
- max8(EN-F4 max24) objective sweep에는 실제로 계산한 invalid/rejected probe도 포함한다. 동일 weight SHA 후보는 deduplicate하며 이전 결과만 재사용한다.
- round의 모든 trial 실패면 탐색을 끝내고 현재 best accepted를 선택한다. budget 끝이면 같은 규칙으로 종료한다. 이를 수렴/KKT 만족으로 부르지 않는다.
- G/chi 비유한, accepted endpoint 비유한, OOM, wrong-state, projector 오류는 기술 실패다. 유한 base에서 너무 큰 trial의 계산값이 비유한이면 해당 trial을 reject하고 backtrack하며 count한다. candidate weight 자체가 비유한이면 model forward 없이 reject한다.
- D_actual=0 또는 목표 감소가 수치 분해능 이하이면 정상적인 NO_RESOLVED_STEP이다. χ=0은 이 loss의 현재 1차 방향이 없다는 뜻이다.

SCALE은 a=1−s∈[0,1], W=W_N−aΔ_N으로 parameterize한다. dL/da=−<G,Δ_N>에서 feasible descent가 있을 때 같은 Polyak/Armijo 규칙을 적용하고 [0,1] 경계에 제한한다. clipping 뒤의 nominal scalar 예측은 `g_a*(a_trial−a_current)`이며 unclipped eta*g_a²로 계산하지 않는다. 최종 수용에는 위 actual weight 이동의 내적을 사용한다. scalar의 여러 국소점/전역최적을 탐색했다는 주장은 하지 않는다. Quality guard가 모두 실패하면 s=1의 N4다.

후보마다 immutable W_N에서 materialize한다. FP32 `+D` 후 `−D`로 복원하지 않는다. native target ball 재투영은 교정에 적용하지 않는다: 그 projection이 DK_E=0을 깨뜨릴 수 있다. original z ball은 native fitting에서 그대로다. 동일 native quadratic cost 이하를 요구하는 추가 guard도 두지 않는다.

EN-F4에서는 ideal D_acc64와 실제 D_actual을 별도로 보존한다. 다음 gradient/loss/cache는 반드시 accepted `FP32(W_N+D_acc64)`에서 계산하고, cached 주입도 D_actual K를 사용한다. 다음 proposed correction은 ideal D_acc64에서 누적한다. D_acc64를 rounded D_actual로 덮어쓰거나 ideal action의 forward를 실제 weight의 forward로 간주하지 않는다.

## 8. 실제 품질·수치 계약

### 8.1 모든 non-native 후보의 공통 finite guard

- 현재 각 rewrite sequence의 new NLL ≤ own W_N의 같은 sequence NLL+1e−4.
- canonical old/new preference에서 W_N의 성공 ID가 하나도 소실되지 않을 것.
- W_N에서 new-target TF-strict였던 current sequence ID가 하나도 소실되지 않을 것.
- Past64에서도 요청별 new NLL+1e−4, canonical preference/strict 성공 ID subset을 동일하게 검사.
- 조건은 개별 sequence/request에 적용한다. 평균 NLL plateau .05를 사용하지 않는다. old target 누락은 pair guard만 제외하며 count한다.

1e−4는 수치/설계 허용오차이며 성능상 허용 손실폭이 아니다. 실제 exact response와 finite guard는 다른 조건이다. 같은 데이터의 finite guard를 CA/KL-P에도 제공하여 정보량 차이를 줄인다.

Past64는 현재 batch 이전에 받은 active (subject,relation) 요청 중 최신 target만 남기고, 현재 overwrite key를 제외한 뒤 `SHA256('ENFC-v1|past|case_id')` 우선64개다. 동률은 case_id. semantic alias 병합은 하지 않는다. label은 metadata proxy다. W_N을 guard reference로 써 D=0 fallback을 가능하게 한다. 이는 **native가 이미 만든 망각을 복원하는 조건이 아니며 correction의 추가 손상 방지**다. entry→native→selected의 과거 손상을 별도로 보고한다.

### 8.2 EN-F/EN-COV/RAND의 invariant 검사

다음은 v1 technical ceilings다. 효과를 본 뒤 완화하지 않는다. technical 단계에서 달성되지 않으면 이 버전의 exact-response 방법을 실행 가능하다고 판정하지 않고 새 수치 설계로 분리한다.

|검사|한도|
|---|---:|
|FP64 Q symmetry/idempotence relative error|1e−10|
|FP64 proposed D의 normalized DK_E residual|1e−10|
|actual FP32 correction의 max-token `||D_actual k||/max(1,||W_N k||)`|1e−5|
|actual correction의 P_star 밖 상대 norm|1e−5|
|보호 sequence valid-token full-vocab logits max absolute difference|1e−3|
|동일 logits RMS difference|1e−4|
|보호 sequence new/old NLL max absolute difference|1e−4|
|canonical preference·new TF-strict ID 변화|0|

전체 logits는 token/vocab streaming reduction으로 검사하고 전체 배열을 매 후보 영구 저장할 필요는 없다. 수치적 bound 이내라는 뜻이지 bitwise logits equality가 아니다. Candidate가 finite guard를 통과해도 invariant 실패면 EN-F 후보로 수용하지 않는다. Proposal 자체가 수학적 nullspace 검사를 실패하면 단순 성능 fallback으로 감추지 않고 기술 실패로 분류한다.

반복/no-op scoring의 max NLL 차이는1e−5 이하, logits max 차이는1e−4 이하를 요구한다. 이 noise band보다 작은 개선은 성능 신호로 세지 않는다. KL reduction은 signed FP64 누적을 사용하며 negative roundoff를 숨기기 위해 loss/gradient를0으로 clamp하지 않는다. |L|≤1e−6이면 numerical floor로 처리하고, L<−1e−6이면 teacher/reduction 기술 오류다. resolved decrease floor는1e−6이다. EN-COV는 자체 단위에 맞춰 roundoff receipt의10배를 사용하고 그 값을 첫 science loss 전에 고정한다.

### 8.3 derivative·forward 검증

T 단계의 독립 cold8 episode에서 direct physical weight leaf의 gradient와 cached all-token suffix VJP를 비교한다. relative norm error1e−4 이하가 필요하다. physical/cached loss 및 forward는 위 반복/동등성 한도 안에 있어야 한다.

EN-F 방향·CA 방향·random 방향에 대해 central finite difference를 `h0=.01||Δ_N||/||v||`, h0/2^j(j=0..11)에서 비교한다. 이 h는 진단용이며 method strength가 아니다. 실제 FP32 plus/minus weight 차이를 기록하고 계단화/ULP 구간을 구분한다. 신호가 noise의10배를 넘는 연속 두 scale에서 상대 AD–FD 오차1% 이하를 확인한다. 작은 AD는 absolute noise envelope로 분리하며 억지 상대오차를 만들지 않는다. 임의로 좋은 한 점만 골라 PASS하지 않는다.

새 방법은 actual weight gradient가 핵심이므로 이 technical 검증이 필요하다. 과거 다른 실험에서 검사를 생략했다는 사실이나 CPU toy PASS가 이번 actual-model derivative PASS를 대신하지 않는다.

## 9. 단계와 실행 범위

|단계|실행|목적|확대 조건|
|---|---|---|---|
|T|W0 cold8 기술 episode, N4 재실행2회|native parity·projector·teacher·gradient·materialization|기술 PASS|
|M|독립 W0 B100 10개; 8 arm과 RAND±/CA-EXACT 진단|동일 target·endpoint에서 공간/목적·계산량 효과 분리|아래 개발 gate|
|S|O0 first1k, N4/SCALE/CA/KL-P/EN-F 각각 W0→B100×10|실제 반복 적용과 history 영향|관측 품질 유지·실제 N/Dev 이득|
|R|O1/O2 first1k, N4/EN-F 각각 W0→B100×10|S의 O0와 합쳐3순서 강건성|방향 일관성; 설정 변경 없음|
|L|O0/O1/O2 fixed10k, N4/EN-F 6 chains|long horizon·최종 Report256|정책 봉인 후 전체 완료|

M에서는10 native fit을 공유한다. S 이후에는 arm별 자기 entry에서100개의 z를 새로 fit하고 자기 native preview를 만든다. 다른 arm의 B2 이후 z·Δ·A·teacher gradient를 재사용하지 않는다. M의 좋은 후보를 연결해 가상의 chain을 만들지 않는다.

M/S/R의 완료 전 P/N 결과를 online policy에 반환하지 않는다. 기술 실패 외에는 단계 안의 예정 batch를 끝까지 완료한다. 과학 gate는 단계 완료 후만 적용한다. 손실이 큰 arm도 native fallback을 포함한 full 분모로 남긴다.

### 9.1 확대/중단 기준

M에서 EN-F의 합법적인 nonzero correction이 하나도 없으면 공간·수치·quality·resource 중 원인을 분류하고 S로 확대하지 않는다. 최소 하나만 좋다고 큰 효과를 주장하지 않는다. EN-F−N4의 평균 Dev128 KL 감소가 noise band를 넘고, NS/PS/RS 점추정이 모두 비음수일 때 S 검증 대상으로 남긴다. TF-strict/joint 하락은 별도 명시해 품질유지 결론을 보류한다. 모든 M 결과는 개발 결과다.

S에서 R로 넘어갈 실용적 목표는 EN-F의 **NS +1.0pp 이상**, Dev128 mean KL **5% 이상 감소**, RS/PS/joint/P-strict 점추정 무손실이다. 이는 효과 크기 목표로 사전 정한 연구 투자 기준이며 algorithm gate가 아니다. NS만 좋아지고 N true NLL 평균이 나빠지면 보존 개선 판정을 보류하고 분포를 분석한다. S64만 감소하거나 PS가 감소하면 원래 주장으로10k를 시작하지 않는다.

점추정 무손실과 통계적 비열화 입증을 구분한다. CI가0을 걸치면 무손실이 입증됐다고 부르지 않고 R의 반복 검증으로 불확실성을 확인한다. R의 O1/O2를 포함한3순서에서 NS·Dev 개선 방향이 모두 같고 각 order의 RS/PS/joint/P-strict 점추정이 모두 비음수일 때 L을 검토한다. pooled 평균이 한 order의 악화를 가리지 못하게 한다. 기준을 통과하지 못한 결과는 STOP/INCONCLUSIVE로 남기며 사후 .5pp PS 허용폭을 만들지 않는다.

L은 등록된6 chain을 기술 실패 외에는 완료하며 Report256은 모두 끝난 후 공개한다. 성공 판정은 NS·W0-correct N retention·Report256 KL의 개선과 RS/PS/joint/strict 무손실 관측을 함께 요구한다. 형식적 비열화 주장은 해당 sampling unit에 타당한 interval이 지지할 때만 한다. 기준 미달이 모든 single-layer correction의 불가능성이라는 뜻은 아니다.

## 10. Observer schedule와 분석

각 batch에서 **W_entry → own W_N → selected endpoint**의 현재 R/P/N을 기록한다. P/N forward는 selection 봉인 뒤 저장 snapshot으로 수행한다. 미리 평가가 필요해도 policy process에 결과를 노출하지 않는다.

- current N entry→native: 이번 native write의 marginal damage.
- current N native→selected: correction 자체의 효과.
- W0→current entry: 과거 batch가 이번 요청의 neighborhood에 이미 만든 손상.
- at-write→final: 이후 batch로 인한 forgetting.

과거에 없었던 entry N을 추가해 arrival degradation과 이번 write의 손상을 분리한다. NS는 true_NLL<new_NLL의 상대 지표이므로 true/new NLL, success lost/gained ID, W0-correct retention을 함께 기록한다.

S/R full-seen은 B5/B10. L full-seen은 B10/B25/B50/B75/B100. current observer는 매 batch이며, 모든 요청의 W0 기준 값은 evaluator에서만 저장한다. Dev128은 M의 각 selected endpoint, S/R B5/B10, L B10/B25/B50/B75/B100. Report256은 L final만이다.

공식 지표: RS, PS, NS, request별 R+두P joint, canonical/P new TF-strict, old/new mean NLL·margin·paired p95/p99, raw와 ACTIVE/SUPERSEDED strata. exact string subject–relation version label을 semantic truth라고 부르지 않는다. 생성 평가는 current canonical에서 do_sample=false, max_new_tokens=32, 원 모델 EOS IDs 종료로 고정한다. generated new-target token-prefix match와 종료/길이를 별도 observer로 기록한다. target 길이가32를 넘는 요청은 censored로 count하며 자동으로 성공 분모에서 지우지 않는다. TF-strict를 자유 생성 정확도라고 부르지 않는다.

M에서는 모든 completed final arm endpoint와 RAND±를 평가한다. S/R에서는 N4 preview와 selected endpoint만 평가하고 rejected candidate의 P/N을 보지 않는다. step별 S64 좋아짐을 official metric 개선처럼 표시하지 않는다.

## 11. 통계와 가설의 판정

M의 paired sampling unit은 독립 cold batch(10개)다. arm 간 같은 batch를 함께 재표집하는10,000회 bootstrap, seed20260918을 사용한다. 각 요청의 두P/열N을 쪼개 독립표본으로 세지 않는다. 10 cluster의 interval은 불안정할 수 있어 모든 batch delta도 공개한다.

S의10 batch는 한 trajectory이므로 독립 반복 bootstrap으로 sequential-policy CI를 만들지 않는다. paired case 통계는 그 trajectory의 기술통계다. R/L의 최상위 반복은 order다. order별효과·범위와 pooled point를 전면에 두고3 order 기반 interval은 탐색적/기술적으로 표시한다. 유의차가 없다는 이유로 PS 동등성을 주장하지 않는다.

Functional bank의 표본 단위는 문서다. M의 Dev128×10 endpoint는 독립1,280문서가 아니므로 batch×document paired matrix를 보존하고, 고정128문서에 대한 batch-cluster interval과 고정10batch 평균에 대한 document-cluster interval을 구분한다. Report256×3 order도 독립768문서로 세지 않는다.

주 비교는 EN-F−N4다. CA/KL-P/EN-S/EN-COV/SCALE는 기전·대조 분석이며 모두 보고한다. 여러 arm 중 최고를 골라 주 방법을 바꾸려면 새 개발 버전이다. 여러 metric/arm의 confirmatory p-value를 주장할 경우 family와 보정을 먼저 등록해야 한다; v1에서는 유리한 p-value만 골라 보고하지 않는다.

기전 연결은 rank/χ와 **own-native→selected**의 actual ΔKL·ΔNS 사이를 분석한다. χ의 값·정규화 비율·step 크기를 각각 보고하며 M/S/L을 섞은 상관으로 인과 주장하지 않는다. χ가0인 경우, 공간이0인 경우, guard 때문에0인 경우, 예산 때문에0인 경우를 분리한다.

## 12. 장기 capacity 진단

N4 S chain에서 현재 full-token key를 저장하고 shadow로 누적 constraint rank를 B1/B5/B10에서 계산한다. 실제 방법에는 누적 exact lock을 적용하지 않는다. 동일 고정 K 함수에서 current-only, subject-mean cumulative, full-token cumulative의 q를 비교한다. streaming FP64 orthogonalization·rank 계산 비용/오차를 별도 기록하고 truncated sketch를 exact rank라고 부르지 않는다.

L에서는 current q/χ와 Past64를 결합한 shadow q를 B10/B25/B50/B75/B100에 측정한다. 모든10k sequence 전체 token을 dense로 저장·SVD하는 것을 필수 실행 비용으로 두지 않는다. 실제 조회하지 않은 전체 lifetime rank를 추정값만으로 확정하지 않는다.

Frobenius norm은 native Δ, correction D, total batch Δ+D, cumulative W−W0를 각각 기록한다. incremental energy 합을 cumulative norm으로 대체하지 않는다. W0/native/history/reference covariance의 metric ID를 모든 cost 열에 붙인다. native와 같은 quadratic 비용 이하라는 별도 조건은 없으며, norm 증가와 functional 개선의 공존도 중요한 결과다.

## 13. 계산 예산

한 주 방법 optimized batch의 상한은 native target100/Adam2400/solve1, 추가 z0, correction gradient1회, objective trial8회, Current/Past finite screen8회다. EN-F4만4gradient/24trial이다. EN-F는 accepted 또는 guard 전 단계에서 필요한 invariant 검사를 포함하고 비용을 숨기지 않는다. 후보에 따라 screen을 조기 종료할 수 있지만 거절 사유와 실제 처리 문서/token 수를 남긴다.

M: native fit10개, optimization controller70개(N4 제외7arm×10). gradient≤100, trial≤720. EN-COV는 activation-only gradient를 neural backward와 따로 계산한다. Random20개는 observer/invariant 평가이며 optimization budget에 끼워 넣지 않는다.

S:5 chains/50 native batch, correction40 controller. R:4 신규 chains/40 native batch, correction20 controller. L:6 chains/600 native batch, correction300 controller. M/S/R/L 총 native target call 상한70,000, native Adam1,680,000, native solve700. **이 전체는 조건부 확대 상한이며 지금 실행할 job 수가 아니다.** T의 기술 호출·teacher 생성·observer는 별도다.

전체 correction controller430개, gradient≤460, objective trial≤3,600. 이 중 M의 EN-COV10 controller는 activation-only다. neural gradient sweep 상한450회이며, S64 gradient scored positions≤3,686,400, input tokens≤7,401,600이다. Trial forward·native/guard/observer token은 이 수치 밖이며 별도 합산한다. 배치 크기·padding·suffix 길이·backbone cache 준비를 포함한 wall-clock을 실측해야 한다.

기존 native10k 시간의 선형 환산을 확정 예산으로 쓰지 않는다. T/M에서 native, key/QR, teacher I/O, gradient, rejected trial, current/past screen, official observer, 저장을 분리 계측한 후 같은 설정의 남은 호출 수로 예상 시간을 산출한다. max controller round를 늘리는 후속 실행은 별도 budget ablation이며 v1 안에서 성능을 보고 연장하지 않는다.

## 14. 산출물·재현·재개 규약

실행 전 capsule: model/tokenizer/data/order/context/P/hparams/native source/adapter source/teacher shard SHA, dtype·library·kernel·microbatch, all split IDs, numeric policy를 봉인한다. 원 파일의 path/size/SHA와 실제 availability를 함께 저장한다.

각 episode에는 다음을 남긴다.

- entry/native/selected L4 weight SHA와 native z·Kbar·A·M-entry identity; M의 공통 endpoint 동일성.
- full-token K_E token/position provenance, basis spectrum/rank/tau/ambiguity, P_raw/P_star 차이.
- 각 gradient의 source-state SHA, G의 norm·projection통계, 방향 artifact 또는 exact 재생 가능한 factor와 SHA.
- 모든 trial의 eta, proposed/actual norm, candidate SHA, signed objective, Armijo 판정, request별 guard violation, invariant max/rms, stop reason와 actual compute.
- selection ledger를 먼저 seal하고 observer reader가 해당 seal 뒤에만 P/N/Dev를 결합.
- native·selected 성공 ID와 paired loss, current entry/at-write/final, active metadata.
- 각 checkpoint의 현재존재 여부·retention policy. 생성했다고 나중에도 남아있다고 가정하지 않는다.

M의 모든 arm final L4 endpoint를 보존한다. S/R/L은 각 batch commit에 L4/M4/RNG/active registry/ledger를 묶은 atomic resume checkpoint를 쓰고, 새 checkpoint 검증 후 직전 rolling checkpoint를 정리할 수 있다. full-seen milestone·final과 그 batch의 native preview는 영구 유지한다. 모든 방향/tensor를 영구 보존할 수 없으면 retention manifest에 범위를 명시하며 삭제된 후보의 exact replay를 주장하지 않는다. raw prompt·teacher·대형 tensor는 local-only, 공유 보고에는 hashes·집계·정확한 경로만 둔다.

history는 실제 commit 후1회 append한다. native fallback도 commit이므로1회 append한다. technical failure는 entry와 history를 원상 보존하며 checkpoint를 성공으로 쓰지 않는다. crash recovery는 commit ID로 중복 append를 막는다. 다른 arm/순서/버전 checkpoint로 resume하지 않는다.

## 15. 구현 재사용 경계와 이후 논문 대조

실제 SL-ZFlow adapter의 all-token cache·physical materialization·durable checkpoint 구조를 참고하되, root checkout에는 CPU reference만 있을 수 있으므로 실행된 frozen worktree/source identity를 사용한다. writer@K 축약 cache를 그대로 쓰면 새 공간을 다시 CA에 가두므로 raw K 또는 정확한 null-basis K를 지원해야 한다. batch-entry reverse-KL 대신 고정 W0 forward-KL을 사용한다.

fixed-z screen의 마지막층 선택, 부분 token equality, ±update로 rollback, native output 직교 rank-one 후보를 그대로 주 방법으로 복사하지 않는다. 기존 EP-TW의 cap/target-ball/mean-E 화면도 v1 계약과 다르다.

M/S/R에서 효과가 없으면 외부 baseline 대규모 실행은 보류한다. 효과가 확인되면 논문 비교에는 BLUE, native AlphaEdit, DOW-KE, KLOD, CrispEdit가 필요하다. author configuration과 matched single-L4/data/compute configuration은 별도 표로 비교하며, 기존 표의 hparams 차이를 allocation의 인과 효과로 해석하지 않는다. 이 외부 runner들의 source·정보 예산을 감사하기 전에는 v1 cells에 실행 가능한 arm처럼 넣지 않는다.

이번 설계의 완료는 실험 실행 완료나 성능 검증을 뜻하지 않는다. 실제 모델의 numerical preflight, source capsule 구축, teacher availability 검증은 실행 준비 단계에 남아 있다. 이들은 새 method의 성능을 보기 전에 해결해야 하는 구체적인 구현 항목이다.
