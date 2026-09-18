# 단일 레이어 방법의 연구 가치와 구체화

작성: 2026-09-18. 성격: 첨부 두 답변·기존 분석의 비판적 검토와 방법 제안. 새 모델 실행이나 성능 개선 결과가 아니다. 비교 출발점은 W0이며, 별도의 paraphrase target·학습·선택 set을 만들지 않는다.

## 1. 판단

우선 투자할 질문은 **이미 성공한 single-layer 편집을 유지하면서, 그 편집이 다른 입력에 일으키는 손상만 줄일 수 있는가**이다. 최소충분 편집(MSSE)은 중요한 대조군이지만 현재 형태로는 주 방법의 차별성이 약하다.

다만 첨부 첫 답변의 `nullspace + functional KL`을 그대로 새 방법으로 주장하는 것도 부족하다. 가치가 생기는 지점은 다음 연결을 실제로 입증하는 데 있다.

1. target residual을 바꾸는 기존 parameterization에는 없는 보정 자유도가 같은 weight matrix 안에 남는다.
2. 그 자유도가 실제 functional 손상을 줄이는 방향과 겹치는지 정량적으로 판정할 수 있다.
3. 해당 방향으로 이동하면 지정된 편집 입력의 반응은 그대로 유지하면서 독립 locality가 개선된다.
4. 이 효과가 관측하지 않은 PS와 장기 편집 유지에도 유효하다.

현재 1은 조건부 선형대수 명제이고, 2–4는 L4 실제 실험에서 아직 검증되지 않았다. 연구의 잠재력과 확보된 성과를 구분해야 한다.

## 2. 두 첨부 답변과 앞선 분석의 비교

|방향|얻으려는 자유도|장점|가장 큰 문제|판단|
|---|---|---|---|---|
|최소충분 편집 MSSE|성공을 만족하는 여러 target·write 중 선택|실제 행동과 비용을 직접 연결|DOW-KE의 실제 write 학습, KLOD의 충분성 목적과 중복; SL-ZFlow 부정적 결과|주 방법보다 강한 대조군|
|편집 반응을 유지하는 교정|같은 편집 반응을 만드는 여러 weight 중 선택|강한 N4를 그대로 출발점으로 활용; 추가 z 불필요|anchor rank 소진, PS 미보장, 기존 projection 연구와 중복|우선 검증할 주 방향|
|앞선 subject patch–all-token write 분석|손상이 발생하는 token 경로를 분리|원인에 대응하는 방법을 정할 수 있음|분해 자체가 해결책은 아님; non-subject 작용이 전부 불필요하지 않음|방법을 뒷받침할 기전 분석|

앞선 분석은 subject patch와 실제 write의 차이가 locality 손상의 주원인이라고 확정하지 않았다. 그 구분은 여전히 가설이며, DOW-KE도 실제 write와 위치별 gradient 문제를 다룬다. 이것을 최초 발견으로 주장해서는 안 된다.

## 3. 기존 근거에서 확실한 것과 정정할 것

### 3.1 Single-layer의 잔여 locality 문제는 크다

[fixed10k 상세 감사](06_blue_l4_detailed_audit.md)에서 AlphaEdit L4-only는 final RS/PS/NS=99.390/95.680/65.348이다. BLUE는 98.880/95.775/63.726이다. L4-only가 모든 지표·요청을 지배하는 것은 아니지만 강한 기준점이다.

L4-only는 W0에서 맞혔던 neighborhood 89,212개 중 26,583개를 잃었다. rewrite와 두 paraphrase를 at-write와 final 모두 유지한 9,174개 요청 중 7,003개도 원래 맞혔던 neighbor를 하나 이상 잃었다. **편집 성공과 원지식 손상은 상당 부분 공존한다.** 이 결과는 원인 규명이나 교정 가능성의 증명은 아니다.

LD의 NS +2.84pp에는 PS −1.25pp가 동반된다. REFIT4도 NS +1.61pp, PS −0.25pp다. 따라서 현재 필요한 것은 편집을 약화시켜 얻는 locality와 구별되는 효과다.

### 3.2 8월 31일 fixed-z screen은 직접적인 L4 교정 성능 근거가 아니다

[원 보고서](/mnt/raid5/janghj/ODE-edit/experiment-reports/servers/server4/fixed-z-nonuniqueness-screen-2026-08-31-v1/fixed-z-nonuniqueness-screen-factual-ko.md)와 per-case CSV·실행 코드를 함께 확인했다.

- 네 model/editor 조합 모두 L4–L8 native 편집 후 **마지막 L8**에서 tangent를 추가했다. 고정 L4-only 실험이 아니다.
- 동일 covariance action은 **후보끼리**의 조건이다. 후보 action은 native의 1.05배다. native와 같은 비용에서의 개선이 아니다.
- 32/32 case에서 8개 후보 중 최저 CVaR는 native보다 낮았다. 평균 절대 개선은 Llama/Alpha 5.52e−5, Llama/MEMIT 3.37e−6, Qwen/Alpha 5.90e−5, Qwen/MEMIT 4.35e−5다. case별 상대 개선의 평균은 각각 약 1.18%, 2.18%, 3.37%, 3.17%다.
- 같은 32개 probe에서 사후 최저 후보를 고른 값이다. 독립 표본에서 선택 가능한 개선이나 lifelong 이득이 아니다.
- CVaR는 32개 prompt의 마지막 token full-vocabulary KL 중 worst 4 평균이다. **최저 CVaR 후보 32개 중 14개는 neighborhood true NLL이 오히려 증가했다.** 256개 후보 모두 canonical/paraphrase TF-strict 성공수는 native와 같았지만 NLL은 변했다. screen KL, neighborhood 보존, 전체 출력 동일성을 구별해야 한다.
- lock은 native 평균 edit key·일부 history subject key·일부 target-position key다. **모든 편집 sequence의 전체 token을 고정한 결과가 아니다.** target-logit 상대 차이도 최대 약 1.18e−3으로 0이 아니다.
- 후보의 output 방향은 native rank-one update의 output 방향과 직교하게 생성됐다. 따라서 native output span만 유지하는 저비용 교정의 근거로 바로 옮길 수 없다.

이 실험의 가치는 “동일한 일부 local constraint와 후보 간 동일 proxy 비용으로도 functional 결과가 달라질 수 있다”는 초기 단서다. 큰 성능 여유가 이미 확보됐다는 뜻은 아니다.

### 3.3 과증폭만으로 locality를 설명하기 어렵다

BLUE L8 target의 95.7%는 zero-step이다. 일부 nonzero residual이 batch solve를 통해 다른 입력에도 영향을 줄 수 있으므로, 모든 요청을 계속 과증폭하는 현상으로 해석하면 부정확하다.

[SL-ZFlow 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-seq1000-review-ko.md)에서는 actual-write 학습 후 NS가 8.42pp 낮아졌지만 canonical new NLL은 오히려 N4가 더 낮았다. 그러므로 “canonical confidence를 낮추면 locality가 해결된다”는 단일 설명도 현재 근거와 맞지 않는다.

### 3.4 기존 보정의 결과는 전부 0도, 확대하면 개선도 아니다

[EP-TW cap 후속 보고](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-sh4-local-z-detailed-review-20260917-v1/experiment-reports/servers/server4/ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)에서 CAP10은 native 대비 최종 성공수 +1R/+4P/+3N이다(분모 1,000/2,000/10,000). CAP100과 NORM_ONLY는 N 성공수가 각각 20개, 30개 감소했고 9/10 batch에서 RAW를 택했다. 단순 cap 확대에 일관된 개선은 없지만 보정이 절대 불가능했다는 근거도 아니다.

해당 family의 finite quality screen은 exact all-token lock과 다르고, 모델 수준 derivative 검증도 보고서상 NOT_ESTABLISHED다. 후속 방법의 공간 차이를 논할 수는 있지만 이 결과를 순수한 geometry 한 요인의 인과 실험으로 간주하지 않는다.

## 4. 주 방법의 핵심: 실제로 성공한 반응을 고정하고 보정 공간을 연다

한 개의 고정 L4 down projection만 편집한다. batch entry를 W_t, native local-z write를 Δ_N, native endpoint를 W_N=W_t+Δ_N이라 한다. W0는 lifelong 시작 전 모델이다.

고정 token sequence X에서 해당 projection의 입력 K(X)는 W의 변화에 영향을 받지 않는다. 다른 parameter를 모두 고정했으므로, 추가 교정 D의 layer output 변화는 정확히

\[
H(W_N+D,X)-H(W_N,X)=D K(X)
\]

이다. 이후 logits까지 선형이라는 뜻은 아니다.

보호할 실제 편집 sequence의 모든 유효 token key를 K_E로 모은다. 기본 보호 범위는 현재 요청과 native target fitting에 이미 사용한 rewrite contexts다. 공식 paraphrase나 neighborhood 평가 prompt를 넣지 않는다. native essence KL prompt는 rewrite contexts와 별도이며, 포함 여부를 명시해야 한다.

\[
\boxed{D K_E=0}
\]

이면 지정된 sequence의 전체 layer output과 이후 logits는 실수 연산에서 그대로다. finite step에도 성립한다. 구현에서는 factorization·FP32 write의 오차를 실제 forward로 확인해야 한다.

**보장의 범위:** new-target teacher-forcing sequence만 포함하면 그 sequence의 logits/NLL/strict를 보존한다. multi-token old target은 다른 입력 경로이므로 pairwise RS까지 정확히 유지하려면 기존에 제공된 old/new teacher-forcing sequence의 union이 필요하다. 미관측 paraphrase와 자유 생성의 모든 가능한 경로는 보장하지 않는다.

### 4.1 기존 target-side correction과의 실제 차이

native writer가 Δ=R A이면 target-side correction은 D=C A다. 이 family에서 동일 편집 반응을 요구하면

\[
C A K_E=0.
\]

AK_E가 full row rank일 때 C=0뿐이다. 반면 전체 weight에서 DK_E=0을 만족하는 nonzero D는 남을 수 있다.

따라서 중요한 구별은 **z가 고정됐는가보다 writer의 right factor까지 고정됐는가**이다. fixed-z와 fixed-native-basis를 동시에 유지하면 자유도가 없을 수 있지만, 같은 편집 반응을 더 넓은 weight 공간에서 구현할 자유도는 별도 문제다.

다만 EP-TW는 exact equality가 아니라 finite quality screen을 썼다. 이 정리만으로 그 실험의 작은 교정이나 RAW 선택 원인을 설명할 수는 없다. 실제 rank와 허용 방향을 측정해야 한다.

### 4.2 AlphaEdit projection을 유지하는 정확한 교정 공간

P=V Vᵀ가 이상적인 직교 projector이고 V가 그 range의 orthonormal basis라고 하자. J=VᵀK_E이면

\[
Q_E=V(I-JJ^\dagger)V^\top
\]

는 range(P)와 ker(K_Eᵀ)의 교집합 projector다. D=B Q_E로 두면 DP=D와 DK_E=0을 함께 만족한다. 두 projector를 임의 순서로 곱해서 같은 결과라고 가정하지 않는다.

실제 저장된 P가 수치적으로 idempotent하지 않다면 이상적 공간과 native 연산을 구분한다. 작은 singular value를 임의로 잘라낸 뒤 전체 원본 key에 대한 exact guarantee를 주장해서도 안 된다.

## 5. 목적과 업데이트: norm보다 실제 원지식 분포를 사용한다

편집 scope 밖 calibration 입력 집합 B에서 W0의 분포를 teacher로 두고

\[
\mathcal L_0(W)=\mathbb E_{x\in B}
\operatorname{KL}(p_{W_0}(\cdot\mid x)\Vert p_W(\cdot\mid x))
\]

를 최소화한다. 실제 full-token forward를 사용하며 평가용 N은 학습하지 않는다. W_t를 teacher로 쓰면 이미 누적된 손상을 정상 상태로 인정하게 되므로 원지식 회복 목적에는 W0가 맞다. 반대로 의도한 편집의 유효 범위를 W0로 되돌리면 안 되므로 calibration의 scope는 별도 검증이 필요하다. subject 문자열 제외만으로 semantic scope가 완전히 분리되지는 않는다.

핵심 최적화는

\[
\min_D\mathcal L_0(W_N+D),\qquad
DK_E=0,\quad D=DP
\]

이다. 전체 Δ를 .75배 하는 식의 attenuation이 아니다. native가 현재 입력에서 만든 실제 반응을 고정하고, 나머지 방향에서 locality를 바꾼다.

가장 단순한 계산은 G=∇_W L0(W_N+D)를 구한 뒤

\[
D\leftarrow D-\eta GQ_E
\]

로 이동하는 것이다. η는 실제 L0 감소를 확인하는 line search로 정한다. 종료 허용오차·최대 계산량은 필요하지만, batch별 .75 같은 임의 강도 메뉴가 방법의 중심이 되지는 않는다. 본질은 projected gradient라는 기존 알고리즘의 발명이 아니라, **보호해야 할 실현 반응과 접근할 보정 공간의 선택**이다.

보정은 같은 weight에 합친다. 추가 layer, inference-time router, 새 local-z optimization을 요구하지 않는다.

### 5.1 먼저 계산할 정량 신호

\[
\boxed{\chi=\|GQ_E\|_F^2}
\]

는 Euclidean metric에서 허용 공간에 남은 functional gradient의 크기다. 방향 −GQ_E에서

\[
\left.\frac{d}{d\eta}\mathcal L_0(W_N+D-\eta GQ_E)\right|_{\eta=0}
=-\chi.
\]

따라서 χ>0이면 해당 calibration loss를 줄이면서 지정 edit response를 유지하는 충분히 작은 step이 존재한다. 이는 **측정한 loss·현재 endpoint·해당 제약 공간에서의 국소 결과**다. held-out NS 개선, PS 보존, 큰 step 성공, 전역 최적을 보장하지 않는다. 추가 active-history guard까지 통과하는 방향이라는 뜻도 아니다. χ=0도 전역적으로 개선 가능한 해가 없다는 뜻은 아니다.

\(\|GQ_E\|_F^2/\|GP\|_F^2\)는 동일 Euclidean 좌표에서 허용된 locality gradient 중 얼마가 남는지 보여준다. 분모가 0이면 별도로 처리한다. 이 비율을 semantic capacity나 보편적인 난이도로 부르지 않는다.

### 5.2 너무 좁은 저비용 parameterization을 다시 만들지 않는다

U=orth(Δ_N)의 output span에 제한하여 D=U B Q_E로 만들면 변수 수를 줄일 수 있다. 이때 gradient는 UUᵀGQ_E이고, 보정 가능성도 이 더 좁은 family에 한정된다.

그러나 8월 31일 screen의 유효 후보는 native output 방향 바깥을 사용했다. 따라서 **첫 원리 검증부터 U로 제한하지 않는 편이 맞다.** 전체 edit-null 방향이 유효함을 확인한 다음, 실제 gradient spectrum이 허용할 때 비용을 줄이는 옵션으로 검토한다. output span 고정은 각 요청의 semantic content 고정과도 다르다.

## 6. 반드시 반영할 norm budget의 제약

분석용 native 목적이

\[
\Delta_N=\arg\min_\Delta\tfrac12\|\Delta K-R\|_F^2
+\tfrac12\operatorname{tr}(\Delta C\Delta^\top),\qquad C\succ0
\]

라고 하자. 같은 허용 공간의 교정 D가 DK=0이면 정상조건에서

\[
\langle\Delta_N C,D\rangle=0
\]

이므로

\[
\boxed{\|\Delta_N+D\|_C^2
=\|\Delta_N\|_C^2+\|D\|_C^2.}
\]

**같은 실현 반응, 같은 native quadratic objective, native 이하 비용을 동시에 요구하면 D=0만 남는다.** 이 조건에서 norm을 줄이려는 방법은 설계부터 막힌다.

그러므로 성공적인 교정은 같은 native proxy 비용을 더 쓰면서 functional locality를 개선할 수도 있다. 이 경우 결과는 “더 작게 편집했다”보다 “proxy가 구분하지 못했던 유효 방향을 찾았다”로 해석해야 한다.

위 식은 같은 C와 native 최적성의 조건부 결과다. W0 대비 누적 변화 E+Δ_N의 비용에는 ⟨E C,D⟩ cross term이 남는다. batch update norm, 누적 W−W0 norm, 원지식 covariance, native history+ridge metric을 서로 바꿔 쓰면 안 된다.

기존 [two-memory v2 감사](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-review-ko.md)도 보호 key의 mapping energy 감소가 같은 bank의 output KL 감소를 보장하지 않음을 보여줬다. 따라서 activation least-squares 복원을 주 방법으로 다시 제안하지 않는다.

## 7. Single-layer의 budget은 무엇으로 측정해야 하는가

최소 세 가지를 분리해야 한다.

|양|의미|의미하지 않는 것|
|---|---|---|
|q=rank(P)−rank(VᵀK_E)|지정 반응을 유지하며 움직일 수 있는 input 방향 수|functional 개선 가능성|
|χ=‖GQ_E‖²|현재 loss를 줄일 수 있는 국소 방향의 크기|held-out 개선·전역 capacity|
|projected preservation key의 singular values|해당 원지식 반응을 교정하는 conditioning|모든 지식의 semantic 분리 가능성|

N이 교정 공간의 orthonormal basis이고 K_B가 보존 입력 key라면 Z=NᵀK_B다. B0=(W_N−W0)K_B를 선형 activation 수준에서 없앨 때 잔여량은

\[
B_{\rm residual}=B_0(I-Z^\dagger Z).
\]

이 표준 least-squares 잔여는 activation 복원의 한계다. NS의 불가능성 정리가 아니다.

직관적으로 edited key와 preservation key가 같은 방향이면 한쪽의 반응을 그대로 두고 다른 쪽만 바꿀 수 없다. 거의 같은 방향이면 큰 교정이 필요할 수 있다. q가 크다는 것만으로 locality 비용이 작다는 결론은 나오지 않는다.

## 8. Long horizon과 PS에서 숨기면 안 되는 한계

### 8.1 모든 과거 sequence를 exact lock하는 방법은 기본 해법으로 삼지 않는다

전체 lifetime의 모든 token key를 계속 쌓으면 rank가 input dimension까지 차고 Q_E=0이 될 수 있다. rank compression이 동일 span을 정확하게 보존하는 경우에만 같은 보장이 유지된다. 근사 sketch는 제한한 정확도만 주장해야 한다.

첫 실용 범위는 **현재 batch의 선언한 sequence에 대한 exact response lock**, 과거 active edit에는 별도 soft preservation 또는 finite quality 확인이다. 이 조합은 현재 편집의 보정 단계에 대한 보장을 갖지만 모든 과거 편집의 lifelong exact preservation을 갖지는 않는다. rank가 소진되면 보정을 생략하고 그 빈도를 기록해야 한다. 작은 방향을 임의로 버려 놓고 exact라고 부르지 않는다.

### 8.2 어떤 과거 상태를 보호하는지가 중요하다

W_N에서 이미 잊힌 history를 DK_H=0으로 잠그면 그 망각도 고정된다. 이는 추가 보정의 피해 방지일 뿐 batch entry 또는 과거 수용 시점의 기억 회복이 아니다.

과거 수용 시점의 down-projection 출력 H_acc=W_acc K_H를 회복하려면

\[
DK_H=H_{\rm acc}-W_NK_H
\]

라는 비동차 제약 또는 해당 teacher에 대한 soft loss가 필요하다. 현재 edit과 충돌하여 infeasible할 수 있다. 이후 다른 target으로 갱신된 fact는 최신 active target을 기준으로 처리해야 하며, 모든 과거 target을 동시에 보호하는 문제가 아니다.

실용적으로 history 보정은 별도 항목으로 평가한다. W_N 기준 no-harm은 D=0이 feasible하지만 native가 이미 만든 손실을 허용한다. W_t 또는 accepted-state 기준은 더 강하지만 D=0조차 feasible하지 않을 수 있다. 이 선택을 숨기지 않는다.

### 8.3 Paraphrase는 observer다

새 paraphrase set을 만들지 않고 official PS를 학습·gradient·line search·후보 선택에 넣지 않는다. native rewrite contexts가 제공하는 보호 범위만 사용한다.

미관측 X에서, P 제약이 없는 간단한 경우

\[
\|DK(X)\|_F\leq\|D\|_2\|(I-K_EK_E^\dagger)K(X)\|_F
\]

가 성립한다. 이것은 anchor span에서 멀리 떨어진 입력이 영향을 받을 수 있음을 설명한다. downstream 민감도까지 없으면 출력 오차 bound가 아니고, PS 보장은 더욱 아니다. PS 손실이 반복되면 “동일 편집을 보존했다”는 표현을 지정 training sequence 범위로 제한하고 방법의 우선순위를 내려야 한다.

## 9. 계산 비용과 단순화

native local-z는 기존처럼 한 번 계산한다. correction 후보마다 새 z를 최적화하지 않는다. K_E·calibration key·W0 teacher는 고정 입력과 single-site 조건에서 재사용할 수 있다. downstream functional gradient는 현재 weight마다 달라지므로 cache된 gradient를 무조건 재사용할 수는 없다.

Q_E를 거대한 dense matrix로 매번 만들 필요는 없다. orthonormal basis를 이용해 G의 보호 방향 성분을 빼는 연산으로 구현할 수 있다. 실제 비용은 rank factorization, calibration suffix forward/backward, line-search forward다.

현재 L4-only 실험의 target optimization은 edit 시간의 약 96.75%였다. 따라서 새 z 호출을 피하는 것은 타당하지만, correction을 붙이면 N4보다 빨라진다고 주장할 수는 없다. 추가 backward와 메모리를 포함한 비용 대비 locality 이득이 필요하다. gradient projection이 작으면 계산을 생략할 근거가 되지만 threshold·실제 이득의 관계는 검증해야 한다.

## 10. 선행연구와의 경계

|연구|이미 다룬 핵심|이번 방향이 확보해야 할 차이|
|---|---|---|
|[EMMET](https://arxiv.org/html/2403.14236), §5 식8–9|동일 edit equality 아래 preservation 최소화|새 equality 원리가 아니라 실제 전체-token response와 functional 손상 사이의 간극|
|[CrispEdit](https://arxiv.org/html/2602.15823v2), §3|capability curvature의 저곡률 방향으로 edit; history 통계 갱신|지정된 성공 반응의 finite 보존과 기존 writer 밖 교정 공간의 유효성|
|[DOW-KE](https://arxiv.org/html/2608.16932v1), Method 식8–10|fixed right factor에서 actual-write residual 학습; 위치별 gradient routing|fixed writer의 residual 재학습 대신, 실현 반응을 유지하며 다른 weight 방향 사용|
|[KLOD](https://arxiv.org/html/2608.27839v1), §2.3 식14–18|충분한 target 확률에서 증폭 중단, non-target/prefix 분포 보존; single-layer FT|새 성공 threshold보다 이미 얻은 실제 edit response를 고정한 교정|

[CoRE](https://arxiv.org/html/2505.23026)의 context robustness, [SUIT](https://arxiv.org/html/2509.24502)의 key subspace 정제도 있으므로, context 증가·공통 key 제거를 독립적인 새 핵심으로 추가하지 않는다.

`projector`, `functional KL`, `single layer`, `finite update`, `제약의 목적함수 전환` 중 어느 하나만으로 최초성을 주장하지 않는다. 문헌 조사가 novelty를 보증하지도 않는다.

첨부 MSSE의 \(b^2/(2a^\top M^{-1}a)\)는 한 개의 선형화된 제약에 대한 표준 최소 비용이다. batch에서는 요청별 값을 합하는 대신 Jacobian 사이의 cross term을 포함한 joint QP가 필요하다. 이미 충분한 요청의 margin 여유도 유지해야 한다. B100의 residual 변수는 4096×100=409,600개이므로 100×100 metric을 푼다는 사실만으로 전체 비선형 최적화가 저렴하다고 볼 수 없다. 이 점에서도 MSSE를 검증 없이 효율적 대체재로 선택하지 않는다.

## 11. 큰 연구 포지션이 되기 위한 판정

강한 포지션은 **편집 성능을 유지하려면 locality 손상을 함께 감수해야 한다고 보였던 상황에서, 손상 중 일부가 편집에 필수적이지 않고 기존 writer의 선택 때문에 남았음을 보여주는 것**이다.

이를 위해 필요한 증거는 대규모 grid가 아니라 다음의 명확한 연결이다.

1. 같은 W0·같은 native endpoint에서 target-side family와 전체 edit-null family의 rank·functional 방향을 비교한다.
2. 지정한 전체-token 편집 반응을 실제로 유지하면서 독립 locality가 개선되는지 확인한다.
3. official PS·strict·active-history 유지가 동반되는지 평가한다. 이 평가를 online 선택에 사용하지 않는다.
4. 단순 KL 보정, 충분성 기반 약화, 같은 정보·비용을 준 기존 방법으로 동일 효과가 설명되는지 확인한다.

q가 거의 없거나 χ가 작다면 현재 exact-lock 방향은 투자가치가 낮다. χ가 커도 독립 locality가 좋아지지 않으면 calibration 또는 목적의 일반화 문제다. N 개선과 PS 손실이 계속 묶이면 기존 LD/REFIT의 교환관계를 넘어선 것이 아니다.

**따라서 지금 채택할 것은 검증된 새 방법이라는 결론이 아니라, “고정 single-layer에서 같은 편집 반응을 보존하는 다른 weight를 찾아 locality를 개선한다”는 한 가지 연구 중심축이다.** MSSE와 layer allocation은 그 주장에 필요한 비교 방법으로 위치시킨다. exact response, 유효 교정 자유도, 독립 기능 보존이 함께 확인될 때 기존 작업들과 구분되는 큰 기여가 된다.

## 확인 범위

첨부 전체, 로컬 감사 문서·CSV, 관련 실행 코드, 위 primary 논문의 본문을 검토했다. 이 문서의 투영·방향미분·quadratic-cost 항등식은 작은 FP64 선형대수 예에서도 확인했다(잔여 오차 약 1e−13 이하). 이는 실제 모델 성능 검증이 아니다. 신규 GPU 실험, 모델 checkpoint 재현, evaluation PS를 이용한 설계 최적화는 수행하지 않았다. 기존 survey·audit 파일은 수정하지 않았다.
