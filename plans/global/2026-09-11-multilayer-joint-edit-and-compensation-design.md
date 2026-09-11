**다중 layer의 공동 편집 부담 분산과 편집 후 손상 보정 — 두 기전을 분리하는 설계**

2026-09-11 KST. 상태: 설계 및 수식 점검. 모델 구현·GPU 실험은 실행하지 않았다. 단일 L4 v2를 그대로 여러 layer에 복제하는 설계가 아니다. 아래 두 방법은 별도의 주 가설이며, 처음부터 결합하지 않는다.

**결정.** A는 We에서 L4·L8의 local target을 공동으로 계획하고 두 layer가 편집을 함께 담당하게 한다. B는 BLUE-L4 편집을 그대로 고정하고 L8에 기능적 보정만 추가한다. 각 방법에서 barrier 없는 one-shot과 두 elastic barrier를 갖는 경로를 비교한다. 새 방법의 차이는 layer 수 자체가 아니라 target의 생성 방식, layer의 역할, 관측하는 손상, 경로상의 재계산에 있다.

|구분|기존 AlphaEdit BLUE|A: 공동 편집 부담 분산|B: 편집 후 손상 보정|
|---|---|---|---|
|시작|We|We|동일 We의 BLUE-L4 endpoint WN|
|L4의 역할|전체 local target을 계산해 먼저 write|공동 목표의 일부를 담당; 전체 목표를 먼저 소진하지 않음|기존 native edit를 그대로 담당하고 이후 고정|
|L8의 목표|바뀐 모델에서 target_new에 대한 local z 계산|L4와 함께 하나의 Current 출력 목표를 만족|Base는 We 반응으로, Current는 WN 반응으로 유지·복원|
|실제 write를 정하는 순서|L4 완료 후 L8 결정|같은 snapshot에서 두 update를 함께 결정|L4 완료 후 L8 보정만 결정|
|손상 관측|native 통계·기존 target 목적|실제 모델의 Base/Past 출력 및 layer별 편집 기여|실제 모델의 Base/Past 출력 및 Current 유지|
|주 결론|강한 baseline|편집 자체의 기능적 부담을 옮겼는가|편집을 담당한 weight를 유지하면서 출력 손상을 보정했는가|

**1. 이번 설계에서 고치는 해석과 관측.** BLUE의 L4 평균 batch-net norm은 L4-only의 99.913%였다. 따라서 기존 BLUE는 실질적인 L4 부담 감소의 대조군이 아니다. 동시에 v2는 보호 key의 mapping energy가 거의 0이어도 동일 bank의 실제 KL/NLL이 악화됐다. 새 방법은 최종 출력에 대한 functional derivative를 사용한다. 보호 위치의 Dk만으로 Base 보존을 대체하지 않는다. [장기 실험 점검](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md), [v2 독립 점검](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-review-ko.md).

현재 공개 결과만으로 A 또는 B의 성공을 보장할 수 없다. A에서 L4의 편집 기여가 그대로이면 부담 분산 주장은 성립하지 않는다. B가 성공해도 편집 부담 분산의 증거로 쓰지 않는다. 어느 경우에도 제한된 algorithm의 실패를 single-layer 또는 multi-layer의 불가능성 증명으로 바꾸지 않는다.

**2. 공통 상태와 보존 대상.** W0는 누적 손상 평가용 원래 모델, We는 이번 batch 직전의 전체 모델이다. WN은 같은 We에서 원형 BLUE-L4를 적용한 모델이다. D4N=WN4−We4이며 다른 layer의 이번 batch delta는 0이다. 두 방법의 Base teacher는 pWe로 고정한다. B가 WN에서 시작하더라도 Base teacher를 pWN으로 바꾸지 않는다.

Current는 이번 batch의 최신 유효 target_new다. Past는 현재 batch가 교체하지 않는 과거 최신 target_new다. Base는 명시적으로 교체한 현재·과거 fact를 제외한 지식이다. Active-fact ledger, 동일 batch latest-wins, W0 평가, request→context→token 평균 규약은 기존 v2에서 이어받는다. 대체된 과거 target을 동시에 보존하라고 요구하지 않는다.

Current 학습은 native와 같은 clean/prefix context inventory와 target tokenization을 사용한다. 실제 held-out rephrase는 학습에 추가하지 않는다. Past는 canonical 및 기존 규약의 paraphrase, Base는 고정 teacher-forced continuation의 full-vocabulary 분포를 쓴다. 모든 derivative는 전체 입력 sequence를 통과한 실제 모델 출력에서 구한다. 예측 위치를 관측하더라도 prefix 전체의 weight 적용 효과가 미분에 포함된다.

Base bank128, Past bank128을 첫 비교의 예산으로 유지한다. 별도 BaseAudit128 및 PastAudit128을 둔다. Sampling은 fact/version hash로 정하고 모든 arm이 공유한다. 특정 방법의 손상 순위로 주 bank를 재선택하지 않는다. Audit와 controller의 canonical뿐 아니라 rewrite/rephrase/neighborhood 전체 문자열 중복도 점검·보고한다. Bank 크기는 current B의 제한이 아니다.

**3. 위험과 Current 관측의 정의.** 모든 평균은 먼저 token, context, request 순으로 계산한다.

\[
F_B(W)=\mathbb E_{x\in\mathcal B}\operatorname{KL}(p_{We}(\cdot|x)\Vert p_W(\cdot|x)),
\]
\[
F_P(W)=\mathbb E_{i,c}\psi_\tau\{\ell_{ic}(W,y_i^{past})-\ell_{ic}(We,y_i^{past})\}.
\]

\(\psi_\tau(v)=0\) for v≤0, v²/(2τ) for 0<v<τ, v−τ/2 for v≥τ. 초기값 τ=0.1 nats/token. 이는 과거 지식의 악화에만 비용을 주는 C1 연속 함수다. KL과 단위가 맞도록 nats 단위를 유지한다.

Current의 평균 NLL을 LE라고 쓴다. 평균-gradient 진척 조건 외에 각 request/context의 NLL 변화와 출력 KL을 quadratic 비용에 반영한다. 성공률 100% 또는 평균 NLL 하나를 개별 편집 유지의 보장으로 쓰지 않는다. 실제 RS/PS, strict teacher-forced accuracy, 평균·중앙값·상위 꼬리 NLL, 문항별 성공→실패를 함께 보고한다.

Native 참조 위험은 rB=FB(WN), rP=FP(WN)이다. 위험의 정규화는 σj=max(rj,10^-3), f_j=F_j/σj로 고정한다. 작은 native 위험에서 비율만 폭증하는 문제를 피하며, 원 단위도 항상 남긴다. 이 scalar calibration 때문에 A도 native reference 계산 비용을 부담한다. A의 target 변수나 weight 방향에 D4N 또는 native L4 hidden z를 넣지는 않는다.

**4. Native 구조를 쓰는 범위와 history 준비.** 실제 N4·BLUE baseline은 기존 구현을 그대로 사용한다. 새 방법은 같은 covariance/history 정보를 사용하는 SPD projected metric과 writer를 쓴다. Praw를 직교화한 P*=QQᵀ의 차이 때문에 baseline과 byte-identical한 solver라고 부르지 않는다.

Layer ℓ의 averaged Current key를 Kℓ∈R^(din×B), native history Gram을 Mℓ, ridge를 λℓ>0으로 두고,

\[
S_\ell=K_\ell K_\ell^\top+M_\ell+\lambda_\ell I,
\quad H_\ell=Q_\ell^\top S_\ell Q_\ell,
\]
\[
\mathcal W_\ell(R_\ell)=R_\ell K_\ell^\top Q_\ell H_\ell^{-1}Q_\ell^\top
\]

로 둔다. Rℓ은 현재 layer의 module-coordinate target residual을 제어하는 RHS다. Native adapter의 module 좌표를 확인해 사용한다. Transformer block z와 down-projection output을 같은 벡터로 간주해 임의로 빼지 않는다.

새 보정 Cℓ은 Cℓ=CℓP*의 전체 허용 공간에서 구한다. 두 scalar layer coefficient 또는 Current key span 안에만 보정을 가두지 않는다. 특히 B에서는 Current key에 거의 작용하지 않으면서 Base prefix 경로를 바꾸는 방향이 필요할 수 있다.

기존 L4-only checkpoint에는 L8의 past history가 없을 수 있다. 이를 M8=0으로 대체하지 않는다. 해당 We에서 과거 승인 요청의 native context keys를 재수집해 L8 Gram을 만든다. 기준 N4의 M4는 원본을 유지하며, 새 L8 history가 현재 We 좌표에서 재구성됐다는 차이를 기록한다. 원형 BLUE를 같은 We에서 재실행할 때도 같은 준비된 M8을 사용한다. 다른 BLUE lifelong chain의 W50/M8을 가져와 현재 We와 섞지 않는다.

각 batch가 끝나면 선택 layer의 최종 실제 모델에서 history를 한 번 갱신한다. Inner target 최적화·OS·ODE node마다 현재 batch를 history에 반복 append하지 않는다. 기존 raw M의 대체된 fact 기여와 최신 ledger의 차이는 그대로 보고한다. 이번 설계가 기존 history를 완전히 정화했다고 주장하지 않는다.

**5. A의 핵심: 공동 local-z 계획 A0.** 두 개의 독립적인 full-target compute_z를 실행하지 않는다. 같은 We에서 R4=R8=0으로 시작하고, 하나의 Current loss로 두 변수를 함께 최적화한다.

\[
D_\ell^0=\mathcal W_\ell(R_\ell),\qquad
W(R)=We+\{D_4^0,D_8^0\},
\]
\[
\min_{R_4,R_8}\ L_E(W(R))+\lambda_{bal}\,\mathcal L_{bal}(D_4^0,D_8^0).
\]

가상 모델 W(R)의 **실제 weight**를 사용해 전체 forward를 실행한다. Subject 위치에만 activation을 주입한 결과로 loss를 대체하지 않는다. 두 weight가 모든 token에 적용되므로 L4 변경→L8 input 변경→최종 출력 변화가 target 계획 단계부터 포함된다. Writer map의 entry K/H는 여기서 고정된 parameterization이고, 실제 forward의 L8 activation은 고정하지 않는다.

Local z는 이 공동 계획에서 얻은 layer별 목표다. Requested residual Rℓ, native map이 실제로 실현한 DℓKℓ, 두 write를 함께 적용한 모델의 delivered local response를 따로 저장한다. Rℓ과 DℓKℓ가 같다고 가정하지 않는다. L8의 delivered response에는 L4의 영향도 포함된다.

계획은 Adam 25회, learning rate0.1, betas=(0.9,0.999), eps=10^-8을 첫 설정으로 둔다. 이는 전체 logical batch의 평균 gradient를 두 R에 함께 적용하는25회다. Request마다 L4 계획을 완성한 뒤 L8 계획을 하는 방식이 아니다. Native의 request별25-step compute_z와 호출 수·메모리·wall time이 같다고 주장하지 않는다. 두 방법의 실제 forward token 수를 별도로 센다. 주 native ridge는 현재 BLUE AlphaEdit의 λℓ=1을 유지한다.

Target planning 자체는 미분 가능한 writer map을 통한 최적화이므로 고정된 오른쪽 factor를 가진 weight 최적화와 대수적으로 연결된다. 이를 새로운 종류의 closed-form target optimizer라고 부르지 않는다. Closed form은 native write map과 아래 quadratic refinement에 적용된다.

**6. A0가 다시 L4로 몰리는 것을 다루는 방식.** 단순 weight norm 균등화 대신 초기 Current loss에 대한 기능적 효율로 action을 정규화한다. A0의 실제 방향 공간은 Vℓ=range(𝒲ℓ)이므로, 효율도 Vℓ에서 계산한다. Bℓ=∂vecDℓ/∂vecRℓ, gRℓ=Bℓᵀ∇Wℓ LE(We), TRℓ=Bℓᵀ𝒮ℓBℓ라 하면 qℓ=gRℓᵀTRℓ†gRℓ다. 𝒮ℓ는 weight-space Sℓ metric이다. 중복 Current key로 생기는 RHS null 방향은 pseudoinverse로 처리한다. 전체 P 공간의 효율을 대신 사용하면 실제 writer가 만들 수 없는 방향까지 분담 능력으로 계산하게 된다.

\[
\mathcal L_{bal}(D)=
\frac{1}{2\sigma_E^2}\sum_{\ell\in\{4,8\}}
q_\ell\,\|D_\ell\|_{S_\ell}^2,
\quad \sigma_E=\max(L_E(We),10^{-3}).
\]

첫 설정 λbal=0.1. q의 정확한 zero는 별도 표시하며, 정규화용 floor는 두 q 평균의10^-6와10^-12 중 큰 값으로 둔다. 이 floor는 layer 성능의 통과 기준이 아니다.

일차 모델에서 layer ℓ이 편집 진척 uℓ을 내는 최소 metric cost는 uℓ²/qℓ다. 위 정규화 후에는 uℓ²가 되어, 처음부터 민감도가 큰 L4가 모든 진척을 맡는 편향을 줄인다. 같은 총 진척이면 여러 layer의 분담에 유리한 soft 비용이다. 손상 위험이 다르면 배분은 달라질 수 있다. 총 update를 사후 unit norm으로 재확대하지 않는다.

50:50을 최종 성공 조건으로 요구하지 않는다. 실제 최적화가 L4에 몰리면 그대로 보고하고 부담 분산의 양성 결과로 세지 않는다. 이 경우를 숨기지 않기 위해 첫 Middle에서 λbal=0의 A0를 한 번 더 둔다. 균형 비용이 실제 편집 기여를 바꾸는지 직접 비교하며 넓은 λ sweep은 하지 않는다.

**7. A-OS: 공동 계획을 실제 지식 손상 기준으로 한 번 재배치.** A0의 실제 virtual endpoint를 WA=We+D0라고 둔다. 이 상태를 선형화 anchor로 사용하되, 최종 모델에는 D0+C를 함께 한 번 적용한다. A0를 실제 배포 모델에 먼저 commit할 필요는 없다.

Current의 reference는 WA의 출력이며 Base/Past reference는 여전히 We다. A0와 같은 편집 반응을 유지하면서 FB/FP를 줄이고, 필요하면 L4의 편집 기여를 L8로 옮기는 C4,C8를 함께 구한다. L4의 correction은 음의 방향도 허용한다. D4N을 고정하거나 D4를 native L4 전체 방향의 양수 배수로 제한하지 않는다.

이를 위해 아래 공통 quadratic에서 support={4,8}, anchor=WA, h=1을 사용한다. 추가 barrier를 넣지 않는 해가 A-OS다. A0→A-OS의 차이는 공동 z만의 효과와 기능적 손상 기반 재배치의 효과를 구분한다.

**8. B-OS: L4 native 편집을 고정한 기능적 보정.** WN에서 시작해 C4=0으로 두고 C8만 구한다. Current teacher는 pWN, Base teacher는 pWe이며 Past reference NLL도 We다. L8에 target_new를 다시 넣는 compute_z는 실행하지 않는다.

L8 보정의 local 목표는 다음 기능적 조건에서 역으로 유도한다: Current는 WN 반응 유지, Base는 We 반응에 가까워짐, Past의 추가 악화 감소. 계산된 C8에 대해 local response correction z8corr(x)=C8 k8WN(x)를 기록한다. 이 z는 Base/Past/Current 출력 목적에서 도출된 response correction이며, 새 정답을 다시 학습한 독립적인 full edit z가 아니다.

Base 지식을 다른 layer의 hidden value로 직접 복사하지 않는다. We와 WN은 L8 input부터 다를 수 있다. 같은 입력 문장의 최종 출력 분포를 연결하는 functional derivative로 C8을 구한다.

B-OS의 anchor=WN, support={8}, h=1이다. 최종 저장 모델은 WN+C8. L4 weight는 WN4와 byte-identical하게 유지한다. 한 batch 안에서 L8 down-projection만 추가로 바꾸면 고정 token sequence의 L8 input key는 고정된다. A보다 geometry가 단순한 이유다. 다만 다음 batch의 L4 write가 과거 보정이 의존한 L8 input을 바꿀 수 있다는 장기 문제가 남는다.

**9. OS와 barrier가 공유하는 functional quadratic.** Anchor b에서 C를 허용 layer의 weight correction을 연결한 벡터로 쓴다. Base/Past의 실제 gradient는 gb,gp다. Current의 실제 평균 NLL gradient는 a, request/context별 NLL Jacobian은 GE다. Gradient와 Jacobian은 예측 위치뿐 아니라 그 출력에 영향을 주는 모든 token의 weight 적용을 미분한다.

Current 보호 curvature FE는 p_b를 teacher로 하는 출력 KL의 generalized Gauss–Newton 항과 GEᵀGE의 합이다. 후자는 per-context 정규화 NLL 변화를 soft하게 억제한다. Base curvature FB는 KL의 generalized Gauss–Newton 항이다. Past curvature FP는 위 ψ∘NLL의 PSD generalized Gauss–Newton 항이다. ψ의 비활성 영역에서는 해당 risk curvature가 0일 수 있고 native ridge가 SPD를 유지한다.

Current profile의 원래 비용은 request/context 평균 KL(pref||pW)/τE와 같은 context의 NLL reference 차이 제곱/(2τE²)의 합이며 τE=0.1로 둔다. GE의 각 row에는 sqrt(request/context weight)/τE가 포함된다. OS에서는 reference=b라 profile gradient가0이다. BF에서는 고정 reference 경로에 대한 실제 profile gradient도 γE/h를 곱해 u에 더한다. 모든 출력 teacher는 response 위치에서 full vocabulary를 사용하며 raw per-context 항을 더한 뒤 전체 B로만 한 번 나누는 오류를 피한다.

예를 들어 Base logit Jacobian Jℓ에 대해,

\[
F_{B,48}=J_{B,4}^\top\{\operatorname{diag}(p_b)-p_bp_b^\top\}J_{B,8}.
\]

이 cross block을 0으로 놓지 않는다. Native preconditioner가 block diagonal이어도 최종 functional quadratic은 joint다. Full Jacobian을 저장하지 않고 JVP/VJP로 matrix action을 구한다. 서로 다른 layer의 residual을 같은 좌표라고 보고 직접 더하지 않는다.

정규화한 φ=2fB+fP와 Current penalty를 사용한다. Native metric \(\mathcal H\)는 layer별 S를 평균 고유값으로 나눈 block metric이다. τH=0.01, γE=1을 첫 설정으로 둔다. 모든 normalization과 gradient norm을 기록한다.

\[
\min_C\ \frac{\tau_H}{2h}\|C\|_{\mathcal H}^2
+g_\phi^\top C+\tfrac12 C^\top F_\phi C
+\frac{\gamma_E}{2h}C^\top F_E C
+\lambda_{bal}\mathcal L_{bal}(D^{cum}+C)
\quad\text{s.t. } a^\top C=t_E.
\]

마지막 balance 항은 A에만 있다. Dcum은 이번 batch 시작 We로부터 anchor b까지의 누적 delta다. B에서는 이 항을 제거하고 L4를 고정한다. Base 우선은 정규화한 Base risk의 가중치2, Past1 및 아래 slack 비용으로 표현한다. Base 때문에 Current/Past를 무조건 희생하는 사전식 최적화는 아니다.

OS에서는 tE=0이다. BF에서는 아래 고정 reference 경로와 비교하여 tE=−max{LE(b)−LE(reference_(s+h)),0}로 둔다. 이전 node의 실제 평균 NLL 악화를 다음 node의 새 기준에 흡수하지 않고 일차적으로 회복시키는 항이다. Current profile 비용 역시 reference에 대한 KL/NLL 차이로 선형화한다. 이를 1/2 CᵀKC+uᵀC로 쓰면 K는 허용 공간에서 SPD이고,

\[
C_u=-K^{-1}u,\quad
C_{eq}=C_u+K^{-1}a\frac{t_E-a^\top C_u}{a^\top K^{-1}a}.
\]

OS의 tE=0일 때 Ceq를 COS라 부른다. a=0이면 scalar 조건을 비우되 FE를 유지한다. 이때 tE≠0이면 일차적으로 회복할 수 없는 Current deficit으로 기록하며 equality가 성립했다고 보고하지 않는다. 거의 0인 a를 임의 clipping한 뒤 진척이 보존됐다고 보고하지 않는다. Whitened row normalization과 실제 residual을 기록한다.

이 식은 **선형화한 평균 진척**의 보존이다. 개별 요청이나 유한 step의 실제 편집 성능을 보장하지 않는다. FE가 이를 soft하게 보완하며, 최종 판단은 실제 Current 결과로 한다. 두 layer 각각에 aℓᵀCℓ=0을 요구하지 않는다. 그 조건은 L4의 편집 기여를 줄이고 L8로 옮기는 방향을 차단하기 때문이다.

Closed form은 이 고정 quadratic의 해를 뜻한다. K inverse는 명시적으로 만들지 않는다. Matrix-free PCG가 필요하면 선형계 반복 수·오차·시간을 모두 센다. 한 번의 모델 write, 한 번의 quadratic, 한 번의 forward/backward는 서로 다른 단위다.

**10. A-BF와 B-BF: predictor–corrector 경로 및 두 barrier.** 주 비교는 Kstep=4, h=1/4, s=k/4다. OS는 barrier 없는 endpoint quadratic이고, BF는 같은 functional objective에 상태 feedback과 두 elastic 위험 조건을 더한다. 엄밀한 continuous-time 안전 보장을 주장하지 않는 이산 경로 실험이다.

A: Wstart=We, nominal velocity=D0. Node k의 predictor b=Wk+hD0를 두 layer에 함께 적용해 관측한다. B: Wstart=WN, nominal velocity=0. Node k의 predictor b=Wk이며 L8 보정 방향을 계산한다. 두 방법 모두 같은 총 correction horizon1을 쓴다. h가 줄어들면 proximal 비용에1/h를 적용하여 네 번의 full OS 보정을 그대로 더하지 않는다.

각 predictor에서 실제 Current/Base/Past 출력, gradient와 functional curvature를 다시 구한다. 시각s의 node가 다음 predictor를 만들 때 Current teacher는 A의 경우 We+(s+h)D0, B는 항상 WN이다. 그 reference와 predictor 사이의 누적 Current KL/NLL deviation도 quadratic의 선형항·curvature에 넣는다. 따라서 이전 node의 편집 손실을 새 teacher에 흡수해 계속 허용하지 않는다. OS에서는 anchor와 Current teacher가 같아 이 추가 선형항이0이다.

Barrier reference는 OS endpoint 위험으로 재설정하지 않고 공통 native WN 위험을 사용한다. 정규화 전 budget은 다음과 같다.

|위험|A의 budget, 시각s|B의 budget, 시각s|
|---|---|---|
|Base|0.5 s² FB(WN)|(1−0.5s) FB(WN)|
|Past|s FP(WN)|FP(WN)|

두 방법의 최종 목표는 Base native 위험의 절반, Past native 위험 이하로 같다. 50%는 보장이나 통과 기준이 아니라 이번 최소-action controller에 요구하는 **elastic 목표**다. Native 위험0도 그대로 정의되며 slack으로 미달을 기록한다. OS보다10%만 좋아지면 되는 목표를 다시 사용하지 않는다.

Node의 quadratic에 다음 두 조건만 더한다.

\[
f_j(b)+g_j^\top C\le \beta_j(s+h)+\xi_j,
\quad\xi_j\ge0,\quad j\in\{B,P\},
\]
\[
\text{slack cost}=\sum_j\frac{\rho_j}{2h}\xi_j^2,
\qquad \rho_B=2,\ \rho_P=1.
\]

Budget도 σj로 나눈 값이 βj다. Current scalar 조건과 soft FE는 OS와 동일하다. 위험별 gradient와 ξ를 따로 남기고 layer별 barrier를 추가하지 않는다. Risk linearization의 오차 때문에 실제 endpoint가 budget을 넘을 수 있으며 predicted slack과 actual violation을 모두 보고한다. 성능 기준 미달로 endpoint를 삭제하거나 native로 몰래 교체하지 않는다.

**11. 두 barrier의 joint solve와 수치 구현.** OS의 equality를 반영한 inverse를

\[
P_K=K^{-1}-\frac{K^{-1}aa^\top K^{-1}}{a^\top K^{-1}a}
\]

로 둔다. a=0이면 PK=K^-1이다. G=[∇fB,∇fP]는 σ로 정규화한 위험의 gradient를 column으로 가지며, v=f(b)+GᵀCeq−β라고 하면,

\[
\nu^*=\arg\min_{\nu\ge0}
\tfrac12\nu^\top\{G^\top P_KG+h\operatorname{diag}(1/\rho)\}\nu-v^\top\nu,
\]
\[
C_{BF}=C_{eq}-P_KG\nu^*,\qquad \xi_j=h\nu_j^*/\rho_j.
\]

2차원 nonnegative quadratic은 네 active set을 직접 비교할 수 있다. Base를 먼저 project한 뒤 Past를 project하는 순서 의존 처리를 하지 않는다. 이전 v2의 잘못된 dual tolerance를 가진 기본 runtime을 복사하지 않는다. 수리된 solver의 nonnegative/KKT 규칙을 이 수식에 맞춰 확인한다.

한 node에서는 같은 b에서 두 layer의 correction을 함께 구하고 함께 반영한다. L4 correction을 먼저 확정한 뒤 L8이 다시 새 정답을 쫓는 순서를 도입하지 않는다. 순서대로 tensor를 저장하더라도 사이에 model forward·재최적화·history append가 없어야 한다.

**12. ODE에서 고정하는 것과 갱신하는 것.** 주 BF는 A0 target과 nominal D0, Base/Past/Current reference, We에서 준비한 native metric을 batch 동안 고정한다. 매 node 갱신하는 것은 실제 모델 출력과 functional gradient/curvature다. A의 L8 input은 실제 forward에서 현재 L4 변화에 따라 다시 계산된다. Fixed native metric을 쓴다는 이유로 L8 input까지 entry 값으로 바꾸면 안 된다.

이 선택은 step마다 두 full compute_z를 다시 수행하는 비용과 목표 변경을 피하면서, Base 손상이 커지는 방향을 current-preserving joint correction으로 L4↔L8 사이에서 옮기게 한다. Native metric 자체를 current keys로 다시 factor하는 효과는 이번 주 비교에서 섞지 않는다.

Frozen-BF4는 첫 predictor의 functional gradient/curvature를 고정하고 나머지 nominal increment·위험 측정·budget·step 수를 동일하게 둔다. 이는 현재 state의 derivative 갱신 효과를 보는 대조다. 위험값과 실제 모델 평가까지 고정하지 않는다. BF4와 비교할 때 추가 backward 비용을 함께 표시한다.

Frozen에서 scalar tE와 두 위험의 RHS는 실제 값으로 갱신하지만 a, profile gradient, risk gradient, functional curvature는 첫 predictor의 값이다. A의 balance 항은 알려진 native quadratic이므로 실제 누적 delta에서 계산한다. Frozen 결과의 Current 유지가 나쁘면 그 차이도 원인 해석에 포함한다.

**13. 실제 분산과 보정을 구분하는 측정.** 최종 delta에 대해 네 실제 모델 We, We+D4, We+D8, We+D4+D8을 평가한다. 이때 We에 이미 있던 이전 batch의 모든 update는 그대로 둔다. Removing-layer 평가는 이번 batch delta만 제거한다.

각 request의 E(S)=LE(We)−LE(We+D_S)로 정의하고,

\[
e_4=\tfrac12\{E(4)+E(4,8)-E(8)\},\quad
e_8=\tfrac12\{E(8)+E(4,8)-E(4)\},
\]

\(I_E=E(4,8)-E(4)-E(8)\)를 보고한다. e4+e8은 총 NLL 개선과 정확히 같다. 이는 선택한 두 block intervention에 대한 대칭 attribution이며 fact가 해당 layer에 저장됐다는 증명은 아니다. 총 개선이0/음수이면 기여율을 안정적인 양의 비율처럼 보고하지 않고 signed 원값을 쓴다.

|판단할 기전|필요한 관측|
|A의 부담 분산|L4의 functional edit 기여와 native-metric action이 줄고, L8이 Current 진척의 일부를 담당하며 전체 Current가 유지됨|
|B의 손상 보정|이번 batch의 L4 delta는 D4N과 같고, L8 correction 제거 시 Current는 대체로 유지되면서 Base/Past 이득이 사라짐|
|재편집에 가까운 B|L8 제거 시 Current도 크게 무너짐; 순수 보정 대신 보조 편집 또는 혼합 기전으로 기록|
|단순 약화|Current NLL·rephrase가 악화되는 만큼 Base가 개선됨; 보존 우위로 단정하지 않음|
|표본 적합|controller bank는 개선되나 별도 Audit에는 개선이 없음|

두 기전이 연속적으로 섞일 수 있으므로 임의의20%/50% 기준으로 결과를 잘라내지 않는다. Layer norm share, signed edit attribution, Base/Past 개선, interaction을 연속값으로 함께 읽는다. ODE에서는 node별 gEℓ·ΔWℓ와 gBℓ·ΔWℓ, 실제 변화의 차이도 저장하여 편집 기여의 이동 시점을 확인한다.

B에서는 WN+C8에서 C8만 제거한 모델이 같은 batch의 WN과 일치해야 한다. 이것은 기전 비교를 위한 상태 identity이며 성능 gate가 아니다. A에서는 native full L4를 먼저 commit하지 않았다는 target/weight 상태 로그가 필요하다.

**14. 최소 실험 구성.** 첫 주 layer set은 L4+L8이다. 모든 arm은 같은 Current requests, We, Base/Past/Audit manifest를 쓴다. 기존 다른 lifelong chain의 endpoint를 동일-entry 비교로 대체하지 않는다.

|arm|확인할 질문|
|N4: 원형 BLUE-L4|강한 단일층 native 기준|
|BLUE: 원형 L4→L8|기존 순차 local-z 방식과의 차이|
|A0: 공동 local-z만|전체 L4 target 선소진을 없앤 효과|
|A-OS|공동 편집의 functional 재배치 효과|
|A-BF4|실제 경로의 두 위험을 보며 추가로 재배치하는 효과|
|B-OS|고정 L4 편집을 한 번의 L8 보정으로 보호할 수 있는가|
|B-BF4|L8 보정을 나누고 기능적 손상을 갱신할 가치가 있는가|

Early/Middle/Late의 기존 세 entry에서 위7 arm을 비교한다. Middle에만 A0의 λbal=0, A0와 같은 writer-aware target optimizer를 L4만 허용하여 실행한 A0-L4, A/B의 Frozen-BF4, N4에서 L4만 허용한 functional OS를 추가한다. A0-L4는 batch joint target 최적화 자체의 효과와 추가 layer의 효과를 구분한다. 마지막 대조는 B의 L8 추가 공간 효과와 functional 보호식 자체의 개선을 구분한다. Core21개와 Middle 추가5개로 총26 endpoint이며, 공유 가능한 native/target 자산은 재사용 여부를 표시한다. 모두 독립 endpoint이며 조합별10k chain을 즉시 실행하지 않는다.

OS↔BF의 차이에는 barrier 유무와 경로 분할이 함께 있다. 따라서 필요하면 Middle의 BF1을 추가하여 barrier 자체의 효과를 분리할 수 있지만, BF4의 모든 이득을 step 수의 효과라고 먼저 주장하지 않는다. Frozen-BF4는 상태 derivative 갱신의 더 직접적인 대조다.

별도 장기 질문은 동일 Middle entry에서 이후10 batches를 각 방법 자체의 state로 이어가는 N4/BLUE/A-BF4/B-BF4 short chain으로 확인한다. 바로 전 batch의 We teacher를 매번 새로 만들고 W0는 평가용으로 유지한다. A의 layer 조합 의존성과 B의 과거 보정이 후속 L4 write로 무너지는지를 본다. 이 short chain도10k lifelong 우위의 증명은 아니다.

**15. B에 구애받지 않는 구현과 검증.** B=0은 no-op, B≥1은 동일 식을 쓴다. Native proposal의 RHS column 수가 B에 맞춰 늘며 rank100/256 절단을 두지 않는다. 현재 proposal을 두 layer coefficient로 압축하지 않는다. Full-space correction도 B가 커졌다고 강제 zero가 되는 exact-all-current-key-null을 요구하지 않는다.

Logical objective는 request 평균이다. Native writer가 raw sum을 사용하면 mean으로 바꿀 때 M/λ까지 동일 비율로 변환해야 같은 operator가 된다. Current KKᵀ만 B로 나누고 history/ridge를 그대로 두는 숨은 변경을 금지한다. Native 방식 자체의 batch partition 의존성까지 제거됐다고 주장하지 않는다.

Physical microbatch는 같은 logical batch의 gradient accumulation에만 사용한다. 모든 Current·Base·Past loss의 정규화를 logical 분모로 고정한 뒤 optimizer update/solve를 한 번 한다. Chunk마다 optimizer 또는 M append를 하면 다른 알고리즘이다.

Middle의 고정100 요청 안에서 nested B1/B7/B100을 사용하고 보호 bank는 가능한 한 동일하게 고정한다. 이들은 요청 수에 따른 관측이며 성능 불변성의 증명은 아니다. 별도로 동일 logical B100의 microbatch1/2/4에서 목적식·gradient·최종 delta의 수치 차이를 확인한다. 앞으로 동일 request stream을 서로 다른 logical B로 나눈 실험은 cumulative 비교로 명시하며 microbatch 검증과 구분한다.

**16. 계산량을 숨기지 않는 실행 예산.** 이 설계의 가장 큰 추가 비용은 layer 수보다 functional curvature action과 공동 target의 실제 weight forward다. Closed form이라는 명칭으로 이 비용을 제외하지 않는다.

|단계|A|B|
|기준 native 계산|공통 위험 calibration용 N4; 비용 포함|실제 편집을 위한 N4|
|추가 target 계획|두 R를 함께 최적화하는25회; Current만 사용|독립적인 target_new용 L8 compute_z 없음|
|One-shot 보정|한 anchor에서 Current/Base/Past gradient 및 implicit curvature solve|같은 계산을 L8 support에서 수행|
|BF4|4개 predictor의 derivative/solve; A0 계획 재사용|4개 L8 보정 derivative/solve|
|실제 target 계산 재시작|node별0|node별0|
|History 준비|L8 재구성·최종 append|보정 metric용 L8 준비·최종 append|

두 selected layer의 gradient는 같은 backward graph에서 함께 얻을 수 있다. 하지만 한 번의 전체 logical pass도 여러 microbatch backward를 포함하며, JVP/VJP matvec과 PCG 반복은 별도다. Target25회를 단순히25개 backward라고 집계하지 않는다.

우선 Middle의 A0, A-OS, B-OS에서 model load와 평가를 뺀 edit-core timer를 계측한다. PCG는 relative residual10^-4 목표, 최대20회로 최초 예산을 고정하고 실제 residual을 보고한다. 상한에서 미수렴한 결과는 수치 근사 결과로 별도 표시하며 그 실패로 capacity 부족을 결론내리지 않는다. 과도한 비용을 피하려고 functional cross block이나 Current 조건을 조용히 삭제하지 않는다. 비용 때문에 근사 operator를 택하면 별도 arm 이름과 근사 오차가 필요하다.

20회는 RHS당 상한이다. 일반적으로 OS에는 u/a의2 RHS, BF에는 u/a/∇fB/∇fP의4 RHS가 필요하다. 따라서 단순한 별도 solve 구현은 OS 최대40, BF4 전체 최대320회의 curvature action까지 갈 수 있다. 실제 구현은 block solve·공유 Krylov 계산을 검토하되 batched JVP도 처리한 vector 수를 숨기지 않는다. 이것이 현재 설계의 큰 계산상 부담이며, 작은2차원 dual solve가 전체 방법을 싸게 만들지는 않는다. 첫 Middle 측정에서 이 비용을 먼저 공개하고 cold native 대비 효율을 판단한다.

이 모델의4096×14336 weight delta는 FP32 약224MiB/layer다. 두 layer의 Krylov vector 하나만 약448MiB이며 여러 vector·factor·activation이 추가된다. Full functional Hessian은 만들지 않는다. 메모리 부족은 microbatch 축소와 host offload로 대응하고 논리적 B나 보호 문항을 몰래 줄이지 않는다.

필수 ledger: model forward tokens; Current/Base/Past backward microbatches; JVP/VJP 수; PCG/KKT solve 수와 residual; teacher/key capture; native/joint-z 시간; protection 시간; history refresh 시간; endpoint 평가 시간; peak allocated/reserved memory; host offload; cache 사용 여부; 총 GPU seconds. Prepared/cached 결과는 cold end-to-end 시간과 분리한다.

**17. BLUE·CAKE와의 차이를 주장하는 범위.** BLUE와 구별되는 A의 조건은 공동 We에서 출발한 writer-aware local targets, 공통 Current objective, 두 layer 동시 결정, Base/Past 기능적 비용에 따른 이후 재배치다. B의 조건은 L4 native weight 고정과 L8의 preservation-only target 정의다. 단순히두 layer를 쓰고 locality가 높다는 결과로 차이를 주장하지 않는다.

CAKE의 §3.2/Algorithm1은 causal weight를 사용해 현재 residual을 남은 layer에 비례 배분하고 순차 write 후 residual을 갱신한다. 실용 알고리즘은 sensitivity를 identity로 근사한다. 따라서 가중 residual 배분 또는 closed-form allocation만을 새로운 기여로 내세울 수 없다. 이번 A는 실제 joint weight forward와 Current/Base/Past 기능적 반응을, B는 edit를 고정한 별도 보정 목적을 검증한다. 이는 설계상 차이이며 CAKE보다 낫다는 실험 결과는 아니다. [CAKE 원문, §3.2](https://aclanthology.org/2026.acl-long.918.pdf).

**18. 결과를 해석하는 방식.** A가 성공하면 공동 편집 target을 먼저 계획한 뒤, Current 진척을 유지하는 joint 공간에서 위험이 큰 기여를 옮길 수 있다는 근거다. B가 성공하면 이미 성립한 편집을 담당하는 weight를 유지하면서 다른 layer에서 일부 손상을 보정할 수 있다는 근거다. 둘 다 성공하면 비용·BaseAudit 일반화·후속 edit 내구성을 비교한다. A만 성공하면 사후 복구보다 초기 배분의 가치가, B만 성공하면 새 편집을 위한 분산보다 보정 공간의 가치가 드러난다.

One-shot이 충분하면 ODE의 추가 비용을 정당화하지 않는다. BF가 bank에서만 좋아지면 광범위한 Base 보존으로 보고하지 않는다. Base 목표 미달, Current 약화, layer 기여 집중, cancellation 의존성, 수치 오차는 각각 관측으로 남긴다. 전체 결과를 통과/실패로 줄이는 추가 scientific gate는 두지 않는다.

**수식 확인 범위.** 작은 CPU 합성 문제120개에서 affine Current 조건과 두 elastic barrier의 네 active set, primal/KKT 조건을 확인했다. 서로 민감도가 다른 두 writer image에서도 q-normalized action의 일차 최소값이 동일 진척을 분담하는지 확인했다. 최대 잔차는 약1.3×10^-14였다. 모델 성능·근사 Jacobian·GPU 비용을 검증한 결과는 아니다. [계산 기록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-11-multilayer-joint-edit-and-compensation-design-math-check.json).
