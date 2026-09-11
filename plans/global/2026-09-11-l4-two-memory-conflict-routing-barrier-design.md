# L4 안에서 과거 편집·Base 지식의 추가 손상을 줄이는 우회 편집

작성·개정: 2026-09-11 KST. 버전: **v2-base-preservation**. 상태: **리뷰 반영 설계 확정 — 모델 구현·GPU 실험 미실행**. 앞선 481행 초안의 Base 복구 항, raw M 혼합, step별 보정 비용 정의를 아래 최종 정의로 대체한다.

이번 결정에서 'Base 보존 우선'은 **이번 batch가 시작하기 직전 We의 Base 반응에 추가 손상을 주지 않는 것**이다. W0로 기존 누적 손상을 복구하는 항은 주 최적화에서 제거한다. W0 대비 손상·복구는 별도 평가한다. Current/Past를 무조건 희생하는 사전식 Base 우선순위를 새로 도입한 것은 아니다.

## 1. 두 주 방법과 범위

현재 편집과 충돌하는 neuron을 고정적으로 삭제하는 대신, **현재 편집의 scalar 목표 반응 진척을 유지하고 context별 반응 변경에 비용을 주면서 과거 편집과 원래 지식에 비용이 큰 write 방향을 다른 방향으로 옮긴다.** 한 layer인 L4 down projection을 유지한다.

사용자의 추가 요구를 반영해 다음 두 가지를 독립적인 주 방법으로 둔다.

- **OS: barrier 없는 one-shot conflict routing.** Native proposal과 고정된 두 보호 통계로 하나의 quadratic 문제를 풀고 weight를 한 번 확정한다. Barrier, stepwise feedback, accepted-prefix 선택을 사용하지 않는다.
- **BF: OS 경로에 실제 output feedback과 두 elastic barrier를 추가.** We에서 OS endpoint 방향으로 진행하며 과거 편집의 context별 NLL 악화와 Base의 We 대비 출력 변화를 각각 관측하고 보정한다. 단회 BF1과 반복 BF8을 구분한다.

OS와 BF 모두 logical batch 크기 B에 대한 100개 제한, 4개 균등 group, 고정 Ub rank100을 갖지 않는다. B=0은 no-op, B≥1은 동일 수식과 API로 처리한다. 물리적 microbatch는 메모리에 맞춰 분할하지만 logical batch의 목적식은 바꾸지 않는다.

**모든 현재 key의 정확한 반응을 고정하는 hard-null 방식은 small-batch 기전 진단으로만 둔다.** B가 커지면 그 조건이 우회 공간을 없앨 수 있기 때문이다. 범용 주 방법은 P의 전체 허용 공간에서 current response 변경을 quadratic 비용으로 억제하고, 목표 반응의 scalar 진척 하나를 보존한다. 이는 요청별 실제 NLL을 모두 보장한다는 뜻이 아니다.

'임의 batch 크기에서 정의·실행 가능'과 '어떤 크기에서도 동일 성능·무손상 보장'은 다른 주장이다. 전자는 설계 조건, 후자는 유한 layer capacity 때문에 일반적으로 보장할 수 없다. 다중 layer, z refresh, 성능 gate/rollback, 광범위한 parameter sweep은 이번 설계에 넣지 않는다.

보호 주장의 강도는 **scalar 목표 진척 + response-change penalty**로 한정한다. 현재 편집의 실제 NLL은 BF의 방향 계산에 직접 들어가지 않으며, Current 실제 성능과 이 조건의 불일치가 첫 실험의 핵심 결과다. 새로운 Current hard gate나 model backward를 추가하지 않는다.

## 2. 근거와 이번 질문

9월 11일 기존 실험에서 Early/Late EP는 0/8회, Middle은 첫 step 1/8회만 활성화됐다. H의 24개 주 step 중 23개에서는 covariance risk가 이미 감소했다. 유일 활성 지점의 q/q_all은 0.998868이며 correction/nominal native norm은 0.576%였다. 이것은 선택한 평균-edit/covariance 관측에서 공간이 부족했다는 증거가 아니다.

기존 risk cap은 R_C(WN)이었으므로 native write까지 발생한 손상을 허용했다. 또한 Middle의 EP−H 일차 예측은 이미 95개 요청 악화와 5개 개선을 평균으로 상쇄했다. 실제 92개 악화는 H와 비교한 작은 차이이며 WN 대비 92개가 붕괴했다는 뜻은 아니다.

이번 질문은 다음과 같다.

> 같은 L4에서 BLUE native의 현재 batch scalar 목표 반응 진척을 유지하고 반응 변형에 비용을 주면서, 과거 편집 및 원래 지식에 대한 기능적 손상을 더 적게 만드는 write를 찾을 수 있는가? 정적 closed-form 재배치보다 실제 output feedback이 추가로 유용한가?

이는 single-layer 전체 capacity가 충분하다는 가정이 아니다. 지정된 입력 공간과 보호 bank에서의 우회 가능성을 먼저 측정한다.

## 3. layer 안의 '위치'를 정의한다

L4 down projection을 y=Wk, W∈R^(4096×14336)로 쓴다. 여기서 k는 MLP intermediate activation이다.

- weight 좌표/개별 neuron: 사람이 볼 수 있는 attribution 좌표지만, 하나의 fact가 한 좌표에 저장된다고 가정하지 않는다.
- 입력 방향: 어떤 key 또는 key 조합에 write가 작용하는가. M/C0와 subspace 분석이 다룬다.
- 입력×출력 방향: 같은 입력 방향이라도 출력 방향에 따라 edit와 locality에 기여하는 정도가 다르다. 실제 loss gradient가 다룬다.

따라서 두 종류의 '충돌'을 구분한다.

1. **구조적 중첩:** 현재 write가 과거/원래 key에 얼마나 크게 작용하는가.
2. **기능적 충돌:** 그 변화가 해당 지식의 실제 정답 NLL 또는 원래 출력 분포를 얼마나 악화시키는가.

key cosine이나 M의 큰 고유값만으로 두 fact의 요구 출력이 모순이라고 판정하지 않는다. 최종 충돌 지도는 (보호 요청, 현재 write의 입력 mode, 예상 변화, 실제 output 변화)를 함께 보여준다.

## 4. 보호 기준과 유효한 과거 사실

- W0: 원래 pretrained 모델의 L4 weight. **누적 손상 평가용**이다.
- We: 현재 batch 직전 L4 weight. OS/BF의 **Base 보존 reference**다.
- WN: 같은 We에서 BLUE-L4 native가 계산한 endpoint.
- ΔN=WN−We: FP64로 올린 WN/We의 차분. Native endpoint identity를 유지한다.
- M_native: 기존 native writer의 raw 누적 Gram. Baseline/proposal용이며 routing용 Cp와 구분한다.
- C0: pretrained L4 input second moment. Broad 구조 진단용이며 named Base bank의 Cb와 구분한다.

Past는 과거에 승인된 최신 target_new를 보호한다. 목표값은 편집된 사실이고, 악화량의 기준 loss는 We다. 이미 잊힌 과거 지식을 모두 복구하라는 목표는 넣지 않는다.

Base는 현재/과거에 명시적으로 교체한 사실을 제외한 지식 요청이다. OS에서는 We Kb, BF에서는 p_We를 teacher로 사용한다. **현재 유지된 Base 반응을 보존**하며, We가 이미 틀린 답을 내는 경우 그 오류까지 유지할 수 있다는 한계를 W0/정답 평가로 드러낸다.

과거 요청의 active-fact ledger는 (subject identity, relation identity)별 최신 승인 버전을 보관한다. CounterFact에서는 relation_id와 정규화된 subject identity를 사용하고, label·원문·version ordinal·해시를 함께 저장한다. 정규화는 Unicode NFC와 기존 tokenizer의 공백 규약에 한정하며 임의 semantic alias 병합은 하지 않는다.

현재 batch가 다루는 fact identity는 Past 보호 집합에서 제외하고 Current로만 평가한다. 과거의 대체된 version도 보호 bank에 넣지 않는다. Base에는 승인된 replacement의 원래 target을 넣지 않는다. 관련 지식 전체를 폭넓게 제외하지 않는다. 동일 batch 안의 상충 target은 데이터의 명시적 순서에 따른 latest-wins를 모든 arm에 공통 적용하고 raw B와 effective B를 기록한다. 이 관계가 없는 semantic 충돌까지 해결됐다고 주장하지 않는다.

## 5. Raw M의 잔류 문제를 처리하는 규칙

M_native는 과거 request별 context 평균 key의 Σkkᵀ다. Clean .5, generated prefix 다섯 개는 각각 .1이며, 개별 context Gram의 평균이 아니다. M_native에는 개별 fact identity/target/version이 없고 대체된 fact의 기여도 남을 수 있다.

**최종 routing Cp에서는 raw M 혼합을 제거한다.** Cp=Kp Ωp Kpᵀ를 active-fact bank에서만 구성한다. 이렇게 하면 Past loss와 routing penalty의 보존 대상이 일치한다. 현재 key를 M에서 임의로 빼거나, key provenance 없이 오래된 사실의 기여를 제거했다고 주장하지 않는다.

M_native는 기존 baseline과 native proposal 계산에 원형대로 남긴다. 따라서 오래된 history의 영향이 native proposal 및 scalar 진척 기준에 간접적으로 남을 수 있다. 이번 변경은 **새 routing penalty의 대상 불일치**를 해소하며 native writer 전체에서 과거 이력을 정화한 것은 아니다. N→OS 해석에 이 범위를 포함한다.

이 선택의 대가는 전체 과거 Gram을 routing 비용에 직접 넣지 않고 표본 Past bank에 의존한다는 것이다. 기존 Fixed/Past held-out 평가와 active-fact 수·coverage를 보고한다. 향후 전체 active-history 통계를 쓰려면 버전별 key ledger로 재구성하고 그 비용을 별도로 측정해야 한다. 이번 실행에 자동 추가하지 않는다.

Single L4 down projection만 수정하고 teacher-forced token sequence를 고정하면, 해당 module의 입력은 고정된다. 현재6B keys와 Past/Base의 보호 위치 keys를 한 번 capture해 모든 arm/step에 재사용한다. 다른 token을 생성하는 자유 생성 경로까지 고정된다는 뜻은 아니다.

## 6. 임의 B에서의 current 통계와 진척 정의

각 request i의 native averaged key를 k̄_i, 기존 BLUE-L4 writer가 사용하는 module-coordinate residual을 r_i라고 둔다. r_i는 기존 adapter에서 직접 받는다. z가 어떤 layer/module output을 뜻하는지 확인하지 않고 r_i=z_i−We k̄_i라고 새로 정의하지 않는다.

\[
K=[\bar k_1,\ldots,\bar k_B],\quad R=[r_1,\ldots,r_B],\quad
\Omega=I_B/B,\quad C_E=K\Omega K^\top,\quad T=R\Omega K^\top.
\]

개별 context 변경 비용은 별도 통계다.

\[
C_{resp}=\frac1B\sum_i\sum_c q_c k_{ic}k_{ic}^\top,
\qquad q_{clean}=.5,\quad q_{prefix}=.1.
\]

C_E는 평균 key의 Gram, C_resp는 context별 Gram 평균이다. 서로 바꿔 부르지 않는다. Native z 학습 loss의 여섯 context 균등 평균도 이 두 통계와 다르다.

\[
\operatorname{tr}(ZC_{resp}Z^\top)
=\frac1B\sum_{i,c}q_c\|Zk_{ic}\|^2.
\]

이 비용에서는 한 요청/context의 양의 변화가 다른 요청/context의 음의 변화와 상쇄되어 무료가 되지 않는다. 다만 soft penalty이므로 개별 변화의 hard bound는 아니다.

목표 반응 진척은 π(D)=〈T,D〉로 둔다. Scale을 표시할 때는 tr(RΩRᵀ)로 나누며0이면 나누지 않는다. Native proposal ΔN의 진척 a_N=〈T,ΔN〉를 기준으로 **추가 보정 Z에 〈T,Z〉=0**을 요구한다. a_N≠0이면 단순히 native write 전체를 축소해 두 보호 위험을 낮추는 해를 막을 수 있다. a_N=0에서는 이 축소 방지 효과가 없으며, 음수인 native 진척을 이 조건이 양수로 고쳐 주지는 않는다.

이 조건은 목표 module response 방향의 scalar projection이다. Actual edit NLL의 평균-gradient equality도 아니고 각 요청의 mapping equality도 아니다. 일부 요청 간 trade-off는 남으며 response-change penalty와 요청별 관측이 이를 보완한다. 음수/0인 native 진척도 숨기지 않는다.

Scalar 진척은 B=1에서도 완전한 반응을 고정하지 않는다. 목표 반응이(1,0)일 때(1,1)도 같은 target projection을 가질 수 있다. C_resp penalty가 이 추가 성분에 비용을 주지만 금지하지는 않는다. 따라서 scalar residual이 작다는 수치적 사실을 Current 성능 유지로 보고하지 않는다.

## 7. 허용 공간과 batch 증가에 대한 정의 가능성

주 방법의 보정은 Z=ZP, 즉 기존 AlphaEdit P의 허용 공간에서 계산한다. Ub의 rank를100 또는256으로 고정하지 않는다. P를 유지하는 이유와 그 근사성은 baseline과 함께 기록하며, 이것이 모든 원래 지식의 exact null space라고 주장하지 않는다.

A가 SPD일 때 허용 inverse operator를

\[
\mathcal T_A=Q_P(Q_P^\top A Q_P)^{-1}Q_P^\top
\]

로 정의한다. Q_P는 P 허용 공간의 orthonormal basis다. A=λ_r I+A_data, λ_r>0이면 동등하게

\[
\mathcal T_A=P[\lambda_r I+P A_{data}P]^{-1}P
\]

를 사용할 수 있다. 거대한 inverse를 명시적으로 만들지 않고 factorization/linear solve/operator action을 사용한다. 다음 규칙으로 저장 P의 근사성과 실제 solver 공간을 고정한다.

Current K의 column 수가 B이거나6B여도 ridge가 양수이므로 작은 B, 큰 B, 중복 key, collinear key에서 주 선형계는 정의된다. BF의 scalar progress projection에서 T의 허용 성분이0이면 Z=ZP의 모든 보정이 자동으로 해당 진척을 보존하므로 observer를 비운다. Approximate zero를 과학적 성공/실패 gate로 쓰지 않는다.

Batch가 커지면 current-response 비용이 더 많은 방향을 포함해 우회가 비싸질 수 있다. 이는 conditioning/공동 보존 여력의 문제이지 'B100만 지원'하는 구현 제한이 아니다.

실행용 직교 공간의 정의:

1. Native baseline은 저장 P_raw를 그대로 사용한다.
2. 새 routing은 S=(P_raw+P_rawᵀ)/2의 FP64 eigen decomposition에서 eigenvalue>1/2인 공간을 Qp로 정의하고, P*=Qp Qpᵀ를 사용한다. 이하 OS/BF 식의 P는 이 P*다. 1/2는 원래 C0의 null-space cutoff를 다시 정하는 값이 아니라, 저장 projector의 0/1 eigenspaces를 수치적으로 복원하는 기준이다.
3. P_raw/P*의 차이, rank, idempotence/orthogonality residual, 1/2에 가장 가까운 eigenvalue, source SHA를 저장한다. 기존 prepared Q는 같은 projector에 대한 것임이 확인된 경우만 재사용하고, 정의가 다른 tolerance의 basis를 그대로 exact 공간이라고 부르지 않는다.
4. 완전한 eigendecomposition 대신 작은 배제 공간을 구하는 계산을 사용해도 위 operator와 같은 공간을 얻었음을 확인한다. Full basis/eigensolve 비용은 최초 geometry ledger에 포함한다. 성능 결과에 따라 rank를 조절하지 않는다.
5. Scalar projection은 whitened T를 unit row로 정규화하여 계산한다. q는 squared norm으로 계산해 cancellation을 피하고, 정확한 zero row만 빈 observer로 둔다. Numerical solver tolerance는 FP64 relative1e−10을 목표로 하되 실제 residual을 보존한다. 이는 과학적 성능 gate가 아니다.

## 8. 충돌 지도와 선택적 낮은-rank 계산

M/C0와 named bank를 이용해 다음 두 지도를 만든다.

- Structural: current write가 보호 key/mode에 작용하는 크기, current/past/base projected Gram spectrum, subspace overlap.
- Functional: 같은 변화가 해당 보호 요청의 실제 NLL 또는 We teacher KL을 높이는지와 그 방향미분.

M/C0의 큰 eigenmode나 key cosine만 보고 출력 요구의 충돌을 확정하지 않는다. Native factor가 있으면 현재 request별 write와 보호 request별 gradient의 내적으로 attribution할 수 있고, factor가 없으면 입력 mode 수준에서 보고한다.

주 OS/BF는 **현재 B와 무관한 rank256 절단을 적용하지 않는다.** Matrix-free solve나 정확한 low-rank factor 재사용은 가능하다. 근사 basis를 사용한다면 native current-response span을 임의로256으로 잘라서는 안 되며, solve residual과 response error를 보고하고 그 근사를 별도의 방법으로 구분한다.

Conflict-based V256 및 random V256 비교는 §9의 small-batch 공간 진단에만 선택적으로 사용할 수 있다. 이번 범용 주 비교의 필수 구성은 아니다.

## 9. 정확한 key-response 우회의 의미와 한계: 보조 진단

Small-batch에서 K*=[개별6B context keys]와 ZK*=0을 요구하면 각 lookup의 native response를 정확하게 유지할 수 있다. P 허용 공간에서 그 projector는

\[
Q=P-PK_*(K_*^\top PK_*)^\dagger K_*^\top P.
\]

P와 일반 key-null projector를 단순 곱한 것은 이 projector와 다르다. 남은 rank는 rank(P)−rank(Q_PᵀK*)다. 큰 B에서0이 될 수 있다. 특히 Ub100 안에서 UbᵀK*가 row-rank100이면 Z=AUbᵀ, ZK*=0은 Z=0만 허용한다.

이 때문에 exact-key-null을 arbitrary-B 주 알고리즘의 필수 조건으로 두지 않는다. 보조 진단에서 공간이0이면 correction0이라는 사실 자체를 기록한다. 다른 batch로 바꾸거나 성능 결과를 버리지 않는다.

작은 예: current key=(1,1), 필요한 변화1, past=(1,0), base=(1,.2)라면 write(.5,.5)→(0,1)로 current 변화1을 유지하면서 past 변화 .5→0, base .6→.2로 낮출 수 있다. Norm은 .707→1로 커진다. 반대로 past=(1,0), base=(0,1) 모두 변화0과 current 변화1을 요구하면 불가능하다. 실제 모델의 지식이 이 좌표별로 저장됐다는 뜻은 아니다.

## 10. OS: Base의 신규 변화만 줄이는 one-shot 닫힌 해

Native proposal ΔN을 계산한 뒤 모델에 최종 적용하기 전에 보정 Z를 구한다. D=ΔN+Z, Z=ZP다. z/native residual은 다시 최적화하지 않는다. Native proposal과 OS의 linear solve 비용을 모두 센다. 최종 write 한 번이며 z 최적화가 사라진 방법은 아니다.

Cp=Kp Ωp Kpᵀ와 Cb=Kb Ωb Kbᵀ를 §11의 유효한 bank·보호 위치에서 구성한다. 두 통계의 local reference는 각각 **We Kp, We Kb**다. C0/global W0 drift는 주 목적식에 넣지 않는다.

\[
\begin{aligned}
\mathcal J_{pres}(D)=&\ \tfrac12\|(DK-R)\Omega^{1/2}\|_F^2
+\tfrac\gamma2\operatorname{tr}[(D-\Delta_N)C_{resp}(D-\Delta_N)^\top]\\
&+\tfrac{\lambda_p}2\operatorname{tr}(DC_pD^\top)
+\tfrac{\lambda_b}2\operatorname{tr}(DC_bD^\top)
+\tfrac{\lambda_r}2\|D\|_F^2.
\end{aligned}
\]

Base 항은 이번 batch가 We의 Base mapping에 만드는 변화다. W0 쪽으로 기존 손상을 되돌리라는 힘은 없다. Base에 유익한 변화도 구조 proxy에서는 비용이 될 수 있다. 이것은 보존을 택한 설계의 절충이며, 실제 W0/정답 개선은 별도 관측한다.

추가 보정의 scalar 진척 조건 〈T,Z〉=0 아래

\[
A=C_E+\gamma C_{resp}+\lambda_pC_p+\lambda_bC_b+\lambda_rI,
\qquad
L=\Delta_N(C_E+\lambda_pC_p+\lambda_bC_b+\lambda_rI)-T.
\]

이제 ∇_Z J=ZA+L이다. Native에서 context deviation은0이므로 L에 γΔN C_resp는 없다. **이전 복구형의 λb(We−W0)Cb만 제거하며 λbΔN Cb는 남긴다.** Base cross term 전체를 삭제하면 We 기준 보존이 아니라 WN 이후 보정 크기만 벌점으로 주는 다른 문제가 된다.

\[
Z_u=-L\mathcal T_A,\quad w=T\mathcal T_A,\quad q=\langle T,w\rangle,
\]
\[
\boxed{Z_{OS}=Z_u-\frac{\langle T,Z_u\rangle}{q}w},
\qquad W_{OS}^{alg}=W_N+Z_{OS},\quad
W_{OS}=\operatorname{cast}_{FP32}(W_{OS}^{alg}).
\]

q=0이면 진척 조건이 모든 허용 보정에서 자동으로 성립하므로 Z_OS=Z_u다. γ=λp=λb=1, λr=10^-3·tr(C_E+C_resp+Cp+Cb)/d+10^-8을 첫 고정 설정으로 둔다. 여기서 'Base 보존 우선'은 복구와 보존 중 후자를 선택한 것이며 λb를 무한대로 키우는 의미가 아니다.

Closed form의 equality는 대수적 해에 대한 것이다. 모델에 저장한 W_OS에서는 FP32 반올림 차이, 실제 〈T,W_OS−We〉−a_N 및 허용 공간 밖 성분을 따로 기록한다. Round-off를0으로 보고하거나, 반올림 차이를 없애려고 별도의 edit 손실 보정을 숨겨서 추가하지 않는다.

Ridge로 동일 quadratic의 해가 정의되고 현재 목적식은 최소화되지만 Current response error가 따로 감소한다는 보장은 없다. OS의 정확한 주장은 scalar 진척 보존과 soft response-change 비용 아래 새 정적 목적식을 최적화한다는 것이다. OS에는 barrier/slack/functional backward/line search/rollback이 없다.

## 11. 보호 bank, token 위치, continuation과 가중치

B100 주 진단에서 Past128/Base128 requests를 사용한다. 일반 입력에서는 실제 유효한 요청 수와 sampling budget 중 작은 수를 쓴다. Empty bank는 해당 Cp 또는 Cb/F/G를0으로 두며 평균을 나누지 않는다. Bank 수는 current B의 상한이 아니다.

선택 규칙:

- Past: active-fact ledger에서 현재 facts와 기존 Fixed/Past 평가 requests를 제외한다. 시기별 hash candidate512 중 hash64+native 구조 영향 상위64를 선택한다. 영향 점수는 canonical prompt의 subject_last key에 대한 ||ΔN k||²다. Pool 부족은 실제 수에 맞게 줄이고, quota/중복 제거/제외 이유를 manifest에 저장한다.
- Base: 현재 batch의 fact identity, 과거 승인된 replacement 및 평가 panel과 겹치지 않는 원래 지식 candidate512에서 같은 hash64+영향64를 선택한다. Base audit128은 이 candidate pool에도 포함하지 않은 별도의 disjoint hash 표본이다.
- 표본 선택에 held-out endpoint loss를 쓰지 않는다. Canonical subject lookup의 영향 점수는 선택용 근사이며 다음의 모든 보호 위치에 대한 점수와 같지 않다.

Hash 순서는 SHA256(seed=20260911, entry identity, bank 종류, fact identity, version ordinal)의 canonical JSON UTF-8 serialization으로 고정한다. Candidate는 그 순서의 첫 min(512, eligible 수)개다. 최종 n=min(128, candidate 수)에 대해 hash 순서의 ceil(n/2)개를 먼저 선택하고, 남은 candidate에서 영향 상위 n−ceil(n/2)개를 고른다. 영향 동점은 같은 hash 순서로 푼다. Base audit는 controller 후보와 평가 panel을 제외한 풀에서 같은 규칙의 hash 표본만 취한다. 모든 arm이 같은 manifest를 공유하며, 동일 batch의 latest-wins 처리로 입력이 달라지면 기존 cached N endpoint를 그대로 재사용하지 않는다.

문장·token 규칙:

1. Past는 canonical rewrite와 dataset의 첫 paraphrase를 사용한다. Paraphrase가 없으면 canonical 하나만 사용한다. Target은 ledger의 최신 target_new이고 전체 target-token NLL을 측정한다. Token 평균은 context 안에서만 먼저 수행한다.
2. Base는 canonical prompt 뒤의 **dataset target_true token sequence 중 첫 최대8 tokens**를 고정 continuation으로 쓴다. W0/We의 생성 결과에서 continuation을 고르지 않는다. Prefix/공백/target tokenization은 기존 evaluator의 동일 packing을 사용하며 input_ids, target positions와 hash를 저장한다. EOS/유효 token 수는 실제 packing 기준이며 padding은 제외한다.
3. Kp/Kb는 subject token 하나가 아니라, **해당 target token을 예측하는 각 logit 위치**의 L4 down-projection input을 수집한다. 첫 target 예측에는 마지막 prompt token의 위치, 그 뒤에는 teacher-forced prefix의 해당 위치를 사용한다. Target token 자신의 위치로 한 칸 밀리지 않게 한다.
4. Past request i의 context 수 m_i, context c의 target 수 L_ic이면 Ωp의 각 token weight는1/(n_p m_i L_ic)다. Base request i의 보호 token 수 L_i이면 Ωb token weight는1/(n_b L_i)다. C_p/C_b는 이 request→context→token 평균 규약의 Gram이다. 이것은 native M의 averaged subject-key 규약과 다르다.
5. BF Base teacher는 같은 고정 위치의 **We full-vocabulary 분포**다. We teacher와 reference loss는 entry 한 번 저장해 모든 arm이 공유한다. W0 분포/정답 NLL은 동일 위치의 별도 누적 손상 평가용이다. Ground-truth continuation이라고 W0/We가 실제로 정답을 알고 있다고 가정하지 않는다.
6. Teacher와 실제 모델은 FULL-FP32 경로를 유지하고 KL의 누적/solver 진단은 FP64로 계산한다. Teacher representation, probability normalization, log-prob precision과 packing hash를 봉인한다. 이론상 음수가 아닌 KL의 작은 음수 rounding은 원값/개수를 기록하고 controller에서는0으로 처리한다.

Capture는 fixed sequence당 한 번이며 각 보호 token 때문에 모델 forward를 따로 호출하지 않는다. Scalar structural mapping 변화와 전체 downstream language 출력 손상은 여전히 다르므로, OS/BF 차이를 그 관측 범위와 함께 해석한다.

## 12. 기능적 손상: Past는 context별 악화, Base는 We 분포 보존

Past request i, context c에 대해 target-token 평균 loss 증가를

\[
d_{p,ic}(W)=\ell_{ic}^{target\_new}(W)-\ell_{ic}^{target\_new}(W_e)
\]

로 정의한다. Context 평균 전에 positive-harm smoothing을 적용한다.

\[
\psi_\tau(d)=\begin{cases}
0 & d\le0\\
d^2/(2\tau)&0<d<\tau\\
d-\tau/2&d\ge\tau,
\end{cases}\qquad
F_p(W)=\frac1{n_p}\sum_i\frac1{m_i}\sum_c\psi_\tau(d_{p,ic}(W)).
\]

τ=.01 nats/token을 고정한다. Rewrite의 개선이 rephrase의 악화를 음수 상쇄할 수 없다. 같은 context 안의 token별 NLL 상쇄는 남으므로 보호 단위는 context의 target-token 평균이라고 명시한다. Token별 악화 분포도 기록하되 새 gate로 쓰지 않는다.

Base는 We의 fixed-position 분포를 직접 보존한다.

\[
F_b(W)=\frac1{n_b}\sum_i\frac1{L_i}\sum_t
KL(p_{e,it}\|p_{W,it}).
\]

각 token-position KL은 이미 비음수이므로 Base에 ψ를 다시 씌우지 않는다. 작은 KL에 squared hinge까지 적용해 출발점 부근의 신호를4차로 약하게 만들지 않는다. F_b(We)=0이고 이 함수는 W0를 향한 복구 목표를 갖지 않는다. We가 이미 잃은 지식을 복구하도록 강제하지 않는다.

W0에 대한 다음 값은 평가로만 보존한다.

\[
K_0(W)=\frac1{n_b}\sum_i\frac1{L_i}\sum_t KL(p_{0,it}\|p_{W,it}),
\qquad \Delta K_0=K_0(W)-K_0(W_e).
\]

F_b를 낮춘 것, W0 대비 누적 차이를 줄인 것, Base 정답/NS를 개선한 것을 서로 다른 결과로 보고한다. 같은 이유로 구조 ΔM_native action과 global C0 risk도 두 functional barrier와 섞지 않는다. Raw NLL/KL, ψ 이후 값, context/request별 median/p90/max를 함께 저장한다.

## 13. BF budget: OS의 손상과 연결하되 elastic하게 적용한다

같은 bank에서 OS endpoint W_OS를 한 번 관측해 F_{j,A}=F_j(W_OS)를 얻는다. OS weight를 임시 적용한 관측이며, BF 경로는 We에서 시작한다. N의 bank 손상도 별도로 기록한다. Cached 진단에도 이 새 bank 평가 비용을 넣고 온라인 비용에도 포함한다.

σ_j=F_{j,A}+τ, f_j=F_j/σ_j로 단위를 고정한다. 주 terminal budget은

\[
b_j(1)=0.9 F_{j,A}/\sigma_j,\qquad b_j(s)=s\,b_j(1).
\]

이는 **Past의 context별 ψ 집계와 Base의 We-teacher KL 각각**을 OS보다10% 낮추려는 첫 진단 operating point이며, 원시 NLL 악화량·실패 문항 수·W0 누적 손상을10% 줄인다는 뜻은 아니다. 이 값은 효과를 관측한 뒤 고른 최적값이 아니다. OS가 손상을 전혀 만들지 않은 channel은 b=0이지만 elastic slack을 허용한다. 두 channel의 N/OS 손상 절대값도 반드시 함께 보고한다.

실제 손상이 작으면 작게 제어하는 것이 정상이다. Barrier 활성률을 높이려고 결과를 보고 budget을 바꾸지 않는다. β=.9가 어떤 모델에서도 적절하다는 주장은 하지 않는다. 수치의 광범위 sweep은 주 실험에 포함하지 않는다.

## 14. BF: We부터 OS 경로를 따라가는 predictor–corrector

D_A=W_OS−We. 주 경로는 s=0,1/8,…,1이며

\[
W_s=W_e+sD_A+Z_s,\qquad Z_s=Z_sP,\qquad \langle T,Z_s\rangle=0.
\]

대수적 OS 해에서는 〈T,D_A〉=a_N이다. 실제 저장 endpoint를 쓰는 BF의 기준은 a_OS=〈T,W_OS−We〉이며, FP64 경로는 s a_OS를 유지한다. a_OS−a_N을 별도 보고한다. 기준 anchor와 누적 Z는 FP64로 계산하고 모델에는 매번 FP32로 변환한 weight를 적용한다. Endpoint anchor는 저장 W_OS를 사용한다. 실제 적용 weight의 scalar/span residual과 rounding norm도 기록해, 이상적 equality와 모델 정밀도를 구분한다.

각 step:

1. 현재 accepted state W_s에서 f_p/f_b를 갖고 시작한다.
2. OS 진행만 추가한 proposal W̄=We+s_next D_A+Z_s를 임시 적용한다.
3. W̄에서 두 bank의 f_j(W̄)와 G_j=∇_W f_j(W̄)를 계산한다.
4. §15의 시간 간격을 반영한 비용과 누적 보정 비용 아래 two-barrier solve로 C=CP, 〈T,C〉=0을 얻는다.
5. Z_next=Z_s+C로 다음 weight를 확정한다.
6. 실제 f_p/f_b를 forward로 관측하고 예측 오차·slack을 기록한다. 나쁜 결과를 rollback하지 않고 다음 step으로 이어 간다.

현재 We에서는 F=0이고 hinge/KL gradient도0일 수 있다. Nominal proposal을 먼저 보는 것이 첫 손상을 발견하는 방법이다. 현재 상태의 zero gradient만 보고 안전하다고 판정하지 않는다.

BF1은 s=0→1 한 번에 같은 절차를 적용한다. **BF1도 barrier를 쓰는 방법이며, barrier 없는 OS와 구분한다.** BF8은8번 관측·보정한다. OS/BF1/BF8의 실제 총 비용과 endpoint를 비교한다.

이는 discrete continuation이다. ODE 수렴이나 continuous-time safety를 입증했다고 부르지 않는다. z/K/M/C0/보호 bank와 metric은 고정하고 실제 output observer를 갱신한다.

BF의 실제 Current loss는 방향 계산에 포함하지 않는다. 저장 state에서 Current rewrite/rephrase/target-response를 관측해 scalar 진척과 실제 편집의 차이를 핵심 진단으로 삼는다. 이 관측은 후속 step을 허용하는 gate가 아니다.

## 15. BF: step 크기를 반영한 비용과 누적 보정 anchor

D_A=W_OS−We는 실제 저장 weight의 FP64 차분이다. a_A=tr(D_A A D_Aᵀ)>0이면 H=A/a_A를 공통 metric으로 쓰고, ||X||_H²=tr(X H Xᵀ)로 정의한다. **D_A가 정확히0이면 BF1/BF8/Frozen 모두 W_OS=We를 그대로 반환하고0개 correction/gradient stage로 기록한다.** We bank 값을 재사용하고 STATIONARY_OS_ENDPOINT를 저장한다. 미달한 edit 성능을 성공으로 간주하지 않는다.

허용 inverse action을 R_H(G)=G T_H라 쓰면 scalar 진척까지 보존하는 연산자는

\[
\mathcal K_H(G)=\mathcal R_H(G)
-\frac{\langle T,\mathcal R_H(G)\rangle}{\langle T,\mathcal R_H(T)\rangle}
\mathcal R_H(T).
\]

분모0이면 모든 허용 방향이 진척을 보존하므로 rank-one 항을 생략한다. 구현은 whitened unit observer로 같은 연산을 수행한다.

Reference calibration을 정확히 다음과 같이 고정한다.

\[
G_j^{ref}=\nabla_W f_j(W_{OS}),\quad
q_{j,ref}=\langle G_j^{ref},\mathcal K_H(G_j^{ref})\rangle,
\quad\varepsilon_j=.1(q_{j,ref}+10^{-4}).
\]

σ_j, H, ε_j는 entry마다 한 번 정하고 BF1/BF8/Frozen에 동일하게 사용한다. q_ref는 raw gradient norm이나 projection 전 값이 아니다. Empty channel은 제거하며 q_ref=0인 비어 있지 않은 channel에는 양의 floor가 남는다.

h=s_next−s, ρ=exp(−κh), κ=2라 두고

\[
c_j=b_j(s_{next})-\rho[b_j(s)-f_j(W_s)],\qquad e_j=f_j(W̄)-c_j.
\]

ξ_j는 이 step의 functional-risk slack이다. 최종 주 문제는

\[
\begin{aligned}
\min_{C=CP,\,\langle T,C\rangle=0,\,\xi\ge0}\quad&
\frac{\|C\|_H^2}{2h}
+\frac\eta2\|Z_s+C\|_H^2
+\sum_j\frac{\xi_j^2}{2h\varepsilon_j}\\
\text{s.t.}\quad&e_j+\langle G_j,C\rangle\le\xi_j,
\end{aligned}
\]

η=1을 고정한다. 첫 항은 보정 increment의 action, 둘째는 OS 경로에서 누적 BF 보정 Z가 벗어나는 비용이다. 같은 총 보정을 균등 N분할하면 Σ||Z/N||²/(1/N)=||Z||²가 되어 분할만으로 action이1/N로 싸지지 않는다. Slack도 ξ=h·violation_rate라는 동일 규약으로 scaling한다. Norm cap이나 rollback은 추가하지 않는다.

닫힌 해를 위한 공통 계수는

\[
a_h=\frac{h}{1+\eta h},\qquad
C_{anc}=-\frac{\eta h}{1+\eta h}Z_s,
\quad e'_j=e_j+\langle G_j,C_{anc}\rangle.
\]

C_anc는 OS 경로 쪽으로 누적 보정을 줄이는 항이며, 구현 변수명은 `anchor_correction`으로 둔다. Pretrained covariance C0와 별개다.

\[
\widetilde w_j=a_h\mathcal K_H(G_j),\quad
\widetilde Q_{ij}=a_h\langle G_i,\mathcal K_H(G_j)\rangle,
\]
\[
\lambda^*=\arg\min_{\lambda\ge0}
\tfrac12\lambda^\top[\widetilde Q+h\operatorname{diag}(\varepsilon)]\lambda
-(e')^\top\lambda,
\quad C=C_{anc}-\sum_j\lambda_j\widetilde w_j,\quad
\xi_j=h\varepsilon_j\lambda_j.
\]

Active set는 ∅,{past},{base},{past,base} 네 가지다. Joint solve 전의 model backward/full-space inverse action은 별도 비용이다. ξ와 ξ/h, Σ||C||²/h, ||Z_final||², Σ||C||²를 모두 기록한다.

이 scaling과 anchor는 BF1/BF8을 같은 endpoint 최적화로 만들거나 완전한 step 수 불변성을 보장하지 않는다. 실제 nonlinear 경로와 관측 횟수는 다르다. BF1→BF8은 반복 방법 전체의 차이, **동일 h/η/ε를 쓰는 Frozen-BF8→BF8은 방향 갱신의 효과**로 해석한다.

## 16. 유한 step과 실제 output의 한계

위 closed form은 **proposal 주위의 두 기능적 손상에 대한 일차 근사**다. 실제 다음 f가 c+ξ 아래라는 hard guarantee가 아니다. 지난 실험의 곡률 문제를 문구로 덮지 않는다.

반드시 raw e 및 anchor를 반영한 e′, 실제 f의 감소, 예측 잔차 f(W_next)−f(W̄)−〈G,C〉를 모두 기록한다. 주요 주장에는 실제 endpoint와 held-out locality를 사용한다. 주 방법의 scalar 목표반응 진척 보존과 보조 exact-key-null은 다른 조건이다. 어느 쪽도 모든 실제 NLL의 유한-step 보존과 같지 않다.

추가 검증용으로 구조 risk는 full-weight correction C의 정확한 quadratic을 계산할 수 있다:

\[
R_j(W̄+C)=R_j(W̄)+\langle A_j,C\rangle
+\tfrac12\operatorname{tr}[CC_jC^\top].
\]

따라서 first-step zero gradient/이차항을 model forward 없이 진단할 수 있다. 기능적 손상과 동일시하지 않는다. 향후 구조 risk 자체를 정확히 제어하고 싶다면 두 방향의 coefficient α_p,α_b에 대한2변수 convex QCQP로 마지막 이차항까지 포함할 수 있다. 이 경우 완전한 closed-form 한 번이라고 부르지 않는다. **이 QCQP는 이번 functional 주 방법에 추가하는 세 번째/네 번째 gate가 아니다.**

Base 추가 변화의 억제와 W0 복구를 분리하기 위해 OS/BF 계산 입력을 고정한 상태에서 audit-only W0 reference를 바꾸어도 update가 바뀌지 않아야 한다. P/C0/데이터·선정 bank까지 함께 바꾸는 다른 실험을 이 불변성으로 묶지 않는다.

## 17. 실제 충돌 지도를 만드는 기록

입력 mode별 다음 양을 함께 저장한다.

- current-response sensitivity, past Gram energy, C0 energy.
- native contribution이 각 named bank에 만든 실제 NLL/KL 변화.
- proposal에서의 functional gradient와 해당 mode의 내적.
- barrier 후 input/output coefficient가 어디로 옮겨졌는가.
- 두 channel의 λ, step slack ξ와 rate ξ/h, Q spectrum/condition, increment norm 및 cumulative norm.

현재 request별 native factor가 재구성 가능한 경우 ΔN=Σ_i ΔN_i에 대해 〈∇_W ℓ_protected,j, ΔN_i〉를 pairwise attribution으로 계산할 수 있다. Subject lookup 하나만 사용한 근사와 모든 teacher-forced token의 full weight gradient에 대한 계산을 구분한다. 저장 native factor가 없으면 basis-mode 수준에서 보고하고 request별 인과 attribution을 꾸며내지 않는다.

개별 neuron의 큰 coefficient는 시각화 보조로만 쓴다. 사실의 고유 저장 위치 또는 정확한 causal localization이라고 부르지 않는다.

## 18. 주 비교: barrier 없는 OS를 독립적으로 평가

Llama-3-8B-Instruct revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, L4 down projection FULL-FP32를 유지한다. 기존 B100 fixture Early We10/B11·Middle We50/B51·Late We90/B91는 알고리즘의 입력 규격이 아니라 첫 성능 진단이다.

| Arm | 방식 | 답하는 질문 |
|---|---|---|
| N | 기존 BLUE-L4 native endpoint | 강한 native 기준 |
| OS | §10의 barrier 없는 one-shot | 두 보호 대상을 고려한 정적 우회만으로 충분한가 |
| BF1 | OS endpoint proposal에서 functional barrier 한 번 | 기능적 충돌 관측 한 번의 추가 효과 |
| BF8 | We→OS 방향을8단계로 진행하며 barrier feedback | 반복 갱신이 단회보다 유용한가 |

세 entry에서 OS/BF1/BF8을 모두 실행한다. N3은 기존 reference이나 새 bank 평가는 추가한다. 새9개 경로이며 BF1/BF8의 logical functional-gradient stage 합27개다. OS에는 기능적 model backward가 없다.

Middle에서만 Frozen-BF8 하나를 보조로 둔다. Direction은 공통 OS endpoint gradient에서 고정하고, actual risk와 RHS·누적 Z anchor는 매 step 갱신한다. BF8과 같은 h=1/8, η=1, H/ε를 사용한다. BF8과 비교하여 방향 갱신의 효과를 본다. 주9+보조1=새10개 경로다. V256/random-V 또는 hard-key-null 진단은 이 주 실행 수에 섞지 않는다.

주 성능을 보며 보조 실행 여부를 바꾸지 않는다. J1/J4/N16/β·τ·ε grid는 이번에 다시 늘리지 않는다. Batch-size 검증은 §23의 별도 명시된 범위다.

## 19. Current–Base–Past를 함께 읽는 평가

가장 먼저 scalar 진척이 실제 Current 보존을 대변하는지 확인한다. Endpoint 및 BF8/Frozen의 s=.25,.5,1 실제 snapshot에서 다음을 기록한다.

- Current의 canonical 및 rephrase NLL·RS/PS, target-response의 목표 방향 성분과 수직 성분, context별 ||Z_total k||²와 p90/max.
- N/OS 기준의 Current 성공→실패·실패→성공 및 NLL 변화 분포. B=1에서도 목표 방향 밖의 성분이 증가할 수 있음을 포함한다.
- Base의 We teacher KL, W0 teacher KL, dataset target_true NLL/NS. **We 반응을 잘 유지한 것과 W0 지식을 회복한 것을 별도 열로 보고한다.**
- Past의 rewrite/rephrase별 raw NLL 증가·ψ 증가와 held-out Fixed/Past retention.
- BF cumulative Z, native 대비 전체 보정 Z_OS+Z_final, 각 context response-change, Σ||C||²/h, ξ/h와 endpoint slack.

B100 endpoint는 기존 Full3900을 재사용하고 controller bank와 identity를 분리한다. Base audit128도 controller의 candidate pool과 분리한다. Current 중간 관측은100 rewrites+200 paraphrases의 실제 candidate NLL이며, 실행을 승인하는 gate나 endpoint 선택 기준으로 쓰지 않는다. BF8/Frozen의3개 entry+Middle frozen, s=.25/.5는8개 중간 관측으로2400 prompt-pair events가 추가된다. Endpoints는 Full에 포함한다. 서로 다른 B에서는 실제 request/prompt 수를 분모로 사용한다.

N→OS는 **새 정적 방법 전체의 효과**다. 공간·통계·목적식·정규화가 함께 바뀌므로 충돌 통계만의 인과 효과로 좁혀 말하지 않는다. OS→BF1은 단회 functional 보정, BF1→BF8은 반복 경로 전체, Frozen-BF8→BF8은 해당 h/anchor 아래 observer 방향 갱신의 효과다.

동일 Fixed100을 세 독립100개처럼 합치지 않고 request-paired bootstrap을 사용한다. Controller bank 개선이 held-out으로 이어지지 않으면 bank 적합으로 해석한다. Current 실제 손실이 악화되면 scalar/soft-response 조건의 한계로 보고하며 Base 개선만으로 방법 전체의 우월성을 선언하지 않는다.

## 20. 계산량: logical B와 physical microbatch를 분리

기존 A6000 B100 측정에서 key capture 두 번은 약40.2–40.6초, native dense solve는 약.31–.32초였다. 새 protected factorization/functional preconditioning의 시간과 B1000 시간은 실측 전 확정하지 않는다. 두 번 capture를 없애는 재사용과 새 bank/teacher 비용을 모두 ledger에 넣는다.

| B | averaged K, FP32 | 개별6B key, FP32 | target4096×B, FP32 |
|---:|---:|---:|---:|
| 1 | .055MiB | .328MiB | .016MiB |
| 10 | .547MiB | 3.281MiB | .156MiB |
| 100 | 5.469MiB | 32.813MiB | 1.563MiB |
| 1000 | 54.688MiB | 328.125MiB | 15.625MiB |

Full14336×14336 FP32 matrix 하나는784MiB, FP64는1568MiB다. 이것은 B와 무관하지만 여러 statistic/factor/solver workspace가 동시에 존재할 수 있다. OS/BF의 full-space 비용을 이전 Ub100의 작은 solve 비용으로 대체해 추산하면 안 된다.

임의 B에서 current Gram과 cross statistic은 streaming으로 누적할 수 있다. k/R을 전부 GPU에 보관하거나 B×B inverse를 반드시 만들 필요가 없다. Protected base factorization이 고정이면 Woodbury/current-key rank update를 사용할 수 있지만 base factorization 비용도 계산한다. B가 커지면 feature-space solve 또는 matrix-free SPD solve를 선택한다. 내부 linear solver 반복은 모델 output-feedback step과 다른 비용이며, iterative solver를 사용했다면 algebraically closed-form과 수치 구현을 구분한다.

Past128×2 prompts + Base128×1 prompt, physical microbatch2이면 한 functional gradient stage는192 model forward/backward calls다. Base의 최대8 target_true teacher positions는 같은 sequence에 들어가므로8개 호출로 세지 않는다. 고정 continuation 생성용 model call은0이다.

B100 주 BF1/BF8 세 entry는27 stage=5184 backward가 계획값이다. BF1의 OS endpoint gradient를 common으로 저장해 Frozen-BF8/ε calibration에 재사용한다. BF8은 다른 proposal에서 새 gradient를 계산한다. 재사용을 못 하면 추가 비용을 별도 기록한다. Accepted risk 관측은 forward-only, full 평가와 generation도 별도다. 이 숫자는 bank가 고정일 때의 functional stage 비용이며 z/native/key capture의 B 의존 비용을 포함하지 않는다.

OS의 추가 functional backward는0이지만 native z 최적화는 그대로 있다. Cold one-shot pipeline에는 요청당 z 최적화, native proposal solve, 보호 statistic 준비 및 OS solve가 들어간다. 최종 weight를 한 번 확정하는 의미의 one-shot이다. 보호 bank의 gradient/teacher가 필요한 것은 BF이며, OS의 구조 bank key 준비와 구분한다.

FP64 dense gradient4096×14336 하나는448MiB다. Base teacher의128 requests×최대8 positions×128256 vocab FP32 저장은 최대 약501MiB이며 We/W0 두 teacher를 모두 저장하면 두 배다. 실제 valid token 수로 ledger를 산출하고 teacher는 host/mmap에서 필요한 microbatch만 GPU로 옮긴다. 모든 full matrix·gradient·factor를 동시에 GPU에 쌓지 않는다. OS의 구조 key 준비, BF의 We teacher 준비와 bank gradient, W0 audit forward, Current 중간2400 pair-events를 별도 계산한다. OS solve 자체에는 teacher logits가 필요 없고, OS의 기능적 성능 평가에는 reference가 필요하다. 새로운 Current backward는0이다.

## 21. 구현 연결점과 산출물

기존 BLUE-L4 binding/native prepare는 재사용한다. 기존 progress-barrier runtime의 WN anchor를 덮어쓰지 않고 새로운 실험 package에서 arbitrary-B 입력, OS affine solve와 We–OS anchor schedule을 구현한다. History append는 accepted batch endpoint 이후 기존 알고리즘대로 한 번만 한다. Active-fact ledger도 그 시점에 한 번 갱신하고 대체된 version을 retired로 표시한다. Microbatch/inner BF step에서는 두 history 모두 갱신하지 않는다. 이번 frozen-entry 진단에서는 새 cumulative chain을 실행하지 않는다.

필요 모듈:

- key/bank preparation: 개별6B context 통계, named bank keys, We teacher/reference, audit-only W0, active-fact ledger, retired-version/exclusion manifest.
- routing geometry: full P-space inverse operator, SPD metric, OS closed form, scalar progress projection.
- functional observer: context-level Past excess/Huber 및 token-position Base KL aggregation, two bank gradient 및 actual-only evaluation.
- joint elastic solve:2×2 active sets, fixed q_ref/ε, h-scaled action/slack, cumulative anchor, moving-budget RHS.
- continuation runtime: immutable We–OS anchor, one correction per stage, actual metric logging.

산출물: bank-manifest.json, geometry-spectrum.csv, conflict-map.csv, native-response-parity.csv, trajectory.csv, per-context-harm.csv, paired-endpoints.csv, compute-ledger.csv, base-preservation-versus-recovery.csv, diagnostic-report-ko.md. Raw prompts/teacher probabilities/weights는 local-only로 보존한다.

수치 correctness는 projector/equality와 joint KKT 두 묶음으로 확인한다. 성능이 낮다는 이유로 fixture를 제거하지 않는다. 형상/비정상수/잘못된 state 적용만 기술 오류로 처리한다.

## 22. 선행연구와 주장 범위

- ROME/EMMET는 response equality를 유지하며 보존 metric 비용을 최소화하는 closed form을 이미 제시했다. 정적 inverse-metric routing 자체가 새 기여는 아니다.
- AlphaEdit는 원래 지식의 P와 이전 편집의 Kp penalty를 이미 구분한다. 이번의 두 bank 구분을 최초라고 주장하지 않는다. 실제 M/C0의 정보와 실제 출력 보존의 차이를 명시한다.
- BLUE는 선택 layer의 residual을 직접 계산하는 강한 native 기준이다. 이번은 L4 내부의 추가 입력 방향과 기능적 손상 제어를 다룬다.
- CAKE의 layer별 sensitivity/weighted allocation과 구별되는 질문은 한 layer 내부의 두 보호 대상에 대한 실제 충돌 반응이다. Weighted constrained allocation의 일반 원리를 최초라고 부르지 않는다.
- Gradient conflict를 projection/QP로 다루는 일반 아이디어는 GEM에도 있다. 이 설계의 검증 대상은 scalar 목표반응 진척의 명시적 보존, context별 Past 악화와 We 기준 Base KL의 두 functional channel, nominal lookahead, feedback의 비용 대비 이득이라는 구체 조합이다.

Primary references:

1. [AlphaEdit §3.2–3.3, Eq12–14](https://arxiv.org/html/2410.02355v4#S3.S3)
2. [EMMET §5](https://arxiv.org/html/2403.14236v3#S5)
3. ROME §3.1: https://arxiv.org/pdf/2202.05262
4. MEMIT §4.2: https://arxiv.org/html/2210.07229v2
5. BLUE §4.3: https://arxiv.org/pdf/2502.03748v3
6. CAKE: https://aclanthology.org/2026.acl-long.918.pdf
7. GEM: https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf
8. CBF–QP framework: https://arxiv.org/abs/1609.06408

Local evidence:

- Prior design: /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-11-blue-l4-only-progress-preserving-barrier-experiment-design.md
- Reviewed report/source snapshot (remote report SHA902ca04d2bb646bfc28fdb8def1a9bb84ea6134a3bcc79cf36fd7ed2960025c3): /tmp/blue-l4-review-20260911-sw2g911y/
- Original server2 report: /mnt/raid5/janghj/.codex/worktrees/odeeditsh2-blue-l4-progress-barrier-integration-v1/experiment-reports/servers/server2/blue-l4-progress-barrier-2026-09-11-v1/diagnostic-report-ko.md
- ABC native source/fixtures: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/

User review source: [첨부 리뷰](/mnt/raid5/janghj/.codex/attachments/29d58f70-fe4d-4292-8e9a-c11c57348569/pasted-text.txt). 리뷰의8개 항목은 §6/19(Current 보존 범위), §4/10/12(Base reference), §5(active 통계), §15/19(BF 누적 비용), §11/12(보호 단위), §7/11/15(구현 정의), §20(비용), §19(N→OS 해석)에 반영했다.

## 23. Batch-size 독립성의 검증 계약

**지원 조건:** B=0은 no-op, B≥1의 임의 정수에서 메서드를 정의한다. 4로 나누어 떨어질 필요가 없다. 현재 batch의 유효 rank가 B보다 작아도 동작한다. 고정-size bank는 보호 관측의 sampling budget이며 현재 edit B의 상한이 아니다.

필수 수치 확인:

1. B=1,7,64,100,257,1000의 synthetic K/R, 중복·collinear keys, past0 cases에서 SPD solve 및 scalar progress residual을 확인한다. 이는 작은 수치 correctness이며 GPU 모델 실험은 아니다.
2. 동일 We/P/history/target/native proposal을 고정하고 모든 current 항을 두 번 복제하면서 Ω를 절반으로 만들면 CE/T/C_resp와 OS/BF의 공통 geometry가 같아야 한다. Native proposal을2B로 새로 계산하면 native의 합 기반 loss 때문에 proposal 자체가 달라질 수 있다. **전체 native pipeline까지 duplicate invariant라고 주장하지 않는다.**
3. 같은 logical B를 physical microbatch1/2/8 및 uneven tail로 나눠도 같은 statistic과 같은 해를 얻는다. Chunk 평균을 단순평균하지 않고 실제 request/context weight로 합산한다.
4. Protected bank를 복제했을 때 평균화된 Cp/Cb가 같아야 한다. Active bank의 Cp/Cb와 native용 raw M_native는 별도로 유지하며 raw M을 routing Cp에 섞지 않는다.
5. Hard-key-null variant에서 rank가 소진되는 사례를 만들어 primary OS/BF는 여전히 정의됨을 확인한다. Exact 보존을 계속 주장하면서 몰래 constraint를 버리는 fallback은 사용하지 않는다.

**v2 수식의 CPU 수치 검산 완료(2026-09-11).** Seed 20260911, input dimension4/output dimension2/P rank3인 작은 예제에서 위 여섯 B × 일반·중복·collinear key × Past 유무의36개 OS 문제를 확인했다. 대수적 OS 해를 독립적으로 구성한 equality-constrained KKT 선형계와 비교했으며 해의 최대 Frobenius 차이는2.64×10^-14, scalar 진척 residual은1.08×10^-16, projected stationarity의 최대 절대 residual은1.24×10^-10이었다. 모든 case에서 SPD Cholesky가 성립하고 J_pres가 native proposal보다 증가하지 않았다.

Physical chunk1/2/8·uneven tail의 OS 해 차이/(1+||Z||)는 최대2.71×10^-13이었다. 큰 크기의 collinear 통계를 포함해 chunk 및 요청 복제의 Gram/T 절대 차이는 각각1.29×10^-11,1.79×10^-11 이내였고 보호 bank 복제의 Gram 차이는1.53×10^-16 이내였다. T=0의 빈 scalar observer와 exact-key-null 잔여 rank0에서도 주 soft-response OS가 정의됨을 확인했다.

새 BF 문제는 h=1,1/8,1/16에서 네 active set을 각각 구성한12개 예제로 검산했다. Dual coefficient 복원 오차는1.00×10^-15, primal violation은1.74×10^-17, projected stationarity는7.63×10^-17, scalar 진척 residual은8.68×10^-19 이내였다. 같은 총 보정의1/8/16분할에서 시간 간격으로 보정한 action 합도 일치했다. 이는 새 수식의 작은 수치 확인이며 모델 성능·FP32 적용·actual nonlinear barrier를 검증한 결과가 아니다. W0 audit 입력의 분리, target packing 및 native parity는 실제 구현 시 확인할 항목으로 남는다.

실제 model의 작은 batch 검증은 Middle의 동일 We와 기존 B100 target pool에서 B=1,7,100을 구성한다. 각 새로운 B마다 native joint solve가 필요하며 기존 WN100을 subset baseline으로 쓰지 않는다. 비균등 B=7은 wrapper의 고정100/4group 가정을 검출한다. 각 B에서 N/OS/BF1을 비교하고 request 내용이 다른 B끼리 성공률만으로 batch-size 인과를 주장하지 않는다.

B100은 §18의 동일 Middle 결과를 재사용한다. B1/B7의 N/OS/BF1은 새6개 경로이므로 주 비교·Frozen10개와 합쳐 **첫 실행은 새16개 경로**다. 보호 bank가 각128/128이면 추가 BF1 두 stage는384 backward이며, 주 비교5184와 합쳐5568이다. 유효 bank 수가 줄면 실제 호출 수를 기록한다. 이 검증은 더 많은 controller arm이나 성능 gate를 추가하는 것이 아니라 같은 구현의 B 규격을 확인하는 것이다. B별 manifest·reference 준비, 새 native solve와 endpoint 평가 비용도 따로 더한다.

B=1000의 실제 검증에는 **동일 We에서** 구성한1000 request의 z/contexts와 새 joint baseline이 필요하다. 서로 다른 historical batch에서 저장된 z를 합치지 않는다. 이는 새로운 target 계산 비용이 들어가는 확장 단계이며 기존 B100 cache만으로 실행 완료나 성능 보장을 주장하지 않는다. 설계상 API/streaming은B1000 이상에도 동일하지만 실제 시간/메모리/품질은 측정해야 한다.

독립 simultaneous batch 비교와 sequential 비교를 분리한다. 같은 We에서1000개를 한 번 푸는 것과100개씩10번 write/history append/retarget하는 것은 다른 문제다. 후자를 memory microbatching이라고 부르지 않는다.

Native BLUE 자체는 requests를 순회하고14336×14336 system을 풀어 B가 가변이다. 제거할 기존 fixture 제약은 ABC binding.py의 offset:offset+100, target100/cache-hit100 assert와 Current100 evaluator다. 새 target cache key에는 entry weight/layer/request/context/hparams identity를 포함한다. Method API에 별도의 batch-size100 assertion을 가져오지 않는다.

## 24. 결과를 읽는 경계

OS가 BF8과 같거나 더 좋으면 barrier 없는 one-shot을 우선한다. BF1과 BF8이 같으면 반복 feedback 비용을 정당화하지 않는다. Functional bank만 좋아지고 held-out locality가 좋아지지 않으면 bank 적합으로 해석한다.

주 scalar 진척이 같아도 per-request response/NLL이 나빠지면 그 보호 조건의 한계다. OS/BF가 arbitrary B에서 계산된다는 사실로 모든 B의 edit–locality capacity가 충분하다고 주장하지 않는다. 큰 B에서 손상이나 slack rate ξ/h가 증가하는 것은 함께 보존할 요청이 늘어난 효과일 수 있으며, 고정된 named bank의 coverage 문제와 구별한다.

우회 성공은 지식이 없는 빈 neuron을 발견했다는 뜻이 아니다. 지정된 native 진척과 허용 공간에서 두 보호 대상에 손상이 더 적은 write를 찾았다는 뜻이다. Barrier의 필요성은 그 정적 해를 넘어 실제 output feedback이 추가 이득을 주는지로 판단한다.
이번 설계의 선택은 Base의 **신규 변화 억제**다. We에 이미 존재한 손상의 복구는 연구 질문에서 분리하며, 자연스럽게 회복된 결과가 있으면 audit으로 보고한다. 별도 복구 arm, Base hard priority, 추가 outcome gate는 주 비교에 넣지 않는다.
