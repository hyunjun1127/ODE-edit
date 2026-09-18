# BG-TW 설계 및 ICLR_2027.pdf 상세 검토

> 2026-09-15 후속 재검토: 아래는 이전 검토·설계의 기록이다. N4 기반 한도와 fixed-budget BG-1을 현재 첫 방법으로 사용하지 않는다. 현행 권고는 [파이프라인 재검토](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pipeline-reset-review-ko.md)와 [사전 보존 한도 없는 EP-TW-1 v3](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-edit-quality-preserving-tw-design-v3.md)를 따른다. 이전 실험 수치·수식 검토는 역사적 근거로 보존한다. 이 표지는 원격 V1 dispatch 변경을 뜻하지 않는다.

작성: 2026-09-15. 상태: global 해석·수식 검토. 신규 모델 편집·GPU 실행 결과가 아니다.

## 1. 판단

**연구 방향은 실행 가능한 write에 실제 출력 보존 피드백을 추가하는 쪽으로 정한다. 첫 구현은 단일 write·단일 보정·고정 generic core로 시작한다. ODE와 barrier bypass의 필요성은 결과로 판단한다.**

“단순 step 분할이 합리적이지 않으므로 barrier-guided ODE가 정당화된다”는 논증은 강도를 낮춰야 한다. 확인한 I2/I4는 특정 budget·write scale·optimizer carry 조건이다. 그 실패 또는 trade-off는 모든 분할 방식을 배제하지 않으며, barrier와 ODE가 유일한 대안이라는 뜻도 아니다. 타당한 동기는 다음과 같다.

> 시험한 반복 fitting과 target continuation은 신규 paraphrase의 달성을 높이기도 했지만 보존과 비용을 함께 개선하지 못했다. 따라서 실제 write 이후의 보존 손실을 관측하여 실행 가능한 residual을 보정한다. 보존 피드백, slack에 따른 가중, 방향 변경, target refresh의 기여를 나누어 검증한다.

사용자가 추가한 요구에 따라 **method test는 W0에서 B1–B10의 1,000개 요청을 실제 순차 처리**한다. W50/W90에서 시작한 결과는 역사적 동기·보조 진단이다. Full lifelong 주장은 W0부터 후보 정책을 고정하여 10,000개에 적용한 결과에서 판단한다.

## 2. 확인한 근거와 한계

- 사용자 원문: [ICLR_2027.pdf](/mnt/raid5/janghj/ODE-edit/plans/global/ICLR_2027.pdf), 13쪽 전체의 텍스트·수식. PDF의 문헌 목록 전체에 대한 독립적인 인용망 감사는 수행하지 않았다.
- 보고서 기준 commit: c2710f3e8fddf2a8df230f076e4519cc6d0d77e5.
- 신규 I2/I4/FROZEN 실행 commit: 8a061ea21661d9480acfc740fb888e7bcbdb4216.
- [완료 보고서 사본](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-method-review-2026-09-15/source/experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1/diagnostic-report-ko.md), final-populations·compute-summary·comparison-capsule 및 관련 집계.
- 실제 singleton fitter와 이전에 보존된 BLUE AlphaEdit source. 보존 source의 SHA256은 79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e로 새 execution capsule의 native source hash와 일치한다. 새 GPU에서 원 모델·tensor 전체를 독립 재실행한 감사는 아니다.
- [출처 manifest](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-method-review-2026-09-15/source-manifest.json), [재계산 CSV](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-reviewed-evidence.csv), [수식·산술 점검](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-design-checks.json).

## 3. PDF의 동기 문장을 먼저 갱신해야 한다

PDF abstract·§1 p.1, §3 p.3의 “short optimization + full optimization”, “10,000 edits”, “세 지표 개선”은 현재 확인한 REFIT4 후보 실험을 정확히 기술하지 않는다.

확인된 REFIT4는 W50/M50에서 시작한 B51–B60, 신규 1,000개다. Adam update 상한은 [24,24], loss 평가 상한은 [25,25], write 계수는 [.75,1]이다. 두 번째 target의 실제 조기 종료가 비용을 줄였으며 첫 target budget을 짧게 나눈 실험이 아니다.

동일 suffix의 terminal 수치를 재계산하면 다음과 같다.

| 정책 | 신규 RS / 1,000 | 신규 PS / 2,000 | 신규 P TF-strict / 2,000 | 신규 NS / 10,000 | Online / N4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| N4 | 1,000 | 1,938, 96.90% | 1,423, 71.15% | 7,108, 71.08% | 1.000 |
| REFIT4 | 999 | 1,950, 97.50% | 1,405, 70.25% | 7,175, 71.75% | 1.223 |
| FROZEN2 | 1,000 | 1,958, 97.90% | 1,467, 73.35% | 7,070, 70.70% | 1.067 |
| I2 | 1,000 | 1,961, 98.05% | 1,456, 72.80% | 7,086, 70.86% | 1.142 |
| FROZEN4 | 999 | 1,969, 98.45% | 1,487, 74.35% | 7,056, 70.56% | 1.193 |
| I4 | 999 | 1,966, 98.30% | 1,489, 74.45% | 7,056, 70.56% | 1.276 |

I2는 N4 대비 PS +1.15%p, P strict +1.65%p, NS −0.22%p다. REFIT4보다 계산이 적고 strict가 높다. 따라서 I2를 모든 목적에서 비합리적이라고 평가할 수 없다. I4는 FROZEN4보다 PS 3문항 적고 NS가 같으며 비용이 더 든다. 시험한 네 단계 continuation의 추가 효용은 약하지만 전체 step splitting에 대한 불가능성 증명이 아니다.

REFIT4도 PS·NS와 strict가 동시에 개선된 결과는 아니다. 원 논문 동기에는 신규 PS/NS 개선과 P strict −0.90%p를 같이 적는다. Full-seen에서 작은 순증은 old edit 손실과 상쇄될 수 있다.

N4는 BLUE-style AlphaEdit singleton L4, L2=1이다. blue=False, L4–L8, L2=10인 원 AlphaEdit와 이름을 섞지 않는다.

## 4. PDF 각 절의 처리

| 위치 | 판정 | 설계 반영 |
| --- | --- | --- |
| Abstract, §1 pp.1–2 | 실험 동기 사실 갱신 | 위 suffix 범위·실제 budget·strict 손실로 교체 |
| §2 식 (4)–(5) p.2 | 이상적 projected quadratic 해는 타당 | native source의 선형 map과 구분하여 구현 |
| §3 식 (6)–(8) p.3 | hybrid 표현과 고정 map endpoint 등가는 유용 | ODE solver 성능·새 write space의 증명으로 쓰지 않음 |
| §3 식 (9) pp.3–4 | matching-key 조건에서 타당 | canonical readout과 context key가 달라 실제 source에 자동 적용하지 않음 |
| §4 pp.4–5 | peak, endpoint, log penalty의 구분을 유지 | finite menu·sampled peak의 한계를 더 명확히 표기 |
| §5 식 (15) p.5 | 실제 write를 통한 피드백을 첫 구현으로 채택 | beam·schedule 탐색보다 먼저 평가 |
| §5 식 (17) p.5 | 조건부 연속시간 명제는 타당 | relaxed barrier와 finite screen에 보증을 전이하지 않음 |
| §6 pp.6–8 | 역할 분리·유한 후보 bound는 타당 | 첫 버전은 고정 64개 자체를 보호 집합으로 명명 |
| §7 pp.8–10 | 선행 연구 경계를 대체로 적절히 서술 | MetaKE와의 거리를 특히 좁게 평가 |
| §8 pp.10–11 | 평가 질문은 유용 | 전수 선행 gate 대신 단계적 W0 SEQ1000 비교로 재배치 |

### 4.1 Native map을 정확히 쓰는 방법

보존된 BLUE source는, key 수를 q, 요청 수를 n이라 할 때, 요청 residual을 필요하면 반복하여 다음 solve를 수행한다.

\[
Q=P(KK^\top+M_t)+\lambda I,\qquad
\Delta^\top=Q^{-1}PKT^\top R^\top .
\]

여기서 T는 요청 residual의 context별 반복을 나타내는 n×q 행렬이다. 따라서 Llama weight orientation에서

\[
\mathcal S_t(R)=RA_t,\qquad A_t=T(Q^{-1}PK)^\top .
\]

이 식은 inverse를 명시적으로 계산하라는 뜻이 아니다. 실제 solve·transpose·repeat 정책에서 선형 map을 추출한다. P가 정확한 orthogonal projector이고 history가 요구 조건을 만족하면 PDF의 이상적 해와 연결할 수 있지만, FP32 projector·solve를 검사하지 않고 이상식으로 교체하지 않는다.

원 source는 residual을 포함한 RHS를 solve한다. 미리 A를 계산하여 RA로 곱하는 방법은 실수 연산에서 같아도 FP32 kernel 결과가 다를 수 있다. Raw native endpoint는 native 연산으로 보존하고, gradient용 map과 실제 materialization의 차이는 기록한다. Factorization 재사용은 허용 가능한 최적화지만 수치적 동일성을 별도로 확인한다.

PDF가 이미 지적했듯 K는 context를 평균하거나 확장할 수 있고 H는 canonical prompt의 layer output이다. H(W+Δ)−H(W)=ΔK를 자동 전제하지 않는다. 입력 key의 고정과 “readout response가 바로 같은 K로 표현된다”는 주장은 다르다.

### 4.2 Residual gradient는 weight gradient의 orthogonal projection이 아니다

\[
\nabla_R F=\eta\nabla_WF A_t^\top
\]

는 고정 parent·A에서 정확하다. 이 gradient로 residual을 한 번 갱신하면, projection 전 weight 보정은

\[
\delta W=-\alpha\eta^2\nabla_W F A_t^\top A_t
\]

가 된다. 이는 writer가 유도하는 metric으로 변환된 방향이다. 일반적으로 weight Frobenius geometry에서의 orthogonal projection과 같지 않다. “Projected”는 target ball/trust projection인지, write family 제한인지 구체적으로 적는다. P만 곱한 임의 weight gradient보다 좁은 current-key write family에 머무른다는 설명은 타당하다.

## 5. Barrier 사양에서 바로 수정할 부분

### 5.1 C2 relaxed extension은 올바르지만 feasibility 보증은 아니다

v1의 함수는 τ에서 함수값·1차·2차 도함수가 이어진다.

\[
b'_\tau(h)=
\begin{cases}-1/h&h\ge\tau\\h/\tau^2-2/\tau&h<\tau,\end{cases}
\quad
b''_\tau(h)=
\begin{cases}1/h^2&h\ge\tau\\1/\tau^2&h<\tau.\end{cases}
\]

그 결과 위반 영역에서도 유한하고 유효한 gradient가 남는다. 다만 h=0에서 발산하지 않으므로 constraint 밖으로 나갈 수 있다. 작은 예로 D(w)=w, ceiling=1, E(w)=−100w, μ=1, τ=.1이면 w=1에서 gradient descent 방향은 +80이다. relaxed barrier만으로 경계를 유지하지 못한다.

이는 방법을 부정하는 반례가 아니다. 보존을 유도하는 penalty와 최종 actual-forward screen의 책임을 구분하는 반례다.

### 5.2 한 번의 barrier gradient는 slack으로 가중한 penalty와 동일하다

\[
h_a=(c_a-L_a)/s_a,\qquad
\nabla_R\{\mu_a b_\tau(h_a)\}
=\lambda_a(R)\nabla_R L_a,\quad
\lambda_a(R)=-\mu_a b'_\tau(h_a)/s_a>0 .
\]

평균·mass weight가 있으면 해당 계수도 포함한다. **한 지점에서 gradient 한 번만 쓰는 경우, 그 지점의 λ를 고정한 선형 penalty는 동일한 gradient와 동일한 projected correction을 만든다.** 작은 수치 예에서도 이 등가를 확인했다.

따라서 첫 버전의 차별점은 slack에 따른 상태 의존 가중이다. Fixed penalty와의 비교는 이를 검증하지만 log 함수만의 유일성을 입증하지 않는다. 이후 slope-matched adaptive penalty가 같다면 당연한 등가일 수 있다. “Barrier가 특별한 곡면을 찾아 우회했다”는 설명으로 확대하지 않는다.

### 5.3 Generic KL의 방향·정규화·관측 지점을 고정해야 한다

- p0는 W0 original teacher로 stream 전체에서 고정한다.
- 동일 prefix와 scored position에서 full-vocabulary KL(p0∥pW)를 계산한다.
- Chunk마다 scored-token 평균, 이후 선언한 mass-weighted 평균을 취한다. 길이가 다르면 단순 평균과 token 평균을 구분한다.
- Native target optimizer 안의 local KL·teacher는 원 source대로 유지하고, 추가 original-reference KL과 별도로 이름 붙인다.
- W0에서 정확한 teacher KL gradient는 0이다. **Native proposal을 반영한 provisional 모델**에서 gradient를 계산해야 첫 finite write의 손상을 관측한다. KL=0인 entry gradient로 chunk의 취약성을 이미 안다고 쓰지 않는다.

### 5.4 Core 통과와 전체 bank 통과를 분리해야 한다

64개에서 KL=0이고 나머지 448개에서 KL=1인 반례는 D64=0, D512=.875다. 작은 core의 screen은 전체 bank의 보증이 아니다.

첫 버전은 단순하게 **고정 64개 집합 자체에 대한 D64 ceiling**만 구현한다. 512개 bank는 필요시 endpoint의 별도 진단에 사용하며 D64 통과를 D512 통과로 표기하지 않는다. 이후 전체 bank acceptance를 주장할 버전은 최종 B100 후보를 512개 전부에서 한 번 검사하고 실패 시 batch 전체 weight를 복원한다. 그 전까지는 해당 보증을 붙이지 않는다.

Full-bank screen도 empirical bank의 committed endpoint에 관한 조건이다. 전체 분포, 모든 중간 보간점, 정답 지식의 유지 보증은 아니다.

### 5.5 새로 발견한 old-edit 위반은 명시적인 상태가 필요하다

Reservoir에 새로 들어온 old request가 이미 acceptance ceiling을 넘었으면 parent 자체가 infeasible하다. “모든 후보 탈락 → parent 유지”는 원 ceiling의 만족을 뜻하지 않는다. Entry loss로 ceiling을 올리면 누적 손상을 숨긴다.

Old-retention 확장에서는 원 ceiling을 고정한 채 다음 상태를 나눈다.

- Entry에서 feasible한 selected old request: 기존 ceiling 유지.
- Entry에서 이미 위반한 selected old request: 원 ceiling을 그대로 기록하고, 이번 batch에서 추가 악화를 허용하지 않는 restoration 조건을 따로 적용.
- 원 ceiling을 회복하지 못한 endpoint: original constraint PASS가 아닌 debt-nonincreasing 상태.

해당 상태에서도 all-request denominator와 acceptance coverage를 유지한다. Global old-retention 보증으로 부르지 않는다.

### 5.6 Acceptance, history, conflict는 서로 다른 규약이다

Weight commit 여부, 개별 요청의 성공, native key history, accepted-label ledger를 분리한다.

제안 규약은 처리한 B100의 key를 native 방식으로 한 번 등록하는 것이다. 모든 weight 후보가 탈락해도 observed-key history는 한 번 갱신한다. Accepted-label ledger에는 final canonical desired response가 TF-strict인 요청만 등록한다. 이는 v1의 모호한 rejected-batch 동작을 채운 설계 선택이며 성공 label 등록과 같지 않다.

Original teacher가 새 desired edit와 충돌할 수 있으므로 generic KL로 old desired loss를 대체할 수 없다. Exact subject/relation의 최신 요청을 기준으로 superseded target을 분리한다. Failed overwrite도 요청 실패로 원분모에 남긴다. Semantic conflict 전체를 해결했다는 주장은 하지 않는다.

## 6. ODE claim은 세 수준으로 나눈다

### 6.1 지금 허용되는 것은 hybrid dynamics 표현이다

PDF 식 (7)의 두 clock과 reset은 반복 target/write를 설명한다. 다만 원 target은 finite Adam이고 write는 큰 유한 변위다. 실제 m,v,t와 early-stop/reset을 생략한 gradient ODE는 수학적 완화 모델이지 원 알고리즘의 정확한 연속시간 실행이 아니다.

단일 gradient 보정, target-ball projection, 후보별 argmin, reject/backtrack을 가진 v1은 hybrid projected optimization이다. Write coefficient η, gradient trust ζ, ODE의 수치 step Δs를 별개로 둔다.

Proposition 1의 time-rescaling은 동일 autonomous field와 trajectory uniqueness 아래 타당하다. Hybrid reset까지 포함하려면 reset의 시각/guard도 같은 reparameterization을 따라야 한다. 물리 시각의 reset만 고정하고 rate를 바꿔도 같은 route라고 할 수는 없다.

### 6.2 실제 ODE solver를 주장하려면 고정 field·horizon을 정의해야 한다

한 stage의 parent와 A를 고정하여 Φ(R)=Wparent+ηRA, F(R)=Ecur(Φ)+μbτ(h(Φ))라 두면,

\[
\dot R=\Pi_{T_{\mathcal K}(R)}[-a(R)\nabla_R F(R)]
\]

라는 projected flow를 정의할 수 있다. K는 target ball과 declared trust set의 교집합이고 a는 고정 규칙의 양수 scale이다. Projection이 있으면 smooth ODE보다 projected dynamics/differential inclusion의 해석이 정확할 수 있다.

이 식은 **provisional model을 최적화하는 inner flow**다. R(0)=native residual이면 Φ(R(0))는 이미 native preview이므로 실제 W0에서 그 preview까지 이동한 전 경로의 보존을 증명하지 않는다.

Euler step 수 1/2/4를 비교하려면 horizon T, vector field, reset 수·위치, objective와 constraints를 고정해야 한다. Gradient 호출 수가 달라지므로 이 비교는 수치해석 진단이며 별도 wall-matched 성능 비교가 필요하다. η=.75/1을 나누는 I2/I4가 이 검증을 대신하지 않는다.

### 6.3 CBF로 실제 write velocity를 바꾸려면 별도 알고리즘이 된다

W(s)=Win+B(s)A로 쓰고 gD=∇W D Aᵀ, nominal residual velocity를 u0라 하면, 단일 generic constraint의 residual-space guard는

\[
\min_u\tfrac12\|u-u_0\|_F^2
\quad\text{s.t.}\quad
\langle g_D,u\rangle_F\le\kappa[b-D(W)]
\]

로 정의할 수 있다. gD≠0이면 해는

\[
u^\star=u_0-
\frac{[\langle g_D,u_0\rangle-\kappa(b-D)]_+}{\|g_D\|_F^2}g_D.
\]

이 구성은 native write family 안에서 방향을 실제로 바꾼다. 반면 gD=0인 infeasible 상태, 여러 old constraints, clamp의 tangent 조건, current progress 조건이 추가되면 feasibility와 solve가 달라진다. ε를 denominator에 임의로 더하면 정확한 부등식 만족도 더 이상 자동 보장되지 않는다.

초기 feasible state, 정확한 미분·연속 실행·필요한 regularity와 constraint 유지 조건 아래의 해석만 허용한다. Actual finite write는 별도로 검사해야 한다. 이는 현재 relaxed-barrier BG-1을 CBF라고 다시 이름 붙이는 변경이 아니라 후속 CBF-flow 방법이다. [Ames et al.](https://arxiv.org/abs/1903.11199)

## 7. Barrier bypass의 관측 주장도 좁혀야 한다

PDF §4의 endpoint frontier 우선은 타당하다. 동일한 최종 W의 신경망 출력은 동일하다. Inner history가 고정된 구현에서 “실제 중간 write”와 “같은 provisional W를 정확히 평가한 뒤 마지막에 한 번 materialize”는 계산상의 저장 방식만 다를 수 있다.

따라서 one-shot 대조는 물리적 copy 횟수를 비교하는 실험으로 정의하지 않는다. Fixed-parent endpoint optimization과 state-dependent target refresh의 알고리즘·정보·비용을 비교한다.

Finite candidate menu의 최소값은 더 넓은 path family infimum보다 높을 수 있고, sampled segment maximum은 continuous maximum보다 낮을 수 있다. 두 근사를 함께 쓴 최종 barrier estimate에는 일반적인 한 방향 bound가 없다. “샘플한 경로에서 peak가 낮았다”와 “continuous bypass가 증명됐다”를 구분한다.

또한 성공한 scaled-ray 후보가 탐색 grid에서 없다고 전체 ray에 성공점이 없다고 할 수 없다. Ray의 탐색 범위·성공 조건·probe 수·비용을 공개한다. 모든 current request의 E_i ceiling을 요구하는 PDF의 Gε를 쓰려면 실제로 그 개별 조건을 검사해야 한다.

## 8. Coreset은 비용 개선 단계에 두는 것이 맞다

PDF 식 (23)은 mass를 옮긴 동일 finite response menu에 대한 loss-value bound로 타당하다. 식 (24)는 주변 영역의 Lipschitz 상수가 알려져야 유용한 조건부 확장이다. 두 식 모두 새 gradient가 만드는 후보의 gradient accuracy를 보증하지 않는다.

첫 버전의 stratified64가 돌아가기 전에 response medoid·vulnerability·witness-trigger를 모두 구현할 필요가 없다. Medoid는 동일 총비용에서 평균 오차·false accept/reject·endpoint를 개선하는지 보며 추가한다.

Full-vocabulary FP32 teacher를 vocab=128,256, chunk당 128 scored tokens로 저장할 경우:

| Teacher 집합 | 최소 probability/log-prob tensor 저장량 |
| --- | ---: |
| 64 chunks | 3.914 GiB |
| 80 chunks | 4.893 GiB |
| 512 chunks | 31.313 GiB |

입력 token·cache metadata·일시 buffer는 별도다. 대표64+위험16이 모두 다를 때 10,240 scored tokens라는 산술은 맞다. 두 gradient pass의 20,480은 이 구성이 두 번 실제 실행된 경우의 control 노출량이며, 집합 중복이나 생략이 있으면 보편적인 “최소”가 아니다.

1–2개의 route gradient는 1–2개의 native z optimizer step과 동등한 비용 단위가 아니다. Full downstream backward, current·old tokens, rejected endpoint forward, teacher I/O를 포함한다. 1.5×는 목표 operating point이고 실측 성공 사실이나 모든 지표를 묶은 통과 gate가 아니다.

## 9. 선행 연구와 주장 범위

- [Baghel et al.](https://aclanthology.org/2025.findings-emnlp.798/)은 updated model에서의 반복 editing과 fixed-target 반복을 다룬다. 반복 호출 자체는 기여가 아니다.
- [MetaKE v3](https://arxiv.org/html/2603.12677v3)는 virtual post-edit의 edit/locality loss로 target을 보정하고 closed-form structural proxy를 사용한다. BG-TW와 가장 직접적으로 겹친다. 단일 L4에서 실제 executable map을 사용한다는 구현 차이만으로 독립적인 큰 novelty가 확보됐다고 쓰면 안 된다.
- [ODE-M v3](https://arxiv.org/html/2605.19409v3)는 continual model merging의 velocity rectification을 다룬다. 다른 task의 route 아이디어이며 knowledge editing에서 barrier가 존재한다는 근거가 아니다.
- [LyapLock](https://aclanthology.org/2025.emnlp-main.327/)은 sequential editing의 long-run preservation constraint를 다룬다. 그 목적·평균 제약과 고정 original-output KL의 finite endpoint ceiling을 구분한다.

검증할 기여는 **fixed-original response와 accepted desired edit를 제한된 native residual family에서 어떻게 제어하여, 계산량을 포함한 장기 endpoint trade-off를 개선하는가**다.

사용자가 지정한 다섯 baseline은 주표에 유지한다. 다만 그 다섯 개보다 좋다는 사실만으로 target-through-write·barrier·ODE 각각의 새로움을 입증하지 않는다. Same-data fixed penalty, scalar guard, edit-only executable gradient 및 matched-budget one-shot이 기전의 비교군이다. MetaKE-style 자체 비교군을 원 논문의 faithful reproduction으로 표시하지 않는다.

## 10. 다음 행동

실행 구조는 [W0부터의 단계적 설계 v2](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-from-base-staged-design-v2.md)에 고정한다. 첫 라운드는 다섯 baseline, W0 REFIT4, 최소 BG-1의 일곱 정책이다. 각 정책을 B100×10 실행하고 어떤 버전도 한 batch 성능으로 선별하지 않는다.

Audit128/MMLU68, E01 전수, NM4, landscape 전수 조사는 구현을 차단하는 선행조건으로 두지 않는다. 성능·기전 claim에 필요한 비교를 방법 실행과 이후 확증 단계에 배치한다.
