# JLZ 간섭 가격 기반 공유 예산 설계

v12-R의 실현 응답 경로와 native ridge writer를 유지하고, 요청별 에너지 제곱합 제약을 **다른 요청에 대한 직접 작용으로 가격을 매긴 공유 norm 예산**으로 교체한다. L4를 선택 규칙에 넣지 않는다. 어떤 층을 사용할지는 편집 손실의 gradient, 측정 가격, 기존 optimizer, cap의 결과로 결정된다.

이 문서는 method와 구현 경계를 확정한 연구 제안이다. 구현·GPU 검증·신규 실험이 완료됐다는 뜻은 아니다. Synthetic/toy 실험, 별도 W0 key 조사, 가격 순위 선행 gate를 두지 않는다. 실제 BS100×20 method trajectory의 첫 후보부터 가격·수치 정합·배분·누적 성능을 함께 측정한다. 실험은 [실험 계획](experiment-plan.md), 코드 변경은 [구현 계약](implementation-contract.json)에 정의한다.

## 출발점과 검증할 주장

기준 source는 `782c4c7a`이며, v12-R 실행 구현은 `635798ba`에서 도입됐다. [확정 W15 보고](../../../experiment-reports/servers/server4/jlz-v12r-realized-response-2k/intermediate-main-W15/report-ko.md)의 MAIN은 1,500 edits까지 검산됐다. 동일 first500 cohort의 PS는 W5 96.20에서 W15 83.00으로, NS는 85.08에서 69.64로 감소했다. B16의 로그 쓰기는 ENOSPC로 실패했으며 W20 결과는 없다. 이 실행 오류와 완료된 prefix의 유지 성능 저하는 구분한다. 1,500개 요청 중 1,445개가 SATISFIED_BASE였으므로 확장 controller만을 붕괴 원인으로 지정하지 않는다.

검증할 주장은 다음과 같다.

> 현재 batch의 다른 요청에 대한 native writer 응답으로 층별 비용을 정하고, 그 비용 아래에서 공동 최적화하면, 편집 획득 성능을 유지하면서 이후 편집의 유지와 이웃 보존을 개선할 수 있는가.

가격이 L4에서 최소라는 가정은 실행 조건이 아니다. L4가 최소가 아니어도 같은 알고리즘과 실험을 진행한다. 가격의 의미는 frozen-entry의 직접 작용 proxy이며, NS·최종 logit·lifelong 손상의 상한은 아니다. 사용자에게 제시됐던 층별 추정 비율과 CPU toy PASS는 이번 구현의 검증 결과로 상속하지 않는다.

## 상태와 기존 경로

한 batch에 B개 요청, eligible layers L={4,5,6,7,8}를 둔다. 각 층의 key 차원은 d_l, output 차원은 h_l이다.

| 기호 | 정의 |
| --- | --- |
| W_e, H_e | 이 batch 직전 실제 commit 상태와 native mean-key history |
| A_l | 기존 native 행렬 15000 C0_l + H_e,l |
| K_l | B개 요청의 rewrite-only native nested mean key, d_l×B |
| a_lr>0 | 기존 entry full-block subject output anchor norm |
| a_star,r | 기존 native anchor layer L8의 norm; norm loss에서 유지 |
| R_lr | 최적화하는 절대 output 좌표 요청, h_l 차원 |
| F_r | 기존 rewrite NLL과 native KL의 합; norm 항은 별도 |

모델·tokenizer·dataset 순서·native contexts·C0·H 갱신·loss reduction은 기존 profile을 그대로 묶는다. c=0.75, lambda_K=0.0625, lambda_N=0.5, lambda_C=15000, Adam lr=0.1, epsilon=1e-8, betas=(0.9,0.999), 최대 candidate 25개와 update 24개, tau_F=0.05를 유지한다. Norm loss는 sum_r (0.5/a_star,r²) sum_l ||R_lr||이다. 아래 예산의 beta와 이 norm 계수를 혼동하지 않는다.

매 candidate의 no-grad BUILD는 아래층의 실제 all-token weight write를 반영해 상위 key와 ridge P를 다시 계산한다. 실제 FP32 weight 차분이 만든 v를 entry-weight subject 경로에 주입한다. Production backward는 같은 층 Lambda M_row^T와 기존 norm gradient를 사용하고 dK/dP·builder.reverse는 사용하지 않는다. 요청 간 off-diagonal response와 subject 경로의 층간 gradient는 유지한다. Terminal에서 마지막 평가에 사용한 FP32 payload를 그대로 commit하고, 최종 rewrite mean-key history를 층마다 한 번 갱신한다.

## Batch entry 가격

R=0인 candidate 0의 BUILD에서 다음 행렬을 얻는다. 상위 P를 이후 candidate에도 고정한다는 뜻이 아니다.

\[
T_l=A_l+K_lK_l^\top,\qquad P_l=T_l^{-1}K_l,
\qquad M_l=P_l^\top K_l\in\mathbb R^{B\times B}.
\]

M_l은 **mean request 대 mean request**의 응답 행렬이다. Subject pullback의 B×전체 context-row 행렬과 구분한다. Native ridge가 이미 계산하는 mean M을 c0 반환값에 노출한다. Price 때문에 새 LM forward나 ridge solve를 수행하지 않으며, c0 BUILD를 중복 실행하지 않는다.

Donor r, receiver j에 대해 j≠r만 사용한다.

\[
d_{lj}=1-M_l[j,j],\qquad
X_l[r,j]=\frac{a_{lr}}{a_{lj}}\frac{M_l[r,j]}{d_{lj}},
\qquad
\kappa_{lr}=\sqrt{\frac1{B-1}\sum_{j\ne r}X_l[r,j]^2}.
\]

Receiver의 leverage d_lj로 나누는 식이다. Output 차원, context 수, 층 수로 다시 나누지 않는다. 동일 subject나 동일 target을 가진 다른 요청도 j≠r이면 포함한다. 따라서 이것은 unseen subject 가격이나 subject-group leave-out 가격이 아니라 **다른 요청 열에 대한 가격**이다. 유익한 교차 작용도 비용으로 센다.

Zero와 near-zero score의 정책은 다음으로 고정한다.

\[
\widetilde\kappa_{lr}=
\max\{\kappa_{lr},10^{-6}\max_q\kappa_{qr},10^{-12}\},
\qquad
\pi_{lr}=\widetilde\kappa_{lr}/\min_q\widetilde\kappa_{qr}.
\]

B=1 또는 요청 r의 모든 층 kappa가 0이면 pi_lr=1로 둔다. Floor는 단순 반올림 처리가 아니라 zero score의 명시적 정책이며, 원래 score와 floor 여부를 기록한다. 이 정의에서 1≤pi≤10^6이다. Floor 값으로 성능 sweep을 하지 않는다.

모든 anchor와 행렬값의 유한성을 확인한다. Production은 native raw A를 대칭화하거나 jitter를 추가하지 않는다. d_lj>1e-8을 수치 허용 조건으로 사용하고, 위반 시 PRICE_NUMERICAL_UNRESOLVED로 해당 batch를 commit하지 않는다. 작은 양의 d가 수학적으로 LOO의 부재를 뜻하는 것은 아니다. 분모를 조용히 clamp하거나 기존 uniform 가격으로 fallback하지 않는다.

가격과 anchor는 **해당 batch의 c0에서 고정**한다. 다음 batch에는 자기 arm의 commit된 W/H로 다시 측정한다. H가 가격에 반영되지만 과거 subject key를 직접 평가하는 replay/protection 항은 추가하지 않는다. 모든 층의 kappa가 같은 배수로 증가하면 상대 가격 pi는 변하지 않는다. 절대 누적 위험량을 자동 억제하는 정책이라는 주장은 하지 않는다.

## LOO 식이 정확한 범위

고정된 T,K,A에서 p_r=T^{-1}k_r라 하자. 현재 receiver 열 j만 Gram에서 뺀 행렬 T_-j=T-k_jk_j^T에 대해,

\[
p_{r,-j}=T_{-j}^{-1}k_r
=p_r+p_j\frac{M[r,j]}{1-M[j,j]},
\qquad
p_{r,-j}^{\top}k_j=\frac{M[r,j]}{1-M[j,j]}.
\]

이는 invertible 행렬의 rank-one inverse identity에서 직접 따른다. 위처럼 p_r의 작용으로 쓰면 native A의 비대칭 roundoff도 표기상 수용한다. SPD인 이상적 ridge에서는 흔히 쓰는 대칭 hat-matrix leave-one-out 형태와 같다. Ridge LOO의 관련 배경은 [Patil 등, 2022](https://proceedings.mlr.press/v151/patil22a.html)를 참고하되, 이 논문의 예측오차 이론을 LLM locality 보장으로 전용하지 않는다.

이 식은 j를 데이터에서 지우고 LM forward·anchor·H를 다시 만든 결과가 아니다. H에 같은 subject가 있으면 그대로 남는다. 한 donor의 상대 요청량 t_lr=||R_lr||/a_lr에 대한 직접 LOO 작용의 receiver RMS는 frozen-entry에서 t_lr kappa_lr이다. 따라서 아래 제약은

\[
\sum_l t_{lr}\kappa_{lr}
\le\min_l\widetilde\kappa_{lr}\,\beta_r
\]

라는 entry proxy의 합을 제한한다. 이 부등식은 측정 pi를 그대로 쓰는 PRICE arm에 대한 것이며, 가격을 바꾸는 FLAT/REVERSE에는 일반적으로 성립하지 않는다. 층별 hidden 작용을 같은 최종 출력 공간의 벡터로 더한 식이 아니므로, 층간 정렬을 측정하지 않아도 이 proxy는 정의되지만 최종 출력 손상의 triangle bound로 해석할 수 없다. 후보 BUILD가 K/P를 갱신한 후의 실제 작용에도 그대로 적용되는 상한은 아니다.

## 공유 예산과 확장

\[
\sum_{l\in L}\pi_{lr}\frac{\|R_{lr}\|_2}{a_{lr}}\le\beta_r,
\qquad \|R_{lr}\|_2\le c a_{lr},
\qquad \beta_{r,0}=c,\quad\beta_{r,\max}=c\max_l\pi_{lr}.
\]

균등 가격이면 v12형의 공유 상대 norm 제약으로 환원된다. **v12 전체 알고리즘으로 환원되는 것은 아니다.** 한 층이면 native clamp와 같은 feasible ball이다. BUILD·loss·optimizer까지 원 native solver와 동일하다는 뜻은 아니다.

최대 예산은 각 층을 단독으로 native cap까지 사용하는 점을 모두 포함한다. 여러 층의 cap을 동시에 모두 허용하지는 않는다. 가격 차가 크면 싼 층 여러 개와 비싼 층 일부를 함께 사용할 수 있으므로 희소 배분·L4 선택·차례로 다음 싼 층 사용은 보장하지 않는다. 현재 목적의 선형 근사에서 단위 비용당 이득은 a_lr||g_lr||/pi_lr이지만, 실제 EfficiencyAdam의 momentum과 좌표 정규화는 이 탐욕적 비율 선택과 다르다.

기존 요청별 controller를 dimensionless beta에 적용한다. 요청의 완료 update 수 t_r와 확장 stage e_r를 별도로 유지한다.

\[
\beta_r(e)=c\exp\left(\frac e4\log\max_l\pi_{lr}\right),\qquad e=0,1,2,3,4.
\]

- 모든 F_r를 매 candidate 평가한다. active |= (F_r≥0.05)이며 한번 active가 된 요청은 이후 satisfied여도 update한다.
- 초기 inactive 요청의 자기 R는 0이다. 다른 요청의 write로 F_r가 상승하면 이후 candidate에서 활성화될 수 있다. Subject loss mask와 reference mask는 동일하게 유지한다.
- Nonterminal 평가 이후, t_r≥12이고 현재 F_r≥0.05이며 e_r<4이면 다음 update 전에 한 stage 확장한다. 만족한 active 요청은 해당 step에서 확장하지 않는다.
- c0부터 계속 미달이면 update1–12는 base, update13–16은 stage1–4다. Candidate16에서 max 예산을 처음 평가한다.
- max pi=1이면 base=max이므로 확장하지 않는다. n_exp=0을 지원하는 code path는 base 고정이며 나눗셈을 하지 않는다. 이번 주실험은 n_exp=4다.
- Candidate24는 평가만 하고 backward·확장·update를 하지 않는다. 유한한 미달 요청도 마지막 후보를 commit한다. 낮은 성능은 기술 오류나 실행 중단 조건이 아니다.

## 절대 좌표의 정확한 사영

EfficiencyAdamAbs가 만든 제안 Z_lr에 대해 sum_l ||R_lr-Z_lr||²/2를 최소화한다. u=R/a 좌표에서 보통 Euclidean 사영을 수행하면 다른 방법이 된다.

\[
s_l=\|Z_{lr}\|,\quad w_l=\pi_{lr}/a_{lr},\quad D_l=c a_{lr},
\qquad t_l(\tau)=\min\{D_l,\max(s_l-\tau w_l,0)\}.
\]

Cap만 적용한 t_l(0)이 sum_l w_l t_l≤beta이면 tau=0이다. 아니면 sum_l w_l t_l(tau)=beta의 비음수 해를 구한다. Breakpoint (s_l-D_l)/w_l과 s_l/w_l의 비음수 구간을 정렬하면 층 수 5에서 정확한 piecewise-linear solve가 가능하다. 한 구간의 free와 capped 집합에 대해

\[
\tau=\frac{\sum_{l\in free}w_ls_l+\sum_{l\in capped}w_lD_l-\beta}
{\sum_{l\in free}w_l^2}.
\]

해가 breakpoint에 놓이거나 plateau 때문에 multiplier가 여러 개여도 projected vector는 유일하다. 분모가 0인 구간에서는 나누지 말고 다음 breakpoint 또는 이미 맞는 boundary를 선택한다. 최종 R_lr=t_l Z_lr/s_l이며 s_l=0이면 나눗셈 없이 0이다.

기하와 사영은 FP64, 저장 R와 모델은 FP32다. FP64에서는 tau≥0, clip 식의 길이 잔차≤1e-10 max(1,s_l,D_l), 공유 예산 초과≤1e-10 max(1,beta), complementarity 절댓값 |tau(spend-beta)|≤1e-10 max(1,tau beta)를 검사한다. FP32로 저장한 뒤에는 local absolute cap excess≤1e-6 및 dimensionless budget excess≤1e-6 max(1,beta)를 별도로 검사한다. 위반 시 수치 오류로 처리하고 결과를 본 뒤 허용치를 바꾸지 않는다. 별도 임의 shrink나 clip 반복으로 다른 projector를 만들지 않는다.

사영은 block 전체를 정확히 0으로 만들 수 있다. 그래도 그 층의 gradient, moments, per-layer EfficiencyAdam gamma를 제거하지 않는다. Norm 항의 zero-block gradient만 0으로 정의하며 task gradient에 의한 재진입은 허용한다. 사영 뒤 moments를 reset하지 않는다.

## 실제 batch 알고리즘

```text
freeze own Wentry/Hentry, anchors, KL teacher
R = 0; request optimizer state = 0
built0 = BUILD(R)                       # 첫 실제 후보, 추가 warmup 아님
raw_kappa, measured_pi = PRICE(built0.mean_M, anchors)
effective_pi = arm_transform(measured_pi)
controller = init(beta0=c, betamax=c*max_layer(effective_pi))
record one compact entry-price receipt

for k in 0..24:
    built = built0 if k == 0 else BUILD(R)
    F_all, subject_graph = EVALUATE_FORWARD(built, R)
    active |= (F_all >= .05)
    observe_all_requests_and_current_budget()
    if k == 24 or no_request_has_ever_been_active:
        break
    subject_adjoint = BACKWARD(sum(F_all[active]), subject_graph)
    g = same_layer_pullback(subject_adjoint, built) + norm_gradient(R)
    controller.expand_before_update(F_all, own_completed_updates)
    Z = EfficiencyAdamAbs(R, g, active)   # 모든 eligible layer 상태 유지
    R = capped_weighted_group_l1_projection(Z, effective_pi/a, beta, c*a)
    increment actual updated requests' t

commit exact last evaluated built.payload
append final native rewrite-mean history once per layer
```

EVALUATE는 기존처럼 terminal 후보에 backward를 수행하지 않는다. Initial zero-step early exit은 모든 요청이 inactive일 때만 가능하다. `built0` 재사용, loss mask 시점, terminal branch는 기존 엔진 API에 맞춰 구현하되 평가·update 순서를 바꾸지 않는다.

## 실제 method 실험으로 바로 연결

첫 비교는 PRICE, FLAT, REVERSE 세 arm만 수행한다. 세 arm 모두 같은 실현 응답 경로·optimizer·weighted-L1 projector를 쓴다.

| Arm | 가격 | 확인하는 차이 |
| --- | --- | --- |
| PRICE | 위 측정 pi | 제안 정책 |
| FLAT | 모든 pi=1 | 가격과 그에 따른 확장을 포함한 정책 전체의 차이 |
| REVERSE | 요청별 가격 multiset의 층별 rank를 반대로 배치 | 측정 가격과 올바른 층의 대응 효과 |

REVERSE는 각 요청에서 (측정 pi, layer index)로 오름차순 정렬한 layer 목록에 정렬된 price 값을 역순으로 배치한다. Anchor, gradient, layer cap, architecture 순서는 바꾸지 않는다. 동률 때문에 바뀌지 않은 assignment 비율을 기록한다. 각 arm은 자기 entry에서 가격을 다시 계산하므로, trajectory가 갈라진 뒤 arm 간 실제 budget·multiset·확장 시점이 동일하다는 주장은 하지 않는다.

FLAT의 max pi=1에서는 확장이 사라진다. PRICE 대 FLAT만으로 순수 가격 ranking의 인과효과라고 부르지 않는다. 새 ENERGY arm을 추가하지 않으므로 L2→L1의 독립 인과효과는 이번 실험의 claim이 아니다. 과거 v12-R·CD·baseline은 관측 범위와 profile 일치 여부를 붙인 역사 참고로만 비교한다.

가격 분포·최소 가격 층·floor 빈도·denominator·요청량·실현량·stage·zero-block 재진입을 실제 실행 중 기록한다. Terminal BUILD의 이미 있는 mean M으로 raw kappa 변화를 집계해 entry price의 stale 정도를 측정할 수 있으나, 이를 보고 같은 batch의 가격이나 예산을 변경하지 않는다. P/N 평가값도 optimizer·price·controller에 입력하지 않는다.

수치 계약은 **실제 B1의 cached c0 및 update tensor**에서 검사한다. Fixed donor/receiver (0,1),(1,0)의 LOO rank-one residual, projector feasibility/KKT, controller count, terminal payload/H once를 검산한다. 신규 synthetic solver benchmark, full-builder gradient audit, 독립 warmup fit, 별도 L4 우위 gate를 넣지 않는다. 실제 key에 대한 residual 검사도 추가 모델 forward나 ridge solve 없이 수행하고 비용을 기록한다.

성공 판단은 동일 cohort의 edit acquisition과 PS/NS·strict·NLL·lost/gained를 함께 본다. NS가 좋아져도 편집 성능이 떨어지면 먼저 보존–획득 tradeoff로 보고한다. 같은 norm·budget을 같은 편집 강도라고 부르지 않으며, 결과를 본 뒤 반경이나 permutation을 조절하지 않는다. 단일 순서·seed의 결과를 보편적 우위로 일반화하지 않는다.

## 구현 완료와 연구 결론의 구분

구현 변경은 과학적으로 entry price와 예산 교체이지만, 실제 코드에서는 price 노출·c0 순서·projection·controller·profile·telemetry·collector가 함께 바뀐다. BUILD·loss·pullback·writer를 다시 설계할 필요는 없다. [구현 계약](implementation-contract.json)이 각 변경 위치를 고정한다.

ENOSPC의 공간 소모 주체는 기존 보고에서 미확인이다. 새 실행은 static price를 entry에 한 번만 기록하고 candidate별 반복과 event 중복을 제거하며, 2k 전체 로그·평가 저장량과 batch 경계의 남은 공간을 확인한다. NoCP를 유지하고 큰 weight/key/history tensor를 영속화하지 않는다. 이 저장 계약은 재실행 낭비를 막는 실행 조건이며 가격 성능을 고르는 gate가 아니다.

현재 상태는 DESIGN_READY다. 이 문서 작성 과정에서 toy·실제 GPU 실험을 실행하거나 기존 job을 변경하지 않았다. 구현과 실험 dispatch가 수행되면 해당 source/config hash와 실제 상태를 별도 receipt로 기록한다.
