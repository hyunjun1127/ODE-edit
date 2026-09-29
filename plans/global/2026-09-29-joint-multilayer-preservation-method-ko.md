**누적 보존 제약 아래의 다층 공동 순차 편집 — METHOD 초안, 2026-09-29**

사용자 지시: 한 edit에서 여러 층을 함께 사용하는 METHOD를 다시 설계한다. 이 문서는 방법의 상태·변수·목적·해법·상태 전이만 다룬다. 실험 cell, 편집 수, checkpoint 선정, 비교군 또는 실행 계획은 만들지 않는다.

제안은 **AlphaEdit의 write subspace 안에서, 원본 지식과 유효한 과거 편집에 대한 누적 손상 상한을 지키도록 여러 층의 update 방향과 크기를 공동 결정하는 online editor**다. 층별 독립 손상 점수에 비례해 residual을 나누는 규칙을 두지 않는다. 배분은 현재 상태의 constrained joint solve 결과다.

**기존 방법에서 재사용하는 것과 이번에 명시하는 것.**

기존 A0는 이미 entry geometry에서 ΔW_l=R_l J_l로 실제 다층 weight를 함께 바꾸며 R4/R8을 공동 최적화했다. 9/19 문서에도 Base/Past functional 목적, 편집 품질 제약, 실제 endpoint 수용 조건이 제안돼 있다. 따라서 공동 residual vector, 실제 weight forward, 보존 loss 자체를 새로운 기여라고 주장하지 않는다. [기존 A0 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-19-joint-z-dynamic-allocation/prior-joint-evidence-ko.md), [기존 공동 목적](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-19-joint-z-dynamic-allocation/math-and-cost-ko.md)

9/27 composition/CARA 감사에도 fixed W0 teacher의 warm gradient, cross-layer curvature, augmented Lagrangian, 실제 endpoint 수용 조건이 이미 명시돼 있다. 이 구성도 재사용 대상이다. [기존 composition 수학 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-27-composition-adaptive-allocation-review/math-method-audit-ko.md:269), [CARA 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-27-cara-final-readiness-review/review-ko.md:130)

이번에 구체화하는 연결은 **anchor의 생성·유지·폐기 → 절대 누적 상한 → 현재 남은 여유 → 시간축 multiplier warm start → feasible 후보를 찾지 못했을 때의 처리**다. 매 edit 진입 손실에 허용 악화량을 더하는 규칙과 구별한다. 현재 오차에 대한 signed marginal risk와 다층 상호작용은 기존 뼈대로 유지하고, 실제로 수용된 모델·유효 history·제약 상태를 다음 요청으로 전달한다. 이 연결의 신규성이나 성능 우위는 확정하지 않는다.

**1. 순차 정책의 action을 공동 update로 바꾼다.**

θ0는 원본 모델, θt는 새 요청 e_t 직전 모델, L은 관리하는 후보층 집합이다. 상태는 θt, 원본 보호 reference, 최신 유효 history ledger, key-side writer 통계, 보존 상한 및 multiplier로 구성한다.

\[
\pi(S_t,e_t)=\{\Delta W_{t,\ell}\}_{\ell\in\mathcal L},\qquad
\theta_{t+1}=\theta_t+\Delta_t.
\]

한 요청의 모든 후보층을 공동 최적화 변수로 열고 한 endpoint로 수용한다. Transformer forward의 계산 순서는 원래대로다. ‘동시’란 독립 편집을 병렬 계산해 합친다는 뜻이 아니라, 모든 변경 weight를 함께 넣은 모델에서 같은 최종 편집 목표를 만족시킨다는 뜻이다.

편집 횟수·weight norm에 따라 층을 가득 찬 저장소로 판정하지 않는다. 층별 고정 사용률이나 합이 1인 배분 비율도 두지 않는다. 여러 층을 함께 쓸 수 있지만 모든 층의 nonzero update를 강제하지는 않는다. 한 층만 필요하다는 해도 가능한 공동 action 공간이며, 방법을 단층 선택으로 제한하는 것은 아니다.

**2. ‘어디에’와 ‘무엇을’을 같은 변수에서 결정한다.**

각 층의 entry-state AlphaEdit key-side map이 허용하는 write의 입력 방향을 Q_{t,l}로 놓는다. 고정 map의 행공간을 rank-revealing QR/SVD로 정규직교화하여 QᵀQ=I로 정의한다. 이는 작은 writer factor의 basis 정리이며 전체 모델 weight의 spectral 분해를 요구하지 않는다.

\[
\Delta W_{t,\ell}=U_{t,\ell}Q_{t,\ell}^{\top},\qquad
U_{t,\ell}\in\mathbb R^{d_{out}\times r_\ell}.
\]

Q는 해당 요청의 live key와 기존 AlphaEdit projector/history 통계로 만든다. 한 요청의 inner solve에서는 Q를 고정하고, 모든 U를 공동 최적화한다. 다음 요청에서는 새 상태에서 다시 만든다. U의 방향까지 자유롭게 바뀌므로, 이미 만든 d_l에 scalar a_l을 곱하는 방법보다 검색공간이 크다. 층마다 독립 compute_z target을 만든 뒤 그것을 달성시키지 않고, 최종 새 목표답에 대한 하나의 기능 목적을 사용한다.

추가 gate g_l와 U_l을 동시에 학습하지 않는다. g_l U_l에는 scale 비식별성이 있으며 배분 변수가 중복된다. 정규직교 Q 아래에서는 ||U_l||F=||ΔW_l||F이므로 실제 write 크기와 계수 크기가 일치한다.

이는 AlphaEdit의 검색공간을 재사용하는 새 공동 writer다. Native AlphaEdit의 solve/target 알고리즘과 수치적으로 동일하다고 표현하지 않는다. 제한된 subspace가 필요한 편집·복원 방향을 포함한다는 보장도 없다.

**3. 보호 기준을 고정하고, 누적 손상의 여유를 정의한다.**

원래 행동은 별도 보존 입력 B의 원본 분포 p0로 보호한다.

\[
R_B(\theta)=\mathbb E_{x\in\mathcal B}
\operatorname{KL}(p_0(\cdot\mid x)\|p_\theta(\cdot\mid x)).
\]

같은 teacher-forced prefix·정해진 출력 위치에서 비교한다. 의도적으로 갱신되는 사실과 정의된 변경 범위는 보존 reference와 충돌하지 않도록 정리한다. Reference 범위를 바꾸면 무엇을 해제했는지 ledger에 남기며, 낮은 손상 점수를 만들기 위해 임의로 보호 대상을 제거하지 않는다. KL 보존은 해당 입력 분포의 행동 보존이며 모든 능력이나 정답률의 보장은 아니다.

과거 edit h는 최신 유효 목표 y_h에 대한 손실 ℓ_h(θ)로 보호한다. 성공적으로 수용할 때 a_h=ℓ_h(θ_accept(h))를 기록하고 이후 상한 b_h=a_h+ε_h를 고정한다. 공식은

\[
\ell_h(\theta)\le b_h,\qquad h\in\mathcal H_t^{valid}.
\]

원본의 옛 답으로 돌아간 것을 history 복원으로 보상하지 않는다. 같은 사실의 새 목표가 들어오면 이전 버전을 superseded로 바꾸고 새 목표를 수용한 뒤 새 anchor를 만든다. History 전체의 평균 하나만 쓰면 신규 항목이 기존 망각을 희석하므로, 기본 방법은 항목별 상한을 둔다.

Base 상한 b_B는 방법 시작 때 한 번 정한다. 이미 손상된 checkpoint에서 시작한다면 원본 reference는 그대로 두고, 그 시작 위험과 허용 여유를 기준으로 초기 feasible 상한을 정할 수 있다. 기존 history의 과거 수용 anchor를 확보하지 못했다면 시작 시점의 유효 목표 손실로 초기 anchor를 명시적으로 만든다. 이 경우 보존 범위는 방법 시작 이후이며 기존 손상을 복구했다는 뜻이 아니다.

현재 여유는 d_{B,t}=b_B−R_B(θt), d_{h,t}=b_h−ℓ_h(θt)다. **상한을 매 edit 현재 손실+ε로 재설정하지 않는다.** 여유가 소진된 뒤에도 새 편집을 위해 계속 ε를 지급하면 누적 안정성 조건이 사라진다. 수용 시점에 가능한 범위에서만 anchor를 추가하며 신규 실패 사례를 보존 성공으로 세지 않는다.

**4. METHOD의 중심 최적화 문제.**

새 목표의 기능 손실을 L_e, 새 편집 품질 상한을 τ_e라 하자. 모든 층에 걸친 실제 write energy는 Ω(U)=½Σ_l||U_l||F²로 둔다. 후보층의 크기나 parameter scale이 크게 다르면 고정된 reference metric으로 먼저 좌표를 정규화하고, inner solve 중 그 metric을 바꾸지 않는다.

\[
\begin{aligned}
\min_{\{U_\ell\}}\quad &\Omega(U)\\
\text{s.t.}\quad
&L_e(\theta_t+\Delta(U))\le\tau_e,\\
&R_B(\theta_t+\Delta(U))\le b_B,\\
&\ell_h(\theta_t+\Delta(U))\le b_h,
\quad h\in\mathcal H_t^{valid},\\
&\|U\|_F\le\rho_t.
\end{aligned}
\]

편집을 약하게 해서 보존만 좋아지는 해는 편집 품질 제약으로 제한한다. 원본 보호와 과거 edit 보호는 별도 제약이며 서로 상쇄하는 하나의 임의 weighted score로만 합치지 않는다. Norm은 기능 손상의 대리값이 아니라, 제약을 만족하는 해 중 불필요한 큰 write와 층간 과도한 상쇄를 줄이는 tie-break 및 trust-region 통제다.

최소 write를 택해도 보이지 않은 입력에서의 큰 상쇄를 완전히 막을 수는 없다. Hidden activation을 모두 원래 값으로 강제하면 합법적인 새 기능까지 막을 수 있으므로, 기본 방법은 기능 제약과 작은 공동 write를 사용한다.

**5. 누적 손상이 실제로 배분을 바꾸는 경로.**

현재 출력 logits z_t에서 원본 KL의 미분은 p_t−p0다. 계수 U_l에 대한 Jacobian을 J_{t,l}로 두면

\[
g^B_{t,\ell}=J_{t,\ell}^{\top}(p_t-p_0).
\]

현재 모델이 원본에서 얼마나 벗어났는지와 그 층의 다음 write가 어떤 출력 방향으로 움직이는지가 곱해진다. 이미 변형이 큰 층이라도 새 방향이 오차를 줄일 수 있고, 거의 쓰지 않은 층도 보호 기능에 민감하면 비쌀 수 있다. ‘층 전체 손상도’ 하나로 이 둘을 판정하지 않는다.

보호 loss의 국소 전개를 공동 계수 u=concat(vec(U_l))로 쓰면

\[
\Delta R_c\approx g_{c,t}^{\top}u
+\tfrac12\sum_{\ell,m}u_\ell^{\top}H_{c,t;\ell m}u_m.
\]

첫 항은 현재 누적 오차의 강화/상쇄, 대각항은 해당 층 변화의 민감도, 비대각항은 공동 write의 상호작용이다. 매 edit live 모델을 새 teacher로 삼으면 KL의 시작 gradient가 0이므로 첫 정보를 잃는다. 현재 geometry를 다시 읽는 것과 보호 기준을 현재 모델로 바꾸는 것은 다르다.

H는 설명을 위한 실제 loss Hessian이다. GGN은 근사이며 잔차를 생략한다. 방법은 dense Hessian을 명시적으로 만들 필요 없이 실제 공동 forward의 autodiff로 구현할 수 있다. Block-diagonal 근사나 층별 독립 위험표만으로 정책을 완전히 대체하면 비대각 상호작용을 잃는다.

**6. 여러 층을 동시에 쓸 때 반드시 살아 있어야 하는 경로.**

층 j의 live 입력을 k_j, 앞층 공동 변화로 달라진 입력을 δk_j라 하면

\[
(W_j+\Delta W_j)(k_j+\delta k_j)-W_jk_j
=\Delta W_jk_j+W_j\delta k_j+\Delta W_j\delta k_j.
\]

W_j=W0,j+E_j로 나누면 E_j δk_j는 앞층 변경이 뒤층의 기존 편집을 다른 입력에서 읽게 하는 성분이다. 앞층 write와 뒤층 write가 같은 최종 기능을 함께 실현하거나 서로 보상할 수 있다. 각 층에 고정 key만 넣어 얻은 local 비용을 합하면 이 경로를 빠뜨린다.

Q를 고정하는 것은 허용하지만, **실제 forward의 k_j를 entry 값으로 고정하면 안 된다.** 모든 token 위치에서 실제 W_j+U_jQ_jᵀ를 사용하는 forward를 한다. Low-rank 형태로 계산한다면 추가 출력은 U_j(Q_jᵀk_j(U_<j))여야 하며, subject 위치에 상수 vector를 주입하는 것으로 대체하지 않는다. 별도의 key-drift penalty를 더하지 않아도 위 기능 목적에는 이 경로가 이미 들어간다.

**7. 배분 정책은 제약의 가격을 이용한 공동 solve다.**

편집·base·history 제약을 c_i(U)≤0로 적고 라그랑지안을

\[
\mathcal L(U,\lambda)=\Omega(U)+\lambda_e c_e(U)
+\lambda_B c_B(U)+\sum_h\lambda_hc_h(U),\quad\lambda\ge0
\]

로 둔다. Trust-region 경계가 비활성이고 Ω=½||U||²인 정칙 stationary point에서는 층별로

\[
U_\ell+\lambda_e\nabla_\ell L_e
+\lambda_B\nabla_\ell R_B
+\sum_h\lambda_h\nabla_\ell\ell_h=0.
\]

보존 여유가 적은 기능의 제약이 활성화되면 그 기능을 해치는 층·방향에 높은 한계 비용이 생긴다. 같은 새 목표를 더 적은 제약 위반으로 달성하는 다른 층·방향이 있으면 공동 해가 그쪽으로 이동한다. λ는 전역 기능 제약의 shadow price이며 독립적인 층별 capacity가 아니다. 이 방향 선택은 다른 층의 update에도 의존한다.

실제 해법은 projected multiplier update를 갖는 augmented Lagrangian과 trust-region/line-search primal step으로 정한다. Inequality augmented term은 β>0에 대해 ([λ_i+β c_i(U)]_+²−λ_i²)/(2β)로 둘 수 있다. Primal step에서는 모든 U를 함께 갱신하고, dual step에서는 λ_i←[λ_i+β c_i(U)]_+로 갱신한다. 같은 inner iteration의 전 층이 같은 endpoint와 제약값을 사용한다.

원래 편집을 아직 못 하는 entry는 infeasible일 수 있으므로, feasible 해를 찾는 단계와 feasible 해 중 작은 write를 찾는 단계를 구별한다. 마지막 optimizer iterate를 자동 수용하지 않고, 실제 기능 제약을 만족하는 후보 중 가장 작은 Ω를 수용한다. Line search/trust region은 augmented objective 또는 feasibility 개선을 확인하며, 최종 수용은 원래 제약으로 확인한다. 비볼록 문제에서 이 절차의 전역 수렴이나 모든 요청의 해 존재를 보장하지 않는다.

**8. 한 요청이 끝난 뒤의 상태 전이.**

1. 새 요청과 충돌하는 이전 사실 버전을 ledger에서 정리하고, live 상태에서 각 층 Q를 구성한다.
2. 모든 U를 0에서 시작해 공동 기능 제약 아래 최적화한다. 편집과 보호 제약을 이미 만족하면 zero write로 끝낼 수 있다.
3. 모든 변경층을 함께 적용한 모델에서 실제 편집·base·history 조건을 확인한다. 실패하면 수용하지 않으며 보존 상한을 자동으로 높이지 않는다. 이는 현재 solver/subspace가 feasible 후보를 찾지 못했다는 뜻이지 전역 불가능성의 증명은 아니다.
4. 수용한 Δ를 함께 materialize하고 새 모델을 다음 θ로 삼는다. 새 목표의 수용 anchor와 유효 버전을 기록한다.
5. Native 누적 Gram을 key-side preconditioner로 계속 쓸 경우, 새 요청의 post-write key를 **모든 관리층**에서 한 번씩 추가한다. 다른 층에 기록된 지식도 향후 어느 층의 write에 의해서든 훼손될 수 있기 때문이다. 중복 append를 막는다.
6. Base/history multiplier는 다음 요청의 warm start로 전달할 수 있다. 새 요청의 edit multiplier는 새 제약에 맞춰 초기화한다. Budget 자체는 재지급하지 않는다.

Step 5의 Gram은 획득 시점 key들의 통계다. 과거 key를 매번 재추출한 live covariance라고 부르지 않는다. 과거 사실의 supersession은 functional ledger에서 처리하고, Gram에서도 제거하려면 항목별 기여나 rebuild 정보가 추가로 필요하다. 그것이 없다면 남은 오래된 Gram은 보수적인 preconditioner 근사로 명시한다. 실제 history 보호의 정의는 현재 모델의 유효 목표 loss다.

이렇게 π는 θt·누적 오차·남은 보존 여유·현재 geometry에 의존하는 시간축의 공동 배분 정책이 된다. 미래 요청의 value function을 최적화하지는 않는다. Stateful greedy constrained control이며, 장기 최적 routing이나 무한 편집 안정성을 주장하지 않는다.

**선행연구에서 가져오는 범위.**

[AlphaEdit](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html)는 보호 key에 대한 write subspace 구성의 출발점이다. 고정 key의 projection 보호를 실제 다층 함수 보존 정리로 확대하지 않는다.

[LyapLock](https://aclanthology.org/2025.emnlp-main.327/)은 순차 편집의 누적 보존을 명시적 제약과 단계별 최적화로 다루는 근거다. 본 방법의 fixed cap·항목별 제약·primal-dual solve는 여기서 제안한 설계이며 LyapLock의 queue 알고리즘이나 이론적 보장을 그대로 가져온 것이 아니다.

[CrispEdit](https://arxiv.org/abs/2602.15823)는 실제 capability 기능과 그 곡률을 보호 대상으로 삼는 근거다. 본 설계의 live warm-state 선형항과 joint cross-layer 비용을 논문이 이미 이 형태로 제안했다고 주장하지 않는다.

[EvoEdit](https://aclanthology.org/2026.findings-acl.75/)는 누적된 편집도 보호 대상으로 반영해야 한다는 관련 선행이다. Projector의 과거 key 누출과 동일 prompt의 upstream-induced key drift를 구별한다. [HiEdit](https://aclanthology.org/2026.acl-long.1855/)는 순차 편집과 동적 층 선택을 이미 결합했으므로, ‘lifelong editing에 층 선택을 도입했다’는 신규성 주장도 성립하지 않는다.

**현재 설계의 범위.**

방법은 정해진 보호 reference와 유효 history를 실제로 재평가할 수 있다는 전제 아래 완결된다. 모든 history 제약을 확인하면 비용은 이력과 함께 증가한다. 제한된 replay memory나 sampling을 쓰면 확인하지 않은 항목의 보존은 근사이며, 고정 비용으로 전체 history를 보호한다고 주장할 수 없다. Native Gram만으로 functional history 제약을 대체하지 않는다.

구체적인 보호 입력 분포·허용 손상·새 편집 품질은 방법의 요구 조건이다. 그것들을 모른 채 어떤 층이 안전한지 자동 판정할 수는 없다. 또한 허용 방향과 보존 제약이 새 목표와 양립하지 않으면 공동 다층 공간에서도 실패할 수 있다. 이 초안은 모든 요청을 반드시 수용한다는 정책보다 누적 보존 상한을 우선한다.

방법의 핵심 문장은 다음과 같다. **현재 누적 손상에 더해지는 공동 기능 변화가 보존 여유를 넘지 않도록, 여러 층의 실제 write 방향과 크기를 함께 정하고 그 결과를 다음 요청의 상태로 넘긴다.**
