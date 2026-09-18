# REFIT feedback 기전 점검과 single-layer barrier-guided write 설계

2026-09-16. 사용자 요청에 대한 조사·설계 산출물. 기존 CSV·소스와 공개 논문을 읽고 CPU 산술을 검산했다. 새 모델 평가·GPU 실험·서버 지시·기존 SL-ZFlow 구현 변경은 수행하지 않았다.

**권고는 single L4의 고정 writer geometry에서 실제 출력 feedback으로 write를 조정하고, 필요한 요청만 target을 refresh하는 것이다. REFIT은 이 방향의 동기지만, ODE의 우월성이나 매 step z 재최적화의 필요성을 입증하지 않는다.** 특히 REFIT의 추가 Adam은 1,000요청에 얇게 분산된 것이 아니라 **853요청 0회 / 146요청 24회 / 1요청 21회**다. 성공한 요청의 demand를 현재 상태로 재설정하는 효과와 부족한 요청을 다시 최적화하는 효과를 분리해야 한다.

## 1. 확인된 실험 범위와 연구 주장

[fixed10k lifelong 정본](../../local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md)의 최종 값은 다음과 같다.

| 정책 | RS | PS | NS |
|---|---:|---:|---:|
| AlphaEdit BLUE L4+L8 | 98.88% | 95.775% | 63.726% |
| AlphaEdit L4-only | 99.39% | 95.680% | 65.348% |
| CAKE native | 98.40% | 88.775% | 62.935% |

CAKE 값은 접근 가능한 [완료 사실 보고](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-sh4-cake-baseline-cap-report-split-20260916-v1/experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)의 동일 fixed10k 표에서 확인했다. 사용자가 언급한 제목의 독립 정본 v2 자체를 읽었다고 주장하지 않는다. CAKE의 PS 차이는 L4-only 대비 −6.905%p다. 설정 차이를 통제한 layer allocation 단독 효과는 아니다.

BLUE의 L4 batch update norm 평균은 10.3723, L4-only는 10.3814다. BLUE L4의 batch별 norm 비중 평균은 73.803%, L8은 26.197%다. 따라서 L8이 작동하지 않았다는 해석은 부정확하지만, L4 write 크기를 실질적으로 줄이는 분담 정책이 아니었다는 관측은 타당하다. Frobenius norm은 기능적 edit 기여율이 아니다.

연구 주장은 **현재 정책에서 edit 달성과 locality 보존의 병목이 다르고, BLUE의 추가 layer 정책이 가용 자유도를 순 preservation 이득으로 전환하지 못했다**로 둔다. 단일층의 intrinsic preservation capacity가 포화됐다는 수학적 결론은 아직 아니다. RS/PS는 두 후보 간 선호이며, L4-only의 최종 paraphrase TF strict는 66.810%이므로 절대 정답 품질까지 거의 완벽하다고 표현하지 않는다.

## 2. REFIT에서 무엇이 바뀌었는가

[원 six-arm 리뷰](../../audits/global/2026-09-14-lowcost-seq10-review-ko.md)와 [후속 write-refresh 완료 보고](../../local/reviews/bg-tw-method-review-2026-09-15/source/experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1/diagnostic-report-ko.md)는 공통 W50/M50에서 B51–B60의 신규1,000요청을 처리했다. 아래는 final W60에서 신규 suffix만 평가한 값이다.

| 정책 | 새 P 선호 성공 / 2,000 | 새 P TF strict / 2,000 | 새 N 성공 / 10,000 | 계측 포함 online / N4 |
|---|---:|---:|---:|---:|
| N4 | 1,938 (96.90%) | 1,423 | 7,108 (71.08%) | 1.000 |
| REFIT4 | 1,950 (97.50%) | 1,405 | 7,175 (71.75%) | 1.223 |
| FROZEN2 | 1,958 (97.90%) | 1,467 | 7,070 (70.70%) | 1.067 |
| I2 | 1,961 (98.05%) | 1,456 | 7,086 (70.86%) | 1.142 |
| FROZEN4 | 1,969 (98.45%) | 1,487 | 7,056 (70.56%) | 1.193 |
| I4 | 1,966 (98.30%) | 1,489 | 7,056 (70.56%) | 1.276 |

REFIT의 P/N 선호 개선은 +0.60/+0.67%p지만 P strict는 −0.90%p다. FROZEN2는 첫 absolute z를 유지한 채 현재 residual을 다시 fitting한다. I2/I4는 최초 anchor·teacher·clamp 기준과 optimizer state를 유지하며 target 최적화를 나눈다. REFIT은 현재 weight에서 native target 최적화를 새로 시작한다. 따라서 위 결과는 **fresh reset 구성에 preservation상 가치가 있을 가능성**을 보여주지만 fresh gradient 하나, teacher reset 하나, ODE 자체의 효과를 식별하지 않는다.

REFIT은 전체 suffix N 이득 +67 중 +28이 at-write 차이, +39가 이후 순변화 차이다. 또한 old active R/P의 net −1/−6과 신규 P strict 손실이 있다. “L4-only를 전 지표에서 이겼다” 또는 “1만 edit에서 검증됐다”로 확대하지 않는다.

**추가로 직접 재집계한 사실:**

- REFIT 첫 target은 1,000개 모두 Adam24회다.
- 두 번째는 853개 0회, 146개 24회, 1개 21회, 총3,525회다.
- 양의 update를 받은 147개만의 평균은 23.9796회다.
- N4/REFIT4/FROZEN2/I2/FROZEN4/I4의 공개 raw-writer-history-links에서 **B51–B60 매 batch의 entry M 및 endpoint M 해시는 6정책 모두 같다.** W trajectory는 달라도 이 기록의 M trajectory는 같다. 기존 보고의 포괄적 “W/M/target이 달라진다”는 표현에서 M은 이번 비교에 적용되지 않는다.

[재집계 결과](../../audits/global/2026-09-16-refit-feedback-mechanism/checks.json)는 공개 해시의 일치 검산이다. 이번 호스트에서 raw M/K tensor를 다시 읽은 독립 검증은 아니다.

## 3. key refresh 설명은 단일층 구조와 맞지 않는다

수정 parameter를 L4 `mlp.down_proj.weight` 하나로 고정하면, 고정된 token IDs·position·mask·context에 대해

\[
k_4(x)=\mathrm{MLP\ intermediate}_4(x),\qquad
\frac{\partial k_4(x)}{\partial W_{4,down}}=0.
\]

입력 key는 down projection보다 앞에서 계산된다. 같은 layer의 down projection을 바꿔도 앞선 attention, gate/up projection 또는 다른 token의 그 layer 입력으로 되돌아가지 않는다. 따라서 동일 입력이면 K는 구조적으로 불변이다. 재생성한 텍스트, 다른 upstream 수정, dropout·backend 변경은 별도 조건이다. BLUE의 L8 key는 앞선 L4 수정으로 달라질 수 있지만 L4-only에서는 해당되지 않는다.

Native compute_ks가 매번 호출되는 것은 재계산이며 새로운 key 정보가 생겼다는 뜻이 아니다. 고정 context와 inner-loop P/M에서 native solve geometry도 고정된다. History는 batch 종료에서 한 번만 append한다.

반대로 **현재 activation Y, suffix logits, loss, gradient, native target initialization은 변한다.** Native compute_z는 호출마다 delta=0, 새로운 Adam, 현재 clean activation a, 현재 essence teacher를 준비한다. delta penalty의 분모와 clamp 반경도 현재 a를 기준으로 한다. 총 loss<0.05이면 backward 전에 종료하며 target=a+delta를 반환한다.

0-step이면 z₂=a(current W)다. 최초 z₁을 재사용하는 것이 아니다. 초기 KL·delta penalty가 0인 조건에서 이 분기는 현재 native rewriting context 평균 NLL이 이미 종료선을 통과했음을 나타낸다. “첫 z를 끝까지 실현해야 한다”는 demand를 내려놓는 효과가 생긴다.

단 a는 target-optimization 입력의 clean subject 위치이고 writer Y는 canonical prompt readout이다. subject까지 token prefix·위치·mask가 같으면 인과적 forward상 같지만, tokenization/batch 수치까지 확인하지 않고 z₂−Y=0이라고 단정하지 않는다. 이 차이는 아래 최소 진단에서 직접 계측한다.

Native 코드는 [보존 compute_z](</mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_z.py>)와 같은 source family의 실제 [REFIT fitter](../../local/reviews/refit4-seq10-review-2026-09-14/source/project/run_scripts/low_cost_write_donor_pilot/fitting.py), [호출 순서](../../local/reviews/refit4-seq10-review-2026-09-14/source/project/run_scripts/low_cost_write_donor_pilot/sequential_runtime.py)에서 확인했다. Fitter는 target/solve를 유지하고 history append만 최종 단계로 분리한다.

## 4. 반복 write의 정확한 의미

한 batch에서 writer K∈R^(d_in×n), canonical key Kc, projector P, entry history M을 고정한다. 현재 pinned singleton은 요청당 평균 key 하나다. 일반적인 residual 반복이 필요하면 아래 A에 해당 expansion을 포함한다.

\[
Q=P(KK^\top+M)+\lambda I,\quad
A=[\operatorname{solve}(Q,PK)]^\top,\quad
\Delta W=RA.
\]

Q는 일반적으로 비대칭이며 임의로 대칭화하지 않는다. Native는 R을 포함한 RHS를 solve하므로 미리 계산한 A와의 곱은 실수 연산에서 동등하고, 실제 FP32 materialization 오차는 별도 확인한다.

\[
W(X)=W_e+XA,\quad Y_c(X)=Y_{c,e}+XAK_c.
\]

여러 write의 합도 W_e+(ΣX_j)A다. **write 횟수가 늘어도 같은 batch의 도달 가능한 선형 공간은 넓어지지 않는다.** 변하는 것은 그 공간에서 선택하는 endpoint다. 단순히 고정 Δ를 여러 번 나누어 총합1로 더하는 Euler는 동일 endpoint인 음성 대조다.

반면 매번 residual을 다시 읽는 반복 fitting은 동일 endpoint가 아니다. R₀=Z₁−Y_e, C=AKc라 하면

\[
\Delta W_{FROZEN2}=[1.75R_0-0.75R_0C]A,
\quad
\Delta W_{FROZEN2}-\Delta W_{N4}=0.75R_0(I-C)A.
\]

이는 75% native write 뒤 현재 residual을 계수1로 fitting한 식이다. 최초 Δ를 총175% 더한 식이 아니다. C=I일 때만 N4로 되돌아가는 특수 경우다. Context-averaged K와 canonical Kc가 다르고 ridge도 있으므로 일반적으로 C≠I다.

스칼라 k=1, ridge=1, residual=1이면 native 최적 Δ=0.5, FROZEN2 최종 Δ=0.6875다. 원래 ridge 목적값은 0.5→0.5703125로 커진다. 따라서 closed form 자체가 부정확하다는 결론보다 **반복 fitting이 기존의 regularization trade-off를 바꾸며, activation residual 감소가 locality 최적화와 같지 않다**는 해석이 맞다.

Frozen-target 연속식 dX/dt=R₀−XC도 만들 수 있지만, C의 성질에 따라 수렴·안정성이 달라진다. 수렴하더라도 z 실현을 강화할 뿐 preservation 개선을 자동 보장하지 않는다. ODE라는 형식만으로 optimal endpoint가 생기지 않는다.

## 5. 권장 flow: native z demand, 실제 출력 feedback, functional barrier

첫 구현의 method identity를 다음처럼 고정한다.

1. L4-only, batch entry W_e/P/M, context, target IDs를 고정한다. 첫 native z₁과 A를 한 번 준비한다.
2. 고정 입력의 **모든 token**에 대해 entry L4 output H_p,e와 down-projection key K_p를 준비한다. 후보 W(X)의 L4 출력은 정확히 H_p,e+XAK_p다.
3. 각 accepted X에서 L5 이후 suffix를 새로 계산하여 실제 edit loss·보존 반응을 읽는다. Prefix는 재사용하지만 suffix는 재사용하지 않는다. Free generation의 token sequence가 바뀌면 다시 prefix를 준비한다.
4. 부족한 요청에는 z demand를 계속 쓰고, 충분한 요청은 추가 confidence 추구를 줄인다. 충분함은 actual-write training contexts의 loss/선호 floor로 정의한다. 공식 평가 P/N은 사용하지 않는다.
5. 그 방향을 functional preservation barrier와 이미 달성한 edit floor에 맞게 수정하여 Euler step을 밟는다.
6. 목표가 오래되었거나 actual edit 개선이 정체된 경우에만 부족한 요청의 native z를 현재 W에서 refresh한다. 이 refresh의 추가 가치는 대조 실험으로 판단한다.
7. Edit 기준을 만족한 이후에도, edit floor를 지키며 locality reference 손실을 줄이는 보정 방향으로 진행할 수 있다. 첫 성공 즉시 종료가 목적은 아니다.

이 구조의 세 refresh는 구별한다.

| 갱신 | 기본 정책 | 이유 |
|---|---|---|
| K/geometry 재계산 | inner loop에서 생략 | single-layer 고정 입력에서는 불변 |
| 실제 후보의 logits/loss/보존 gradient | accepted state마다 갱신 | weight write의 실제 효과를 feedback |
| native z 재최적화 | 선택적·사건 기반 | 필요성이 아직 분리되지 않음 |

Nominal residual velocity는

\[
V_0(X)=\big[(z_i(X)-Y_{c,i}(X))\,g_i(X)\big]_{i=1}^n,
\]

로 둔다. g_i는 actual edit가 부족하면1, 만족하면0인 gate 또는 고정된 좁은 전이구간의 연속 gate다. 만족 판정은 irreversible freeze가 아니다. 다른 요청 write 때문에 품질이 떨어지면 재활성화한다. 별도 성공 floor 제약도 둔다. 요청 i의 velocity column이0이어도 VA의 공동 효과로 i의 activation은 변할 수 있기 때문이다.

**보호 데이터:** W0 기준의 독립 pre-edit reference와 과거 active edit reference를 구분한다. Pre-edit reference에는 현재 edit와 관계가 가깝지만 수정 대상이 아닌 subject/context를 포함한다. 보존할 답은 검증된 기존 label 또는 명시한 W0 teacher다. 경쟁 답은 현재 edit의 target_new 또는 사전에 고정한 대안이다. Official evaluation neighborhood/paraphrase를 controller에 넣지 않는다. Reference 수와 token/F+B 비용은 동일 정보 대조군에도 맞춘다.

Pre-edit reference a에서

\[
m_a(X)=\mathrm{NLL}(y_{comp}\mid x_a,W(X))-
\mathrm{NLL}(y_{keep}\mid x_a,W(X)),
\]

\[
h_a^{margin}(X)=m_a(X)-m_{a,min},\quad
h_a^{answer}(X)=\ell_{a,max}-\mathrm{NLL}(y_{keep}\mid x_a,W(X)).
\]

두 항을 분리하는 이유는 true 답이 약해지는 경우와 새 target이 잘못 침투하는 경우를 모두 보기 위해서다. 기존 fixed10k 감사에서 L4의 W0-success→failure N 중4,242/26,583은 true NLL이 증가하지 않았는데도 실패했다. True NLL 보호만으로 이 실패 유형을 막을 수 없다.

기준은 W0의 margin/NLL과 사전 개발 split에서 정한 허용 변화로 고정한다. 매 Euler step 또는 batch마다 현재의 손상된 모델을 새 기준으로 삼지 않는다. W0에서 이미 틀린 reference는 보존 성공 집합과 분리한다. W0 teacher를 보존하는 실험은 진실 정답 보존과도 구분한다.

과거 edit reference는 최신 active target과 그 target을 수용했을 때의 기준을 사용한다. 정당하게 overwrite된 target은 보존 제약에서 교체한다. 초기 W0나 current entry teacher 하나로 두 종류의 지식을 모두 대표하지 않는다. Historical M도 output reference를 대체하지 않는다.

Writer-space metric을 H=AAᵀ+εI로 두면, 다음 QP가 W-space 방향 변경을 reduced coordinates에서 계산한다.

\[
V^*=\arg\min_V\frac12\operatorname{tr}[(V-V_0)H(V-V_0)^\top]
\]
\[
\text{s.t.}\quad
\langle\nabla_Xh_a,V\rangle_F\ge-\kappa_a h_a,
\qquad a\in\mathcal A_{preserve}\cup\mathcal A_{edit\ floor}.
\]

\[
X_{k+1}=X_k+\eta_kV_k^*,\qquad
W_{k+1}=W_e+X_{k+1}A.
\]

실제 weight 속도는 VA다. H의 AAᵀ 항은 ||(V−V₀)A||²에 해당하며 ε는 좌표 null direction의 수치 regularization이다. 각 barrier gradient는 실제 all-token write와 nonlinear suffix를 통해 얻는다. Residual norm 또는 weight norm만을 locality barrier라 부르지 않는다.

Edit가 충분해진 뒤의 보정 단계에서는 V₀를 −∇_X L_preserve H⁻¹로 바꾸고, current/old edit floor 제약은 유지한다. 이는 더 좋은 preservation 지점까지 진행할 수 있게 하는 구체 규칙이다. 해당 목적의 제약 정상점과 finite resource stop을 구분하며 전역 optimum을 주장하지 않는다.

QP는 0 velocity가 허용된 feasible 상태에서 local direction을 결정한다. 안전 집합이 존재해도 유효한 edit 진행 방향이 존재한다는 보장은 없다. 정체 시에는 edit 진행 slack, reference별 충돌, current writer family의 제약을 기록한다. Barrier에 막혔다고 z refresh가 해결한다고 가정하지 않는다. A의 row span 밖 자유도를 탐색한 것도 아니다.

CBF의 연속시간 조건은 [Ames 등](https://arxiv.org/abs/1609.06408)의 framework를 사용한 설계다. **유한 Euler step에는 별도 actual-forward 검사**가 필요하다. Tangent step은 아무리 작게 해도 곡률 때문에 경계를 벗어날 수 있어 단순 halving의 성공을 보장하지 않는다. 후보가 위반하면 step 축소, interior margin을 넣은 방향 재계산 또는 0-step/실패 판정을 사용한다. Entry가 이미 기준 밖이면 repair 문제로 분류하고 forward-invariance를 주장하지 않는다. Soft slack을 쓰면 실제 위반량을 결과에 남긴다.

관측 reference의 제약 만족과 held-out N 보존은 별개다. Full-gradient per-anchor 비용이 크면 active constraint selection과 전체 reference의 candidate forward screen을 사용할 수 있으나, 그 실행은 모든 reference를 연속시간에서 보호한 것과 같지 않다. Cache 크기·reference F/B·rejected 후보 비용을 모두 기록한다.

## 6. REFIT 기전을 가르는 최소 추가 점검

이미 FROZEN2/I2/FROZEN4/I4 순차 실험은 완료됐다. 같은 실험을 미수행으로 취급해 다시 요구하지 않는다. 다음 질문은 **REFIT의 pass/fail gate와 실패 요청의 fresh z 중 어느 부분이 이득을 남기는가**다.

우선 보존된 공통 W50→B51에서 첫 target과 partial S=W_e+.75Δ₁를 공유한다. 두 번째 entry를 쓰면 보존된 REFIT W55→B56을 선택하되 그 entry의 첫 fitting을 새로 공유한다. N4 chain B56을 대신 사용하지 않는다. 모든 분기에서 P/M/context/RNG를 복원하고 inner history append는0이다.

| 대조 | 두 번째 target | 식별할 것 |
|---|---|---|
| N4 / S75 | 첫 full endpoint / partial endpoint | 이미 존재하는 기준 |
| FROZEN2 | 모든 요청 z₁ | 첫 target 잔차 재피팅 |
| GATE2 | native 첫 평가에서 pass면 현재 a_s, fail이면 z₁ | **추가 Adam 없이** 성공 요청의 demand만 재설정 |
| REFIT4 | 현재 S에서 native fresh target | gate 이후 실패 요청의 새 target 구성의 추가 효과 |
| NM4, 필요 시 | 같은 Δ₁의 norm을 REFIT net norm에 맞춘 진단 endpoint | 단순 update 크기로 설명되는 범위 |

GATE2의 gate는 S의 native initial loss<0.05로 고정한다. REFIT의 초기 forward를 공유하거나 직접 수행하며, 추가 Adam0회여도 forward 비용은0이 아니다. Pass의 a_s−canonical Y도 보존한다. Zero-step 요청을 solve에서 삭제하지 않는다. 그 K가 KKᵀ에 계속 참여하므로 RHS를0으로 두는 것과 요청을 제거하는 것은 다른 개입이다.

동일 partial 상태에서 GATE2와 REFIT의 차이는 fail 요청의 z₂−z₁에 국한되도록 구성한다. 그렇더라도 그 차이는 extra optimization, anchor·teacher·clamp reset을 포함한 target 구성 효과다. 그 안의 각 reset 효과까지 필요하면 **추가1대조만** 둔다: 현재 S에서 같은 initial target/mask와 Adam budget으로 최적화하되 teacher와 regularizer/clamp 기준을 첫 stage에 고정한다. 상대 delta/absolute target 좌표를 명시하고 moment reset은 동일하게 맞춘다.

필수 계측은 다음과 같다.

- K₁/K₂ hash·maxabs, canonical Kc 및 target-init forward의 subject-prefix identity.
- 853/147 그룹의 ||a_s−Y_s||, ||z₁−Y_s||, ||z₂−Y_s|| 분포.
- 동일 A를 유지한 RHS의 pass/fail 분해 D₂=D₂,pass+D₂,fail. 선형 분해의 norm을 metric 인과 기여율로 해석하지 않는다.
- Actual Δ₁, .75Δ₁, Δ₂, net Δ의 norm/cosine 및 net에서 Δ₁에 수직인 성분.
- 동일 문항의 P/N lost/gained, desired/competing NLL, P strict와 tail. Old active와 superseded를 분리한다.

GATE2가 REFIT의 N 이득을 남기면 재최적화보다 **state-dependent demand suppression**이 우선 후보가 된다. REFIT이 GATE2보다 더 나으면 fail 요청의 fresh target 구성에 추가 가치가 있다. NM4가 대부분 설명하면 방향 최적화 claim을 줄인다. 어느 경우도 현재 자료만으로 사전 결론 내리지 않는다.

## 7. ODE·barrier·refresh의 분리 실험과 종료

최소 flow 비교는 fixed first z / selective fresh z와 barrier off / on의2×2다. Gate, actual-write forward, 입력/reference, initial z, A, stop rules를 맞춘다. Barrier off도 같은 reference 값을 관측해 정보·평가 비용 차이를 별도 표기한다. Native N4와 REFIT을 reference로 포함한다.

여기에 **같은 reduced parameterization·semantic objective·reference·허용오차를 쓰는 직접 constrained optimizer**를 대조한다. Native fixed-Δ 단순 분할도 algebraic 음성 대조다. ODE만의 기여는 위 비교 없이 주장하지 않는다. DOW-KE는 이미 실제 배포되는 weight update를 최종 loss로 최적화하므로, actual-weight feedback 자체가 신규성은 아니다. [DOW-KE 원문](https://arxiv.org/abs/2608.16932).

모든 조합의 full10k를 선행조건으로 만들 필요는 없다. 작은 동일-entry 기전 비교로 구현과 질문을 점검하고, 개발용 순차 구간에서 정책을 고정한 뒤 독립 order/구간과 충분한 lifelong 길이에서 평가한다. 단일 batch 성능만으로 장기 우위를 확정하지 않는다. 기존 Middle1,000 suffix와 W0→1,000은 시작 상태가 다르므로 수치를 직접 섞지 않는다.

주 평가를 current N at-write, 동일 과거 N 성공집합의 이후 lost/gained, final N으로 분리한다. RS/PS ranking ceiling만으로 edit 유지 판정을 내리지 않고 strict·desired/competing NLL·old active 손실을 함께 본다. 공식 test P/N은 trajectory·hyperparameter·endpoint 선택에 사용하지 않는다. Request당 P2/N10을 독립10배 표본으로 세지 않는다.

계산량은 요청별 native Adam/F/B, whole-batch suffix F/B, reference F/B, refresh 횟수, accepted/rejected Euler, 실제 wall을 구분한다. 같은 step 수는 같은 비용이 아니다. 최종 accepted X를 실제 dtype W로 적용한 parity와 history 한 번 append를 확인한다. 물리적 copy 횟수는 algorithmic feedback 유무와 별개다.

## 8. 기존 SL-ZFlow 결과를 어떻게 반영할 것인가

저장소의 [기존 SL-ZFlow SEQ1000 리뷰](../../audits/global/2026-09-16-single-layer-zflow-seq1000-review-ko.md)에는 실제 W(X)의 all-token 효과를 통한 fresh suffix gradient가 이미 있다. 그런데 λ_flow=1, geometric cost+entry essence KL, barrier off, max25oracle 설정에서 N4 대비 PS−2.95%p/NS−8.42%p였다. Middle REFIT와 다른 W0 entry 실험이므로 차이의 크기를 직접 비교하지 않는다.

따라서 기존 구현을 “write 후 feedback이 없어서 실패했다”고 진단하면 틀린다. 이번 설계의 변화는 **이미 충분한 edit를 더 밀어붙이지 않는 demand 조정, competitor 침투까지 보는 실제 보존 기준, W0/accepted-history 기준의 누적 보존, 선택적 native target reset**이다. 각각 유용한지는 새 대조가 필요하다.

Single layer의 장점은 고정 K/A, exact all-token affine cache, cumulative X로 표현 가능한 write, 명료한 attribution이다. Single L4 이후 많은 suffix block이 남으므로 계산이 무조건 저렴하다는 뜻은 아니다. ODE의 기여 후보는 그 구조를 이용해 **같은 layer와 같은 writer family 안에서 edit 품질을 유지하며 더 나은 preservation endpoint를 선택하는 정책**에 있다.

검산: `python3 audits/global/2026-09-16-refit-feedback-mechanism/recompute.py`. Stored CSV 산술, 고정 A의 반복-write 식, RHS 분해, Euler 경계 반례를 검산한다. LLM 성능·새 방법의 feasibility 또는 최적성을 검증하는 실험은 아니다.
