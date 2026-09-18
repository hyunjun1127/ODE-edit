# 사전 손상 한도를 두지 않는 target–write editing — 실험 방향 논의안

> 후속 상세화: 이 논의안의 품질 조건 아래 KL 최소화 방향을 [v3 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-edit-quality-preserving-tw-design-v3.md)에 반영했다. 실제 native endpoint를 기준으로 한 correction 좌표, 수치 비교, 기존 admission 교체와 1000요청 실행 범위를 보충했다. 원격 실행 변경은 아직 수행하지 않았다.

작성: 2026-09-15 KST. 상태: 논의·수식 검토. 실행 계약 확정, controller 구현, GPU 실험, 원격 지시 변경을 수행한 문서가 아니다.

최신 요구는 N4 선행 실행뿐 아니라 **사전에 허용 손상량을 정하는 과정 자체를 없애는 것**이다. 따라서 N4 손상의 90%를 쓰는 규칙과, 고정 TV 허용치 ρ를 KL 한도 b로 바꾸는 후속 초안을 모두 권고안에서 제외한다. 후속 고정 ρ/b 초안은 active=false로 표시했고 기존 실행 계약에 통합하지 않았다.

## 1. 권고: 편집 품질을 유지하는 조건에서 original KL을 최소화한다

문제를 다음처럼 바꾼다.

- 기존: 편집 손실을 줄이되 D(W) ≤ b를 만족한다.
- 변경: 선언한 편집 품질을 만족하는 후보 중 D(W)를 가장 작게 만든다.

W0에 대한 고정 C4 출력 분포는 계속 필요하다. 제거하는 것은 reference가 아니라 reference에서 허용할 손상량 b다. W0 teacher는 stream 중 바꾸지 않으며 C4는 고정 prefix·position의 full-vocabulary KL을 제공한다.

이 방향은 [첨부 PDF](/mnt/raid5/janghj/ODE-edit/plans/global/ICLR_2027.pdf) p.5 식 (14)의 endpoint preservation 최소화와 연결된다. 첫 버전은 그 식의 경로 peak 항을 넣지 않고 endpoint만 다룬다. 식 (15)의 dmax log-barrier 및 식 (17)의 고정 dmax 불변성 조건은 채택하지 않는다.

단, 아무 우선순위도 없이 editing과 preservation의 최적 균형이 데이터에서 유일하게 결정되지는 않는다. 서로 충돌하면 어떤 해를 선택할지는 여전히 필요하다. 여기서는 **현재 편집 품질을 우선 유지하고, 그 범위 안에서 보존을 개선한다**는 순서를 선언한다. 손상 예산과 loss 혼합 가중치를 없애는 것이며, 방법의 모든 설계 선택을 없애는 것은 아니다.

## 2. 대안들을 구분한다

| 접근 | 사전 손상 한도 제거 여부 | 이번 판단 |
| --- | --- | --- |
| N4 손상의 일정 비율 | 제거하지 않음 | 외부 baseline 경로와 임의 비율에 의존하므로 제외 |
| TV·KL 허용치를 직접 지정 | 제거하지 않음 | 부등식은 수치의 의미를 설명하지만 적절한 수치를 정해주지 않음 |
| 평균·분산·분위수로 adaptive b 추정 | 일반적으로 제거하지 않음 | 기준 집합·분위수·배수·갱신 규칙에 선택이 남음 |
| E + λD, 또는 b를 둔 dual update | 해결하지 않음 | 고정 λ에 균형 선택을 옮기거나, dual update에도 b가 남음 |
| 편집 성공 조건 아래 D 최소화 | 제거 가능 | 과업 기반 정의이나 경계에서 약한 편집·feasibility 문제가 남음 |
| 현재 native proposal의 품질을 유지하며 D 감소 | 제거 가능 | 첫 구현에 권고. 이미 계산한 자기 proposal을 품질 기준으로 사용 |

W0 self-KL 오차나 표본 오차에서 허용 손상을 자동 산출하는 것도 첫 방법으로 권하지 않는다. 수치적 오차와 실제로 허용할 의미적 변화는 다르다. 유의수준·허용 검정력까지 도입하면 선택이 사라지는 것도 아니다.

## 3. 최소 구현: 자기 native proposal 이후의 품질 유지 보정

각 B100의 자기 branch entry를 Wt라고 한다. Native target과 writer를 한 번 계산해 residual Rp와 실제 preview를 얻는다.

\[
\Phi_t(R)=W_t+RA_t,\qquad V_p=\Phi_t(R_p).
\]

At, 현재 key, history, tokenized contexts, scoring positions는 이 보정 동안 고정한다. 수정 weight는 모든 입력 token에 적용한다. 임의 full-weight gradient나 subject 위치만의 activation hook으로 대신하지 않는다.

**Vp는 별도로 학습한 N4 chain의 checkpoint가 아니다.** 제안 방법이 원래 수행하는 현재 batch의 native target/write에서 바로 얻는다. B2 이후에도 자기 branch에서 계산하며 다른 정책의 미래 target이나 weight를 가져오지 않는다.

Current loss는 기존 설계의 canonical request loss를 유지한다.

\[
E_t(V)=\frac1{100}\sum_{i=1}^{100}\frac1{|y_i|}
\sum_k-\log p_V(y_{i,k}\mid x_i,y_{i,<k}).
\]

각 요청의 desired-token 평균 후 request 평균이다. Native target optimizer의 rewrite contexts와 내부 local KL는 원 설정대로 유지한다. 이 둘을 같은 objective라고 부르지 않는다.

Quality anchor는 실제 preview에서 측정한 Ep = Et(Vp)다. Ap는 Vp에서 canonical desired answer의 모든 teacher-forced token이 top-1인 현재 요청의 ID 집합이다. Tie 처리도 evaluator와 일치시킨다.

첫 후보 선택 문제는 다음과 같다.

\[
\min_{R\in\mathcal C_t}D_{64}(\Phi_t(R))
\quad\mathrm{s.t.}\quad
E_t(\Phi_t(R))\le E_p,\quad
\mathcal A_p\subseteq\mathcal A(\Phi_t(R)).
\]

Ct는 실제로 평가하는 작은 후보 menu이며 Rp를 반드시 포함한다. 성공 수만 같도록 요구하지 않고 **이미 성공한 요청의 ID 집합을 유지**한다. Native proposal에서 실패한 요청도 원 requested denominator에 포함한다. Quality anchor를 맞추기 위해 실패한 요청을 빼지 않는다.

이 규약에는 b, ρ, μbase가 없다. Ep는 사전 조절하는 edit-loss 허용치도 아니며, 같은 batch에서 이미 달성한 품질이다. 다만 native proposal의 품질을 우선한다는 설계 선택은 남는다. Native보다 조금 낮은 edit quality를 허용해 훨씬 낮은 KL을 얻는 영역은 첫 방법이 의도적으로 탐색하지 않는다.

### 보장 범위를 좁혀 읽는다

- 평균 NLL과 이미 성공한 canonical 요청을 보호한다. 모든 요청의 NLL 비악화나 공식 paraphrase 성능을 보장하지 않는다.
- Per-request NLL 악화 분포와 canonical margin을 함께 기록한다. 평균 상쇄가 문제면 후속 버전에서 고정 request groups 또는 개별 NLL 조건을 검토한다.
- 공식 P/N, Historical 정답, Audit/MMLU, FutureN은 gradient·screen·후보 선택에 넣지 않는다.
- Native proposal이 과거 edit를 잊었다면 그 preview를 유지하는 것만으로 old retention이 확보되지 않는다.
- D64 개선은 고정 control64에서의 결과다. Dev128·Report256·NS·일반능력으로 자동 확대하지 않는다.

## 4. 손실 혼합 가중치 없이 한 번의 방향 보정

Rp의 실제 post-write 모델에서 두 gradient를 따로 구한다.

\[
g_E=\nabla_R E_t(\Phi_t(R_p)),\qquad
g_D=\nabla_R D_{64}(\Phi_t(R_p)).
\]

At가 고정이므로 각 gradient는 해당 weight gradient에 At의 transpose를 곱한 값이다. Native solve inverse에 대한 differentiation, HVP, PCG는 필요 없다.

Preservation descent −gD를 current loss의 일차 비증가 반공간에 투영한다.

\[
d^\star=\arg\min_d\frac12\|d+g_D\|_F^2
\quad\mathrm{s.t.}\quad \langle g_E,d\rangle_F\le0.
\]

gE가 0이 아닌 경우:

\[
d^\star=
\begin{cases}
-g_D,&\langle g_E,g_D\rangle_F\ge0,\\
-g_D+
\dfrac{\langle g_E,g_D\rangle_F}{\|g_E\|_F^2}g_E,
&\langle g_E,g_D\rangle_F<0.
\end{cases}
\]

이는 residual 좌표의 Frobenius projection이다. Weight Frobenius metric이나 이전 native-metric projection과 동일하다고 부르지 않는다. At의 null direction 때문에 executable 이동이 0인 경우도 별도로 처리한다.

정확한 일차 모델에서는 ⟨gE,d⟩ ≤ 0이고 ⟨gD,d⟩ ≤ 0이다. 두 gradient의 충돌로 projection이 0이 될 수 있다. gE=0이면 일차 제약은 비어 −gD를 제안할 수 있지만, current loss가 유한 이동에서 유지된다는 결론은 없다.

수식상 두 loss를 각각 양의 상수배해도 gE의 반공간은 같고 d의 방향도 같다. 아래 executable norm으로 step 크기를 정하면 단위 변경을 loss trade-off 가중치로 쓰지 않는다. 수치 임계값, clamp, finite 후보 선택까지 완전 불변이라고 주장하지는 않는다.

### 실제 후보는 전체 write가 아니라 보정량을 줄여 만든다

우선 d를 기존 executable trust 규약으로 정규화한다.

\[
\alpha=\zeta\,\frac{\|R_pA_t\|_F}{\|dA_t\|_F}.
\]

첫 수치 설정 ζ=.25는 기존 보정 크기 제한을 재사용하는 시작점이다. 손상 허용치가 아니지만 solver hyperparameter이며 결과·비용에 영향을 준다. Zero/tiny/nonfinite denominator, alpha cap은 수치 규약으로 분리한다. 모델 성능에 맞춰 numeric epsilon을 키우지 않는다.

Rp+αd를 native target의 per-request ball에 투영하고, executable correction norm이 위 trust bound를 넘으면 correction만 축소한다. Rp가 그 ball 안에 있는지 먼저 확인한다. 얻은 correction을 C라 하면 실제 menu는 다음 네 개다.

\[
R_p,\quad R_p+C,\quad R_p+\tfrac12 C,\quad R_p+\tfrac14 C.
\]

기존의 η(Rp+C)를 사용하는 menu와 다르다. 여기서는 Rp 자체를 줄이지 않고 **추가 correction C를 backtrack**한다. ζ와 candidate 수는 compute/optimizer 설정으로 공개한다. “하이퍼파라미터가 전혀 없다”는 표현은 쓰지 않는다.

모든 후보의 실제 materialized model에서 quality screen과 D64를 측정한다. 통과 후보 중 D64가 가장 작은 것을 선택하고 수치적으로 동등하면 Rp를 우선한다. 원 proposal보다 D64가 높은 보정은 선택하지 않는다. 새로운 gradient를 각 probe에서 다시 구하지 않는다.

유효한 보정이 없으면 **native proposal을 commit**한다. 손상 상한 초과를 이유로 parent를 유지하는 기존 BG rejection 정책과 다른 방법이다. 이를 “preservation 제약을 만족한 성공”으로 표기하지 않고 “보정 없이 native proposal 채택”으로 기록한다. Raw proposal 자체가 nonfinite이면 정상 fallback이 아니라 별도 수치 실패다.

모든 요청의 at-write 및 terminal 평가를 유지한다. 후보 preview에서 history를 append하지 않으며, 최종 선택 모델에서 processed B100당 native finalizer를 한 번 호출한다. Accepted-label ledger와 native key/history 등록은 계속 구분한다.

## 5. 이 방법에서 가장 먼저 확인할 실패 가능성

**일차 투영은 실제 유한 이동의 feasible direction을 보장하지 않는다.** 설명용 smooth loss 예시로 R=(x,y), 원 후보 (0,0), E=x+y², D=((x−1)²+(y+1)²)/2를 두면 gE=(1,0), gD=(−1,1), 투영 방향은 d=(0,−1)이다. 이는 모델 측정값이 아니다.

임의 α>0에 대해 E(αd)=α²>Ep이므로 직선 backtracking 후보는 모두 탈락한다. 그러나 곡선 위의 (−α²,−α)는 E=Ep를 만족하면서 0<α≤1/2에서 D를 줄인다. 따라서 작은 menu의 실패가 더 나은 feasible endpoint의 부재를 증명하지 않는다.

첫 SEQ1000에서 다음을 나눠 기록한다.

1. Gradient conflict가 없고 raw preservation 방향이 가능한 경우.
2. Projection 후 executable 방향이 거의 0인 경우.
3. Current 평균은 유지되지만 개별 strict-success ID가 탈락한 경우.
4. 일차 조건은 만족하지만 curvature 또는 clamp 뒤 actual quality가 악화된 경우.
5. Quality는 유지되지만 actual D64가 줄지 않는 경우.
6. 실제로 native proposal보다 D64가 낮은 보정이 선택된 경우.

4번이 지배적이면 다음 버전에서 **quality restoration 또는 constrained endpoint optimizer**를 추가할 근거가 된다. 이때도 actual Ep를 유지하며, 통과율을 높이려고 임의의 NLL allowance를 더하는 것으로 바꾸지 않는다. 실패 원인을 보기 전에 두 stage, 많은 gradients, 큰 search menu를 동시에 넣지는 않는다.

W0에서 KL gradient가 0인 문제도 구분한다. 여기서는 W0 자체가 아니라 실제 native preview에서 gradient를 계산하므로 처음부터 KL gradient가 반드시 0인 것은 아니다. Native preview에서도 해당 writer family의 KL gradient가 0일 수 있으며 이 경우 gradient-only 보정에는 정보가 부족하다.

## 6. Lifelong 보존과 barrier/ODE 주장

각 batch에서 D64(Vselected) ≤ D64(Vp)가 성립해도 D64(Wt+1) ≤ D64(Wt)는 보장하지 않는다. **D가 원 모델보다 얼마나 커질 수 있는지에 대한 전역 상한은 없다.** Native proposal보다 국소적으로 낮은 손상을 선택하는 규칙이다.

B2 이후의 proposal은 이미 달라진 자기 branch에서 나오므로, 각 batch의 국소 개선을 더해 독립 N4 chain보다 terminal damage가 반드시 낮다고 증명할 수도 없다. 이것이 W0에서 1000개를 순차 편집해야 하는 직접적인 이유다.

W0부터 D가 절대 증가하지 않도록 요구하면, D≥0이므로 모든 선택 모델이 reference에서 D=0을 유지해야 한다. 이는 학습을 항상 막는다는 뜻은 아니지만, reference에 조금이라도 변화를 주는 필요한 편집을 허용하지 않는 매우 강한 다른 문제다.

고정 b를 없앤 뒤에도 h=b−D의 보존 barrier 또는 그 불변성 보증이 남는다고 주장하면 안 된다. Ep−E를 품질 slack으로 삼을 수는 있으나 시작점에서 0이므로 보통의 log barrier는 정의되지 않는다. 임의 positive slack을 더해 문제를 숨기는 대신 첫 구현에서는 projection과 actual screen을 사용한다.

따라서 첫 이름과 주장은 **편집 품질을 유지하는 executable preservation refinement** 정도로 둔다. ODE는 projected flow의 연속시간 표현으로 기술할 수 있지만, 한 번의 correction을 ODE의 필요성·barrier bypass의 증거로 사용하지 않는다. 이후 실제 multistep의 이득은 동일 공간·정보·비용의 one-shot endpoint comparator와 비교해야 한다.

### 선행 연구 및 기존 작업과의 관계

[GEM, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)은 과거 episodic loss의 비증가를 제약으로 두고 gradient를 투영한다. [A-GEM, ICLR 2019, 식 (9)–(11)](https://arxiv.org/pdf/1812.00420)은 평균 memory loss의 단일 제약으로 줄여 내적 기반 projection을 사용한다. 여기의 단일 반공간 수식은 이 계열과 연결되며 projection 자체를 새로운 기여로 주장하지 않는다.

이번 적용은 fixed-W0 full-vocabulary response를 보존 목적에 쓰고, 자기 native preview의 현재 편집 품질을 지키며, executable residual 공간에서 실제 후보를 다시 확인한다. 이는 적용 설계의 차이이지 이 조합의 novelty나 성능을 증명한 것이 아니다.

저장소의 [9월 10일 EP-Free 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-edit-progress-preserving-damage-removal-design.md)에도 편집 진척을 보존하는 공간에서 risk를 제거한다는 선행 구상이 있다. 그 문서는 nominal velocity·group observer·native metric·covariance risk를 사용한다. 이번 논의는 actual W0 response KL, native endpoint 품질 anchor, finite 후보 선택으로 구체화하는 방향이며 기존 아이디어를 새로 발견한 것으로 서술하지 않는다.

## 7. 단계별 실험: 첫 과학적 단위는 W0부터 B100×10

### 1차: 단순한 보정 정책 한 개를 1000개에 적용

- W0·초기 native history에서 시작한다. W50/W90 중간 weight를 사용하지 않는다.
- 공통 고정 요청 1000개, B100×10, 각 batch 자기 branch에서 native proposal 1회·추가 보정 최대 1회.
- C4 S64만 controller에 사용하고 W0 teacher를 고정한다. Dev128은 W5/W10의 개발 확인, Report256은 policy lock 뒤 독립 평가다.
- Old-edit feedback, vulnerability sampling, medoids, 두 번째 fresh target은 첫 버전에 넣지 않는다. Accepted active/superseded ledger와 old 결과는 기록한다.
- 이미 보유한 baseline 결과는 W0·model/config·order·평가 규약이 맞는 범위에서 사용한다. N4가 없어도 ours의 정의와 시작은 성립한다.
- B1은 기술 검증과 continuation 확인용이다. B1의 성능만으로 후속 B2–B10을 실행할지 선택하는 효능 gate로 쓰지 않는다.

이는 **신규 정책 1개, 10 batches**의 첫 실험 제안이다. 기존 실행 지시를 변경하거나 새 job을 제출했다는 뜻은 아니다.

각 B100의 raw preview에서 이미 얻는 E/D/strict 기록과 실제 채택 endpoint를 짝지어 남긴다. 필요한 소수 고정 batch의 raw preview P/N 평가는 evaluation-only 진단으로 할 수 있으나, 이를 매 batch 후보 선택에 사용하지 않는다. 이 자료는 같은 상태의 보정 효과이고 별도 native 1000개 trajectory를 대체하지 않는다.

### 2차: 같은 1000개 sequential 대조로 기전을 분리

| 추가 정책 | 같은 정보·screen 아래 달라지는 점 | 구분할 질문 |
| --- | --- | --- |
| 투영 없는 preservation 보정 | d=−gD, 같은 correction trust·후보 상한·actual quality screen | 방향 projection이 screen만으로는 얻지 못한 이득을 주는가 |
| 품질을 맞춘 scalar 후보 | R=βRp, raw β=1 포함, 사전 고정 scale menu에서 같은 품질을 지킨 후보 중 D 최소 | 방향 변화가 단순한 write 축소보다 필요한가 |

이들도 각각 W0 B100×10이다. 첫 정책을 바꾸지 않았다면 1차 chain을 재사용하며, 변경했다면 변경된 정책의 chain을 새로 구분한다. 대조군을 모두 첫 방법 구현의 선행조건으로 삼지 않는다.

투영 없는 대조의 불필요한 current backward는 비용을 맞추기 위해 강제로 수행하지 않는다. 같은 정보·수치 규약·후보 상한의 비교와 실제 wall/token 차이를 함께 보고한다. Scalar도 gradient가 없어 싸다는 점을 숨기지 않는다. 사전 고정 후보 menu와 실제 비교 비용은 실행 계약 작성 시 잠근다.

단일 B100의 matched-state 방향 진단은 sequential 결과의 해석을 보완하며 1000개 정책 대조를 대신하지 않는다.

### 3차: 필요성이 확인된 요소만 확장

1. **Old retention:** fixed-W0 KL은 새롭게 수락한 정답을 보호하지 못한다. 별도 active accepted-label 자료를 사용한다. Task-defined strict correctness나 acceptance-time loss의 비악화처럼 사전 손상 allowance가 없는 조건을 검토하되, 이미 잊힌 요청·충돌·비feasibility 시 정책을 명시한다. Current-only 버전의 raw fallback을 old까지 보호한 것처럼 재사용하지 않는다.
2. **Quality restoration/group constraints:** curvature 또는 평균 상쇄가 첫 결과의 제한일 때만 추가한다. 공식 P/N를 training observer로 쓰지 않는다.
3. **두 번째 target refresh/write:** REFIT4 경로에 같은 보정을 넣는 경우와, 같은 executable space에서 한 번의 endpoint 최적화에 추가 연산을 쓰는 경우를 비교한다.
4. **Full10k:** method lock 후 W0에서 100 batches를 실제 수행한다. 사용자가 지정한 AlphaEdit, MEMIT, AlphaEdit-BLUE, MEMIT-BLUE, AlphaEdit-L4_only를 최종 비교 목록으로 유지한다. 추가 order와 독립 Audit/MMLU는 최종 주장 검증에 배치한다.

## 8. 첫 결과에서 판단할 것

| 관측 | 허용되는 해석 |
| --- | --- |
| Native preview 대비 E/strict를 유지하며 D64 감소 | 해당 batch·후보 menu의 국소 보정 효과 |
| 1000개 terminal PS/NS·old 유지·독립 KL도 개선 | sequential endpoint 개선을 지지 |
| Control KL만 감소하고 PS 또는 old가 악화 | 선택한 품질 observer 또는 preservation surrogate가 부족 |
| 대부분 raw fallback | 현재 방향·clamp·finite menu에서 보정 실용성이 약함; 일반적인 개선 가능성 부재는 아님 |
| 투영 없는 보정과 동등 | explicit quality screen은 유용할 수 있으나 projection의 추가 필요성 미입증 |
| Scalar와 동등 | 더 싼 정책을 우선할 근거; 방향 제어의 독자 효과 미입증 |
| 두 단계와 one-shot이 동등 | actual-output endpoint 최적화는 남기되 ODE/multistep 필요성 주장을 제외 |

필수 기록은 all-request R/P/N·TF-strict·NLL 및 paired tail, 각 batch at-write→terminal, first500 W5→W10, old active/superseded, canonical 성공 ID의 유지, control/Dev/Report KL 분리다. W0 시작의 첫 실험에 과거5000/전체6000 표기를 재사용하지 않는다.

기전은 gE/gD norm·내적·projection 전후 방향·clamp·실행 가능한 correction norm·후보별 quality와 D·raw fallback 원인으로 기록한다. No-correction 비율과 requested edit의 실제 성공률은 서로 다른 분모다.

비용에는 native target/solve, current와 C4의 분리된 backward, 후보 검사, teacher 준비·읽기, logging/evaluation을 모두 포함한다. Projection에는 두 loss의 gradient가 따로 필요하다. Disjoint current/control microbatch를 각각 계산하면 C4 backward를 두 번 반복할 이유는 없지만, 이를 scalar 하나를 추가한 공짜 연산으로 부르지 않는다.

S64는 8192 scored positions, 16448 input tokens다. 최대 네 endpoint를 모두 별도 검사하면 C4 scored-token 노출은 32768이며 current 평가가 추가된다. Raw preview의 gradient용 forward와 후보 측정을 정확히 재사용한 경우 실제 절약만 ledger에 반영한다. 이 token 산술은 latency 예측이 아니다.

## 9. 계약에 반영할 변경 목록

논의안을 채택하는 다음 계약은 dataset 생성 규약과 method 규약을 분리해야 한다.

- C4 source·문서 분리·tokenization·고정 W0 teacher·cache checksum은 유지한다.
- N4 calibration stage, b_recipe, TV→KL 변환, μbase/τ의 preservation barrier, CALIBRATION_MISSING 실행 차단은 제거한다.
- Ep와 성공 ID는 batch의 자기 raw preview에서 측정한다. 다른 batch로부터 상향 갱신하는 보존 허용치로 쓰지 않는다.
- 전체 update scale backtracking을 correction-only backtracking으로 바꾼다.
- D 상한에 의한 batch rejection을 raw proposal fallback으로 바꾸고, old 제약을 넣는 후속 버전의 비feasibility 정책은 별도로 둔다.
- 기존 BG-1과 목적·선택·fallback이 다르므로 새 method version으로 구분한다. 과거 fixed-b 결과와 같은 정책명 아래 합치지 않는다.
- 변경된 finite screen, zero-gradient 처리, gradient weighting, history finalization을 좁은 기술 검사로 확인한 뒤 1000개 순차 실행으로 넘어간다.

현재 문서는 이 선택을 구체적으로 검토할 수 있게 만든 논의안이다. 기존 method/reference/dispatch 파일과 원격 작업이 이 새 정책으로 이미 전환됐다는 상태 보고가 아니다.
