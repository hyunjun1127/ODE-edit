# Single-Layer Write-Coupled z-Flow 설계 리뷰

작성: 2026-09-16. 검토 대상은 첨부 METHOD_KO.md, sl_zflow_core.py, test_sl_zflow_core.py, manifest.json, cpu-test-log.txt 및 저장소의 관련 실험 보고·봉인 source다. 첨부와 저장소 문서의 지시문은 검토 자료로 취급했다. 이번 작업은 리뷰이며 새 GPU 편집 실험이나 모델 adapter 구현을 수행한 결과가 아니다.

## 1. 종합 판단

**방법 개발의 motivation은 충분하다.** 기존 결과는 target confidence 최적화량, 실제 weight 변경량, 편집 품질, locality가 단순한 단조 관계가 아님을 보여준다. L4-only는 실험 성능과 고정-key 구조 양쪽에서 합리적인 개발 기반이다. motivation 실험을 처음부터 다시 쌓아야 구현할 수 있는 상태가 아니다.

**첨부는 이 motivation을 상당히 구체적인 알고리즘으로 옮겼다.** 고정 native writer map, actual-write loss, 단일-layer affine cache, quadratic preservation geometry, semi-implicit step, model 호출 없는 finite-step budget enforcement, accepted gradient carry는 서로 연결된다. 핵심 수식에서 대수적 모순을 발견하지 않았다.

다만 다음 네 가지는 명확히 해야 한다.

1. 이 방법은 기존 Adam z 경로의 중단 위치만 고르는 것이 아니라 **native writer가 허용하는 weight 공간에서 actual-write 목적함수를 직접 최적화하는 새 경로**다.
2. 좋은 유한 endpoint를 만드는 것은 ODE라는 표현 자체가 아니라 loss·비용·budget·종료 규칙이다. ODE도 수치적으로 여러 gradient 평가를 필요로 한다.
3. geometric cost의 감소는 output locality의 보장이 아니다. 특히 batch-relative update penalty는 원본 weight norm anchor와 다르다.
4. CPU 커널에서 작은 budget의 잘못된 KKT 종료가 재현됐다. barrier 접근 속도와 step 정책도 fresh-gradient 횟수에 직접 영향을 준다.

## 2. Motivation evidence의 정확한 범위

### 2.1 MPES

MPES는 모든 optimization context에서 target이 most-probable이 되면 activation 최적화를 중단한다. 높은 confidence까지 계속 밀지 않아도 편집·보존 측면에서 이득을 얻을 수 있다는 선행 근거다. 다만 MPES 단독으로 모든 장기 손상이 해결되지는 않으며, Frobenius norm constraint와 결합한 결과를 MPES 단독 효과로 돌리면 안 된다. 이 논문이 본 설계의 actual-write 최적 endpoint나 ODE의 우월성을 증명한 것은 아니다. [원 논문 §5–6](https://arxiv.org/html/2502.01636v2)

따라서 motivation의 주장은 “더 높은 target confidence가 실제 편집 효용의 적절한 대리 목적이라고 보장되지 않는다”가 적절하다. “모든 요청에서 native Adam 경로의 내부점이 endpoint보다 좋다”는 더 강한 주장은 현재 필요하지 않다.

### 2.2 L4-only

동일 fixed10k의 최종 실제 W100:

| AlphaEdit 구성 | RS % | PS % | NS % |
|---|---:|---:|---:|
| BLUE L4+L8 | 98.880 | 95.775 | 63.726 |
| BLUE-style L4-only | 99.390 | 95.680 | 65.348 |
| 차이 L4−BLUE | +0.510 | −0.095 | +1.622 |

분모는 각각 10,000/20,000/100,000이다. 편집 성능을 대체로 유지하면서 locality에서 우위라는 해석은 맞다. 모든 지표의 지배 관계는 아니다. L4가 보편적으로 최적 layer라는 뜻도 아니다. [기존 감사 및 수치 출처](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-13-server4-fixed10k-attached-analysis-review-ko.md:9)

Singleton down-projection의 입력은 그 weight의 수정에 영향을 받지 않는다. 이것이 fixed K와 suffix cut을 정당화한다. 다만 이 성질은 다른 singleton에도 성립하므로 L4의 성능 순위 자체를 설명하는 원인과 구분한다. [구조적 범위](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-13-server4-fixed10k-attached-analysis-review-ko.md:64)

BLUE의 동작 설명도 “두 layer의 z를 모두 최적화한 뒤 write”보다 “L4 local z 계산·write 후, 바뀐 모델에서 L8 local z 계산·write”가 정확하다. 두 layer가 요청이나 residual을 절반씩 나누어 맡는 구성은 아니다. [봉인 source에 근거한 설명](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-13-server4-fixed10k-attached-analysis-review-ko.md:52)

### 2.3 REFIT4

공통 W50/M50에서 B51–B60 신규 1,000개를 처리한 결과다. 첫 write .75, 두 번째 same-L4 fit/write를 적용했다.

| 정책 | 신규 RS % | 신규 PS % | 신규 P TF-strict % | 신규 NS % | Online/N4 |
|---|---:|---:|---:|---:|---:|
| N4 | 100.00 | 96.90 | 71.15 | 71.08 | 1.000 |
| REFIT4 | 99.90 | 97.50 | 70.25 | 71.75 | 1.223 |

신규 active R은 양쪽 994/994이며 REFIT4의 R 실패 1개는 superseded다. 과거 active R/P에서는 net −1/−6이 남는다. 따라서 유용한 품질–보존 trade-off가 관측됐다는 것은 맞지만, 모든 편집 품질이 좋아졌다는 해석은 넓다. W50 suffix 결과를 W0부터 적용한 full10k 정책 결과와 섞지 않는다. [검산된 REFIT4 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md:18)

“최적화를 더 했는데도 좋은 결과가 있었다”는 방향은 유지할 수 있다. 그러나 과최적화 자체를 입증한 것으로 쓰지는 않는 편이 정확하다.

- 첫 fit: 1,000개 모두 Adam update 24회.
- 두 번째 fit: update 0회 853개, 21회 1개, 24회 146개.
- 총 update 27,525회, 평균 27.525회/request.
- W50→W60의 net L4 displacement norm은 REFIT4 30.657836, N4 34.335700으로 REFIT4가 작다.

즉 추가 계산, 높은 confidence, 큰 write는 동일한 개념이 아니다. REFIT4는 **부분 write 후 바뀐 상태를 반영하는 정책이 유용한 endpoint를 만들 수 있음**을 지지한다. 기존 Adam trajectory 위의 특정 내부점이 최적임을 직접 식별하지는 않는다. [step와 norm](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md:64)

### 2.4 후속 I2/I4/FROZEN 결과까지 포함한 판단

동일 W50→W60 신규 1,000개:

| 정책 | PS % | P strict % | NS % | Online/N4 |
|---|---:|---:|---:|---:|
| N4 | 96.90 | 71.15 | 71.08 | 1.000 |
| REFIT4 | 97.50 | 70.25 | 71.75 | 1.223 |
| FROZEN2 | 97.90 | 73.35 | 70.70 | 1.067 |
| I2 | 98.05 | 72.80 | 70.86 | 1.142 |
| FROZEN4 | 98.45 | 74.35 | 70.56 | 1.193 |
| I4 | 98.30 | 74.45 | 70.56 | 1.276 |

I2는 12+12, I4는 6+6+6+6의 Adam budget과 carry를 사용한다. FROZEN arm은 최초 target을 유지한다. 이 결과는 단순한 분할·refresh 빈도 증가만으로 모든 지표를 개선할 수 없음을 보여준다. 편집 이득과 보존 비용을 함께 보는 제어 규칙의 필요성을 더 구체화한다. [검산 CSV](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-reviewed-evidence.csv:2)

더 직접적인 보조 근거도 있다. 기존 same-horizon B100에서 target field evaluation을 8→16으로 늘리자 z rephrase NLL은 1.424842→1.384642로 좋아졌지만 실제 W rephrase NLL은 2.138524→2.315501로 나빠지고 Gen은 92.5→89.0%가 됐다. 기존 8-write 방법의 한 B100이라는 범위를 유지해야 하지만 target loss와 actual-write utility의 불일치를 직접 보여준다. [원 보고](/mnt/raid5/janghj/ODE-edit/experiment-reports/servers/server4/p1r52-target-timescale-same-horizon-b100-2026-08-23-native-v3/report-ko.md:13)

## 3. 이 설계가 실제로 최적화하는 대상

수학적 본체는 다음 reduced-space 문제다.

\[
\min_X\; L_{edit}(W_e+XB)+\beta L_{essence}(W_e+XB;W_e)
 +\frac{\lambda}{2}\operatorname{tr}(XSX^\top),
\qquad C(X)\le b\;\text{(선택)}.
\]

여기서 B는 native writer에서 얻은 고정 right factor, X는 학습하는 left factor다. 학습 가능한 weight update는 \(\{XB\}\)로 제한된다. 이는 native preservation geometry가 정한 공간 안에서 actual-write loss를 보는 저랭크 weight 최적화다.

기존 compute_z와 달라지는 요소는 다음과 같다.

- subject 위치의 가상 hidden intervention → 실제 weight 변화에 해당하는 모든 token의 변화.
- 각 request의 개별 z fitting → shared writer를 반영한 joint gradient.
- native delta penalty/clamp → write-space quadratic cost 및 선택적 budget.
- Adam iteration → fixed-metric semi-implicit step.
- 기존 loss/iteration 종료 → stationarity 또는 resource 종료.

따라서 성능이 개선되어도 그 기여가 모두 ODE 때문이라고 해석할 수 없다. actual-write feedback과 목적함수 변경이 가장 직접적인 개선 이유일 수 있다.

### Z 명칭을 더 정확히 해야 한다

첨부의 \(Z=H_{c,e}+X\)는 요구하는 target 좌표로 정의할 수 있다. 그러나 실제 canonical subject hidden은

\[
H_c(W_e+XB)=H_{c,e}+XBK_c
\]

이며 일반적으로 \(BK_c\ne I\)다. X 자체가 실제 hidden displacement인 것처럼 쓰면 native fitting의 실현 오차를 다시 숨기게 된다. X를 writer residual coordinates로 명시하고 desired Z와 realized H를 분리하는 것이 좋다. [첨부 정의](/mnt/raid5/janghj/.codex/attachments/37d8ebce-6326-4b81-b268-520949633099/METHOD_KO.md:15)

## 4. 수학적으로 잘 구체화된 부분

### 4.1 Native map

\[
N=P(KK^\top+M)+\lambda_W I,\quad D=N^{-1}PK,\quad B=ED^\top
\]

이면

\[
XB=[N^{-1}PK(XE)^\top]^\top
\]

가 실수 연산에서 성립한다. 일반적으로 비대칭인 N을 임의로 대칭화하지 않는 점도 맞다. 실제 pinned BLUE compute_ks는 context group 내부 평균 후 group 간 평균을 사용해 request당 한 key를 만든다. 이 실행에서는 q=m, E=I가 맞으며, 모든 context를 별도 key column으로 펼치는 것으로 바꾸면 native writer가 달라진다.

### 4.2 Exact single-layer cut

\[
H_p(X)=H_{p,e}+X(BK_p)=H_{p,e}+XA_p
\]

는 fixed input에서 정확한 affine identity다. \(g_X=\sum_p G_pA_p^\top\)도 맞다. 이 식이 suffix의 비선형성을 근사한 것은 아니다. L5 이후 forward/backward는 매 queried state에서 새로 수행해야 한다. Llama block의 residual/MLP 구조와 일치한다. [공식 v4.44.2 source](https://raw.githubusercontent.com/huggingface/transformers/v4.44.2/src/transformers/models/llama/modeling_llama.py)

고정 prompt의 모든 token에 injection해야 한다. subject-only injection으로 바꾸면 actual-write equivalence가 사라진다. 한 request의 loss가 X의 여러 column에 영향을 줄 수 있으므로 성공한 request column을 임의로 freeze하는 것도 같은 목적함수의 정확한 최적화가 아니다.

### 4.3 Cost와 semi-implicit step

\[
C(X)=\tfrac12\operatorname{tr}(XSX^\top),\quad \nabla C=XS,
\quad H=S+\epsilon I\succ0
\]

에서

\[
\dot X=-(g_X+\lambda XS)H^{-1}
\]

이면 \(dF/ds=-\|\nabla_XF\|_{H^{-1}}^2\le0\)다. 제시한 step

\[
Y=(XH-\eta g)(H+\eta\lambda S)^{-1}
\]

는 해당 proximal surrogate의 유일한 minimizer다. 비싼 nonlinear gradient는 한 번, quadratic term은 analytic하게 처리하는 구성이 병목에 맞는다. 이러한 update의 수치적 해석은 preconditioned proximal gradient/IMEX다. 일반 원리는 [Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)에 해당한다.

### 4.4 Finite-step barrier

\[
C(Y)\le C(X)+(1-e^{-\kappa\eta})(b-C(X))
\]

를 직접 강제하는 방식은 \(X=0\)에서 \(\nabla C=0\)인 문제를 피한다. multiplier를 추가한 scalar search가 constrained quadratic surrogate를 정확히 푸는 것도 타당하다. LLM 호출은 필요하지 않는다. continuous CBF의 일반적인 invariance 관점과 연결되지만, 본 discrete 결합과 knowledge-editing 효능은 별도의 설계·검증 대상이다. [CBF 원 논문](https://arxiv.org/abs/1609.06408)

## 5. Optimal endpoint의 의미를 수정해야 하는 이유

### 5.1 중단 시점 선택과 새 목적함수의 정상점은 다르다

기존 Adam 경로 \(X_A(t)\)에서 optimal stopping을 선택한다면 문제는

\[
t^*=\arg\min_t\{L_{actual}(X_A(t))+\lambda C(X_A(t))\}
\]

다. 경로를 유지하면서 위치를 선택한다.

첨부는 처음부터 \(F=L_{actual}+\lambda C\)의 gradient flow를 만든다. 이상적인 flow에서 F는 계속 감소하므로, 자기 F에 대해 더 이른 중간점이 이후의 정상점보다 더 좋은 구조를 기본적으로 만드는 것은 아니다. native confidence endpoint와 다른 trade-off endpoint를 찾는 것이다. Barrier가 켜지면 최종 budget에 대한 constrained stationarity가 목표다.

이 차이는 motivation을 약화하지 않는다. 원하는 것은 실제 이득과 비용의 균형이므로 actual-write 목적함수로 바꾸는 쪽이 더 직접적일 수 있다. 다만 “Adam 경로의 최적 중단점”과 동일한 수학 문제라고 쓰지 않는다.

### 5.2 ODE가 스스로 유한 endpoint를 만드는 것은 아니다

예를 들어 \(L(x)=\log(1+e^{-x})\), lambda=0, barrier=None이면 finite minimizer가 없다. 첨부 커널에서도 25 oracle에서 target probability .9585577, 128 oracle에서 .9920742이며 모두 RESOURCE_STOP이다.

따라서 main trade-off 구성은 유효한 양의 penalty 또는 유한 budget을 명시해야 한다. lambda=0/budget=None은 confidence-only 비교 설정으로 구분하는 편이 맞다. lambda>0이고 write norm penalty가 유효하면 적어도 write 공간에서 유한 최소점의 존재를 뒷받침하지만, nonconvex optimization이 그 최적점에 도달한다는 보장은 별개다.

또한 first-order stationarity는 local minimum이나 높은 RS/PS의 인증이 아니다. X 공간에서의 stationarity는 전체 W 공간에서의 stationarity와도 다르다. 문서는 이 한계를 이미 상당 부분 적절히 구분했다.

## 6. 우선 수정할 CPU 종료 판정

### 6.1 작은 budget을 경계로 잘못 취급

커널은 boundary 여부를 다음 기준으로 판정한다.

`budget - C <= active_tolerance * max(1.0, budget)`

budget이 작으면 절대 1e-8의 영역이 feasible set 대부분을 덮는다. 여기에 절대 complementarity tolerance 1e-7이 결합한다. [문제 위치](/mnt/raid5/janghj/.codex/attachments/37d8ebce-6326-4b81-b268-520949633099/sl_zflow_core.py:125), [종료 조건](/mnt/raid5/janghj/.codex/attachments/37d8ebce-6326-4b81-b268-520949633099/sl_zflow_core.py:234)

재현: S=[1], x0=0, L=(x−1)^2/2, lambda=.2, eta=1, b=1e-16, 기본 나머지 설정.

| 항목 | 결과 |
|---|---:|
| 반환 상태 | CONVERGED_LOCAL |
| oracle calls | 2 |
| x | 1.12438477e-8 |
| 실제 constrained optimum | sqrt(2b)=1.41421356e-8 |
| C(x)/b | .63212056 |
| 계산한 stationarity residual | 0 |
| multiplier | 8.89375e7 |
| complementarity | 3.27183e-9 |

분명한 내부점에서 경계 multiplier를 허용해 gradient를 지워 버린다. 단위에 민감한 잘못된 수렴 판정이다. b가 매우 작다는 사실만으로 무시하면 안 된다. 본 설계는 native endpoint를 이용한 비용 정규화를 제거했으므로 실제 C/b 단위를 명시해야 한다.

**수정 방향:** 상대 slack으로 active set을 판정하고, primal feasibility·dual residual·complementarity를 선언한 cost/objective scale로 정규화한다. 수치적으로 표현 불가능한 작은 budget은 명시적으로 처리한다. 기존 b=.3 테스트 외에 단위를 바꾸어 같은 문제의 종료가 일관되는 검사를 추가해야 한다.

### 6.2 Barrier가 NFE를 늘리는 구조

discrete condition을 누적하면

\[
b-C_n\ge e^{-\kappa\sum_{j<n}\eta_j}(b-C_0).
\]

따라서 상대 slack delta에 도달하려면 적어도 \(\sum\eta_j\ge\log(1/\delta)/\kappa\)의 pseudo-time이 필요하다. 경계가 최적인 문제에서는 이 제한이 실제 계산 횟수에 작용한다.

동일 1차원 quadratic에서 b=.1, eta=1:

| kappa | oracle | status | 최종 C/b |
|---:|---:|---|---:|
| 1 | 18 | CONVERGED_LOCAL | .999999959 |
| .1 | 128 | RESOURCE_STOP | .999996949 |
| .01 | 128 | RESOURCE_STOP | .719168378 |
| .001 | 128 | RESOURCE_STOP | .119266327 |

모두 rejection 0회다. scalar search가 cheap해도 barrier 때문에 추가되는 suffix F+B가 cheap한 것은 아니다. 현재 step은 rejection 시 감소만 하므로 초기에 작아지면 kappa·eta도 작게 남는다. [step 정책](/mnt/raid5/janghj/.codex/attachments/37d8ebce-6326-4b81-b268-520949633099/sl_zflow_core.py:257)

제한적 step 회복, kappa와 계산 budget의 관계, fixed cap C(Y)<=b와 exponential cap의 차이를 명시하는 것이 좋다. 이는 ODE 전체 가설을 폐기할 이유가 아니라 구현할 controller를 구체화할 문제다.

## 7. 비용이 보호하는 대상과 barrier의 실제 역할

M=K_hist K_hist^T라면

\[
\operatorname{tr}(\Delta W M\Delta W^\top)=\|\Delta W K_{hist}\|_F^2.
\]

Singleton의 고정 key에서는 history key 위치의 **이번 hidden-output 변경량**을 정확히 측정한다. 단순한 추상 norm보다 의미가 있다. 그러나 해당 key는 평균된 subject representation이며 모든 token/context를 대표하지 않는다. 이후 nonlinear suffix에서의 locality, competing-target margin, 일반능력 보존까지 보장하지는 않는다.

또한 다음은 서로 다르다.

- 이번 update norm \(\|\Delta W\|^2\).
- 최종 weight norm \(\|W_e+\Delta W\|^2\).
- 원본으로부터 누적 변화 \(\|W_e+\Delta W-W_{pre}\|^2\).

v1은 첫 번째와 history response를 사용한다. 작은 batch budget을 계속 허용해도 누적 drift가 작다는 보장은 없다. M은 누적 Gram이라 그 크기와 history 방향이 batch에 따라 달라진다. /m 정규화만으로 모든 chronology·batch size에서 lambda와 b의 의미가 자동으로 같아지지는 않는다. 각 batch의 spectrum, C, history 수, lambda/b를 기록하는 편이 좋다. 기존 설계처럼 cumulative anchor를 별도 확장으로 남기는 판단은 타당하다.

추가 수학적 해석:

1. H=S인 극한에서는 \(Y(\nu)=[(1+\eta\lambda)/(1+\eta(\lambda+\nu))]Y(0)\)다. 이 barrier는 raw candidate의 radial shrink가 된다. epsilon>0이면 방향별 축소가 달라질 수 있지만, 실제 locality 손상 방향을 관측해 독립적으로 회전시키는 controller는 아니다.
2. B를 aB로 균일 scaling하면 S와 코드의 relative-damped H가 a^2배가 된다. X를 X/a로 바꾸면 actual write·cost·flow가 동일해진다. 따라서 native attenuation의 계수를 물려받았다는 사실만으로 최종 update 크기도 native처럼 억제된다고 볼 수 없다. 명시적인 cost/budget이 중요한 이유다.

## 8. Fresh compute와 실제 속도

기존 기록상 BLUE-L4의 edit 전체 7.9046h 중 target/z가 7.6480h, **96.753%**다. native solve는 100 batches 전체 약 12초다. 사용자의 병목 진단은 맞다. [시간 계측](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:175)

따라서 fixed writer solve 재사용은 반복 refinement를 싸게 만드는 좋은 구조지만, 원 native 대비 주된 속도 이득을 solve 절감으로 설명하면 안 된다. 핵심은 suffix oracle 횟수와 oracle 한 번의 token work다.

첨부의 정책은 적절하다.

- X=0에서 한 번 L/g 계산.
- candidate 계산은 작은 행렬 연산.
- candidate에서 fresh L/g 한 번.
- accepted L/g를 다음 step에 carry.
- rejected 후보의 비용도 계수.
- \(N_{oracle}=1+N_{accepted}+N_{rejected}\).

다만 1 oracle는 전체 m requests·모든 context의 논리적 sweep이며, 단일 request의 native step과 같은 단위가 아니다. Microbatch 수, padding token 수, target 길이, backward 호출, rejection, wall time을 함께 남겨야 한다.

Pinned L4-only native는 layer4 block output에 delta를 넣고 v_loss_layer=31을 사용한다. 기존에도 frozen L4 prefix backward는 발생하지 않는다. 새 cached suffix의 L5–31, 27 blocks는 native L4 fitting과 같은 backward 깊이다. Prefix cache는 주로 앞의 5개 block 반복 forward를 줄인다. Target prediction/KL 위치만 gather한 full-vocabulary head도 유효한 절감이다. Vocabulary 자체를 줄이지 않는 것은 올바르다.

Native max25는 최대 25번 loss forward/24번 Adam backward이며 early stop도 있다. 새 method의 기본 max_oracle_calls=128이 25보다 적은 비용이라는 뜻은 아니다. Barrier 수렴까지 기다리는 정책이 오히려 더 많은 feedback을 요구할 수 있다.

비용 모델은 다음처럼 기록할 수 있다.

\[
T_{SL}=T_{prefix}+T_{map}+N_{oracle}T_{suffix\ sweep}+T_{commit}.
\]

Request별 조기종료가 있는 native의 실제 token work·wall time과 비교해야 한다. “ODE step 몇 개”만으로 효율을 비교하지 않는다. Reference scalar search의 반복 float 변환은 GPU에서 synchronization을 일으킬 수 있으므로, 작은 spectrum/energy를 한 번 CPU로 옮겨 탐색하거나 device 안에서 수행하는 구현도 검토할 만하다.

## 9. 실제 adapter 구현에서 고정할 계약

1. **좌표/anchor:** W_e, B, S, teacher는 inner loop 동안 고정. 후보는 항상 W_e+XB에서 구성. desired Z와 realized hidden 분리.
2. **Native context semantics:** key의 group 평균, canonical readout, teacher-forced target shift, loss의 context/token weighting을 각각 보존. Key aggregation과 NLL weighting은 같다고 가정하지 않는다.
3. **Cache parity:** full actual-write와 cached suffix의 logits/NLL/X-gradient 비교. 좌·우 padding, position_ids/cache_position/RoPE, attention backend, all-token injection 포함.
4. **Numerical parity:** direct native solve와 factored XB의 FP32 차이, (W_e+XB)K와 W_eK+XBK의 materialization 차이 모두 확인. 실제 파라미터 dtype/backend를 기준으로 허용오차 설정.
5. **종료:** scale-aware KKT; 마지막 accepted state의 L_edit/KL/C/F/residual/relative slack 저장. RESOURCE_STOP과 stationarity 구분. 실제 parity 실패는 별도 상태.
6. **Transaction:** terminal candidate의 parity를 확인한 뒤 weight/history를 일관되게 확정. reject 또는 parity fail에서 W와 M 모두 entry 복원. 중복 commit을 막는 batch token 필요.

기존 NativeSingletonFitter.finalize는 한 호출 안의 중복 physical layer는 막지만 여러 호출의 중복 history append까지 막지 않는다. 기존 helper의 사용 자체가 exactly-once 보장은 아니다. Native history는 CPU FP32 K K^T 누적 순서를 보존해야 한다.

대조한 로컬 source는 첨부에서 지정한 git blob과 일치했다.

| 소스 | 확인한 blob | 핵심 위치 |
|---|---|---|
| NativeSingletonFitter | b1dc90f02aec23ea39b646ec86223dfec17f273a | [finalize와 caller commit token](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-method-review-2026-09-15/source/project/run_scripts/low_cost_write_donor_pilot/fitting.py:167) |
| DirectObjective | 747e4773d6ac8a92dad36fa377eedaf690c583a2 | [teacher·loss·기존 normalization](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/objective.py:39) |
| BLUE writer | 01f910b71655d96119545da2f85a2f4ee40ff38c | [layer-local target](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/AlphaEdit_main.py:58) |

추가 근거: [key group aggregation](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_ks.py:39), [native gradient loop](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_z.py:175), [pinned config](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/config.json:10).

Pinned 기본 context는 clean 한 개와 generated 다섯 개다. Writer key는 clean 1/2와 generated 각각 1/10의 가중치이고 NLL은 여섯 context 각각 1/6이다. 이 차이를 구현에서 없애면 안 된다. Reference runtime은 FP32/eager attention/TF32 off/right padding을 사용하므로 최초 adapter parity의 기준으로 고정할 수 있다. `pretraining_tp>1`처럼 module hook을 우회할 수 있는 구현은 지원 범위를 별도로 명시한다.

메모리 측면에서는 B가 정해진 뒤 prefix를 microbatch별로 처리해 A_p=BK_p를 만든 다음 큰 raw K_p를 해제할 수 있다. 주 cache는 H_entry와 A다. S의 eigendecomposition은 O(m^3), X의 basis 변환은 O(d_out m^2)이므로 “작은 행렬 공간”은 m의 실제 운용 범위와 함께 제시한다. Scalar 이분탐색의 O(m)과 전체 step 비용은 구분한다.

## 10. 구현 후 구분해야 할 기여

광범위한 motivation 재검증을 선행조건으로 둘 필요는 없다. 다음 비교는 새 방법의 기여를 해석하는 데 필요하다.

| 비교 | 분리할 질문 |
|---|---|
| 기존 L4-only / 단순 early-stop L4 | 최대 confidence까지 가지 않는 것만으로 얻는 이득인가? |
| 같은 B·actual-write L/C·oracle를 쓰는 Adam vs IMEX | 새 loss/parameterization의 이득과 integrator의 이득은 무엇인가? |
| IMEX barrier off/on | hard cost cap의 추가 가치와 추가 NFE는 무엇인가? |

Optimizer 비교에서 cache·context·head·평가 budget이 달라지면 integrator 효과로 읽기 어렵다. 고정 oracle budget의 품질과 같은 품질 수준까지의 wall time을 함께 보면 사용자 motivation에 직접 답할 수 있다. 모든 개발 후보마다 전체 baseline 재실행이나 별도 보호 데이터 수집을 강제할 필요는 없다.

개발 단계에서 lambda/b를 선택할 수 있지만 최종 평가 neighborhood를 보고 매 batch 설정을 고르면 locality의 독립 평가가 아니다. 기존 development 실험으로 정책을 정하고 이후 고정한다. 이것은 현재 리뷰의 추가 실행 지시가 아니라 claim과 평가의 구분이다.

## 11. 검증 범위와 최종 권고

- 첨부 manifest의 네 파일 크기와 SHA-256이 모두 일치했다.
- CPU 9개 unittest를 독립 재실행해 모두 통과했다. 명령은 첨부 디렉터리에서 `PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project --with torch python -m unittest -v test_sl_zflow_core`, 실행 기록 3.034초다.
- 작은-budget KKT, kappa에 따른 NFE, lambda=0의 finite endpoint 부재는 별도 CPU 예제로 재현했다.
- 실제 Llama/GPU suffix adapter, checkpoint materialization, knowledge-editing 성능은 이번에 검증하지 않았다. 첨부 역시 이를 구현 완료라고 주장하지 않는다.

**권고는 설계 진행이다.** 가장 먼저 수정할 것은 scale-aware 종료와 실제 suffix/commit parity이고, method 서술의 중심은 “native writer 공간에서 실제 편집 효과와 변경 비용을 함께 반영한 endpoint 최적화”로 두는 것이 맞다. ODE/IMEX는 이를 계산하는 방법으로 제시하고, 그 고유한 효율·성능 기여는 동일 objective의 대조를 통해 확인한다. 기존 실험은 이 작업을 시작할 충분한 이유이며 새 controller의 성공을 선기록하는 근거는 아니다.
