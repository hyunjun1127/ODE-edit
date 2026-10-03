# JLZ 실현 subject 주입: T′ 교정 설계

2026-10-03, revision 3. **T′(actual key로 만든 local 변화를 subject fit에 이식)**, native 그룹 가중 norm, active clamp 없음을 유지한다. 사용자 지시로 B1 강도 보정을 철회하고 **A/B의 λ_n=.5를 고정**했다. Method 선택은 정리됐지만 production 구현 및 모델 qualification은 아직 수행하지 않았다.

## 1. 무엇을 정확히 계산하는가

세 가지 정확성을 구별한다.

1. **Ridge 해의 정확성:** 주어진 A, K, 목표 계수 R에 대해 ridge 최적해 U를 구했는가.
2. **상태의 정확성:** K가 실제 하층 가중치 편집 이후 모델에서 읽은 key인가.
3. **학습과 배포의 일치:** 손실을 계산한 forward가 U를 모든 token에 적용한 배포 모델인가.

Subject-only 경로에서 계산한 K로도 ridge 식은 정확히 풀 수 있다. 그 U를 가중치에 더하는 연산도 정확하다. 그러나 실제 all-token 하층 write 이후의 K와는 다를 수 있다. 이때 근사는 **subject-only forward를 배포 모델의 대리 경로로 사용하는 것**이다.

실제 K로 다시 풀면 2번을 충족한다. 다만 commit 직전에만 다시 풀면 학습 때 평가한 U와 다른 U를 배포한다. 최적화 중부터 actual K로 U를 만들고, 최종 평가한 U를 그대로 commit한다. **T′는 실제 context key에서의 local 변화까지 fit에 전달하지만, subject-only 기저와 all-token 기저의 차이는 남긴다.** 실제 모델 성능을 정확히 직접 최적화하는 경로라는 주장은 하지 않는다.

## 2. 유지하는 사용자 요구

- 모든 configured eligible layer를 처음부터 공동 최적화한다. 층 제거, 사전 subset, 강제 균등·강제 분산을 도입하지 않는다.
- Native rewrite/KL 문장, subject lookup, target 위치, NLL readout, KL(current∥batch-entry), 정규화 분모를 유지한다. Paraphrase/neighborhood는 평가에만 사용한다.
- 모델 가중치를 optimizer의 직접 변수로 두지 않고, 층별 local writer 계수를 공동 최적화한다. 편집 손실 경로에서는 subject 위치에만 local 변화를 넣는다.
- MEMIT-H의 ridge 식, C0와 native key history H를 유지한다. History replay 및 추가 reference 문장은 없다.
- 기존 두 arm의 층별 비용 결합 방식은 유지한다. A는 층별 root-cost의 합, B는 제곱합의 root다. 특정 층 집중도 유효한 결과다.
- Adam, warm-up 없음은 우선 유지한다. Adam만 바꿔 배분 모양을 만드는 것을 구조 수정으로 삼지 않는다.
- Batch 크기 B, 층 수 m, 차원, context 수, target 길이는 runtime 입력이다. 마지막 부분 batch도 같은 정의를 따른다.

## 3. 변수와 writer

Batch entry 가중치를 W_t, 고정된 history를 H_t라 둔다. 각 층 l에 대해:

\[
A_l=\lambda_C C_{0,l}+H_{t,l},\quad
K_l=[\bar k_{l1},\ldots,\bar k_{lB}],\quad
P_l=(A_l+K_lK_l^\top)^{-1}K_l,
\]
\[
R_l\in\mathbb R^{d_{out,l}\times B},\qquad U_l=R_lP_l^\top.
\]

기존 D는 **writer에 주는 목표 계수 R**로 표기한다. R 자체를 실제 hidden 변화 δ라고 부르지 않는다. Native nested context mean을 유지하며 KL rows는 K의 rewrite-context 평균에 포함하지 않는다. 실제 구현은 명시적 역행렬 대신 solve를 쓴다.

실제 하층 write 이후의 context별 subject key를 k^a, subject-only fit 경로의 key를 k^s라 한다. **채택한 주입은 k^a 기준**이다.

\[
v^a_{lrc}=U_l k^a_{lrc},\qquad z^s_{lrc}=h^{s,pre}_{lrc}+v^a_{lrc}.
\]

모든 요청·context에 동일한 R_lr를 그대로 더하던 v9와 달리, ridge shrinkage·context 차이·batch cross-talk를 포함한 **actual 경로의 변화**를 주입한다. 각 v는 일반적으로 R의 모든 열과 하층 R에 의존한다. v^s=Uk^s는 차이를 측정하는 관측값이며 T′의 주입값이 아니다. k^a와 v^a를 teacher처럼 detach하지 않는다.

위 식은 실수 연산의 표현이다. FP32 모델에서는 `W_eff = W_entry + cast_fp32(R_fp64 @ P.T)`를 한 번 materialize하고, **actual subject 입력**에 대한 차이 `v_eff = linear(k_actual, W_eff) - linear(k_actual, W_entry)`를 구해 fit의 subject에 더한다. Bias는 동일하게 처리한다. 단순 `cast(U64 k64)`는 같은 FP32 연산이 아니다. T 초안처럼 fit subject 행을 `linear(k_subject_fit, W_eff)`로 교체하면 Uk^s가 들어가므로 **T′ 구현이 아니다.** 차이의 재가산에는 반올림이 생길 수 있어 bitwise 모델 일치 대신 같은 입력에서의 FP32 허용오차를 검증한다. Norm과 주입은 같은 v_eff tensor를 사용하며, ideal FP64 v와 effective FP32 v를 telemetry에서 구분한다. Model adapter는 linear write에서 full-block local 주입으로의 가법성을 검증한다.

## 4. 실행 경로: actual causal builder + 실제 변화의 subject 주입

### S — subject-only에서 key와 U를 함께 구성 (미채택)

각 층에서 subject-only 하층 변화를 반영한 key를 모아 P/U를 구하고, 그 U를 subject에만 적용한 뒤 다음 층으로 간다. Terminal U를 그대로 all-token 가중치로 commit한다.

- 장점: 같은 fit 경로 안에서 주입량과 writer 직접 작용이 일치한다. 별도 actual builder 계산을 줄일 여지가 있다.
- 한계: K^s는 실제 하층 all-token write의 K^a와 다를 수 있다. Native ridge **식**은 같지만 native actual-key **상태**는 다르다.
- 따라서 'actual 상태의 key를 사용한다', '상속 gap이 사라진다', 'fit NLL=commit NLL'을 주장하지 않는다.

### T′ — actual key로 writer 구성 + Uk^a subject 주입 (채택)

사용자의 “상층 key를 재계산 하는 쪽으로 해”를 유지하고, 추가 리뷰의 실제 local 변화 주입까지 반영한다. Terminal에만 갱신하는 방식이 아니라 공동 최적화의 매 후보에 적용한다.

1. 후보 R마다 아래층부터 실제 all-token write를 활성화한다.
2. 그 상태의 자기 층 write 직전 context별 k^a_lrc를 보존하고, rewrite rows만 native 평균하여 K^a_l을 만든다.
3. 동일 ridge 식으로 U_l(R)를 구하고 다음 층으로 간다. 전체 logical batch의 key로 한 번 푼다.
4. 각 rewrite/KL context의 actual subject key로 v^a_lrc를 구한다. 이를 subject-only loss 경로의 해당 위치에만 더한다. KL 문장에 rewrite 평균 key나 canonical v를 대신 넣지 않는다.
5. Native 손실과 write-energy 비용을 합쳐 모든 R을 공동 갱신한다. R_lower→K^a→P→U의 gradient도 유지한다.
6. 마지막 후보의 U로 actual all-token 관측을 수행하고, **관측한 바로 그 materialized weights**를 commit한다. Commit 때 새 writer를 만들어 바꾸지 않는다.

후보별 실제 builder의 K/P는 해당 R와 batch-entry W_t/H_t를 기준으로 새로 구성한다. 후보들을 평가할 때 W_t나 H_t에 누적 write하지 않는다. 최소 write layer의 입력은 하층 edit이 없으므로 그 geometry는 batch 내 재사용할 수 있지만, 상층은 하층 R가 바뀔 때 갱신한다. 전체 후보 종료 후 한 번만 terminal weight/history를 admission한다.

V9의 actual causal builder를 재사용하되 **기존 builder는 rewrite rows만 처리한다는 점을 수정해야 한다.** T′의 native KL에도 v^a가 필요하므로 KL rows를 같은 U의 all-token 하층 경로로 전달하고, 각 층의 actual KL subject key를 저장한다. KL rows는 K의 평균·ridge solve·history Gram에는 포함하지 않는다. Whole-B의 모든 요청과 native context를 포함하며 부분 batch도 같은 정의다.

실수 연산에서 각 층의 직접 변화는 fit과 actual에 동일한 v^a이므로:

\[
h^{a,post}_{lrc}-z^s_{lrc}
=h^{a,pre}_{lrc}-h^{s,pre}_{lrc}.
\]

자기 층 local 변화의 차이는 없어지지만 **하층 all-token write에서 시작된 기저 상태 차이**는 남는다. Attention만이 아니라 residual/MLP/normalization 등 이후 경로가 이를 전달한다. T′가 전체 gap norm을 줄이거나 fit/actual NLL을 일치시킨다는 보장은 없다. 기존 항과 기저 차이의 상쇄가 사라져 전체 gap이 커질 수도 있다.

T 초안은 Uk^s, T′는 Uk^a를 주입한다. 둘의 gradient도 다르다. T′의 주입항에는 masked key로 가는 U·dk^s 경로가 없고 actual builder를 통한 U·dk^a가 있다. 따라서 단순 cache 최적화가 아니라 **공동 목적의 계산 그래프 변경**이다. Subject 위치와 native 손실 구조는 유지하지만 native의 context-불변 단일 δ와 동일한 parameterization은 아니다.

S와 T의 local gap은 각각 U(k^a−k^s) 형태지만, U와 상태가 서로 달라 **크기가 같다는 결론은 성립하지 않는다.** Q가 commit U의 에너지라는 사실은 S에서도 성립한다. T/T′의 고유한 차이는 actual key 상태에서 ridge를 푼다는 것이다. Key 상대 오차가 작아도 ||U||, 방향 및 기저 차이 때문에 task 차이가 작다고 단정할 수 없다.

### 피할 조합 — S로 fit한 뒤 terminal에만 actual K로 재작성

그 마지막 writer는 actual key에 대한 정확한 ridge 해일 수 있지만, 해당 U를 사용한 목적을 최적화한 것은 아니다. 이를 '학습과 commit의 차이를 해결한 정확한 버전'이라 부르지 않는다. 비교 대상으로 쓰려면 별도 endpoint로 식별한다.

## 5. 목적함수

채택한 목적은 다음이다.

\[
\mathcal L(R)=\frac1B\sum_r\left[\mathrm{NLL}^{s}_r+
\lambda_K\mathrm{KL}(p^s_{K,r}\Vert p^0_{K,r})\right]
+\lambda_n\mathcal N(v^a)+\lambda_{alloc}\mathcal R(U).
\]

Teacher p^0와 canonical entry anchor a_lr=||h^0_lr||는 batch 동안 고정한다. Native norm의 **비제곱 형태**와 a² 분모를 유지하고 **rewrite native 그룹 가중 평균**을 채택한다.

\[
\mathcal N(v^a)=\frac1B\sum_r\sum_l\sum_{g=1}^{J}\sum_{c\in g}
\frac{1}{J n_g}\frac{\|v^{a,eff}_{lrc}\|_2}{a_{lr}^2}.
\]

현재 [canonical 1개, prefix 5개]의 두 그룹에서는 w=(.5,.1,.1,.1,.1,.1)이다. **이것은 writer key의 그룹 평균 가중치다. Native NLL은 6개 context에 각각1/6을 준다.** NLL의 target-token 평균과 context 평균은 baseline대로 유지한다. 다른 benchmark/context 수에서는 w=1/(J n_g)를 adapter에서 유도하고 숫자를 hardcode하지 않는다. KL rows의 v도 주입·관측하지만 이 rewrite norm 평균에는 넣지 않는다.

Context 변화가 동일하면 기존 δ norm으로 환원된다. `||Σ_c wv||`는 상쇄로 norm을 숨기므로 위 norm 평균과 교환하지 않는다. 하층 변화까지 포함한 전체 hidden 차이를 local norm으로 쓰면 하층 효과를 반복 계산하므로 구분한다. q/R/v=0에서 비제곱 norm의 subgradient는0으로 정의하고 구현에서 NaN을 만들지 않는다.

보존 비용은 U의 에너지를 기준으로 한다.

\[
Q_l=\operatorname{tr}(U_lA_lU_l^\top),\quad
\sigma_l^2=B^{-1}\sum_r a_{lr}^2,\quad
c_l=\frac{\sqrt{Q_l}}{\sqrt B\sigma_l},
\]
\[
\mathcal R_A=\sum_lc_l,\qquad
\mathcal R_B=\sqrt{\sum_lc_l^2}.
\]

E=||UK−R||²에 해당하는 v9 미실현 비용은 제거한다. 새 편집 경로에서 이미 U의 작용을 평가하므로 R을 숨겨진 정답으로 두고 이를 추종하게 하지 않는다. 이는 root-cost를 raw quadratic Q의 합으로 바꾸는 것과 다르다. A/B의 같은 λ가 같은 규제 강도는 아니며, 이전 v9의 λ를 그대로 옮겨도 동일한 유효 강도가 아니다.

Zero initialization의 Q=0에서도 subgradient0을 일관되게 구현한다. `sqrt(Q)`의0점에서 무한 미분과0이 곱해져 NaN이 생기지 않도록, 가능한 경우 energy factor의 vector norm 또는 검증된0점 처리를 사용한다. 목적을 바꾸는 임의 smoothing을 조용히 추가하지 않는다.

Q는 ideal FP64 U의 write 공간 보존 proxy다. 실제 FP32 effective update의 에너지도 관측하여 rounding 차이를 구분한다. 전체 semantic locality, 층 간 최종 출력 교차항, 새 이웃 문장의 안정성을 보장하지 않는다. H가 증가하면 실제 최적 배분이 반드시 특정 층에서 다른 층으로 옮겨간다는 보장도 없다.

## 6. Pulse, clamp, history

Detached same-candidate teacher를 추종하는 physical pulse는 학습에서 제거한다. Actual 경로는 writer·v^a 구성과 관측에 사용한다. Native 문장으로 actual NLL/KL, subject-only 대 all-token target KL, key/state drift를 관측하되 새 locality 목적이나 replay를 자동 추가하지 않는다.

**Active clamp는 사용하지 않는다.** R cap도 별도로 두지 않는다. ||v^a||/a의 분포와 .75 초과 비율을 사전 정의된 관측값으로 남긴다. Rewrite/KL, layer/context별 분모를 구분하고, 후보별 전부 및 terminal을 기록한다. .75는 v9의 비교용 반경이며 다른 model에 보편적 안전 한계라는 뜻은 아니다. 초과 여부로 clip·후보 거절·층 제외·조기 중단을 하지 않는다. Nonfinite/state corruption 같은 기술 실패 처리는 유지한다.

Commit 후 history는 최종 실제 all-token 모델 경로의 rewrite mean key로 CPU FP32 H+=K_history K_history^T를 한 번 수행한다. T′에서는 동일 후보 actual builder와 observer key의 정합을 검증한다. History 관측을 위해 U를 재작성하지 않는다. W/H transaction과 rollback/hash 검증은 유지한다.

## 7. 정확한 식이 알려주는 것과 알려주지 않는 것

고정 A≻0, K에서 S=K^T A^-1 K, M=S(I+S)^-1이라 두면 ridge의 mean-key 변화는 Y=RM이다. K가 full-column-rank라면:

\[
U_{ridge}=YS^{-1}K^TA^{-1}=U_{exact}(Y),\qquad
Q=\operatorname{tr}(YS^{-1}Y^T).
\]

**같은 K/A와 같은 전체 Y를 맞추면 ridge와 exact는 에너지뿐 아니라 U도 같다.** Cross-talk 차이가 따로 남지 않는다. V9 exact probe는 같은 R을 비교했으므로 같은 Y 비교가 아니었다. Causal 비교에서는 K까지 달라질 수 있다. Probe의 NS 차이를 순수 강도 차이라고 확정하지 않는다. Rank-deficient K는 Y의 실현 가능 부분공간 조건이 필요하며 본 방법에 inverse-M 보상을 추가하라는 뜻이 아니다.

M_ii/(1−M_ii)는 일반적인 batch에서 S_ii가 아니라, 다른 요청 key를 covariance에 넣은 조건부 용량 k_i^T(A+K_-i K_-i^T)^-1 k_i다. 층 평균 대각에 변환을 취한 값으로 full-matrix 배분 비용을 대체하지 않는다.

리뷰의 κ_L4=1.63, κ_L8=.80을 **단일 key·동일 anchor** 예시에 대입하면:

| 비용의 정의 | L8/L4 |
|---|---:|
| v9 G+E root, 같은 요청량 R | 1.209 |
| v9 G+E root, 같은 실현량 Y | 1.686 |
| 새 G-only root, 같은 실현량 Y | 1.427 |
| 새 G-only quadratic energy, 같은 실현량 Y | 2.038 |

따라서 새 비용이 리뷰의 '1.7배' 신호를 그대로 복원한다는 설명은 부정확하다. 실제 B100의 배분에는 행렬의 off-diagonal, 방향, anchor, task Jacobian이 함께 작용한다.

## 8. Native norm 계수 고정과 baseline 비교

Scalar 단일-key, fixed geometry·같은 실현량 y·같은 anchor에서 실현률 γ=κ/(1+κ)라 하면:

\[
\frac{\text{v9 decay}}{\text{새 decay}}=\frac1\gamma,\qquad
\frac{\text{v9 G+E root}}{\text{새 G root}}=\frac1{\sqrt\gamma}.
\]

γ=.60에서는1.667/1.291, γ=.45에서는2.222/1.491이다. 이는 **같은 y에 대한 벌점 비율**이다. 최적 y가 그 비율로 커진다는 식도, '최대1.67배' 상한도 아니다. Task 경로·norm 가중·pulse 제거·finite Adam trajectory도 바뀌므로 실제 크기와 PS/NS의 방향은 측정해야 한다. Batch cross-talk에서는 요청별 단일 γ로 이 식을 일반화할 수 없다.

**현재 profile에서 λ_n=.5를 A/B와 모든 batch에 고정한다.** λ_alloc=.1 및 나머지 확정 profile도 유지한다. B1의 계수 탐색, 추가6 fit, 계수 선택 함수, matching 판정은 모두 철회한다. V9의 실현량은 원칙적인 최적 강도로 정해진 값이 아니므로 새 method의 목표값으로 사용하지 않는다. B1은 일반적인 첫 sequential batch이며 calibration 단계가 아니다.

고정 .5는 native 기본 계수를 유지하면서 실제 주입량에 norm을 부과한다는 설계 선택이다. 다만 **v9의 D도 virtual fit에는 실제 주입된 변화**였으므로, v9 norm 자체가 native 형태를 전혀 따르지 않았다고 설명하지 않는다. 문제는 그 virtual 변화와 actual writer 실현량의 차이였다. V10은 실제 local 변화 v^a를 fit과 norm에 함께 사용해 그 관계를 개선한다.

여러 층의 norm 합, context별 변화·가중치, anchor 및 추가 allocation 비용 때문에 같은 .5라도 native 단일 δ와 **동일한 유효 규제 또는 최적 강도**를 보장하지 않는다. 다른 model/benchmark에서는 해당 native profile의 계수와 reduction을 실행 전에 명시하고 고정한다. .5를 모든 model의 보편적 native 기본값으로 주장하지 않는다. 이번 profile은 .5이며 관측된 실현량·PS/NS에 맞춰 조정하지 않는다.

주 비교는 **BLUE·MEMIT-H·AlphaEdit의 native 기본 설정**이다. V9는 내부 method 변화의 참고 결과이며 v9 대비 순수 구조 효과 분리를 본 실험의 선행 조건으로 두지 않는다. 동일 요청·순서·초기 모델·평가 정의를 확인하고 실행 환경 차이는 표기한다. 같은 norm 계수만으로 완전히 동일 조건이라고 부르지 않는다. 구체 비교 계약은 [baseline 비교 문서](comparison-ko.md)에 둔다.

S_mean=(1/(Bm))Σ_lr ||(U_l K_l)_r||/a_lr, context별 effective v/a, Q와 R/P/N은 계속 기록한다. 이는 설명용 관측이며 λ 선택·후보 채택·실행 gate의 기준이 아니다. 기존 v9 B1 실현량은 [참고 artifact](math/v9-b1-strength-reference.json)로만 보존한다. 다른 profile에서 이를 맞추기 위해 v9를 추가 실행하지 않는다.

## 9. 구현 계약과 적용 순서

1. R/v/U와 K_fit/K_actual을 분리한다. Native prompt/readout/mean과 all-layer 범위를 먼저 고정한다.
2. Actual 입력에서 materialized weight의 local 변화 v_eff를 추출하여 subject-only fit에 전달하는 연산을 만든다. Uk^s를 적용하는 T의 operator와 구별한다.
3. Rewrite/KL actual context를 포함하도록 builder를 확장한다. Whole-B rewrite key만 평균·solve하고, 모든 native row의 v^a graph를 보존한다.
4. R 직접 주입을 v^a 주입으로 교체하고 G-only 비용과 그룹 가중 v^a norm을 연결한다. Detached pulse와 active clamp를 제거한다.
5. Terminal actual 관측과 동일-weight commit을 확인한다. Fit/actual 차이는 연구 관측값이며 이를0으로 강제하는 gate를 만들지 않는다.

Microbatch마다 별도 ridge를 풀면 다른 방법이다. 권고 backward는 **v^a tensor 경계에서 adjoint를 모아 shared builder에 한 번 전달**하는 방식이다.

1. Candidate의 whole-B builder graph에서 U/P/k^a와 모든 v^a를 만든다. 또는 동일 상태의 checkpoint 재계산으로 복구한다.
2. v^a의 임시 leaf를 사용해 masked native loss를 microbatch로 계산하고 bar-v=∂L/∂v^a를 누적한다. Norm의 v gradient도 같은 경계에 합친다. 임시 leaf 분리는 메모리 분할이며 원래 graph로 adjoint를 반드시 전달한다.
3. 누적 bar-v를 원래 v^a graph에 seed하고, allocation의 직접 gradient를 합쳐 builder를 한 번 역전파한다. 이 경로가 ∂v/∂U와 ∂v/∂k^a, R_lower→K^a→P→U를 모두 포함한다.

대안으로 U/k^a 경계에서 각각 bar-U/bar-k를 모아도 되지만 v 경계와 동시에 seed하면 같은 경로를 중복 계산하므로 둘 중 하나만 사용한다. KL row gradient도 빠짐없이 포함한다. 전체 요청 mean과 v9의 request-SUM optimizer bridge 및 q scale을 각각 한 번 적용한다. 임시 detach 후 adjoint를 반환하지 않거나 P를 영구 detach하는 것은 다른 근사다.

작은 CPU nonlinear causal 모델에서 dense autograd, 분할 adjoint, K/P 경로 포함 중앙 유한차분을 비교한다. Production에서는 FP32 materialization/operator VJP, mask, 실제 adapter의 key 인과성, whole-B microbatch parity를 추가 검증한다. [CPU 재현](math/validate_tprime_adjoint.py)은 production 통과 증거가 아니다.

수정 대상은 jlz_realization의 profile.py, subject.py, allocation.py, optimize.py, physical_aux.py, telemetry.py, qualification.py, causal_builder.py 및 writer.py 계약이다. Inputs/prompts의 native NLL reduction과 key-group weights를 재사용하고 변경하지 않는다. Entry/history/readout adapter도 가능한 범위에서 재사용한다.

Telemetry에는 layer/context별 k^a−k^s의 절대/상대 norm, v^a와 v^s, U(k^a−k^s), pre-state gap, 실제 주입/commit parity, mean-key 및 context 실현 share, Q와 ||v||/a 분포를 포함한다. Key 상대 오차만으로 S/T/T′의 동등성을 판정하지 않는다. Share는 효과의 방향·최종 기여도가 아니라 local norm 비중이다. 0분모는 null로 기록한다.

성분별 NLL/KL/norm/allocation의 **R 및 q gradient**를 매 batch 후보2/9/25에서 기록한다. 계수 적용 전·후 norm, 층별 norm, total과의 cosine 및 radial 성분, weighted component 합과 total의 오차를 남긴다. 실제 candidate 수가 다르면 첫 갱신 후/사전 지정 중간/terminal로 profile에 매핑한다. 매 후보에는 total gradient와 모든 실현량 요약을 남긴다. 성분 분해로 늘어난 backward도 비용에 포함한다.

성공 조건은 배분이 불균등해지는 모양이 아니다. B1 균등이나 B5 L8 감소를 gate로 두지 않는다. Actual key/weight/state 정합 같은 기술 검증과 연구 결과 gate를 구분한다. T′는 builder와 masked loss 두 경로를 유지한다. Rewrite builder에 KL rows 추가, v/adjoint 보존, terminal backward 때문에 추가 비용이 생길 수 있어 '추가 계산 거의 없음'을 보장하지 않는다. 호출 수·token 수·최대 메모리·재계산·관측 시간을 따로 측정한다.

## 근거 범위

Frozen v9 source commit: b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66. 경로는 [이전 구조 리뷰](../../../experiment-reports/global/2026-10-03-jlz-v9-deep-review/structural-review-ko.md)의 근거 목록을 따른다. NLL/context 가중치는 frozen subject.py와 jlz_pilot/prompts.py, rewrite-only builder는 causal_builder.py에서 확인했다. [CPU 대수 검산](math/validate_contract_math.py) 및 [T′ adjoint 검산](math/validate_tprime_adjoint.py)을 수행한다. 새로운 실제 모델 성능이나 GPU 처리시간을 측정한 결과는 없다.
