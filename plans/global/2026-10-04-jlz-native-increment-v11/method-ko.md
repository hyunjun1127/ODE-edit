# JLZ v11: 공동 local 증분과 native ridge 비용에 따른 배분

2026-10-04, revision 1. 사용자 실행 지시: `ODEEDIT-USER-GH-SH4-JLZ-V11-INCREMENT-2K-20261004-R1`.

## 1. 목적과 범위

MEMIT-H의 상층 z 목표와 하층 write 사이의 불일치, 그리고 남은 잔차/남은 층 수라는 분배 규칙을 교정한다. 모든 configured write layer에서 사용할 local 증분을 한 subject-injection 경로로 공동 최적화하고, 현재 배경 통계·편집 history·key에 근거한 ridge 비용을 배분 목적에 더한다. Native 문장, subject 위치, NLL/KL, ridge 식, 실제 상층 key 재측정, history 갱신을 유지한다.

모든 층을 처음부터 고려한다. 일부 층 집중은 허용되는 결과다. 특정층을 사전 제외하거나 균등 share, 분산, 비제로층 수를 gate로 강제하지 않는다. Claim은 layer-local 목표 생성과 writer 비용 기반 배분이며 모든 의미적 locality failure를 해결한다는 주장이 아니다. ACC를 선호율로 대체하지 않는다.

이번 버전은 **direct local increment** 계약이다. Absolute virtual target의 `z_l−h_current_l` tracking, 하층 미실현의 상층 자동 전가, T′ 실제 write 주입, inverse-M 증폭은 사용하지 않는다. Native L8 residual feedback의 의도적 대체이며 완전한 native 복제라고 부르지 않는다.

## 2. Entry 상태와 native 입력

순차 batch t의 entry 모델 W_t, 이전 편집 key history H_t를 고정한다. 요청 수 B, 편집층 집합 L, m=|L|, 층 출력 차원 d_l은 runtime 값이다. 마지막 부분 batch도 실제 B로 정규화한다.

요청 r마다 baseline과 동일한 rewrite contexts와 KL 문장을 사용한다. Target token ID, subject-last lookup, target scoring 위치, NLL readout, final-layer KL readout, tokenizer padding/position 의미를 보존한다. Eval paraphrase/neighborhood는 학습·계수 선택에 사용하지 않는다.

Canonical entry subject의 full transformer-block output을 h^0_lr, 그 norm을 a_lr=||h^0_lr||>0이라 한다. Native KL teacher p^0_Kr, a_lr, W_t와 H_t는 fit 동안 고정한다. 주입은 full block 출력의 subject lookup에 적용한다. Writer key는 down_proj 입력이다. Adapter가 이 두 위치 사이의 국소 가법성을 검증해야 한다.

각 층의 entry rewrite key를 native 그룹 평균하여 Khat_l를 얻는다. Canonical1개와 prefix5개의 현재 profile에서 key 가중치는(.5,.1,.1,.1,.1,.1)이다. NLL은 rewrite6개에 각각1/6을 주므로 key 평균과 섞지 않는다. KL rows는 writer key 평균에 포함하지 않는다.

## 3. 공동 local 증분과 task 목적

층 l, 요청 r의 변수 δ_lr를 그 요청의 모든 native rewrite/KL context에 동일하게 주입한다. D_l=[δ_l1,...,δ_lB]. 한 forward에서 해당 요청의 모든 층 δ가 활성화되므로 하층 δ→상층 hidden→최종 task loss의 gradient를 유지한다. 층마다 전체 편집을 따로 성공시키는 loss를 만들지 않는다.

\[
L_N(D)=\frac1B\sum_r\left[\mathrm{NLL}_r^v+\lambda_K\mathrm{KL}(p^v_{K,r}\Vert p^0_{K,r})
+\lambda_n\sum_{l\in L}\frac{\|\delta_{lr}\|_2}{a_{lr}^2}\right].
\]

NLL은 native의 target-token 평균 후 context 평균이다. KL은 current||entry full-vocabulary 방향이다. Norm은 비제곱 L2, canonical anchor 제곱 분모다. 같은 δ가 context 모두에 주입되므로 context별 realized-v norm은 쓰지 않는다. 여러 layer norm의 합은 native 단일 δ norm의 확장이지 동일 유효 강도라는 주장이 아니다.

Model weights는 optimizer 변수가 아니다. Fit 동안 실제 weight update를 적용하지 않는다. Entry 이하 prefix cache는 검증된 동일 token/position/cache state에서만 재사용한다.

## 4. Entry geometry의 full-batch ridge 비용

\[
A_l=\lambda_C C_{0,l}+H_{t,l},\quad
\widehat P_l=(A_l+\widehat K_l\widehat K_l^T)^{-1}\widehat K_l,
\quad \widehat M_l=\widehat P_l^T\widehat K_l.
\]

A/Khat/P_hat/M_hat는 batch-entry에서 한 번 구성하고 모든 fit 후보에 고정한다. 다음 batch에는 그때의 W/H로 다시 만든다. 따라서 history와 순차 모델 상태에 반응하지만, 같은 batch의 하층 candidate 변화에 따른 상층 geometry 변화를 fit 비용이 추적하는 것은 아니다.

A가 SPD인 실수 연산에서:

\[
\widehat C_l(D_l)=\min_U\{\|U\widehat K_l-D_l\|_F^2+\operatorname{tr}(U A_l U^T)\}
=\operatorname{tr}[D_l(I-\widehat M_l)D_l^T].
\]

이는 entry geometry에서의 native ridge 목적 최적값이며 **actual commit 비용의 proxy**다. G=P_hatᵀAP_hat, E=(M_hat−I)(M_hat−I)ᵀ라 하면 G+E=I−M_hat다. G-only로 바꾸지 않는다.

수치 구현은 명시적 inverse나 I−M 뺄셈 대신 SPD solve를 사용한다. S=Khatᵀ A⁻¹Khat, I+S=TTᵀ일 때 C=||T⁻¹Dᵀ||_F²로 계산할 수 있다. Native FP32 C0 normalization/H accumulation의 미세 비대칭을 임의로 symmetrize하거나 jitter로 숨기지 않는다. 실제 asset의 native solve와 해당 factorization parity를 qualification에서 확인하고 허용오차 밖이면 typed technical failure로 보고한다.

\[
\sigma_l^2=\frac1B\sum_r a_{lr}^2,\qquad
c_l=\frac{\sqrt{\widehat C_l(D_l)}}{\sqrt B\sigma_l},\qquad
\mathcal R_A(D)=\sum_l c_l.
\]

이번 본 방법은 **A형 root 합**, MAIN λalloc=.1을 사용한다. NOALLOC은 같은 구조에서 λalloc=0이다. B형 root-of-sum-of-squares arm은 추가하지 않는다. Root의0점은 norm의 최소-norm subgradient0으로 정의한다. 임의 smoothing은 넣지 않는다.

\[
J(D)=L_N(D)+\lambda_{alloc}\mathcal R_A(D).
\]

Full-batch I−M의 off-diagonal을 유지한다. Native loss forward는 요청별로 계산할 수 있지만 allocation cost를 통한 요청 간 coupling은 claim(ii)의 일부다. Microbatch마다 별도 ridge/cost를 계산하거나 diagonal/singleton 비용으로 몰래 바꾸면 다른 방법이다.

## 5. Optimizer·clamp·후보 채택

\[
\delta_{lr}=s_{lr}q_{lr},\quad s_{lr}=\frac{a_{lr}}{\sqrt{d_lm}},\quad q_0=0.
\]

q Adam: lr=.1, betas(.9,.999), eps1e−8, weight_decay0, foreach=false, warm-up0. Model/q/D/moments FP32, geometry와 allocation reduction FP64를 기본으로 한다. Native δ Adam과 물리 step이 같다고 주장하지 않는다.

기록하는 loss는 **request MEAN인 J**다. Optimizer에는 기존 JLZ와 같이 B배 한 request-SUM gradient를 한 번 전달한다: q.grad=B*s*∂J/∂D. B와 s를 두 번 적용하지 않는다. Native row loss는 microbatch별로 누적하고 full-batch allocation은 후보당 한 번만 추가한다.

Native-form 물리 cap은 각(l,r)에 ||δ_lr||≤c_native*a_lr다. 현재 c_native=.75. Adam step 후 D를 projection하고 q=D/s로 반영하며 moment는 유지한다. Model의 hidden을 별도 clip하지 않는다. 합계 budget이나 전체층 cap을 추가하지 않는다. 여러층 cap은 native 한층과 같은 총변화 제한이 아니다.

**종료 계약을 확정한다.** 최대25회 후보평가/24회 optimizer update다. 후보1은 q0=0. 매 후보에 전체 J를 평가한 뒤:

1. Nonfinite/state corruption이면 technical failure.
2. **J_mean<.05**이면 whole logical batch를 종료한다. Allocation까지 포함한 전체 목적이다.
3. 후보25이면 예산 종료한다.
4. 그 외 backward→Adam→물리 δ projection→다음 후보.

이는 native 총loss threshold의 공동 목적 확장이며 native 요청별 종료를 그대로 복제한 것이 아니다. 완료 요청별 freeze, NLL-only 충분성 정지, ACC/PS/NS 기반 stop은 없다. 모든 arm에서 같은 규칙을 쓰되 NOALLOC에는 λalloc=0이므로 그 arm의 목적을 평가한다. Stop 차이는 실측으로 보고한다.

채택은 **마지막으로 평가된 유한 후보**다. 수렴하지 않아도 기술 정합을 통과하면 정해진 예산의 후보를 write하고 다음 batch로 간다. 과거 후보 중 평가 성능이 좋은 것을 선택하거나 추가 step을 허용하지 않는다.

## 6. Write: 실제 현재 key, 계획한 local 증분

Fit이 끝나면 D를 고정한다. W_t/H_t의 transaction 안에서 낮은 층부터 native 순서로 진행한다.

1. 앞층의 실제 weight write가 적용된 현재 모델에서 자기층 rewrite key를 재측정한다.
2. Native nested context mean으로 K_l^a를 만든다.
3. P_l^a=(A_l+K_l^a K_l^{aT})⁻¹K_l^a를 native 방식으로 푼다.
4. U_l=D_l P_l^{aT}, W_l←W_t,l+cast_FP32(U_l)로 적용한다.
5. 다음층의 key를 새 상태에서 측정한다.

**Writer residual은 D_l 자체다.** L8 target residual/remaining-layer divisor, final virtual absolute z−h_actual, extra residual top-up, inverse-M compensation을 넣지 않는다. Final virtual z는 필요하면 관측값으로만 남긴다.

Native weight orientation/transpose/FP64→FP32 cast/add 순서를 유지한다. 실제 effective update와 ideal FP64 update를 구분한다. 실제 key로 다시 푸는 것은 필수이며 entry P를 그대로 commit에 쓰지 않는다.

실현은 Y_l=U_lK_l^a=D_lM_l^a이며 계획과 다를 수 있다. 전체 layer D_l=0이면 U_l=0이지만 개별 요청 δ_lr=0이어도 다른 요청의 열 때문에 그 key에서 변화가 생길 수 있다. Layer별 실현량은 직접 작용 U_l k_l로 기록하고 하층 상속까지 포함한 전체 hidden 차이와 혼용하지 않는다.

Virtual loss는 모든 context의 subject에 동일 D를 더하고 actual weight는 모든 token에 작용하므로 fit/commit 차이가 남는다. Local target은 층 간 목표 위치 불일치를 줄이는 설계이지 완전 실현·최종 task 일치 보장이 아니다.

모든 write 후 최종 모델의 native mean keys를 재추출해 H_l+=K_history,l K_history,lᵀ를 CPU FP32로 딱 한 번 추가한다. 순방향 인과성으로 builder key를 재사용하려면 final-key parity가 먼저 성립해야 한다. Replay 문장/teacher는 없다. Nonselected weights 보존과 rollback을 검증한다.

## 7. Native와 공유하는 부분과 확장한 부분

| 공유 | 명시적 차이 |
|---|---|
| Native 문장·token·subject 위치·NLL/KL 방향·readout | L8 단일 δ 대신 모든 write층 direct δ의 공동 경로 |
| Native 비제곱 norm·canonical anchor | 단일 norm 대신 층별 합, q 재매개화, layer별 cap |
| Native ridge·C0/H·actual key 갱신·history append | L8 residual/divisor 대신 local increment D_l |
| Native 총loss 기반 정지라는 형식 | Batch 전체 J_mean<.05, 요청별 종료가 아님 |
| 순차 모델 상태와 동일 요청 순서 | Entry geometry G+E 배분 비용 추가 및 요청 간 coupling |

같은 λ가 같은 유효 강도라는 주장을 하지 않는다. 본 설정은 기존 native profile과 JLZ의 고정 계수를 유지하는 사전 선택이다. B1 결과로 λ 보정·강도 matching·sweep을 하지 않는다.

## 8. 효율화와 범용성

T′ actual builder/solve/weight materialization을 매 후보 수행하지 않는다. Entry의 geometry 한 번, final sequential writer 한 번으로 분리한다. 이것이 새 절감이며 기존 selected-position head·prefix 재사용을 중복 절감으로 계산하지 않는다.

기존 token IDs를 그대로 사용한 microbatch별 오른쪽 padding trim, 필요한 readout 위치만 full-vocabulary head, key 추출의 마지막 필요지점 조기 종료를 parity 확인 후 허용한다. 논리적 batch는 분할하지 않는다. 원목적/정규화/dtype/후보 규칙을 바꾸는 context·vocabulary·평가 표본 축소는 금지한다. Microbatch 확대는 FLOPs 감소와 GPU 활용 개선을 구분해 기록한다.

B,m,d_l,context 수,target 길이,write modules,주입/readout 위치와 계수는 model/benchmark adapter 및 native profile에서 얻는다. 현 실행은 Llama-3-8B-Instruct/CounterFact, L4–L8/loss layer31, λK=.0625, λn=.5, λC=15000, clamp=.75다. 다른 모델에 숫자를 무조건 전용하지 않는다. 지원되지 않는 architecture는 silent fallback 대신 qualification 미지원으로 보고한다.

## 9. 검증·해석 계약

CPU math 검산, 작은 실제 모델 qualification, BS2×2 pilot, 실제 B100 admission을 거쳐 main으로 진행한다. 상세 허용오차와 분모는 구현·실험 설계에 있다. 성능 gate는 없으며 유한성·native parity·mask/gradient·state/commit 정합 같은 기술 오류만 막는다.

Claim(ii) 대조는 λalloc=0인 **NOALLOC**이며 uniform이 아니다. Main−NOALLOC은 비용항 제거의 효과이고 추가 규제 강도와 배분 선택이 모두 포함된다. NOALLOC−MEMIT-H는 공동 local-target pipeline의 효과이며 locality 하나만 분리한 인과실험이 아니다. 모든 비교는 ACC(strict/token/prompt 정의 명시), RS/PS/NS, NLL, current cohort와 순차 retention을 함께 보고한다.
