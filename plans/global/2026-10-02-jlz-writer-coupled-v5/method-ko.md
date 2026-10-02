# JLZ v5 실제 writer를 통한 local target 공동 최적화

2026-10-02. 사용자 요청에 따른 재설계다. **모든 편집 가능 층의 local target을 실제 공유 가중치의 효과로 공동 학습하고, 학습에 사용한 가중치를 그대로 commit한다.** 현재 요청과 보존 입력 모두 native 문장으로 한정한다. 과거 native 문장 재사용은 이번에 사용자가 명시적으로 선택했다.

이 문서는 method 정본이다. 구현 완료·GPU pilot·locality 개선을 주장하지 않는다. 기존 v4 source, 설계문, 실행 중인 chain을 변경하지 않는다. [구현 및 검증 명세](implementation-ko.md), [기계 판독 계약](contract.json), [수식 TeX](../../../docs/methods/jlz-writer-coupled-v5.tex)를 함께 제공한다.

## 1 연구 질문과 이번 선택

목표는 순차 편집에서 새 사실을 학습하면서, 실제 편집 효율과 보존 비용에 따라 local target의 크기·방향·층을 공동 결정하는 것이다. 모든 eligible layer를 처음부터 고려하며, 한 층 집중과 여러 층 분산을 모두 허용한다. 강제 균등화, 최소 사용 층 수, 사전 top-k, 층별 NS gate는 없다.

v4의 가상 subject injection은 실제 writer가 만드는 다른 요청·다른 토큰의 변화를 task loss에 넣지 않았다. v5에서는 이 경로를 제거한다. 각 batch entry에서 writer의 기저 P를 한 번 정하고, 그 기저가 만드는 실제 모델로 모든 손실을 계산한다. P를 fit 동안 고정하고 commit에서도 재계산하지 않는 것이 **방법의 정의**다. Current-key triangular writer를 근사 실행한 뒤 마지막에 다른 writer로 바꾸는 방식이 아니다.

이 선택은 exact current-key writer를 매 후보에서 미분하는 방법보다 표현 범위가 제한되지만, 반복적인 geometry solve·층별 전체 batch key 동기화를 피한다. 현재 key가 변하는 효과 자체는 실제 모델의 forward/backward에 포함된다. P는 정의상 상수이므로 dP/dD 항은 없다.

## 2 유지하는 것과 바뀌는 것

| 항목 | v5 정의 |
|---|---|
| 현재 편집 학습 입력 | 해당 model/benchmark baseline의 native rewrite·KL 문장, target IDs, lookup, readout, 가중치 그대로 |
| 편집 후보 층 | model profile의 전체 eligible 집합; 모든 층 D=0에서 시작 |
| 공동 변수 | 각 층 출력 공간의 요청 local target residual D_l |
| Task 경로 | W_t + D_l P_l^T가 모든 토큰에 작용하는 실제 모델 하나 |
| Native 항 | 동일 NLL, current‖own-entry KL, nonsquared payload norm, 개별 native ball |
| Writer 기저 | entry의 문맥별 key second moment를 사용해 batch당 한 번 계산 |
| Commit | 마지막 accepted 후보와 동일한 materialized weight; 재-solve 없음 |
| 보존 입력 | 이미 관측한 native KL·rewrite 문장만; 외부 G/E, 공식 P/N, 미래 edit 없음 |
| 보존 비용 | 과거 native KL anchor + 과거 target NLL 악화 비용; B에만 추가 |
| 배분 규제 | v4의 별도 quadratic V 비용 제거; 강제 분산 항 없음 |
| Solver | 공통 scalar step의 group proximal gradient; 모든 trial을 고정 예산에 포함 |

**Native 목적과 완전히 동일한 알고리즘이라는 주장은 하지 않는다.** 문장·NLL/KL 의미·norm 형태를 유지하지만, 가상 activation intervention에서 실제 weight model로 계산 그래프가 바뀐다. D에 적용하는 norm은 요청 local target에 대한 native-form 규제이며, 실제 hidden 변화량에 대한 원 native 규제와 동일하지 않다.

## 3 변수와 실제 local z

Batch t의 entry 가중치를 W_t, 실제 요청 수를 B_t, 순서 있는 전체 편집층 집합을 L이라 한다. 층별 shape를 허용한다.

\[
D_l=[d_{l1},\ldots,d_{lB_t}]\in\mathbb R^{d_{out,l}\times B_t},\quad
a_{lr}=\|h^0_{lr}\|_2>0,\quad v_{lr}=d_{lr}/a_{lr}.
\]

h^0는 canonical native anchor의 entry block output이다. Nominal local target은 z_nom=h^0+d로 정의할 수 있다. 하지만 실제 모델에서 자기 층 write가 만드는 local action은

\[
u_{lrc}(D)=D_lP_l^\top k_{lrc}(D_{<l}),
\qquad z^{real}_{lrc}=h^{pre-own-write}_{lrc}(D_{<l})+u_{lrc}(D).
\]

실제 z에는 하층 write의 전달과 자기 층의 공유 가중치 작용이 함께 들어간다. u=d 또는 z_real=h^0+d를 가정하지 않는다. Context와 요청 간 간섭도 허용되는 실제 현상으로 계산한다. D는 local target 좌표이면서 저랭크 write 좌표라는 사실을 숨기지 않는다. Claim은 **writer를 통과하는 local target 공동 최적화**이며, 독립적인 local activation을 마음대로 지정할 수 있다는 주장이 아니다.

수학적으로 fit은 고정 오른쪽 기저 P에 대한 저랭크 업데이트 최적화와 동등하다. “joint-z”라는 명칭만으로 그 등가성을 부정하거나 신규성을 주장하지 않는다. Local-target 해석은 P를 현재 요청의 문맥과 순차 history로부터 유도한다는 데 있으며, 검증할 차별점은 이 구조·native 보존 목적·전체 층의 동적 배분을 결합한 효과다.

## 4 문맥 전체를 사용하는 고정 writer

각 native key context의 entry key를 k^0_lrc, native pooling 가중치를 α_rc라 한다. 요청별 Σ_c α_rc=1이다. 현재 Llama profile의 canonical .5, 나머지 각 .1을 보존한다. Task의 rewrite 가중치 w_rc는 별개로 native 값을 유지한다.

\[
\bar k_{lr}=\sum_c\alpha_{rc}k^0_{lrc},\quad
A_{tl}=\lambda_C C_{0l}+H_{tl},
\]
\[
M_{tl}=A_{tl}+\sum_{r,c}\alpha_{rc}k^0_{lrc}(k^0_{lrc})^\top,
\qquad P_{tl}=M_{tl}^{-1}\bar K_l.
\]

명시적 역행렬 대신 FP64 SPD solve를 사용한다. 이는 다음 entry ridge 문제의 해 U_l=D_l P_l^T를 parameterization으로 사용하는 것이다.

\[
\min_U\ \frac12\sum_{r,c}\alpha_{rc}\|Uk^0_{lrc}-d_{lr}\|^2
+\frac12\operatorname{tr}(UA_{tl}U^\top).
\]

평균 key outer product 대신 문맥 second moment를 사용하므로 평균에서 상쇄되는 문맥 차이를 geometry가 본다. 실제 current key는 fit 동안 바뀌지만, P는 갱신하지 않는다. 특정 층이나 요청 column을 미리 제거하거나 Q의 작은 고유값을 역수로 증폭하지 않는다.

Context 열을 C=[sqrt(α_rc) k^0_lrc], 요청 incidence를 R_(rc,r)=sqrt(α_rc)라 두면 CR=Kbar다. A factor를 사용하여 T=solve(A,C), S=I+C^T T, P=T solve(S,R)로 계산한다. 큰 context 수에서는 같은 M의 primal/blocked solve로 전환한다. 어느 경로에서도 실제 logical batch 전체를 사용한다. P의 열 수와 D의 변수 수는 context 수가 아닌 B_t다.

## 5 하나의 실제 모델과 commit 계약

기준 forward는 다음 materialization을 후보마다 한 번 수행한다.

\[
U_l^{64}=D_l^{64}(P_l^{64})^\top,\qquad
W_{eff,l}^{32}=W_{t,l}^{32}+\operatorname{cast}_{32}(U_l^{64}).
\]

수정 linear는 W_eff로 **한 번만** 계산한다. 원 linear를 실행한 뒤 hook에서 다시 F.linear를 실행하지 않는다. Base 모델은 fit 동안 불변이며, 후보 weight는 임시 RAM 상태다. 모든 native·보존 행에는 모든 요청 column의 D_l P_l^T가 작용한다.

마지막 accepted 후보의 W_eff를 RAM에 유지하고 commit에서 동일 tensor를 복사한다. 재-solve, current-key refit, 별도 strength scaling, Q 보정은 없다. 같은 입력·같은 runtime에서 candidate와 committed model의 출력 일치를 검증한다. Low-rank 분리 forward XW_t^T+(XP)D^T는 실수에서 같지만 FP32 반올림이 다르므로 기본 forward가 아니다.

하층 수정→상층 hidden/key→상층 수정→최종 출력, 다른 요청 payload의 간섭, subject 이외 토큰의 효과가 모두 이 경로에 포함된다. A도 요청별로 분리 최적화할 수 없다.

## 6 Native task와 두 arm

L_NLL,r와 L_KL,r는 baseline의 native readout·token/context 평균을 유지한다. KL teacher는 이번 batch의 실제 W_t이며 full vocabulary current‖entry다.

\[
\bar J_A(v)=\frac1{B_t}\sum_r\left[
L_{NLL,r}(W_{eff}(v))+\lambda_{KL}L_{KL,r}(W_{eff}(v)\Vert W_t)
+\sum_l\frac{\lambda_{norm}}{a_{lr}}\|v_{lr}\|_2\right],
\quad\|v_{lr}\|\le c_{native}.
\]

\[
\bar J_B=\bar J_A+R_{past},\qquad
R_{past}=\lambda_K R_K+\lambda_E R_E.
\]

| Arm | 정의 | 해석 |
|---|---|---|
| v5 A, η=0 | J_A | 실제 writer를 통한 공동 local target control |
| v5 B, η=1 | J_A+R_past | 동일 구조에 과거 native 입력의 기능적 보존 추가 |

두 arm의 geometry·norm·solver·초기화·native 문장은 같다. A도 C0/H writer prior를 사용한다. B만 기능적 보존 loss를 사용한다. v4의 η=1과 항의 의미가 달라지므로 결과를 같은 method arm으로 이어 붙이지 않는다. 과거 bank가 빈 B1에서는 두 arm의 목적이 같아야 한다.

V(D)는 필요하면 진단 값으로 계산하지만 목적에 넣지 않는다. 따라서 v4 quadratic가 동등한 경로 사이에서 유도하던 추가 분산 선호를 제거한다. Ridge parameterization과 실제 손실에도 선호가 있으므로 모든 의미에서 배분 중립이라고 주장하지 않는다. 공동 L2/L1 hard budget도 이번 기본형에 추가하지 않는다.

## 7 이미 사용한 native 문장에 대한 순차 보존

이번 기본형은 외부 reference를 만들지 않는다. 유한 RAM의 native memory에는 과거에 실제 처리한 fact만 저장한다. 평가용 paraphrase/neighborhood, 평가 정답, 아직 도착하지 않은 요청을 읽어 teacher·샘플·계수를 정하지 않는다.

### 7.1 Native KL anchor

새 resident KL identity의 teacher는 admission batch의 **편집 전 실제 모델**에서 저장한 분포다. Bank에 동일 KL token/mask/position/readout identity가 이미 있으면 먼저 보관한 teacher를 공유하며 이후 entry로 갱신하지 않는다. Eviction으로 identity가 사라졌다가 다른 새 fact를 통해 재등장하면 새 admission-entry teacher다. 복구할 수 없는 옛 teacher를 가졌다고 표기하지 않는다.

\[
R_K=\frac1{|S|}\sum_{r\in S}\sum_q\beta_{rq}
KL\big(p_{W_{eff}}(\cdot\mid x^{KL}_{rq})\Vert\pi^{anchor}_{rq}\big).
\]

β는 profile의 native KL 분포 가중치다. 각 reference fact가 같은 평균 단위를 갖도록 사용한다. 동일 token identity를 공유해 forward를 재사용해도 원 fact 가중치를 보존한다. Teacher는 W0가 아니며, 처음 관측하기 전의 손상을 되돌리는 기준이 아니다.

### 7.2 과거 target NLL 악화 비용

과거 native rewrite context c의 target 평균 NLL을 ℓ_rc(W), 해당 fact의 최신 실제 commit 직후 값을 b_rc라 한다.

\[
R_E=\frac1{|S|}\sum_{r\in S}\sum_c w_{rc}
\frac12\left[\ell_{rc}(W_{eff})-b_{rc}\right]_+^2.
\]

과거보다 target NLL이 개선되는 것은 벌하지 않고 악화만 비용으로 부과한다. Context별 비교 후 native 가중 평균을 사용해 한 context의 악화를 다른 context의 개선이 가리지 않게 한다. 전체 commit 분포 KL을 고정하지 않으므로 그때의 비목표 logit 변화까지 보호하지 않으며, 저장량도 scalar NLL로 줄인다. NLL 차이의 제곱 항은 새 보존 목적이며 native NLL 자체와 혼동하지 않는다.

고정 참조치 b는 다음 batch entry의 값으로 갱신하지 않는다. 이후 자연스럽게 더 좋아진 최고 수준으로도 자동 갱신하지 않는다. 다만 사용자가 같은 fact를 다시 편집하면 현재 batch에서 해당 fact의 replay를 제외하고, commit 뒤 최신 target/version과 b로 교체한다. 기본형에서는 K도 같은 sampled fact 집합을 사용하므로 현재 재편집 fact는 K/E 모두 이번 replay에서 제외한다. 직접 충돌을 피하는 version 규칙이며 품질 gate가 아니다. 이후 bank의 KL anchor는 기존 것을 유지한다. Soft penalty이므로 b보다 나빠지는 후보를 불가능하게 만드는 hard constraint는 아니다.

약한 편집이 commit되면 b도 약한 상태다. 이는 달성한 target 품질을 보호하는 항이지 과거 성공을 보장하는 항이 아니다. RS 성공 여부로 memory 삽입을 결정하지 않는다.

### 7.3 Bounded memory와 샘플링

초기 실행 profile의 제안값은 capacity M=128 unique fact, 후보당 reference 최대 m=16 fact다. 일반 interface는 임의 M,m를 받는다. Sample 수는 min(m,B_t,현재 유효 memory 크기)로 정하며, empty이면 R_K=R_E=0이다. 각 fact의 **모든 native context와 target 위치**를 재사용한다. 긴 target·다른 모델에서는 profile의 예산을 새로 정한다.

Fact ID는 benchmark adapter가 정의한다. Algorithm R reservoir를 **최초 관측 unique fact** 순서에 적용한다. Unique counter n≤M은 모두 삽입; 이후 j∼Uniform{1,…,n}, j≤M이면 해당 slot을 교체한다. RNG는 arm 공통 seed와 stream metadata만 사용한다. 이미 관측했지만 bank에 없는 fact의 반복은 자동 재입장시키지 않는다. Bank에 있는 fact의 반복은 slot을 유지하고 rewrite 입력·target·b만 최신 version으로 갱신한다. KL 입력의 token/mask/position/readout tuple과 teacher는 함께 고정한다. Subject alias가 바뀌어도 새 KL 입력에 옛 teacher를 붙이지 않는다.

공유 KL identity는 owner fact의 reference count로 관리한다. Commit 시 pending event를 stream 순서로 적용하고, eviction으로 마지막 owner가 없어지면 해당 teacher를 제거한다. 이후 같은 batch의 새 fact가 같은 identity를 다시 추가해도 새 admission-entry teacher다. Fit 중에는 이 pending 상태를 참조하지 않는다. M은 teacher/context payload의 fact 수 상한이며, 중복 식별용 seen_fact_ids metadata는 unique fact 수에 비례해 증가한다. Memory 밖의 반복 fact에 옛 teacher가 복구되는 숨은 refresh는 없다.

이번 batch의 반복 fact를 제외한 bank에서 deterministic seeded uniform subset을 한 번 뽑고, 전체 candidate에서 고정한다. 제외 단위는 fact다. 다른 과거 fact가 같은 KL prefix를 공유하면 그 입력은 여전히 replay될 수 있으며, 정당한 새 편집과의 충돌은 soft loss의 trade-off로 남는다. Arm 간 선택 ID는 같고 teacher/b 값은 각 arm 자신의 실제 과거 상태다. Sample의 품질을 보고 갈아 끼우지 않는다. Current batch의 새 record는 commit 후에만 보인다.

한 batch 안에 상충하는 같은 fact가 있으면 학습 요청을 삭제하지 않는다. Memory는 마지막 stream occurrence를 최신 version으로 삼되, 한 batch의 서로 모순되는 목표를 해결했다고 주장하지 않는다. 해석 시 occurrence와 active latest fact를 분리한다.

## 8 Batch 크기와 계수의 의미

수식은 요청 평균 단위다. 구현이 기존 request SUM을 사용하면 smooth 부분은

\[
f(v)=\sum_r(L_{NLL,r}+\lambda_{KL}L_{KL,r})
+\eta B_t(\lambda_K R_K+\lambda_E R_E),
\quad \Omega(v)=\sum_{l,r}\lambda_{norm}\|v_{lr}\|/a_{lr}
\]

로 계산한다. Reference 평균에 B_t를 곱하지 않은 채 task SUM과 더하지 않는다. Context/token 평균·reference 평균·native norm을 각각 기록한다. v4의 batch 평균 anchor로 V를 나누는 항은 없다.

초기 Llama/CounterFact profile 제안은 λ_K=λ_KL=.0625, λ_E=1이다. Norm .5, clamp .75, C0 coefficient 15000은 기존 profile을 따른다. λ_E는 nat 단위 NLL 악화의 제곱에 대한 새 계수이며 최적값이라는 근거가 없다. η=0/1은 arm switch일 뿐 모든 모델에서 같은 보호 강도를 뜻하지 않는다. 계수 변경은 별도 revision으로 기록하며 공식 PS/NS에 따라 실행 중 조절하지 않는다.

이 정규화는 단위를 보존할 뿐 logical batch 불변성을 보장하지 않는다. Batch가 바뀌면 P, 요청 간 간섭, commit 경계, history가 바뀐다. Microbatch만 바꿀 때 같은 목적·동일한 joint update를 유지해야 한다. 마지막 partial batch를 그대로 처리한다.

## 9 집중을 허용하는 proximal solver

일반 Adam 대신 상대 좌표 v에서 공통 scalar step의 proximal gradient를 사용한다. 모든 층·요청 gradient를 전체 native/reference microbatch에 걸쳐 합산한다. Smooth g=∇f, w_lr=λ_norm/a_lr에 대해

\[
y_{lr}=v_{lr}-\tau g_{lr},\qquad
v'_{lr}=\frac{y_{lr}}{\|y_{lr}\|}
\min\{c_{native},[\|y_{lr}\|-\tau w_{lr}]_+\}.
\]

y=0이면 v'=0이다. 이는 native-form nonsquared norm과 ball의 정확한 Euclidean prox다. 각 group의 gradient를 unit norm으로 만들지 않는다. 영점에 남는 층도 다음 iteration에 모든 gradient를 다시 계산하므로 이후 활성화될 수 있다.

Zero 후보를 최초 accepted state로 둔다. 첫 step은 ν=c_native/4, τ=ν/max_lr||g_lr||로 둔다. g=0이고 v=0이면 그대로 반환 가능하다. 수락 후 다음 시작 step은 min(직전 accepted τ,ν/max||g||)이고, g=0이면 이전 τ를 쓴다. 거절 시 accepted (v,f,g)는 유지하고 **현재 active trial τ를 절반으로 줄여** 다시 시도한다. 거절 때마다 큰 초기 τ로 되돌리지 않는다. ν는 새 solver hyperparameter이며 native Adam .1을 그대로 이식한 값이 아니다. 기존 feasible set은 유지하므로 반복 후 여러 층이 cap에 도달할 가능성은 남는다.

Trial은 smooth majorization

\[
f(v')\le f(v)+\langle g,v'-v\rangle+\|v'-v\|^2/(2\tau)+\epsilon_{num}
\]

으로 수락한다. ε_num은 runtime의 고정 후보 반복 오차를 근거로 qualification에서 결속한다. 성능을 보고 넓히지 않는다. 이는 최적화 정합 판정이며 locality 점수·배분 비율·성공률을 이용한 gate가 아니다. 수치 오차 범위를 제외하면 composite 목적은 감소하지만 개별 RS/PS/NS 단조 개선이나 수렴은 보장하지 않는다.

초기 계산 예산은 25 candidate 평가다. 1번은 zero loss/gradient, 2–24번은 trial loss/gradient, 25번은 trial loss만 계산한다. 거절도 예산에 포함하며 최대24 backward, accepted update≤24다. 마지막 accepted 후보를 반환한다. 모든 trial이 거절되면 zero 후보를 반환하고 no-change를 기록한다. 예산을 몰래 늘려 24번 수락을 보장하지 않는다. Proximal mapping의 영점 또는 v'==v이면 조기 종료하고 실제 횟수를 보고한다. FP32 materialized weight만 같다는 이유로 stationary라고 판정하지 않는다.

## 10 History와 시간 방향의 보존

공통 geometry history는 최종 accepted 실제 모델의 문맥 key로

\[
H_{t+1,l}=H_{t,l}+\sum_{r,c}\alpha_{rc}
k^{commit}_{lrc}(k^{commit}_{lrc})^\top
\]

를 한 번 append한다. 평균 key Gram이 아닌 전체 문맥 second moment다. Entry key를 그대로 append하지 않는다. 마지막 accepted forward에서 모은 subject key가 native key 입력과 동등한 causal adapter이면 재사용하고, 그렇지 않으면 commit 후 정확한 key 입력을 한 번 forward한다.

H는 전 발생 occurrence를 포함하고 재편집된 사실의 과거 항도 자동 삭제하지 않는다. **오래된 H key의 staleness는 여전히 남는다.** 이번 기본형은 별도 대규모 history refresh를 도입하지 않는다. B의 sampled native replay는 현재 실제 모델로 직접 계산하므로 그 표본에서의 key drift·층간 결합·anchor 이후 누적 악화를 본다. 그것이 미표본 history 전체를 현재 좌표로 갱신한다는 뜻은 아니다.

Memory teacher·target anchor는 entry마다 재설정하지 않으므로 매 step 손상을 새 정상 상태로 삼는 문제를 줄인다. 다만 native KL의 관측 위치와 bounded memory 밖의 지식은 직접 보호되지 않는다. W0 대비 전체 모델 보존 또는 neighborhood NS 보장은 없다.

## 11 계산량을 줄이는 구현

1. Entry에서 native 입력을 no-grad로 한 번 처리해 anchor, KL teacher, 문맥 key를 얻고 P를 한 번 구성한다. 그 뒤 zero 후보의 differentiable forward를 수행한다. 두 계산을 근거 없이 한 번으로 세지 않는다.
2. 첫 수정 **선형 연산 직전**까지 cache한다. 수정층 위의 subject-prefix KV는 무효다. Profile이 지원하면 첫 수정층의 앞부분 attention·norm·gate/up도 cache할 수 있다.
3. Candidate별 W_eff를 한 번 materialize하고 모든 microbatch에서 공유한다. Dense weight gradient를 만들지 않고 ∇D=G^T(XP), 입력 gradient=G W_eff를 사용한다. P는 고정이므로 P gradient가 없다.
4. Microbatch별 right-padding crop, native target/ KL 위치만 full-vocabulary head, causal readout 뒤 토큰 prune를 유지한다. Vocabulary 근사나 context 삭제는 하지 않는다.
5. Past native KL teacher는 RAM에 저장하고, rewrite는 commit NLL scalar만 저장한다. 필요한 native reference prefix cache는 매 batch entry에서 새로 만든다.
6. Accepted candidate의 current-request NLL과 context key를 보관하면 memory/history 업데이트를 위한 추가 모델 pass를 피할 수 있다. 마지막 trial이 거절됐으면 rejected tensor를 commit용으로 재사용하지 않는다.

P가 고정돼 있으므로 각 microbatch의 graph를 backward 뒤 해제할 수 있다. 전체 logical batch key를 매 후보·매 층 모으는 barrier가 없다. 모든 D column gradient만 모아 joint step한다.

현재 Llama profile에서 B=100이면 D 변수는 2,048,000개이고 full weight gradient는 293,601,280개다. 비율143.36은 전체 가속 배율이 아니다. Full-context dual 크기는600이며 pooled100보다 geometry 준비는 증가한다. B100·reference16·native6+1이면 후보당 native 행 수는700→최대812로16% 증가하지만, token 길이·cache·head·backward가 달라 시간 증가율로 환산하지 않는다. Materialization 비용, 새로운 cache 경계의 손실, line-search rejection도 별도 집계한다.

M128·KL1·vocab128256·FP32 teacher는 dedup 전 약62.63 MiB다. Rewrite scalar·token metadata·history·P·W_eff·activation 메모리는 별도다. 실측 없이 ETA나 가속 배율을 확정하지 않는다.

## 12 배포 및 이식 계약

Model adapter는 전체 eligible set, 층별 shape, weight orientation/alias, native anchor/readout/head, key 입력과 context 가중치, 첫 write cache 경계, causal 재사용 가능성을 제공한다. Site의 additive local effect와 선형 writer가 연결되지 않는 구조는 별도 adapter 없이는 지원하지 않는다. 상수 L4–L8, hidden4096, key14336, six-context, loss-layer31을 공통 코드에 넣지 않는다.

Benchmark adapter는 ordered fact/version identity, native 입력/target, baseline 가중치, conflict 규칙, 실제 평가 분모를 제공한다. 같은 모델이어도 benchmark에 따라 native input 규약이 달라질 수 있다. Native 규약이 없으면 새 profile을 명시해야 하며 임의로 CounterFact template을 적용하지 않는다.

W/H/memory/RNG/pending admission은 하나의 RAM transaction이다. 기술적 오류는 모두 rollback한다. 유한하고 정합한 후보는 품질이 낮아도 정해진 규칙으로 commit한다. 영속 checkpoint나 대형 재개 payload는 만들지 않고 기존 저장 정책을 유지한다. 설계·source·config·scalar/raw 평가 결과와 해시만 기록한다.

## 13 검증할 claim과 남는 한계

검증할 claim은 다음 세 가지다. 첫째, 실제 writer를 본 공동 local target이 가상 개입의 배포 차이를 없애는가. 둘째, native-only 기능적 보존이 같은 구조 A보다 과거 편집과 locality를 개선하는가. 셋째, 그 결과가 단순 strength 축소만이 아니라 실제 layer/direction 배분의 변화와 함께 나타나는가. RS/PS/NS와 비용을 분리 보고하고, norm share를 인과적 기여율이라고 부르지 않는다.

층별 단독 NS 검사를 selection gate로 넣지 않는다. 실제 local action, payload gap, cap fraction, 각 손실/gradient, accepted/rejected trial, at-write→누적 retention을 보고한다. 우위의 실증 전에는 이러한 구조가 W5 NS 손실을 해결했다고 쓰지 않는다.

기존 [구조 심층 점검](../../../experiment-reports/global/2026-10-02-jlz-v4-method-structure-audit/report-ko.md)의 문제 중 실제 fit/commit 차이·요청 간 task 간섭·문맥 second moment·norm 영점 처리는 설계상 다룬다. History 전체 refresh, 미관측 이웃 지식, model별 계수 보정, 작은 고정 예산의 최적화 품질은 남는 한계다.
