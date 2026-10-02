# JLZ v7 실제 key와 writer를 함께 갱신하는 공동 최적화

2026-10-02. 사용자의 “하층 write로 변한 key에 맞춰 상층 closed-form writer를 다시 풀고, 그 의존성을 z 공동 최적화에도 반영한다”는 요구를 구체화한다. [구현 계약](implementation-ko.md), [기계 판독 계약](contract.json), [TeX](../../../docs/methods/jlz-causal-writer-v7.tex), [500-edit 실행 설계](experiment-500/experiment-ko.md)를 함께 사용한다. 실제 모델 구현·GPU 검증·성능 개선은 아직 확인하지 않았다.

핵심 변경은 매 후보마다 **실제 하층 write → 상층 current key → 상층 writer solve → 배분·보존 loss → 모든 δ의 Adam update**를 하나의 미분 그래프로 연결하는 것이다. Native subject δ 주입을 주 task로 유지하며, v6의 fixed-entry writer와 geometry를 이 그래프로 교체한다. 기존에 전달한 v6 정본은 이 revision과 구분하여 보존한다.

## 1 목표와 변경 범위

모든 eligible layer를 처음부터 고려하고, 현재 batch의 편집 성공과 순차 보존에 유리한 δ의 방향·크기를 공동 학습한다. 한 층 집중은 허용한다. 사전 top-k, 최소 사용 층 수, 균등 분배, 층별 품질 gate는 없다.

v6도 하층 subject δ가 상층 hidden/key를 바꾸는 native 경로와, 하층 실제 weight가 상층 입력을 바꾸는 physical 경로의 gradient는 포함했다. 빠졌던 것은 **그 current key에 맞춰 상층 writer 자체가 적응하는 효과**였다. P/G/E를 batch entry에서 고정하면 그 적응 경로와 배분 비용의 상태 변화가 빠진다. V7은 이를 포함한다. 과거 PS/NS 차이가 이 누락 때문에 발생했다고 단정하지 않는다.

Native 학습 문장·lookup·NLL readout·current‖entry KL·nonsquared norm, absolute δ, Adam, 사후 clamp, 네 auxiliary pulse, native-only memory, 두 arm의 aggregation 차이는 유지한다. Final-layer z를 나눠 쓰는 방식이나 층별 독립 z fitting으로 돌아가지 않는다. 실제 weight 모델의 target NLL로 primary task를 교체하지 않는다.

## 2 고정 상태와 공동 변수

한 batch의 pretrained 또는 직전 commit 모델을 W_t, history를 H_t라 한다. 실제 요청 수는 B, ordered eligible write sites는 L, 모든 current native rewrite context 수는 C다. 다음을 batch 동안 고정한다.

- 원본 W_t, A_l=λ_C C0_l+H_t,l 및 그 SPD factor.
- Canonical clean full-block anchor a_lr=‖h0_lr‖>0와 σ_l²=B⁻¹Σ_r a_lr².
- Native 입력·target·lookup·readout·context/token reduction, own-entry KL teacher.
- Context 소유 행렬 Z∈R^(B×C), writer 가중치 Ω=diag(α_c), native task 가중치 w_rc.
- Native 과거 memory 기준·이번 batch의 replay 표본·네 current/past partition.

D_l=[δ_l1,…,δ_lB]∈R^(d_out,l×B)는 모든 층·요청에 존재하며 0에서 시작한다. 모델 weight를 Adam parameter로 등록하지 않는다. Layer 수·차원·B·context/target 길이를 profile 상수로 가정하지 않는다.

α_c≥0이고 각 요청의 α합은1이다. Writer α와 native task w를 혼동하지 않는다. 현재 profile의 α는 canonical .5, 나머지5개 각각 .1이며, task w는6개 각각1/6이다. 실제 저장된 부동소수 가중치를 보존한다. 미세한 합 오차를 없애려고 몰래 재정규화하지 않는다. Ω의 역행렬을 쓰지 않으므로 α=0도 처리할 수 있다.

## 3 Native subject 주경로

Frozen W_t에 모든 eligible layer의 δ를 각 native 문장의 subject lookup 위치에만 더한다.

\[
h^v_{lrc}[s_{rc}]=h^{v,pre}_{lrc}[s_{rc}]+\delta_{lr},
\qquad z^v_{lrc}=h^v_{lrc}[s_{rc}].
\]

h_v,pre는 이미 하층 δ의 전달을 포함한다. 상층의 virtual key도 매 forward 바뀐다. 이것을 actual weight 경로의 key와 동일하다고 가정하지 않는다.

\[
L_{native}(D)=\frac1B\sum_r\left[
L^v_{NLL,r}+\lambda_{KL}KL(p_r^v\Vert p_r^{entry})
+\lambda_{norm}\sum_l\frac{\|\delta_{lr}\|_2}{a_{lr}^2}\right].
\]

학습 문장은 baseline의 native rewrite/KL 그대로다. Official paraphrase/neighborhood와 미래 edit는 사용하지 않는다. Norm anchor·own-entry KL teacher를 매 후보의 변경된 상태로 갱신하면 기준이 움직이므로 고정한다. Current key 갱신과 보존 기준 갱신을 혼동하지 않는다.

## 4 후보마다 실제 writer를 인과 순서로 구성한다

한 후보 D를 고정하고 낮은 write site부터 높은 site로 진행한다. K_l^a는 **이 후보의 하층 실제 weight write가 적용되고 자기 층 write는 아직 적용되지 않은 상태**에서 추출한, 모든 current native rewrite 문맥의 subject pre-down-projection key다.

\[
K_l^a(D_{<l})=\operatorname{Keys}_l
\bigl(W_t+\{U_j(D):j<l\};\text{all current native RW}\bigr).
\]

Physical builder에는 subject δ hook을 설치하지 않는다. 하층 D는 이미 U로 작용하므로 δ까지 더하면 같은 편집을 이중 적용한다. 다른 요청의 D 열도 공유 U를 통해 해당 문장의 key에 영향을 준다.

각 층의 현재 K=K_l^a로

\[
\bar K=K\Omega Z^\top,\quad M=A_l+K\Omega K^\top,\quad
P_l(D_{<l})=M^{-1}\bar K,
\]

\[
U_l(D)=D_lP_l(D_{<l})^\top,\qquad
\widetilde W_l=W_{t,l}^{32}+\operatorname{cast}_{32}(D_l^{64}P_l^{64\top})
\]

를 구성한다. 역행렬을 직접 만들지 않고 solve한다. 이 실제 가중치를 해당 층의 모든 token 위치에 적용하고 다음 층으로 진행한다. Layer별 forward 순서는 인과적 구성 순서이며 optimizer는 층별로 따로 step하지 않는다.

Adapter는 자기 write weight가 자기 input key에 영향을 주지 않고, 이후 write가 앞선 key에 영향을 주지 않는 feed-forward dependency와 write parameter의 독립성을 보장해야 한다. 현재 Llama down_proj 경로는 이 구조다. Weight alias, recurrent/shared invocation, 다른 architecture는 유효한 의존 그래프와 write map을 검증하기 전 지원 완료로 표기하지 않는다.

이 구조는 causal DAG다. 같은 후보 안에서 fixed-point 반복이나 하층 재수정은 필요 없다. 최저 write site는 앞선 current write가 없어 K/P를 batch 내 재사용할 수 있다. 그보다 높은 site의 K/P는 모든 후보에서 새로 구성한다. Batch commit 뒤에는 최저층을 포함해 entry-dependent 상태를 갱신한다.

## 5 실제로 fitting하는 목표

현재 층의 writer는 다음 ridge 문제의 해다.

\[
U_l=\arg\min_U\;\frac12\|(UK_l^a-D_lZ)\Omega^{1/2}\|_F^2
+\frac12\operatorname{tr}(UA_lU^\top).
\]

목표는 기존과 같은 **local incremental δ**다. 새 actual key에서 그 증분을 실현한다. Absolute virtual z에 맞추기 위해 R_lc=z_lc^v−h_lc^{a,pre}로 목표를 교체하지 않는다. 그 경우 context별 R, writer 표현, G/E 목적이 달라져 이번 변경과 섞이기 때문이다.

Same-layer block-output 좌표가 유효한 adapter에서

\[
h_l^a-h_l^v=(h_l^{a,pre}-h_l^{v,pre})+(U_lK_l^a-D_lZ).
\]

Current-key writer는 두 번째 항의 fitting 기준을 맞춘다. 첫 번째 inherited state 차이, ridge shrinkage, 여러 context를 한 δ로 표현하는 제약, 모든 token에 작용하는 부수 효과는 남는다. 이를 다음 보조 loss가 다룬다. Key refresh만으로 virtual z와 actual model의 완전 일치 또는 locality 보존을 주장하지 않는다.

## 6 현재 상태의 배분 비용

매 후보의 K/P로

\[
G_l(D)=P_l^\top A_lP_l,\qquad
E_l(D)=(P_l^\top K_l^a-Z)\Omega(P_l^\top K_l^a-Z)^\top
\]

를 계산한다. 그러면

\[
\operatorname{tr}(D_lG_lD_l^\top)=\operatorname{tr}(U_lA_lU_l^\top),\quad
\operatorname{tr}(D_lE_lD_l^\top)=\|(U_lK_l^a-D_lZ)\Omega^{1/2}\|_F^2.
\]

\[
G_l+E_l=Z\Omega Z^\top-\bar K_l^\top P_l.
\]

이상적으로 정규화된 가중치면 ZΩZᵀ=I다. 수치 검증은 실제 저장된 Ω로 왼쪽과 오른쪽을 비교한다. Off-diagonal을 유지한다. FP64 이론적 U에 대한 비용과 FP32로 materialize된 실제 효과는 구분하고 그 차이를 telemetry에 남긴다.

\[
g_l=\frac{\sqrt{\operatorname{tr}(D_lG_l(D)D_l^\top)}}{\sqrt B\,\sigma_l},\quad
e_l=\frac{\sqrt{\operatorname{tr}(D_lE_l(D)D_l^\top)}}{\sqrt B\,\sigma_l}.
\]

| 항목 | Arm A | Arm B |
|---|---|---|
| R_W | Σ_l g_l | sqrt(Σ_l g_l²) |
| R_E | Σ_l e_l | sqrt(Σ_l e_l²) |
| 나머지 | 동일 | 동일 |

\[
L_{base}=L_{native}+\lambda_W R_W+\lambda_E R_E.
\]

G/E가 D_<l에 의존하므로 이를 D 전체의 고정 norm 또는 convex penalty라고 부르지 않는다. 동일하고 고정된 geometry·동일 기능 경로를 가정한 분할 예시에서만 A는 같은 방향 분할에 중립, B는1/√L 할인을 준다. 실제 동적 모델에서 A가 완전히 분산 중립이라는 주장은 하지 않는다. 공통 hidden 정합 항도 분산에 영향을 줄 수 있다.

G/E 값만 새로 계산하고 detach하면 안 된다. P/K에 대한 비용 gradient까지 전달한다. 전체 logical B의 Gram을 사용하고 root를 microbatch별로 나눠 합하지 않는다. 0 norm의 선택 subgradient는0이다. Norm smoothing·임의 jitter·diagonal 근사를 추가하지 않는다. FP64 quadratic의 작은 음수 roundoff 처리는 구현 계약에 고정하고 성능에 따라 바꾸지 않는다.

## 7 Solve와 선형 적용의 역전파

A_l=L_l L_lᵀ를 batch당 한 번 분해해 재사용한다. 안정적인 full-context dual은

\[
F=L_l^{-1}K\Omega^{1/2},\quad S=I_C+F^\top F,\quad
V=S^{-1}\Omega^{1/2}Z^\top,\quad T=FV,\quad P=L_l^{-\top}T.
\]

G=TᵀT로 계산할 수 있다. S와 RHS는 요청 수 B만이 아니라 전체 context 수 C에 의존한다. Backend가 바뀌어도 같은 전체-context solve를 수행해야 한다. A factor만 고정하며 F/S/P는 상층에서 후보마다 새로 계산한다.

\[
dP=M^{-1}\{dK\Omega(Z^\top-K^\top P)-K\Omega dK^\top P\}.
\]

Q=∂L/∂P, Y=M^{-ᵀ}Q라면 solve 경로의 key adjoint는

\[
\bar K_{solve}=Y(Z-P^\top K)\Omega-P(Y^\top K)\Omega.
\]

여기에 E의 명시적 K 의존성과 activation 경로의 key gradient를 더한다. A는 batch 내 상수여서 A gradient는 학습하지 않는다. M solve의 forward factor/dual operator를 backward에서 재사용하되 다른 후보의 factor를 섞지 않는다.

Modified linear의 입력을 X, 출력 adjoint를 B_out이라 하면 실수 연산 기준

\[
\bar D=B_{out}^\top(XP),\qquad
\bar P=X^\top(B_{out}D),\qquad
\bar X=B_{out}\widetilde W
\]

다. 전체 weight gradient를 만들 필요는 없지만 **D/P/input 세 방향을 모두** 반환해야 한다. 기존 v5/v6 direct-D 함수의 P gradient=None 규칙은 사용할 수 없다. Reassociation·cast는 표준 autograd의 cast 미분 관례에 맞춰 실제 dense reference와 검증한다. FP32 rounding 함수에 대한 고전적 의미의 정확한 미분 또는 bitwise 동일성은 주장하지 않는다.

## 8 실제 writer의 정합과 native 보존

Native 후보5·10·15·20의 **Adam update 전**에만 기존 physical auxiliary를 추가한다. 완료한 update는 각각4·9·14·19회다. Current 요청은 seeded four disjoint partition I_j, 고정 replay S_t는 four partition J_j로 나눠 각 요청을 총 한 번 포함한다. 빈 partition은 loss/forward를 생략한다.

Native teacher는 같은 후보 D의 native joint forward에서 얻은 post-injection subject hidden과 native NLL readout의 full-vocabulary target-position 분포를 stop-gradient한다. Actual 후보는 위 causal builder가 구성한 전체-B weight다.

\[
C_h(I)=\operatorname{mean}_{r\in I}\frac12\sum_l\sum_{c\in RW_r}w_{rc}
\frac{\|h^a_{lrc}[s]-\operatorname{sg}(h^v_{lrc}[s])\|^2}{a_{lr}^2},
\]

\[
C_d(I)=\operatorname{mean}_{r\in I}\sum_cw_{rc}\operatorname{mean}_{target\ token}
KL(\operatorname{sg}(p^v_{rc})\Vert p^a_{rc}),
\]

\[
C_{current}(I)=\beta_h C_h(I)+\beta_d C_d(I)
+\lambda_{KL}\operatorname{mean}_{r\in I}KL(p_{r,KL}^a\Vert p_r^{entry}).
\]

Teacher detach와 writer P detach는 다르다. Teacher는 학습 목표로만 고정하지만, actual P/K builder와 모든 δ의 연결은 유지한다. Selected request만 loss에 포함해도 physical weight는 전체 B로 구성한다. Current actual target NLL은 별도 primary loss로 추가하지 않는다.

Past memory는 두 arm에 동일하다. 이미 사용한 native rewrite/KL만 저장하며 capacity128 unique facts, 이번 sample M_p=min(16,B,eligible resident facts)를 batch 동안 고정한다. Current re-edit fact version은 replay에서 제외한다. KL teacher는 admission 당시 실제 entry 분포를 유지하고 동일 token-input identity의 teacher 소유권을 공유한다. Rewrite baseline b_rc는 해당 fact의 최신 actual commit 직후 NLL이다. 성공 여부로 admission/eviction을 정하지 않고, 기존 v5 Algorithm R의 unique-fact/version/shared-teacher/refcount/eviction 계약을 유지한다. 모델·benchmark가 바뀌면 memory identity를 섞지 않는다.

\[
R_{past}(J)=\lambda_{KL}\operatorname{mean}_{r\in J}KL(p_r^a\Vert\pi_r^{admission})
+\lambda_{past,E}\operatorname{mean}_{r\in J}\sum_cw_{rc}\frac12[\ell_{rc}^a-b_{rc}]_+^2.
\]

Probe에서 L=L_base+(|I_j|/B)C_current+(|J_j|/M_p)R_past, 그 외에는 L=L_base다. 빈 항은0이다. 네 partition 가중치는 각각 합1이며 임의4배 보정이나 stale auxiliary gradient 재사용은 없다. 이는 정해진 횟수의 teacher update 알고리즘이며, 고정된 단일 objective의 정확한 gradient나 수렴 보장을 주장하지 않는다.

## 9 Optimizer와 후보 순서

Absolute D의 Adam: native profile LR 고정(현재 .1), betas(.9,.999), eps1e-8, weight_decay0. **Warmup은 없다.** 전체 평균 loss에 실제 B를 곱해 request-SUM으로 backward한다. Native·policy·current·past 모두 같은 전체 scale을 사용한다.

후보1은 D=0, 후보25는24번 update 이후다. 후보1–24에서 전체 native gradient, dynamic policy gradient, 해당 pulse의 actual gradient를 같은 D.grad에 누적하고 한 번만 Adam step한다. 모든 checkpoint/VJP가 끝난 뒤에만 D를 변경한다. 각 δ_lr를 c_native a_lr ball로 사후 projection한다. Moment를 임의 초기화하지 않는다.

후보25에는 backward와 추가 Adam update가 없다. 기술 오류는 실패로 기록하지만 clamp 비율·층 집중·낮은 RS/PS/NS·고정 예산 미수렴을 후보 거절이나 추가 step의 이유로 삼지 않는다. 실험 중 profile을 재튜닝하지 않는다.

## 10 최종 candidate와 commit

후보25의 D로 **동일한 causal writer 정의**를 다시 구성한다. 전체 current native/actual 관측을 수행하고, actual forward에서 재포착한 각 pre-own-down key와 builder key를 비교한다. 같은 causal model에서는 이후 write가 이전 key를 바꾸지 않으므로 두 key는 수치 허용오차 안에서 일치해야 한다. 최종 학습된 모델의 inference에 subject δ hook은 없다.

평가한 materialized FP32 tensors를 그대로 commit한다. 이는 처음부터 정의된 후보의 구성 과정이며, 평가 이후 다른 key로 다시 solve하는 후처리가 아니다. Final strength rescale, absolute-z residual 보상, 평가 후 다른 후보 선택은 없다.

History에는 actual terminal full-context key second moment를 요청당·층당 한 번 누적한다. Actual terminal rewrite NLL을 memory b로 사용한다. Terminal gradient는 null, gradient_measured=false다. 최종 P 자체를 다음 batch에서 재사용하지 않는다.

## 11 계산량과 정확성을 보존하는 구현

Reference 실행은 후보마다 전체 current native RW 문맥의 physical key-builder sweep을 수행한다. 마지막 write site의 input key까지면 geometry 계산에 충분하다. 각 층에서 모든 microbatch key를 모으는 barrier 후 전체-B solve를 하고 다음 층으로 간다. Layer마다 모델 처음부터 다시 forward할 필요는 없다.

Native branch의 subject 이전 causal-prefix cache와 actual branch의 첫 modified down_proj 이전 cache를 구분한다. 하층 write 이후의 상층 entry cache를 실제 current key 대신 쓰지 않는다.

Qualified causal adapter에서는 **key 추출용 별도 builder row만 subject 위치까지** 계산할 수 있다. 미래 token은 그 subject key에 영향을 주지 않으며 원 token IDs/mask/position/context 소유권을 보존한다. Native NLL/KL 학습 입력과 실제 보조/terminal 전체 prompt는 그대로다. 이 최적화는 full-row reference와 key 및 전체 D gradient parity를 확인한 뒤 사용한다. 잘린 builder의 hidden으로 full-prompt target suffix를 바로 대체하지 않는다.

Factor CPU 보관과 layer별 streaming, immutable candidate stage checkpoint, direct D/P VJP, 필요한 위치의 full-vocabulary head를 사용한다. Checkpoint 재실행 시 같은 후보의 실제 weights/solve를 복구해야 하며 종료된 hook context에 의존하면 안 된다. Loss가 일부 요청에서 계산돼도 builder의 전체-B gradient 경로를 보존한다.

**v6의 27B+M_p forward /25B+M_p backward bundle과 추가시간40% 목표는 v7의 비용 모델로 사용할 수 없다.** 분리 원장은 다음과 같다.

- Native full input:25회 forward,24회 backward.
- Whole-current RW key builder:25회 forward,24회 backward의 reference 일정; full-depth/head 계산과 구분.
- A factorization:batch당 각 eligible layer1회. 첫층 P는1회, 나머지 P는각25회 재계산. 현재5층이면 상층 solve100회, 각24회 backward.
- 네 selected physical pulse:current 합B, past 합M_p의 full native bundle. 최종 actual current B forward. Builder와 검증된 재사용이 있으면 실제 호출 수에서 중복을 차감한다.
- Checkpoint replay·solve VJP·materialization·data transfer·setup/history/official 평가를 별도 계측한다.

d_in14336의 FP64 dense A factor 저장량은 층당 약1.64GB,5층 약8.22GB다(임시 buffer 제외). Dual context600의 RHS가 작은B=100으로 줄어든다고 계산하지 않는다. 시간·GPU memory는 실제 B100 경로 측정 전 보장하지 않는다. 자원 제한을 맞추려고 key를 detach하거나 context를 줄여 같은 method라고 표기하지 않는다.

## 12 첫 실험과 해석 범위

두 arm은 norm aggregation만 다르며 이 dynamic writer는 공통으로 적용한다. 현재 profile은 Llama3-8B-Instruct/CounterFact, L4–L8, FP32 model/FP64 geometry, native LR .1·norm .5·KL .0625·clamp .75·C0 weight15000, λ_W=λ_E=β_h=β_d=.1, past NLL coefficient1이다. 이전과 동일한 시작값이며 optimal 값을 주장하지 않는다.

첫 실행 범위는 각 arm cold W0/H0의 BS100×5=500 edits다. Native 문장과 순서를 유지하고 official P/N을 tuning에 사용하지 않는다. Method 자체는 B·model·benchmark에 고정되지 않는다. W5에서 R500/P1000/N5000와 pre-edit locality 및 at-write retention을 보고한다. 상세 pilot·기술 tolerance·실행 상태는 별도 실험 및 구현 계약으로 고정한다.

V7은 실제 key 기준의 층별 fitting과 그 상태 적응 gradient를 추가한다. Subject-only 목표와 모든-token weight 편집의 차이, bounded native memory 밖의 locality, 순차 history의 과거 표현 문제는 완전히 해결한 것이 아니다. 기존 v4/v5 성능이나 v6 CPU 대수 검증을 v7의 실제 모델 성능 증거로 대신하지 않는다.
