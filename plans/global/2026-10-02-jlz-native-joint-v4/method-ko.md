# JLZ v4: native local-z 공동 최적화와 상태 기반 편집 배분

2026-10-02. Method 설계와 CPU 수학 검증 단계다. 생산 runner·GPU pilot·성능 검증 완료를 뜻하지 않는다. 이 문서는 직전 shared-subject v3의 후속 방법 방향이며, v3의 R 최적화·full/subject 혼합 목적을 계승하지 않는다. 과거 실행 결과와 파일은 보존한다.

**Compute revision r1:** native 목적과 A/B 정의를 유지하면서 subject-prefix cache, batched head, factorization 재사용, streaming writer와 history key 재사용을 방법의 실행 명세에 반영했다. 실제 GPU 구현에 적용됐다는 뜻은 아니다. 이 문서는 **method의 수식·계산 그래프·불변조건**이며, BS100×20 pilot/main/평가/자원 계획은 별도 [실험 설계문](experiment-2k/experiment-ko.md)에 있다. GH 전달 진입점은 [인계 안내](GH-HANDOFF.md)다.

## 1. 연구 질문과 주장 범위

질문은 **순차 편집의 현재 상태에서, 새 사실을 학습하면서 보존 비용이 낮은 층·방향에 필요한 local 변위를 배분할 수 있는가**이다. Model profile에 선언된 편집 가능 층 전체를 후보로 두고, 층별 local z를 하나의 native task loss에서 공동 결정한다. 이번 Llama3 profile은 L4–L8이다. 한 층 전담도 유효한 해다. 균등 배분, 최소 사용 층 수, 사전 top-k는 없다.

HJ는 native L8 target을 유지한 채 writer 배분을 바꾸면 PS/NS의 균형이 달라진다는 동기를 제공했다. Misalignment가 NS 손실의 유일한 원인이라는 증거는 아니다. 기존 JLZ의 낮은 PS도 목적·변수·규제 공간·solver가 함께 바뀐 결과다. V4는 그 결과를 local-z 공동 최적화 자체의 실패로 취급하지 않는다.

BLUE는 boundary-layer local target, CAKE는 causal-weighted residual assignment, FE는 first-anchor forward replay를 다룬다. HiEdit는 lifelong preservation과 adaptive layer selection을 이미 결합한다. 신규성 후보는 배분 일반론이 아니라 **native local 변위의 공동 결정, 모든 허용 층의 연속 편집량, 실제 writer geometry/history를 통한 z 단계 비용**의 구체적인 조합이다. 전체 문헌에 대한 최초성이나 성능 우위를 확정하지 않는다.

## 2. 고정할 것과 바꿀 것

| 항목 | v4 정의 |
|---|---|
| 이번 실험 요청 | 기존 fixed10k 첫2k, BS100×20, 동일 순서·분모 |
| 학습 문장 | 요청당 기존 native rewrite6개 + KL1개; 새 문장·공식 P/N 추가 없음 |
| Target/packing | 기존 native tokenizer·leading-space/BOS 처리·teacher forcing·subject-last 위치 |
| NLL | target token 평균, rewrite context 1/6 평균, 요청 합 |
| Key | native mean-of-group-means: canonical .5 + 나머지 각각 .1 |
| 학습 변수 | 모든 층·요청의 실제 subject intervention δ_l,r |
| Task forward | entry 모델에 모든 δ를 동시에 주입하는 한 경로 |
| Norm/clamp | δ 자체에 native 계수와 entry anchor로 적용 |
| 보존 정보 | 기존 C0와 누적 history H; B에서 z 단계 비용에 사용 |
| 제거 | R norm, full/subject 2:1 혼합, 별도 정합 loss, 새 G/E prompt-loss |
| 실제 평가 | 최종 weight 모델에서만 RS/PS/NS·retention 평가 |

기존 JLZ의 general/replay **문장 기반 loss는 v4의 기본 방법에서 제거**한다. Sequential history Gram H는 유지한다. 따라서 native KL 문장 외에 새 reference 문장을 학습에 추가하지 않는다. H를 유지하는 것과 별도 reference NLL/KL를 학습하는 것은 다르다.

입력 identity는 `inputs/exact-native-inputs.json`에 결속한다. 현지 동일 입력 검사는 기존 요청·target·context identity의 재사용이며, 아직 존재하지 않는 v4 runner의 token equivalence 검증으로 과장하지 않는다.

### 2.1 일반 method와 이번 실험 profile의 구분

**B=100, L4–L8, hidden4096, key14336, rewrite6/KL1, readout31, CounterFact의 P2/N10은 method 상수가 아니다.** 아래 표의 일반 interface를 사용하고 이번 실험에서만 해당 값을 profile로 결속한다. `portability-contract.json`이 interface와 지원 조건의 정본이다. Adapter/runner 구현 완료 또는 다른 모델·benchmark에서의 실증을 주장하지 않는다.

| 구분 | 일반 method | 이번 실행 profile |
|---|---|---|
| Logical batch | 임의의 양의 B_t, 마지막 partial batch 포함 | B_t=100,20회,2k |
| 편집층 | 순서가 있는 전체 eligible 집합 ℒ; 층별 d_in,l / d_out,l | L4–L8;14336→4096 |
| 학습 행 | request r의 native rewrite 집합 C_r와 KL 집합 K_r, native 가중치 | rewrite6+KL1 |
| Intervention | model adapter가 지정한 native site와 request별 anchor 위치 | block output, subject-last |
| Writer | native site의 additive displacement와 호환되는 layer-local linear write map | MLP down_proj |
| Readout | model/native profile의 NLL·KL readout과 full vocabulary | NLL31, KL final |
| Native 계수 | baseline/model profile의 KL/norm/clamp/learning-rate | .0625/.5/.75/.1 |
| 데이터 | benchmark adapter의 ordered requests, target/identity/conflict 규칙 | fixed CounterFact 첫2k |
| 평가 | benchmark 고유 test family와 실제 prompt/target 분모 | R/P/N=2000/4000/20000 |

Model adapter는 layer/module/hook 위치, native delta→writer additive 연결, lookup/RoPE/mask, tokenizer/BOS/EOS, 원 norm·head, C0와 history 저장 규약을 제공한다. 합쳐진 projection이나 weight transpose는 adapter가 명시하며 dimension을 layer마다 검증한다. 의미가 다른 output에 같은 크기의 벡터를 넣는 것으로 alignment를 대체하지 않는다. 같은 parameter가 여러 위치에서 공유되거나 module 입력이 자기 weight에 의존하는 구조는 단순 순차 writer/history key 재사용의 조건을 다시 검증한다. 여러 eligible site가 같은 parameter를 쓰면 alias를 반영한 하나의 유효한 write 규약이 필요하다. 이를 제공하지 않는 adapter에서는 cache만 꺼서 지원된다고 취급하지 않는다.

Benchmark adapter는 요청별 가변 개수의 native 학습 context/KL 입력·가중치, target token, intervention anchor, key용 입력/pooling, 평가 panel 및 claim-conflict 식별을 반환한다. 새 benchmark에서 native policy가 제공되지 않으면 profile을 먼저 정의하고 기록한다. CounterFact의 `{} is a`, six-context, subject+relation conflict 규칙을 다른 benchmark에 자동 이식하지 않는다. Paraphrase/neighborhood가 없는 benchmark에 가짜 P/N 분모나 점수를 만들지 않는다. Native와 같은 학습 입력이라는 원칙은 **동일 model+benchmark profile의 baseline과 동일**하다는 뜻이다.

B_t는 실제 request 수에서 구하고 D_l∈R^(d_out,l×B_t), K_l∈R^(d_in,l×B_t)를 만든다. 예컨대2k를BS128로 구성하면15개 full batch와 마지막80개를 처리한다. Padding용 가짜 request는 Q·손실·H·평가에 넣지 않는다. Logical batch 크기를 바꾸면 request 결합과 순차 history 경계가 바뀌므로 **batch-size 불변 결과를 보장하는 것은 아니다**. Microbatch는 같은 logical objective의 계산 분할일 뿐이며 Adam step이나 write의 단위가 아니다.

B가 작으면 dual B×B Cholesky를, B가 커서 메모리에 맞지 않거나 d_in보다 크면 primal d_in×d_in factor와 blocked RHS를 사용할 수 있다. Q/E를 전부 저장할 필요 없이

\[
E_l X=X-K_l^T\operatorname{solve}(A_l+K_lK_l^T,K_lX),\quad
G_l=(E_lD_l^T)^T/s_l^2
\]

의 정확한 operator action을 column tile로 계산한다. 각 KX와 KK^T는 실제 B_t 전체 요청의 기여를 모은 뒤 solve한다. Request tile마다 별도 solve/Adam/write를 하지 않는다. P와 U도 RHS/output tile로 materialize할 수 있다. 이는 cross-request 비대각 결합을 모두 유지한다. Near-cancellation·solve residual은 qualification 대상이며 대각화/top-k/저랭크 근사나 logical batch 분할로 몰래 대체하지 않는다. 큰 B의 Q spectrum telemetry는 `NOT_MATERIALIZED`로 둘 수 있다. 임의 B를 받는 interface와 유한한 GPU/RAM에서 무제한 B를 실행할 수 있다는 보장은 다르다.

Prefix 최적화는 causal decoder와 tokenwise layer 연산 등 정확한 의존성 조건이 성립하는 경우에 켠다. Cache 경계는 첫 intervention **연산 직전**이다. e=min ℒ의 block output에 개입하는 이번 profile에서만 L0–Le의 출력까지 재사용하고 e+1부터 반복한다. Block 내부 개입이면 같은 block의 남은 연산도 후보마다 다시 계산하며 실제 readout까지 이어진다. Bidirectional attention, encoder–decoder의 cross-attention, recurrent/shared-weight 구조 등에서 같은 증명이 성립하지 않으면 해당 cache/future-prune/streaming 최적화만 끄고 **동일 adapter 목적의 full reference forward·key 재측정**을 사용한다. Valid native site/write map 자체가 없는 모델은 adapter 구현이 필요하며 “이미 지원”이라고 표기하지 않는다. 따라서 모델별 capability에 따라 계산 경로를 선택하되 모든 eligible 층과 공동 목적은 유지한다.

## 3. 학습 변수와 local z의 의미

Batch t의 entry 모델을 W_t, 편집 층 집합을 ℒ라 한다. 각 요청 r마다 δ_l,r∈R^(d_out,l)를 직접 최적화하고, D_l=[δ_l,1,…,δ_l,B_t]∈R^(d_out,l×B_t)로 묶는다. 이번 profile은 ℒ={4,5,6,7,8}, d_out,l=4096이다. D_l은 원래 JLZ의 writer residual 좌표 R_l을 이름만 바꾼 것이 아니다. 이하 L4/L8, six-context 등 구체 표현은 이번 profile의 계산 예이며 일반 경계는 §2.1을 따른다.

원 native `compute_z`와 같은 block-output subject-last 위치에서

\[
h_{l,r,c}^{+}=h_{l,r,c}(\delta_{<l})+\delta_{l,r}
\]

를 적용한다. 여섯 rewrite context와 native KL 문장에서 해당 요청의 같은 δ_l,r를 사용한다. 상위 층의 입력은 하위 층 intervention이 반영된 실제 joint forward의 값이다. Independent local fit 다섯 개를 수행한 뒤 합치지 않는다.

이때 absolute local target은 문맥별 현재 표현에 대한 `z_l,r,c=h_l,r,c(δ_<l)+δ_l,r`다. `entry h_l,r^0+δ_l,r`라는 고정 absolute target으로 상층을 덮어쓰지 않는다. h_l,r^0는 canonical entry block output이며 norm/clamp 기준으로만 고정한다.

Llama에서 down_proj output 뒤의 residual addition은 additive 변위를 block output으로 연결한다. 그러나 absolute down_proj output과 block output은 같지 않다. 학습은 native block-output hook을 기준으로 하고, 실제 writer의 down_proj 변위와 비교할 때는 additive δ를 비교한다. FP32 덧셈 순서까지 bitwise 동일하다는 주장은 별도 검증 대상이다.

## 4. Native 공동 목적

W_t는 fit 동안 frozen이고, 모든 D_l을 0에서 초기화한다. Native KL teacher는 동일 입력의 W_t 분포이며, 방향은 current‖entry다.

\[
J_A(D)=\sum_{r=1}^{B_t}\left[
\mathrm{NLL}^{\rm native}_r(F_D)+\lambda_{\rm KL}\,\mathrm{KL}^{\rm native}_r(p_D\Vert p_t)
+\sum_{l\in\mathcal L}\lambda_{\rm norm}\frac{\|\delta_{l,r}\|_2}{\|h^0_{l,r}\|_2^2}\right],
\qquad \|\delta_{l,r}\|_2\le c_{\rm clamp}\|h^0_{l,r}\|_2.
\]

이번 profile의 λ_KL=.0625, λ_norm=.5, c_clamp=.75다. NLL은 Llama3의 native loss layer31을, KL은 원래 final logits의 subject-last를 사용한다. 층 수에 따른 임의의 1/|ℒ| 평균은 넣지 않는다. 단일 층만 유효하고 η=0이면 해당 층의 native 목적 형태로 돌아간다. 다만 공동 solver의 궤적·정지 규칙까지 원래의 독립 native 호출과 같다는 뜻은 아니다.

Norm은 L2 노름이며 제곱이 아니다. Native KL, native norm, 아래의 quadratic burden은 별도 항으로 기록한다. 모두를 “native loss”로 묶어 표시하지 않는다.

## 5. Writer를 통한 상태 의존적 배분 비용

Batch entry의 각 층에서 target-free native pooled key K_l∈R^(d_in×B)를 측정한다. History는 batch 시작 시점에 고정한다.

\[
A_l=\lambda_C C_{0,l}+H_{t,l},\qquad
P_l=(A_l+K_lK_l^\top)^{-1}K_l,\qquad Q_l=K_l^\top P_l.
\]

역행렬을 생성하지 않고 FP64 solve를 사용한다. A_l≻0을 전제로 Q_l은 대칭이며 0≼Q_l≺I다. 수치 계산에서는 대칭화하여 사용하고, solve residual과 고유값의 반올림 오차 범위를 기록한다. 임의의 jitter, 층 제외, inverse-gain 보정은 수행하지 않는다.

이번 profile은 λ_C=15000이며 B=B_t다. Geometry/history dtype과 C0 원본은 model profile에 결속한다. 큰 B에서 Q/E를 implicit operator로 표현하는 것은 같은 수식의 구현이며, Q를 대각 근사하는 것은 다른 방법이다.

임의의 요청 local 변위 D_l을 이 geometry로 쓰는 ridge 문제는

\[
V_l(D_l)=\min_U\left\{
\frac12\|UK_l-D_l\|_F^2+\frac12\operatorname{tr}(UA_lU^\top)\right\}.
\]

그 해와 최적값은

\[
U_l^*(D_l)=D_lP_l^\top,\qquad
V_l(D_l)=\frac12\operatorname{tr}\left[D_l(I-Q_l)D_l^\top\right].
\]

여기서 Q−Q²=PᵀAP이므로 비용은

\[
V_l=\underbrace{\tfrac12\|D_l(Q_l-I)\|_F^2}_{\text{실현하지 못한 local 변위}}
+\underbrace{\tfrac12\operatorname{tr}[D_l(Q_l-Q_l^2)D_l^\top]}_{\text{보호 geometry에 대한 write 비용}}
\]

으로 나뉜다. 거의 쓸 수 없는 Q≈0 방향에서는 write energy만 보면 작아 보이지만, V는 미실현분을 포함한다. 따라서 “목표를 반영하지 못해서 비용이 낮은 것”을 좋은 배분으로 오인할 가능성을 줄인다.

고정된 K,D에서 H가 PSD 방향으로 증가하면 V는 감소하지 않는다. 다만 실제 다음 batch에서는 K와 W도 달라지므로 전체 trajectory에 같은 단조성을 주장하지 않는다. H나 V는 기능적 NS 손상 자체가 아니라 local geometry에 따른 proxy다. 모든 층에서 Q가 충분히 작으면 V≈.5||D||²가 되어 세밀한 상태 차이를 반영하기 어려워진다. 이러한 포화도 기록한다.

정규화와 gradient는

\[
s_l^2=\frac1B\sum_r\|h^0_{l,r}\|_2^2,\quad
\mathcal V_t(D)=\sum_l V_l(D_l)/s_l^2,\quad
\nabla_{D_l}\mathcal V_t=D_l(I-Q_l)/s_l^2.
\]

실제 writer 문제는 정규화하지 않은 D를 사용한다. D의 column을 anchor로 미리 변환한 뒤 V를 계산하면 다른 writer 문제가 되므로 그렇게 하지 않는다. Q의 off-diagonal을 버리지 않고 같은 batch 내 request 간 key 결합을 유지한다.

Q의 대칭화, I−Q, V 및 analytic gradient까지 FP64로 계산한다. 최종 비용 gradient만 δ의 FP32 dtype으로 변환해 native task/norm gradient에 누적한다. 이 연산 규약도 구현의 일부이며, FP64 CPU 수학 검산이 실제 FP32 누적 오차까지 검증한 것은 아니다.

## 6. 두 arm과 dynamic allocation

\[
J_\eta=J_A+\eta\mathcal V_t,\qquad \eta\in\{0,1\}.
\]

| Arm | z fit | 해석 |
|---|---|---|
| A | native joint 목적만 사용, η=0 | 공동 local-z 자체의 control. History를 사용하는 writer는 공통이지만 z 단계의 명시적 보존 비용은 없음 |
| B | A + writer 최적값 V, η=1 | current task 효과와 history 의존적 실현·보호 비용으로 z 단계부터 배분 |

모든 데이터·teacher·native norm/clamp·solver·writer·예산은 공통이다. 차이는 추가 V와 그 gradient뿐이다. **B는 제안하는 상태 의존적 배분이고, A는 그 추가 비용을 분리하는 control**이며, A에도 같은 손상 인식 정책이 있다고 설명하지 않는다.

η=1은 최초의 명시적 설정이며 native 상수에서 유도하거나 실측으로 보정한 값이 아니다. 너무 강하면 strength를 낮출 수 있다. B의 NS만 평가하지 않고 같은 평가 분모의 PS·retention·비용을 함께 기록한다. 공식 P/N으로 η나 후보를 자동 선택하지 않는다.

별도의 자유로운 배분 계수 a_l을 δ_l에 곱하지 않는다. a_l과 δ_l을 모두 자유롭게 두면 scale을 식별할 수 없게 된다. 방향과 크기는 D 자체에서 공동 결정한다. 보고용으로 ||δ_l,r||²/||h_l,r^0||²의 층별 비율을 사용할 수 있지만, 이는 norm share이며 인과적 기여율이 아니다. 분모가 0이면 NA다.

V는 여러 층이 분담하는 것을 유리하게 만들 수 있지만 균등화 제약은 아니다. 단일 층·여러 층·0개 층 사용을 허용한다. 특정 층에 집중한 것 자체를 실패나 재배분 trigger로 삼지 않는다.

## 7. Solver와 정지·후보 채택

초기 구현은 native에 가까운 Adam을 사용한다. lr=.1, betas=(.9,.999), eps=1e-8, optimizer weight_decay=0이다. Objective의 norm 항은 명시적으로 미분하며 norm=0의 subgradient는 PyTorch/native와 같은 0으로 둔다. 모든 δ의 gradient를 한 번 모아 모든 층을 같은 Adam step으로 갱신하고 block별로 ball에 투영한다.

δ와 Adam state는 FP32다. 초기 δ=0에서도 injection hook을 미분 가능한 덧셈으로 유지해야 한다. 값이 0이라는 이유로 hook을 생략하면 그 층의 최초 task gradient까지 사라지므로 허용하지 않는다.

고정 예산은 **joint loss/gradient 후보 평가 25회, Adam 갱신 최대 24회**다. 마지막 후보는 loss 평가 후 갱신하지 않고 반환한다. 마지막 평가에서 backward를 생략할 수 있으면 logical25, physical backward24로 구분해 기록한다. Microbatch마다 Adam step을 수행하지 않는다. Native norm과 B의 V는 후보당 한 번만 더하고 microbatch 수를 곱하지 않는다.

독립 native의 request별 정지나 기존 JLZ의 entryNLL<.05 고정 column mask는 계승하지 않는다. Joint forward, 특히 Q의 request 결합에서는 다른 변수가 해당 request를 바꿀 수 있기 때문이다. 모든 requested 항목을 loss에 포함하며 초기 상태만으로 request나 layer를 동결하지 않는다. 이는 명시적 solver 변경이다.

중간 PS/NS, layer share, realization residual로 후보를 선별하지 않는다. 예산 종료 시 최종 유한 후보를 사용하며 미수렴을 실패와 혼동하지 않는다. Nonfinite 등 수치 오류는 transaction을 중단하고 기록한다. 과학적 quality gate와 기술적 성립 조건을 구분한다.

## 8. 실제 write: z 단계에서 정한 payload 유지

Fit 후 D_l^*를 고정하고 학습 hook을 제거한다. 실제 모델 W_t에서 L4→L8 순서로 다음을 수행한다.

1. 이미 반영한 하층 write를 포함하는 모델에서 해당 층의 target-free native pooled key K_l^cur를 다시 얻는다.
2. 같은 batch-entry A_l을 사용하여 P_l^cur=solve(A_l+K_l^cur(K_l^cur)ᵀ,K_l^cur)를 구한다.
3. `U_l=D_l^* (P_l^cur)ᵀ`를 FP64로 만들고 기존 FP32 materialization 순서로 W_l에 더한다.
4. 이상적인 pooled 실현량 D_l^*Q_l^cur, 실제 materialized update의 pooled 실현량, D_l^*와의 차이를 기록한다.

실제 materialized action은 FP32 덧셈이 끝난 두 가중치로 `ΔW_eff=W_new.double()−W_entry.double()`를 구한 뒤 `ΔW_eff @ K_current.double()`로 측정한다. 이는 단순한 `cast32(U) @ K`와 다를 수 있다. 실제 두 `linear()` output의 차이에는 GEMM 반올림도 추가될 수 있으므로 별도로 구분한다.

local 목표는 해당 시점의 현재 출력에 δ_l^*를 더한 것이므로 writer에 전달하는 residual은 D_l^* 자체다. `target=z_l^virtual−h_l^actual`이라는 상층 보정으로 대체하지 않는다. L8 잔차, remaining-layer divisor, 사후 비율 조정, Q의 역수에 의한 gain 증폭도 사용하지 않는다.

이를 통해 z 단계에서 정한 **요청 변위의 배분**을 write 단계에서 재배분하지 않는다. 다만 요청 D와 실현 UK는 다를 수 있다. 실제 weight allocation까지 z의 norm share와 같다고 주장하지 않는다.

Fit 중 V는 **entry K에 대한 고정 proxy writer의 최적값**이다. 실제 write는 갱신 후 current K를 사용하므로 V가 최종 writer의 정확한 최적값이라고 주장하지 않는다. Entry/current Q 차이도 기록한다. Fit 중 current-key writer를 후보마다 다시 실행하는 고비용 bilevel 방법은 이번 기본 방법에 넣지 않는다.

모든 층을 write한 뒤 post-all-layer native keys로 H를 층마다 한 번 갱신한다. 기본 실행은 각 층의 write 직전에 얻은 current key를 보관했다가 사용하며, 마지막 전체 key 재측정 forward는 생략한다. 해당 down_proj의 **입력**은 자기 weight 변경이나 뒤의 상층 weight 변경에 의존하지 않으므로, 같은 target-free 입력에서 `K_l^prewrite=K_l^post-all-write`이기 때문이다. 이는 상층에 entry key를 그대로 쓰는 것과 다르다. Native FP32 pooling·Gram·H 덧셈 순서를 그대로 보존하고, 모든 write 성공 후에만 append한다. 변위가 0인 층도 H의 commit 규약은 공통으로 적용한다. Fit의 가상 intervention이나 geometry 계산에서는 H를 갱신하지 않는다. 도중에 technical failure가 발생하면 전체 5개 층과 H를 batch-entry로 rollback한다.

## 9. 무엇을 align하고 무엇이 남는가

해소하려는 것은 “L8에서 구한 변위를 다른 층의 local 변위로 취급하는” target-space 불일치다. 각 δ는 처음부터 해당 write 층에서 학습한다.

남는 것은 ridge shrinkage, context-pooled key와 각 context key의 차이, entry proxy와 current geometry의 차이, 이상적 subject intervention과 실제 all-token write의 차이다. 이를 숨긴 채 완전한 align이나 locality 보장이라고 부르지 않는다. V는 미실현분을 z 단계 비용에 포함하지만 정확한 실현 constraint는 아니다.

편집 성공 측면의 cross effect는 하나의 joint forward/backward에 포함된다. 보존 측면의 V는 층별 geometry의 합이며 nonlinear cross-layer 손상을 전부 계산한 것은 아니다. 둘을 구분한다.

## 10. 연산 최적화를 포함한 실행 방법

아래 항목을 **구현 대상 기본 경로**로 채택한다. FP32 kernel shape·합산 순서가 달라지는 항목은 같은 입력/고정 후보의 loss·모든 δ gradient·writer state를 reference와 확인한다. 아직 runner가 없으므로 “기본 경로로 채택”은 설계 결정을 뜻한다. 정합을 확인하지 못한 빠른 경로 대신 같은 method의 단순 reference를 사용하고 선택과 실제 비용을 기록할 수 있다. Quality 기준으로 경로를 선택하지 않는다.

### 10.1 직접 δ oracle과 정확한 인과 prefix 재사용

Fit은 후보당 논리적으로 native7B행의 subject-intervention 경로 하나다. 모델 weight는 frozen이고 δ만 학습한다. Dense weight gradient, R 좌표, 후보별 weight materialization, full/subject 이중 loss가 없다.

행 i의 subject-last index를 s_i라 하면 첫 개입은 L4 **block output**의 s_i에서 일어난다. 배치 진입 모델에서 다음을 고정한다.

1. L0–L4의 개입 이전 계산과 L4 출력. L4의 s_i 이후 출력도 첫 개입 이전 값이므로 entry 상수다.
2. L5–L31의 각 attention에서 j<s_i인 prefix K/V. Prefix에는 subject token s_i를 포함하지 않는다.
3. 원 absolute position IDs/RoPE, 유효 토큰 mask, 각 행의 s_i·길이 및 context/request 매핑.

후보마다 L4 출력의 s_i에 δ4를 더하고 L5–L31의 **s_i부터 끝까지**를 다시 계산한다. L5–L8의 동일 위치에 각 δ를 추가한다. 각 suffix query는 cache의 전체 유효 prefix와 현재 계산한 causal suffix K/V에 attention한다. Subject 자체와 이후 위치의 상층 hidden/K/V는 하층 δ에 의존하므로 cache로 고정하거나 detach하지 않는다. 이 조건에서 공동 gradient와 층간 cross effect가 유지된다.

Cache는 own arm·batch-entry weight state·행 token/mask/position·layer/backend·dtype에 귀속한다. 다음 batch·다른 arm·weight 변경 후 재사용하지 않는다. Padding 길이가 다른 prefix를 결합할 때 padding K/V는 mask하고 original absolute position을 복원한다. `is_causal` flag만으로 비정방형 cached attention의 offset을 추정하지 않는다.

### 10.2 Head, padding과 KL의 사용하지 않는 미래 계산

- 필요한 NLL target prediction 위치와 KL subject 위치의 hidden을 microbatch별로 모아 **한 번의 full-vocabulary head**로 계산한다. 위치를 scatter하여 기존 요청별 token mean/context mean/request sum을 유지한다. NLL layer31의 norm/readout과 final-logit KL의 readout은 각각 원 정의를 보존하며, 같은 readout일 때만 head 호출을 함께 묶는다. 이미 있던 selected-position 원칙을 새 절감으로 중복 계산하지 않는다.
- 원 token IDs를 다시 tokenize하지 않고 microbatch의 불필요한 오른쪽 padding을 제거한다. 길이별 stable bucket은 fixed row map·scatter/reduction 순서와 함께 고정한다. Request index·target 위치·KL lookup을 원래 좌표에서 suffix 좌표로 정확히 변환한다. Bucket 정렬은 요청 stream이나 writer의 column 순서를 바꾸지 않는다.
- KL 행은 s_i 위치 하나만 읽으므로 s_i보다 뒤의 ` is a` 위치 계산을 생략할 수 있다. **원 KL 문장과 token IDs는 manifest에 그대로 유지**하고, 준비된 token 배열의 관측 위치 이후 계산만 prune한다. 원문을 잘라 다시 tokenize하면 경계 token이 달라질 수 있으므로 허용하지 않는다. 이 가지치기를 writer의 target-free full-token 전파나 공식 evaluator에 무조건 적용하지 않는다.

### 10.3 초기 준비와 첫 후보 평가의 결합

δ=0의 첫 full candidate에서 detached teacher, clean canonical anchor, entry key와 prefix cache를 함께 얻을 수 있다. Teacher는 같은 현재 logits의 detach이며, zero-valued injection도 graph에 남겨 최초 task gradient를 유지한다. Captured anchors/keys/cache는 detach하여 고정 상수로 쓴다.

Teacher-forced rewrite와 target-free key 입력의 subject까지 token·position·lookup이 동일한 행에만 key capture를 합친다. CPU 계수에서는 rewrite12,000행 모두 이 조건을 만족했지만 실행 tokenizer에서도 확인한다. Layer별 h0는 own δ를 더하기 전의 zero-entry 값이다. Pooling은 native mean-of-group-means를 그대로 사용한다.

준비 forward가 첫 loss/gradient를 실제 반환하면 이것이 **candidate1**이고 이후24후보만 추가한다. `no_grad` 준비만 했다면 candidate를 소비하지 않으며 추가 물리 forward로 계상한다. 첫 forward 전체를 `no_grad`로 감싸고 δ gradient를 얻었다고 보고하지 않는다. B의 Q 준비가 완료되기 전 update하지 않으며 준비 중에는 W/H를 바꾸지 않는다.

### 10.4 작은 정책 gradient와 큰 geometry의 재사용

배치 동안 A_l=15000C0_l+H_l는 고정이다. FP64 Cholesky `A_l=L_l L_l^T`를 층당 한 번 준비하고, entry 또는 current K에 대해

\[
T_l=\operatorname{cholsolve}(L_l,K_l),\quad
S_l=I+K_l^T T_l,\quad
P_l=T_l S_l^{-1}
\]

를 계산한다. 마지막 오른쪽 solve는 전치된 linear solve로 구현하고 명시적 큰 inverse는 만들지 않는다. 이는 원 `solve(A+KK^T,K)`와 대수적으로 동일하다. A는 actual writer에서만, B는 entry proxy와 actual writer에서 **같은 batch-entry factor**를 공유한다. 하층 write는 current K를 바꾸지만 A는 바꾸지 않는다. L4는 하층 write가 없으므로 K와 pooling/dtype/state identity까지 같으면 B의 entry P4도 재사용한다.

작은 행렬 `E_l=I−Q_l=S_l^{-1}`를 FP64 small Cholesky solve로 만들면 Q≈I에서 직접 뺄셈하는 취소오차를 줄일 수 있다. Q는 별도로 `sym(K^T P)`로 기록하고 `E+Q≈I`를 확인한다. 이 수치 경로는 원 `I−sym(K^T P)`와 bitwise 같다고 주장하지 않으며 reference parity와 scaled solve residual을 기록한다. Eigenvalue clipping·jitter·diagonal Q 근사를 넣지 않는다. 분해 경로만 실패하면 원 FP64 direct solve로 돌아가 검증하고, 명시적 SPD 가정 또는 원 solve의 기술적 실패는 보고한다.

각 후보에서 `G_l=D_l E_l/s_l^2`를 한 번 계산하고 `V_l/s_l^2=.5<D_l,G_l>`로 비용도 얻는다. Norm과 ηG는 모든 microbatch의 task gradient를 모은 뒤 후보당 한 번 더한다. Q/E의 비대각 성분을 보존한다. 실제 U=DP^T를 매 iteration 만들어 비용을 구하지 않는다.

14336차원 factor는 FP64 층당1.53125GiB,5개7.65625GiB다. Cholesky는 약0.982TFLOP/층,100 RHS의 두 triangular solve는 약0.041TFLOP/층이다(2 FLOP/MAC 근사). 큰 factor는 host RAM 보관·층별 전송을 우선 검토하고 작은 Q/E만 fit GPU에 둔다. CPU/GPU 양쪽 peak에 C0/H·transaction weight·activation·transfer buffer를 모두 포함한다. 메모리 부족 시 factor 재계산/direct 경로로 돌아갈 수 있지만 디스크 resume payload로 저장하지 않는다.

**배치 간 Cholesky rank update는 이번 경로에서 제외한다.** 원 FP32 `H += KK^T`의 Gram/누적 반올림 후 FP64 A를 만드는 규약과 이상적 저랭크 갱신은 다를 수 있다. 각 새 batch에서 실제 entry H로 factor를 다시 만든다.

### 10.5 실제 weight를 사용하는 layer streaming writer

대상은 teacher-forced 학습 행이 아니라 원 target-free rewrite6B행이다. L0–L3를 한 번 계산하고 L4부터 각 층의 down_proj **직전**까지 전 토큰을 처리한다. 모든 요청·context의 key를 모아 B개 column으로 native pooling한 뒤 해당 층의 U를 한 번 결정하고 materialize한다. 그다음 저장한 전 토큰 MLP 입력과 residual에 **W_new**를 적용하여 다음 층으로 넘긴다.

Microbatch마다 key를 일부 모아 즉시 write하면 batch geometry가 바뀌므로 금지한다. Layer-major로 모든 microbatch의 필요한 activation을 GPU/CPU RAM에 보관하고, 전체 B key 수집 → 한 번 write → 전체 행 전파 순서를 지킨다. Activation 보관 대신 block을 다시 계산하면 그 비용을 기록한다. 오래된 hidden에 학습 δ를 더해 actual weight 전파를 대체하지 않는다. Sequential writer 중에도 backward/optimizer는 실행하지 않는다.

각 층 prewrite key를 보관해 §8의 history append에 재사용한다. 마지막 post-key forward가 필요 없고, 별도 key capture를 fallback으로 사용해도 마지막 필요한 L8 down_proj input에서 종료한다. 실제 평가용 L31 forward는 이 절감과 별도로 남는다.

첨부의 층 방문192/44/약9는 조건부 구조 계수다. 192는 `5×32 + history 동시capture32`,44는 `5+6+7+8+9+history9`; history 재사용만 먼저 적용하면35다. 약9는 L0–L8을 이어 계산하며 down_proj 직전에서 멈춰 원래 linear의 중복까지 피한 이상적인 block 방문수다. Block 재실행, layer별 history 재측정, padding·attention·geometry 비용이 있으면 달라지며 전체 runtime 배율이 아니다.

### 10.6 메모리·동기화·공통 경로

BS100,d=4096,5층의 δ/gradient/Adam 두 moments는 FP32 약31.25MiB다. B의 정책 gradient는 후보당 약0.4096GFLOP이며 Transformer F/B를 추가하지 않는다. Fit graph는 microbatch backward 후 해제하고 loss·finite flag·per-request 성분은 device에서 누적한다. 후보 경계에서 finite/clamp를 검사하고 update하며, microbatch마다 `.item()`/CPU 동기화를 수행하지 않는다. CUDA 오류나 발견된 nonfinite를 묵인하는 변경은 아니다.

메모리가 허용하면 microbatch를 늘려 GPU 활용률을 높일 수 있다. 이것은 FLOPs 감소와 구분한다. 공통 경로와 row schedule은 A/B에서 동일하게 선택하고, 메모리 때문에 다른 설정을 쓰면 실행 차이를 기록한다. Main 도중 P/N을 보고 바꾸지 않는다.

Arm A는 요청별 목적이 분리되므로 이론적으로 요청 묶음마다 고정25후보를 수행한 뒤 모두 모아 write할 수 있다. 그러나 B는 Q의 비대각 항으로 요청이 결합된다. **첫 A/B 실험은 두 arm 모두 전체 BS100 joint-step schedule을 유지**하고 A 전용 requestwise 최적화는 후속 구현 항목으로 남긴다. Global clipping·공통 조기종료·정규화·weight state가 바뀌면 A의 분리 가능성도 다시 확인해야 한다.

### 10.7 CPU 입력 계수와 검증 범위

같은 첫2k의20개 배치에 대해 후보당 한 번씩 계산할 입력을 합한 계수다. 25회 반복이나 두 arm을 이미 곱한 값이 아니다. `compute/native-work-counts.json`에 tokenizer/source/input hash와 행별 조건 검증을 결속한다.

| 계수 | 수량 |
|---|---:|
| 원 유효 token | 215,075 |
| subject 이전 cache token | 160,115 |
| 매 후보 subject 포함 suffix token | 54,960 (25.55%) |
| KL 미래 위치 계산 생략 후 suffix | 50,960 |
| NLL/KL head 위치 | 14,144 (6.58%) |
| MB4, 배치 전체 폭 유지 | 354,200 slots |
| MB4, 원 행 순서에서 crop | 261,676 slots |
| MB4, 길이 정렬 후 crop | 215,720 slots |
| ideal unpadded FP32 prefix KV, batch 최대 | 1.6964 GiB |

KV 수치는 GQA의 실제 KV head8개, head dimension128, L5–L31의27층 기준이며 expansion/padding/allocator·suffix activation·모델·geometry를 제외한다. Padding/head/suffix 절감을 서로 독립된 배율처럼 곱하지 않는다. Native L8 개입의23개 후속층과 v4의27개 후속층 비율1.174는 같은 입력/반복이라는 조건의 층 수 비교일 뿐 실제 baseline 시간 배율이 아니다.

Reference 대비 fixed candidate의 loss·gradient, 짧은 Adam 궤적·clamp, 실제 writer weight/key/H의 정합을 GPU pilot에서 확인한다. CPU causal-attention/geometry toy 결과는 `math/compute-reuse-check.json`에 별도 기록한다. 실제 Llama FP32·캐시 kernel·GPU 가속·paraphrase 품질을 검증한 것으로 확대하지 않는다. 원25후보/24update, 모든층, native 입력과 A/B eta만의 차이는 그대로다.

## 11. 구현 단위와 평가

새 namespace는 `project/run_scripts/jlz_native_joint/`로 하며 기존 실험을 hotpatch하지 않는다.

- `inputs.py`: native preparer 공유와 token identity.
- `entry.py`: zero-candidate teacher/anchor/key capture, L4 output·causal prefix cache와 entry identity.
- `allocation.py`: FP64 direct/reference 및 Cholesky/Woodbury geometry, Q/E·V/analytic gradient와 RAM factor 수명.
- `oracle.py`: subject suffix 공동 δ graph, batched full-vocabulary selected head, row/position mapping.
- `optimize.py`: joint Adam, norm/clamp, 고정25후보, microbatch 정규화.
- `writer.py`: full-token layer streaming, 전체 B key 후 FP32 materialization, prewrite key의 post-all-layer H 재사용과 transaction.
- `observe.py`: 실제 모델만의 기존 R/P/N·strict·cohort retention, 요청/실현 배분, 비용.

Pilot과 main은 서로 다른 cold 상태에서 시작한다. Main은 A/B가 각자 W0/H0에서 BS100×20으로 시작한다. Pilot은 구현 성립을 소규모로 확인하며 특정 PS/NS 값이나 배분 형태를 main 진행 gate로 삼지 않는다. 이번 설계 작업에서 실험을 시작하지 않는다.

공식 성적은 실제 weight 모델에서 측정한다. Subject-only fit의 NLL을 실제 편집 성공 대신 사용하지 않는다. Current·all-seen·cohort의 at-write→endpoint를 구분하고 PS와 NS를 같은 분모로 제시한다. B가 edit를 약하게 만든 것만으로 NS가 오른 경우를 개선으로 취급하지 않는다. Norm share는 요청과 실현을 별도로 보고한다.

## 12. 검증과 자료

`math/`의 작은 CPU FP64 모델에서 ridge 최적값·gradient, history 비용, 공동 δ의 미분, 단일 층으로의 환원과 잘못된 absolute overwrite를 확인한다. 이는 실제 Llama·FP32·Adam trajectory·GPU 속도·PS/NS를 검증하는 것이 아니다.

이번 CPU 검산 결과는 다음과 같다.

- Ridge primal/reduced value의 최대 차이 5.55e-17, analytic gradient 차이 1.35e-16.
- 공동 목적의 방향 유한차분: epsilon=1e-5에서 A/B 절대오차 2.47e-11 / 3.37e-12.
- L4–L8 mixed derivative의 유한차분 오차 2.01e-14; 각 단일 활성 층의 native 형태 loss/gradient 환원 오차 0.
- 보호 행렬 증가 예제에서 write energy 단독은 감소하지만 미실현분을 포함한 V는 증가함을 확인.
- 입력 검사에서 기존 첫2k의 순서·target·rewrite12k행·KL2k행·native key 가중치 일치. Compute-r1에서 CPU tokenizer 계수와 subject-prefix identity를 추가 확인하며, 실제 모델/GPU는 실행하지 않았다.

이 디렉터리에서 `python3 inputs/check_exact_native_inputs.py`와 `/mnt/raid5/janghj/EasyEdit/.venv/bin/python math/validate_native_joint.py`로 재현한다. Source SHA는 `math/math-check.json`에 기록한다. TeX는 구조 검사만 했으며 컴파일러가 없어 PDF는 생성하지 않았다.

TeX pipeline은 `docs/methods/jlz-native-joint-v4.tex`다. Machine-readable contract는 `contract.json`이다. V3의 CPU 결과를 v4 결과로 재사용하지 않는다.

1차 자료: [BLUE](https://papers.neurips.cc/paper_files/paper/2025/file/70d4ef44dc973586cfa3ea92b4868b72-Paper-Conference.pdf), [CAKE](https://aclanthology.org/2026.acl-long.918.pdf), [FE](https://arxiv.org/html/2605.00358v2), [HiEdit](https://aclanthology.org/2026.acl-long.1855.pdf). Native loss의 근거는 로컬 vendor `easyeditor/models/memit/compute_z.py`의 δ 가산, NLL/KL, norm, clamp다.
