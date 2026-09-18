# Single-layer direct editing과 cumulative-risk 방향의 진단실험 설계

작성: 2026-09-10 KST. 상태: 연구 설계 초안. 이번 작업에서 GPU 실험은 실행하지 않았다.

사용자 요구: 진단실험을 자세히 설계하되 strict gate와 gate 수를 줄인다. 따라서 이 문서는 성능 통과에 따라 다음 단계를 잠그는 계약이 아니다. 정해진 세 실험 묶음을 실행하고, 성능·방향·비용의 연속적인 결과를 함께 해석한다. 이전 proposal의 fixed-z closure, per-request monotonicity, hard safety certificate를 이 진단에 승계하지 않는다.

## 1. 연구 질문과 이번 실험의 범위

이번에 구분할 질문은 세 가지다.

1. Native target/write를 실제 output optimization으로 바꾸면 어떤 strength–locality 범위를 얻는가?
2. 같은 모델에서 누적 변형을 줄이는 방향이 local update 축소나 임의 방향보다 locality에 유리한가?
3. 그 방향을 매번 현재 모델에서 갱신하는 것이 고정 방향이나 추가 optimization만 수행하는 것보다 유용한가?

세 묶음을 각각 A, B, C로 부른다. A의 결과가 나쁘더라도 B는 실행한다. B의 결과가 약하더라도 예정된 소형 C는 실행한다. 이후 한 번의 종합 해석으로 다음 연구의 초점을 정한다.

이번 결과만으로 CBF의 필요성, continuous ODE의 우월성, 전체 10k sequential 안정성을 주장하지 않는다. 먼저 실제 출력 최적화와 위험 방향의 정보 가치를 확인한다.

## 2. 공통 출발점: 기존 L4 lifelong의 세 checkpoint

주 모델은 Llama3-8B-Instruct, 수정 parameter는 `model.layers.4.mlp.down_proj.weight` 하나다. 기존 AlphaEdit_BLUE_L4_ONLY의 FP32 상태를 사용한다.

| 지점 | 복원 상태 | 다음 편집 | 원래 sample의 0-based ordinal | 용도 |
|---|---|---|---|---|
| Early | W10, M10 | B11, 100 requests | 1000–1099 | 초기 누적 상태 |
| Middle | W50, M50 | B51, 100 requests | 5000–5099 | 중간 상태, 공통 설정의 소규모 탐색 |
| Late | W90, M90 | B91, 100 requests | 9000–9099 | 후기 누적 상태 |

각 지점 안에서는 모든 방법이 같은 W, M, 다음 batch에서 시작한다. 서로 다른 지점의 다음 batch는 다르므로 Early/Middle/Late 차이를 누적 age만의 인과효과로 해석하지 않는다.

W100은 원래 stream에 B101이 없으므로 새 편집의 기본 출발점으로 쓰지 않는다. W100의 고정 panel 및 cumulative risk 측정은 읽기 전용 보조 분석에 사용할 수 있다.

### 2.1 확인된 자료와 아직 확인하지 않은 사항

기존 보고서에는 W10/W50/W90/W100의 실제 checkpoint와 CPU reload/hash/finite 확인이 기록돼 있다. 각 checkpoint에는 L4 W `[4096,14336]` FP32와 history M `[1,14336,14336]`, contexts/RNG/seen IDs 및 base/config/source binding이 있다. 파일 크기는 각각 약 1.057 GB다.

원본 경로 형식:

`/data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/B050/W-method-state.pt`

B010과 B090도 같은 구조다. 이 경로는 원본 보유 server4의 경로이며 현재 호스트의 로컬 파일로 확인한 것이 아니다. 현재 확인 수준은 기존 보고서·manifest·CPU 감사 기록이고, 원본 서버의 현존 여부와 GPU continuation은 아직 새로 확인하지 않았다.

복원에 필요한 나머지 재료:

- 같은 base revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`의 immutable snapshot. W0는 여기서 가져온 원래 L4 weight다.
- 저장된 L4 projector P. 기존 5-layer asset의 index 0이며 기록된 L4 slice SHA는 `8c50a474f01a28afbe52f3212e79da3dc4a22f5c753a53dedc3d31ed10c2ec68`다.
- 기존 L4 Wikipedia `mom2_100000.npz` C0. checkpoint의 covariance 필드는 빈 dict이므로 checkpoint만 읽어서 C0가 복원됐다고 생각하면 안 된다.
- 기존 contexts와 sample/order. 각 target string의 tokenization도 그대로 사용한다.

기존 실행 config의 nullspace threshold는 **0.02**다. 논문 설명의 0.01을 대입해 새 projector를 만들지 않고 실제 저장 P를 사용한다. Q가 필요하면 P의 range를 orthonormal basis로 표현한다. Q 구성 전후 projector 차이는 수치 기록으로 남긴다.

B11/B51/B91의 `native-targets.pt`도 inventory에 있다. entry와 target provenance가 대응하면 native 기준점 계산에 재사용한다. Direct-B/C의 loss나 endpoint target에는 z를 넣지 않는다.

원본 checkpoint를 사용할 수 없으면 먼저 저장된 다른 인접 지점을 쓰거나 짧은 새 native chain에서 공통 entry를 만든다. 그 결과를 기존 W50/W90 continuation이라고 부르지 않으면 된다. 복원을 위해 10k 전체를 먼저 재실행하는 것을 기본 경로로 삼지 않는다.

### 2.2 비교를 위한 복원 확인은 한 묶음으로 처리

모델·L4 W·history·sample이 의도한 입력인지, 실제 forward가 유한한지, weight update가 실제 forward graph에 들어가는지를 확인한다. 작은 gradient/finite-difference 점검도 같은 묶음에서 처리한다.

여기서 새로운 scientific gate를 만들지 않는다. 작은 backend 수치 차이는 기록하며 전 실험을 중단시키지 않는다. 입력 자체가 잘못됐거나 update가 모델에 적용되지 않는 구현 오류는 고친다. Native 재계산과 옛 보고 값의 차이가 크면 원래 trajectory의 정확한 재개라는 주장을 보류하고, 복원한 공통 entry 안의 비교와 분리한다.

## 3. 데이터 사용과 평가 panel

### 3.1 Controller가 읽는 정보

기본 A/B/C는 현재 B100의 rewrite requests와 기존 prefix contexts, 기존 request-derived essence KL inputs, P/C0, past key statistic M만 사용한다. Held-out paraphrase와 neighborhood는 gradient·learning-rate 선택·step 수 결정에 쓰지 않는다.

과거 요청의 functional replay는 기본 묶음에 추가하지 않는다. B/C의 risk 방향 때문에 과거 edit가 되돌아가는 패턴이 나오면 §15의 작은 replay sensitivity로 분리해서 확인할 수 있다. 이 선택은 정보 접근을 섞지 않기 위한 것이며 과거 요청 사용을 금지하는 원칙은 아니다.

### 3.2 평가용 300-request panel

| Panel | 구성 | 주 질문 |
|---|---|---|
| Current100 | 각 지점의 다음 B100 전체 | 신규 editing, paraphrase generalization, 해당 neighborhood |
| Fixed100 | 원래 B1 전체, 세 지점에서 동일 | 같은 과거 요청의 누적 손상과 추가 손상 |
| Past100 | 해당 entry의 seen requests에서 Fixed100과 겹치지 않는 100개 | 특정 early cohort에 한정되지 않는 retention |

Past100은 남은 과거 ordinal을 네 구간으로 나누고 구간당 25개를 선택한다. 고정 salt와 case ID의 hash 순서를 사용한다. 성능이나 loss를 보고 뽑지 않는다. 고유 request 목록과 선택 규칙만 저장하면 충분하며 별도의 승인 절차를 만들지 않는다.

각 panel의 전체 평가 분모는 RS 100, PS 200, NS 1000 prompt pairs다. 전체 300 requests는 **3900 prompt pairs**, 두 candidate를 모두 평가하면 7800 candidate sequences다. 실제 context 길이와 target 길이에 따른 token 수는 따로 센다.

Trajectory의 모든 지점에 3900 pairs를 반복하지 않는다. 두 해상도를 사용한다.

| 해상도 | 구성 | 총 prompt pairs |
|---|---|---:|
| Curve panel | 세 panel RS 전부 300 + Current PS 200 + request당 고정 neighbor 2개, 600 | 1100 |
| Full panel | 세 panel의 RS/PS/neighbor 10개 전부 | 3900 |

Neighbor 2개는 같은 request에서 항상 같은 두 index를 hash로 고른다. 이는 빠른 곡선용 sample이며 full NS처럼 이름 붙이지 않는다. Full 평가가 있는 state에서는 curve rows를 재사용하고 중복 실행하지 않는다.

W0, 해당 entry, native endpoint는 full panel로 한 번씩 측정해 branch 사이에서 재사용한다. 서로 다른 entry의 Current100/Past100은 구성이 다르므로 W0 결과도 해당 목록에 대해 별도로 측정한다.

### 3.3 오래된 label과 충돌

전체 panel은 원래 분모를 유지한다. 같은 subject–relation에 다른 target이 나중에 들어온 경우 metadata로 표시하고 전체 결과와 그 stratum을 같이 제시한다. Entry에서 틀린 요청도 전체 panel에서 삭제하지 않는다.

Retention은 전체 RS/PS와 함께 entry 성공→실패, entry 실패→성공을 기록한다. Supersession을 제거한 보조 수치는 모델 출력과 무관한 규칙으로 계산하고 원래 지표를 대체하지 않는다.

Neighbor가 edit stream의 다른 요청과 정확히 겹치는 등 식별 가능한 중복도 표시한다. 완전한 semantic conflict 검출을 새로운 선행 gate로 요구하지 않는다.

### 3.4 W0 / entry / post의 기능적 변화 분해

모든 category에서 기존 정의인 `margin = true_NLL - new_NLL`을 유지한다. RS/PS에서는 양수가 새 target 선호, NS에서는 음수가 기존 target 선호다.

같은 neighborhood q에서 다음을 계산한다.

\[
I_q=m_q(W_e)-m_q(W_0),\qquad
A_q=m_q(W_{post})-m_q(W_e).
\]

I는 해당 entry까지의 변화, A는 현재 branch의 추가 변화다. NS의 경우 양수 방향 변화가 새 competing target 쪽으로 기운 것이다. 서로 다른 prompt의 평균 차이를 이 분해로 대체하지 않는다.

## 4. A: Native / Direct-B / Direct-C 비교

### 4.1 세 arm

| Arm | 실제 parameterization | 최적화 대상 |
|---|---|---|
| N | 기존 AlphaEdit_BLUE_L4_ONLY | 기존 z 계산과 native write |
| B | W = We + A Ub^T | 실제 모델 output loss |
| C | W = We + A Q^T | 실제 모델 output loss |

Native input basis는 실제 실행식에서 가져온다. 보고된 AlphaEdit orientation에서는

\[
S=P(KK^\top+M)+L2 I,\qquad B_{native}=S^{-1}PK,
\]

이며 native update는 residual R에 대해 `R B_native^T` 형태다. Ub는 B_native의 span을 QR/SVD로 직교화한 basis다. B 계산에는 z나 R을 사용할 필요가 없다. 실제 구현에서는 이미 검증된 writer map을 재사용하고 transpose 방향을 맞춘다.

Q는 기존 P의 range를 나타내는 orthonormal basis다. Ub와 Q의 실제 rank를 기록한다. C를 임의 low-rank random subspace로 대체하고 full-P C라고 부르지 않는다.

두 parameterization 모두 `||A U^T||F = ||A||F`이므로 같은 scalar-metric optimization을 적용할 수 있다. Raw B의 조건수와 coefficient Adam 효과를 support 차이와 섞지 않는다.

### 4.2 실제 forward와 공통 loss

A=0에서 출발한다. 수정 W가 모든 token 위치에 실제로 적용되는 forward에서 loss를 계산한다. Subject 위치만 바꾼 activation intervention으로 학습하지 않는다. Gradient routing/output projector 같은 별도 개선은 기본 arm에 추가하지 않는다.

현재 edit loss는 request별 target-token 평균 NLL을 prefix contexts에 대해 평균한 뒤 100 requests를 동일 가중 평균한다. 긴 target이나 context 수가 많은 request가 의도치 않게 더 큰 가중치를 갖지 않게 한다.

\[
L_E(W)=\frac1{100}\sum_i\frac1{|C_i|}\sum_{c\in C_i}
\left[-\frac1{|o_i^*|}\log p_W(o_i^*\mid c\oplus x_i)\right].
\]

기존 native quadratic의 M 대 L2 비율을 유지한다.

\[
J_N(\Delta)=\operatorname{tr}\{\Delta(M+L2 I)\Delta^\top\},
\quad
\widehat J_N(\Delta)=J_N(\Delta)/J_N(\Delta_N).
\]

여기서 Delta_N=W_N-We는 해당 지점의 native update다. Direct 공통 objective의 출발 설정은

\[
L_{direct}=L_E+0.1\widehat J_N(W-We)+0.0625 L_{essence}.
\]

0.1은 이번 진단의 제안값이지 기존 실험으로 검증된 최적값이 아니다. M과 L2 항을 따로 재정규화하지 않는다. 기존 config에서 L2=1, kl_factor=0.0625가 확인됐다. Essence 입력·KL 방향·token 위치·reduction은 복원한 native source의 recipe를 그대로 사용하되, teacher는 공통 batch-entry We, student는 실제 수정된 W다. 위 식의 L_essence는 계수를 곱하기 전 값이며 0.0625를 중복 적용하지 않는다.

이는 z-space loss와 완전히 같은 objective라는 뜻은 아니다. 동일한 입력 정보와 preservation 구성요소를 사용해 B/C를 비교한다는 뜻이다. 기존 activation clamp와 z weight decay를 임의의 weight clamp로 옮기지 않는다. 그 차이는 direct/native formulation 차이에 포함된다.

J_N(Delta_N)이 수치적으로 0인 특수 경우에는 큰 epsilon으로 의미를 바꾸지 않는다. 정규화 불능을 기록하고 해당 지점에서는 고정 physical scaling으로 바꾼 별도 표기를 사용한다. 이런 경우를 일반적 실행 중단 조건으로 두지 않는다.

### 4.3 Optimizer와 작은 tuning 범위

- Optimizer: momentum SGD, momentum 0.9, decoupled weight decay 없음.
- 첫 momentum buffer는 첫 gradient로 초기화한다. 이후 `v = 0.9 v + g`를 사용한다. 이 선택과 실제 update norm을 기록한다.
- 32 full-batch optimization steps. 한 step은 Current100와 모든 공통 context를 한 번씩 사용한 누적 gradient update다.
- 시작 request microbatch=2, context microbatch는 메모리에 맞춰 고정한다. Gradient accumulation으로 full-batch objective를 유지한다.
- B/C의 microbatch와 request 순서를 동일하게 사용한다. 메모리 때문에 바꾸면 paired arm도 같은 logical batching을 사용한다.
- Snapshot indices: 0,1,2,4,8,16,24,32. 모든 step에 train loss와 각 penalty를 기록한다.
- Curve panel은 4,8,16,24,32에서 평가한다. Full panel은 32에서 평가한다.

Learning rate는 native update를 공통 길이 단위로 사용한다.

\[
\eta_U(\alpha)=\alpha\|\Delta_N\|_F/\|\nabla_A L_{direct}(0)\|_F,
\qquad \alpha\in\{0.02,0.06,0.2\}.
\]

첫 실제 SGD step의 Frobenius norm이 약 alpha||Delta_N||가 되게 한다. 이후 gradient를 매번 normalize하지 않는다. Near optimum에서도 일정한 크기로 계속 움직이는 별도 효과를 피하기 위함이다.

Middle W50/B51에서 B와 C 각각 세 alpha를 32 steps 실행한다. N 1개+B 3개+C 3개=7 writers다. Support별로 마지막 네 step의 평균 공통 train objective가 가장 낮은 alpha를 선택한다. Finite endpoint가 남지 않은 arm은 기술 실패로 그대로 보고하며, 남은 finite 후보 중 선택한다. 후보가 모두 불안정하면 그 사실을 보고하며 임의 성공 설정으로 바꾸지 않는다.

선택에는 PS/NS를 사용하지 않는다. 세 후보의 loss 및 curve도 모두 남긴다. Alpha 선택은 W10/W90의 성능을 기다리는 gate가 아니라 작은 learning-rate 교정이다. Early와 Late는 각 N/B/C 3 writers를 실행한다. A 전체는 **13 writers: native 3 + direct 10**이다.

이 normalization과 learning-rate calibration은 native Delta를 참조한다. 따라서 이번 비교는 **native-assisted diagnostic**이며 최종 독립 editor의 closed-form-free runtime을 측정한 결과가 아니다. Direct forward와 loss에는 z가 없다는 점과 구분한다.

### 4.4 Native scalar curve

각 지점에서 `W(alpha)=We+alpha Delta_N`, alpha={0.25,0.5,0.75,1,1.25}를 forward-only 평가한다. Alpha=1의 full 결과는 N에서 재사용하고 나머지는 curve panel로 평가한다.

이 비교는 단순 편집량 변화의 기준선이다. Alpha<1로 NS가 좋아지는 것과 risk-directed update의 효과를 비교할 수 있다. 이것을 native 방식의 모든 hyperparameter를 탐색한 최적 frontier라고 부르지는 않는다.

### 4.5 A에서 읽는 결과

- N→B: target/write 분리를 실제 output optimization으로 바꾸는 효과. 동시에 loss formulation도 바뀌므로 순수한 solver-only 비교는 아니다.
- B→C: 공통 loss·physical optimization 아래 support를 넓히는 효과.
- Direct trajectory vs native scaling: 추가 locality가 단순 under-editing 범위와 겹치는지.
- Current train NLL은 낮은데 PS가 낮으면 prompt fitting 문제, Past RS가 낮으면 retention 문제로 구분한다.

어떤 NLL/RS/PS를 반드시 넘어야 한다는 성능 gate는 두지 않는다. 모든 finite endpoint와 curve를 보고한다.

## 5. B: 같은 native endpoint에서 위험 방향에 개입

### 5.1 공통 상태와 reference

세 지점 각각의 W*=W_N에서 모든 후보를 독립적으로 출발시킨다. 앞 후보의 결과에서 다음 후보를 시작하지 않는다. 이 묶음은 native write 이후 추가 방향만 바꾸므로 target construction과 initial endpoint가 공통이다.

허용 공간은 Q로 고정한다. A의 B/C와 전 조합을 만들지 않는다.

\[
D_G=W^*-W_0,\qquad D_L=W^*-We=\Delta_N.
\]

Local reference는 batch-entry We다. Repair 시작점 W_N으로 reference를 reset하지 않는다. Entry에서 local gradient=0인 자명한 비교도 피한다.

Frobenius risk의 global gradient는 과거 누적분+현재 update, local gradient는 현재 update다. 따라서 global/local 비교가 무엇을 바꾸는지 명확하다.

### 5.2 측정할 risk

\[
R_F(W;W_{ref})=\tfrac12\|W-W_{ref}\|_F^2,
\]
\[
R_{op}(W;W_0)=\tfrac12\|W-W_0\|_2^2,
\]
\[
R_C(W;W_0)=\tfrac12\operatorname{tr}\{(W-W_0)C_0(W-W_0)^\top\}.
\]

주 인과 비교는 global/local Frobenius다. Operator/covariance 방향은 저비용의 나란한 probe로 넣고, 이것들을 바로 controller의 확정 metric으로 승격하지 않는다.

C0의 원래 normalization과 scale을 확인해 raw 값과 per-unit 설명을 함께 적는다. 성능을 보고 covariance의 eigenvalue weight를 재설정하지 않는다. P 안에서 covariance gradient가 거의 0이면 그대로 관측 결과다.

Spectral norm은 같은 설정의 반복 추정으로 측정한다. 예: 두 fixed initialization의 30-step power iteration, 큰 Rayleigh 값과 각 residual을 저장. 이 값은 upper bound가 아니다. 필요하면 대표 state에만 block power/SVD를 추가한다. Top singular direction이 불안정하거나 중복이면 단일 방향의 불확실성으로 표시하며 전체 실험을 중단하지 않는다.

### 5.3 요청별 hard equality 대신 soft gradient filtering

Current100를 case ID hash로 정렬해 10개 group, 각 10 requests로 나눈다. 각 group에서 A와 같은 context-averaged NLL의 coefficient gradient를 계산한다. Group membership은 세 방법에서 같고 성능에 따라 바꾸지 않는다.

각 gradient row의 norm을 정규화하고 1/sqrt(10)을 곱해 J를 만든다. Near-zero row는 noise를 크게 증폭하지 않고 0 row로 두며 그 수를 기록한다.

\[
S_\gamma(g)=g-J^\top(JJ^\top+\gamma I)^{-1}Jg,
\qquad \gamma=0.1.
\]

벡터화된 coefficient 좌표에서 계산하며 gradient matrix와 내적만 저장해도 된다. Gamma=0.1은 제안된 공통 감쇠 설정이고 결과를 보고 checkpoint별로 바꾸지 않는다.

이는 edit gradient와 겹치는 성분을 부드럽게 줄인다. Jd=0을 요구하지 않으며 개별 요청의 NLL 유지도 보장하지 않는다. Projection 후 unit norm으로 맞추면 leakage가 다시 커질 수 있으므로 `||Jd||`, row별 Jd, filtering 전후 norm을 반드시 같이 남긴다.

기본 J에는 과거 request loss gradient를 넣지 않는다. 추가 historical replay 없이 방향 정보를 시험하는 조건이다. Past retention은 별도로 평가하므로 global 감소가 과거 edit를 단순히 지우는지 확인할 수 있다. 이 조건의 실패를 모든 possible history-aware correction의 실패로 일반화하지 않는다.

### 5.4 방향 arm과 amplitude

Risk gradient는 먼저 Q coefficient 공간으로 pull back하고 같은 S_gamma를 적용한다. 그 뒤 실제 weight Frobenius norm이 1이 되게 정규화한다.

| Arm | 추가 방향 | 무엇과 비교하는가 |
|---|---|---|
| B0 | 0, native endpoint 그대로 | 공통 기준점 |
| GF− | −S(grad global Frobenius) | 누적 변형 감소 |
| GF+ | GF−의 정확한 반대 | risk 방향의 부호 효과 |
| LF− | −S(grad batch-local Frobenius) | 현재 update 축소와 비교 |
| Random1 | S(random coefficient), seed1 | 임의 방향 대조 |
| Random2 | S(random coefficient), seed2 | 임의 방향의 변동 예시 |
| OP− | −S(grad global operator risk) | spectral 방향이 추가 정보를 주는지 |
| COV− | −S(grad global covariance risk) | 기존 C0 방향의 정보 가치 |

Random coefficient는 Q 좌표에서 isotropic Gaussian으로 만들고 같은 filtering과 norm matching을 적용한다. 두 seed로 random 방향의 일반 분포를 추정했다고 주장하지 않는다. Random이 높은 차원에서 자연스럽게 J에 직교할 수 있으므로 실제 leakage를 비교한다.

\[
W_{trial}=W_N+\epsilon\widehat d Q^\top,
\qquad \epsilon/\|\Delta_N\|_F\in\{0.03,0.1,0.3\}.
\]

Amplitude 세 개는 모두 제안값이다. 첫 것은 local response, 마지막 것은 finite-step 비선형성이 보이는지 확인하는 용도다. 작은 진단이라고 해서 모든 step에서 loss가 같아야 한다고 요구하지 않는다.

방향 norm이 너무 작거나 0이면 강제로 증폭해 norm을 맞추지 않는다. 계산된 raw norm과 실제 적용 norm을 남기고, undefined direction은 별도 표시한다. 다른 arm과 checkpoint는 그대로 진행한다.

기본 수는 7 directions × 3 amplitudes × 3 checkpoints = **63 forward-only trial endpoints**다. B0는 A의 N 결과를 재사용한다. 0 gradient 때문에 실제 비영 update 수는 적을 수 있다.

63개 전체에 curve panel을 평가한다. 가운데 amplitude=0.1의 7×3=21 endpoints에는 full panel을 평가한다. 가장 NS가 좋아 보이는 amplitude를 골라 full 평가하는 방식은 쓰지 않는다. 중복 curve rows는 재사용한다.

### 5.5 Risk 계산과 결과 해석

Frobenius risk에는 다음 finite-change 식이 정확하다.

\[
R_F(W+E;W_{ref})-R_F(W;W_{ref})
=\langle W-W_{ref},E\rangle+\tfrac12\|E\|_F^2.
\]

Risk 방향은 감소 방향이더라도 큰 amplitude에서 실제 risk가 증가할 수 있다. 이를 폐기하지 않고 일차 항과 이차 항으로 분해한다. Gradient 방향과 finite trial의 실제 risk 감소를 구분한다.

GF−/GF+ pair에서는 NLL·margin·risk에 대해 중심 차분과 curvature를 기록한다. RS/NS는 이산 지표이므로 중심 차분의 주 해석은 NLL/margin으로 한다.

주 비교는 다음이다.

1. GF− vs GF+: 감소 방향이 증가 방향보다 실제 locality에 유리한가?
2. GF− vs Random1/2: norm이 같은 임의 수정 이상의 방향 정보가 있는가?
3. GF− vs LF− 및 native scaling: 과거 누적 상태가 추가로 유용한가?
4. OP−/COV− vs GF−: 더 복잡한 risk가 실질적으로 다른 trade-off를 만드는가?

Global NS가 좋아져도 Past RS/PS가 내려가면 useful old edit를 되돌린 trade-off로 기록한다. 이를 성공으로 숨기지도 않고, 작은 하락 때문에 전체 실험을 실패로 닫지도 않는다.

## 6. C: 방향을 다시 계산할 가치가 있는가?

Middle W50/B51의 공통 native endpoint에서 8-step trajectory 다섯 개를 실행한다. A/B의 성과 통과를 선행 조건으로 두지 않는다. Frozen/refreshed comparison의 최소 범위다.

모든 arm의 nominal update u_k는 같은 direct objective를 실제 현재 weight에서 미분해 얻는다. Support는 Q, We/M/essence teacher는 W50 batch-entry 기준으로 고정한다. 매 arm의 momentum은 native endpoint에서 새로 초기화하고, A에서 native endpoint까지 이어진 optimizer state가 있었던 것처럼 취급하지 않는다.

Learning rate eta는 Middle A에서 선택한 Direct-C의 실제 scalar eta를 다섯 arm에 공통 재사용한다. Native endpoint에서 norm calibration을 다시 하지 않는다. Nominal momentum buffer를 v_k라 하면 u_k=-eta v_k이며, A와 같은 momentum 정의를 사용한다.

| Arm | 추가 항 | 해석 |
|---|---|---|
| C0 Continue | 없음 | native 이후 추가 실제-output optimization 효과 |
| C1 Frozen-global | 시작점의 global gradient와 J로 만든 방향 고정 | 고정된 risk correction |
| C2 Refreshed-global | 현재 global gradient와 현재 J로 방향 재계산 | 현재 상태 feedback 전체의 효과 |
| C3 Refreshed-local | batch-entry 기준 local gradient와 현재 J 재계산 | global reference의 추가 가치 |
| C4 Soft-global | 고정 lambda의 global Frobenius penalty gradient | 고정계수의 현재 penalty-gradient correction |

C1/C2/C3의 correction 길이는 각 step `c=0.1||Delta_N||F/8`로 동일하게 제안한다. 한 trajectory의 correction path budget은 0.1||Delta_N||F다. 이 제한은 최종 risk/strength 통과 조건이 아니라 probe amplitude를 맞추는 실험 설정이다.

\[
a_{k+1}=a_k+u_k+c\widehat d_k.
\]

C1은 d0를 고정하고 C2/C3은 매 step 새로 계산한다. 현재 loss gradient로 만든 nominal update는 C1에서도 갱신한다. 따라서 비교 대상은 nominal optimizer의 feedback 유무가 아니라 risk/protection 방향의 refresh다.

C4는 `u_k - eta lambda_R grad R_G`를 사용한다. 첫 상태의 penalty correction norm이 c와 같도록 `lambda_R = c / (eta ||grad_a R_G(W_N)||)`를 한 번 정하고 이후 고정한다. Gradient가 0이면 이 calibration은 정의되지 않는 것으로 기록한다. Lambda_R는 locality를 보고 선택하지 않는다. 위험 penalty를 momentum에 넣어 누적하는 다른 알고리즘과 섞이지 않도록, nominal momentum update와 별도의 현재 penalty-gradient correction으로 구현한다. 따라서 표준적인 L_direct+lambda_R R_G 전체에 momentum SGD를 적용한 baseline과는 구분한다.

C2는 g와 J를 함께 갱신하므로 이 둘 각각의 효과를 분리하지 않는다. C4와 C2는 filtering과 amplitude 정책도 다르므로 차이를 곧바로 barrier 효과로 부르지 않는다. 이 실험에는 hard barrier 자체가 없다.

Steps 0,1,2,4,8의 train/risk telemetry를 저장하고 curve panel은 2,4,8, full panel은 8에서 평가한다. Every-step J는 10 group gradients이므로 parameter-gradient 저장과 backward 비용을 따로 기록한다. 평균 edit gradient는 group gradients에서 재사용할 수 있다.

NS가 일시 하락하거나 risk가 증가했다는 이유로 rollback/backtracking하지 않는다. NaN처럼 모델을 평가할 수 없는 기술 실패만 해당 branch의 종료로 기록한다. 8-step의 finite endpoint는 성능에 관계없이 모두 보고한다.

## 7. 무엇을 항상 기록할 것인가?

### 7.1 기능 측정

- RS/PS/NS의 numerator/denominator와 strict comparison 정의.
- New NLL와 true NLL 각각의 mean, median, p90. Margin 및 entry 대비 변화.
- Teacher-forced exact: 모든 target token top-1인 비율. 자유 생성 정확도라고 부르지 않는다.
- Current request별 NLL 변화: 평균 외에 p90 악화량, 악화된 요청 수, 개선된 요청 수.
- Past/Fixed의 success loss와 recovery. 전체 분모와 entry-success 조건부 분모 모두.
- Panel별 결과. Current/Fixed/Past를 섞은 단일 NS로 결론 내리지 않는다.

Fixed decoded generation 보조 panel은 Current100의 metadata hash로 고른 20 requests의 rewrite+paraphrase 2개, 총 60 prompts다. Native, 각 지점의 선택된 B/C step32, Middle C의 step8 endpoints에서 greedy max_new_tokens=32로 평가한다. Decode 설정과 raw output을 저장한다. 이 작은 표본의 literal target-prefix/string match는 의미적 정확도와 구분하고, NS나 learning-rate 선택에 사용하지 않는다. 주 paired NLL 평가를 대체하지 않는다.

### 7.2 구조 측정

- Global D와 batch-local Delta의 Frobenius norm, operator estimate, covariance risk.
- Native action J_N, current/history/L2 contribution을 분리한 설명용 값.
- Step norm, batch-net norm, sum of step norms, cumulative W-W0 norm을 구분.
- Risk gradient norm, GF/LF cosine, risk-gradient와 edit-gradient의 cosine.
- Soft filtering 전후 norm, 최종 Jd와 request-group별 predicted leakage.
- Q/Ub rank, coefficient norm, projection consistency의 수치 크기.
- Spectral estimator 설정·초기화·residual. 검증되지 않은 upper-bound 표현 금지.

### 7.3 계산 비용

- Model load, checkpoint load, P/Q/C0/native-basis 준비 시간.
- Native z cache hit/miss, native solve 시간.
- 실제 training forward/backward sequence 수와 token 수.
- Group-Jacobian 생성 시간, risk-gradient 시간, small solve 시간.
- Curve/full evaluation 시간, 생성 평가 시간.
- 최대 GPU memory, parameter/optimizer/J storage.

기존 z cache를 사용한 실제 비용과 cold-native 비용을 분리한다. Cache-hit native와 cold direct를 같은 runtime 표에서 공정한 online speed comparison으로 해석하지 않는다.

## 8. Strength matching을 gate 대신 분석으로 사용

Native의 요청별 NLL을 반드시 맞추도록 ceiling을 두지 않는다. Mean NLL/RS가 정확히 같은 endpoint만 남기는 필터도 사용하지 않는다.

모든 관측 snapshot을 다음 companion plots에 표시한다.

1. Current train NLL ↔ Current/Fixed/Past NS 변화.
2. Current RS ↔ NS, point color는 Current PS.
3. Current PS ↔ Past RS/PS 손실.
4. Global risk 변화 ↔ neighborhood margin 변화.
5. Applied correction norm ↔ 현재·과거 edit loss 변화.

Past/Fixed PS는 full panel을 평가한 state에서만 표시한다. 나머지 trajectory의 past retention은 관측된 RS/NLL로 표시하며, 미측정 PS를 보간하거나 평가했다고 쓰지 않는다.

먼저 실제 관측된 범위가 겹치는지 본다. 공통 strength 구간이 있으면 그 안의 실제 snapshots를 나란히 비교한다. Matching 지점 선택에는 train quantities를 사용하고, NS가 높은 점을 사후 선택해 primary result로 만들지 않는다. PS와 per-request loss 분포는 matching이 얼마나 충분했는지 보여주는 보조 축이다.

Interpolation은 시각화 보조다. 관측되지 않은 중간 weight에서 NS를 측정한 것처럼 쓰지 않는다. 범위가 겹치지 않으면 그대로 `관측 범위 내 matched comparison 없음`으로 보고하며 그 arm을 삭제하지 않는다.

하나의 정답 숫자나 합성 score를 최대화하지 않는다. 특히 높은 RS만 맞춘 상태에서 PS가 크게 다르면 strength-matched라고 단정하지 않는다.

## 9. 통계와 개발/확인 경계

이 세 checkpoint는 개발용 mechanism diagnosis다. 이미 본 기존 stream과 checkpoint를 새 held-out benchmark라고 부르지 않는다.

각 checkpoint 내부의 paired delta를 우선한다. Request 단위로 그 request의 rewrite/paraphrase/neighborhood를 묶어 2000회 paired bootstrap CI를 계산한다. 같은 subject-relation 그룹이 중복되는 경우 가능한 group-cluster sensitivity도 보조로 제시한다. Prompt를 모두 독립 표본으로 취급해 표본 수를 부풀리지 않는다.

CI는 해당 panel의 sampling variability 설명이다. 단일 edited checkpoint의 bootstrap이 다른 edit order나 학습 seed에 대한 변동까지 추정하지는 않는다. P-value가 특정 수치를 넘는다고 후속 실험을 차단하지 않는다.

W50의 LR 선택은 train objective만 사용한다. 이후 이 진단의 PS/NS를 보고 method를 바꾸면 세 panel은 계속 개발용이다. 최종 주장에는 별도 request/order에서 확인한다. 과도한 사전 봉인 대신 어떤 데이터를 보고 어떤 설정을 바꿨는지 한 change log에 적는다.

## 10. 실제 실행 순서와 기본 규모

1. 원본 보유 서버에서 복원 재료를 확인하고 Middle 상태를 로드한다. 입력/forward/gradient 확인을 한 묶음으로 처리한다.
2. Middle A의 native 및 B/C LR 세 점을 실행한다. 미리 정한 train-only 기준으로 alpha를 선택한다.
3. Early/Late A를 실행한다. Middle의 성능이 좋아야만 실행하는 조건은 없다.
4. 세 native endpoints에서 B의 방향·amplitude grid를 평가한다.
5. Middle C의 다섯 8-step trajectories를 실행한다.
6. 정해진 full panel과 생성 보조 panel을 마치고 한 번 종합 해석한다.

| 계산 종류 | 기본 수 |
|---|---:|
| Native writers | 3 |
| Direct 32-step writers | 10 |
| Direct optimizer steps | 320 full-batch steps |
| Native scalar trial | 12 추가 endpoints, alpha1은 재사용 |
| B 방향 trial | 63 추가 endpoints |
| C 8-step trajectories | 5 |
| C nominal optimizer steps | 40 full-batch steps |
| C에서 J 재평가가 필요한 arm | Frozen 초기1회, refreshed global/local은 step별 |

B의 risk gradients/J는 공통 native state에서 계산해 amplitude들 사이에 재사용한다. OP/COV도 방향을 한 번 구한 뒤 amplitude만 바꾼다. 수십 번의 backward가 필요한 실험으로 잘못 구현하지 않는다.

기본 추가 training은 direct 320 + repair 40 = **360 full-batch steps**다. 이것이 360번의 단일 microbatch backward라는 뜻은 아니다. Context 수, microbatch 수, sequence 길이에 따른 actual backward와 token 수를 보고한다.

원본 서버에서 측정한 step/evaluation 시간으로 예상 잔여 비용을 계산한다. 사전 근거 없이 GPU-hours를 고정하지 않는다. 메모리 때문에 physical microbatch를 줄여도 logical full-batch를 유지하면 연구 질문은 그대로다.

## 11. 비용이 부족할 때 줄이는 순서

성능 결과를 보고 나쁜 arm만 제거하지 않는다. 자원 한계라면 사전에 다음 축소 순서를 적용하고 실제 실행 범위를 표시한다.

1. 생성 보조 panel을 Middle의 N/B/C endpoint로 축소.
2. Curve evaluation를 steps 8/16/32로 축소. Full terminal panel은 유지.
3. OP/COV 방향을 Middle에서만 실행. Global/local Frobenius와 random/sign 비교는 세 지점 유지.
4. C4 soft-penalty companion을 생략하고 continuation/frozen/refreshed-global/refreshed-local을 유지.

모델 수·전체 10k rerun·full factorial lambda sweep은 기본 범위에 없다. 줄인 결과를 원래 전체 설계를 완료한 것으로 보고하지 않는다.

## 12. 결과 패턴별 해석: 탈락표가 아니라 다음 질문

| 관찰 | 우선 해석 | 이어지는 질문 |
|---|---|---|
| B/C가 native와 비슷한 strength에서 NS 개선 | 실제 weight optimization이 유용할 가능성 | 위험 제어 없이도 충분한가? |
| B는 안정적이고 C의 PS가 낮음 | 넓은 support의 fitting/specificity 비용 가능성 | C가 꼭 필요한가, gradient 위치가 문제인가? |
| GF−가 GF+/random보다 NS에 유리하고 Past strength도 비슷함 | 누적 변형 방향에 유용한 정보가 있음 | feedback을 넣은 긴 trajectory에서도 유지되는가? |
| GF− NS 개선과 Past RS 손실이 동반 | 이전 edit를 되돌리는 trade-off | 같은 방향을 historical functional observer로 다듬을 수 있는가? |
| Risk는 감소하지만 margin/NS 변화가 약함 | structural proxy의 방향 정보가 약함 | 다른 metric이나 downstream 정보가 필요한가? |
| LF−와 GF−가 비슷함 | 이번 범위에서 global reference 추가 가치가 약함 | Gradient들이 실제로 비슷했는가, 변화 폭이 너무 작았는가? |
| C2와 C1이 비슷함 | 8 steps에서 refresh의 추가 가치가 약함 | 더 비선형적인 구간에서도 같은가? |
| C2가 C0보다 좋지만 C4와 비슷함 | 보존 regularization으로 효과 설명 가능 | Hard barrier가 실용적으로 필요한가? |
| 효과가 특정 checkpoint에서만 나타남 | 상태 또는 다음 batch 의존 가능성 | 같은 probe batch를 여러 entry에서 쓰면 어떤가? |

한 번의 음성 결과를 모든 cumulative-risk 방법의 반증으로 확대하지 않는다. 그렇다고 성능이 나올 때까지 통제 변수를 무제한 추가하지도 않는다. 관측된 실패 형태에 맞는 작은 후속 질문 하나를 고른다.

## 13. Gate 정책

Scientific pass/fail gate는 두지 않는다. 다음은 모두 결과다: RS/PS의 소폭 하락, NS 악화, risk 증가, 큰 correction leakage, 방향의 near-zero norm, alpha 간 불안정, common strength 범위 부재.

입력 오인·update 미적용·NaN/Inf·자원 오류처럼 결과를 계산하거나 해석할 수 없는 문제는 기술 문제로 수정/기록한다. 실패한 branch의 행을 없애지 않는다. 작은 수치 차이에 맞춰 gate threshold를 반복 조정하지 않는다.

초기 복원 확인 한 묶음과 종료 후 종합 해석 한 번이면 충분하다. 실험마다 별도의 성과 승인이나 promotion 절차를 추가하지 않는다.

## 14. 최종 산출물

Raw artifacts는 execution storage에 두고 compact tables와 한국어 해석 문서만 정리한다.

- `run-index`: entry, batch IDs, method, alpha, support rank, seed, state path, completion/exception.
- `trajectory`: step, train losses, risk, norms, elapsed time.
- `request-metrics`: state, panel, request/prompt identity, new/true NLL, margin, strict bits.
- `direction-probes`: direction/amplitude, actual applied norm, Jd, predicted/actual risk change.
- `paired-summary`: panel별 변화, loss/recovery, cluster uncertainty.
- `compute-summary`: cache 포함/제외, optimization/evaluation 시간, sequence/token/backward counts.
- 그림: strength–locality, past retention, risk–margin 방향 산점도, frozen/refreshed trajectories.
- `diagnostic-discussion-ko.md`: 확인된 사실, 가능한 해석, 다음에 분리할 한 질문.

## 15. 첫 결과에 따라 붙일 수 있는 작은 후속 비교

아래는 기본 완료를 막는 gate가 아니며 첫 round의 mandatory factorial에 포함하지 않는다.

### 15.1 Checkpoint age와 batch difficulty를 분리

공통 probe B91을 W10/W50/W90에서 편집한다. W90/B91은 기본 실험에 이미 있으므로 W10/B91과 W50/B91 두 상태를 추가하면 된다. N과 대표 direct arm만 사용한다. B91은 앞 entry들에서 아직 도착하지 않은 요청이지만 이것은 진단용 counterfactual branch이지 실제 online 순서를 재현한 run이 아니다.

### 15.2 History-aware direction sensitivity

Global 감소가 과거 edit를 지우는 경우, Middle에서 기존 seen requests 64개를 네 age 구간별 16개씩 고정 선택한다. Fixed100/Past100 평가 requests와 겹치지 않게 한다. Supersession이 식별된 항목은 유효 target 정책을 명시한다.

Canonical rewrite만 사용해 4 group gradients를 J에 추가한다. 기존 current block은 그대로 유지하고, 정규화한 history rows에 sqrt(0.5/4)를 곱해 append한다. 같은 gamma를 사용해 current protection의 가중치를 줄이는 별도 변경을 피한다. GF−/LF−/Random의 amplitude0.1만 다시 측정하면 처음부터 모든 arm에 replay를 추가하는 것보다 정보 효과를 분리하기 쉽다. 이 variant는 edit-stream replay를 사용한 것으로 별도 표기한다.

### 15.3 Prompt-position 또는 regularizer sensitivity

C의 training fit은 좋지만 PS/NS가 낮으면 Middle에서 subject 이전 coefficient-gradient routing 유무를 한 쌍으로 비교한다. 다른 option과 교차 탐색하지 않는다. 또는 common action coefficient 0.1을 0.03/0.3으로 바꾼 sensitivity 중 하나를 먼저 고른다. 둘 다 동시에 바꿔 rescue 원인을 흐리지 않는다.

### 15.4 짧은 실제 sequential continuation

방법 자체의 누적 효과가 필요하면 W50부터 B51–B55의 5×B100을 native, direct 기준점, cumulative correction 세 arm으로 수행한다. 각 arm은 자신의 직전 endpoint에서 이어지고 M은 commit 후 native 정책으로 한 번 append한다. 이제 cached native targets를 다른 arm에 그대로 쓰지 않는다. 새로운 native-assisted calibration이 필요하면 해당 arm의 실제 entry에서 다시 계산하고 비용에 포함한다.

Fixed panel은 W50에서 고정해 끝까지 사용한다. Current/all-new-seen RS/PS와 Fixed/Past retention을 함께 평가한다. 이 소형 continuation은 10k 안정성 증명이 아니라 one-step 이득이 다음 batch에서 사라지는지 보는 실험이다.

## 16. 근거 자료

아래 기존 결과를 실제 새 실험을 수행한 결과와 혼동하지 않는다.

- [9월9일 BLUE L4/L8 lifelong v2 factual report](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-fixed-counterfact10k-20260909/experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2/factual-report-ko.md)
- [실제 checkpoint tensor inventory](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-fixed-counterfact10k-20260909/experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2/checkpoint-tensors.csv)
- [Source/config 및 P binding](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-fixed-counterfact10k-20260909/experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2/source-config-compatibility.csv)
- [Raw member inventory](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-fixed-counterfact10k-20260909/experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2/raw-member-inventory.csv)
- [C0/P와 source asset inventory](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-fixed-counterfact10k-20260909/experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2/source-member-inventory.csv)
- [DOW-KE 원문](https://arxiv.org/html/2608.16932v1): 직접 coefficient optimization의 선행연구이며 본 B arm은 output projector/gradient routing까지 포함한 원 방법의 재현이 아니다.
- [CrispEdit 원문](https://arxiv.org/html/2602.15823v2): 직접 constrained editing 및 sequential preservation의 선행연구. 이번 진단을 마친 뒤 solver 비교를 설계할 때 참고한다.
