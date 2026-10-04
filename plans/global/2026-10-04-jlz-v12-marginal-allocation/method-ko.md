# JLZ v12 공유 예산 공동 local z 방법

2026년 10월 4일. 첨부 v12를 구현 가능한 계약으로 구체화한 method 명세다. 모든 후보 쓰기 층의 subject 변화를 공동 계획하고, 상대 변화 예산 하나를 공유한 뒤, 층별 목표를 현재 모델에서 순차적으로 추적해 쓴다. 이번 문서는 method와 수치 참조 구현을 다룬다. GPU 실행 지시나 기존 실험의 취소·재시작 명령은 포함하지 않는다.

## 1 목적과 claim

MEMIT-H의 native z 목표는 한 상층에서 정의되며, 순차 writer는 그 목표에 대한 잔차를 남은 쓰기 층 수로 나눈다. 이 방법은 각 쓰기 층에서 직접 목표를 정의하고, 모든 층을 함께 최적화해 편집 요구를 배분한다. 고정된 층 부분집합이나 균등 배분을 먼저 지정하지 않는다.

v12의 배분 기준은 **공유된 상대 변화 예산 안에서의 편집 목적 개선**이다. writer의 보존 에너지나 history 용량을 계획 비용으로 직접 최소화하는 방법은 아니다. 배치 entry 모델 상태가 바뀌면 목적과 gradient가 달라져 계획이 달라질 수 있지만, locality 개선과 lifelong 안정성은 평가로 확인해야 한다.

다음 두 주장으로 범위를 제한한다.

1. 모든 쓰기 층의 subject local 목표를 한 계산 경로에서 공동 생성한다.
2. 공유 예산과 층간 gradient 크기를 반영하는 optimizer로 요청별 계획을 적응시킨다.

KKT는 이상적인 정지점의 조건과 진단 기준이다. 아래 유한 횟수 EfficiencyAdam은 그 조건에 도달한다고 보장하지 않는다. 한 층 집중, 여러 층 사용, 요청 간 같은 배분은 모두 허용되는 결과다. 활성 집합의 다양성이나 locality 개선을 hard gate로 두지 않는다.

## 2 기호와 native adapter

배치 entry 모델을 \(W_t\), 요청을 \(r\), 순서가 있는 전체 쓰기 층 집합을 \(\mathcal L\), 층 수를 \(m\), block 출력 차원을 \(d_l\)로 둔다. \(\ell_\star\)는 native z anchor 층이다. NLL readout 층과 writer module은 별도 항목이며, anchor와 같다고 가정하지 않는다.

\[
h^0_{lr}=\operatorname{BlockOut}_{l,s_r}(W_t;\text{canonical native row}),\qquad
a_{lr}=\|h^0_{lr}\|_2,\qquad a_r^\star=a_{\ell_\star r}.
\]

이 anchor들은 계획 동안 고정한다. \(a_{lr},a_r^\star\)는 양의 유한값이어야 하며, 실패 시 임의 epsilon을 더해 계속하지 않는다. anchor 층이 쓰기 후보 집합 밖인 adapter도 허용하며 그 entry 출력은 별도로 측정한다.

anchor norm은 native activation과 norm reduction dtype으로 entry에서 한 번 계산한다. 현재 profile에서는 FP32이며, 이후 geometry가 FP64라는 이유로 anchor norm을 FP64로 다시 계산하지 않는다. 저장·전달 과정에서 값은 유지한다.

adapter는 다음을 제공한다.

- native rewrite·KL 문장, target tokenization, lookup 위치, attention/position/target mask, context 및 target-token reduction.
- full block 출력의 주입 위치, NLL readout, KL 측정 위치, canonical row, native z anchor, 쓰기 층과 weight orientation.
- native key 추출과 nested context 평균, \(C_{0,l}\), history \(H_l\), ridge 계수, dtype과 cast/add 순서.
- 요청 식별자와 평가 문항. 평가 문장은 계획 loss나 후보 선택에 들어가지 않는다.

모델·benchmark·batch 크기는 위 adapter와 profile로 결정한다. \(B\), \(m\), \(d_l\), context 수와 target 길이를 method 내부 상수로 두지 않는다. 층마다 \(d_l\)이 달라도 목적·사영은 정의된다. RMS 기반 optimizer의 층간 보폭을 이 경우 동일한 norm 비율이라고 주장하지 않는다.

native writer의 출력 변화가 주입하는 block 출력 공간에 가법적으로 대응하는지도 adapter 검증 대상이다. 두 공간의 차원만 같다는 이유로 이를 가정하지 않는다. write module 뒤의 정규화 등이 이 대응을 바꾸는 architecture는 검증된 변환이 없으면 `UNSUPPORTED_ADAPTER`로 처리한다.

## 3 공동 목표 생성

### 3.1 변수와 forward

요청별 변수는 \(u_{lr}\in\mathbb R^{d_l}\), 실제 주입량은 \(\delta_{lr}=a_{lr}u_{lr}\)다. 초기값은 모든 층에서 0이다.

\[
h^{v,+}_{lrc}[s_{rc}]=h^{v,-}_{lrc}[s_{rc}]+a_{lr}u_{lr}.
\]

같은 요청의 native rewrite·KL row에는 같은 \(\delta_{lr}\)를 해당 subject lookup에 더한다. full block 출력에서 주입하며, canonical target 저장 역시 이 출력 공간을 사용한다. 아래층 주입이 위층 hidden과 손실에 미치는 gradient를 끊지 않는다. 반면 모델 weight, entry anchor, KL teacher에는 gradient를 적용하지 않는다.

모든 후보 층의 변수와 gradient 경로는 처음부터 끝까지 유지한다. 사영으로 0이 된 층도 다음 갱신에서 다시 예산을 받을 수 있다. 0인 층의 moment 삭제, 영구 pruning, 특정 층 선제 제외는 하지 않는다.

### 3.2 목적과 공유 예산

\[
F_r(u)=\operatorname{NLL}^{native}_r(u)
  +\lambda_K\operatorname{KL}(p^v_{K,r}(u)\Vert p^0_{K,r}),
\qquad \beta_r=\lambda_n/a_r^\star,
\]
\[
J_r(u)=F_r(u)+\beta_r b_r(u),\qquad
b_r(u)=\sum_{l\in\mathcal L}\|u_{lr}\|_2\le c.
\]

KL teacher는 배치 entry의 native KL 분포이며 full vocabulary를 쓴다. 방향은 current‖entry다. NLL·KL의 native token/context reduction은 유지한다. NLL context 가중치와 key 평균 가중치를 서로 바꿔 쓰지 않는다.

decay는 상대 변화 1단위에 층 공통 가격 \(\beta_r\)를 준다. 배분 비용, 별도의 용량 계수, reference loss, history replay, pulse, 추가 locality loss를 넣지 않는다. 기존 native \(C_0\) 통계는 writer에 유지되며, 새 reference 목적을 추가한다는 뜻은 아니다.

\(b_r=c\) 위에서는 decay가 \(\beta_r c\)로 일정하다. 따라서 경계 위의 층간 선호를 이 항 자체가 정하지 않는다. 경계에서 충분히 정확한 해를 구했다면 편집 목적의 상대 단위 한계 개선이 배분을 정한다. 실제 24회 이하 갱신에서는 optimizer의 방향과 이력도 결과에 영향을 준다.

이 예산이 제한하는 것은 **계획된 상대 주입량의 합**이다. weight 변화, write 잔차, 최종 hidden 변화, logits 변화 또는 locality 손상에 동일한 상한을 보장하지 않는다. 서로 다른 층의 변화는 중간 계산을 거쳐 전달되므로 상대 norm의 합은 명시적으로 선택한 계획 단위다.

### 3.3 EfficiencyAdam

첨부 optimizer의 형태를 유지한다. 요청별 moment와 step counter를 독립적으로 관리하며, 다음 gradient를 사용한다.

\[
g^F_{lr}=\nabla_{u_{lr}}F_r,\quad
e_{lr}=\begin{cases}u_{lr}/\|u_{lr}\|,&\|u_{lr}\|>0,\\0,&u_{lr}=0,\end{cases}
\quad g_{lr}=g^F_{lr}+\beta_r e_{lr}.
\]

이는 norm의 0점 subgradient를 0으로 고른 \(\nabla J_r\)다. model backward는 \(F_r\)에 대해 한 번만 수행하고 norm gradient는 분석식으로 더한다. norm을 loss graph에서 다시 backward해 이중으로 더하지 않는다. zero subgradient와 예산 사영의 결합도 정확한 nonsmooth solver라고 주장하지 않는다.

갱신 번호 \(k\ge1\)에 대해 좌표별 moment \(M,V\)와 층별 scalar moment \(s\)를 둔다.

\[
M_{lr}^{k}=b_1M_{lr}^{k-1}+(1-b_1)g_{lr},\quad
V_{lr}^{k}=b_2V_{lr}^{k-1}+(1-b_2)g_{lr}^{\odot2},
\]
\[
s_{lr}^{k}=b_2s_{lr}^{k-1}+(1-b_2)\operatorname{mean}_j(g_{lr,j}^2),
\quad\gamma_{lr}^k=\sqrt{\widehat s_{lr}^k/\operatorname{mean}_{q\in\mathcal L}\widehat s_{qr}^k}.
\]

hat은 해당 요청의 실제 갱신 횟수에 따른 bias correction이다. 평균은 모든 후보 층을 포함하며 현재 비제로 층만으로 재정규화하지 않는다. 모든 \(\widehat s\)가 0이면 \(\gamma=1\)로 정의한다.

\[
\eta_r=\eta_{native}/a_r^\star,\qquad
\epsilon_r=a_r^\star\epsilon_{native},\qquad
\widetilde u_{lr}=u_{lr}-\eta_r\gamma_{lr}
 \frac{\widehat M_{lr}}{\sqrt{\widehat V_{lr}}+\epsilon_r}.
\]

epsilon의 좌표 변환은 첨부 참조 코드에서 교정한 부분이다. 단일 변수층이 native anchor 층인 제한된 경우 \(\delta=a^\star u\)와 \(g_u=a^\star g_\delta\)를 적용했을 때 native Adam의 delta 갱신과 수학적으로 일치한다. 실제 저정밀 구현 parity는 별도로 검사한다.

다른 층을 단독 변수로 두면 물리적 보폭은 \(a_l/a^\star\)의 영향을 받고 decay도 \(\lambda_n\|\delta_l\|/(a^\star a_l)\)다. 이를 해당 층 native compute_z나 FE와 완전히 동일하다고 부르지 않는다. 다층 문제의 최종 비제로 층이 하나인 경우도 단일 변수 문제로 환원되지는 않는다.

\(\gamma\)는 과거 total gradient의 RMS 차이를 보폭에 전달한다. 현재 task-only 한계 효율, 정확한 최적 배분, 일반적인 gradient norm 비율 보존을 보장하지 않는다. warm-up, gamma clipping, 별도 weight decay, line search, 성능 기반 learning rate 변경은 넣지 않는다.

### 3.4 정확한 예산 사영

갱신 후보를 Euclidean group norm ball에 사영한다.

\[
u^{k+1}=\arg\min_{b_r(x)\le c}\frac12\sum_l\|x_l-\widetilde u_l\|_2^2,
\quad
u_l^{k+1}=\left(1-\frac{\tau_{proj,r}}{\|\widetilde u_l\|_2}\right)_+\widetilde u_l.
\]

예산 안이면 \(\tau_{proj}=0\), 밖이면 \(\sum_l\max(0,\|\widetilde u_l\|-\tau_{proj})=c\)를 만족하는 임계값을 구한다. 0 norm block은 0으로 둔다. plan projection 이후 Adam moment는 그대로 유지한다. 사영 계산만 FP64로 수행하고 plan은 model의 FP32로 저장하며, cast 후 실제 저장값의 feasibility를 검사한다. 허용오차를 넘는 위반은 기술 실패로 남기며 숨은 반복 축소나 추가 정책을 적용하지 않는다.

\(\tau_{proj}\)는 이 사영 문제의 승수다. 다음 절의 원래 목적 KKT 승수와 혼용하지 않는다. Euclidean 사영 자체의 정확성이 EfficiencyAdam 전체의 최적성을 뜻하지 않는다.

### 3.5 요청별 종료와 terminal 목표

각 요청은 candidate 0에서 시작해 다음 순서를 따른다.

1. 현재 \(u_r\)의 native forward로 \(F_r,J_r\)를 평가한다. 같은 forward에서 모든 canonical post-injection block 출력 \(z^v_{lr}\)를 저장한다.
2. \(J_r<\varepsilon_{stop}\)이면 종료한다. candidate 0에서도 같은 규칙을 쓴다.
3. 이번이 최대 평가 횟수의 마지막 후보이면 종료한다.
4. 그 밖에는 task backward 한 번, analytic norm gradient, Adam, 예산 사영을 수행하고 다음 후보로 이동한다.

현재 profile은 최대 25회 평가와 24회 갱신이다. terminal은 **마지막으로 실제 평가한 유한 후보**다. 평가 성능이나 이전 후보 중 최소 loss로 바꾸지 않는다. 고정 횟수 종료는 허용하며 수렴으로 표기하지 않는다. 종료한 요청의 변수·moments·counter는 이후 갱신에서 동결한다.

terminal에서 backward를 추가로 수행하지 않는다. 따라서 그 후보의 gradient/KKT는 `NO_BACKWARD_TERMINAL`로 기록한다. 바로 전 후보의 진단을 terminal 값으로 재표기하지 않는다. \(z^v\)는 이미 평가 forward에서 얻었으므로 목표를 얻기 위한 추가 forward도 필요 없다.

독립성은 고정된 \(W_t\)에서의 계획에 해당한다. 실행은 여러 요청을 묶을 수 있으나 optimizer에 전달하는 gradient는 각각 \(\nabla J_r\)여야 한다. request 평균으로 줄인 gradient, 활성 요청 수에 따른 보폭 변경, 여러 요청의 moment 공유는 금지한다. context를 microbatch로 나눌 때도 요청별 native reduction을 복원한다.

## 4 순차 writer

모든 요청의 terminal \(u,\delta,z^v\)가 정해지면 계획 graph를 해제하고 native writer를 수행한다. \(\mathcal L\)의 순서대로 다음을 반복한다.

\[
K_l=\operatorname{NativeKeys}_l(W^{cur};\mathcal R_t),\quad
R_l=[z^v_{lr}-h^{cur}_{lr}]_r,\quad A_l=\lambda_C C_{0,l}+H_{t,l},
\]
\[
P_l=\operatorname{solve}(A_l+K_lK_l^\top,K_l),\qquad U_l=R_lP_l^\top.
\]

\(h^{cur}_{lr}\)는 하층 write를 적용한 실제 모델의 canonical full block 출력이며 자기 층의 현재 write 전 상태다. \(K_l\)도 같은 실제 모델에서 다시 추출한다. terminal virtual key나 배치 entry key를 대신 쓰지 않는다. canonical 출력과 key는 같은 모델 상태에서 capture해야 하며 context reduction은 native 그대로다.

adapter가 native weight orientation에 맞춰 \(U_l\)을 변환하고, native dtype/cast/add 순서로 실제 weight에 더한다. 남은 층 수 divisor, \(M^{-1}\) 증폭, 실현율로 나누는 보정은 하지 않는다. 모든 층을 쓴 후 최종 모델에서 native key를 다시 구해 각 층 history에 \(K_l^{final}(K_l^{final})^\top\)를 정확히 한 번 append한다. 성공 요청만 골라 append하거나 중복 요청을 제거하지 않는다.

수식의 \(U_l\)은 FP64 이상적 update다. 직접 실현 telemetry는 가능한 경우 실제 저장 weight 차이 \(U_l^{applied}=W_l^{after}-W_l^{before}\)를 native orientation으로 복원한 값을 사용한다. \(U_lk\)와 \(U_l^{applied}k\)를 혼용하지 않으며, 실제 모델 forward의 저정밀 연산 차이는 별도 parity 허용오차로 다룬다.

배치 writer는 요청 간 결합을 유지한다. 계획에서 zero-step으로 종료한 요청도 원래 요청 집합의 key/target column에 남는다. \(\delta_r=0\)을 다른 요청의 편집 아래에서도 실제 변화가 0이라는 뜻으로 해석하지 않는다.

### 4.1 계획과 write의 차이

\[
r_{lr}=\delta_{lr}+b^{inh}_{lr},\qquad
b^{inh}_{lr}=h^{v,-}_{lr}-h^{cur}_{lr}.
\]

상속항은 순수한 ridge 미실현분만이 아니라 virtual과 actual 경로 차이 전체다. 하층 변화의 전달, 모든 token에 작용하는 write, context 차이와 다른 요청의 영향이 섞일 수 있다. 따라서 plan에서 0인 층도 상속항 때문에 write한다.

각 층은 ridge 자기 key에서 일부만 실현할 수 있다. 다만 다음 층은 변경된 실제 상태를 다시 측정하므로 그 결과를 자신의 target residual에 반영한다. 이를 **목표 추적 feedback**으로 정의한다. 계획 배분을 정확히 유지하는 실행이나 실현 축소를 전혀 보상하지 않는 실행으로 부르지 않는다.

모든 층의 virtual 목표는 하나의 forward에서 생성되지만 실제 모델이 모두 정확히 실현한다는 보장은 없다. canonical target을 추적해도 다른 rewrite context와 paraphrase의 정합성은 별도 측정 대상이다.

## 5 KKT와 배분 진단

원래 제약 목적의 정지점은 \(\mu_r\ge0\), \(b_r\le c\), \(\mu_r(b_r-c)=0\)과 함께 다음을 만족한다.

\[
u_{lr}\ne0:\quad g^F_{lr}+(\beta_r+\mu_r)e_{lr}=0,
\qquad
u_{lr}=0:\quad\|g^F_{lr}\|\le\beta_r+\mu_r.
\]

활성 층에서는 크기와 방향을 모두 검사한다. norm만 같아도 위 식을 만족하지 않을 수 있다. \(g_J\) 대신 task gradient \(g_F\)를 사용해 zero block의 nonsmooth 문제를 명시적으로 처리한다.

진단 시 예산 내부면 \(\widehat\mu=0\), 경계이며 활성 층이 있으면 \(\widehat\mu=\max(0,\operatorname{mean}_{l\in A}[-\langle g_l^F,e_l\rangle-\beta])\)로 추정한다. 이는 진단용 후보 승수이며 optimizer를 수정하는 계수가 아니다. 모든 active/inactive 잔차, primal violation, complementarity를 따로 남긴다. 반경 0의 수치 검사에서는 \(\widehat\mu=\max(0,\max_l\|g_l^F\|-\beta)\)를 쓴다.

저정밀 진단의 활성 tolerance와 예산 경계 tolerance는 schema에 고정하고 raw norm도 함께 저장한다. 진단상 작은 값 분류가 실제 변수를 0으로 바꾸지 않는다. 특히 \(\tau_{proj}\)로 \(\widehat\mu\)를 대체하거나 KKT residual을 갱신·채택 gate로 쓰지 않는다.

배분 기록은 다음을 구분한다.

| 기록 | 정의와 해석 |
|---|---|
| 계획 share | \(p_{lr}=\|u_{lr}\|/b_r\). \(b_r=0\)이면 0벡터와 `NO_PLANNED_EDIT` 표시 |
| writer residual | \(r_l,\delta_l,b_l^{inh}\)의 norm과 각도. norm들은 일반적으로 가법적이지 않음 |
| 직접 실현 | 실제 저장 update에 의한 \(U_l^{applied}k_{lrc}^{cur}\), native mean-key와 canonical/context를 구분. FP64 이상값은 별도 필드 |
| 실제 실현 share | canonical \(\|U_l^{applied}k_{lr,canon}^{cur}\|/a_{lr}\)를 층간 정규화. 순차 서로 다른 위치의 국소량을 요약하며 최종 효과의 가법 분해가 아님 |
| 최종 순변화 | 최종 모델 출력 − entry 출력. 직접 실현과 상속 변화를 합쳐 실제로 측정 |
| optimizer 진단 | task/norm/total gradient norm, gamma, 사영 전후 norm, \(\tau_{proj}\), 실제 step norm, 정확한 후보 번호 |

## 6 범위와 비교 해석

보존하는 native backbone은 문장, subject lookup, NLL·KL 의미, candidate 예산, 요청별 종료 형식, ridge solve, \(C_0\), history 갱신이다. 다층 공통 decay·공유 예산·EfficiencyAdam·층별 목표 추적 residual은 명시적인 method 변경이다.

\(\mathcal L=\{\ell_\star\}\)인 제한된 문제에서 native z의 loss·optimizer·clamp 대응을 확인할 수 있다. 이것은 native MEMIT-H의 여러 층 residual/divisor writer 전체가 같다는 주장이 아니다. L4 고정 계획은 요청별 배분의 대조군으로 사용할 수 있으나 FE 자체와 동일한 구현으로 표시하지 않는다.

v12와 v11의 성능 차이는 budget·decay·optimizer·종료·writer가 함께 바뀐 결과다. 이를 배분 하나의 인과효과로 해석하지 않는다. 향후 배분 효과 대조는 v12의 나머지 조건을 유지한 고정 계획 또는 gamma 대조로 설계해야 하며, 이번 method 문서가 추가 실험 실행을 지시하지는 않는다.

기존 원고의 HJ 동일 에너지, v9 gradient 원인, clamp 원인 등은 이 method의 입증된 전제로 사용하지 않는다. 수치 출처와 비교 조건을 다시 결속하기 전까지 연구 동기로만 다룬다. 특히 과거 total q-gradient를 native task-only 효율로 대체 해석하지 않는다.

## 7 검증과 구현 효율

CPU reference는 예산 사영, 좌표 변환, 진단 정의, state 규칙을 검증하는 작은 수치 증거다. 실제 model 주입·native loss·writer의 production 검증을 대신하지 않는다. EfficiencyAdam의 non-KKT 반례도 회귀 검사에 포함해 제한을 숨기지 않는다.

production 적용 전 필요한 기술 검증은 다음과 같다.

- 같은 후보에서 native single-layer loss·gradient·주입 위치·clamp parity. tiny gradient와 anchor가 1이 아닌 좌표 변환 포함.
- 요청별 독립 실행과 묶음 실행의 candidate별 loss, 변수, moments, 종료 번호 일치. request/context microbatch 크기 변경과 마지막 작은 batch 포함.
- actual lower write 후 상층 key/hidden 재측정, terminal target의 post-injection 시점, ridge input/cast/history parity, write 실패 rollback.
- 계획 feasibility와 모든 eligible 층의 재진입, 실제 실행된 후보·backward·optimizer 횟수 기록.

편집 정확도와 보존은 strict ACC, token micro ACC, prompt macro ACC와 RS/PS/NS 선호율을 함께 보고한다. teacher-forced strict ACC를 자유 생성 exact match라고 부르지 않는다. 현재 batch 직후와 과거 cohort retention, 누적 평가에 동일 정의와 분모를 적용한다. 모델별 baseline과 비교할 때 문항·순서·runtime·평가기를 결속한다.

계획에서 writer geometry를 매 후보 재계산하지 않는다. native prompts는 한 번 토큰화하고, entry anchor/KL teacher와 첫 쓰기 층 이전의 고정 prefix를 안전한 경우 재사용한다. 필요한 loss 위치의 full-vocabulary head, microbatch별 기존 오른쪽 padding 제거, terminal target의 평가 forward 내 capture를 사용한다. 의미를 바꾸는 context 축소·어휘 근사·평가 축소·candidate 축소는 하지 않는다. 정확한 cache 범위와 native adapter parity는 [구현 계약](implementation-ko.md)에 따른다.

## 8 현재 profile과 산출물

현재 예시 profile은 Llama 3 8B Instruct와 고정 CounterFact이며, 쓰기 층 L4–L8, native anchor L8, NLL readout L31이다. \(\lambda_K=0.0625\), \(\lambda_n=0.5\), \(c=0.75\), native learning rate 0.1, Adam betas (0.9,0.999), native epsilon \(10^{-8}\), ridge 계수 15000, stop threshold 0.05를 사용한다. model/plan은 FP32, geometry와 projection 계산은 FP64, history는 native CPU FP32다. 이 수치는 method 상수가 아닌 profile이다.

이전 사용자 범위인 2000 edits와 BS100×20은 후속 실험 설계의 현재 workload로 보존한다. method는 임의의 유효 batch 크기와 다른 model/benchmark adapter에 적용되도록 정의했다. 실행 arm·자원·제출은 별도 실험 계약으로 정한다.

구현자가 참조할 파일은 [기계 판독 계약](contract.json), [구현 계약](implementation-ko.md), [telemetry schema](telemetry-schema.json), [CPU reference](reference/README-ko.md), [TeX](../../../docs/methods/jlz-v12-shared-budget.tex)다.
