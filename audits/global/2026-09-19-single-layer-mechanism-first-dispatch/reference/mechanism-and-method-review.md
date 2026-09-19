# Single-layer의 구조적 이점, 조건부 손상 정리, EN 이후의 방법 판단

2026-09-19. 분석 문서다. 새로운 LLM/GPU 실험이나 실행 코드 변경은 하지 않았다. 아래 수학은 명시한 선형대수·미분 조건에서의 유도이며 그 자체의 최초성을 주장하지 않는다. [기존 capacity 정리](/mnt/raid5/janghj/layer_allocation/07_single_layer_capacity_theorems.md), [EN 감사](/mnt/raid5/janghj/layer_allocation/en_r512_g256_capacity_scope_audit_20260919.md), [실증·문헌 인벤토리](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-19-key-aware-routing-and-joint-z/empirical-literature-ko.md)를 함께 검토했다.

## 1. 판단

Single-layer의 이점은 정적인 key만이 아니다. 같은 고정 입력에서 **전체-token response의 정확한 가산성**, **같은 편집 반응을 내는 update 집합의 정확한 표현**, **고정 downstream 함수를 통한 유한 변화 평가**를 함께 사용할 수 있다.

그러나 activation 변화가 곧 손상인 것은 아니며, downstream 함수가 고정돼도 Jacobian은 endpoint에 따라 변한다. 이 차이가 이전 two-memory/EN 결과를 다시 설명할 때 중요하다.

제안된 인과 사슬은 다음처럼 수정해야 한다.

> 구별하기 어려운 허용 key 방향 **그리고** 그 방향에서 양립하기 어려운 residual 요구 → exact fitting에 필요한 큰 write → reference와 방향이 겹칠 때 activation 변화 → 그 변화가 downstream margin을 불리하게 움직일 때 판단 손실.

구별성 저하가 target 요구의 증가를 유발한다는 일반 명제는 없다. 큰 norm이 곧 NS 손상이라는 명제도 없다. 실제 AlphaEdit가 exact interpolation과 같은 역특이값 증폭을 하는지도 관측해야 한다.

## 2. 정확한 좌표계와 누적 상쇄

한 down-projection W만 편집하고 token IDs, prefix, mask, position, 다른 모든 parameter를 고정한다. 입력 x의 모든 필요한 token key를 K_x, base에서의 module output을 H_x^0라 하면

\[
H_x(W)=H_x^0+(W-W_0)K_x,
\qquad f_W(x)=F_x\big(H_x^0+(W-W_0)K_x\big).
\]

F_x에는 고정 residual 경로·mask·이후 network가 포함된다. 이것은 해당 고정 입력에서의 정확한 함수 분해이며 logits의 선형화를 뜻하지 않는다. 새로 생성한 prefix가 바뀌면 K_x와 F_x의 입력도 달라진다.

Entry 누적 변화 E=W_entry−W0, 신규 write Δ, B_x=EK_x, V_x=ΔK_x이면

\[
\|B_x+V_x\|_F^2-\|B_x\|_F^2
=2\langle B_x,V_x\rangle_F+\|V_x\|_F^2.
\]

교차항은 상쇄/증폭을 정확히 측정한다. 하지만 이것은 **누적 activation 변위**다. KL·NS·일반 능력 손상이라는 이름으로 바꾸면 안 된다. C4와 active history의 기준도 구분한다. 전자는 W0 행동, 후자는 overwrite를 반영한 유효 target/수용 상태가 기준이다.

Cold B1에서는 E=0이므로 누적 교차항이 없다. 누적 상쇄를 핵심으로 주장하는 방법은 B1만으로 검증할 수 없다. 반대로 첫 batch에도 N 손상이 있으므로 과거 edit collision은 모든 locality 손상의 필요조건이 아니다.

## 3. 같은 편집 반응의 동치류와 제거 가능한 activation 변위

허용 basis U의 열을 orthonormal이라고 하고, 보호할 전체-token key를 K_E, X=U^T K_E로 둔다. Native endpoint W_N의 반응을 그대로 유지하는 모든 추가 보정은

\[
D=B(I-XX^\dagger)U^T.
\]

교집합의 orthogonal projector는 Q_E=U(I−XX†)U^T다. 임의 두 projector의 곱을 교집합 projector라고 가정하지 않는다. 필요한 new/old teacher-forcing sequence 전체를 포함한 경우 그 입력의 downstream 출력이 실수 연산에서 동일하다. 평균 subject key만으로 같은 보장을 주장할 수 없다. Paraphrase·unseen reference는 이 보장의 대상이 아니다.

N을 Q_E의 orthonormal basis, Z=N^T K_R, B_R=(W_N−W0)K_R라고 하자. 같은 edit 반응 아래 reference activation 변위를 가장 많이 제거하는 문제는

\[
\min_C\|B_R+CZ\|_F^2.
\]

한 최소해는 C*=−B_R Z†이고, residual 및 제거 가능한 energy는

\[
B_{\rm remain}=B_R(I-Z^\dagger Z),\qquad
J_{\rm removable}=\|B_RZ^\dagger Z\|_F^2.
\]

이 값은 정확한 **activation 복원 가능량**이며 functional 회복 가능량이 아니다. 기존 two-memory-v2는 보호 bank의 mapping energy 감소와 output KL/NLL 악화를 함께 관측했다. 따라서 이 closed form은 후보·진단용이며 그대로 새 주 방법으로 채택할 근거가 없다. [기존 감사](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-review-ko.md:60).

Native가 동일 quadratic metric C≻0 아래 같은 편집 반응을 만드는 최소해라면, DK_E=0인 허용 보정에 대해

\[
\|\Delta_N+D\|_C^2=\|\Delta_N\|_C^2+\|D\|_C^2.
\]

따라서 같은 native 비용을 더 줄이는 nonzero 보정은 없다. 누적 E+Δ_N의 비용에는 E와의 교차항이 남을 수 있다. Native increment 비용과 W0 기준 누적 비용을 혼동하지 않는다. Functionally 유리한 다른 해가 norm을 더 사용할 수도 있다.

## 4. 정리: key 구별성과 target 요구의 결합

허용 update Δ=AU^T, projected key X=U^TK, 동일 entry/module 좌표의 residual R에 대해 AX=R를 요구하자. 임의 a에서 Xa≠0이면

\[
\boxed{\|\Delta\|_F\ge\frac{\|Ra\|_2}{\|Xa\|_2}.}
\]

증명: Ra=AXa와 Cauchy–Schwarz/operator norm bound를 사용하면 ||Ra||≤||A||_F||Xa||, 그리고 ||AU^T||_F=||A||_F다. □

근사 fitting ||AX−R||_F≤ε이면 분자는 [||Ra||−ε||a||]_+로 바뀐다. Xa=0, Ra≠0이면 exact write는 존재하지 않는다. 가까운 두 요청은 a=e_i−e_j를 사용해 구별성 대비 residual 차이를 측정한다.

Compatible한 exact 문제에서 X=LΣV^T의 compact SVD를 사용하면

\[
\Delta_*=RX^\dagger U^T,
\qquad
\boxed{\|\Delta_*\|_F^2=\sum_j\frac{\|Rv_j\|_2^2}{\sigma_j^2}.}
\]

조건은 R=RX†X다. 작은 σ_j만으로 비용 증폭을 말하지 않는다. 해당 방향의 target loading Rv_j가 함께 커야 한다. 이것은 표준 pseudoinverse 결과다.

과거 반응을 exact 보호하는 허용 공간의 projector Q에서 새 key k, residual r, u=Qk≠0라 두면

\[
\Delta_*={ru^T\over\|u\|^2},\quad
\|\Delta_*\|_F={\|r\|\over\|u\|}.
\]

증명: Δk=Δu=r에서 norm 하한을 얻고 위 rank-one 해가 이를 달성한다. Qk=0, r≠0이면 infeasible이다. r도 ||u||에 비례해 작아지면 norm 증폭은 없다.

**실제 writer와의 구분:** simple ridge의 해는

\[
A_\lambda=RV\operatorname{diag}\left({\sigma_j\over\sigma_j^2+\lambda}\right)L^T.
\]

고정 λ>0에서 σ→0이면 coefficient는 0으로 간다. 대신 해당 target 실현율 σ²/(σ²+λ)이 낮아진다. Native AlphaEdit의 soft-history·ridge·수치 projector를 exact-history inverse formula로 대신 설명할 수 없다. 실제 solver metric, realized residual, actual write를 함께 확인해야 한다.

## 5. 조건부 margin 손상 정리와 반례

위 rank-one 해에서 δ=||u||, v=u/δ, reference 전체 key K_j라 하면

\[
V_j=\Delta_*K_j={r(v^TK_j)\over\delta},\qquad
\|V_j\|_F={\|r\|\|v^TK_j\|\over\delta}.
\]

Reference와 v의 overlap이 0이면 큰 write에도 반응 변화는 없다.

Entry의 전체 module output H_j, 그 입력의 정확한 판단 margin φ_j(H_j)=γ_j>0를 정의하자. Pairwise target NLL margin이면 두 sequence 경로를 함께 포함한다. V_j≠0, e_j=V_j/||V_j||라 하고 선분 전체에서

\[
\langle\nabla\phi_j(H_j+sV_j),e_j\rangle_F\le-a_j<0
\quad(0\le s\le1)
\]

라고 가정하면

\[
\phi_j(H_j+V_j)\le\gamma_j-a_j{\|r\|\|v^TK_j\|\over\delta}.
\]

증명: φ_j(H_j+V_j)−φ_j(H_j)=∫_0^1〈∇φ_j(H_j+sV_j),V_j〉ds를 적분한다. □

따라서 우변이 음수이면 지정 판단이 실패한다. 미분 불가능한 max competitor margin은 적절한 절대연속성/거의 모든 점의 derivative 조건으로 처리하거나 fixed competitor pair margin을 사용한다.

중요한 미확인 가정은 **불리한 signed sensitivity a_j>0**다. Lipschitz 상한이나 한 지점의 gradient만으로 이 가정을 보증할 수 없다. 함수 F가 고정된 것은 gradient가 고정된 것과 다르다.

반례:

1. u가 작아도 r도 같이 작아지면 큰 write가 필요 없다.
2. Write가 커도 v^TK_j=0이면 reference 변화가 없다.
3. 큰 response가 downstream의 무감각 방향이거나 margin을 높이는 방향일 수 있다.
4. 최소 norm 해의 reference 피해가 모든 feasible 해의 피해는 아니다. k=e1, r=1, b=e1+e2에서 (1,0)은 b를 바꾸지만 (1,−1)은 같은 edit을 하며 b를 보존한다.
5. 여러 singular mode의 reference 영향은 교차항을 갖는다. 이를 독립적 양수 손상의 합으로 해석할 수 없다.

## 6. EN 다음 방법에 세 이점을 결합하는 방식

가장 가치 있는 검증 명제는 **native가 일으킨 손상 중, 현재와 유효한 과거 편집을 유지하면서 제거할 수 있는 functional 성분이 존재한다**는 것이다. Norm 축소나 평균 KL 축소만으로 이 명제를 대체하지 않는다.

구조를 세 부분으로 나눈다.

1. **정확한 상태:** E=W_entry−W0 및 후보 Δ의 전체-token 반응을 계산한다. 신규 write 크기만 보는 대신 누적 상태에서 어떤 reference 판단이 위험한지 확인한다.
2. **편집 효과의 동치류:** 우선 DQ_E로 동일 native response 아래 개선 가능성을 분리한다. 여기서 실패했다고 single-layer가 불가능한 것은 아니다. 최종 정답 유지보다 강한 full-token equality가 유용한 방향을 제거했는지 구분한다.
3. **실제 downstream 판단:** reference별 margin·허용 NLL 약화·active-history 유효 target을 보호 조건으로 둔다. 평균 KL은 관측량으로 유지할 수 있지만 유일한 수용 조건으로 두지 않는다.

현재 endpoint에서 reference margin의 weight gradient는 정확히

\[
G_i=A_iK_i^T,\quad A_i=\nabla_{H_i}\phi_i,
\qquad h_i=G_iQ_E.
\]

이것은 single linear site의 chain rule이다. Key Gram 부분은 캐시 가능하지만 A_i는 endpoint가 바뀌면 갱신이 필요하다. projected gradient Gram은

\[
\langle h_i,h_j\rangle_F
=\sum_{s,t}(a_{is}^Ta_{jt})(k_{is}^TQ_Ek_{jt}).
\]

이 식은 **입력 공유와 downstream 방향의 결합**을 직접 나타낸다. Raw key cosine만으로 충돌이나 보호 가능성을 판단하지 않는다.

한 가지 구체적 local solver는 다음이다. 보호 기준 b_i와 허용 metric M을 명시하고

\[
\min_D\tfrac12\|D\|_M^2
\quad\text{s.t. }D=DQ_E,\quad
\phi_i(W_N)+\langle G_i,D\rangle_F\ge b_i
\]

를 reference와 active-history 조건에 대해 공동으로 푼다. 모든 조건을 만족하는 해가 없을 수 있으며, 선형화 feasible이어도 실제 endpoint가 feasible이라는 보장은 없다. 실제 전체 reference와 Current/active-history endpoint 검증이 필요하다. Trust region/허용 오차를 사용한다면 그 값을 숨기지 않는다. 현재 key equality는 native에서 이미 잊힌 과거 판단을 복구하지 않으므로 history 기준을 W_N으로 자동 설정하면 안 된다.

이 QP는 기존 BPCW류 및 표준 constrained optimization이다. **새 method 이름을 붙이는 것만으로 EN의 실패나 novelty 문제가 해결되는 것은 아니다.** 새로 확보할 것은 어떤 유해 mode를 왜 수정하는지, equality 아래 얼마나 회복 가능한지, 독립 locality로 얼마나 전달되는지다.

Q_E 안에 유용한 방향이 거의 없다면 더 넓은 family를 구분한다. General allowed D는

\[
D=ZX^\dagger U^T+B(I-XX^\dagger)U^T,\qquad Z=ZX^\dagger X
\]

로 쓰며 DK_E=Z다. Z=0이 EN의 exact response family이고 Z≠0은 edit의 내부 반응도 바꾸는 family다. 후자는 실제 Current success/NLL와 active-history 조건을 만족하는 범위에서만 고려한다. 선형화된 edit Jacobian을 보존한 것과 finite output을 보존한 것은 다르다. 이것을 기본값으로 바로 채택하거나 공식 PS를 fitting set에 넣지 않는다.

Reference512 전체는 유지한다. Active-set/GSS류 선택은 필요한 gradient 제약의 중복을 줄이는 내부 장치일 수 있지만 전체512 평가·제약 검사와 제외된 위반의 재추가는 필요하다. 단순 recency 가중 평균은 오래된 active fact를 버릴 수 있으므로 no-harm 여부와 우선순위 가중을 구분한다. GSS를 사용한다고 일반화·계산 절약이 자동 보장되지는 않는다.

## 7. 증명 다음에 무엇을 실증해야 하는가

논문용 universal collapse theorem을 먼저 완성할 필요는 없다. 앞의 exact identities와 conditional theorem으로 가설을 명확히 한 뒤, 한 episode에서 연결을 관측하고 개입해야 한다.

- Key geometry: 실제 writer가 쓰는 projected/whitened key. Current-current, current-past, current-reference를 구분한다.
- Target demand: 동일 좌표의 residual과 방향별 loading. 작은 σ_j만의 순위와 비교한다.
- Actual write: 해당 mode가 실제 Δ에 얼마만큼 나타나며 fitting error가 얼마인지 확인한다.
- Reference response: ΔK_j와 누적 cross term을 계산한다. 평균 subject 위치가 아니라 해당 판단의 전체 입력 경로를 사용한다.
- Functional effect: 동일 neighbor의 entry→endpoint margin 변화, signed directional effect, 실제 성공 전이를 연결한다.
- Intervention: 가설상 유해 mode를 제거/대체한 반응을 frozen suffix에서 확인한다. Activation patch 진단과 모든 입력에서 구현 가능한 단일 weight update를 구분한다. 이후 같은 편집 품질을 유지하는 실제 weight intervention으로 이득을 확인한다.

Official N을 사후 기전 감사에 사용할 수는 있지만 그 사례를 reference·선택 신호로 넣고 동일 N에서 방법의 성능을 주장하면 안 된다. Reference로 선택한 신호가 별도 batch/독립 N에 전달되는지 확인한다. Paraphrase는 observer로 유지한다.

현재 fixed10k에는 batch scalar write/target norm과 at-write/final NLL이 있지만 이를 묶는 projected target loading과 같은 neighbor의 전체 response/Jacobian은 없다. 기존 norm↔NS 손실 상관은 batch 추세를 제거하면 크게 낮아지고 이번 write의 증분 손상이 아닌 W0 대비 수준차다. EN의 condition 1.60e9는 Current all-token lock K_E에 대한 값이므로 native mean-key100/past collision 증거가 아니다.

사용자가 정한 cold 시작은 유지한다. B1에서 intra-batch/base interference를 확인한 뒤 같은 W0에서 시작하는 짧은 sequential로 past overlap과 누적 상쇄를 관측한다. 5000-edit checkpoint를 새 비교의 출발점으로 사용하지 않는다.

## 8. 선행연구와 포지션의 한계

|요소|이미 존재하는 가까운 연구|
|---|---|
|같은 edit equality 아래 preservation 최소화|[EMMET §5](https://aclanthology.org/2024.findings-emnlp.903.pdf)|
|누적 W−W0, 과거 key-value와 anchor 비용의 공동 closed form|[RLSEdit §3.1–3.2](https://arxiv.org/html/2601.15686v1)|
|Downstream curvature/GN와 history 통계|[CrispEdit §3.2–3.4](https://arxiv.org/html/2602.15823v2)|
|Reference covariance 분포의 중요성, separable output metric의 취소|[Moir §5.1 및 Appendix B.2](https://arxiv.org/html/2607.20433)|

특히 output metric G≻0 하나를 붙인 tr(GΔCΔ^T), ΔK=R 문제에서는 G가 최적해에서 취소된다. Sensitivity를 넣었다는 말만으로 방향이 바뀌지 않는다. 실제 문서/token별 sensitivity 또는 decision inequality가 해를 바꾸는지 보여야 한다.

기여 후보는 교차항·QP·single-layer 자체가 아니다. **같은 강한 편집을 구현하는 update 중 일부만 preservation에 해롭다는 것을 실제 모델에서 분리하고, 제거 가능한 유해 성분을 정량적으로 찾아 낮은 추가 비용으로 억제하는 것**이다. 이 성분의 존재와 held-out 기능 개선은 아직 확인되지 않았다. 선행연구 조사도 최초성을 보증하지 않는다.

## 9. 이번 확인 범위

Primary 논문·기존 감사·코드를 읽고 조건부 정리와 반례를 검토했다. 작은 NumPy 행렬로 누적 cross term, exact/approximate writer 차이, 동일 edit의 다른 해, reference 피해의 반례, 제거 가능한 activation 비용을 19개 검산했고 모두 통과했다. 이는 LLM 실험이나 수학적 증명의 대체물이 아니다.

- [CPU 검산 코드](/mnt/raid5/janghj/layer_allocation/empirical/single_layer_mechanism_checks_20260919.py)
- [검산 결과](/mnt/raid5/janghj/layer_allocation/empirical/single_layer_mechanism_checks_20260919.json)

실행: `/mnt/raid5/janghj/EasyEdit/.venv/bin/python /mnt/raid5/janghj/layer_allocation/empirical/single_layer_mechanism_checks_20260919.py`. 기본 python3에는 numpy가 없어 기존 EasyEdit 환경으로 실행했다. 모델 forward/backward, 새 GPU job, optimizer 구현, commit/push는 하지 않았다.
