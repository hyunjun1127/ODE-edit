**원본 모델과 현재 편집 모델의 차이에서 손상 신호를 찾기 — 2026-09-29**

이번 논의의 목적은 배분 규칙을 확정하는 것이 아니라, 그 규칙에 입력할 손상 신호를 정의하는 것이다. 앞선 [배분 설계](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-29-knowledge-editing-survey/history-aware-allocation-design-ko.md)는 보존 비용을 얻을 수 있다고 가정했다. 여기서는 순차 안정성·보존 논문이 실제로 무엇을 측정하고 개입했는지 다시 대조한다. 새 모델 실험, GPU 실행, 기존 실험 원자료의 재감사는 수행하지 않았다.

M0는 편집 시작 전 원본, Mt는 t회 편집 뒤 live 모델, Mt+d는 다음 후보를 적용한 모델이다. 원본과 live 사이의 **변화**, 그 변화에 의해 잃은 **행동**, 다음 쓰기에 대한 **취약성**을 별도로 측정한다. 특정 입력·방향에 따라 위험이 다르므로 층 전체에 고정된 ‘손상도’ 하나가 있다고 가정하지 않는다.

**문헌이 제공하는 관측량과 근거의 강도.**

| 연구 / 확인 위치 | 무엇을 실제로 보았는가 | 가져올 insight | 해석의 한계 |
|---|---|---|---|
| [REVIVE](https://aclanthology.org/2026.acl-long.1384.pdf), §2.3–3 | 원래 weight의 큰 singular-value 방향과 작은 방향에 동일 Frobenius norm의 perturbation을 가함. 순차 편집의 dominant reconstruction/vector similarity도 추적 | 같은 크기라도 원래 중요한 입출력 방향에 닿는 변화가 더 해로울 수 있음. 방향을 지정한 개입 근거가 있음 | 실제 live 모델의 모든 손상이 그 방향에서 발생했다는 보장은 아님. Main text는 original basis를 설명하지만 Appendix Algorithm 1은 current weight라고 표기하므로 구현 기준 시점은 추가 대조 필요 |
| [CrispEdit](https://arxiv.org/abs/2602.15823), §3.2–3.4 | 보호 출력의 Bregman divergence와 Gauss–Newton 곡률. K-FAC의 입력 공분산과 출력방향 민감도 | 입력 key에 작용한 변화가 최종 capability에 얼마나 중요한지까지 봐야 함 | Reference distribution·국소 근사·곡률 재사용에 의존. 작은 곡률이 모든 질문의 보존을 보장하지 않음 |
| [AlphaEdit](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html), §3 | 고정 보호 key K0에 대한 업데이트 readout을 null-space로 억제하고 과거 edit key 항도 사용 | Weight 변화량 대신 보호할 입력에 실제로 작용한 변화량을 측정 | 선정 key/위치, projector threshold, 변하는 live 입력과 출력 비선형성이 남음 |
| [EvoEdit](https://aclanthology.org/2026.findings-acl.75/), §4, Fig.2 | 정적 projector P에 대해 누적 과거 key Kp의 누출 ||PKp||F를 관측하고 projector를 갱신 | 최초 보호 집합과 나중에 수용한 편집은 별도의 보호 대상 | 논문의 null-space drift는 동일 입력의 k0→kt 이동과 다름. 모든 과거 key를 현재 모델로 재추출하는 알고리즘이라고 설명하지 않음 |
| [LyapLock](https://aclanthology.org/2025.emnlp-main.327/), §2–3 | PLt=||Wt K0−V0||F², 과거 편집 reconstruction loss, PL 초과를 반영한 virtual queue | 현재까지 쌓인 보호 mapping 오차를 다음 write와 연결할 수 있음 | PL은 고정 key의 국소 surrogate. 이론은 장기 평균 제약이며 매 시점 모든 입력의 보존 보장이 아님 |
| [MPES + norm constraint](https://aclanthology.org/2025.findings-emnlp.1234/), §5–6, App.D | Target 과최적화, weight norm, 일반 입력에서 edited MLP output norm 및 residual 구성요소의 norm 비중 | Weight 팽창이 실제 일반 입력의 activation을 지배하는지 확인할 이유 | Norm 제한 ablation은 있으나 norm 증가→특정 출력 손상의 유일한 매개경로를 분리한 증명은 아님 |
| [NAS](https://arxiv.org/abs/2602.02543), §3, App.B | 현재 weight/value와 solved target value의 norm feedback, original-model pilot에서 얻은 target norm 기준 | 다음 후보가 현재 readout보다 얼마나 큰 target을 요구하는지 볼 수 있음 | 지수 증가·유계성은 empirical scaling 가정에 의존. Value inflation이 곧 기능 손상은 아님 |
| [SPHERE](https://openreview.net/pdf?id=CHsdtzCip6), §2–4, App.C | 정규화된 neuron 방향들의 hyperspherical energy와 성능 상관, 주요 방향의 update 억제 | Weight의 크기 외에 방향 분포도 변함 | HE 관련 작은-perturbation 관계는 출력 drift 하한을 주는 형태. HE 변화가 작으니 안전하다는 역추론 불가 |
| [PRUNE](https://proceedings.iclr.cc/paper_files/paper/2025/file/2c15b0221da28bc6f4373a7e78b896dd-Paper-Conference.pdf), §3–4 | Condition number와 누적 update singular-value 팽창; 누적 update의 큰 singular values를 압축하는 회복 개입 | 과도한 spectral amplitude를 감시할 이유 | 이론은 inverse key-compensation 문제이며 실제 forward 손상과 동일하지 않음. 주 실험은 누적 편집 후 회복; 반복판은 부록 C.6 |
| [StableEdit](https://arxiv.org/abs/2605.11836), §3, §5.4, B.6 | Value-gradient 평균·공분산, update 정렬·norm, running-statistics normalization ablation | 비슷한 bias 방향이 계속 쌓이는지 볼 수 있음 | Lifelong Normalization은 LayerNorm 수정이 아님. 제거 시 update가 거의 0으로 줄며 under-fitting하는 경우도 있음. Per-step bound가 누적 변형 유계성을 뜻하지 않음 |
| [OTE–SE](https://arxiv.org/abs/2605.26670), §3, App.C.4/D.2 | 누적 OLS 목적과 순차 업데이트의 일치, 보정 제거의 hidden drift 및 edit forgetting | 원래 모델에 가까워진다는 이유로 순차 보존이 좋아졌다고 할 수 없음 | 이론은 OLS/shifted-quadratic 범위. 저자도 매우 긴 stream의 일반능력 저하가 완전히 사라지지는 않음을 인정 |

위 표는 관련 본문·방법·부록을 선별 재검토한 기록이다. 독립 재현 결과가 아니다. 이후의 진단식과 우선순위는 이 논문들에서 도출한 제안이며 각 논문이 모두 같은 알고리즘을 주장한다는 뜻이 아니다.

**1. 첫 후보: 원래 중요한 weight 방향의 어떤 mapping이 바뀌었는가.**

W0=U0 Σ0 V0ᵀ, E_t=Wt−W0라 하자. 원래 주요 입출력 방향에 대한 projector를 PU, PV로 고정하고 다음 성분을 정의할 수 있다.

\[
S(E)=E-(I-P_U)E(I-P_V).
\]

이는 주요 입력 또는 주요 출력 방향과 접촉하는 변화다. 단순히 양측 top-top block만 보는 것과 다르다. 진단량은

\[
\|S(E_t)\|_F^2
=\|E_tV_{0,r}\|_F^2
+\|U_{0,r}^{\top}E_t(I-P_V)\|_F^2.
\]

첫 항은 원래 주요 입력 방향의 출력 mapping 변화, 둘째는 나머지 입력이 주요 출력 방향으로 섞이는 변화다. 원모델 좌표를 고정해야 시점 간 비교가 가능하다. 이 quantity 자체는 여기서 제안하는 REVIVE-inspired 진단이며 논문의 고유 metric 이름을 붙이지 않는다. 큰 singular value가 중요 기능 전체를 대표한다고 가정하지 않고 행동 손상과 대조한다.

지금까지의 중요한 방향 침범량이 커도, 다음 후보 d가 이를 더 키울지는 별도다.

\[
\|S(E_t+d)\|_F^2-\|S(E_t)\|_F^2
=2\langle S(E_t),S(d)\rangle_F+\|S(d)\|_F^2.
\]

이미 큰 오차를 가진 위치도 d가 반대 방향이면 일부 회복할 수 있다. ‘누적 변화가 큰 층을 제외’하는 정책에는 이 정보가 없다.

**2. 둘째 후보: 보호할 입력의 실제 MLP readout이 어떻게 변했는가.**

동일 입력·동일 token prefix에서 원본과 live의 key를 k0, kt, 편집되는 행렬의 출력을 m0=W0k0, mt=Wtkt라고 한다. 정확한 대수적 분해는

\[
m_t-m_0
=\underbrace{E_tk_0}_{\text{해당 weight의 변화}}
+\underbrace{W_0(k_t-k_0)}_{\text{입력 표현의 변화}}
+\underbrace{E_t(k_t-k_0)}_{\text{상호작용}}.
\]

이는 벡터 분해다. 각 항의 norm을 더한 값을 전체 변화량으로 취급하거나, 해당 항을 곧바로 출력 손상의 인과 기여율로 부르면 안 된다. 누적된 여러 층의 차이를 다시 합치면 같은 upstream 효과를 중복 계산할 수 있다.

**중요한 조건:** 동일한 MLP Wout 하나만 편집하고 그 입력을 만드는 계산이 고정되어 있다면, 같은 teacher-forced prefix의 해당 key는 바뀌지 않는다. 이 경우 kt=k0이고 뒤 두 항은 0이다. Live generation에서 서로 다른 prefix를 생성한 결과를 이 비교에 섞지 않는다. 그 층 뒤의 attention 패턴은 입력 activation 변화로 달라질 수 있다.

모든 보호 질문에서 cached-key response E_tK0와 live response WtKt−W0K0를 함께 관측한다. Subject-last 같은 일부 위치만 확인하면 다른 token 위치의 변화가 뒤 attention을 통해 영향을 주는 경로를 놓칠 수 있다. 모든 위치를 일괄 보호하겠다는 방법 제안과, 원인 확인 때 영향 위치를 조사한다는 것은 구별한다.

MPES에서 가져올 추가 관측은 이 response가 일반 입력에서 residual의 다른 가지에 비해 얼마나 커졌는지다. MLP output norm, residual stream norm, 방향/cosine을 원본/live 동일 입력에서 함께 본다. Norm들의 비중은 residual 합에 대한 인과 기여율이 아니며, 가지 간 상쇄가 있을 수 있다.

**3. 셋째 후보: 그 변화가 최종 정답을 얼마나 밀어내는가.**

CrispEdit의 핵심 insight는 representation 변화 전체를 같은 비용으로 취급하지 않는 것이다. 작은 후보의 층 출력 변화 δm에 대해 현재 모델의 downstream Jacobian J와 출력 loss 곡률 F를 이용한 δmᵀJᵀFJδm은 기능적 민감도를 반영하는 국소 근사다. 원래 모델부터 지금까지의 큰 변화 전체에 이 근사를 그대로 적용하지 않는다.

실제로 확인해야 할 행동은 보호 질문의 정답률, 정답 score, 경쟁 오답/새 edit target score, 원본 대비 출력 분포 변화다. 정답 score가 덜 나빠졌거나 좋아졌더라도 잘못된 edit target이 더 크게 올라오면 locality는 실패할 수 있다. Multi-token 답은 고정 scoring 규약을 사용하고 token-mean NLL 차이를 sequence log-odds라고 부르지 않는다. KL 감소도 정답 보존 자체는 아니다.

원래 보호 행동의 기준은 M0이지만, 수용한 과거 편집의 기준은 최신 유효 목표답 또는 수용 시점의 행동이다. 원래 틀린 답으로 돌아가는 것을 안정성으로 보상하지 않는다. OTE–SE App.C.4의 hidden drift 감소와 edit forgetting 동반 결과가 이 구분을 뒷받침한다.

보충 선행인 [SADR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/35cb54b887e7aafe74829677cce6c5c6-Abstract-Conference.html)은 base attention을 patch하는 개입으로 특정 relation/distractor 실패의 회복을 보인다. 이는 MLP 변화가 downstream에서 증폭되는 경로를 확인할 이유다. Single-edit 연구의 결과를 모든 lifelong collapse의 주원인으로 확장하지 않는다. Attention 변화 전체를 되돌리면 필요한 새 편집도 사라질 수 있다.

**순차성의 핵심: 누적 오차와 다음 write의 정렬.**

LyapLock/AlphaEdit의 보호 mapping 관점을 사용하여 고정 reference keys K0와 V0=W0K0를 두면

\[
A_t=W_tK_0-V_0,\qquad PL_t=\|A_t\|_F^2.
\]

후보 d가 추가될 때 다음 식은 정확하다.

\[
PL(W_t+d)-PL(W_t)
=\underbrace{2\langle A_t,dK_0\rangle_F}_{\text{기존 오차의 강화 또는 상쇄}}
+\underbrace{\|dK_0\|_F^2}_{\text{새로 가하는 변화의 크기}}.
\]

핵심은 첫 항이다. 이것이 빠지면 같은 크기의 후보들이 현재 이력에서 다르게 작동하는 이유를 놓친다. 이 등식은 고정 key의 국소 linear mapping에 대한 것이며 LLM 기능 손상의 정확식은 아니다. Upstream을 고정하고 해당 행렬만 추가 수정하는 후보라면 live error B_t=WtKt−W0K0에 대해서도 2〈B_t,dKt〉+||dKt||²로 대응되는 mapping 변화량을 계산할 수 있다. 동시에 앞층까지 바꾸는 후보에서는 Kt도 달라지므로 이 식을 그대로 쓸 수 없다.

출력 위험 R_ref를 원본/유효목표에 고정해 놓고 live 모델에서 전개하면

\[
R_{\rm ref}(\theta_t+u)-R_{\rm ref}(\theta_t)
\approx \nabla R_{\rm ref}(\theta_t)^\top u
+\tfrac12u^\top H_{\rm ref}(\theta_t)u.
\]

이미 원본에서 벗어난 live 모델에서는 첫 항을 일반적으로 버릴 수 없다. CrispEdit의 Bregman이 기준점에서 1차항을 없앤다는 것과 이 식은 충돌하지 않는다. GGN/K-FAC로 H를 근사하면 비선형 잔차가 생기며, 여러 층의 후보에는 교차항도 남는다. 따라서 실제 후보의 출력 검사를 통해 예측 오차를 확인해야 한다.

**Norm·condition number·방향 에너지는 어디까지 쓰는가.**

NAS의 특별한 C=I exact rank-one write에서는 다음 norm 변화식이 성립한다.

\[
\|W_{t+1}\|_F^2-\|W_t\|_F^2
=\frac{\|v_{\rm new}\|^2-\|v_{\rm old}\|^2}{\|k\|^2}.
\]

일반적인 writer에서는 2〈Wt,d〉+||d||²를 직접 계산한다. 이는 다음 weight norm 증가의 정확식이지 행동 손상의 식이 아니다. Target/value inflation, 실제 보호 입력의 response, residual 내 비중까지 연결할 때 원인 후보가 구체화된다.

Normalized HE와 condition number만으로 mapping 보존을 인증할 수 없는 간단한 반례도 있다. Wt=cW0(c>0)이면 두 지표는 같지만 선형 출력은 달라진다. Wt=W0Q인 직교 우측 회전도 두 지표와 singular values를 유지하지만 동일 입력의 출력은 일반적으로 달라진다. 이는 전체 모델의 출력이 항상 나빠진다는 반례가 아니라, 해당 지표들만으로 mapping 불변성을 식별할 수 없다는 반례다.

따라서 PRUNE/SPHERE의 지표는 특정 수치적·기하적 붕괴 양상의 보조 경보로 둔다. StableEdit의 gradient 통계도 업데이트 방향이 반복되는지를 알려주는 보조 정보다. Update가 거의 0이 되어 under-fitting하는 경우를 안정성 향상으로 오인하지 않는다.

**직접적인 손상 연결을 확인하는 두 종류의 개입.**

첫째는 현재까지의 손상에 대한 부분 복원이다. Live 모델에서 E_t의 원래 중요한 singular 성분만 조금 되돌리고, 같은 norm·rank의 다른 성분 복원과 비교한다. 또는 의심한 MLP response/attention을 같은 입력의 원본 값으로 부분 patch한다. 보호 출력의 회복과 새·과거 편집의 손실을 동시에 본다. Norm 복구와 방향 복구를 분리하면 MPES/NAS의 크기 가설과 REVIVE의 방향 가설을 비교할 수 있다.

이 개입은 현재 모델에서 그 성분을 되돌렸을 때의 조건부 효과다. 여러 층의 보상과 interaction, hybrid activation의 부자연스러움이 있으므로 유일한 최초 원인이나 손상의 가산적 기여율로 해석하지 않는다. 필요하면 쌍별 joint 복원도 확인한다.

둘째는 다음 손상의 예측이다. 같은 live checkpoint와 같은 새 edit에서 후보 write를 만든다. 새 답 성공 및 개발용 paraphrase 일반화를 비교 가능한 수준으로 맞춘 뒤, 내부 신호가 별도 보호 질문의 실제 추가 손상을 순위화하는지 본다. 같은 norm으로 방향만 비교하는 기전 실험과, 같은 편집 품질로 실제 배분 효용을 비교하는 실험은 구분한다.

**지금 우선할 최소 관측 묶음.**

| 단계 | 꼭 볼 것 | 해소할 질문 |
|---|---|---|
| 행동 기준 | 동일 보호 입력의 원본/live 정답·오답 score, 생성 성공, 유효 과거답 | 무엇이 실제로 손상됐는가? |
| 구조적 후보 | 원래 singular 좌표의 E_t, 주요 입출력 방향 침범 | 손상에 민감한 원래 weight 방향을 건드렸는가? |
| 계산 경로 | E_tK0, live response, 조건에 따라 key drift, 여러 token 위치, residual 내 비중 | 어느 입력·위치에서 어떤 변화가 생겼는가? |
| 기능 연결 | 해당 response에 대한 signed output sensitivity 및 부분 복원 효과 | 어떤 변화가 최종 오답에 영향을 주는가? |
| 다음 write | 기존 오차와 dK의 정렬, 후보의 실제 출력 손상 증가 | 이번 쓰기가 그 손상을 더 키우는가? |

보호 패널은 일반 능력·편집 주변의 보존 대상·유효 과거 편집을 구분한다. Official test locality를 선택 정책에 노출하지 않고 개발용 진단 패널과 최종 평가 패널을 나눈다. 평균뿐 아니라 tail과 편집 age별 손실도 관측한다. 모든 내부 지표를 임의 가중합으로 만들어 allocator에 넣기 전에, 각 신호가 추가 손상을 예측하는지와 복원 개입의 방향이 맞는지를 확인한다.

현재 문헌을 종합한 우선순위는 **REVIVE의 중요한 방향 → AlphaEdit/LyapLock의 보호 입력 반응 → CrispEdit의 출력 민감도**다. MPES/NAS의 activation 지배·target inflation은 이 경로를 악화시키는 별도의 원인 후보로 비교한다. 이 조합은 연구 가설이며, 어느 신호가 우리 모델에서 지배적인지는 아직 확인되지 않았다.
