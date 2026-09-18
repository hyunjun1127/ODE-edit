# Preservation repair의 가능성 및 repair layer 선정 근거 점검

2026-09-17. 사용자 후속 질문에 따른 수학·논문·기존 실험·구현 검토. 새 GPU 실험이나 실제 층별 repair 성능 측정은 수행하지 않았다.

## 1. 결론과 설계 수정

조건부 preservation repair는 수학적으로 가능한 경우가 있지만, 현재 Llama에서 full native L4 이후의 성공과 최적 repair layer는 미검증이다. **L8 고정은 해제하고 후보 중 하나로 둔다.** L4–L8은 기존 edit window이며 repair 가능 구간의 경계가 아니다.

L4 writing을 고정하는 첫 비교에서는 L5/L6/L7/L8을 주 후보, L9/L12를 범위 밖 대조로 삼는다. L9는 기존 window 바로 다음 층, L12는 그보다 떨어진 downstream 대조다. 이는 bounded 진단용 사전 선택이며 L12의 최적성이나 모든 후반 층의 대표성을 주장하지 않는다. 결과가 불명확하면 미조사 층의 가능성은 남는다.

L4 자체를 다시 고치면 full-native-L4 weight 동결 조건과 충돌한다. L4 refinement는 이미 있는 REFIT4와 같은 별도 정책으로 분리한다. L4 이전 층의 변경은 L4에 들어가는 입력까지 바꾸므로 첫 post-L4 repair 비교에 넣지 않는다.

현재 [방법 v1](2026-09-17-l4-preserving-batch-repair-v1.md)의 단일 L8 표기는 층 ℓ에 적용할 수 있는 repair template으로 일반화한다. 아래 feasibility 진단 전에는 ℓ=8 또는 다른 층을 최종 방법으로 확정하지 않는다. 기존 5-arm lifelong 계획은 층/정책 확정 이후의 조건부 계획이다.

## 2. Causal tracing과 이번 repair는 다른 개입이다

| 항목 | 일반적인 factual causal tracing | 이번 preservation repair |
|---|---|---|
| 출발 | 원래 모델 + subject 입력의 인위적 corruption | 실제 native L4 weight edit 이후 모델 |
| 목표 | 원래 object의 회복 | 새 edit를 유지하면서 비편집 입력의 손상을 감소 |
| 수정 객체 | 특정 token/layer activation을 clean 값으로 치환 | 하나의 공유 down-projection weight update |
| 자유도 | 입력별로 다른 clean activation을 직접 주입 | 모든 입력/token에 같은 ΔW 적용 |
| 주요 측정 | 원래 target 확률의 causal recovery | edit 제약 아래 W0-output KL 감소와 독립 PS/NS |

ROME §2.1–2.2의 tracing은 subject embedding에 noise를 넣고 내부 activation을 복원하는 방식이다. MEMIT §4.1의 GPT-J 구간은3–8이며, 원래 논문 모델에서 얻은 결과를 현재 Llama3의4–8 구간 선정 실험으로 바꿔 읽으면 안 된다. [ROME](https://arxiv.org/pdf/2202.05262), [MEMIT](https://arxiv.org/html/2210.07229v2).

Hase et al.은 GPT-J/CounterFact652개 사실에서 weight-edit 성능의 설명력이 layer만으로 R²=.947이고 tracing 효과 추가시 .948임을 보고했다. 이는 tracing과 weight editing 위치의 관계가 단순하지 않다는 직접 근거이며 Llama repair를 직접 검증한 것은 아니다. [Does Localization Inform Editing? §4–6](https://arxiv.org/pdf/2301.04213).

BLUE §4.3은 마지막 critical layer 사용을 기존 locate-then-edit의 마지막층 residual 계산 구조와 연결한다. L8이 preservation repair에 최적이라는 실험을 근거로 삼지 않는다. [BLUE](https://arxiv.org/html/2502.03748v3).

CAKE Appendix E.1은 모든 층을 tracing한 것이 아니라 MEMIT/AlphaEdit의 기존 critical set 안으로 tracing을 제한한다. Llama3는{4,5,6,7,8}이다. 알려진 fact1000개, noise10회, 마지막 subject token의 post-MLP clean activation 복원이며 Llama3 noise=.02다. 따라서 Fig.6은 범위 밖 repair 불가능성뿐 아니라 범위 밖 tracing score가 낮다는 근거도 아니다. [CAKE Appendix C.2/E.1](https://aclanthology.org/2026.acl-long.918.pdf).

현재 실행과 직접 연결된 Llama 전층 causal map 원자료·sample ID·receipt는 조사 범위에서 찾지 못했다. 그러므로4–8 설정 자체는 확인됐으나 그 설정의 전층 탐색 과정을 이번에 재현했다고 말할 수 없다.

### Activation 복원 성공을 weight repair 성공으로 읽으면 안 되는 이유

수정이 L4뿐인 native chain에서 L4 뒤 어느 층이든 **전체 token의 전체 block-output residual stream**을 W0 값으로 치환하면, 이후 suffix weight가 W0와 같으므로 최종 출력도 W0로 돌아간다. 이런 full-state patch는 어느 층이 좋은 공유-weight repair 위치인지 구분하지 못한다. 일반 causal tracing의 제한된 token/module patch와도 다른 개입이다.

또한 edit 문장까지 W0 상태로 복원하면 원하는 새 사실도 취소될 수 있다. 필요한 것은 원래 응답을 무조건 되찾는 능력이 아니라 편집 입력과 보존 입력을 구별하여 보상하는 능력이다.

## 3. 어떤 조건이면 repair가 가능한가

같은 post-L4 anchor WN에서 층 ℓ의 작은 update를 d라 하자. 보존 loss gradient는 g_B,ℓ, 보호할 edit 응답의 Jacobian은 J_E,ℓ이다.

강한 일차 equality 이상화는 다음과 같다.

    J_E,ℓ d = 0                  편집 응답의 일차 유지
    g_B,ℓ^T d < 0               보존 loss의 일차 감소

Euclidean equality 모델에서는 Πker(J_E,ℓ) g_B,ℓ가 비영이면 projected descent가 존재한다. 양의 정부호 metric H를 쓰면 v=H^(-1/2)g_B, C=J_E H^(-1/2)이고, trust 제약 없는 quadratic equality 모델의 gain은 0.5||Πker(C)v||²다. Raw gradient norm만 비교해서는 edit과 충돌하지 않는 성분을 알 수 없다.

실제 방법은 모든 logit equality가 아니라 mean NLL·성공 margin inequalities와 finite acceptance를 사용하므로 위 projector 수치를 실제 inequality-QP gain과 동일시하지 않는다. Active linear guard가 A d≤0일 때 descent가 없는 경우는 −g_B가 A의 nonnegative row cone에 속하는 경우와 연결된다. 즉 loss 감소 방향이 보호해야 할 edit을 훼손하는 방향과 완전히 충돌할 수 있다.

### 가장 작은 가능/불가능 예시

- Edit 응답이 d1, preservation 오차가1+d2라면 d=(0,−1)은 edit을 유지하면서 오차를 없앤다.
- Edit 응답도 d1, preservation 오차도1+d1이고 edit을 정확히 고정해야 한다면 d1=0이므로 repair할 수 없다.

같은 parameter 수여도 응답의 분리 가능성에 따라 결과가 달라진다. 실제 transformer의 어느 층이 첫 경우에 가까운지는 측정해야 한다.

### Layer key 관점에서도 같은 제약이 생긴다

고정 token과 upstream weights 아래 한 층의 down-projection을 바꾸면 그 층의 input key K는 고정이다. Local module에서 edit key K_E를 유지하고 base key K_B를 T_B만큼 바꾸려면

    ΔW K_E=0,   ΔW K_B=T_B

를 동시에 만족해야 한다. K_E의 column span을 지우는 직교 projector를 P라 하면 ΔW=ΔW P이므로, 필요한 보존 변화가 P K_B를 통해 실현 가능해야 한다. Edit와 Base key가 같은 방향에 묶여 있는데 서로 다른 변화를 요구하면 정확한 local 보정은 불가능할 수 있다.

그러나 module-level 보존은 최종 logit 보존보다 강한 조건일 수 있다. 이 key 예시로 최종 transformer repair의 필요충분조건을 주장하지 않는다. 실제 비교에는 전체 출력 Jacobian과 forward를 사용한다.

### 일차 가능성과 실제 채택 가능성을 분리한다

g_B≠0이고 모든 실제 guard에 엄격한 양의 slack이 있으며 smooth하면 충분히 작은 descent는 존재할 수 있다. 따라서 gradient가 있어도 원리적으로 항상 불가능하다는 설명도 부정확하다. 문제는 그 변화가 FP32에서 실현되고 ε_B보다 큰 의미 있는 감소를 내며 실제 품질 조건을 지키는가다.

또한 linearized cone과 실제 feasible tangent cone은 퇴화 제약에서 다를 수 있다. 예를 들어 d²≤0은 anchor에서 gradient0이지만 허용되는 유한 d는0뿐이다. QP gain·KKT 통과만으로 finite repair 가능성을 선언하지 않는다. 현재 QP mean slack0는 실제 acceptance의 ε_L>0보다 보수적이라는 차이도 기록한다.

## 4. 왜 edit-critical 층에서도 가능할 수 있고 실패할 수 있는가

가능한 경우는 edit에 영향을 주는 방향과 unrelated 입력의 손상을 보상하는 방향이 일부 분리될 때다. Critical layer라는 말이 그 층의 모든 parameter 방향이 모든 사실에 똑같이 필수라는 뜻은 아니다. 입력별 activation/key 및 downstream response가 다르면 기능을 나눌 여지가 있다.

반대로 factual recall에 강하게 관여하는 층은 새 edit에도 민감할 수 있으므로, 보존 복구가 편집을 되돌리는 방향과 겹칠 수 있다. 더 깊은 층에는 상쇄할 자유도가 있을 수 있지만 원래 정보가 충분히 남아 있지 않거나 response가 얽혀 있으면 쉽지 않다. 이런 설명은 가능한 기전이며 L5나 L8의 우열을 예측한 사실은 아니다.

Hydra Effect는 뒤쪽 층의 기능 보상 현상을 보여주지만 activation ablation에 대한 모델 반응이지, 이번 shared-weight preservation repair의 직접 검증은 아니다. [Hydra Effect](https://arxiv.org/html/2307.15771v1).

## 5. 로컬 실험이 실제로 보여준 범위

실제 수정 객체는 Llama3-8B-Instruct의 `model.layers.{l}.mlp.down_proj.weight`, 0-based physical index다. Block-output z intervention과 영구 down-projection write를 구분한다. `v_loss_layer=31`은 repair 층이나 z write층이 아니다.

| 기존 관측 | 지지하는 것 | 지지하지 않는 것 |
|---|---|---|
| L4-only의 강한 RS/PS | L4를 주 writer로 유지할 근거 | L8 repair 최적성 |
| L5–L8 singleton edit 성능 | 각 층의 독립 target-writing 정책 성능 | 동일 L4 손상을 어느 층이 잘 repair하는가 |
| LD9/10 (.75,.5) | 약한 L4와 local L8의 조합 이득 | full L4 이후 L8 preservation repair |
| B-OS의 L8 보정 실패 | 그 solver/finite step의 실패 | 모든 L8 repair 방향의 부재 |

참고로 singleton의 final10k RS/PS/NS는 L4 99.390/95.680/65.348, L5 99.340/94.195/62.822, L6 96.830/85.740/58.396, L7 92.400/80.970/56.277, L8 93.960/77.780/54.703이다. 서로 다른 lifelong edit chain이며 repair layer 순위표가 아니다.

직접 post-full-L4인 B-OS는 Base risk .034895→.113661, Current NLL .047468→.062879, Current NS71.1→70.6이었다. 두 PCG RHS 미수렴 및 finite acceptance 부재를 분리해야 한다. 현재 성공한 L8 repair의 실증 근거로 사용할 수 없다.

근거: [singleton 통합리뷰](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md), [B-OS](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server2/multilayer-damage-compensation-b-2026-09-11-v1/partial-recall-r1/diagnostic-report-ko.md), [cold7 독립리뷰](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/global/2026-09-17-local-z-seven-arm-independent-review.md).

## 6. 먼저 할 비교: 같은 손상에 대한 layer별 repair 가능성

### Anchor와 입력

- 동일 pretrained W0에서 시작한 native N4 chain의 B1/B5/B10 직후 WN을 사용한다. 과거5000-edit warm checkpoint를 사용하지 않는다.
- 기존 cold7 N4 checkpoint를 재사용할 경우 source/model/token/history 연결을 확인한다. 이는 W0에서 출발한 동일 anchor의 재사용이지 후보별 누적 trajectory 비교가 아니다.
- 각 anchor에서 현재100 rewrite, 수신한 과거의 canonical Past64, S64, 같은 W0 teacher를 모든 층에 공유한다. B1 Past는 비어 있다.
- 별도 paraphrase 세트·생성·target·guard는 없다. Official P/N과 Dev128은 선택을 봉인한 뒤 observer로만 사용한다.

### 후보와 동일 조건

- L5/L6/L7/L8: 기존 critical band 안의 추가층.
- L9/L12: band 밖 downstream 대조.
- Repair off: 공통 WN.
- 3anchors×6layers=18개의 독립 layer-anchor cells. 모든 후보는 공통 WN에서 분기하고 rollback하며 서로의 update를 이어받지 않는다. Repair를 baseline N4 anchor chain에 commit하지 않는다.

각 층은 같은 목적·guard·최대3방향 및 같은 probe 제한을 사용한다. W4와 비선택 weight는 WN에 고정한다. 같은 weight Frobenius norm만으로 공정성을 정하지 않는다. 층별 activation scale과 downstream 감도가 다르기 때문이다.

층 비교의 주 trust budget은 0.5 c^T Htilde_l c≤τ, τ=B(WN)로 공통화하고 τ/4^k, k=0..5의 nested level에서 실제 반응을 비교한다. Whitening 후 ||u||2≤sqrt(2τ)인 작은 convex QCQP/SOCP다. 기존 coordinate box는 방향기저 회전에 따라 범위가 달라지므로 주 layer ranking에 사용하지 않는다. 이 수정은 비교용 contract에 명시하며 기존 box 알고리즘과 같은 solver라고 부르지 않는다.

Γfree와 Γsafe는 동일 layer·basis·목적·Htilde·τ에서 계산한다. 서로 다른 accepted radius의 gain 비율을 섞지 않는다. 각 cell은 중복되지 않는 최대6개 guarded proposal을 평가하고, τ0에서 guard-free shadow1개를 추가한다. 최대126개 proposal 평가이며 derivative 검사와 observer 비용은 별도다. 각 proposal 평가는 여러 문서/token의 forward를 포함하므로126 model calls라는 뜻이 아니다. 단일층 controller를 모사하는 선택은 큰 τ부터 첫 finite-accepted 후보 또는 zero로 봉인한다.

λnum, 실제 ||ΔW_l||, undamped/damped Fisher energy와 ratio, numerical residual을 모두 기록한다. Damping이 큰 층에서 metric이 어떻게 달라졌는지 공개한다. Predicted energy가 같아도 실제 KL·edit 변화가 같다고 가정하지 않는다.

### 각 cell에서 구분할 네 질문

1. **감도:** 실제 L4 손상이 있는 상태에서 g_B,l과 output JVP가 유의하고 FD와 맞는가.
2. **편집과의 분리:** 같은 trust budget에서 Γfree 대비 Γsafe가 얼마나 남는가. Mean/margin 제약별 active status를 기록한다.
3. **실현:** 검증된 작은 문제의 해를 실제 weight에 적용했을 때 KL 감소와 rewrite 보호가 함께 성립하는가.
4. **전이:** 봉인 후 Dev128·official P/N·과거 cohort에서도 보존 이득이 나타나는가.

동일 response model의 guard-free proposal도 shadow로 평가한다. KL만 감소하고 rewrite를 잃는다면 보존 방향이 없는 것이 아니라 해당 방향이 edit과 충돌하는 것이다. Shadow는 chain에 commit하지 않는다. Accepted/rejected probe를 모두 보존한다.

제한된3방향에서 실패하면 작은 search family의 실패로 기록한다. 그것을 layer 전체 불가능성으로 확대하지 않는다. 이 경우 active per-request guard gradient를 추가하는 별도 확장 진단으로 representation 제한과 실제 충돌을 분리할 수 있으나 이번18cell 결과에 소급 혼합하지 않는다.

## 7. 이 진단이 끝난 뒤의 방법 선택

- 한 층이 일관되게 좋은 finite repair를 제공하면 먼저 그 고정층을 사용한다. Dynamic layer routing 자체를 의무로 삼지 않는다.
- 동일 anchor 비교에서 batch/누적 state에 따라 유리한 층이 바뀌고 차이가 수치 잡음보다 크면, 온라인 S64+rewrite guard 기반 층 선택을 후속 가설로 둔다. Official P/N으로 온라인 winner를 바꾸지 않는다.
- Band 밖 층이 우세하면 repair 후보를4–8로 제한할 이유가 없다.
- Critical band 전부에서 실패하고 다른 층에서도 이득이 없으면 full-L4 고정 repair 가설을 보류한다. LD처럼 edit 자체를 약화·분담하는 정책과 혼동하지 않는다.

18cell은 개발 진단이다. Official P/N을 읽고 global layer나 method를 변경하면 그 데이터에 대해 독립 최종 성능이라고 주장하지 않는다. Layer/policy를 lock한 뒤 별도 cold lifelong 비교에서 RS·PS 유지와 locality 이득을 검증한다. 단순히 같은 데이터로 W0에서 다시 실행하는 것은 독립 검증이 아니며, 개발에 쓰지 않은 평가 입력을 확보하고 편집 순서/seed 분리 범위를 명시해야 한다.

새 method의 feasibility와 layer 선택이 아직 미확정인 동안 기존 5-arm lifelong 제출은 하지 않는다. 이는 사용자의 상세 점검 요청에 따른 설계 순서 수정이며 자동 승인 요청이나 GPU 실행 중단 명령이 아니다. 현재 새 GPU 작업은 없다.

## 8. 구현 경계

Cold7 Runtime은 W4/W8를 관리하며 NativeSingletonFitter에는4/8 지원 가정이 있다. L5/6/7에 native target fit을 호출하여 repair를 구현하지 않는다. Native L4 writer만 재사용하고, repair는 `model.layers.{l}.mlp.down_proj.weight` 선택과 functional/physical forward 검사를 일반화한다.

모든 후보에 공통 S64 forward·gradient를 재사용할 수 있는 부분은 묶되, 여러 층 gradient를 얻는 것과 여러 층을 동시에 수정하는 것은 구분한다. 층별 JVP/곡률·메모리·copy 비용을 별도로 계측한다. Repair에는 별도 native P_l/M_l/compute_z가 필수인 것이 아니다.

이 문서의 성과는 layer8 고정 근거의 부재 확인과 가능한 검증 절차의 구체화다. 실제 L5–L12 repair 성공률·best layer·장기 capacity 숫자는 아직 없다.

실행 사양: [구조화 계약](2026-09-17-repair-layer-feasibility-contract-v1.json), [18개 비교 셀](2026-09-17-repair-layer-feasibility-cells-v1.csv). 계약 일관성 검사는 모델 성능 검증과 구분한다.
