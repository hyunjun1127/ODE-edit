# EN 파이프라인 수식 해설: single-layer의 역할과 lifelong 보호의 범위

작성: 2026-09-18. 기존 [상세 설계 v1](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-single-layer-edit-preserving-correction-design-v1.md)의 동반 해설이다. 기존 실행 계약·수치 기준·arm을 변경하지 않는다. 아래 수식은 방법과 한계를 설명하며, 실제 모델에서 성능이나 numerical parity가 검증됐다는 뜻은 아니다.

관련 문서: [방법 포지셔닝](/mnt/raid5/janghj/layer_allocation/07_single_layer_method_positioning.md), [계산량·경량화 제안](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-single-layer-edit-preserving-correction-efficiency-review-v1.md).

## 1. 전체 흐름과 기호

**현재 batch는 L4-only로 먼저 편집한다. 그 결과가 실제로 만든 편집 반응을 유지하는 방향 안에서, 같은 L4 weight를 추가 보정한다.**

```math
W_{t-1}
\xrightarrow{\text{L4 local-z + native write}}
W_{N,t}=W_{t-1}+\Delta W_{N,t}
\xrightarrow{\text{correction + acceptance}}
W_t=W_{N,t}+D_t.
```

|기호|정의|
|---|---|
|$`W\in\mathbb R^{d_o\times d_i}`$|수정하는 L4 down-projection weight 하나. 여기서는 $`d_o=4096,d_i=14336`$|
|$`W_0`$|pre-edit L4 weight. 다른 모든 parameter는 전체 과정에서 고정|
|$`p_W`$|L4 weight만 $`W`$로 설정한 전체 모델의 출력 분포|
|$`t`$|현재 편집 batch 번호|
|$`W_{t-1}`$|현재 batch 진입 weight. 첫 batch는 $`W_0`$|
|$`W_{N,t}`$|현재 batch의 native L4-only write 직후 weight|
|$`\Delta W_{N,t}`$|native write의 실제 weight 변화|
|$`K_{E,t}\in\mathbb R^{d_i\times n_t}`$|현재 보호 sequence의 모든 유효 token에서 얻은 input key|
|$`D_t`$|최종 수용된 correction. fallback이면 0|
|$`P_*`$|native projector의 허용 range를 나타내는 correction용 직교 projector|

행렬 곱은 column-vector 규약이다. 구현이 token-major tensor를 사용하면 전치해 적용한다. $`W`$는 모델 전체 parameter가 아니라 선택한 행렬 하나다.

## 2. Local-z는 언제 계산하고 무엇을 고정하는가

현재 요청 집합을 $`\mathcal E_t`$, native history 상태를 $`M_{t-1}`$라고 하면 첫 단계는 다음처럼 쓸 수 있다.

```math
Z_t=\mathrm{NativeZOpt}_{\ell=4}(W_{t-1},\mathcal E_t),
```
```math
\Delta W_{N,t}
=\mathrm{NativeWrite}_4
(W_{t-1},Z_t,K_t^{\mathrm{write}},P_{\mathrm{raw}},M_{t-1}).
```

이는 기존 local-z optimizer와 native writer를 나타내는 연산자 표기다. 유한 step으로 계산한 z를 전역 최적해라고 가정하지 않으며, 기존 writer를 새 closed-form 식으로 대체하지 않는다. $`K_t^{\mathrm{write}}`$는 native write에 쓰이는 key이고, correction의 full-token 보호 집합 $`K_{E,t}`$와 구별한다.

Correction 단계에서는 $`Z_t`$를 재최적화하지 않는다. 하지만 **z 고정 자체가 편집 반응 보존 조건은 아니다.** Native activation target과 down-projection 출력은 동일한 객체라고 가정할 수 없으며, 실제 write의 target 실현 오차도 있을 수 있다. 따라서 $`W_{N,t}K_{E,t}=Z_t`$라고 놓지 않는다.

보호할 대상은 $`W_{N,t}`$가 실제로 만든 반응이다. 기존 L4-only의 강한 편집 성능을 먼저 확보하고 이 결과를 기준점으로 쓰는 것이 single-layer의 경험적 이점이다.

## 3. Single layer이기 때문에 얻는 고정 key와 선형 변화식

고정 token sequence $`x`$의 위치 $`j`$에서 L4 down-projection 입력을 $`k_j(x)`$, 그 모듈 출력을 $`u_W(x,j)`$라고 하자. 다른 parameter가 고정이고 이 행렬 앞의 계산에 변화가 없으므로

```math
k_j(x;W)=k_j(x),\qquad
u_W(x,j)=Wk_j(x).
```

따라서 유한 크기의 correction에도

```math
u_{W+D}(x,j)-u_W(x,j)=Dk_j(x)
```

가 성립한다. 이는 전체 모델을 1차 근사한 식이 아니라 해당 선형 모듈의 정확한 변화식이다. 고정된 bias가 있다면 차분에서 상쇄된다.

현재 보호 sequence 집합 $`\mathcal X_{E,t}`$의 모든 유효 token key를 모으면

```math
K_{E,t}=\mathrm{concat}_{x\in\mathcal X_{E,t},\ j\in\mathrm{valid}(x)}k_j(x).
```

Native rewrite contexts의 new/old teacher-forcing 경로 및 canonical 포함 규약은 상세 설계 §4를 따른다. 별도 paraphrase target은 추가하지 않는다.

```math
DK_{E,t}=0
\quad\Longrightarrow\quad
(W_{N,t}+D)K_{E,t}=W_{N,t}K_{E,t}.
```

모든 보호 token에서 해당 모듈 출력이 같고 다른 parameter도 같으므로, 정확 산술과 deterministic 계산 아래 그 sequence의 downstream logits도 같다. 이는 보호된 고정 sequence의 보장이며, 임의의 paraphrase·자유 생성 경로 전체의 보장이 아니다.

여러 layer를 동시에 수정하면 앞 layer 변화가 뒤 layer의 key를 바꿀 수 있다. 그 경우 고정 key 식들을 단순히 독립 적용할 수 없다. 다층 보존이 불가능하다는 주장은 아니며, 현재 단일 행렬 구조에서 제약이 간단해진다는 뜻이다.

## 4. 보정 공간과 실제 업데이트 방향

현재 correction이 속해야 하는 공간은

```math
\mathcal D_t
=\{D\in\mathbb R^{d_o\times d_i}:D=DP_*,\ DK_{E,t}=0\}.
```

첫 조건은 native preservation projector의 허용 공간을 따르고, 둘째 조건은 현재 편집 반응을 유지한다. 두 projector를 임의 순서로 곱하는 것으로 교집합을 대신하지 않는다.

$`P_*=VV^\top`$, $`V^\top V=I_r`$, $`r=\mathrm{rank}(P_*)`$이고 $`J_t=V^\top K_{E,t}`$라 두면

```math
Q_t=V(I_r-J_tJ_t^\dagger)V^\top.
```

$`\dagger`$는 Moore–Penrose pseudoinverse다. 정확 산술에서는

```math
Q_t^\top=Q_t,\quad Q_t^2=Q_t,\quad
Q_tK_{E,t}=0,\quad Q_tP_*=Q_t.
```

따라서 임의의 gradient $`G_t`$를 $`G_tQ_t`$로 투영하면 합법적인 보정 방향을 얻는다. Key 방향의 잔여 차원은

```math
q_t=r-\mathrm{rank}(V^\top K_{E,t}),
\qquad \dim\mathcal D_t=d_o q_t.
```

$`q_t>0`$이 실제 locality 개선을 보장하지는 않는다. 유용한 gradient가 이 공간에 남아 있어야 한다. Rank 판정과 FP32 적용의 수치 기준은 원 설계 §5·§8을 따른다.

## 5. 원지식 목적함수와 한 번의 correction

$`\mathcal S`$는 S64, $`\mathcal I_x`$는 문서당 지정된 128개 logit 위치다. 문서별 평균을 먼저 취하는 preservation loss는

```math
\mathcal L_{\mathrm{pres}}(W)
=\frac{1}{|\mathcal S|}\sum_{x\in\mathcal S}
\frac{1}{|\mathcal I_x|}\sum_{j\in\mathcal I_x}
\sum_v p_{W_0}(v\mid x_{\le j})
\log\frac{p_{W_0}(v\mid x_{\le j})}{p_W(v\mid x_{\le j})}.
```

Vocabulary 전체를 합산하고 teacher는 $`W_0`$로 고정한다. 현재 설계가 지향하는 제약 문제는

```math
\min_{D\in\mathcal D_t}\mathcal L_{\mathrm{pres}}(W_{N,t}+D)
```

이다. 실제 EN-F는 이 문제의 전역해를 구하지 않고, 한 번의 projected gradient와 제한된 후보 탐색을 수행한다.

```math
G_t=\nabla_W\mathcal L_{\mathrm{pres}}(W_{N,t}),\qquad
H_t=G_tQ_t,\qquad
\chi_t=\langle G_t,H_t\rangle_F=\|H_t\|_F^2.
```

$`\chi_t>0`$이면 기존 설계의 초기 step과 후보는

```math
\eta_0=\frac{\mathcal L_{\mathrm{pres}}(W_{N,t})}{\chi_t},\qquad
\eta_j=\eta_0 2^{-j},\qquad D_{t,j}^{\mathrm{ideal}}=-\eta_jH_t.
```

원형 EN-F는 $`j=0,\ldots,7`$이다. KL의 하한 0을 쓰지만, 편집 제약 아래 0에 도달 가능하다는 뜻은 아니다. 경량화 문서의 2회 후보는 별도 제안이며 기존 계약에 적용된 변경이 아니다.

실제 FP32 후보와 실제 변화량은

```math
W_{t,j}^{\mathrm{cand}}=\mathrm{FP32}(W_{N,t}+D_{t,j}^{\mathrm{ideal}}),\qquad
D_{t,j}^{\mathrm{actual}}=\mathrm{FP64}(W_{t,j}^{\mathrm{cand}})-\mathrm{FP64}(W_{N,t}).
```

따라서 이상적인 $`DK_E=0`$만 확인하지 않고 actual correction 및 보호 logits도 검사한다. Armijo 조건은 실제 이동을 기준으로

```math
a_j=\langle G_t,D_{t,j}^{\mathrm{actual}}\rangle_F<0,\qquad
\mathcal L_{\mathrm{pres}}(W_{t,j}^{\mathrm{cand}})
\le\mathcal L_{\mathrm{pres}}(W_{N,t})+10^{-4}a_j
```

이며, 수치 오차보다 분명한 감소와 편집 보호 검사도 요구한다. 무한 값·rank 불확실성·noise floor 등 예외 처리는 원 설계를 따른다.

## 6. 현재·과거 검사와 최종 수용

$`\mathcal C_t`$는 현재 검사 sequence, $`\mathcal H_t`$는 Past64 표본이다. 대표적인 NLL 조건은 각 요청·sequence별로

```math
\ell_i(W_{t,j}^{\mathrm{cand}})
\le\ell_i(W_{N,t})+10^{-4},
\qquad i\in\mathcal C_t\cup\mathcal H_t
```

이다. 평균 조건이 아니며 canonical preference 및 TF-strict 성공 ID 유지 조건도 함께 적용한다. EN-F는 전체 보호 logits 등 추가 invariant 검사도 통과해야 한다.

```math
W_t=
\begin{cases}
W_{t,j}^{\mathrm{cand}},&\text{accepted},\\
W_{N,t},&\text{fallback}.
\end{cases}
```

위 식의 accepted는 objective·수치·품질 조건을 모두 통과한 후보를 수용하는 경우, fallback은 정상적인 탐색 실패로 native 결과를 유지하는 경우다. 기술 오류를 정상 fallback으로 감추지는 않는다. 다음 batch는 이 $`W_t`$에서 시작한다. 공식 P/N은 후보 선택 이후 observer이며 gradient·step 선택에는 쓰지 않는다.

## 7. Horizon이 쌓이면 무엇이 보호되지 않는가

현재 exact 조건은 $`D_tK_{E,t}=0`$이다. 과거 batch $`s<t`$에 대해서는 일반적으로

```math
D_tK_{E,s}\ne0
```

일 수 있다. 과거 고정 sequence의 L4 출력 변화를 실제 수용된 increment로 분해하면

```math
(W_T-W_s)K_{E,s}
=\sum_{t=s+1}^{T}\Delta W_{N,t}K_{E,s}
+\sum_{t=s+1}^{T}D_tK_{E,s}.
```

첫 항은 이후 native 편집의 변화, 둘째 항은 이후 correction의 추가 변화다. 이는 해당 모듈 출력의 정확한 분해이지, 비선형인 최종 NLL·RS 변화의 가산 분해는 아니다. 누적 변화가 상쇄될 수도 있으므로 단조 망각을 단정할 수 없지만, current-only lock으로 둘째 항을 0이라고 보장할 수도 없다.

Past64는 모든 과거 요청이 아니다. 또한 검사 기준이 $`W_{N,t}`$이므로 correction의 추가 손상을 제한할 뿐, 그 batch의 native 단계에서 이미 발생한 망각을 복원하지 않는다. 이전 correction으로 trajectory가 달라졌다면 native fallback 역시 독립 N4 chain으로 돌아간다는 뜻이 아니다.

$`W_0`$ teacher는 과거의 새 edit target을 모른다. Calibration 입력에서 과거 편집과 원지식 복원이 충돌하면 $`\mathcal L_{\mathrm{pres}}`$ 감소가 과거 편집의 약화와 함께 나타날 수 있다. 모든 과거 편집에서 반드시 발생한다는 주장은 아니지만, 현재 목적함수로 배제되지 않는다.

만약 history key까지 누적 exact lock한다면

```math
K_{\le t}=[K_{E,1},\ldots,K_{E,t}],\qquad
q_{\le t}=r-\mathrm{rank}(V^\top K_{\le t})
```

이고, $`P_*`$가 고정된 이상화된 누적 집합에서는

```math
q_{\le t+1}\le q_{\le t}.
```

제약이 차면 $`q_{\le t}=0`$이 되어 보정 자유도가 사라질 수 있다. 최신 active target만 남기거나 제약을 제거하면 단조성 조건 자체가 달라진다. **누적 exact lock은 현재 v1에 적용된 방법이 아니라 capacity 진단·후속 설계의 대상이다.**

또한 이미 발생한 망각을 되돌리는 조건은 단순한 $`DK_H=0`$과 다르다. 과거 수용 시점의 출력을 $`H_{\mathrm{acc}}`$라 두면 복원에는

```math
DK_H=H_{\mathrm{acc}}-W_{N,t}K_H
```

같은 비동차 조건이 필요하고, 현재 편집과 충돌해 feasible하지 않을 수 있다. 이것도 현재 v1이 해결했다고 주장하는 부분이 아니다.

## 8. 이 설계에서 single layer의 역할과 미검증 질문

|역할|수식·파이프라인에서의 위치|한계|
|---|---|---|
|강한 편집 기준점|$`W_{N,t}=W_{t-1}+\Delta W_{N,t}`$|모든 요청에서 완벽한 edit을 보장하지 않음|
|고정 input key|$`u_{W+D}-u_W=Dk`$|고정 token 입력 및 다른 parameter 고정 조건 필요|
|현재 edit과 보정 방향 분리|$`DK_{E,t}=0`$, $`G_tQ_t`$|미관측 PS와 모든 과거 edit에는 동일 보장이 없음|
|일부 계산 재사용|key·upstream·teacher cache|L5 이후 forward/backward 비용은 남음|

주 가설은 **L4-only의 실제 편집 반응을 유지하는 공간에, 원지식 손상을 줄일 유효 방향이 남아 있는가**다. Lifelong 방법으로서의 추가 질문은 **그 보정이 과거 편집을 손상하지 않으면서 horizon이 길어져도 유용한가**다. 현재 수식과 설계는 첫 질문의 공간을 정의하지만, 두 번째 질문의 해결이나 실증 성능을 보장하지 않는다.
