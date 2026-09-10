# BLUE(L4-only) 기반 단일-layer 진척 보존 barrier 진단 실험

작성: 2026-09-11 KST. 상태: **설계 작성·독립 검토 완료 — 모델/GPU 실험 미실행**.

## 1. 실험 질문과 범위

**강한 BLUE(L4-only) 편집을 이미 성립시킨 동일 모델에서, native가 제공한 입력 공간을 유지하면서 편집 진척을 보존하는 보정으로 누적 손상을 줄일 수 있는가?**

주 방법은 `BLUE(L4-only) native write → 동일 native 입력 basis의 actual-weight refinement → 진척 보존형 soft barrier`이다. Native endpoint를 정확히 공통 시작점으로 사용한다. 처음부터 barrier만으로 새 편집을 성립시키는 실험이나, 여러 layer에 residual을 배분하는 실험은 이번 주 비교에 포함하지 않는다. 따라서 긍정 결과도 **single-layer의 native-assisted refinement 가능성**을 지지하며, 일반적인 single-layer capacity 충분성이나 전체 lifelong 안정성을 증명하지 않는다.

사용자의 이전 지시를 반영해 성능 통과를 다음 비교의 실행 조건으로 만들지 않는다. Risk 증가, slack, barrier 비활성, edit 약화, 작은 효과를 모두 결과로 남긴다. 실행 중 성능을 보고 cap·step 수·basis·observer를 바꾸지 않는다. 본 문서는 실행 설계이며 이번 작성 작업에서 GPU 작업을 제출하지 않는다.

이 문서가 9월 10일 설계에서 바꾸는 핵심은 다음과 같다.

- `EP-Free`의 **기존 free 성분 삭제** 대신, 편집 진척을 유지하는 공간에 **새 보정 방향도 허용하는 soft barrier**를 주 방법으로 삼는다.
- `||보정 후 step|| ≤ ||nominal step||` 조건을 제거한다. 같은 진척을 만드는 최소 metric-norm 방향에서 출발하면 이 조건이 모든 유용한 방향 변경을 막을 수 있다.
- 평균 edit 진척 하나를 주 observer로 사용한다. 네 group 보호는 작은 보조 비교로 둔다.
- 한 layer의 native update에 scalar 하나만 곱하지 않는다. Native 입력 basis 안의 **전체 output coefficient 행렬**을 움직인다.
- Native target, key, projector, history와 basis를 inner loop에서 재계산하지 않는다. 현재 실제 출력 gradient와 누적 risk gradient만 갱신한다.

## 2. 기존 근거와 이번에 분리할 가설

단일-layer ABC에서 같은 Fixed100의 RS는 Early/Middle/Late 진입점에서 100/100/99%였지만, 같은 주변 지식의 NS는 82.8/69.3/66.2%였다. 편집 성공의 포화가 원래 지식의 보존을 뜻하지 않는다. Native Current RS도 세 entry 모두 100%이므로 RS만을 성공 기준으로 쓰면 안 된다. [ABC 사실 보고서][abc-facts]

기존 BGODE-FBP는 combined-model 출력에서 barrier를 관측했지만, risk-only projection으로 locality drift와 함께 편집 강도도 크게 줄였다. Llama rewrite exact는 Native 8/8에서 barrier N2/N4 모두 0/8이었다. 따라서 **risk가 감소한 것과 필요한 편집 진척을 유지한 것을 분리**해야 한다. [FBP 결과][fbp]

JV의 공동 response·반복 후보 갱신, `ode_alloc`의 실제 출력 coefficient gradient, P1R52 Phase 3의 target refresh는 이미 repo에 존재한다. 이번의 기여를 ODE·feedback·projection의 최초 도입으로 설명하지 않는다. CAKE와 BLUE의 여러-layer 배분 비교도 이번 단일-layer 진단에서는 주 질문이 아니다. [JV 감사][jv], [기존 coefficient gradient][alloc], [target refresh 실험][phase3]

검증할 가설은 세 가지다.

| 가설 | 직접 비교 | 결과를 읽는 기준 |
|---|---|---|
| Native support 안에 유용한 보정 여력이 있다 | H 대 N | 추가 비용과 함께 실제 edit–locality 조합이 개선되는가 |
| Risk-only 보정은 편집 진척도 깎을 수 있다 | R 대 H, 같은-state matched-risk probe | 같은 위험 감소를 얻을 때 edit 방향을 얼마나 바꾸는가 |
| 진척 보존형 보정이 그 trade-off를 개선한다 | EP 대 H/R | 일차 보호가 실제 NLL·PS와 원래 지식 보존에도 유용한가 |

`H`는 native metric을 사용하는 barrier-off refinement, `R`은 같은 metric의 risk-only soft barrier, `EP`는 edit-progress equality를 추가한 같은 soft barrier다. 세 방법의 기본 loss·시작점·basis·nominal field는 동일하다.

## 3. 모델, checkpoint, 평가 패널

모델: Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`. 수정 weight는 `model.layers.4.mlp.down_proj.weight`, shape `[4096,14336]` 하나다. 모델·actual write는 기존 FULL-FP32 경로를 사용한다.

| Entry | 편집 전 모델/history | 현재 batch | Current ordinal, 0-based | Prepared 경로 — 아래 artifact root 기준 |
|---|---|---|---|---|
| Early | W10/M10 | B11 | 1000–1099 | `A/Early/native-r4/prepared.pt` |
| Middle | W50/M50 | B51 | 5000–5099 | `A/Middle/native-r1/prepared.pt` |
| Late | W90/M90 | B91 | 9000–9099 | `A/Late/native-r4/prepared.pt` |

Artifact root: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1`.

이 root의 `imports/entries/B010`, `B050`, `B090`에 `W-method-state.pt`, `B011`, `B051`, `B091`에 `native-targets.pt`가 있다. 이번 문서 작성 시 세 checkpoint, target, prepared 파일의 로컬 존재를 확인했다. 새 GPU 복원이나 tensor 일치 검사를 수행한 것은 아니다. 원격 checkpoint 재수집이나 W0→W90 재실행은 기본 준비에 필요하지 않다.

- **Current100:** 위 자연 next batch.
- **Fixed100:** B1, ordinal 0–99. 세 entry에서 같은 100개다.
- **Past100:** `[100,current_start)`를 네 구간으로 나누어 각 25개를 고정 hash로 뽑은 기존 패널. 특정 과거 batch 하나가 아니다.
- 정확한 request/neighbor 목록은 artifact root의 `input.lock.json` 및 기존 `panel-metadata.csv`를 재사용한다. 현재 결과를 보고 다시 뽑지 않는다.

Full은 패널별 RS100/PS200/NS1000, 세 패널 합계 **3900 prompt pairs**다. Curve는 RS300, Current PS200, 세 패널의 neighbor 각 200으로 **1100 pairs**다. 중간 Fixed/Past PS는 Curve에 없다. [패널 목록][panels]

| Native 기준 | Early | Middle | Late |
|---|---:|---:|---:|
| Current RS / PS / NS, % | 100 / 97.5 / 81.3 | 100 / 98.5 / 71.1 | 100 / 96.5 / 62.6 |
| Six-context train NLL | 0.004195 | 0.047468 | 0.023872 |
| Native update Frobenius norm | 8.770752 | 10.744017 | 11.466632 |
| Native history+L2 action | 82.639404 | 136.660172 | 177.651993 |

이 값은 기존 실행 관측이다. Train NLL과 canonical rewrite NLL은 다른 측정이다. Middle canonical rewrite NLL 약 0.02667을 위 0.047468 대신 쓰지 않는다.

## 4. Native baseline을 보존하는 경계

기호는 다음과 같다.

- `W0`: 원래 pretrained L4 weight. Cumulative risk의 reference.
- `We`: 현재 batch 편집 전 weight. Essence teacher와 전체 batch action의 reference.
- `WN`: BLUE(L4-only)가 실제로 만든 native endpoint.
- `ΔN = WN − We`: native batch update.
- `M`: 현재 batch 이전 history. `MN`은 현재 batch가 한 번 append된 다음 history.
- `K`: native가 사용한 request별 key `[14336,100]`.
- `U = Ub`: native input direction의 orthonormal basis `[14336,r]`. 세 fixture의 관측 rank는 `r=100`이다.

Native 설정은 `blue=true`, `layers=[4]`, `L2=1`, `z_steps=25`, `z_lr=.1`, `z_decay=.5`, `clamp=.75`, `loss_layer=31`, `KL=.0625`, `subject_last`를 유지한다. Cached target 100개를 재사용하며 주 진단에서 새 `compute_z` 호출은 0이다. 25는 기존 구현상 최대 target forward 수이며 실제 Adam update는 최대 24회다. [Native 설정][native-config]

Context는 clean 1개와 generated 5개다. **Native K는 clean 1/2, generated 각각 1/10**인 context-type 평균이고, actual edit loss는 여섯 context의 **flat 1/6 평균**이다. 이를 통일한다는 이유로 K나 loss를 바꾸지 않는다. [Native key reduction][native-keys]

Native target은 초기 방향과 baseline의 provenance다. Refinement 동안 target을 activation에 재주입하지 않으며, target fitting loss를 actual task loss로 오인하지 않는다.

N은 정확한 WN 자체다. `We + (ΔN U)Uᵀ`로 재구성한 근사 모델을 baseline 또는 공통 시작점으로 사용하지 않는다. Native wrapper를 inner step마다 호출하면 복원·target cache·history append가 섞이므로 금지한다.

## 5. 한 layer에서 움직일 공간

\[
W(X)=W_N+XU^\top,\qquad X_0=0,\qquad X\in\mathbb R^{4096\times r}.
\]

주 공간은 409,600개의 coefficient를 갖는다. `W = WN + c ΔN`의 scalar amplitude 조절과 다르다. Scalar 하나라면 edit derivative가 0이 아닌 상태에서 진척 equality를 걸 때 보정이 0으로 제한된다.

Full null-space Q의 rank14326 전체로 공간을 넓히지 않는다. 이 실험은 **같은 native 입력 공간에서 output 방향을 조정하는 가능성**을 확인한다. 이 공간에서 실패해도 L4 전체 또는 모델 전체의 capacity 불가능을 증명하지 않는다.

U는 prepared `Ub`를 그대로 사용한다. 생성 의미는 native right factor `S = [P(KKᵀ+M)+I]⁻¹PK`의 column space를 orthonormal하게 표현한 것이다. `orth(K)`나 `orth(PK)`로 바꾸면 history metric이 회전시킨 공간을 바꿀 수 있으므로 대체하지 않는다.

L4 down-projection만 수정하고 teacher-forced 입력이 고정되어 있으므로 이 module의 입력 key는 변하지 않는다. K/U/P/M은 inner loop에서 고정한다. Downstream의 실제 출력·loss·gradient는 바뀌므로 매 step 다시 계산한다. 여러 write layer 사이의 key drift는 이번에 제거되는 변수지만, token·context 간 weight 공유와 downstream 비선형성은 남는다.

## 6. 모든 refinement 방법의 공통 loss와 native metric

\[
L(X)=L_E(W(X))+0.0625L_{\mathrm{essence}}(W(X);W_e)
+0.1\,J_N(\Delta_N+XU^\top)/j_N,
\]
\[
J_N(\Delta)=\operatorname{tr}[\Delta(M+I)\Delta^\top],\qquad j_N=J_N(\Delta_N).
\]

`L_E`는 request 평균 × 여섯 context 평균 × target-token 평균 NLL이다. Essence는 기존 `{subject} is a` 입력과 **KL(student || We teacher)**, 기존 reduction을 유지한다. Held-out paraphrase/neighborhood와 과거 평가 정답은 controller loss에 넣지 않는다.

\[
H_R=U^\top(M+I)U,\quad C_N=\Delta_N(M+I)U,\quad Z=U^\top K,
\]
\[
J_N(\Delta_N+XU^\top)=j_N+2\langle C_N,X\rangle+
\operatorname{tr}(XH_RX^\top),\quad H_N=ZZ^\top+H_R.
\]

Native 초기 write까지 포함한 **We 기준 전체 action**을 정규화한다. WN에서 refinement coefficient가 0이라는 이유로 penalty를 0으로 reset하지 않는다. `H_R`는 loss의 보존 비용, `H_N`은 방향을 계산할 metric이다. H_N을 actual-output Hessian이나 Fisher라고 부르지 않는다.

기존 helper의 `Uᵀ M U + I_r`는 U가 orthonormal이라는 표현을 사용한다. 이번 공통 엔진은 저장된 U의 Gram까지 사용한 위 식으로 물리적 action을 계산하고, 기존 helper와의 작은 차이를 기록한다. 모든 새 arm에 동일하게 적용하며 N의 WN은 바꾸지 않는다.

`MN`에는 KKᵀ가 이미 한 번 들어 있으므로 M 대신 MN을 쓰거나 KKᵀ를 두 번 더하지 않는다. 기존의 일반적으로 비대칭인 `P(KKᵀ+M)+I`를 SPD Hessian으로 취급하지 않고, 허용 basis 안의 H_N을 별도로 구성한다.

Actual gradient는 모든 token에 실제 weight W(X)가 적용된 forward에서 얻는다. 기존 `AffineForward`/`DirectObjective`의 기능을 재사용하되 anchor를 WN, teacher를 We, action cross term을 C_N으로 명시한다. [기존 objective][objective]

## 7. Nominal flow와 step 예산

\[
G_s=\nabla_X L(X_s),\qquad v_{0,s}=-\nu G_sH_N^{-1}.
\]

같은 entry의 모든 arm은 같은 ν를 사용한다. `ν`는 공통 X=0에서 한 번 정한다.

\[
\nu=\frac{0.08\|\Delta_N\|_F}
{\|(G_0H_N^{-1})U^\top\|_F}.
\]

주 horizon은 `T=1`, `N=8`, `h=1/8`이며 X의 update는 `X_next = X + h v`다. 따라서 첫 **barrier 전 nominal displacement**는 native norm의 1%다. 보정 후 norm을 다시 맞추지 않는다. Momentum, Adam, optimizer-state carryover, step별 gradient normalization은 사용하지 않는다.

`H_N`을 c배 재표현하면 ν도 c배 되어 같은 physical nominal field가 된다. Effective metric은 `B_right = H_N/ν`이고, vector 표기의 B는 coefficient 행마다 이 metric을 적용한다.

\[
\|v\|_B^2=\operatorname{tr}(vB_{\mathrm{right}}v^\top),\qquad
v_0=-B^{-1}g_{\mathrm{total}}.
\]

이하의 controller는 **velocity**에 대해 쓴다. Barrier 우변에 h를 또 넣지 않는다. N16 감도 실험도 T, ν, B, barrier gain, elasticity를 그대로 두고 h만 1/16로 바꾼다.

ΔN 또는 초기 nominal gradient가 정확히 0이면 강제 길이를 만들지 않고 stationary/no-native-action branch로 기록한다. 이번 fixture의 ΔN과 j_N은 양수다. 수치 재계산에서 비정상값이 나오면 입력·정밀도 문제인지 먼저 확인한다.

## 8. 편집 진척 observer

주 observer는 **edit NLL만**의 gradient다.

\[
j_s=\operatorname{vec}(\nabla_X L_E(X_s))^\top,\qquad
j_s(v_s-v_{0,s})=0.
\]

Total objective의 gradient를 보호하지 않는다. 그러면 edit NLL 악화를 KL/action 감소로 상쇄할 수 있기 때문이다. 이 equality는 각 현재 상태에서 nominal의 평균 NLL 변화율을 유지한다. 개별 요청·PS·과거 edit의 유지나 두 trajectory의 동일 endpoint를 보장하지 않는다.

`j v0 > 0`이면 nominal부터 edit NLL을 높이는 상태다. EP는 그 변화율도 유지하므로 이를 ‘편집을 진전시켰다’고 보고하지 않는다. 이것은 삭제할 결과가 아니라 nominal objective의 중요한 관측이다.

요청을 case-ID hash `L4-EP-GROUP-V1|case_id`로 정렬해 25개씩 네 group으로 고정한다. 모든 arm에서 group loss/gradient를 같은 방식으로 모으고 평균으로 j를 만든다. Middle 보조 arm만 네 row J4를 모두 보호한다. 성능이나 loss를 보고 group을 재구성하지 않는다.

## 9. 단일 cumulative covariance barrier

\[
D(X)=W_N-W_0+XU^\top,\qquad
R_C(X)=\tfrac12\operatorname{tr}[D(X)C_0D(X)^\top].
\]

C0는 기존 native second moment를 재사용한다. Raw 파일은 `/mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz`다. 기존 source와 동일하게 Torch FP32 `mom2.mom2 / mom2.count`로 읽고, reduced 연산은 FP64로 준비한다. [기존 covariance 복원][runtime]

\[
s_F=\tfrac12\|\Delta_N\|_F^2,\quad
r(X)=\{R_C(X)-R_C(0)\}/s_F,\quad h_B(X)=-r(X).
\]

Cap은 **R_C(WN)**으로 batch의 refinement 동안 고정한다. Risk의 중심은 계속 **W0**다. s_F는 양수인 수치·보고 단위일 뿐, covariance risk를 Frobenius risk로 바꾸지 않는다. 이 cap은 native 이후 추가 증가를 줄이려는 기준이며, native가 이미 만든 모든 손상을 복구하라는 조건이 아니다. 매 batch 새 RN을 사용하는 확장도 하나의 고정 lifelong cap이라고 부르지 않는다.

\[
H_C=U^\top C_0U,\quad C_C=(W_N-W_0)C_0U,
\]
\[
r(X)=\{\langle C_C,X\rangle+\tfrac12\operatorname{tr}(XH_CX^\top)\}/s_F,
\quad a(X)=\nabla_Xr=(C_C+XH_C)/s_F.
\]

Risk와 gradient는 작은 이차식으로 계산하므로 추가 model backward가 필요 없다. 실제 displacement d의 risk 변화는

\[
r(X+d)-r(X)=\langle a(X),d\rangle+
\tfrac1{2s_F}\operatorname{tr}(dH_Cd^\top)
\]

로 정확히 분해된다. Actual FP32 materialized weight의 risk도 계산해 reduced 표현과의 차이를 기록한다.

이 risk는 원래 지식의 기능적 손상의 proxy다. P가 허용한 공간에서 risk가 거의 보이지 않을 수도 있고, risk를 줄여도 locality가 좋아지지 않을 수 있다. 새로운 external reference bank나 여러 risk의 조합을 주 실험에 추가하지 않는다.

## 10. 주 controller: 진척 보존형 elastic barrier의 closed form

κ=2를 고정한다. EP는 각 현재 state에서 다음 문제를 푼다.

\[
\begin{aligned}
\min_{v,\xi\ge0}\quad&\tfrac12\|v-v_0\|_B^2+\tfrac{\xi^2}{2\varepsilon}\\
\text{s.t.}\quad&J(v-v_0)=0,\\
&a^\top v\le\kappa h_B+\xi.
\end{aligned}
\]

주 arm의 J는 j 한 row, J4 보조는 네 row다. B는 위 native effective metric이다. Risk slack만 soft하게 허용하고, 별도의 step-norm cap·request별 loss ceiling·rollback·line search·성공 threshold를 붙이지 않는다.

\[
\mathcal P_B=B^{-1}-B^{-1}J^\top(JB^{-1}J^\top)^\dagger JB^{-1},
\]
\[
w=\mathcal P_B a,\quad q=a^\top w,\quad
e=a^\top v_0-\kappa h_B,
\]
\[
\boxed{v_{EP}=v_0-\frac{[e]_+}{q+\varepsilon}w},\qquad
\boxed{\xi=\frac{\varepsilon[e]_+}{q+\varepsilon}}.
\]

`Jw=0`이므로 선택한 nominal 진척을 일차적으로 유지한다. `q=0`이면 보정은 0이고 필요한 risk slack을 남긴다. 이는 현재 공간과 observer 아래에서 risk를 독립적으로 움직일 일차 여력이 없다는 뜻이며, 모델 전체의 capacity 한계를 뜻하지 않는다.

**Elasticity는 entry에서 한 번 보정하고 고정한다.** `q_all,0 = a0ᵀ B⁻¹ a0`로 두고, `H_c = H_C/s_F`, `λ_c = ||B_right^(-1/2) H_c B_right^(-1/2)||_2`, `d_ref = (1/8)v0,0`를 계산한다.

\[
q_{ref}=q_{all,0}+\lambda_c^2\|d_{ref}\|_B^2,\qquad
\varepsilon=0.1q_{ref}.
\]

두 번째 항은 기준 이동에서 생길 수 있는 risk-gradient 변화의 크기를 반영해, a0=0이지만 risk curvature가 있는 경우를 다룬다. Floating-point epsilon이 아니라 **고정 elasticity calibration**이다. N8/N16 모두 d_ref의 1/8을 유지한다. Risk의 보고 단위를 바꾸면 a/H_c는 같은 비율, q_ref/ε는 그 제곱 비율로 바뀌어 physical correction은 유지된다.

q_ref=0일 때는 stationary nominal과 실제 flat risk를 구분한다. Stationary 시작은 앞 절의 no-action으로 기록한다. Nonstationary인데 a0=0, H_c=0이면 이 support에서 risk가 상수이므로 R/EP는 H와 같게 두고 risk-blind 상태를 보고한다. 작은 q 때문에 임의의 큰 역수를 만드는 대신 FP64 native-whitened 공간에서 w와 q를 직접 계산한다.

이 식은 **현재 gradient의 local quadratic controller**의 closed form이다. 최종 nonlinear 편집 문제 전체를 closed form으로 풀었다는 뜻은 아니다. 실제 finite step에서는 edit curvature, 양의 risk 이차항, slack이 남으므로 hard safety나 finite-step progress equality를 주장하지 않는다.

## 11. 대조군과 고정 실행 목록

Risk-only R은 EP에서 edit equality만 제거한 같은 elastic 문제다.

\[
w_R=B^{-1}a,\quad q_{all}=a^\top w_R,\quad
v_R=v_0-\frac{[e]_+}{q_{all}+\varepsilon}w_R.
\]

Risk-only slack은 `ξ_R = ε[e]_+/(q_all+ε)`로 기록한다.

ε/κ/cap/nominal field를 EP와 공유한다. H는 v=v0, N은 추가 refinement가 없다.

| 실행 묶음 | 조건 | 새 trajectories / logical steps |
|---|---|---:|
| 주 비교 | Early/Middle/Late × H, R, EP; 각 N8/T1 | 9 / 72 |
| 기존 삭제 방식 비교 | Middle × EP-Free, N8/T1, J1 | 1 / 8 |
| 평균 observer 확인 | Middle × EP-J4, N8/T1 | 1 / 8 |
| 유한 step 확인 | Middle × EP, N16/T1, 동일 vector field | 1 / 16 |
| 합계 | N 세 endpoint는 정확한 기존 결과 재사용 | **12 / 104** |

EP-Free는 9월 10일의 선택적 삭제 식을 같은 geometry/nominal field에서 적용한다. `v_prog = B⁻¹Jᵀ(JB⁻¹Jᵀ)^†Jv0`, `v_free=v0−v_prog`라 하면

\[
v_{Free}=v_0-\frac{[a^\top v_{free}]_+}{q}w
\]

이며 q=0에서는 v0다. 이 arm은 새 보상 방향을 허용하는 EP와 기존 free 성분 삭제의 차이를 설명하는 보조다. 순수 edit gradient의 native steepest descent에는 삭제할 free 성분이 0일 수 있다. Free 성분이나 효과가 작다고 observer를 실행 중 바꾸지 않는다.

별도 Euclidean/native 2×2, 여러 risk 종류, α×κ×ε 전체 sweep, z refresh, Q 공간 확장은 이번 목록에 추가하지 않는다. Main 세 방법의 세 entry 비교는 Middle의 성능과 무관하게 모두 완료한다.

## 12. 같은-state 검증: 진척 보존과 제거량의 혼동 방지

**Nominal shadow.** EP 주 arm의 pre-state 0/3/7에서 같은 X의 nominal displacement `h v0`와 실제 `h vEP`를 비교한다. Pre0 nominal은 H 첫 step 관측을 재사용하고, pre3/pre7에서는 forward-only shadow 후 원상 복원한다. 세 entry 합계 새 edit-only forward sweep **6회**다. 다른 trajectory의 같은 step 번호를 같은 state라고 취급하지 않는다.

각 group/request의 다음 값을 저장한다.

- Algebraic leakage: `J(vEP−v0)`와 실제 적용 displacement에 대한 값.
- 실제 loss 변화와 그 일차 예측의 차이.
- 같은-state nominal endpoint 대비 보정 endpoint의 NLL 차이.
- 평균·median·p90·최대 악화, 개선/악화 요청 수. PS는 별도 held-out 결과다.

**Matched-risk 단발 probe.** R과 EP는 같은 ε에서도 제거하는 risk 양이 다를 수 있다. 공통 WN에서 EP가 만드는 일차 감소량 `t = aᵀ(v0−vEP)`를 계산하고

\[
v_{match}=v_0-(t/q_{all})B^{-1}a
\]

를 구한 뒤 공통 X=0에서 **X_probe=(1/8)v_match**를 적용한다. t는 velocity 기준 감소량이고, 실제 displacement의 일차 risk 감소량은 t/8로 EP의 N8 첫 step과 같다. q_all=0이면 t=0이고 v_match=v0다. 세 entry에서 각 하나, 총 **3개 단발 endpoint**를 공통 objective와 Curve로 평가한다. 추가 backward는 없다. EP 첫 step과 같은 일차 risk 감소량에서 실제 edit 차이를 보므로, 단순히 EP가 덜 줄였기 때문에 강도가 유지된 상황과 구분할 수 있다. 이 probe는 trajectory의 다음 상태로 사용하지 않는다.

## 13. 평가와 통계

| 상태 | 평가 |
|---|---|
| W0, We, WN | 정확한 기존 Full 결과 재사용; state/입력/평가 설정이 맞는지 확인 |
| N8의 step1/2/4 | Curve1100 |
| N8의 step8 | Full3900 + terminal train/risk; Curve는 Full에서 추출 |
| N16의 step2/4/8/16 | 위와 같은 t=1/8,1/4,1/2,1에 대응 |
| Matched-risk probe | Curve1100 + objective/risk |

RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이고 tie는 실패다. 주변 true NLL과 competing new NLL을 각각 보고하여, 원래 답이 약해진 것과 competing 답만 강해진 것을 구분한다. TF exact와 자유 생성은 candidate preference와 구분한다.

Middle N/H/R/EP의 terminal에는 기존 20 rewrite + 40 rephrase 생성 패널과 같은 decoding 설정을 사용한다. N의 동일 결과를 재사용하면 새 생성은 **180 sequences**다. 작은 패널의 literal/의미 관측을 전체 자유 생성 정확도로 확대하지 않는다.

주 결과는 `EP−H`, `EP−R`, 각 refinement의 `−N`이다. Current edit NLL/PS, Fixed/Past retention, 세 패널 locality를 별도로 제시한다. Training objective 감소·RS100·risk 감소를 하나의 성공 점수로 합치지 않는다.

실제 관측 snapshot의 NLL–NS, PS–NS, risk–NS 곡선을 그린다. Common strength 구간이 없으면 그대로 보고하며, 측정하지 않은 weight의 NS를 보간해 strength-matched 결과로 만들지 않는다. Terminal8은 고정 endpoint이며 가장 좋은 NS 시점을 사후 endpoint로 선택하지 않는다.

Request 단위로 rewrite/paraphrase/neighborhood와 모든 arm을 묶어 paired bootstrap 2000회를 수행한다. 같은 Fixed100의 세 entry 반복을 독립 300 requests로 합산하지 않는다. 세 entry는 history와 Current batch가 함께 달라 age-only 인과효과가 아니며, 기존 개발 패널의 재사용이므로 독립 benchmark 검증이라고 부르지 않는다.

## 14. 필수 진단과 해석

| 관측 | 해석 |
|---|---|
| R은 같은 risk 감소에서 NLL 진척을 깎고 EP는 더 잘 유지 | Progress-null 보정의 실용적 근거 |
| Algebraic leakage는 작지만 actual NLL 차이가 큼 | 유한 step/비선형성; N16와 같은-state shadow 확인 |
| Mean은 유지되지만 일부 요청 또는 PS 악화 | 평균 observer의 한계; Middle J4와 분포 확인 |
| EP-Free는 거의 움직이지 않고 EP는 유용한 보정 | 삭제만으로 부족하고 새 보상 방향이 필요한 가능성 |
| q/q_all이 작고 slack이 큼 | 현재 support에서 edit와 risk를 독립적으로 움직일 일차 여력이 작음 |
| Risk는 감소하나 locality는 개선되지 않음 | 구조적 proxy의 한계; capacity 실패로 대체 해석하지 않음 |
| Current는 유지되지만 Fixed/Past 악화 | 현재 edit 진척 보호가 과거 edit 보호를 대신하지 못함 |
| N16도 같은 성능·비용 관계 | 작은 step 또는 추가 feedback의 가치가 제한적 |
| H도 N보다 일관되게 나쁨 | Native 이후의 공통 refinement objective/공간/노출부터 재검토 |

기하 진단은 `H_N` spectrum, native/physical norm, `q_all`, `q`, `q/q_all`, free-energy 비중, correction norm, slack, cap 초과와 risk 이차항으로 제한한다. q/q_all이 작다는 이유로 해당 entry를 제외하지 않는다. Input rank100과 coefficient409600이라는 수치만으로 usable capacity가 충분하다고 주장하지 않는다.

## 15. 구현 재사용과 수치 확인

기존 ABC worktree의 `project/run_scripts/single_layer_cumulative_risk/`에서 preparation, contexts/panels, actual-weight objective, evaluation, structural measurement를 재사용한다. 실행 source는 별도의 task branch/worktree에서 기존 결과를 덮어쓰지 않는 방식으로 구현한다.

| 책임 범위 | 구체 변경 |
|---|---|
| 공통 gradient accumulation | Group edit gradient, 전체 edit 평균, 기존 essence, 전체 batch action을 같은 X에서 계산; edit backward 중복 방지 |
| Geometry | U Gram, H_R/Z/H_N, FP64 factorization, C_N/H_C/C_C와 고정 risk calibration |
| Controller | H/R/EP/EP-Free, J1/J4, 같은 field의 N8/N16, native-whitened closed form |
| Observer | 기존 panels + same-state shadow/matched-risk + 계산 ledger |

`evaluate()` 후 `group_edit_gradients()`를 그대로 다시 호출하지 않는다. 네 group을 한 번씩 처리해 네 edit gradient를 모으고 평균으로 전체 edit gradient를 만든 다음, essence와 analytic action gradient를 더한다. 모든 방법에 같은 group/microbatch/reduction 순서를 사용하며 group 사이에 X를 바꾸지 않는다.

Actual forward는 WN+XUᵀ weight가 모든 token에 적용되는 기존 경로를 우선 유지한다. 저랭크 hook 최적화를 도입한다면 cached activation이 아니라 현재 입력에 작용해야 하며, actual materialized weight의 출력과 차이를 확인한다. 이번 설계의 계산 절감 근거로 아직 구현하지 않은 batching 최적화를 가정하지 않는다.

B-whitened 공간에서 J의 row space를 구하고 a를 직교 투영하여 q를 제곱 norm으로 계산한다. 거대한 409600×409600 행렬은 만들지 않는다. 작은 H_N과 최대 네 row만 사용한다. FP64 row-normalized rank-revealing decomposition을 사용하고 relative singular-value tolerance `1e-10`을 실행 전에 고정한다. 0인 row는 0으로 기록하며, 버린 row에 대한 leakage도 함께 저장한다. 기존 Euclidean ridge `soft_filter()`로 exact native projection을 대체하지 않는다.

Edit gradient j=0이지만 G_total≠0이면 stationary 상태가 아니다. 이때 J1 조건만 비어 EP가 R과 같아진다. 작지만 0이 아닌 edit gradient는 row normalization으로 유지한다. G_total=0인 nominal stationary 상태, edit observer만 0인 상태, covariance risk가 flat인 상태를 서로 다른 진단값으로 기록한다.

실행 전 계산 확인은 두 묶음이다.

1. **공통 상태·적용 확인:** WN/We teacher/M/native assets/패널 identity, L4만 수정되는지, overlay/materialized 출력이 의도한 모델을 나타내는지.
2. **수학 확인:** 작은 합성 행렬에서 EP/R closed-form KKT, progress-null, q=0·rank-deficient J, 정확한 risk 이차 전개, N8/N16 field identity와 단위 재표현 불변성.

이는 계산이 의도대로 성립하는지 확인하는 작업이다. 성능·risk·slack 임계값을 scientific pass/fail gate로 만들지 않는다. NaN/Inf나 잘못된 적용은 실행 오류로 고치고, 유효하지만 나쁜 결과는 보존한다.

## 16. 계산 예산

기존 ABC의 RTX A6000, request microbatch2/context microbatch1에서 full logical gradient는 약 41초, terminal train forward 약 23초, Curve 약 42초, Full 약 148초였다. 새 엔진의 비용은 첫 공통 gradient와 평가에서 다시 측정해 추산을 갱신한다. [ABC compute 원자료][abc-compute]

네 group을 25개씩 처리하면 rewrite는 `4×ceil(25/2)×6=312`, essence는 `ceil(100/2)=50`으로 full gradient당 **362 model forward/backward**를 계획한다. Group이 네 개라고 전체 모델 backward가 네 배가 되는 것은 아니다. Penalty 및 small solve는 별도 ledger로 센다.

| 항목 | 주 비교 | 보조 포함 고정 묶음 |
|---|---:|---:|
| Trajectories | 9 | 12 |
| Logical steps | 72 | 104 |
| 같은 G0 공유 후 실제 full-gradient sweep | 3+9×7=66 | 3+(104−12)=95 |
| Model backward 호출 계획 | 23,892 | 34,390 |
| Terminal full-objective forward sweep | 9 | 12 |
| Matched-risk objective sweep | 3 | 3 |
| Same-state nominal edit-only sweep | 6 | 6 |
| 새 panel prompt pairs — matched probe 포함 | 68,100 | 89,700 |
| 새 candidate sequences | 136,200 | 179,400 |
| 새 자유 생성 | 180 | 180 |
| 새 compute_z / inner native 재호출 | 0 / 0 | 0 / 0 |

전체 묶음의 training forward 계획은 `95×362 + 12×362 + 3×362 + 6×312 = 41,692`회다. Teacher 준비, panel evaluation, generation, load/geometry, penalty gradient와 small solve는 따로 센다. Initial gradient 공유는 같은 X/teacher/objective/순서일 때만 성립한다.

단순 기존 계수로 전체 묶음의 training/evaluation 시간을 합하면 약 **2.2 GPU시간 + model/asset load·geometry·generation 비용**이다. Group accumulation 증가와 새 controller overhead를 포함한 운영 계획 범위는 **대략 2.5–4 GPU시간**으로 둔다. 실측을 대신하는 약속이 아니며 새 GPU의 시간을 기존 A6000 시간에 섞어 계산하지 않는다. 기존 rank100 direct peak allocated 약 34.6 GiB에 새 group/FP64 reduced tensor와 준비 자산의 peak를 추가 측정한다.

이 비용은 **cached native 이후의 진단 비용**이다. 과거 Blackwell 측정의 BLUE(L4-only) 약 284.6초/B100과 같은 GPU의 end-to-end 속도 비교로 사용하지 않는다. 온라인 방법의 총비용은 cold L4 z, native write, refinement, history와 평가를 분리해 별도로 측정해야 한다.

## 17. 실행 순서, 산출물, 후속 범위

1. Middle의 공통 자산과 small-matrix 계산을 확인한다.
2. Middle H/R/EP를 실행해 구현·실제 비용을 확인한다. 과학적 성능이 낮아도 Early/Late의 같은 비교를 이어간다.
3. Early/Late 주 비교와 같은-state probes를 완료한다.
4. Middle EP-Free/J4/N16 및 고정 terminal/생성 평가를 완료한다.
5. Native 대비 실용적 이득, 같은-state 보호 효과, proxy 한계, 비용을 한 번에 분석한다.

산출물은 `run-index.csv`, `trajectory.csv`, `same-state-probes.csv`, `request-metrics.csv`, `paired-summary.csv`, `compute-summary.csv`, `diagnostic-report-ko.md`로 충분하다. Run index에는 실행/미실행/실행 오류를 구분한다. X snapshots와 U/WN reference를 저장해 0/1/2/4/8 및 N16 대응 상태를 재구성한다. Report에는 실제 관측, 가능한 설명, 미분리 요인을 분리한다.

**짧은 sequential 확인은 별도 두 번째 묶음으로 정의한다.** 이번 12-trajectory 진단과 비용에 자동 합산하지 않는다. 자연 후속은 W50에서 B51–B55의 5×B100이며 N/H/EP를 사전 고정해 비교한다. 각 방법은 자기 endpoint와 history로 계속 진행하고, W50에서 정한 Fixed/Past 패널을 유지하면서 새 500 edits의 all-new-seen retention을 따로 평가한다. 이 후속에서 EP가 실제로 유리하지 않아도 결과를 지우지 않는다.

Sequential 구현에서는 각 batch의 native z와 WN을 **그 arm의 현재 모델**에서 생성한다. B52 이후 historical cached z를 분기 arm에 재사용하지 않는다. 첫 공통 B51의 native 준비만 공유할 수 있다. M은 inner loop 동안 고정하고 현재 batch history는 한 번만 append한다. Prepared MN 또는 native wrapper가 이미 append한 상태를 쓰면 재append하지 않는다. Single L4와 고정 teacher-forced prompt에서는 refinement가 K를 바꾸지 않지만, 다른 batch의 prompt/target/모델 상태까지 같다고 가정하지 않는다.

이 단계에서 한 layer 안의 보정이 유용하다는 근거를 얻은 뒤에야, We부터 barrier가 write를 guide하는 경로 또는 여러 layer의 local-z 배분으로 질문을 넓힌다. 후속 선택은 결과의 원인을 설명하기 위한 연구 판단이며, 이번 사전 지정 비교를 중간에 멈추는 gate가 아니다.

## 18. 계보와 참조

이번 작성에서 fixture 존재·기존 code/report·실행 수와 비용 산술을 대조하고 독립 수식 검토를 받았다. 별도 CPU 합성 계산에서 80개 SPD 문제의 closed-form/KKT 및 risk 단위 재표현을 검산했다. 최대 잔차는 약 `9.24e-14`였고, q=0, 빈 edit observer, 같은 진척에서 norm이 커지는 보정, covariance 이차 전개도 확인했다. 이 검산은 실제 LLM controller 실행이나 성능 검증이 아니다. 문서의 로컬 참조 16개가 존재하는 것도 확인했다.

- 이전 동기: [단일-layer cumulative-risk 진단 설계][old-single], [native-metric refinement 설계][old-native], [EP-Free/EP-CBF 설계][old-ep]. 세 문서는 실행 결과와 구분한다. 이번 문서의 주 방법·observer·elasticity·실행 목록이 최신이다.
- 연구 proposal: [기존 response-barrier proposal][proposal]. 이 문서는 단일-layer 후속 진단이며 이전 여러-layer 방법의 성능을 승계하지 않는다.
- 실제 근거: [ABC 최종 보고서][abc-final], [ABC A facts][abc-facts], [FBP][fbp], [JV][jv], [Phase 3][phase3].

[old-single]: /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-single-layer-cumulative-risk-diagnostic-design.md
[old-native]: /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-native-metric-barrier-refinement-diagnostic-design.md
[old-ep]: /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-edit-progress-preserving-damage-removal-design.md
[proposal]: /mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-03-ordered-response-barrier-ode-edit-proposal.md
[abc-final]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/final/final-diagnostic-report-ko.md
[abc-facts]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/A/A-factual-report-ko.md
[abc-compute]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/A/auxiliary/compute-summary.csv
[panels]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/A/metadata/panel-metadata.csv
[native-config]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/config.json
[native-keys]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_ks.py
[runtime]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/runtime.py
[objective]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/objective.py
[fbp]: /mnt/raid5/janghj/ODE-edit/experiment-reports/servers/server1/2026-08-28-bgode-fbp-minimal-fixed-basis-barrier-ode-v1/bgode-fbp-minimal-fixed-basis-barrier-ode-factual-ko.md
[jv]: /mnt/raid5/janghj/ODE-edit/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v1/gh-mechanism-review-ko.md
[alloc]: /mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_alloc/p1_runtime.py:814
[phase3]: /mnt/raid5/janghj/ODE-edit/experiment-reports/servers/server1/2026-08-23-p1r52-c-writer-phase3/exhaustive-v1/p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-exhaustive-factual-ko.md
