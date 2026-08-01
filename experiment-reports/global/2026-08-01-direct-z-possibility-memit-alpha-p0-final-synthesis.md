# Direct-z 최종 synthesis — MEMIT × AlphaEdit p0

> **Discussion 갱신 시각:** 2026-08-02 00:47:30 KST (`UTC+09:00`)
>
> **갱신 범위:** 완료된 raw output, canonical analyzer, gate와 기존 수치는
> 변경하지 않았다. 아래 갱신은 post-experiment Discussion과 연구 단계 판정만
> 추가한다.

## 최종 판정

이 fixed-panel 실험으로 닫히는 것은 **ODE/BF의 국소적·proposal-conditioned
motivation**이다. 닫히는 문장은 다음과 같다.

> 동일한 endpoint C-energy에서 state-refreshed genuine-Alpha BF 경로는
> one-pass ordered genuine-Alpha native writer보다 두 모델 모두에서 frozen direct-z를
> 더 충실히 실제 weight에 쓰고, local edit proxy도 더 높이는 endpoint를
> 선택할 수 있다.

그러나 이것은 ODE-Edit의 절대 성능 우월성이 아니다. `llama3-8b-inst`와
`qwen2.5-7b-inst` 모두에서 Alpha-BF의 family-relative z/edit lift는 양수지만,
Alpha-BF endpoint 자체의 z/edit는 MEMIT-BF보다 낮다. 반대로 Alpha-BF의
Frobenius norm은 두 모델 모두 MEMIT-BF보다 작다. Held-out preservation
endpoint는 Llama에서 Alpha-BF가 명확히 낫지만 Qwen에서는 4/8, 평균
`+0.00268541`로 사실상 중립에 가깝다.

따라서 최종 evidence는 “state refresh가 주는 국소 방향 보정의 가능성”을
지지하고, “더 나은 deployable method”, “일관된 model preservation”, “더 작은
update의 보편적 보장”, “lifelong/sequential collapse 방지”는 지지하지 않는다.

## 분석 계약과 네 비교량

Cross-track artifact는 두 모델 각각 replay-lock에 고정된 case/request,
direct-z artifact/tensor/target, W0 lineage와 endpoint C-budget에 대해 8/8
cross-bound join을 검증했다. Held-out payload는 track·model 사이 equality를
검증하지만 replay-lock pin은 아니다. Alpha의 context/state/snapshot/parameter/
solver anchor는 외부 cross identity가 아니라 별도의 self-bound provenance로
8/8 검증했다. 기술 판정은
`PASS_CROSS_BOUND_JOIN_ALPHA_SELF_BINDING_AND_C_BUDGET`, 최종 analyzer 판정은
`CROSS_TRACK_AXIS_SIGNALS_REPORTED_NO_MODEL_POOLING`이며 analysis SHA-256은
`32b7bf86d28a3ee5feafb540ca7d352c3c70a78d79e87dc5f8274d91feeb89c0`이다.

모든 비교는 모델별로 독립적이다. Llama와 Qwen의 효과를 합산하거나 평균내지
않는다. Precommitted lenient gate는 **sign-normalized paired mean > 0**이고
**positive cases가 5/8 이상**인 경우다. Bootstrap CI는 설명용이며 gate가
아니다. 아래에서 양수는 각 축에서 왼쪽 항이 더 좋다는 뜻이다.

동일 metric의 sign-normalized 성능을 `S`라 할 때 네 비교량은 엄격히
분리한다.

\[
\begin{aligned}
A_f &= S(\mathrm{native}_f)-S(\mathrm{no\mbox{-}op}),\\
L_f &= S(\mathrm{BF}_f)-S(\mathrm{native}_f),\\
I_{\alpha-M} &= L_{\mathrm{Alpha}}-L_{\mathrm{MEMIT}},\\
E_{\alpha-M} &= S(\mathrm{AlphaBF})-S(\mathrm{MEMITBF}).
\end{aligned}
\]

`A`는 native baseline의 absolute actual write, `L`은 동일 family 안의 BF lift,
`I`는 Alpha−MEMIT 2×2 interaction, `E`는 두 BF endpoint의 절대 대비다.
Positive interaction은 positive endpoint 대비를 함의하지 않는다.

## 1. Native baseline은 direct-z를 실제로 얼마나 쓰는가

Representation metric은

\[
r_z(W)=\frac{\lVert h_8(W)-z^*\rVert}
{\lVert h_8(W_0)-z^*\rVert},\qquad
g_z(W)=r_z(W_0)-r_z(W)
\]

이며 `g_z>0`이면 no-op보다 frozen target에 가까운 actual weight endpoint다.
아래 MEMIT native mean은 joined cross-track identity
`MEMITBF endpoint − MEMIT BF lift`로 복원했고, Alpha native 값은 해당
absolute no-op contrast다. 이는 같은 case/target/C-budget의 aggregate mean
비교다. Cross artifact가 native-vs-no-op case count를 별도 출력하지 않는
MEMIT 열에는 새로운 count gate를 추정해 붙이지 않는다.

| 모델 | MEMIT native z gain / 잔여 residual | Alpha native z gain / 잔여 residual | MEMIT native output-NLL gain | Alpha native output-NLL gain |
|---|---:|---:|---:|---:|
| Llama | `+0.65307184` / `~0.34692816` | `+0.28445854` / `~0.71554146` (8/8) | `+10.12363116` | `+4.16866735` (8/8) |
| Qwen | `+0.75959129` / `~0.24040871` | `+0.46528274` / `~0.53471726` (8/8) | `+10.53310070` | `+10.45983613` (8/8) |

따라서 native baseline이 direct-z를 전혀 쓰지 못한다는 가설은 기각된다.
두 family 모두 mean상 target을 부분적으로 실제 weight에 반영한다. 동시에
native endpoint는 oracle처럼 residual을 0으로 만들지 못하며, 특히 Alpha
native는 같은 C cap에서 MEMIT native보다 두 모델 모두 absolute z gain이
작다. 이후 Alpha의 큰 relative BF lift를 해석할 때 이 더 낮은 출발점을
반드시 보존해야 한다.

## 2. ODE/BF는 동일-family native를 개선하는가

### Llama — BF minus same-family native

| 축 | MEMIT BF lift; +/8 | Alpha BF lift; +/8 |
|---|---:|---:|
| z-residual 감소 | `-0.13500684`; 0/8 | `+0.02056762`; 7/8 ✓ |
| generated mean error 감소 | `-0.11268154`; 0/8 | `+0.02623012`; 7/8 ✓ |
| target delta gain | `+0.01511731`; 7/8 ✓ | `+0.14513059`; 8/8 ✓ |
| target delta cosine | `-0.09460918`; 0/8 | `-0.18616539`; 0/8 |
| output progress | `+0.76534057`; 8/8 ✓ | `+4.91699534`; 8/8 ✓ |
| output-NLL reduction | `+0.16067749`; 8/8 ✓ | `+3.78883688`; 8/8 ✓ |
| exact margin | `+1.18858397`; 8/8 ✓ | `+4.70847762`; 8/8 ✓ |
| paraphrase-NLL reduction | `+0.80229617`; 6/8 ✓ | `+3.26029124`; 8/8 ✓ |
| preservation (`-heldout KL`) | `-0.09631979`; 0/8 | `-0.14487684`; 0/8 |
| off-token spill 감소 | `-0.02665687`; 0/8 | `-0.03609012`; 0/8 |
| Frobenius norm 감소 | `-0.04089099`; 0/8 | `-0.02535538`; 0/8 |
| NFE 감소 | `0`; ties | `-19`; 0/8 |

Llama MEMIT에서는 BF가 output 축을 개선하지만 direct-z fidelity는 native보다
낮다. Alpha에서는 BF가 z와 edit를 함께 개선한다. 두 family 모두
preservation과 spill은 악화하고, Alpha-BF norm은 native보다 약간 크다.

### Qwen — BF minus same-family native

| 축 | MEMIT BF lift; +/8 | Alpha BF lift; +/8 |
|---|---:|---:|
| z-residual 감소 | `-0.05456161`; 2/8 | `+0.08233046`; 6/8 ✓ |
| generated mean error 감소 | `-0.02368673`; 3/8 | `+0.10533109`; 7/8 ✓ |
| target delta gain | `-0.00699852`; 4/8 | `+0.13776107`; 8/8 ✓ |
| target delta cosine | `-0.02069393`; 0/8 | `-0.01412005`; 2/8 |
| output progress | `-0.11081997`; 2/8 | `+0.71733285`; 7/8 ✓ |
| output-NLL reduction | `+0.00002117`; 2/8 | `+0.05692602`; 7/8 ✓ |
| exact margin | `-0.12983727`; 1/8 | `+0.72356713`; 6/8 ✓ |
| paraphrase-NLL reduction | `+0.20916645`; 7/8 ✓ | `+0.88091911`; 7/8 ✓ |
| preservation (`-heldout KL`) | `-0.08699755`; 2/8 | `-0.20097198`; 1/8 |
| off-token spill 감소 | `-0.00360239`; 3/8 | `-0.01056363`; 2/8 |
| Frobenius norm 감소 | `+0.83019237`; 7/8 ✓ | `+0.10153338`; 7/8 ✓ |
| NFE 감소 | `0`; ties | `-19`; 0/8 |

Alpha-BF의 공통 signal은 명확하다. 두 모델 모두 z residual, generated error,
delta gain, output progress/NLL, exact margin, paraphrase 축에서 native 대비
lenient gate를 통과한다. 반면 cosine은 두 모델 모두 낮아져 단순 radial
확대가 아니라 방향 재배치임을 시사한다. Preservation과 spill은 두 모델
모두 악화하고, smaller Frobenius lift는 Qwen에서만 성립한다. 따라서
“BF가 같은 budget에서 target/edit에 유리한 방향을 고를 수 있다”는 claim은
지지되지만, “BF가 native보다 더 잘 보존하거나 항상 더 작은 update를 쓴다”는
claim은 지지되지 않는다.

## 3. Alpha−MEMIT 2×2 interaction은 무엇을 말하는가

Interaction은 Alpha에서의 BF lift가 MEMIT에서의 BF lift보다 큰지를 묻는다.
Endpoint 자체의 우열은 묻지 않는다.

| 축 | Llama interaction; +/8 | Qwen interaction; +/8 |
|---|---:|---:|
| z-residual 감소 | `+0.15557446`; 8/8 ✓ | `+0.13689207`; 8/8 ✓ |
| generated mean error 감소 | `+0.13891166`; 8/8 ✓ | `+0.12901782`; 8/8 ✓ |
| generated worst error 감소 | `+0.10040715`; 8/8 ✓ | `+0.12035681`; 8/8 ✓ |
| target delta gain | `+0.13001328`; 8/8 ✓ | `+0.14475959`; 8/8 ✓ |
| target delta cosine | `-0.09155620`; 0/8 | `+0.00657388`; 4/8 |
| output progress | `+4.15165477`; 8/8 ✓ | `+0.82815281`; 6/8 ✓ |
| output-NLL reduction | `+3.62815939`; 8/8 ✓ | `+0.05690485`; 7/8 ✓ |
| exact margin | `+3.51989365`; 7/8 ✓ | `+0.85340440`; 7/8 ✓ |
| paraphrase-NLL reduction | `+2.45799507`; 6/8 ✓ | `+0.67175266`; 6/8 ✓ |
| preservation (`-heldout KL`) | `-0.04855705`; 4/8 | `-0.11397443`; 3/8 |
| off-token spill 감소 | `-0.00943325`; 1/8 | `-0.00696124`; 2/8 |
| Frobenius norm 감소 | `+0.01553561`; 6/8 ✓ | `-0.72865899`; 1/8 |
| NFE 감소 | `-19`; 0/8 | `-19`; 0/8 |

두 모델에서 z/generated/edit interaction이 같은 방향으로 gate를 통과한다.
이는 refreshed path의 유용성이 solve/proposal geometry와 상호작용한다는 가장
강한 cross-track signal이다. 그러나 preservation·spill interaction은 둘 다
음수이고 norm interaction은 모델별로 반대다. 또한 Alpha native의 absolute
출발점이 MEMIT native보다 낮으므로, 큰 positive interaction의 일부는
“Alpha-BF endpoint가 더 좋다”가 아니라 “Alpha 안에서 BF가 더 많은 간극을
회복한다”는 뜻이다.

## 4. Alpha-BF endpoint는 MEMIT-BF endpoint보다 좋은가

아래는 native를 빼지 않은 동일 case의 absolute BF endpoint 대비
`AlphaBF − MEMITBF`다.

| 축 | Llama endpoint contrast; +/8 | Qwen endpoint contrast; +/8 |
|---|---:|---:|
| z-residual 감소 | `-0.21303884`; 0/8 | `-0.15741648`; 0/8 |
| generated mean error 감소 | `-0.17573496`; 0/8 | `-0.12978544`; 0/8 |
| generated worst error 감소 | `-0.13321338`; 0/8 | `-0.12775790`; 0/8 |
| target delta gain | `-0.27241866`; 0/8 | `-0.13824734`; 0/8 |
| target delta cosine | `-0.14765366`; 0/8 | `-0.04516524`; 0/8 |
| output progress | `-5.35787861`; 0/8 | `-0.89626947`; 0/8 |
| output-NLL reduction | `-2.32680442`; 0/8 | `-0.01635973`; 0/8 |
| exact margin | `-5.36988688`; 0/8 | `-0.67598736`; 0/8 |
| paraphrase-NLL reduction | `-2.54894562`; 1/8 | `-0.10662905`; 2/8 |
| preservation (`-heldout KL`) | `+0.33627280`; 8/8 ✓ | `+0.00268541`; 4/8 |
| off-token spill 감소 | `+0.02888428`; 8/8 ✓ | `+0.00009203`; 6/8 ✓ |
| Frobenius norm 감소 | `+0.46195763`; 8/8 ✓ | `+1.66568573`; 8/8 ✓ |
| NFE 감소 | `-21`; 0/8 | `-21`; 0/8 |

절대 endpoint 결론은 relative lift와 다르다.

- Llama z gain은 Alpha-BF `0.30502616`, MEMIT-BF `0.51806500`이고,
  output-NLL gain은 `7.95750422` 대 `10.28430864`다.
- Qwen z gain은 Alpha-BF `0.54761319`, MEMIT-BF `0.70502968`이고,
  output-NLL gain은 `10.51676214` 대 `10.53312187`다.
- 따라서 z/generated/edit endpoint는 두 모델 모두 MEMIT-BF가 더 높다.
  반대로 Alpha-BF는 Frobenius norm이 두 모델 모두 더 작고 spill도 낮다.
- Preservation endpoint는 Llama에서 Alpha-BF가 `+0.33627280`, 8/8로
  명확히 낫다. Qwen은 `+0.00268541`, 4/8로 방향성이 없고 0에 가깝다.
- Alpha-BF는 24 NFE/case이고 endpoint NFE 대비가 `-21`이므로 MEMIT-BF의
  3 NFE/case보다 훨씬 비싸다.

즉 “Alpha에서 BF lift가 더 크다”를 “Alpha-BF가 MEMIT-BF보다 우수하다”로
바꾸면 안 된다. 이 실험의 absolute z/edit winner는 두 모델 모두 MEMIT-BF다.

## BF와 DF를 수식적으로 어떻게 구분해야 하는가

이 실험에는 `DF`라는 정식 deployable arm이 없다. DF가 synchronous cone을
뜻한다면 그것은 W0의 unit-C responses에 대한 non-negative ridge
**비배포성 diagnostic ceiling**이다. BF와 cone의 차이는 두 method의 공정한
ranking이 아니다. DF가 frozen/current direction을 한 번만 쓰는 direct
one-shot을 뜻한다면, BF의 가능한 이점은 numerical splitting 자체가 아니라
**state dependence**에서만 나온다.

같은 smooth autonomous vector field `F`를 비교하는 이상화에서

\[
\Delta_D=T F_0,
\]

이고 `h=T/K`인 K-step refreshed Euler/BF path는

\[
\Delta_{BF,K}
=T F_0+\frac{T^2}{2}\left(1-\frac1K\right)J_FF_0+O(T^3).
\]

Exact ODE endpoint는

\[
\Delta_{ODE}=T F_0+\frac{T^2}{2}J_FF_0+O(T^3)
\]

이므로 one-shot의 leading endpoint error는
`(1/2)T^2J_FF_0`, K-step의 leading error는
`(1/(2K))T^2J_FF_0`다. K=4에서 BF가 one-shot에 추가하는 correction은

\[
\frac{3}{8}T^2J_FF_0,
\]

exact ODE에 남는 leading error는 `(1/8)T^2J_FF_0`다. `J_FF_0=0`이면 이
차이는 사라진다. 따라서 단순히 동일한 fixed update를 네 조각으로 나누는
것은 이점을 만들지 않는다.

### C-normalized endpoint에서 필요한 조건

C-inner product와 tangent projection을

\[
\langle A,B\rangle_C=\operatorname{tr}(ACB^\top),\qquad
\Pi_{\perp_C F_0}G
=G-F_0\frac{\langle F_0,G\rangle_C}{\langle F_0,F_0\rangle_C}
\]

로 두자. Raw BF correction의 `F_0` 방향 성분은 endpoint를 같은 C-energy로
normalize하면 1차에서 사라진다. 그러므로 matched-C에서 BF가 direct보다
다른 endpoint를 만들기 위한 최소 기하 조건은

\[
\Pi_{\perp_C F_0}J_FF_0\ne0
\]

이다. 이것만으로 성능 이득은 충분하지 않다. Lower-is-better task loss
`\mathcal L`에 대해서는 추가로

\[
\left\langle \nabla\mathcal L,
\Pi_{\perp_C F_0}J_FF_0\right\rangle<0
\]

같은 task-direction alignment가 필요하다. Alpha에서 두 모델의 z/edit
lift가 양수이고 cosine이 낮아진 관측은 이 “radial 확대가 아닌 tangential
재배치”와 일관되지만, Jacobian 항을 직접 식별한 causal proof는 아니다.

### Update norm이 작아지는 조건

Unnormalized path에서는

\[
\lVert TF_0+\epsilon J_FF_0\rVert_F^2
=T^2\lVert F_0\rVert_F^2
+2T\epsilon\langle F_0,J_FF_0\rangle_F+O(\epsilon^2),
\]

이므로 leading-order smaller norm에는
`\langle F_0,J_FF_0\rangle_F<0`가 필요하다. 하지만 matched C-energy에서는
이 radial 조건만으로 충분하지 않다. C-Rayleigh quotient를

\[
\rho_C(\Delta)=\frac{\lVert\Delta\rVert_C^2}
{\lVert\Delta\rVert_F^2}
\]

로 두면 같은 C-energy에서 Frobenius norm이 작으려면 endpoint direction의
`\rho_C`가 더 커야 한다. 실제 Alpha BF−native norm lift가 Llama
`-0.02535538`이지만 Qwen `+0.10153338`인 것은 smaller-norm이 BF의 보편적
결과가 아님을 보여준다. Alpha-BF가 MEMIT-BF보다 두 모델 모두 norm이 작은
것은 family endpoint의 absolute 차이이며 순수 integrator theorem이 아니다.

### Preservation이 보장되지 않는 이유

Weight update를 vectorize한 `\Delta`에 대해 held-out KL의 국소 근사는

\[
\mathrm{KL}_{hold}(\Delta)
\approx\frac12\Delta^\top F_{hold}\Delta
\]

이다. `\Delta_{BF}=\Delta_D+\delta`이면

\[
\mathrm{KL}_{hold}(\Delta_{BF})-
\mathrm{KL}_{hold}(\Delta_D)
\approx
\Delta_D^\top F_{hold}\delta
+\frac12\delta^\top F_{hold}\delta.
\]

C-tangent correction이라는 사실은 이 cross term의 부호를 정하지 않는다.
`C`와 `F_hold`의 geometry도 같을 필요가 없다. 실제 Alpha BF−native
preservation은 Llama `-0.14487684` (0/8), Qwen `-0.20097198` (1/8)로
둘 다 악화했다. 따라서 state refresh, matched C, smaller Frobenius 중 어느
것도 downstream 보존을 자동 보장하지 않는다.

## Genuine Alpha와 post-hoc projection의 비가환성

Genuine P-inside-solve와 post-hoc right projection은 각각

\[
\Delta W_{gen}
=R(\lambda I+K^\top PK)^{-\top}(PK)^\top,
\qquad
\Delta W_{post}
=R(\lambda I+K^\top K)^{-\top}K^\top P
\]

이므로 일반적으로 같지 않다. 첫 식에서는 `P`가 normal system과 RHS를
함께 바꾸고, 둘째 식에서는 identity-P base update를 이미 정한 뒤 투영한다.

| Genuine C − post-hoc C | Llama mean; +/8 | Qwen mean; +/8 |
|---|---:|---:|
| z-residual 감소 | `-0.01133774`; 0/8 | `-0.09317304`; 0/8 |
| output-NLL reduction | `-0.24165694`; 0/8 | `-0.04695102`; 0/8 |
| preservation | `+0.00407069`; 8/8 ✓ | `+0.09821588`; 7/8 ✓ |
| off-token spill 감소 | `+0.00010584`; 4/8 | `+0.00322912`; 6/8 ✓ |
| Frobenius norm 감소 | `-0.00282356`; 0/8 | `-0.01858916`; 3/8 |

비가환성은 실제로 trade-off를 바꾸지만 scalar dominance를 만들지 않는다.
이 panel에서는 post-hoc가 두 모델 모두 direct-z/edit에 유리하고 genuine은
preservation에 유리하다. 따라서 Alpha null-space geometry의 의미는
“보존을 고려한 다른 feasible direction”이지 “target fidelity와 norm까지
동시에 항상 개선”이 아니다.

## Direct-z fidelity는 충분조건도 필요조건도 아니다

**충분조건이 아니다.** Alpha native와 BF는 두 모델 모두 no-op보다 z를 더
충실히 쓰지만 모든 deployable Alpha write의 preservation과 spill은 no-op보다
나쁘다. Alpha BF가 native보다 z/edit를 더 개선한 바로 그 비교에서도
preservation은 두 모델 모두 악화한다. Direct-z를 잘 쓰는 것은 representation
diagnostic이지 benign downstream write의 보증이 아니다.

**필요조건도 아니다.** Llama MEMIT BF는 native보다 z-residual fidelity가
`-0.13500684` (0/8)로 낮지만 output-NLL은 `+0.16067749` (8/8), output
progress는 `+0.76534057` (8/8) 높다. 즉 frozen canonical z와 downstream
edit objective 사이의 nonlinear mapping 때문에 z fidelity가 낮아져도 edit
proxy가 개선될 수 있다.

## 순수 integrator attribution과 lifelong claim의 경계

앞의 Taylor 식은 **동일한 F를 동일한 조건에서 평가한다는 이상화**다. 실제
Alpha 비교는 one-pass ordered native proposal과 synchronous state-refreshed
controller를 대비하며 NFE도 5 대 24다. Proposal timing, descendant state,
direction construction, normalization, compute가 함께 달라진다. 따라서 관측된
lift를 순수 numerical integrator order 하나에 귀속할 수 없다. 허용되는
attribution은 “state-dependent re-proposal/controller path가 유용한 local
direction을 만든다”까지다.

또한 모든 case는 fresh W0에서 시작하는 하나의 atomic edit다. K=4는 한 fact를
쓰는 내부 numerical path이지 네 개의 sequential fact가 아니다. 이 자료에는
edit history, retention trajectory, interference accumulation, collapse onset가
없다. 따라서 sequential collapse 방지, lifelong retention, long-horizon model
preservation claim은 실험적으로 정의조차 되지 않는다.

## Claim ledger

| Claim | 판정 | 현재 evidence가 허용하는 범위 |
|---|---|---|
| Native baseline이 frozen direct-z를 실제 weight에 쓴다 | **지지** | 두 family/model의 mean z gain이 양수이며 Alpha는 각 모델 8/8. 완전 write는 아님. |
| Alpha BF가 동일-family native보다 z actual-write를 개선한다 | **지지** | Llama `+0.02056762` 7/8, Qwen `+0.08233046` 6/8. |
| Alpha BF가 동일-family native보다 edit signal을 개선한다 | **지지** | 두 모델 output NLL/progress/margin/paraphrase gate 통과. |
| Alpha에서 BF lift가 MEMIT보다 크다 | **지지, 상대효과 한정** | z/edit 2×2 interaction이 두 모델 모두 양수. Alpha native의 낮은 출발점 포함. |
| Alpha-BF endpoint가 MEMIT-BF보다 z/edit가 좋다 | **기각** | z와 주요 edit 축이 두 모델 모두 0/8; MEMIT-BF endpoint 우세. |
| Alpha-BF endpoint norm이 MEMIT-BF보다 작다 | **지지, family 대비 한정** | Llama `+0.46195763`, Qwen `+1.66568573`, 둘 다 8/8. |
| BF가 native보다 항상 더 작은 update를 쓴다 | **기각** | Alpha 내 Llama는 norm 증가, Qwen만 감소. C-match도 Fro ordering을 고정하지 않음. |
| BF가 model preservation을 개선한다 | **기각** | Alpha BF−native preservation이 두 모델 모두 음수. Absolute AlphaBF−MEMITBF는 Llama만 명확, Qwen 중립. |
| Direct-z fidelity가 edit/preservation의 충분조건이다 | **기각** | 더 나은 z와 preservation 악화가 동시 관측. |
| Direct-z fidelity가 edit improvement의 필요조건이다 | **기각** | Llama MEMIT에서 z lift 음수이나 output lift 양수. |
| Genuine Alpha가 post-hoc projection을 지배한다 | **기각** | Genuine은 preservation, post-hoc는 z/edit에 유리한 trade-off. |
| BF가 deployable DF보다 우월하다 | **판정 불가** | 정식 DF arm 없음. Cone은 nondeployable diagnostic; one-shot 대비라면 state-dependence 조건부. |
| 관측 효과가 순수 integrator 효과다 | **판정 불가** | ordered native 대 refreshed controller, 5 대 24 NFE의 복합 대비. |
| ODE-Edit가 lifelong collapse를 방지한다 | **판정 불가/범위 밖** | Atomic edit뿐이며 sequential history가 없음. |
| ODE-Edit/AlphaEdit/MEMIT의 일반 method superiority | **기각** | 축별·모델별 결과가 상충하고 absolute z/edit endpoint는 MEMIT-BF 우세. |

## Motivation을 어디까지 닫을 수 있는가

현재 실험은 세 가지를 충분히 닫는다.

1. Frozen direct-z는 oracle에서만 의미 있는 추상 target이 아니라 native
   weight writer가 부분적으로 실제화할 수 있는 reachable representation이다.
2. Same-C Alpha geometry에서는 state refresh가 두 architecture 모두에서
   one-pass ordered native보다 더 나은 z/edit endpoint를 선택한다. Positive 2×2
   interaction은 이 신호가 bare step splitting보다 proposal/solve geometry와
   결합되어 있음을 뒷받침한다.
3. 그 이점의 수식적 후보는 `J_FF_0`의 C-tangential, task-aligned component다.
   이는 동일 C-energy에서도 endpoint direction을 바꿀 수 있지만 preservation,
   Fro norm, compute를 동시에 개선한다는 보장은 없다.

동시에 최종 method claim은 닫히지 않는다. Alpha-BF는 MEMIT-BF보다 두 모델
모두 absolute z/edit가 낮고 21 NFE 더 비싸다. Preservation은 family-relative로
악화하며 absolute endpoint에서는 Llama만 명확하고 Qwen은 중립이다. 따라서
motivation의 최종 형태는 **“ODE-style state refresh가 proposal-conditioned
local geometry를 유용하게 바꿀 수 있다”**이고, **“ODE-Edit이 baseline을
전반적으로 능가하거나 lifelong model을 더 잘 보존한다”**가 아니다.

## Discussion — direct-z possibility에서 method로 넘어가는 경계

### Evidence freeze와 이 실험의 정확한 역할

이 실험은 ODE-Edit의 최종 method 성능 검정이 아니라, frozen direct-z를 실제
weight에 쓰는 과정에서 state-dependent re-proposal이 동일 endpoint C-budget의
parameter realization을 바꿀 수 있는지 확인한 **atomic possibility diagnostic**이다.
모델별 8 cases를 fresh W0에서 하나씩 독립 평가했고, gate는 model별 paired
mean이 양수이면서 5/8 이상이 양수인 lenient directional gate다. CI exclusion을
요구하지 않으므로 결과를 benchmark-level 통계적 우월성으로 읽지 않는다.

별도 표시가 없는 아래 실증 문장은 terminal MEMIT × genuine-Alpha direct-z
fixed panel만 근거로 한다. 별도 GH track의 completed adaptive-MEMIT 진단은
terminal status와 evidence boundary로만 분리해 표기한다. Projected-Alpha
A/B/C/G와 4-edit micro-sequential stage는 이 Discussion evidence freeze 시점에
terminal evidence로
포함하지 않았다. 이후 결과도 자기 report와 audit에서 판정해야 하며, partial
runtime output을 이 fixed-panel estimate에 소급해 합치지 않는다.

### BF, DF, adaptive layer allocation을 분리하면 무엇이 남는가

서로 다른 layer parameter block을 하나의 joint editable-parameter vector에
zero-embed하는 연산자를 `\iota_l`라 두고, waypoint `s`의 actuator를

\[
b_{s,l}:=\iota_l\operatorname{vec}\!\left(\widehat B_l(W_s)\right),\qquad
D_s=[b_{s,1},\ldots,b_{s,L}]
\]

로 정의한다. Non-negative layer share를 `v_s`, total integration length를
`T=\sum_s h_s`라 두면 현재 BF endpoint의 joint-parameter displacement는

\[
\Delta_{BF}=\sum_s h_sD_sv_s.
\]

초기 same-snapshot fixed control `TD_0v_0`와의 차이는

\[
\begin{aligned}
\Delta_{BF}-TD_0v_0
={}&\sum_s h_s(D_s-D_0)v_0 &&\text{(direction/state relinearization)}\\
&+\sum_s h_sD_0(v_s-v_0) &&\text{(adaptive layer-share redistribution)}\\
&+\sum_s h_s(D_s-D_0)(v_s-v_0) &&\text{(interaction)}.
\end{aligned}
\]

하지만 실제 native writer는 위 fixed synchronous control이 아니다. Native
MEMIT은 앞 layer write 뒤 residual/key를 다시 계산하는 ordered
Gauss--Seidel형이며, 국소적으로

\[
\Delta_{native}\simeq
\sum_l U_l(W_0)
+\sum_l\sum_{i<l}DU_l(W_0)\!\left[U_i(W_0)\right]+\cdots
\]

처럼 schedule 자체의 Fréchet-derivative 교차항을 갖는다. 위 세 항을 각각
`R_{dir}`, `R_{alloc}`, `R_{int}`라 하고

\[
R_{sched}:=TD_0v_0-\Delta_{native}
\]

로 정의하면
`\Delta_{BF}-\Delta_{native}=R_{dir}+R_{alloc}+R_{int}+R_{sched}`다.
`R_{sched}`에는 ordered-vs-synchronous snapshot, denominator와 commit-order
차이가 포함된다. 현재 결과를 layer weighting, direction refresh 또는 numerical
ODE 하나의 순수 효과로 단독 귀속할 수 없는 이유다.

여기서 DF는 용어를 고정해야 한다.

- DF가 proposal의 **layer-axis ordered native execution**을 뜻하면, native는
  완전 open-loop가 아니지만 각 layer를 한 번 방문하고 joint waypoint에서 모든
  layer를 재경쟁시키지 않는다. BF와의 비교에는 `R_sched`가 섞인다.
- DF가 `sync_z_cone`을 뜻하면, 이는 W0의 unit-C response들을 frozen target에
  맞추는 target-aware nondeployable diagnostic ceiling이다. BF와 method ranking을
  할 arm이 아니다.
- DF가 동일 synchronous field를 한 번 적용하는 K=1 direct policy를 뜻하면,
  현재 artifact에는 formal equal-scheduler/equal-NFE arm이 없다.

따라서 이 report는 **BF가 deployable DF보다 낫다**고 보이지 않는다. 보인 것은
same-C에서 state-conditioned synchronous path가 one-pass ordered native와 다른
endpoint trade-off를 선택할 수 있다는 사실이다.

현재 waypoint layer score는 개념적으로

\[
v=\frac{[a]_+}{\lVert[a]_+\rVert_2}
\]

이고, 모두 non-positive이면 best one-hot으로 fallback한다. 이는
`v\ge0,\lVert v\rVert_2=1`에서 현재 rewrite slope `a^\top v`를 최대화하는
greedy rule이다. Simplex probability가 아니며, history, cumulative capacity,
held-out preservation 또는 remaining layer slack을 목적함수에 넣지 않는다.
구현이 hop마다 coefficient를 다시 계산한다는 사실과, 그 변화가 direct-z gain의
원인임을 인과적으로 보였다는 주장은 서로 다르다.

필요한 최소 local factorial은 하나의 common shadow state `W_s`를 고정한 뒤
direction basis `B`와 share `v`를 각각 고정/refresh하는 다음 네 cell이다.

| Cell | Direction basis | Layer share | 식별하는 역할 |
|---|---|---|---|
| C | `B_0` fixed/transported | `v_0` fixed | static split control |
| B | `B_0` fixed/transported | `v_s` refreshed | share adaptation |
| D | `B_s` refreshed | `v_0` fixed | direction relinearization |
| A | `B_s` refreshed | `v_s` refreshed | bundled BF path |

현재 direct-z BF는 A형으로 direction과 share를 함께 refresh한다. 빠진 D cell을
같은 shadow state에서 추가하면

\[
\mathcal I_{B\times v}(W_s)
=\mathcal S_s(B_s,v_s)-\mathcal S_s(B_s,v_0)
-\mathcal S_s(B_0,v_s)+\mathcal S_s(B_0,v_0)
\]

로 두 요소의 local interaction을 분리할 수 있다. 서로 다른 descendant state를
각 cell이 스스로 만드는 full rollout에서 같은 second difference를 계산하면
state-distribution mediation까지 포함한 policy-level interaction이지 위 pure
local interaction은 아니다. 별도 GH adaptive-MEMIT 진단에는 terminal report가
존재하지만, 그 A/B/C/G arm과 teacher-forced estimand는 본 direct-z fixed panel의
actual-write contrast와 다르다. 따라서 그 수치와 selector 판정은 본 식의
direction/share 항을 식별하는 근거로 사용하지 않으며, 해당 GH 결과의 해석은
자체 report에만 남긴다.

### 현재 ODE는 어디까지 실제로 들어갔는가

Proposal의 목표 vector field는

\[
\dot W_l=v_l^*(W,\Delta,\Phi)\widehat B_l(W,z)
\]

이며, `v_l^*`가 rewrite deficit뿐 아니라 cumulative capacity와 barrier slack을
보면서 layer responsibility와 total velocity를 조절하고 first-hitting state에서
멈추는 구조다. 현재 direct-z experiment의 구현 범위는 다음과 같다.

| ODE 요소 | 현재 상태 | 해석 |
|---|---|---|
| edit당 frozen direct-z 1회 | 구현 | semantic target은 고정 |
| same-snapshot READ--PROPOSE와 simultaneous partial commit | 구현 | layer-axis BF skeleton |
| descendant-state direction/share 재계산 | 구현 | state-dependent vector field |
| unit-C normalization, fixed `K=4` Euler-like path | 구현 | controlled local trajectory |
| cumulative `\Psi_l`를 controller state로 사용 | 미구현 | history-aware routing 아님 |
| minimum-capacity QP와 per-layer barrier/cap | 미구현 | capacity-aware ODE 아님 |
| deficit-dependent total velocity, first-hit/free terminal | 미구현 | budget을 항상 끝까지 사용 |
| trust-ratio-based step accept/shrink/reject와 rejected-step rollback | 미구현 | adaptive solver stability claim 불가 |
| adaptive `h/K`, Heun, refinement convergence | 미구현 | ODE accuracy claim 불가 |
| persistent sequential history/cache | 미구현 | lifelong claim 불가 |

따라서 현재 허용되는 명칭은 **waypoint-conditioned synchronous
re-proposal/controller path**, 더 넓게는 fixed-step state-dependent ODE skeleton이다.
아직 empirical하게 **cumulative-capacity-aware ODE editor**라고 부를 수는 없다.
ODE가 충분히 발휘되는지를 method 단계에서 보려면 단순 K 증가가 아니라, state가
바뀔 때 `B_s`, `v_s`, `\Psi_s`, step acceptance와 stopping이 실제로 달라지고 그
변화가 static/frozen control보다 나은 frontier를 만드는지를 검증해야 한다.

### BF preservation 악화 관측과 가능한 설명

Same-family BF preservation은 MEMIT에서 Llama `-0.09631979`, Qwen
`-0.08699755`, genuine-Alpha에서 각각 `-0.14487684`, `-0.20097198`로 모두
평균 악화했다. 이 관측과 일관된 한 가지 가능한 설명은 controller-objective
mismatch다. 현재 panel은 preservation 악화의 원인을 인과적으로 식별하지
않는다. 현 controller는 immediate rewrite slope만 최대화하며 다음을 고려하지
않는다.

1. C-energy와 held-out Fisher/KL geometry는 일반적으로 다르다.
2. 모든 K=4 hop과 고정 budget을 사용하므로 target이 이미 충분해도 over-write할
   수 있다.
3. rewrite에 민감한 layer가 preservation에도 민감할 수 있으나 현재 share에는
   Fisher, old-edit curvature, capacity slack이 없다.
4. Alpha projector가 고정돼도 descendant-state의 functional Jacobian/key와
   approximate null-space leakage는 변할 수 있다.

`\Delta_{BF}=\Delta_D+\delta`일 때

\[
\mathrm{KL}_{hold}(\Delta_{BF})-\mathrm{KL}_{hold}(\Delta_D)
\simeq\Delta_D^\top F_{hold}\delta
+\frac12\delta^\top F_{hold}\delta
\]

의 부호는 matched C-budget이나 state refresh만으로 정해지지 않는다. 그러므로
preserved-key geometry, accumulated-write state 또는 norm anchor를 새 controller
objective에 결합하면 나아질 **가능성**은 있지만, 이는 현재 결과의 설명이
아니라 새 method hypothesis다. Direct-z fidelity가 더 높아도 preservation이
악화했고, Llama MEMIT에서는 direct-z fidelity가 낮아도 output proxy가 좋아졌다.
이 관측은 direct-z fidelity가 preservation의 충분조건이 아니며, local
output-proxy improvement의 필요조건도 아님을 강조한다.

### Sequential 강점의 수식적 가설과 현재 증거의 공백

Stream의 `t`번째 update와 그 직전 누적 displacement, positive normalization을

\[
u_{t,l}:=W_{t,l}-W_{t-1,l},\qquad
S_{t-1,l}:=W_{t-1,l}-W_{0,l},\qquad
D_l:=\operatorname{tr}\!\left(W_l^0C_l(W_l^0)^\top\right)+\varepsilon>0
\]

로 두자. Normalized covariance load는

\[
\Psi_{t-1,l}=\frac{\lVert S_{t-1,l}\rVert_{C_l}^2}{D_l},
\]

이고 한 edit의 marginal cost는

\[
\Delta\Psi_{t,l}:=\Psi_{t,l}-\Psi_{t-1,l}
=\frac{2\langle S_{t-1,l},u_{t,l}\rangle_{C_l}
+\lVert u_{t,l}\rVert_{C_l}^2}{D_l}.
\]

핵심은 per-edit norm만이 아니라 이미 쌓인 write와 새 write의 cross term이다.
동일 규모 update가 반복되는 이상화에서 같은 C-direction으로 정렬되면 squared
cumulative load가 `O(t^2)`로 커질 수 있고, 서로 C-orthogonal하면 `O(t)`에
가까울 수 있다. 따라서 dynamic BF의 설득력 있는 sequential 가설은 “매 edit
update가 더 작다”가 아니라, matched acquisition에서 high-marginal-cost layer의
progress를 다른 feasible layer로 넘겨

\[
\sum_l\Delta\Psi_l(\mathrm{dynamic})
<\sum_l\Delta\Psi_l(\mathrm{fixed/ordered\ control})
\]

를 만드는 것이다. 여기서 comparator는 향후 sequential protocol에서 ordered
native 또는 fixed-allocation control 중 하나로 사전 고정해야 한다. Proposal의
capacity QP와 barrier가 필요한 정확한 이유다.

그러나 covariance capacity는 preservation theorem이 아니다.
`F_{hold}:=F_{hold}(W_0)`,
`g_{j,t-1}:=\nabla\ell_j(W_{t-1})`,
`H_{j,t-1}:=\nabla^2\ell_j(W_{t-1})`라 두면, quadratic/local approximation
아래 누적 held-out drift와 과거 edit `j`의 incremental damage는 각각

\[
\begin{aligned}
\Delta\mathrm{KL}_t
&\simeq S_{t-1}^\top F_{hold}u_t+\tfrac12u_t^\top F_{hold}u_t,\\
\Delta\ell_{j\leftarrow t}
&\simeq g_{j,t-1}^\top u_t+\tfrac12u_t^\top H_{j,t-1}u_t
\end{aligned}
\]

이므로 `C_l`, `F_{hold}`, `H_{j,t-1}`의 정렬을 별도 sequential intervention으로
검증해야 한다. 현재 direct-z panel은 매 case 종료 후 W0로 rollback하므로
`S_{t,l}`, edit age, old-edit retention과 collapse onset이 존재하지 않는다.

### K=4의 의미와 adaptive K의 단계

K=4는 **한 rewrite case를 네 개의 sequential fact로 편집한 것**이 아니라, 한
fact를 쓰는 내부 integration waypoint를 네 번 둔 것이다. Direct-z는 case당
한 번 계산해 고정하고, 각 waypoint에서 current-state proposal/share를 다시
계산한다. 동일한 frozen additive update를 네 등분해 합치면 exact arithmetic에서
one-shot endpoint와 같으므로, waypoint 수만 늘리는 것은 다른 endpoint를 만들지
않는다. 별도 control의 수치를 이 terminal direct-z panel의 arm처럼 사용하지
않는다.

반대로 K=4 하나만으로 solver convergence나 optimal K도 말할 수 없다. 이 kill
test에서는 K=4를 고정하는 편이 맞다. K와 stopping까지 함께 바꾸면 direction,
allocation, compute와 numerical resolution이 동시에 변하기 때문이다. Adaptive
`h/K`, first-hit, trust-ratio와 Euler/Heun consistency는 local mechanism이 살아남은
뒤 method stage에서 `K\in\{1,2,4,8\}`, frozen split, equal-NFE control과 함께
검증한다.

### Motivation closure와 다음 stage 판정

현재 판정은 범위를 세 층으로 나눠야 한다.

| 층 | 판정 | 현재 허용되는 결론 |
|---|---|---|
| direct-z atomic/local mechanism | **closed-positive within this fixed panel** | state-conditioned path가 genuine-Alpha native보다 same-C에서 더 나은 local z/edit endpoint를 선택할 수 있음 |
| sequential capacity/preservation motivation | **open; no terminal evidence in this report** | 별도 precommitted GH 4-edit sentinel은 evidence freeze에서 제외되며 direct-z panel만으로는 판정 불가 |
| deployable ODE-Edit method superiority | **not established; method-stage evaluation not entered** | preservation, compute, BF-vs-DF, adaptive solver와 long horizon 검증 필요; tested absolute z/edit는 MEMIT-BF 우세 |

Atomic 개선은 가능성 실험으로는 유의미하다. Genuine-Alpha BF−native z lift는
Llama `+0.02056762` (7/8), Qwen `+0.08233046` (6/8), output-NLL lift는 각각
`+3.78883688` (8/8), `+0.05692602` (7/8)다. 두 모델의 방향 일치와 positive
Alpha−MEMIT interaction은 mechanism signal을 준다. 하지만 `n=8/model`, lenient
gate, teacher-forced/local proxy, fixed K=4이고 Alpha-BF의 absolute z/edit는
MEMIT-BF보다 낮으며 21 NFE 더 비싸다. 따라서 논문 method improvement의
효과크기나 일반성을 확정한 것은 아니다.

추가 Motivation 실험 범위를 제한한다면 별도 precommitted 4-edit chain은 **마지막
harm/specialization sentinel로만** 사용한다. 이는 sequential capacity/preservation
claim을 닫기에 충분한 실험이 아니다. 양성이어도 short-horizon common directional signal까지만,
무신호면 sequential preservation narrative를 버리고 atomic mechanism만 남기며,
해로우면 현재 always-refresh cumulative controller를 kill한다. 이를 구제하기
위해 Motivation에서 capacity-aware weighted ODE, 여러 edit order, 16--32 edits,
10K stream까지 확장하는 것은 너무 무겁다. 1-batch-size, 10-step 한 order 실험도
4-edit 결과가 정말 모호할 때만 선택적 sanity check이며 collapse proof가 아니다.

그러므로 bounded GH sentinel은 한 번 precommit대로 마치되, 결과와 무관하게
full capacity-QP/barrier, 2×2 direction×share ablation, deployable DF/K=1,
compute-matched K sweep와 multi-order sequential evaluation은 method stage로 넘기는
것이 맞다. Original proposal의 H3--H5와 long-horizon GO는 이 direct-z report로
닫히지 않으며, local mechanism closure와 구분한다.

이 Discussion이 허용하는 최종 claim은 다음 한 문장이다.

> 고정 atomic panel에서 proposal direction과 layer share를 함께 갱신하는
> waypoint-conditioned synchronous re-proposal/controller path는 genuine-Alpha
> geometry 아래 one-pass native보다 frozen direct-z와 local edit proxy가 더 좋은
> same-C endpoint를 선택할 수 있었다. 이 bundled contrast는 direction refresh와
> share refresh의 개별 기여를 식별하지 않으며, BF-over-DF, smaller-update 또는
> preservation 보장, capacity-aware stability, sequential collapse 방지,
> compute-efficient method superiority를 확립하지 않는다.
