# Direct-z 최종 synthesis — MEMIT × AlphaEdit p0

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
