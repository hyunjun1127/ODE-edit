# AlphaEdit direct-z 가능성 진단 — Llama/Qwen pair synthesis p0 v1

## Pair-level 판단

두 모델의 locked analyzer verdict는 각각
`PAIRED_ALPHA_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION`이며 기술적으로 모두
유효하다. 모델을 합치거나 평균내지 않았고, 모든 gate는 모델별로 따로
적용했다. Precommitted lenient gate는 **paired mean > 0과 positive cases
5/8 이상**이며 bootstrap CI는 설명용이다.

모델별 판단을 나란히 놓으면 다음의 제한된 motivation은 닫힌다.

- Genuine native Alpha는 no-op보다 frozen direct-z를 실제 weight update로
  부분적으로 쓴다. Z-residual 감소는 Llama `+0.28445854` (8/8), Qwen
  `+0.46528274` (8/8)이다.
- 동일 endpoint C-energy에서 genuine-Alpha BF K=4는 native보다 두 모델
  모두 direct-z residual과 edit proxy를 개선한다. BF−native z-residual
  개선은 Llama `+0.02056762` (7/8), Qwen `+0.08233046` (6/8),
  output-NLL reduction은 각각 `+3.78883688` (8/8), `+0.05692602`
  (7/8)이다.
- 이 공통 local signal은 preservation, off-token spill, compute 개선이
  아니다. BF−native preservation score는 Llama `-0.14487684` (0/8),
  Qwen `-0.20097198` (1/8), spill 개선은 `-0.03609012` (0/8)와
  `-0.01056363` (2/8), NFE 개선은 두 모델 모두 `-19` (0/8)이다.
- Frobenius norm은 모델 의존적이다. BF가 Llama에서는 `0.02535538` 더
  크고(감소 effect `-0.02535538`, 0/8), Qwen에서는 `0.10153338` 더
  작다(7/8). 따라서 “BF가 update norm 자체를 더 작게 만든다”는 공통
  claim은 성립하지 않는다.

가장 강한 허용 문장은 “state-refreshed genuine-Alpha BF 경로가 one-pass
ordered genuine baseline과 같은 C-budget에서 두 모델 모두 더 나은 local direct-z
actual-write와 edit signal을 선택할 수 있다”이다. 이 실험만으로 순수
integrator 효과, model preservation, method superiority, lifelong retention,
sequential-collapse 방지를 주장할 수는 없다.

## 실행·분석 contract

- 고정 모델은 `llama3-8b-inst`와 `qwen2.5-7b-inst`, 각 8 cases, 8 arms,
  64 outcomes이다. 두 모델 모두 ITD 8/8, 실패 0, technical validity
  `true`다.
- Frozen direct-z는 모델·case별로 정확히 한 번 load되고 recompute는 0회다.
  Replay/target anchor, precomputed projector와 covariance only, projector
  byte integrity, zero-`cache_c`, receipt-before-outcome, scalar firewall,
  exact W0/RNG rollback이 두 모델 모두 통과했다.
- Genuine ordered solve는 case당 한 번이고, BF의 네 refresh는 모두 genuine
  Alpha direction이다. Post-hoc arm은 identity-P Alpha base 뒤의 right
  projection만 사용한다.
- Genuine C-matched, post-hoc C-matched, BF endpoint가 각 case의 같은
  reference C-energy를 만족하며 cone도 같은 cap 안에 있다. Projector leak는
  관찰 축이지 downstream preservation 보증이나 validity gate가 아니다.
- Dedicated two-GPU Slurm job은 `15755`다. Llama child `15755.0`은
  `COMPLETED`, exit `0:0`, elapsed `01:56:47`, MaxRSS `5,830,048K`; Qwen
  child `15755.1`은 `COMPLETED`, exit `0:0`, elapsed `02:14:18`, MaxRSS
  `8,904,392K`다.
- 각 모델의 총 ITD NFE는 `488`이다. Analysis SHA-256은 Llama
  `bf3eb0e774e571a102659600f089d982f40083c016c664531ee19c2735e54eb0`,
  Qwen
  `5c4676adec7de5e0fa5495842702fd991b8f93251b4d95c312490d1340756c1e`다.

## BF 대 genuine native — 공통 signal과 상충 축

아래 값은 모두 BF−genuine C-matched의 sign-normalized paired effect다.
양수는 BF가 해당 축에서 더 좋다는 뜻이며, 각 모델의 count를 독립적으로
표기했다.

| 축 | Llama mean; +/8 | Qwen mean; +/8 | Pair 해석 |
|---|---:|---:|---|
| z-residual 감소 | `+0.02056762`; 7/8 | `+0.08233046`; 6/8 | 두 모델 PASS |
| target delta gain | `+0.14513059`; 8/8 | `+0.13776107`; 8/8 | 두 모델 PASS |
| target delta cosine | `-0.18616539`; 0/8 | `-0.01412005`; 2/8 | 두 모델 fail |
| generated mean error 감소 | `+0.02623012`; 7/8 | `+0.10533109`; 7/8 | 두 모델 PASS |
| generated worst error 감소 | `+0.04220109`; 7/8 | `+0.10699570`; 7/8 | 두 모델 PASS |
| output-NLL reduction | `+3.78883688`; 8/8 | `+0.05692602`; 7/8 | 두 모델 PASS |
| output progress | `+4.91699534`; 8/8 | `+0.71733285`; 7/8 | 두 모델 PASS |
| exact margin | `+4.70847762`; 8/8 | `+0.72356713`; 6/8 | 두 모델 PASS |
| exact satisfaction | `+0.5`; 4/8 | `0`; 0/8, 8 ties | 두 모델 gate fail |
| paraphrase-NLL reduction | `+3.26029124`; 8/8 | `+0.88091911`; 7/8 | 두 모델 PASS |
| preservation score (`-KL`) | `-0.14487684`; 0/8 | `-0.20097198`; 1/8 | 두 모델 악화 |
| off-token spill 감소 | `-0.03609012`; 0/8 | `-0.01056363`; 2/8 | 두 모델 악화 |
| endpoint C-energy 감소 | `-3.41e-13`; 3/8 | `-2.42e-9`; 4/8 | Contract상 matched |
| Frobenius norm 감소 | `-0.02535538`; 0/8 | `+0.10153338`; 7/8 | 모델별 반대 |
| projector violation 감소 | `-3.18e-8`; 3/8 | `-2.48e-9`; 4/8 | 공통 signal 없음 |
| NFE 감소 | `-19`; 0/8 | `-19`; 0/8 | 두 모델 compute 악화 |

두 모델에서 target delta gain은 커지지만 cosine은 낮아진다. BF가 native
방향을 단순 확대했다기보다, target 축 성분을 키우면서 전체 방향을 다시
배치했다는 local geometry signal이다. 이 재배치는 z-residual·generated
error·edit proxy에는 공통 이득이지만 held-out KL과 spill에는 공통 손해다.

동일 C-energy가 동일 Frobenius norm을 뜻하지 않는 것도 중요하다. C-energy를

\[
E_C(\Delta W)=\operatorname{tr}(\Delta W C\Delta W^\top)
\]

로 보면, 같은 `E_C`에서도 covariance의 고유방향에 따라
`||Delta W||_F`는 달라질 수 있다. 실제로 그 ordering이 Llama와 Qwen에서
반대다. Qwen의 smaller-norm signal은 모델-local 결과이고, pair 공통
수식적 보장은 아니다.

## Native/BF actual-write와 direct-z fidelity의 의미

| Model / arm vs no-op | z-residual 감소; +/8 | output-NLL reduction; +/8 | preservation; +/8 | spill 감소; +/8 | Fro norm 감소; +/8 | NFE/case |
|---|---:|---:|---:|---:|---:|---:|
| Llama genuine C | `+0.28445854`; 8/8 | `+4.16866735`; 8/8 | `-0.12773744`; 0/8 | `-0.04110590`; 0/8 | `-0.33386235`; 0/8 | 5 |
| Llama BF | `+0.30502616`; 8/8 | `+7.95750422`; 8/8 | `-0.27261428`; 0/8 | `-0.07719602`; 0/8 | `-0.35921773`; 0/8 | 24 |
| Qwen genuine C | `+0.46528274`; 8/8 | `+10.45983613`; 8/8 | `-0.92632730`; 0/8 | `-0.04357059`; 0/8 | `-2.57325908`; 0/8 | 5 |
| Qwen BF | `+0.54761319`; 8/8 | `+10.51676214`; 8/8 | `-1.12729928`; 0/8 | `-0.05413422`; 0/8 | `-2.47172570`; 0/8 | 24 |

Native와 BF 모두 두 모델에서 direct-z actual-write gate를 통과한다. 그러나
actual write가 exact edit success와 동일하지도 않다. Qwen은 native와 BF가
모두 exact satisfaction 8/8인 반면, Llama는 native 0/8, BF 4/8이다.
그리고 네 deployable write 모두 no-op보다 preservation과 spill이 나쁘다.
따라서 direct-z fidelity는 유용한 representation diagnostic이지만 edit
success나 model preservation의 충분조건은 아니다.

## Genuine P-inside-solve 대 post-hoc right projection

Genuine Alpha와 post-hoc ablation은 각각

\[
\Delta W_{gen}=R(\lambda I+K^\top PK)^{-\top}(PK)^\top,
\qquad
\Delta W_{post}=R(\lambda I+K^\top K)^{-\top}K^\top P
\]

이므로 일반적으로 같지 않다. 첫 식은 `P`가 normal system 안에 있고,
둘째 식은 identity-P base를 푼 뒤 이미 정한 update를 투영한다.

| Genuine C − post-hoc C 축 | Llama mean; +/8 | Qwen mean; +/8 | Pair 해석 |
|---|---:|---:|---|
| z-residual 감소 | `-0.01133774`; 0/8 | `-0.09317304`; 0/8 | Post-hoc fidelity 우세 |
| target delta gain | `-0.01038429`; 0/8 | `-0.09336335`; 0/8 | Post-hoc 우세 |
| target delta cosine | `-0.01227809`; 0/8 | `-0.02618068`; 0/8 | Post-hoc 우세 |
| generated mean error 감소 | `-0.01003402`; 0/8 | `-0.08737324`; 0/8 | Post-hoc 우세 |
| output-NLL reduction | `-0.24165694`; 0/8 | `-0.04695102`; 0/8 | Post-hoc edit 우세 |
| output progress | `-0.25153236`; 0/8 | `-0.79576240`; 0/8 | Post-hoc edit 우세 |
| preservation score | `+0.00407069`; 8/8 | `+0.09821588`; 7/8 | Genuine 두 모델 PASS |
| off-token spill 감소 | `+0.00010584`; 4/8 | `+0.00322912`; 6/8 | Qwen만 PASS |
| Frobenius norm 감소 | `-0.00282356`; 0/8 | `-0.01858916`; 3/8 | Genuine norm 이점 없음 |
| projector violation 감소 | `-5.61e-9`; 4/8 | `+1.18e-8`; 4/8 | 분리 signal 없음 |
| NFE 감소 | `0`; 8 ties | `0`; 8 ties | 같은 5 NFE |

수식적 비가환성은 분명하지만 scalar 우월 ordering을 만들지는 않는다. 이
panel에서는 post-hoc가 두 모델 모두 fidelity/edit에 유리하고, genuine은
preservation에 유리하다. Genuine의 spill 이점은 Qwen에서만 gate를 넘는다.
따라서 null-space geometry는 fidelity와 보존 사이 trade-off를 바꾸는
메커니즘이지, 모든 축을 동시에 개선하는 보증이 아니다.

## Cone reachability와 oracle ceiling

| Model / diagnostic vs no-op | z-residual 감소; +/8 | output-NLL reduction; +/8 | preservation; +/8 | spill 감소; +/8 | NFE/case |
|---|---:|---:|---:|---:|---:|
| Llama genuine cone | `+0.35262485`; 8/8 | `+6.78356875`; 8/8 | `-0.20876647`; 0/8 | `-0.04260142`; 0/8 | 15 |
| Llama oracle do-z | `+0.99999999`; 8/8 | `+10.42197685`; 8/8 | `-1.18917908`; 0/8 | `0`; ties | 1 |
| Qwen genuine cone | `+0.37876187`; 8/8 | `+10.43335807`; 8/8 | `-0.94572745`; 0/8 | `-0.02567594`; 0/8 | 15 |
| Qwen oracle do-z | `+0.99999999`; 8/8 | `+10.54005588`; 8/8 | `-1.01225985`; 0/8 | `0`; ties | 1 |

두 cone이 모두 z/output gate를 통과하므로 같은 C-cap의 genuine Alpha response
set 안에 replayed direct-z를 부분적으로 향하는 방향이 존재한다. 다만 cone은
W0의 다섯 unit-C response에 대한 non-negative ridge 진단으로 비배포성이다.
Oracle은 frozen delta를 activation에 직접 넣는 weight-free ceiling이며 두
모델 모두 residual을 사실상 완전히 제거한다. 두 ceiling 모두 preservation을
악화시키므로 reachability나 direct-z 충실도가 benign downstream write를
보장하지 않는다.

## Natural full budget이 보여 주는 fidelity–cost 관계

Genuine full−C-matched의 z-residual 개선은 Llama `+0.69010628` (8/8),
Qwen `+0.52851892` (8/8)이고 output-NLL reduction도 각각
`+6.25140007` (8/8), `+0.07670711` (8/8)이다. 동시에 preservation은
`-0.89236980` (0/8), `-0.06808518` (2/8), spill은 `-0.09050256`
(0/8), `-0.02819981` (0/8), Frobenius norm 감소는 `-0.61223190`
(0/8), `-1.71993665` (0/8)다.

Post-hoc full−matched도 두 모델 모두 z와 output은 8/8 개선하지만
preservation, spill, C-energy, Frobenius norm은 개선하지 못한다. 즉 더 큰
natural budget으로 direct-z를 더 충실히 쓰는 것은 가능하지만, 작은 update나
보존 이점으로 자동 전환되지 않는다.

## NFE와 compute trade-off

두 모델은 동일하게 no-op 1, oracle 1, genuine full 5, genuine C-matched 5,
post-hoc full 5, post-hoc C-matched 5, BF 24, cone 15 NFE/case를 사용한다.
모델별 총계는 `61 x 8 = 488`이다. BF는 native보다 19 NFE/case, 즉
`4.8x`를 쓴다. 따라서 관찰된 BF 이점은 favorable accuracy/compute
frontier가 아니라 추가 relinearization을 허용했을 때의 mechanism signal이다.

## 순수 integrator attribution이 불가능한 이유

고정 vector field라면 네 Euler quarter-step은 한 full step과 같다.

\[
W_4=W_0+\sum_{k=0}^{3}\tfrac14 F(W_0)=W_0+F(W_0).
\]

따라서 단순히 같은 update를 네 조각으로 나누는 것만으로는 endpoint 이득이
생기지 않는다. BF signal이 생기려면 최소한 `F(W_k)`가 descendant state에
따라 달라져야 한다. 이번 결과에서 같은 C-energy인데도 delta gain은 두 모델
모두 증가하고 cosine은 감소한 것은 그런 방향 변화와 양립한다.

그러나 현재 BF−native contrast를 **순수 ODE integrator 효과**로 단독
귀속할 수는 없다.

- Native control은 layer 4--8을 ordered하게 제안하며, 앞 layer의 temporary
  write 뒤 현재 key와 z를 측정한다.
- BF는 각 quarter-step descendant에서 다섯 layer의 genuine direction을
  synchronous하게 다시 만들고 probe-selected path를 누적한 뒤 endpoint를
  reference C-energy로 한 번 더 normalize한다.
- BF는 native의 5 NFE가 아니라 24 NFE를 사용한다.

즉 관찰 contrast에는 descendant relinearization뿐 아니라 ordered-versus-
synchronous construction, 반복 probe/controller, 추가 compute가 함께 들어간다.
Static synchronous K=1 control 또는 동일 scheduler를 유지한 ordered K=4
control이 없으므로 각각의 기여를 식별할 수 없다. 현재 수식적 이점은 “고정
field splitting”이 아니라 **state-dependent proposal field와 경로 선택의
가능성**으로 표현해야 한다.

## 최종 claim boundary

이번 Alpha pair가 지지하는 것은 다음 네 가지다.

1. 두 모델의 native Alpha baseline이 frozen direct-z를 실제 weight update로
   부분적으로 구현한다.
2. 같은 C-budget의 refreshed BF가 두 모델 모두 native보다 local z-write와
   여러 edit proxy를 개선한다.
3. Genuine response cone 안에 두 모델 모두 bounded reachability가 있다.
4. Fidelity, edit, preservation, spill, C-energy, Frobenius norm, NFE는 서로
   분리된 축이며 direct-z fidelity가 보존을 보장하지 않는다.

지지하지 않는 것은 method superiority, 보편적 smaller-norm claim, broad
downstream preservation, pure integrator attribution이다. 각 case는 fresh W0의
한 atomic edit이고 K=4는 그 한 edit 안의 numerical path다. Sequential edit,
long-horizon retention, collapse trajectory가 없으므로 lifelong editing이나
sequential-collapse 방지 claim도 전혀 평가되지 않았다. Llama와 Qwen 결과는
끝까지 모델-local로 유지하며 합산 score나 pooled gate를 만들지 않는다.
