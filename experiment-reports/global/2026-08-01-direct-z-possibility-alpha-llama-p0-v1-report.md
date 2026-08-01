# Llama AlphaEdit direct-z 가능성 분석 — p0 v1

## 결론

`llama3-8b-inst`의 고정 8-case AlphaEdit panel은 기술적으로 유효하며,
analyzer 판정은
`PAIRED_ALPHA_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION`이다. 가장 유용한
국소 signal은 다음 세 가지다.

1. 같은 endpoint C-energy에서 genuine native Alpha와 BF 모두 no-op보다
   frozen direct-z를 실제 weight write로 반영했고 lenient directional gate를
   통과했다. Native의
   z-residual 감소는 평균 `0.28445854` (8/8), BF는 `0.30502616` (8/8)이다.
2. BF는 같은 C-energy의 genuine native보다 z-residual을 평균
   `0.02056762` 더 줄였고(7/8), output-NLL reduction을 `3.78883688` 더
   높였다(8/8). 따라서 **Alpha proposal geometry 안에서는 state refresh가
   direct-z write와 edit proxy를 함께 높이는 local directional signal**이
   있다.
3. 그 이득은 보존이나 작은 update와 동반되지 않았다. BF는 native보다
   held-out preservation score가 `0.14487684` 낮고(0/8), spill은
   `0.03609012` 나쁘며(0/8), Frobenius norm은 `0.02535538` 더 크고(0/8),
   NFE도 5에서 24로 늘었다. 즉 이 run은 “BF가 더 작은 norm으로 더 잘
   쓴다”가 아니라, **동일 C-budget을 다른 방향과 경로에 배분해 더 강한
   local edit signal을 얻는 대신 preservation·spill·compute를 지불한다**는
   증거다.

Genuine P-inside-solve는 같은 C-budget의 post-hoc right projection보다
direct-z fidelity나 edit proxy에서 우세하지 않았다. 유일하게 gate를 넘은
genuine 이점은 작은 preservation 개선 `+0.00407069` (8/8)이다. 따라서
이 Llama panel만으로 genuine Alpha, BF, ODE-Edit 또는 어떤 baseline의
방법 우월성을 판정할 수 없다.

## 실행 및 기술 contract

- Run/model: `dzf_alpha_llama_p0_v1` / `llama3-8b-inst`; claim boundary는
  `atomic_geometry_signal_only_not_lifelong_or_method_superiority`이다.
- Slurm child `15755.0`은 `COMPLETED`, exit `0:0`, elapsed `01:56:47`,
  MaxRSS `5830048K`이다.
- Analyzer technical validity는 `true`: 8/8 cases 성공, 실패 0,
  `8 arms x 8 cases = 64` outcomes이다. Raw stream도 features 8, case events
  8, actions 8, exclusive receipts 8, outcomes 64이며 각 arm이 정확히 8개다.
- Frozen target은 case마다 정확히 한 번 load되고 compute-z는 총 0회다.
  Replay/target anchor가 exact이며 paired MEMIT source는
  `dzf_llama_p0_v2`, source manifest SHA-256은
  `2eda8963ae8b854ee58c1835ba6184bf1cdf88e8748ac46a208bb80542235e15`,
  replay lock은
  `8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e`이다.
- Existing covariance와 `(5, 14336, 14336)` projector만 mmap/read-only로
  사용했다. Projector start/end integrity, precomputed-only policy,
  zero-`cache_c` isolated first-edit policy가 모두 통과했다.
- Genuine ordered solve는 case당 한 번, BF는 네 descendant에서 모두
  genuine Alpha direction을 refresh했다. Post-hoc 두 arm은 동일한
  identity-P Alpha base를 만든 뒤 right projection만 했다.
- Receipt-before-outcome, scalar firewall, 모든 case의 W0/RNG exact rollback,
  BF endpoint binding이 통과했다. Genuine C-matched, post-hoc C-matched,
  BF endpoint의 reference C-energy match와 cone cap도 통과했다.
- 총 ITD NFE는 `488`이다. Analysis SHA-256은
  `bf3eb0e774e571a102659600f089d982f40083c016c664531ee19c2735e54eb0`이다.

아래 모든 paired effect는 analyzer와 같이 sign-normalized했다. 즉 양수는
표의 lhs가 해당 축에서 더 좋다는 뜻이다. Gate는 **평균 > 0 그리고 최소
5/8 positive cases**이며, 4,000회 paired-bootstrap interval의 zero exclusion은
요구하지 않는다.

## Native/BF의 absolute direct-z write

No-op의 평균 z-residual ratio는 `1.0000000004`이다. Genuine native
C-matched는 `0.71554146`, BF는 `0.69497384`로 내려간다. 따라서 두 방법
모두 weight write를 통해 frozen direct-z의 일부를 실제로 구현하며, BF가
이 panel에서는 더 많이 구현한다. 다만 residual 한 축만으로 edit나 보존을
대체하지 않는다.

| 축 (lhs vs no-op) | Genuine native C: mean (positive/8) | BF: mean (positive/8) |
|---|---:|---:|
| z-residual 감소 | `+0.28445854` (8/8) | `+0.30502616` (8/8) |
| target delta gain | `+0.29598827` (8/8) | `+0.44111886` (8/8) |
| target delta cosine | `+0.91338530` (8/8) | `+0.72721992` (8/8) |
| generated mean error 감소 | `+0.25771673` (8/8) | `+0.28394685` (8/8) |
| generated worst error 감소 | `+0.19492569` (8/8) | `+0.23712678` (8/8) |
| output-NLL reduction | `+4.16866735` (8/8) | `+7.95750422` (8/8) |
| output progress | `+4.29762298` (8/8) | `+9.21461831` (8/8) |
| exact margin | `+4.51496148` (8/8) | `+9.22343910` (8/8) |
| exact satisfaction | `0` (0/8; 8 ties) | `+0.5` (4/8; 4 ties) |
| paraphrase-NLL reduction | `+2.33755288` (8/8) | `+5.59784411` (8/8) |
| preservation score (`-KL`) | `-0.12773744` (0/8) | `-0.27261428` (0/8) |
| off-token spill 감소 | `-0.04110590` (0/8) | `-0.07719602` (0/8) |
| endpoint C-energy 감소 | `-0.000204936` (0/8) | `-0.000204936` (0/8) |
| endpoint Frobenius norm 감소 | `-0.33386235` (0/8) | `-0.35921773` (0/8) |
| projector violation 감소 | `-4.35864e-6` (0/8) | `-4.39042e-6` (0/8) |
| NFE 감소 | `-4` (0/8) | `-23` (0/8) |

BF의 delta gain은 더 크지만 cosine은 더 낮다. 즉 BF의 local 이득은 native
방향을 단순히 더 크게 복제한 결과가 아니다. Target-axis 성분을 더 크게
만들면서 전체 방향 정렬은 낮아지는 재배치가 일어났고, 이것이 residual과
output proxy에는 이득이지만 보존·spill에는 손해로 나타난다.

## Genuine P-inside-solve 대 post-hoc projection

Genuine isolated Alpha는

\[
U=PK,\quad M=\lambda I+K^\top U,\quad
\Delta W_{gen}=R M^{-\top}U^\top
\]

를 사용한다. Post-hoc arm은 같은 Alpha base를 `P=I`로 먼저 푼 뒤

\[
\Delta W_{post}=R(\lambda I+K^\top K)^{-\top}K^\top P
\]

만 적용한다. 일반적으로 `P`가 normal system 안에 들어가는 것과 이미 정한
update를 오른쪽 투영하는 것은 같지 않다. 이 비교는 MEMIT update를 post-hoc
가공한 것이 아니라 동일 Alpha base 안에서 그 차이만 격리한다.

| Genuine C − post-hoc C 축 | Normalized mean | Positive/8 | Gate |
|---|---:|---:|:---:|
| z-residual 감소 | `-0.01133774` | 0/8 | — |
| target delta gain | `-0.01038429` | 0/8 | — |
| target delta cosine | `-0.01227809` | 0/8 | — |
| generated mean error 감소 | `-0.01003402` | 0/8 | — |
| generated worst error 감소 | `-0.00697011` | 0/8 | — |
| output-NLL reduction | `-0.24165694` | 0/8 | — |
| output progress | `-0.25153236` | 0/8 | — |
| exact margin | `-0.21263736` | 0/8 | — |
| exact satisfaction | `0` | 0/8 (8 ties) | — |
| paraphrase-NLL reduction | `-0.13370255` | 0/8 | — |
| preservation score (`-KL`) | `+0.00407069` | 8/8 | PASS |
| off-token spill 감소 | `+0.00010584` | 4/8 | — |
| endpoint C-energy 감소 | `-8.07e-12` | 2/8 | — |
| endpoint Frobenius norm 감소 | `-0.00282356` | 0/8 | — |
| projector violation 감소 | `-5.61e-9` | 4/8 | — |
| NFE 감소 | `0` | 0/8 (8 ties) | — |

절대 평균도 같은 결론이다. Genuine/post-hoc의 z-residual은
`0.71554146 / 0.70420372`, output-NLL reduction은
`4.16866735 / 4.41032428`, preservation score는
`-0.12773744 / -0.13180813`, Frobenius norm은
`0.33386235 / 0.33103878`이다. P-inside-solve는 이 panel에서 작은 보존
이점은 보이지만 fidelity·edit·norm 이점은 보이지 않는다. Projector leak
차이는 `1e-8` 이하 규모이고 4/8이라 directional signal로 해석하지 않는다.

## BF 대 genuine native: 같은 C-budget의 핵심 비교

BF와 native의 평균 endpoint C-energy는 각각
`0.00020493583190`과 `0.00020493583156`이며, analyzer의 matched-budget
contract를 통과했다. C-energy 차이의 normalized mean은
`-3.41e-13`로 사실상 0이다.

| BF − genuine native C 축 | Normalized mean | Positive/8 | Gate |
|---|---:|---:|:---:|
| z-residual 감소 | `+0.02056762` | 7/8 | PASS |
| target delta gain | `+0.14513059` | 8/8 | PASS |
| target delta cosine | `-0.18616539` | 0/8 | — |
| generated mean error 감소 | `+0.02623012` | 7/8 | PASS |
| generated worst error 감소 | `+0.04220109` | 7/8 | PASS |
| output-NLL reduction | `+3.78883688` | 8/8 | PASS |
| output progress | `+4.91699534` | 8/8 | PASS |
| exact margin | `+4.70847762` | 8/8 | PASS |
| exact satisfaction | `+0.5` | 4/8 (4 ties) | — |
| paraphrase-NLL reduction | `+3.26029124` | 8/8 | PASS |
| preservation score (`-KL`) | `-0.14487684` | 0/8 | — |
| off-token spill 감소 | `-0.03609012` | 0/8 | — |
| endpoint C-energy 감소 | `-3.41e-13` | 3/8 | — |
| endpoint Frobenius norm 감소 | `-0.02535538` | 0/8 | — |
| projector violation 감소 | `-3.18e-8` | 3/8 | — |
| NFE 감소 | `-19` | 0/8 | — |

이 결과가 지지하는 ODE/BF 쪽 claim은 좁다. 네 번의 descendant
relinearization으로 같은 C-budget에서 native와 다른 유용한 local direction을
찾을 수 있다는 것이다. 더 작은 update, 더 낮은 spill, downstream 보존,
또는 계산 효율의 claim은 지지하지 않는다. 특히 C-energy match가 Frobenius
norm match를 뜻하지 않으며, BF는 모든 case에서 Frobenius norm이 더 컸다.

## Cone/oracle ceiling

| Diagnostic arm vs no-op | z-residual 감소 | output-NLL reduction | preservation score | generated mean error 감소 | spill 감소 | Positive-count 요약 |
|---|---:|---:|---:|---:|---:|---|
| Genuine cone | `+0.35262485` | `+6.78356875` | `-0.20876647` | `+0.32489132` | `-0.04260142` | z/output/generated 8/8; preservation/spill 0/8 |
| Oracle do-z | `+0.99999999` | `+10.42197685` | `-1.18917908` | `+0.99999996` | `0` | z/output/generated 8/8; preservation 0/8 |

Cone의 절대 z-residual은 `0.64737515`, C-energy는 cap 안의
`0.00020493583503`, Frobenius norm은 `0.29241373`, NFE는 15다. 이는
동일 budget의 genuine Alpha response cone 안에 direct-z를 향하는 실제 write
방향이 존재한다는 controllability signal이다. 그러나 cone은 non-negative
ridge로 고른 비배포 진단용 ceiling이다.

Oracle의 절대 z-residual은 `6.77e-9`로 frozen direct-z를 사실상 완전히
구현하고 exact satisfaction도 8/8이지만, weight를 쓰지 않는 activation
intervention이다. 동시에 held-out KL은 평균 `1.18917908`이다. 따라서
“direct-z를 충실히 구현하면 preservation도 좋아진다”는 보장은 이 가장
강한 ceiling에서도 성립하지 않는다.

Cone과 BF의 절대 평균 역시 단일 ordering을 주지 않는다. Cone은 BF보다
z-residual (`0.6474 < 0.6950`), preservation 손실 (`0.2088 < 0.2726`), spill
(`0.0426 < 0.0772`), Frobenius norm (`0.2924 < 0.3592`)이 작지만, BF의
output-NLL reduction은 더 크다(`7.9575 > 6.7836`). 이는 fidelity, edit,
보존, norm이 서로 대체 가능한 하나의 score가 아님을 재확인한다.

## Full natural budget의 민감도

| Full − C-matched 축 | Genuine Alpha: mean (positive/8) | Post-hoc Alpha: mean (positive/8) |
|---|---:|---:|
| z-residual 감소 | `+0.69010628` (8/8) | `+0.64494383` (8/8) |
| generated mean error 감소 | `+0.45504425` (8/8) | `+0.44736882` (8/8) |
| output-NLL reduction | `+6.25140007` (8/8) | `+6.00916520` (8/8) |
| output progress | `+12.85250372` (8/8) | `+12.45855566` (8/8) |
| exact satisfaction | `+1.0` (8/8) | `+1.0` (8/8) |
| preservation score (`-KL`) | `-0.89236980` (0/8) | `-0.85309715` (0/8) |
| off-token spill 감소 | `-0.09050256` (0/8) | `-0.08511156` (0/8) |
| endpoint C-energy 감소 | `-0.001468742` (0/8) | `-0.001331412` (0/8) |
| endpoint Frobenius norm 감소 | `-0.61223190` (0/8) | `-0.56869850` (0/8) |
| NFE 감소 | `0` (all ties) | `0` (all ties) |

Natural full genuine Alpha는 평균 z-residual을 `0.02543518`까지 낮추고
8/8 exact satisfaction을 달성하지만, C-energy는 `0.00167368`, Frobenius
norm은 `0.94609425`, held-out KL은 `1.02010725`, spill은 `0.13160846`이다.
즉 direct-z를 훨씬 충실히 쓰는 것이 edit proxy에는 유리하지만 보존과 작은
update에 자동으로 유리하지 않다는 budget-response를 아주 선명하게 보인다.

## NFE와 효율

Arm별 평균 NFE는 no-op 1, oracle 1, genuine full 5, genuine C-matched 5,
post-hoc full 5, post-hoc C-matched 5, BF 24, cone 15이며 총 `61 x 8 = 488`이다.
BF는 genuine native 대비 case당 19 NFE를 더 사용한다. 따라서 현재 signal은
sample/compute efficiency가 아니라, 추가 relinearization으로 얻는 local
directional benefit이다.

## 기존 Llama MEMIT 결과와의 제한적 방향 대조

기존 MEMIT Llama report의 동일 C-budget 비교에서 BF−native는 z-fidelity
`-0.13500684` (0/8), output-NLL reduction `+0.16067749` (8/8), preservation
`-0.09631979` (0/8), raw Frobenius norm 차이 `+0.04089099` (BF가 큼, 8/8)이었다.
이번 Alpha comparison은 각각 `+0.02056762` (7/8), `+3.78883688` (8/8),
`-0.14487684` (0/8), raw Frobenius 차이 `+0.02535538` (BF가 큼, 8/8)이다.

따라서 MEMIT에서는 보이지 않았던 BF의 positive z-write signal이 genuine
Alpha geometry에서는 나타났다. 이는 BF의 잠재 이점이 단순 step splitting
자체가 아니라 proposal/solver geometry와 state-dependent refresh의 결합에
의존할 수 있다는 동기와 부합한다. 그러나 이 문단은 두 기존 model-local
report의 방향을 나란히 둔 것이며, exact cross-track join이나 formal
difference-in-differences가 아니다. Alpha가 MEMIT보다 우월하다는 판정으로
사용할 수 없다.

## 최종 해석 경계

이 run에서 허용되는 가장 강한 해석은 다음과 같다.

- Frozen direct-z는 Alpha의 genuine weight geometry 안에서 부분적으로
  reachable하며, BF refresh는 같은 C-budget의 native보다 z-write와 여러
  edit proxy를 높이는 local signal을 보였다.
- Genuine P-inside-solve와 post-hoc projection은 수식적으로 다르지만,
  이 panel에서 genuine의 관찰 이점은 작은 preservation signal에 한정됐다.
- Direct-z fidelity, output edit, preservation, spill, C-energy, Frobenius
  norm과 NFE는 분리된 축이다. 한 축의 양의 결과를 종합 method score로
  바꾸지 않는다.
- 이 실험은 fresh W0에서 한 fact를 편집하는 atomic diagnostic이며 K=4는
  한 edit 내부의 numerical path다. Lifelong editing, sequential retention,
  sequential-collapse 방지, 장기 downstream capability 보존을 측정하지 않았다.
- Llama 단일 8-case panel이므로 Qwen과 pooling하지 않으며, AlphaEdit,
  MEMIT, BF, ODE-Edit, cone 또는 oracle 사이의 방법 우월성을 주장하지 않는다.

따라서 motivation에 사용할 수 있는 문장은 “ODE/BF refresh가 genuine Alpha
proposal geometry 안에서 동일 C-budget의 direct-z actual-write와 edit
signal을 높일 수 있다”까지다. “더 작은 update로 보존까지 개선한다” 또는
“lifelong collapse를 막는다”는 문장은 이 결과로 지지되지 않는다.
