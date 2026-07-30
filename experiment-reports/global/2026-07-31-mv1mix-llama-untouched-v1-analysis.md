# MV-1 untouched Llama 독립 분석

## 결론

Llama3-8B-Instruct의 locked untouched 20-case run은 기술적으로 유효하며,
MV-1 spec §10.9의 단일-model 규칙을 적용하면 **`clear input`**이다.
Primary `score_mix - frozen_static_mix`는 mean `0.010262799263000489`,
trim20 `0.007329424222310384`, median `0.007647037506103516`,
positive sign `20/20`이고, paired case bootstrap mean 95% CI는
`[0.005930529832839966, 0.015292656421661378]`이다.

이 판정은 Llama 한 model의 입력일 뿐이다. Pair, cross-model, method gain,
MV-2 개방, ODE claim은 계산하거나 승인하지 않는다.

## 네 범주 구분

| 범주 | 내용 |
| --- | --- |
| proposal에서 온 내용 | 이 독립 분석의 허용 입력에는 proposal 원문이 없었다. 따라서 새로운 proposal claim을 추가하지 않았다. |
| repo/protocol에서 확인한 사실 | 고정 analyzer, Llama run manifest/summary, frozen Llama policy, MV-1 spec §10.9에서 아래 artifact·metric·gate 규칙을 확인했다. |
| GH 추정 | §10.9 규칙을 수치에 기계적으로 적용한 단일-model 판정은 `clear input`이다. 이는 paper claim이나 pair verdict가 아니다. |
| 사용자 확인 필요 | 단일-model 입력 계산에는 없음. Pair 판정 및 MV-2 개방은 별도 red audit 없이는 결정할 수 없다. |

## 실행 및 artifact lock

- Run: `mv1mix_llama_untouched_v1`
- Model: `llama3-8b-inst`
- Slurm: job `15586`, `odeedit_mv1mix_untouched_pair_v1`
- 실행 commit: `a7c929f11da540c7435f9b0f3d367c5009c1ff2d`
- 실행 당시 tracked worktree: clean
- 분석 확인 HEAD: `51d85971f825b237d7a79320c9c8eb9cf9cf92bd`
- Analyzer blob: `92ca2d491c1c8bc930f4b0bbe37520a92db3dfce`
- Analyzer는 실행 commit부터 위 확인 HEAD까지 unchanged
- Split/cases: canonical `untouched`, exact `20`
- Seed / `q`: `17` / `1/256`
- Arms: exact `6`, 총 outcome `120`
- Count: event `20`, feature `20`, action `20`, receipt `20`,
  local-only `direct_z` artifact `20`
- Completion: planned/attempted/pass `20/20/20`, failure `0`,
  abort 미실행 `0`
- `expected_counts_exact=true`, `all_rollbacks_exact=true`
- Equal-`C`/rollback validation 및 outcome firewall/receipt validation: 모두 true
- Receipt는 feature와 frozen action/forecast가 outcome보다 먼저
  write/flush/fsync되고 exclusive 생성되는 lock을 통과했다.
- Covariance: verified read-only file `5`개 사용; 재계산·download 없음
- Projector: preflight hash/size만 검증하고 deserialize `0`
- Structured Git output에는 prompt, target, logits, weights, generation,
  activation을 기록하지 않았고 `direct_z`는 local-only로 유지됐다.

## Primary와 oracle

Replay envelope `e_m`은 `1e-12`이다.

| Metric | Primary adaptive-static | Finite-panel oracle |
| --- | ---: | ---: |
| mean | `0.010262799263000489` | `0.010279297828674316` |
| trim20 | `0.007329424222310384` | `0.007356921831766765` |
| median | `0.007647037506103516` | `0.007647037506103516` |
| positive sign | `20/20` | `20/20` |
| paired bootstrap mean 95% CI | `[0.005930529832839966, 0.015292656421661378]` | `[0.005953561663627625, 0.015286592841148378]` |

Primary bootstrap은 seed `20260731`, oracle bootstrap은 seed `20260732`,
각 `4000` replicate다. Oracle mean과 primary mean 차이는 매우 작고,
현재 signal이 oracle에만 남은 형태가 아니다.

## D0+D1 기대효과와 untouched 실현값

Calibration 단계에서 고정한 D0+D1 예상 adaptive-static gap은
`0.010963672438775659`, case-bootstrap 95% interval은
`[0.005291678279052575, 0.017507945627132517]`였다. Untouched에서 실현된
mean `0.010262799263000489`는 이 사전 기대값보다
`0.0007008731757751699` 낮지만 interval 안에 있으며, 상대 차이는 약
`-6.39%`다.

Untouched 20개 case의 frozen forecast를 동일 case 실현값과 비교하면 다음과
같다.

| 항목 | 값 |
| --- | ---: |
| predicted mean | `0.008498514037922367` |
| realized mean | `0.010262799263000489` |
| mean error (`realized - predicted`) | `0.0017642852250781215` |
| MAE | `0.002590510556911647` |
| median absolute error | `0.0010548005386717201` |
| zero-intercept slope (`realized ~ predicted`) | `1.1849713069832435` |
| Pearson | `0.9270568073422046` |
| positive-direction concordance | `1.0` |

따라서 Llama에서 motivation-stage 기대효과의 방향은 전 case에서 맞았고,
case 간 크기 순서도 강하게 보존됐다. 다만 slope `>1`과 양의 mean error는
frozen casewise forecast가 평균적으로 보수적이었음을 뜻하며, 이 한 run만으로
일반화된 개선량을 claim할 수는 없다.

## Compute/resource

| 항목 | 관측값 |
| --- | ---: |
| visible GPU | `1` |
| GPU peak allocated | `40,712,323,584 B` (`37.9163 GiB`) |
| GPU peak reserved | `43,518,001,152 B` (`40.5293 GiB`) |
| Host MaxRSS | `10,747,568 KiB` (`10.2497 GiB`) |
| Wall time | `6281.279592 s` (`01:44:41.280`) |

## Locked 단일-model gate

§10.9 untouched 규칙은 mean이 replay envelope보다 양수이고,
trim20·median·positive sign `>=11/20` 중 하나 이상이 같은 방향이면
단일-model clear 입력으로 본다.

- mean `> e_m`: true
- trim20 `> e_m`: true
- median `> e_m`: true
- positive sign `>=11/20`: true (`20/20`)
- `clear input`: **true**
- `scientific kill input`: false
- `controller/static-policy pivot input`: false
- `gray input`: false

고정 analyzer의 follow-up JSON은 claim boundary상 pair verdict가 아니라
single-run metric input만 내므로 `single_model_gate_inputs`의 follow-up
boolean을 모두 false로 유지한다. 위 gate는 analyzer boolean을 재해석한 것이
아니라, 검증된 metric에 §10.9 untouched 규칙을 별도로 적용한 결과다.

## Claim boundary

이 보고서는 Llama untouched 20-case single-model replication이 clear
입력을 제공한다는 데까지만 유효하다. 다른 model 결과를 읽거나 결합하지
않았고, case-level 수치·raw prompt·`direct_z` tensor를 노출하지 않았다.
Pair/MV-2 결정은 별도 red-team pair audit의 책임이다.
