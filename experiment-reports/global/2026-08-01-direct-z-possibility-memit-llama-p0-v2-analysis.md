# Llama MEMIT direct-z 가능성 분석 — p0 v2

## 결론

`llama3-8b-inst`의 고정 8-case MEMIT panel에서 analyzer의 **lenient** 판정은
`CONE_FEASIBLE_CURRENT_BF_GAP`이다. 즉, frozen direct-z 자체에는 행동적으로
유용한 oracle ceiling이 있고, 같은 BF endpoint C-budget 안의 synchronous
z-cone도 direct-z residual을 일관되게 줄일 수 있다. 반면 현재
`bf_current_refreshed_k4`는 그 direct-z fidelity를 더 잘 따르는 것으로
확인되지 않았고, analyzer가 명시한 것은 `current_bf_directional_gap_clear =
true`이다.

이는 방법 우월성 판정이 아니다. 특히 cone과 `oracle_do_z`는 모두 진단용
ceiling이며, 전자는 비배포성 ridge/cone 해이고 후자는 weight를 전혀 쓰지
않는 layer-8 frozen-delta intervention이다. 둘을 ODE-Edit 또는 배포 가능한
편집기로 제시할 수 없다.

## 증거 범위와 실행 무결성

이 문서는 다음 단일 Llama run의 manifest/summary/analyzer projection 및
지정된 Slurm accounting metadata만 사용한다. raw outcome, feature/action,
다른 모델, 기존 report, 구현 코드는 해석에 사용하지 않았다.

- Run/model: `dzf_llama_p0_v2` / `llama3-8b-inst`; manifest의 claim boundary는
  `possibility_only_not_method_superiority`이다.
- 고정 arm 순서는 `no_op_replay`, `oracle_do_z`, `native_ordered_full`,
  `native_alpha_c_matched`, `bf_current_refreshed_k4`,
  `sync_z_cone_oracle`이고, case 수는 8이다.
- 기술 유효성은 `true`: 8/8 cases pass, failed cases 0, outcomes 48,
  direct-z는 case당 한 번만 생성, receipt-before-outcome / scalar firewall /
  exact rollback은 모두 `true`이다. Analyzer SHA-256은
  `2a0b8413f12020651fdda1e2d736b166ae936bcb8ef0acdc169e29f2b13fc494`이다.
- `nfe_total_itd`는 144이고, BF와 `native_alpha_c_matched`의 endpoint
  C-energy match는 `true`이다.
- `sacct` snapshot에서 parent `15740` (`odeedit_dzf_pair_v2`)는 `RUNNING`,
  16 CPU / 2 GPU / 130000M / `devbox`로 기록되었다. 지정 step `15740.0`
  (`bash`)는 `COMPLETED`, exit `0:0`, elapsed `02:05:39`, 8 CPU / 1 GPU /
  65000M / `devbox`로 기록되었다. 이는 이 report가 의존하는 Llama child
  step의 정상 종료 audit일 뿐, parent allocation의 최종 pair-level 상태를
  판정하지 않는다.

Analyzer gate는 positive mean과 strict majority 5/8만 요구하며, 4,000회
paired bootstrap CI의 zero exclusion은 요구하지 않는다. 아래 CI는 그래서
**설명용**이며 유의성 gate가 아니다.

## Direct-z ceiling과 cone feasibility

제공된 schema에는 `DF`라는 arm ID가 없다. 따라서 여기서 direct-z 쪽은
임의의 새 DF 방법으로 부르지 않고 정확한 arm 명칭인 `oracle_do_z`와
`sync_z_cone_oracle`로 적는다.

| 비교 (analyzer 표기) | 평균 paired effect | 양의 방향 | 95% paired-bootstrap CI | 해석 |
|---|---:|---:|---:|---|
| `z_residual_gain_noop_minus_oracle` | `0.9999999936392909` | 8/8 | [`0.9999999870486395`, `0.9999999983697394`] | Oracle direct-z가 residual을 사실상 전부 제거한 ceiling이다. |
| `output_nll_gain_oracle_minus_noop` | `10.421976848245322` | 8/8 | [`8.902477643962266`, `12.00300762009224`] | Oracle ceiling의 output-NLL reduction이 모든 case에서 양수다. |
| `output_progress_gain_oracle_minus_noop` | `17.68893551826477` | 8/8 | [`15.791485786437988`, `19.50961297750473`] | 같은 방향의 output progress다. |
| `z_residual_gain_noop_minus_cone` | `0.5602797111419432` | 8/8 | [`0.537570787387574`, `0.5818180776418448`] | Cone이 no-op보다 direct-z residual을 줄여 `cone_feasible` lenient gate를 충족한다. |
| `output_nll_gain_cone_minus_noop` | `9.47441476280801` | 8/8 | [`8.313102737534791`, `10.730333522241562`] | Cone의 output-NLL reduction도 8/8에서 양수다. |
| `generated_error_mean_gain_noop_minus_cone` | `0.5156427195760885` | 8/8 | [`0.4936746850073744`, `0.5346339808942864`] | Shared-delta generated-context mean relative error는 cone에서 낮아진다. |
| `generated_error_worst_gain_noop_minus_cone` | `0.3822139601412816` | 8/8 | [`0.3393458335590186`, `0.42192894928765967`] | Worst generated-context relative error도 같은 방향이다. |
| `preservation_gain_cone_minus_noop` | `-0.4033672825898975` | 0/8 | [`-0.7366527661215514`, `-0.14180672238580883`] | `preservation_score=-heldout_kl`이므로 cone은 이 제한된 preservation proxy에서는 악화다. |

Oracle은 cone보다 여전히 더 가깝다. `z_residual_gap_cone_minus_oracle`은
`0.43972028249734774` (8/8 양수, CI
[`0.41838171878787317`, `0.46226746586377976`])이고, analyzer의
`output_gap_oracle_minus_cone`은 `0.9475620854373119` (8/8 양수, CI
[`0.2887383528552164`, `1.7238126669708436`])이다. 따라서 cone feasibility는
direct-z target의 완전한 실현도, oracle과의 동등성도 뜻하지 않는다.

## BF의 fidelity, edit, preservation 분리

### C-matched native alpha와의 비교

BF와 `native_alpha_c_matched`는 endpoint C-energy가 match된 조건이다.
실제 `endpoint_c_difference_bf_minus_native_alpha`의 평균은
`-4.774847184307873e-12` (median `-4.320099833421409e-12`, 5/8 음수,
CI [`-1.6257217794191092e-11`, `5.6274984672199935e-12`])로 수치적으로
0 근방이다. 따라서 아래 방향은 서로 다른 C-budget 때문으로 읽으면 안 된다.

| BF − native alpha paired contrast | 평균 | 방향 case 수 | 95% paired-bootstrap CI | 의미 |
|---|---:|---:|---:|---|
| `z_gain_bf_vs_native_alpha` | `-0.13500684119681627` | 0/8 양수, 8/8 음수 | [`-0.15598219788156936`, `-0.11229616018573961`] | BF의 direct-z fidelity gain은 native alpha보다 낮다. |
| `output_nll_gain_bf_vs_native_alpha` | `0.1606774851679802` | 8/8 양수 | [`0.01702705061179586`, `0.3113187204056885`] | BF의 NLL-reduction edit proxy는 높다. |
| `output_progress_gain_bf_vs_native_alpha` | `0.7653405666351318` | 8/8 양수 | [`0.38277438133955`, `1.2278283037245274`] | Output progress도 높다. |
| `exact_margin_gain_bf_vs_native_alpha` | `1.1885839700698853` | 8/8 양수 | [`0.6173873364925384`, `1.780961087346077`] | Exact target margin은 높다. |
| `paraphrase_gain_bf_vs_native_alpha` | `0.8022961665410548` | 6/8 양수 | [`0.15361711268196815`, `1.5827625251025894`] | Evaluation-only paraphrase NLL reduction도 평균상 높다. |
| `preservation_gain_bf_vs_native_alpha` | `-0.09631979279220104` | 0/8 양수, 8/8 음수 | [`-0.20602652803063393`, `-0.026229727081954472`] | BF의 제한된 preservation proxy는 낮다. |

즉, C가 맞은 이 비교에서는 **BF의 edit proxy 상승이 direct-z fidelity와
preservation의 동반 상승을 요구하지 않는다.** 오히려 BF는 8/8에서 더 높은
output-NLL reduction을 보이면서 8/8에서 더 낮은 z gain과 preservation score를
보였다. 이는 spec의 `bf_output_without_z` 설명과 양립하는 패턴, 즉 target을
더 가깝게 추적하지 않아도 task-relevant projection으로 output을 바꿀 수 있다는
가능성이다. 그러나 projection에는 BF−no-op z-residual contrast가 없으므로,
이 자료만으로 formal `bf_tracks_z` 판정을 내릴 수는 없다.

### Norm efficiency와 off-token spill

동일 C-energy에서 `endpoint_frobenius_difference_bf_minus_native_alpha`는
평균 `0.04089099070579942` (8/8 양수, CI
[`0.02624779405861065`, `0.05409964680369916`])이다. 즉 BF는 이 저랭크
Frobenius norm 기준에서 native alpha보다 **더 큰** norm을 사용했으며,
이 panel에서는 BF의 norm-efficiency 이점이 관찰되지 않았다.

Cone과 비교하면 `off_token_spill_gap_bf_minus_cone`은 평균
`0.06422967418397758` (8/8 양수, CI
[`0.044976011012588316`, `0.0875265903790709`])이다. Layer-8
full-sequence off-token spill은 BF가 cone보다 더 크다는 방향이다. BF의
generated mean-error gap도 `0.055960909687108006` (7/8 양수, CI
[`0.0247667408721518`, `0.0901406614612452`])이고, worst-error gap은
`0.01187379647147399` (5/8 양수, CI
[`-0.018996019163620967`, `0.0452464995478641`])이다. 따라서 BF의
off-token containment 또는 generated-context fidelity 우세를 이 run에서
주장할 수 없다.

## BF 대 cone: 하나의 우월 관계로 환원되지 않음

Analyzer가 `current_bf_directional_gap_clear`로 분류한 직접 근거는
`z_objective_gap_bf_minus_cone = 0.042214712806779346` (6/8 양수, CI
[`0.00977229380192623`, `0.07294419696676485`])이다. 같은 projection의
`output_objective_gap_cone_minus_bf`는 `-0.8098938816110604` (8/8 음수,
CI [`-1.397002308505762`, `-0.25614441893412726`])이고,
`exact_margin_gap_cone_minus_bf`는 `-4.060324788093567` (8/8 음수, CI
[`-5.646415644884109`, `-2.4757689237594604`])이다. 후자는 BF의 exact
margin이 cone보다 높은 방향임을 뜻한다.

동시에 `preservation_difference_bf_minus_cone`은
`-0.2055198021698743` (7/8 음수, CI
[`-0.4069858085655142`, `-0.0599794473557267`])이므로 BF의 preservation
score는 cone보다 낮다. 따라서 margin, generated error, z objective,
spill, preservation, 그리고 analyzer가 이름 붙인 output-objective가 서로
동일한 ordering을 주지 않는다. 이 이유로 cone ceiling과 현재 BF 사이의
단일 method ranking이나 ODE-Edit 우월성을 결론내릴 수 없다.

## Fidelity–edit–preservation 관계의 한계

Cone은 no-op보다 direct-z residual과 generated error를 8/8에서 개선하고
output-NLL reduction도 8/8에서 높였지만, preservation score는 8/8에서
낮았다. BF는 native alpha보다 edit proxy를 높이면서 z gain 및 preservation
score를 8/8에서 낮췄다. 따라서 이 fixed panel에서 direct-z fidelity가
자동으로 preservation을 보장하거나 edit quality의 단조 대리변수라는 증거는
없다.

Analyzer의 case-level correlations도 모두 **descriptive, noncausal**이다.

| 관계 | Pearson r | 95% paired-case bootstrap CI |
|---|---:|---:|
| z gain vs output gain | `0.02378818989336336` | [`-0.46518769056989884`, `0.4256211393202519`] |
| z gain vs preservation gain | `-0.11675961457542432` | [`-0.48895282873787316`, `0.1981446107741745`] |
| output gain vs preservation gain | `0.5110919012903055` | [`-0.4996182085092137`, `0.8277986259491803`] |

또한 BF-vs-native-alpha의 joint-direction count는
`z_and_output_bf_better = 0`, `z_and_preservation_bf_better = 0`,
`output_and_preservation_bf_better = 0` (각각 n=8)이다. 이는 mediation,
인과, 혹은 일반적 fidelity–preservation alignment를 뒷받침하지 않는다.

## 허용되지 않는 결론

이 결과는 새 8개 atomic CounterFact case의 두-snapshot 가능성 진단일 뿐이다.
다음은 보장하거나 주장할 수 없다.

- lifelong/sequential retention, sequential-collapse 방지, 또는 장기 안정성;
- 일반 downstream capability, benchmark gain, accuracy, 혹은 다른 model로의
  일반화;
- direct-z fidelity가 preservation 또는 edit outcome을 인과적으로 만든다는 주장;
- MEMIT, BF, cone, direct-z diagnostic arm, AlphaEdit, 또는 ODE-Edit 사이의
  method superiority.

따라서 이 Llama report가 제공하는 가장 강한 판단은 “frozen direct-z는 이
고정 panel에서 실현 가능하고 output과 함께 움직일 수 있으나, 현재 BF의
edit gain·fidelity·preservation·norm/spill trade-off는 분리되어 있으며,
lifelong 또는 방법 우월성을 보장하지 않는다”이다.
