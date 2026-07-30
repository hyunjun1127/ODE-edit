# MV-1 Qwen D0+D1 calibration forecast v1

## 판정 경계

- `[repo 사실]` Qwen D1 `mv1mix_qwen_d1_v1`은 `COMPLETED`, ExitCode
  `0:0`이며 sanitized summary의
  `planned/attempted/pass=12/12/12`, `failure=0`,
  `features/actions/outcomes/events/receipts=12/12/60/12/12`,
  `all_pass=true`, `all_rollbacks_exact=true`,
  `expected_counts_exact=true`다.
- `[제안/명세 사실]` D1은 calibration 완료와 feature-only static policy
  freeze 단계다. D1 outcome과 아래 D0+D1 forecast는 descriptive calibration
  output이며, C1 held-out outcome이 아니다.
- `[GH 추정]` 아래 adaptive-static forecast는 17개 calibration case에
  조건부인 raw progress 예측이다. benchmark accuracy, retention,
  long-horizon 성능 또는 완성된 ODE-Edit gain으로 외삽할 근거는 아직 없다.
- `[사용자 확인 필요]` C1 전에는 method gain, GO, confirmatory success,
  MV-2 진입 또는 ODE claim을 승인하지 않는다. 다음 실행이 필요하다면
  사전 고정된 paired C1 fold 0과 동일 policy/hash를 사용할지 별도 확인해야
  한다.

## D1 descriptive `G_mix`

`G_mix = P(score_mix) - P(uniform)`이며 단위는 raw progress다.

| 항목 | 값 |
| --- | ---: |
| planned / finite / missing | `12 / 12 / 0` |
| mean `G_mix` | `0.1624626417954763` |
| median `G_mix` | `0.1634688377380371` |
| observed range | `[0.01695537567138672, 0.4432029724121094]` |
| D1 replay envelope | `1e-12` |
| replay envelope 초과 | `12/12` |

모든 case의 replay absolute progress는 `0.0`이었다. 이 12개 값에는
population inference 또는 confidence interval을 붙이지 않았으며, 양수라는
사실만으로 held-out claim을 만들지 않는다.

## Frozen static policy

입력은 independently valid D0 `[3:8]` 5개와 D1 `[8:20]` 12개의
feature-only single-layer slope다. Outcome field는 사용하지 않았다.

| action | pooled mean slope | frozen weight |
| --- | ---: | ---: |
| `layer_4` | `54.251433822573546` | `0.30583144454336836` |
| `layer_5` | `66.30922853286133` | `0.37380481435908547` |
| `layer_6` | `91.00231619663018` | `0.5130070830980947` |
| `layer_7` | `85.55742747190439` | `0.4823126282841114` |
| `layer_8` | `92.3340575601931` | `0.5205145046771928` |

- controller: `relu(sbar)/L2-else-max-onehot`
- controller branch: `positive-relu-l2`
- weight L2 norm: `1`
- predicted mean score: `177.38997997271153`
- feature cases: `17`

## D0+D1 calibration-only adaptive-static forecast

| 항목 | 값 |
| --- | ---: |
| calibration case | `17` |
| eligible `x_AU` case | `17` |
| nonnegative zero-intercept robust `beta` | `1.1595737381191145` |
| mean feature-only `x_AS` | `0.09660042026961474` |
| expected adaptive-static raw progress gap | `0.11201531043591464` |
| case-bootstrap 95% percentile interval | `[0.06476441445994054, 0.17629666495305207]` |
| bootstrap seed / replicates | `20260731 / 4000` |
| calibration residual envelope | `0.03236109868826046` |
| D0+D1 replay envelope | `1e-12` |

`beta`만 calibration adaptive-uniform outcomes를 사용한다. Adaptive weights와
leave-one-out static weights는 feature-only이며, confirmatory outcome은
정책 fit과 forecast 어디에도 사용하지 않았다. Bootstrap interval도
calibration-case resampling uncertainty일 뿐 C1 effect의 confidence interval이
아니다.

## Identity와 hashes

| 객체 | SHA-256 |
| --- | --- |
| selection manifest | `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce` |
| selection order | `01cd37bca74dba6e3405cbbd36bc5036c9a6bcfc4ce1b20c8871f561f560585d` |
| selection split | `f2e23e4135e077f3401485e647c21c9224d93dd889e31aeace0b11a47c1f2547` |
| D1 manifest | `00779516e7728fa2050fc6e81be1c6fdcd1d53276de3011266b3f076ff61b069` |
| D1 features | `a57497ab05a831865b056192a2fb1d8f8154fb127b1694fba6a845b3473449c3` |
| D1 actions | `9ccb3c3cd98e31ba2424d6bb6a8728b6bc8d3e5a9da3a13207106521ea9b44f2` |
| D1 outcomes | `1861488acbc2122569d060c39bcaa300ed20f36aaf6767c5eb902b2d1d50fb7a` |
| D1 events | `3569f4b6d4dfc82278b2aa3d4c8aab93ab2702c8096bd6b2558dbaacd526d8e8` |
| static feature panel | `9399206584fe18ee5977b79625b48bc114c42957416d3efec2ebd5a9d0ebfbe5` |
| static policy | `381e22334e0e3d07c37f07db16c22e8cdd1b689deba66ec7b032dd32ba0b2955` |
| feature-only actions | `6e5e948d42559a6922dd980e2413986827a7f3c473100bc47c2bd60b9262461b` |
| calibration payload | `440f1eadd495d7e9a36cde258d711acfc0207867dab4d468b4c920765df0f122` |
| forecast policy | `a1feb93481f64785539c102a1ac58340fee71ba4e91a9ccc4b8f699cc6c04c99` |
| forecast analysis | `cabff79e285f252b7219de3f016e2ef280ad506a4b79264d5806dd8909ba6e22` |

## 독립 검증

- D0와 D1의 model/run/slice/selection identity가 일치하고 case가 겹치지
  않았다.
- Sanitized manifest, summary, four JSONL streams 및 12개 receipt의
  schema, payload chain, receipt hash, artifact hash, exact rollback,
  equal-`C`, no-op replay, selection/commitment/outcome firewall이 모두
  통과했다.
- 생성된 모든 numeric JSON leaf는 finite이고 JSON은 `allow_nan=false`로
  직렬화된다.
- Raw `direct_z/*.pt`는 열거나 역직렬화하지 않았다.
