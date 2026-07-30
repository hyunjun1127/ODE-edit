# MV-1 Llama D0+D1 calibration forecast 독립 분석

## 결론

- **[repo/protocol에서 확인한 사실]** `llama3-8b-inst`의 D1 artifact는 `valid=true`, `panel_complete=true`, error code 없음으로 독립 검증을 통과했다. D1의 equal-C `G_mix = P(score_mix) - P(uniform)`는 12/12 case에서 replay envelope `1e-12`를 초과했고, 평균은 `0.0075978041`, 관측 범위는 `[0.0004110336, 0.0462369919]` progress unit이다.
- **[GH 추정]** D0+D1 17-case calibration에서 고정한 feature-only static comparator와 비교할 때, adaptive score mix의 예상 gap은 `+0.0109636724` progress unit이다. 4,000회 case bootstrap calibration interval은 `[+0.0052916783, +0.0175079456]`이며 전 구간이 양수다.
- **[GH 추정]** 이 결과는 confirmatory에서 검증할 가치가 있는 양의 핵심 신호다. 다만 이는 calibration-conditional forecast이며 ODE-Edit의 end-to-end method gain, EasyEdit baseline 대비 개선률, GO, confirmatory success 또는 MV-2 진입 근거가 아니다.

## 네 범주 구분

### Proposal에서 온 내용

- 이 문서는 generated analysis JSON만을 근거로 작성했다. 따라서 proposal의 claim을 별도 사실로 가져오거나 검증 완료로 표시하지 않는다.
- 현재 수치가 다루는 질문은 request-specific adaptive layer mix가 calibration에서 고정한 static layer mix보다 나을 수 있는지에 한정된다.

### Repo/protocol에서 확인한 사실

- source run은 D0 `mv1mix_llama_d0_v1` 5 case와 D1 `mv1mix_llama_d1_v1` 12 case로, calibration denominator는 정확히 17 case다.
- D1 descriptive 상태는 `d1_descriptive_complete`, claim 상태는 `d1_calibration_descriptive_only`다.
- D1 `G_mix` finite denominator는 12/12이며 12/12가 D1 replay envelope `1e-12`를 초과했다.
- 17/17 case가 `abs(x_AU) > 1e-12` 조건으로 β 적합에 포함되었다.
- β는 nonnegative zero-intercept median-ratio 규칙으로 고정되었고 값은 `0.9497786623`이다.
- 평균 feature-only `x_AS`는 `0.0115433973`이고, 이에 따른 expected realized adaptive–static gap은 `0.0109636724`다.
- calibration residual envelope는 `0.0012365315`, calibration replay envelope는 `1e-12`다.
- adaptive/static weight는 outcome을 사용하지 않았다. β만 calibration adaptive–uniform outcome을 사용했으며 confirmatory outcome 사용 목록은 비어 있다.

### GH 추정

- 예상 gap `0.0109636724`는 residual envelope의 약 `8.87배`이고 bootstrap lower bound도 약 `4.28배`다. 이는 단순 replay/noise scale보다 큰 신호라는 calibration 수준의 근거다.
- bootstrap interval 전체가 양수이므로 confirmatory adaptive–static 비교를 진행할 실용적 근거는 살아 있다.
- D1 `G_mix`는 adaptive–uniform 관측치이고 forecast는 adaptive–static 예측치이므로, 두 값을 같은 효과 크기로 직접 비교하면 안 된다.
- 이 raw progress-unit gap을 percentage improvement로 변환할 분모가 generated JSON에 없으므로 백분율 개선 기대치는 보고하지 않는다.

### 사용자 확인 필요

- 현재 기술 산출물에 필요한 추가 확인은 없다.
- confirmatory 실행, pair-level gate 및 GO 판단은 이 단일 모델 독립 보고의 권한 밖이며 GH가 별도로 결정해야 한다.

## D1 descriptive signal

| 항목 | 값 |
| --- | ---: |
| finite D1 case | `12 / 12` |
| replay envelope 초과 | `12 / 12` |
| mean `G_mix` | `0.0075978041` |
| min `G_mix` | `0.0004110336` |
| max `G_mix` | `0.0462369919` |
| D1 replay envelope | `1e-12` |

이 표는 D1 calibration에서 실제 관측된 adaptive score mix 대 uniform의 equal-C 차이다. population effect나 adaptive–static 효과가 아니다.

## D0+D1 17-case adaptive–static forecast

| 항목 | 값 |
| --- | ---: |
| calibration case | `17` |
| β eligible case | `17 / 17` |
| β | `0.9497786623` |
| mean feature-only `x_AS` | `0.0115433973` |
| expected adaptive–static gap | `+0.0109636724` |
| 95% case-bootstrap calibration interval | `[+0.0052916783, +0.0175079456]` |
| calibration residual envelope | `0.0012365315` |
| calibration replay envelope | `1e-12` |
| bootstrap | seed `20260731`, `4,000` replicates |

β는 adaptive–uniform의 feature-only 예측 차이 `x_AU`와 관측 차이 `y_AU`를 calibration하는 값이다. 예상 adaptive–static gap은 β를 leave-one-out static comparator 대비 feature-only 차이 `x_AS`에 적용한 값이다.

## Frozen static comparator

정적 comparator는 D0+D1 17개 feature의 다섯 single-layer slope 평균만으로 고정되었고 outcome field는 사용하지 않았다. weight는 확률 simplex가 아니라 `relu(sbar)`의 L2 정규화 결과다.

| layer | mean slope `sbar` | frozen weight |
| --- | ---: | ---: |
| `layer_4` | `446.5681554` | `0.5292497592` |
| `layer_5` | `388.3547601` | `0.4602582176` |
| `layer_6` | `366.0476803` | `0.4338210064` |
| `layer_7` | `337.7034696` | `0.4002288963` |
| `layer_8` | `337.1655060` | `0.3995913293` |

- static policy hash: `e37443fa075245aa52828773ffc062217d983fb46254d1f40010d396be3d5e06`
- feature panel hash: `57d9581c58e60f7dff1d8f5ddf4637b99cbdf3a7a2795a1eaf1960e16d3e3328`
- forecast policy hash: `da5b352cbae20a44ff24e22f9c598f4a5df88aeab373995df3025ce2d0894533`
- calibration hash: `1f1ecf1a393df1f5474e523be49ca7d1dbecc9de3b812d766a1c3e41f9ca51a0`
- feature-only action hash: `fc71dca8773139a8afbabac024c7b116dbe8b41a993a65af93784c7cb3134346`
- analysis hash: `d8b371b37c76877c06349251d7abd3f9c5d828421c441f31e272cacd2b2e2094`

## 제한과 판정

- 표본은 calibration 17 case뿐이며 confirmatory generalization uncertainty를 측정하지 않는다.
- bootstrap interval은 같은 17-case calibration resampling interval이지 독립 test confidence interval이 아니다.
- β가 adaptive–uniform calibration을 adaptive–static gap으로 전달한다는 구조적 가정을 포함한다.
- 보고된 gap은 low-level equal-C `progress` proxy의 차이다. reliability, locality, paraphrase, multi-edit 성능을 합친 ODE-Edit method gain이 아니다.
- 이 문서의 판정은 **`CALIBRATION FORECAST POSITIVE; CONFIRMATORY REQUIRED`**다.
