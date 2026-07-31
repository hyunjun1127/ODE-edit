# MV-2 refresh pair v2 사후 red audit

범위: 지정된 implementation spec 217–244행과 두 compact analysis summary의 허용 투영만 검토했다. 각 summary는 투영 전에 SHA-256을 산출했다. 코드, raw/case 효과, replay per-case, 다른 report/audit 및 GH 기록은 열람하지 않았다.

## Summary hashes

| summary | SHA-256 |
|---|---|
| `2026-07-31-mv2refresh-llama-e0-v2.analysis.summary.json` | `62536b253c252c53490a6f608d7a0aeaf3d39640f69b351ac36ba09871c13da4` |
| `2026-07-31-mv2refresh-qwen-e0-v2.analysis.summary.json` | `371c15d4c7d5b53119b96bd65f84cd0407532d5fbab91d80eb613cc73d444b8e` |

## 집계 표

| 항목 | Llama (`llama3-8b-inst`) | Qwen (`qwen2.5-7b-inst`) |
|---|---|---|
| technical validity | pass=`true`; failures=`[]`; denominator 실패 case=`0` | pass=`true`; failures=`[]`; denominator 실패 case=`0` |
| replay/sham envelope (e_m) | `0.0001` | `0.0001` |
| decision | coefficient clear=`true`, null=`false`; direction clear=`false`, kill=`false`, null=`true`; total clear=`false`, nonnegative=`false`; `PIVOT_FIXED_DIRECTION_DYNAMIC_COEFFICIENT` / fixed-direction dynamic coefficient로 전환 | coefficient clear=`true`, null=`false`; direction clear=`true`, kill=`false`, null=`false`; total clear=`true`, nonnegative=`true`; `DIRECTION_REFRESH_CLEAR` / direction mechanism과 current-method 총 기대효과 모두 명확 |
| direction refresh: mean / trimmed mean 10% / median / positive / bootstrap mean CI95 | `-0.0044080813725789385` / `-0.0034751415252685545` / `-0.0024871826171875` / `0` / `[-0.007394531369209289, -0.0021643350521723427]` | `0.006887376308441162` / `0.004586529731750488` / `0.0011625289916992188` / `7` / `[-0.002343200147151946, 0.017991075416406]` |
| coefficient refresh: mean / trimmed mean 10% / median / positive / bootstrap mean CI95 | `0.00010367234547932942` / `0.00005345344543457031` / `0.000038623809814453125` / `12` / `[0.00003484785556793213, 0.0002209345499674479]` | `0.0016649017731348674` / `0.0016055703163146973` / `0.0013937950134277344` / `11` / `[0.0004207629710435867, 0.002964677413304647]` |
| total refresh: mean / trimmed mean 10% / median / positive / bootstrap mean CI95 | `-0.004304409027099609` / `-0.0034216880798339845` / `-0.0024406909942626953` / `0` / `[-0.0071797053019205725, -0.002130027612050374]` | `0.008552278081576029` / `0.005599361658096313` / `0.002131938934326172` / `7` / `[-0.0003046554823716481, 0.01953920697172484]` |
| compute: case count / controlled NFE / proposal builds / probe panels / wall seconds | `12` / `516` / `48` / `36` / `8676.61980325263` | `12` / `516` / `48` / `36` / `11564.389404653572` |

Llama coefficient refresh의 trimmed mean 10%는 summary 투영값 `5.345344543457031e-05`이며, 위 표의 과학적 표기와 같은 값이다.

## §7 first-match 순차 trace

1. Rule 1 — 두 model 모두 technical validity pass이고 failures가 없으며 denominator 실패 case도 `0`이다. `TECHNICAL BLOCK`에 일치하지 않는다.
2. Rule 2 — 어느 model도 `DIRECTION_MECHANISM_ONLY_REDESIGN_COEFFICIENT`가 아니다. Llama는 fixed-direction coefficient pivot, Qwen은 direction clear이므로 일치하지 않는다.
3. Rule 3 — 양쪽 direction clear가 아니다(Llama=`false`, Qwen=`true`). 일치하지 않는다.
4. Rule 4 — direction clear은 정확히 하나(Qwen)이고 Llama는 technical valid이다. 그러나 (e_m=0.0001), 즉 기준은 −`0.0001`이며, 다른 model인 Llama의 원본 aggregate mean을 직접 대조하면 direction=`-0.0044080813725789385` < −`0.0001`, total=`-0.004304409027099609` < −`0.0001`이다. 두 조건 모두 불충족이므로 이 rule에 일치하지 않는다. `total_nonnegative` flag는 이 판정의 대체 근거로 사용하지 않았다.
5. Rule 5 — direction clear이 전혀 없어야 하나 Qwen에 clear가 있으므로, Llama의 pivot에도 불구하고 전제 불충족이다.
6. Rule 6 — 양쪽이 static routing pivot이 아니다. 일치하지 않는다.
7. Rule 7 — 위 first-match 조건 어느 것도 일치하지 않은 혼합 결과다. 여기서 판정을 종료한다.

## 비용 및 claim boundary

- 보고된 model-run 합계는 case count `24`, controlled NFE `1032`, proposal builds `96`, probe panels `72`, wall time `20241.009207906202`초다. case 수 합계는 고유 case 수를 뜻하지 않는다.
- 이 증거는 Qwen의 direction-clear와 Llama의 fixed-direction coefficient pivot이 공존함만 뒷받침한다. 교차-model 일반화, 일반 ODE-Edit 우위, locality/retention 개선, 논문 claim, 또는 MV-3 성공을 주장하는 근거가 아니다.
- 판정 후 재튜닝, threshold/metric 변경, 또는 추가 fold로 결과를 구제하지 않는다. MV-3를 제안하거나 시작하지 않는다.

## 분류 기록

| 범주 | 기록 |
|---|---|
| proposal | 현 증거에 대한 후속 MV-3 proposal은 없다. 허용되는 다음 조치는 별도 지시가 있는 비-MV-3 범위의 기록 또는 분석뿐이다. |
| repo | 지정된 spec 부분과 두 summary의 허용 필드만 읽었고, 이 audit 하나만 작성했다. 코드/실험 raw/다른 audit은 검사하지 않았으며 commit/push도 수행하지 않았다. |
| GH 추정 | launcher runtime metadata가 `model=gpt-5.6-terra`, `reasoning effort=ultra`였다는 것은 사용자 제공 진술이다. GH 기록을 열람하지 않았으므로 독립 검증·재현성·원인에 관한 GH 추정은 하지 않는다. |
| 사용자 확인 필요 | 본 판정 자체에는 추가 확인이 필요 없다. 코드 변경, 새 실행, 대외 claim, 또는 이후 계획은 별도 사용자 승인과 범위 지정이 필요하다. |

NO MV3
