# P1R34 W-anchored finite-demand P1R24 Atomic — terminal 분석

## 결론

**패키지 판정은 `SCIENTIFIC_FAIL / scientific_promotion=false`다.** Llama는 Neutral/Soft 모두 `10/10 Eff, 20/20 Gen`을 회복했고 P1R33보다는 명확히 강해졌지만, P1R24 대비 연속 NLL 강도는 완전히 회복하지 못했다. Qwen Neutral은 K8까지 갔으나 paired action-freeze/evaluator가 생성되기 전에 Soft가 k5 pre-route에서 `NON_SEMANTIC_TARGET_MOVE`를 기록했다. 따라서 Qwen endpoint를 보간하거나 Llama 결과로 package recovery를 주장하지 않는다.

핵심 발견은 finite demand가 단순히 old-gradient demand를 대체하는 만능 복구 장치가 아니라는 점이다. Qwen Soft k5에서 old demand는 `+4.3544`였지만 실제 frozen-W endpoint NLL은 `1.19457→1.26176`으로 악화하여 finite demand가 `-0.06719`였다. 계약대로 이를 뒤집지 않고 scientific boundary로 종료했다.

## 범위와 무결성 — FACT

- base `ce8c6c36348752f1407f7d713d30e6b5c727379b`; scientific child `d38b70cb8fc2150ed47f29ead1baac5a42e133a0`; execution TECH-R1 `ac321d86a634a20888cb57003a65467d40aa5ddd`.
- seal `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`; order `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`; RS, Neutral/Soft, K8/h=1/8, H/history0.
- smoke attempt `19426`은 새 worktree의 ignored session-boundary 파일 누락으로 model action0 상태에서 실패했다. TECH-R1 `19428`은 양 alias paired N/S K8/W0 restore를 통과했다.
- production `19430`: Llama task COMPLETED0, Qwen task typed scientific boundary. Llama accepted 16/16; Qwen Neutral 8/8, Soft 4/8 뒤 pre-route boundary 1건이다.
- 모든 accepted step은 endpoint full-six hooked forward를 정확히 한 번의 microbatch plan으로 수행했고 추가 backward는 0이다. P1R24 target gradient/KL/field/slope backward는 그대로다.
- `L_base`는 authoritative no-hook W-only full-six NLL을 재사용했고 same-state identity residual은 전 step 0이다. `u=d+lag/(K-k)`, h 적용1, second remaining division0, debt0을 유지했다.
- accepted field 최대 strength equality residual은 `8.88e-16`; retry/backtracking/imputation/inner heldout=0.

## Terminal endpoint

| model | arm | status | Eff | Gen | Loc | Eff new-NLL | Gen new-NLL | z8 full6 NLL | W8 full6 NLL | cum.P | capacity sum |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | Neutral | PASS | 10/10 | 20/20 | 81/100 | 0.07096 | 1.26093 | 0.04464 | 0.11984 | 0.00161899 | 0.20118 |
| Llama | Soft | PASS | 10/10 | 20/20 | 82/100 | 0.06935 | 1.25162 | 0.04317 | 0.12077 | 0.00161010 | 0.20330 |
| Qwen | Neutral | K8 complete, pair incomplete | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 0.16415 full6 내부값만 기록 | 0.0505541 | 1.33012 |
| Qwen | Soft | typed incomplete at k5 pre-route | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | k4 0.0189939 | k1–4 0.63625 |

Qwen Neutral의 K8 `W8 full6`는 k8 scientific-path terminal objective(`0.16415`)로 계산되었지만 paired action-freeze 및 official evaluator가 없으므로 공식 endpoint로 승격하지 않는다. Qwen 실패 경로의 W0 restore는 코드 `finally` 경로가 실행됐지만 별도 durable restore receipt가 생성되지 않아 `NOT_ESTABLISHED`다.

## 단계별 finite-demand 및 W 실현 추이

표의 k는 accepted transition 1..8이다. `actual`은 다음 refreshed `L_base` 재사용, 마지막은 terminal objective다. Qwen Soft k4 actual은 다음 boundary-state `L_base`로부터 계산한 observation이다. stepwise Eff/Gen/Loc는 계약상 `NOT_RECORDED`다.

### Llama Neutral

| k | L_base | L_endpoint | old rho | finite rho/pred | actual | R | cum.P | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|8.36549|5.23219|4.83006|3.13330|2.05123|0.655|0.0000715|0.02466|
|2|6.31427|3.07317|3.40912|3.24110|3.10060|0.957|0.0002961|0.04926|
|3|3.21366|1.16294|2.22375|2.05072|1.75904|0.858|0.0005247|0.03641|
|4|1.45462|0.58493|1.50782|0.86970|0.62425|0.718|0.0006617|0.01712|
|5|0.83037|0.55732|0.56249|0.27305|0.17235|0.631|0.0007561|0.00871|
|6|0.65802|0.34923|0.57648|0.30879|0.21605|0.700|0.0009143|0.01257|
|7|0.44197|0.18574|0.35979|0.25623|0.16920|0.660|0.0012133|0.02828|
|8|0.27278|0.04406|0.49635|0.22872|0.15294|0.669|0.0016190|0.02419|

### Llama Soft

| k | L_base | L_endpoint | old rho | finite rho/pred | actual | R | cum.P | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|8.36549|5.23219|4.83006|3.13330|2.05174|0.655|0.0000702|0.02466|
|2|6.31375|3.06763|3.40708|3.24612|3.11325|0.959|0.0002893|0.04958|
|3|3.20050|1.15609|2.22105|2.04441|1.75338|0.858|0.0005124|0.03635|
|4|1.44713|0.58847|1.51477|0.85865|0.61529|0.717|0.0006456|0.01696|
|5|0.83183|0.55250|0.56151|0.27933|0.17039|0.610|0.0007402|0.00903|
|6|0.66145|0.34872|0.57821|0.31273|0.22051|0.705|0.0008993|0.01265|
|7|0.44094|0.17924|0.36007|0.26170|0.17006|0.650|0.0012109|0.03046|
|8|0.27088|0.04349|0.50183|0.22739|0.15012|0.660|0.0016101|0.02362|

### Qwen Neutral

| k | L_base | L_endpoint | old rho | finite rho/pred | actual | R | cum.P | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|9.30923|5.91534|10.77124|3.39388|1.65986|0.489|0.0009965|0.05528|
|2|7.64937|3.51209|4.91959|4.13728|4.02437|0.973|0.0126948|0.54565|
|3|3.62499|1.41233|4.51053|2.21267|1.66571|0.753|0.0169928|0.14245|
|4|1.95928|1.05050|3.25052|0.90878|0.73661|0.811|0.0227578|0.16238|
|5|1.22267|1.04505|4.13841|0.17762|0.15229|0.857|0.0241470|0.01652|
|6|1.07038|0.50879|5.28864|0.56159|0.40449|0.720|0.0338442|0.21604|
|7|0.66589|0.50008|1.06690|0.16581|0.16018|0.966|0.0363606|0.01133|
|8|0.50571|0.05769|3.09644|0.44802|0.34156|0.762|0.0505541|0.18046|

### Qwen Soft와 boundary

| k | status | L_base | L_endpoint | old rho | finite rho/pred | actual | R | cum.P | capacity |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|accepted|9.30923|5.91534|10.77124|3.39388|1.64768|0.485|0.0008573|0.04157|
|2|accepted|7.66155|3.54502|4.92543|4.11653|4.04428|0.982|0.0108408|0.39962|
|3|accepted|3.61726|1.39180|4.48381|2.22546|1.69670|0.762|0.0146173|0.10056|
|4|accepted|1.92057|1.01931|3.24800|0.90125|0.72600|0.806|0.0189939|0.09450|
|5|`NON_SEMANTIC_TARGET_MOVE` pre-route|1.19457|1.26176|4.35441|-0.06719 / —|—|—|—|—|

## P1R24 및 P1R33 동일 샘플 대비

| model/arm | P1R24 E/G/L | P1R33 E/G/L | P1R34 E/G/L | Eff NLL P1R24→P1R34 | 해석 |
|---|---:|---:|---:|---:|---|
| Llama N | 10/20/82 | 4/7/83 | 10/20/81 | 0.0218→0.0710 | count 회복, 연속 강도·Loc 미완전 |
| Llama S | 10/20/81 | 4/7/83 | 10/20/82 | 0.0265→0.0694 | count 회복, 연속 강도 미완전, Loc +1 |
| Qwen N | 10/19/80 | 9/16/80 | official endpoint 없음 | — | K8 내부 궤적만, 비교 불가 |
| Qwen S | 10/20/80 | 8/12/80 | k5 typed incomplete | — | finite demand가 비semantic target move를 검출 |

- FACT: Llama k1 finite demand `3.1333`은 P1R24 old demand `4.8301`보다 작고 actual `2.051–2.052`라서 R32식 k0 폭발이 없다.
- FACT: Llama는 z8 full-six NLL `0.043–0.045`지만 W8은 `0.120–0.121`; 추가 writer gap은 약 `0.075–0.078`다.
- FACT: Llama Soft는 Neutral과 k1 동일 demand/strength를 유지하고 terminal cumulative P를 `0.0016190→0.0016101`로 아주 소폭 낮췄지만 capacity sum은 `0.20118→0.20330`으로 늘었다. Loc는 `81→82`, terminal functional-P mean은 `0.004559→0.004180`으로 낮았다.
- FACT: old rho가 수치적으로 0에 가까운데 finite rho만 양수인 qualifying event는 관측되지 않았다. 반대로 Qwen Soft k5에서는 old rho가 크게 양수인데 finite rho가 음수였다.
- INFERENCE: finite endpoint difference는 local gradient dot보다 writer demand의 실제 방향성을 더 엄격히 판별했지만, alias-common package를 완결시키지는 못했다.

## Compute

- Llama 완결 arm당 `220F/125B/8 materializations`, processed tokens `35,263`. P1R24보다 endpoint forwards가 40회 늘고 backward는 동일하다.
- edit-core: Neutral `91.65s`, Soft `90.94s`; finite endpoint-forward phase `2.376s/arm`.
- scheduler: Llama `268s`, MaxRSS `7,116,308 KiB`; Qwen typed-incomplete task `271s`, MaxRSS `10,581,680 KiB`.
- model-load-only 및 exact Native compatible pure-edit denominator는 `NOT_RECORDED`; formal Native ratio를 만들지 않는다.

## 판정 경계

### FACT

- Llama endpoint 2/2는 action-freeze, manifest, endpoint evaluation, W0 pointer restore PASS.
- Qwen boundary는 수치/serializer/solver 문제가 아니라 source-defined material negative finite demand다.
- Soft는 accepted step에서 matched predicted strength를 낮추지 않았다.

### SCIENTIFIC_FAIL

- Qwen paired package가 완결되지 않았고, Llama 연속 NLL도 P1R24 strength region을 완전히 회복하지 못했다.
- finite demand가 old-gradient collapse를 일반적으로 복구한다는 가설은 `NOT_ESTABLISHED`다.

### NOT_RECORDED

- stepwise heldout Eff/Gen/Loc.
- Qwen official terminal Eff/Gen/Loc, paired action-freeze/manifest, durable W0 restore receipt.
- fresh sample, BG, Sequential/Historical, B100, formal Native ratio.

이 결과는 outcome-selected 재사용 B10의 Atomic mechanistic/descriptive evidence다. `scientific_promotion=false`; 본 보고서는 추가 GPU/Slurm 실행이나 tuning을 승인하지 않는다.
