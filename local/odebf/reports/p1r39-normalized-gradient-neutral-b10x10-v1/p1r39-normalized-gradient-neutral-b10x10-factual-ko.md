# P1R39 정규화 복합-gradient Neutral B10×10 결과 (사실 기록)

- 생성 시각(Asia/Seoul): 2026-08-14T12:50:00.531591+09:00
- instruction: `ODEEDIT-S05-P1R39-PR-P1R38-PERREQUEST-NORMALIZED-GRADIENT-NEUTRAL-B10X10-V1`
- source: `763457560f2efb177a56310dfd87526772cf8158`; exact P1R38 base: `6f48ac2800b257ceb16368fff5137212dfa6037f`
- scheduler submission: job `19652`, array `0-1%2`; result terminals: 2/2
- attempts/endpoints/failures: 20/20/0; accepted K steps: 160/160; request-step rows: 1,600
- stream root/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- scientific_promotion: `false` (administrative boundary)

## 1. 실행 및 불변식

|model|cases|endpoints|K steps|action-freeze|W0 pointer/bytes|controller reset|cross-case state|
|---|---:|---:|---:|---:|---|---:|---:|
|llama3-8b-inst|10|10|80|10/10|10/10|10/10|0|
|qwen2.5-7b-inst|10|10|80|10/10|10/10|10/10|0|

- target_direction_policy: `PER_REQUEST_NORMALIZED_COMPOSITE_GRADIENT` 160/160
- physical h/second h: 160/0; remaining/fresh/lag division: 0/0/0; semantic debt: 0
- Adam state/LR/bias-correction/raw-cap access: 0/0/0/0; raw-cap application: 0; semantic velocity decay access: 0
- selection added F/B: 0/0; per-request Python model calls: 0; persistent/carried mask influence: 0/0
- inner heldout evaluation: 0/160; terminal evaluator controller/routing influence: 0/20

## 2. 모델별 terminal 지표

|model|A/E/F|Eff|Gen|Loc|Eff NLL|Eff margin|z8 NLL|W8 NLL|W-z gap|held|reactivated|rejected|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|10/10/0|100/100|179/200|875/1000|0.1480|11.7334|0.092282|0.144211|0.051929|62|0|97|
|qwen2.5-7b-inst|10/10/0|96/100|155/200|844/1000|0.8292|10.4578|0.680512|0.734158|0.053646|132|2|155|

## 3. Writer/P/capacity/compute

|model|finite demand Σ mean|old rho Σ mean|predicted Σ mean|actual Σ mean|realization mean|negative steps|P terminal mean|capacity mean|energy mean|functional-P mean|F/B|tokens|mat|edit-core s|eval s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|13.1507|18.1321|13.1507|10.2443|0.7284|0|0.006599|0.054000|0.087323|0.021421|2200/1250|371778|80|1025.13|189.17|
|qwen2.5-7b-inst|13.1505|26.7436|13.1505|9.7203|0.6742|1|0.213970|1.026097|0.862239|0.003490|2200/1250|340727|80|1437.04|212.78|

- peak GPU/host memory: `NOT_RECORDED` in case receipts.

## 4. Target terminal request 분포

|model|case-median mean|case-p90 mean|worst max|count NLL<.05 /100|
|---|---:|---:|---:|---:|
|llama3-8b-inst|0.006208|0.141485|3.707090|89/100|
|qwen2.5-7b-inst|0.026968|1.896820|8.140265|69/100|

## 5. Exact case-paired 산술 비교

`Δ`는 P1R39 − comparison이다. P/capacity/energy ratio는 case ratio의 matched-case 평균이다.

|Model|comparison|matched/attempts|ΔEff|ΔGen|ΔLoc|ΔEff NLL|Δz NLL|ΔW NLL|Δgap|P ratio|capacity ratio|energy ratio|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|P1R38-Neutral|10/10|1|-7|48|-0.446278|-0.326560|-0.484774|-0.158213|0.364816|0.340129|0.390519|
|llama3-8b-inst|P1R36-RS-Neutral|10/10|0|-2|0|0.012612|-0.000027|-0.003322|-0.003295|1.059770|3.655423|3.639086|
|llama3-8b-inst|Official-AlphaEdit|10/10|0|-6|16|0.146856|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|qwen2.5-7b-inst|P1R38-Neutral|10/10|-2|-18|2|0.314223|0.358742|0.325679|-0.033063|0.316586|0.305272|0.525447|
|qwen2.5-7b-inst|P1R36-RS-Neutral|8/10|0|-1|-1|0.316079|0.331927|0.291233|-0.040694|1.069183|2.142759|2.477104|
|qwen2.5-7b-inst|Official-AlphaEdit|10/10|-4|-37|16|0.796559|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|

- Official AlphaEdit identity: exact same model×batch references 20/20 matched through immutable P1R38/P1R31 identity fields.
- Official AlphaEdit latent-z/W/P/capacity/energy: `NOT_RECORDED`; corresponding Δz/ΔW/Δgap/ratios are `NOT_RECORDED`.
- P1R36 Qwen RS-Neutral endpoints: 8/10; its typed-incomplete two batches are comparison `matched=false` and are not imputed.

## 6. Stepwise 기록

|model|k|rows|active mean|held|reactivated|rejected|z NLL|W NLL|gap|finite rho|predicted|actual|realization|negative|P|capacity|energy|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|1|10|10.00|0|0|0|7.707201|10.388497|2.681296|2.681296|2.681296|1.794568|0.670343|0|0.000087|0.025764|0.051529|
|llama3-8b-inst|2|10|10.00|0|0|0|4.796100|8.593928|3.797828|3.797828|3.797828|3.578483|0.938860|0|0.000644|0.098595|0.195092|
|llama3-8b-inst|3|10|10.00|0|0|1|2.621268|5.015446|2.394178|2.394178|2.394178|1.868793|0.774581|0|0.001516|0.139705|0.265171|
|llama3-8b-inst|4|10|9.70|3|0|3|1.307883|3.146653|1.838770|1.838770|1.838770|1.335419|0.733808|0|0.002457|0.111128|0.201444|
|llama3-8b-inst|5|10|9.20|8|0|7|0.585949|1.811234|1.225286|1.225286|1.225286|0.855126|0.701183|0|0.003609|0.113523|0.197681|
|llama3-8b-inst|6|10|8.60|14|0|15|0.290003|0.956108|0.666104|0.666104|0.666104|0.434725|0.644942|0|0.004503|0.063533|0.106344|
|llama3-8b-inst|7|10|8.20|18|0|29|0.146584|0.521383|0.374799|0.374799|0.374799|0.255885|0.679560|0|0.005657|0.077463|0.128034|
|llama3-8b-inst|8|10|8.10|19|0|42|0.093070|0.265498|0.172428|0.172428|0.172428|0.121286|0.684163|0|0.006599|0.054000|0.087323|
|qwen2.5-7b-inst|1|10|10.00|0|0|0|7.272526|10.454479|3.181953|3.181953|3.181953|1.978152|0.610076|0|0.003173|0.167013|0.334026|
|qwen2.5-7b-inst|2|10|10.00|0|0|0|4.248613|8.476327|4.227715|4.227715|4.227715|3.677981|0.867376|0|0.027278|0.907555|1.678655|
|qwen2.5-7b-inst|3|10|9.90|1|0|2|2.376931|4.798347|2.421416|2.421416|2.421416|1.718514|0.703416|0|0.050364|0.794091|1.063626|
|qwen2.5-7b-inst|4|10|9.60|4|0|10|1.512813|3.079833|1.567020|1.567020|1.567020|1.117677|0.716627|0|0.076875|0.784808|0.836785|
|qwen2.5-7b-inst|5|10|8.60|14|0|25|1.138731|1.962155|0.823425|0.823425|0.823425|0.561119|0.684150|0|0.104375|0.617854|0.611238|
|qwen2.5-7b-inst|6|10|7.10|29|0|32|0.895114|1.401037|0.505923|0.505923|0.505923|0.364117|0.682624|0|0.137438|0.771452|0.651395|
|qwen2.5-7b-inst|7|10|6.00|40|1|42|0.789733|1.036919|0.247187|0.247187|0.247187|0.177440|0.649422|0|0.176578|0.953833|0.735317|
|qwen2.5-7b-inst|8|10|5.60|44|1|44|0.683617|0.859480|0.175863|0.175863|0.175863|0.125322|0.480263|1|0.213970|1.026097|0.862239|

- Stepwise heldout Eff/Gen/Loc: `NOT_RECORDED`; 위 표는 scientific-path raw-free receipts이다.

## 7. Machine-readable 산출물

- `p1r39-per-case.json/csv`: 20 rows
- `p1r39-per-step.json/csv`: 160 rows
- `p1r39-per-request.json/csv`: 1,600 rows
- `p1r39-paired-comparisons.json/csv`: 60 case-comparison rows + 6 aggregate rows(JSON)

## 8. 기록 경계

- 포함: identities, denominators, raw metrics, arithmetic deltas, mechanical PASS/FAIL, NOT_RECORDED.
- scientific interpretation, causal claim, superiority/promise, recommendation: not included.
- `scientific_promotion=false`.
