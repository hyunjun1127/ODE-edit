# P1R36 P1R35 Independent B10×10 Full-Matrix factual comparison

- 생성 시각(Asia/Seoul): 2026-08-14T00:57:31.813433+09:00
- 정책: `SH_FACTUAL_ONLY_REPORTING`
- 실험: 8 cells × 10 independent B10 = 80 attempts; 각 B10은 W0에서 시작하고 종료 후 W0를 복원
- `scientific_promotion=false` (administrative boundary)
- 모델/evaluator 재실행: 0; 분석 중 scientific source/result mutation: 0

## 1. Source, stream, comparison identity

- P1R36 source head/tree/parent: `87efd168fc9680cabde878cd3b595e98db66d8fa` / `0df5a4318b7886145e6b083d48db982de56b37ed` / `79ffa93205f4d800f4bf54bc07296a30a4a702ab`
- P1R35 scientific parent: `a625e3d1cded3ced0e9128ef7a44205953041447`; TECH-R1 ancestor: `117ece2ee1132e7277a0b43ec5ac77feebfb2646`
- stream root/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- P1R31 report SHA-256: `e487668709934786dc8415fe7d34be0078df858e34c3318a07990616ec75047f` (expected `e487668709934786dc8415fe7d34be0078df858e34c3318a07990616ec75047f`)
- P1R31 identity checks: 79 rows; matched=true
- Official AlphaEdit: P1R31의 동일 model+batch 공통 baseline 필드를 사용; arm별 별도 실행이 아님
- P1R36 stepwise heldout Eff/Gen/Loc: `NOT_RECORDED`; terminal action-freeze 뒤 endpoint evaluator만 실행

## 2. Scheduler and terminal integrity

|task|model|alloc|arm|job|state|elapsed|MaxRSS KiB|attempts|endpoints|failures|W0|
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
|0|llama3-8b-inst|RS|Neutral|19473|COMPLETED0|22:49|7464124|10|10|0|true|
|1|llama3-8b-inst|RS|Soft|19474|COMPLETED0|22:43|7486216|10|8|2|true|
|2|llama3-8b-inst|BG|Neutral|19475|COMPLETED0|26:37|7435400|10|10|0|true|
|3|llama3-8b-inst|BG|Soft|19476|COMPLETED0|24:17|7395044|10|10|0|true|
|4|qwen2.5-7b-inst|RS|Neutral|19477|COMPLETED0|27:31|10778104|10|8|2|true|
|5|qwen2.5-7b-inst|RS|Soft|19478|COMPLETED0|28:59|10873120|10|9|1|true|
|6|qwen2.5-7b-inst|BG|Neutral|19479|COMPLETED0|29:40|10791372|10|8|2|true|
|7|qwen2.5-7b-inst|BG|Soft|19472|COMPLETED0|27:53|10680336|10|8|2|true|

- attempts/endpoints/typed incomplete: `80/71/9`
- technical failures: `0`
- 성공 endpoint: K8/tau1/materialization8/action-freeze/W0 restore PASS 71/71
- typed incomplete: W0 pointer+byte restore 및 next-case continuation PASS 9/9
- accepted scientific transitions: 629 (성공 568 + incomplete prefix 61)
- retry/backtracking/imputation: 0

## 3. P1R36 endpoint counts and continuous metrics

성공 endpoint 분모만의 metric 합계이며 attempts/endpoints/failures를 함께 표기한다.

|model|alloc|arm|A/E/F|Eff|Gen|Loc|Eff NLL|Eff margin|Gen NLL|Loc NLL|z8 full6 NLL|W8 full6 NLL|W−z gap|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|BG|Neutral|10/10/0|98/100|178/200|871/1000|0.4133|9.7995|2.4993|10.3080|0.3097|0.4347|0.1249|
|llama3-8b-inst|BG|Soft|10/10/0|97/100|180/200|873/1000|0.4536|9.7315|2.5146|10.3166|0.3088|0.4550|0.1462|
|llama3-8b-inst|RS|Neutral|10/10/0|100/100|181/200|875/1000|0.1354|11.9863|2.2791|10.3932|0.0923|0.1475|0.0552|
|llama3-8b-inst|RS|Soft|10/8/2|80/80|145/160|705/800|0.1447|11.5957|2.2904|10.3978|0.0698|0.1377|0.0679|
|qwen2.5-7b-inst|BG|Neutral|10/8/2|76/80|131/160|662/800|0.8078|8.9355|3.9008|10.0224|0.5131|0.6573|0.1442|
|qwen2.5-7b-inst|BG|Soft|10/8/2|76/80|124/160|676/800|0.8299|8.7553|4.0343|9.9863|0.5440|0.7325|0.1885|
|qwen2.5-7b-inst|RS|Neutral|10/8/2|77/80|125/160|683/800|0.5334|11.0298|4.2610|10.3000|0.3356|0.4319|0.0963|
|qwen2.5-7b-inst|RS|Soft|10/9/1|89/90|142/180|758/900|0.4570|11.6138|4.1921|10.2565|0.2793|0.3457|0.0664|

## 4. P1R36 writer/routing/P/compute receipts

`actual Σ mean`은 기록된 actual transition의 합을 case별로 계산한 평균이다. Typed incomplete 9건의 마지막 accepted-prefix actual은 `NOT_RECORDED`이며, 전체 actual 기록 수는 620/629이다. P/capacity/energy는 각 case의 endpoint 또는 last-valid prefix 값이다.

|model|alloc|arm|finite demand Σ mean|old rho Σ mean|predicted Σ mean|actual Σ mean|realization mean|negative steps|P last mean|capacity last mean|energy last mean|F/B mean(endpoint)|tokens mean(endpoint)|mat total|edit-core s mean|eval s mean|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|BG|Neutral|13.2427|24.6509|13.2427|9.9538|0.7120|0|0.005406|0.063410|0.105915|220.0/125.0|37177.8|80|115.77|18.99|
|llama3-8b-inst|BG|Soft|13.2092|24.8758|13.2092|9.9335|0.6946|1|0.005280|0.052165|0.087949|220.0/125.0|37177.8|80|103.02|18.85|
|llama3-8b-inst|RS|Neutral|13.1706|18.8027|13.1706|10.2410|0.7228|1|0.006240|0.043241|0.071148|220.0/125.0|37177.8|80|93.87|19.05|
|llama3-8b-inst|RS|Soft|13.0952|18.8170|13.0952|10.2230|0.7286|1|0.005949|0.032897|0.054254|220.0/125.0|37208.5|78|97.18|19.03|
|qwen2.5-7b-inst|BG|Neutral|13.7083|45.2849|13.7083|9.5368|0.6751|0|0.179915|0.866062|0.667600|220.0/125.0|34124.1|77|132.22|21.00|
|qwen2.5-7b-inst|BG|Soft|13.6321|46.1775|13.6321|9.6744|0.6933|0|0.119436|0.487617|0.503569|220.0/125.0|34041.2|77|131.69|21.22|
|qwen2.5-7b-inst|RS|Neutral|13.6170|33.0085|13.6170|10.0299|0.7225|0|0.179577|0.587238|0.451934|220.0/125.0|34080.2|78|125.23|21.25|
|qwen2.5-7b-inst|RS|Soft|13.4918|33.1062|13.4918|10.1284|0.7432|0|0.113459|0.149811|0.149014|220.0/125.0|34159.4|79|129.15|21.27|

## 5. Accepted-step trajectory

Heldout Eff/Gen/Loc는 stepwise `NOT_RECORDED`이다. 아래 값은 frozen scientific path의 accepted-prefix receipts이다. `actual n`은 다음 accepted refresh 또는 terminal에서 actual progress가 기록된 행 수다.

|model|alloc|arm|k|prefix n|L_base|L_endpoint|finite rho|old rho|predicted|actual (n)|realization|negative|cum P|capacity|P31 predicted|P31 actual|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|BG|Neutral|1|10|10.3885|7.9814|2.4071|4.1044|2.4071|1.3992 (10)|0.5764|0|0.000055|0.016846|4.1044|NOT_RECORDED|
|llama3-8b-inst|BG|Neutral|2|10|8.9893|5.4445|3.5449|4.2433|3.5449|3.1866 (10)|0.8971|0|0.000458|0.078179|3.4332|NOT_RECORDED|
|llama3-8b-inst|BG|Neutral|3|10|5.8027|3.4769|2.3258|3.7228|2.3258|1.7141 (10)|0.7291|0|0.000986|0.083730|3.4008|NOT_RECORDED|
|llama3-8b-inst|BG|Neutral|4|10|4.0887|2.0217|2.0670|3.4789|2.0670|1.6672 (10)|0.8018|0|0.001935|0.120762|2.9777|NOT_RECORDED|
|llama3-8b-inst|BG|Neutral|5|10|2.4214|1.1051|1.3163|3.2201|1.3163|0.9120 (10)|0.6808|0|0.002930|0.102743|2.9298|NOT_RECORDED|
|llama3-8b-inst|BG|Neutral|6|10|1.5095|0.7626|0.7468|2.4179|0.7468|0.5477 (10)|0.7488|0|0.003679|0.052897|2.8378|NOT_RECORDED|
|llama3-8b-inst|BG|Neutral|7|10|0.9617|0.4758|0.4859|2.1030|0.4859|0.3002 (10)|0.5965|0|0.004431|0.042888|2.3081|NOT_RECORDED|
|llama3-8b-inst|BG|Neutral|8|10|0.6615|0.3127|0.3488|1.3606|0.3488|0.2269 (10)|0.6657|0|0.005406|0.063410|1.0264|0.3025|
|llama3-8b-inst|BG|Soft|1|10|10.3885|7.9814|2.4071|4.1044|2.4071|1.3986 (10)|0.5761|0|0.000053|0.016846|4.1044|NOT_RECORDED|
|llama3-8b-inst|BG|Soft|2|10|8.9899|5.4427|3.5471|4.2412|3.5471|3.1946 (10)|0.8987|0|0.000454|0.079232|3.4351|NOT_RECORDED|
|llama3-8b-inst|BG|Soft|3|10|5.7953|3.4749|2.3204|3.7105|2.3204|1.7058 (10)|0.7265|0|0.000994|0.086661|3.4027|NOT_RECORDED|
|llama3-8b-inst|BG|Soft|4|10|4.0895|2.0296|2.0599|3.5135|2.0599|1.6586 (10)|0.7999|0|0.001962|0.123399|2.9729|NOT_RECORDED|
|llama3-8b-inst|BG|Soft|5|10|2.4309|1.1091|1.3218|3.2470|1.3218|0.9298 (10)|0.6907|0|0.002972|0.100456|2.9401|NOT_RECORDED|
|llama3-8b-inst|BG|Soft|6|10|1.5011|0.7649|0.7362|2.4363|0.7362|0.5401 (10)|0.7553|0|0.003643|0.040152|2.1405|NOT_RECORDED|
|llama3-8b-inst|BG|Soft|7|10|0.9610|0.4959|0.4652|2.1514|0.4652|0.2963 (10)|0.6238|0|0.004355|0.038436|2.1427|NOT_RECORDED|
|llama3-8b-inst|BG|Soft|8|10|0.6648|0.3131|0.3516|1.4716|0.3516|0.2098 (10)|0.4861|1|0.005280|0.052165|1.0068|0.3023|
|llama3-8b-inst|RS|Neutral|1|10|10.3885|7.7071|2.6814|3.6310|2.6814|1.7944 (10)|0.6702|0|0.000087|0.025767|3.6310|NOT_RECORDED|
|llama3-8b-inst|RS|Neutral|2|10|8.5941|4.7925|3.8016|4.0152|3.8016|3.5795 (10)|0.9383|0|0.000645|0.098810|3.2616|NOT_RECORDED|
|llama3-8b-inst|RS|Neutral|3|10|5.0145|2.6266|2.3879|3.2560|2.3879|1.8546 (10)|0.7721|0|0.001504|0.137443|3.1276|NOT_RECORDED|
|llama3-8b-inst|RS|Neutral|4|10|3.1599|1.3176|1.8423|2.7864|1.8423|1.3402 (10)|0.7348|0|0.002421|0.106371|2.4565|NOT_RECORDED|
|llama3-8b-inst|RS|Neutral|5|10|1.8197|0.5847|1.2350|2.2473|1.2350|0.8674 (10)|0.7039|0|0.003536|0.108071|1.8669|NOT_RECORDED|
|llama3-8b-inst|RS|Neutral|6|10|0.9523|0.2793|0.6730|1.7232|0.6730|0.4440 (10)|0.6594|0|0.004419|0.060992|1.3863|NOT_RECORDED|
|llama3-8b-inst|RS|Neutral|7|10|0.5084|0.1379|0.3705|0.7426|0.3705|0.2363 (10)|0.5806|1|0.005516|0.071150|0.6594|NOT_RECORDED|
|llama3-8b-inst|RS|Neutral|8|10|0.2721|0.0932|0.1788|0.4011|0.1788|0.1245 (10)|0.7234|0|0.006240|0.043241|0.4042|0.1017|
|llama3-8b-inst|RS|Soft|1|10|10.3885|7.7071|2.6814|3.6310|2.6814|1.7928 (10)|0.6696|0|0.000086|0.025767|3.6310|NOT_RECORDED|
|llama3-8b-inst|RS|Soft|2|10|8.5957|4.7937|3.8019|4.0125|3.8019|3.5863 (10)|0.9400|0|0.000642|0.099853|3.2584|NOT_RECORDED|
|llama3-8b-inst|RS|Soft|3|10|5.0093|2.6273|2.3820|3.2468|2.3820|1.8592 (10)|0.7759|0|0.001522|0.140487|3.1267|NOT_RECORDED|
|llama3-8b-inst|RS|Soft|4|10|3.1502|1.3147|1.8355|2.7808|1.8355|1.3424 (10)|0.7378|0|0.002467|0.109192|2.4522|NOT_RECORDED|
|llama3-8b-inst|RS|Soft|5|10|1.8078|0.5781|1.2297|2.2395|1.2297|0.8703 (10)|0.7085|0|0.003576|0.103319|1.8833|NOT_RECORDED|
|llama3-8b-inst|RS|Soft|6|10|0.9375|0.2775|0.6600|1.6924|0.6600|0.4588 (10)|0.6979|0|0.004447|0.055865|1.4160|NOT_RECORDED|
|llama3-8b-inst|RS|Soft|7|10|0.4787|0.1292|0.3495|0.8411|0.3495|0.2644 (8)|0.6929|0|0.005352|0.048308|0.5847|NOT_RECORDED|
|llama3-8b-inst|RS|Soft|8|8|0.2648|0.0710|0.1938|0.4662|0.1938|0.1271 (8)|0.5523|1|0.006067|0.035708|0.7850|0.0759|
|qwen2.5-7b-inst|BG|Neutral|1|10|10.4545|7.6779|2.7766|7.5664|2.7766|1.4870 (10)|0.5144|0|0.001706|0.090860|7.5664|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Neutral|2|10|8.9675|4.9238|4.0436|6.1091|4.0436|3.4383 (10)|0.8476|0|0.019259|0.672692|5.3657|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Neutral|3|10|5.5291|3.2680|2.2611|6.0390|2.2611|1.5665 (10)|0.6903|0|0.036554|0.570368|5.5058|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Neutral|4|10|3.9626|2.1778|1.7848|6.2119|1.7848|1.2245 (10)|0.6751|0|0.062613|0.778049|5.9149|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Neutral|5|10|2.7382|1.5594|1.1787|5.1113|1.1787|0.8211 (10)|0.6903|0|0.089522|0.676881|4.9771|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Neutral|6|10|1.9171|1.1630|0.7540|5.7839|0.7540|0.4648 (9)|0.6128|0|0.117301|0.670140|6.2096|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Neutral|7|9|1.3510|0.7881|0.5630|5.2188|0.5630|0.3651 (8)|0.6705|0|0.150186|1.085519|5.4995|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Neutral|8|8|1.0187|0.5153|0.5034|4.7081|0.5034|0.3613 (8)|0.6749|0|0.194971|0.950217|3.5527|0.2316|
|qwen2.5-7b-inst|BG|Soft|1|10|10.4545|7.6779|2.7766|7.5664|2.7766|1.4838 (10)|0.5132|0|0.001388|0.065714|7.5664|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Soft|2|10|8.9707|4.9256|4.0451|6.1158|4.0451|3.4695 (10)|0.8555|0|0.015173|0.459773|5.3425|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Soft|3|10|5.5011|3.2809|2.2202|6.0413|2.2202|1.5509 (10)|0.6945|0|0.027596|0.325958|5.6064|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Soft|4|10|3.9502|2.1822|1.7681|6.3717|1.7681|1.2417 (10)|0.6892|0|0.044897|0.362811|5.9487|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Soft|5|10|2.7085|1.5299|1.1787|5.0813|1.1787|0.8566 (10)|0.7266|0|0.063748|0.345041|5.1763|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Soft|6|10|1.8519|1.1924|0.6595|5.9677|0.6595|0.4869 (9)|0.6885|0|0.077755|0.221722|6.5203|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Soft|7|9|1.4332|0.8512|0.5821|4.9488|0.5821|0.4101 (8)|0.7060|0|0.105843|0.698839|5.0163|NOT_RECORDED|
|qwen2.5-7b-inst|BG|Soft|8|8|1.1144|0.5393|0.5751|5.7242|0.5751|0.3819 (8)|0.6618|0|0.129801|0.546623|5.7880|-0.1552|
|qwen2.5-7b-inst|RS|Neutral|1|10|10.4545|7.2725|3.1820|6.6223|3.1820|1.9771 (10)|0.6097|0|0.003174|0.167020|6.6223|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Neutral|2|10|8.4774|4.2427|4.2347|5.7994|4.2347|3.6801 (10)|0.8664|0|0.027320|0.909280|5.0551|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Neutral|3|10|4.7974|2.4302|2.3672|5.1518|2.3672|1.6732 (10)|0.7005|0|0.048761|0.710265|4.8234|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Neutral|4|10|3.1242|1.5678|1.5564|4.4875|1.5564|1.1123 (10)|0.7310|0|0.069823|0.514029|4.3300|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Neutral|5|10|2.0119|1.0536|0.9583|3.6776|0.9583|0.6781 (10)|0.7039|0|0.096022|0.570664|3.5391|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Neutral|6|10|1.3338|0.6852|0.6486|3.1913|0.6486|0.4726 (10)|0.7248|0|0.124344|0.578479|3.5353|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Neutral|7|10|0.8612|0.4497|0.4115|2.5252|0.4115|0.3121 (8)|0.7312|0|0.157469|0.671782|2.4900|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Neutral|8|8|0.6656|0.3428|0.3228|1.9418|0.3228|0.2337 (8)|0.7214|0|0.180300|0.566816|2.6398|0.5409|
|qwen2.5-7b-inst|RS|Soft|1|10|10.4545|7.2725|3.1820|6.6223|3.1820|1.9801 (10)|0.6106|0|0.002577|0.119402|6.6223|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Soft|2|10|8.4743|4.2551|4.2192|5.7958|4.2192|3.6922 (10)|0.8723|0|0.020919|0.578999|5.0406|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Soft|3|10|4.7822|2.4198|2.3623|5.1292|2.3623|1.7044 (10)|0.7161|0|0.036165|0.370850|4.8219|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Soft|4|10|3.0778|1.5493|1.5284|4.4840|1.5284|1.1078 (10)|0.7410|0|0.050589|0.245969|4.2737|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Soft|5|10|1.9699|1.0778|0.8921|3.5008|0.8921|0.6709 (10)|0.7525|0|0.067081|0.258518|3.9061|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Soft|6|10|1.2991|0.6757|0.6233|3.1679|0.6233|0.4407 (10)|0.7214|0|0.084985|0.261048|3.1165|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Soft|7|10|0.8583|0.4447|0.4137|2.8631|0.4137|0.3534 (9)|0.7758|0|0.100985|0.201981|3.1825|NOT_RECORDED|
|qwen2.5-7b-inst|RS|Soft|8|9|0.5836|0.2828|0.3008|1.7146|0.3008|0.2379 (9)|0.7675|0|0.114743|0.148977|2.1818|0.2196|

## 6. Matched P1R31 P1R24 comparison

각 `Δ`는 `P1R36 − P1R31` 산술 차이다. 각 방법의 endpoint 분모를 별도로 표기한다.

|model|alloc|arm|P36 E/F|P31 E/F|Eff P36/P31/Δ|Gen P36/P31/Δ|Loc P36/P31/Δ|Eff NLL P36/P31/Δ|W full6 P36/P31/Δ|z full6 P36/P31/Δ|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|BG|Neutral|10/0|10/0|98/98/0|178/175/3|871/871/0|0.4133/0.4188/-0.0055|0.4347/0.4400/-0.0053|0.3097/0.3290/-0.0192|
|llama3-8b-inst|BG|Soft|10/0|9/1|97/89/8|180/164/16|873/787/86|0.4536/0.2631/0.1905|0.4550/0.2891/0.1659|0.3088/0.2654/0.0434|
|llama3-8b-inst|RS|Neutral|10/0|10/0|100/99/1|181/178/3|875/873/2|0.1354/0.1430/-0.0076|0.1475/0.1459/0.0016|0.0923/0.0834/0.0090|
|llama3-8b-inst|RS|Soft|8/2|10/0|80/99/-19|145/177/-32|705/873/-168|0.1447/0.1795/-0.0348|0.1377/0.1875/-0.0498|0.0698/0.0847/-0.0149|
|qwen2.5-7b-inst|BG|Neutral|8/2|10/0|76/94/-18|131/165/-34|662/843/-181|0.8078/1.2933/-0.4855|0.6573/1.1138/-0.4565|0.5131/0.8403/-0.3272|
|qwen2.5-7b-inst|BG|Soft|8/2|10/0|76/94/-18|124/164/-40|676/842/-166|0.8299/1.2732/-0.4433|0.7325/1.1584/-0.4259|0.5440/0.7649/-0.2209|
|qwen2.5-7b-inst|RS|Neutral|8/2|10/0|77/98/-21|125/155/-30|683/843/-160|0.5334/0.5851/-0.0517|0.4319/0.5603/-0.1284|0.3356/0.2879/0.0477|
|qwen2.5-7b-inst|RS|Soft|9/1|10/0|89/100/-11|142/163/-21|758/844/-86|0.4570/0.4922/-0.0352|0.3457/0.4267/-0.0810|0.2793/0.2946/-0.0153|

## 7. Official AlphaEdit common model-level reference

Official AlphaEdit는 model×batch당 1개 공통 baseline이며 RS/BG/Neutral/Soft 실행으로 반복하지 않았다.

|model|matched batches|Eff|Gen|Loc|Eff NLL mean|Eff margin mean|full6 z/W|P/capacity|
|---|---:|---:|---:|---:|---:|---:|---|---|
|llama3-8b-inst|10|100/100|185/200|859/1000|0.0012|14.7516|NOT_RECORDED|NOT_RECORDED|
|qwen2.5-7b-inst|10|100/100|192/200|828/1000|0.0326|14.2824|NOT_RECORDED|NOT_RECORDED|

## 8. Typed incomplete cases

|model|alloc|arm|batch|last valid k|classification|exception|message SHA|W0 pointer/bytes|
|---|---|---|---:|---:|---|---|---|---|
|llama3-8b-inst|RS|Soft|2|7|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|llama3-8b-inst|RS|Soft|6|7|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|qwen2.5-7b-inst|BG|Neutral|4|6|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|qwen2.5-7b-inst|BG|Neutral|6|7|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|qwen2.5-7b-inst|BG|Soft|1|7|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|qwen2.5-7b-inst|BG|Soft|3|6|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|qwen2.5-7b-inst|RS|Neutral|2|7|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|qwen2.5-7b-inst|RS|Neutral|9|7|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|
|qwen2.5-7b-inst|RS|Soft|9|7|SCIENTIFIC_FAIL|P1R34NonSemanticTargetMove|`b823c15dcf2090f4cdb19050b6e1004c281c96dd92597216e507128f983044f0`|true/true|

## 9. Identity and counter facts

- all accepted steps: `u=d+lag=target_next-current_terminal`; remaining/fresh/lag division counters 0; semantic debt 0
- physical h application 1/step; second h application 0; finite endpoint backward 0
- inner heldout count 0; history count 0; retry count 0
- terminal evaluator controller/routing influence count 0
- peak GPU memory per case: `NOT_RECORDED`; scheduler MaxRSS는 §2에 표기
- Official AlphaEdit stepwise writer/P/capacity/compute fields: `NOT_RECORDED`

## 10. Machine-readable artifacts

- `p1r36-comparison-per-case.json` / `.csv`: 80 rows, 실패 분모 및 P1R31/Official join 포함
- `p1r36-comparison-per-step.json` / `.csv`: 629 accepted-prefix rows, P1R31 동일 k arithmetic delta 포함
- 세부 batch별 값은 per-case 파일, step별 값은 per-step 파일에 기록

## 11. Boundary

- report contents: facts, values, denominators, arithmetic deltas, identities, typed failures, paths/checksums, NOT_RECORDED
- scientific interpretation/recommendation/promotion judgment: not included
- `scientific_promotion=false`
