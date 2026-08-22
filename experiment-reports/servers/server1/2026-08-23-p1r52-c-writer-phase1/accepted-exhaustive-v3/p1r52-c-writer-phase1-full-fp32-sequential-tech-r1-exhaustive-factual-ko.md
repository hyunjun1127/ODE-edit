# P1R52 C-writer Phase 1 FULL-FP32 Alpha-cache sequential 5-arm — 최종 상세 사실 보고서

> **최우선 실행 경계:** C0/C1/C3는 이전 J0와 달리 각 B100에서 P1R52 accepted-z를 K1→K8까지 전부 최적화한 뒤 writer를 한 번만 적용했다. K-step 표는 write 이전 target/z trajectory이지 중간 writer endpoint가 아니다.

> 이 문서는 v2 상세 분석을 그대로 보존하면서 요청된 accuracy 및 W−z NLL gap을 명시적으로 보강한 canonical v3다. 모든 arm은 FULL FP32이며 BF16/FP16/autocast/quantization/numeric cast=0, imputation=0, scientific promotion=false다.

## A. accepted-z / native-z: success·accuracy·NLL

AlphaEdit native-z, MEMIT latent-z, P1R52 accepted-z는 provenance가 다르다. 특히 MEMIT 값을 P1R52 accepted-z로 해석하지 않는다.

|arm|prompt|success|accuracy|success strict|accuracy strict|target-new mean/median/p90|max|target-true mean/median/p90|max|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|alphaedit|rewrite|1000/1000|1000/1000|1000/1000|1000/1000|0.001392/0.000743/0.003357|0.025194|15.196513/14.896466/20.951120|31.449064|
|alphaedit|rephrase|1987/2000|1631/2000|988/1000|694/1000|0.797691/0.069259/2.721582|12.862535|12.183187/11.778222/17.397113|24.276031|
|memit|rewrite|1000/1000|1000/1000|1000/1000|1000/1000|0.002097/0.000903/0.004589|0.043856|14.776316/14.372349/20.241045|27.454245|
|memit|rephrase|1980/2000|1626/2000|981/1000|689/1000|0.771799/0.081747/2.646872|12.733783|11.586472/11.335657/16.863792|24.677603|
|c0|rewrite|999/1000|996/1000|999/1000|996/1000|0.058140/0.020117/0.087212|6.398037|10.795876/10.268672/16.056259|24.007744|
|c0|rephrase|1940/2000|1394/2000|947/1000|551/1000|1.388216/0.603655/3.972267|13.433740|8.992932/8.784340/13.507415|21.037389|
|c1|rewrite|1000/1000|997/1000|1000/1000|997/1000|0.039680/0.014133/0.060342|6.349159|11.218118/10.740218/16.260996|23.287218|
|c1|rephrase|1937/2000|1420/2000|941/1000|564/1000|1.334686/0.486903/3.972267|13.100144|9.222917/9.087888/13.712799|20.709798|
|c3|rewrite|998/1000|994/1000|998/1000|994/1000|0.059838/0.020173/0.084153|6.350801|10.862917/10.352556/16.015209|23.499222|
|c3|rephrase|1930/2000|1400/2000|941/1000|550/1000|1.382232/0.597201/3.972267|13.832182|9.085106/8.901217/13.746833|23.287881|

## B. immediate-post W: success·accuracy·NLL

|arm|prompt|success|accuracy|success strict|accuracy strict|target-new mean/median/p90|max|target-true mean/median/p90|max|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|alphaedit|rewrite|1000/1000|1000/1000|1000/1000|1000/1000|0.001493/0.000804/0.003582|0.026080|15.100299/14.800235/20.819164|31.112593|
|alphaedit|rephrase|1948/2000|1511/2000|955/1000|613/1000|1.159276/0.290779/3.386415|13.746431|10.758104/10.754892/15.675660|23.082351|
|memit|rewrite|996/1000|989/1000|996/1000|989/1000|0.076587/0.003204/0.022911|12.664020|12.895130/12.687969/18.405182|24.849831|
|memit|rephrase|1841/2000|1344/2000|879/1000|528/1000|1.678150/0.603241/4.716975|18.988174|9.286820/9.209508/14.490139|21.984386|
|c0|rewrite|999/1000|996/1000|999/1000|996/1000|0.052697/0.018326/0.082074|6.204568|10.798762/10.335991/16.108162|24.129406|
|c0|rephrase|1661/2000|965/2000|719/1000|294/1000|2.651556/1.954561/6.214141|13.705603|7.129028/6.777652/11.718503|19.612071|
|c1|rewrite|1000/1000|997/1000|1000/1000|997/1000|0.035539/0.012934/0.052743|6.171869|11.232102/10.763325/16.391319|23.266331|
|c1|rephrase|1686/2000|1021/2000|739/1000|320/1000|2.540690/1.845693/6.067925|13.710615|7.296926/7.000891/11.920898|19.580434|
|c3|rewrite|998/1000|992/1000|998/1000|992/1000|0.053884/0.018877/0.075232|6.184371|10.874818/10.489680/15.930799|23.573730|
|c3|rephrase|1682/2000|1023/2000|742/1000|332/1000|2.526564/1.821163/5.940355|13.703525|7.309436/7.039367/11.763669|21.440433|

## C. W−z target-new NLL gap

|arm|prompt|mean/median/p90|max|min|
|---|---|---:|---:|---:|
|alphaedit|rewrite|0.000101/0.000037/0.000203|0.016238|-0.000607|
|alphaedit|rephrase|0.361585/0.040936/1.071371|13.481135|-3.833974|
|memit|rewrite|0.074490/0.001810/0.015034|12.663803|-0.007608|
|memit|rephrase|0.906351/0.165543/2.917542|17.543275|-3.376095|
|c0|rewrite|-0.005443/-0.000497/0.004904|0.109437|-0.911330|
|c0|rephrase|1.263340/0.655936/3.587641|13.644160|-3.410434|
|c1|rewrite|-0.004140/-0.000423/0.003218|0.042536|-0.983912|
|c1|rephrase|1.206004/0.597848/3.532439|13.649172|-3.457015|
|c3|rewrite|-0.005954/-0.000424/0.005593|0.358097|-0.999555|
|c3|rephrase|1.144332/0.554258/3.350031|13.642082|-3.023500|

---

## D. 전체 상세 본문


> **실행 경계:** C0/C1/C3는 이전 J0와 달리 각 B100에서 P1R52 accepted-z를 K1→K8까지 모두 최적화한 뒤 writer를 정확히 한 번 적용했다. K1→K8은 write 전 target/z trajectory이고 writer endpoint 8개가 아니다.

> 모든 arm은 FULL FP32이며 BF16/FP16/autocast/quantization/numeric storage cast가 0이다. 기존 raw/result/log는 변경하지 않았고 분석 재실행, imputation, promotion은 0이다.

## 1. 완전성·동일성

- scheduler 5/5 COMPLETED(0:0), B100 endpoint 50/50, request 5,000/5,000, C target step 240/240. technical/scientific failure, retry, imputation은 모두 0이다.
- source HEAD/tree `251e616cf95972e230ce40718e1e390ef7dc9eb4` / `34d464c7fedcd21bd7786a74b7540ae2bdf39be9`; stream/order SHA `467e5946…` / `018be113…`.
- C-arm B_r commit→B_(r+1) entry 27/27 exact, 모든 arm terminal W0 pointer/bytes restore exact. C0/C1/C3 target hash는 B1 K1→K8 전부 같고 최초 divergence는 B2 K1이다. 이는 B1 writer가 서로 다른 W를 만든 뒤의 정상적인 sequential closed loop다.
- Alpha cache applicable denominator는 AlphaEdit/C0/C1/C3 4/4: entry/consume width 0,100,…,900, 성공 B 뒤 append100 exactly once, current-batch/prefix history inclusion0. MEMIT은 Alpha cache N/A이며 native static COV_CACHE만 쓴다.

## 2. arm/writer 정의

|arm|z/target|writer|
|---|---|---|
|AlphaEdit|EasyEdit native compute_z|Official `apply_AlphaEdit_to_model`; native dynamic cache_c|
|MEMIT|EasyEdit native latent-z|Official `apply_memit_to_model`; native static COV_CACHE, Alpha cache N/A|
|C0|P1R52 K8 accepted-z|PIR-U control allocation + remaining-residual L4→L8 prefix sequential writer|
|C1|P1R52 K8 accepted-z|Joint P+C epigraph minimax allocation + remaining-residual L4→L8 prefix sequential writer|
|C3|P1R52 K8 accepted-z|direct Official AlphaEdit writer adapter, native compute_z=0|

Native AlphaEdit-z, MEMIT latent-z, P1R52 accepted-z는 provenance가 다르며 같은 target algorithm이라고 주장하지 않는다.

## 3. accepted-z / native-z NLL

|arm|prompt|success|strict|target-new mean/median/p90/max|target-true mean/median/p90/max|
|---|---|---:|---:|---:|---:|
|alphaedit|rewrite|1000/1000|1000/1000|0.001392/0.000743/0.003357/0.025194|15.196513/14.896466/20.951120/31.449064|
|alphaedit|rephrase|1987/2000|988/1000|0.797691/0.069259/2.721582/12.862535|12.183187/11.778222/17.397113/24.276031|
|memit|rewrite|1000/1000|1000/1000|0.002097/0.000903/0.004589/0.043856|14.776316/14.372349/20.241045/27.454245|
|memit|rephrase|1980/2000|981/1000|0.771799/0.081747/2.646872/12.733783|11.586472/11.335657/16.863792/24.677603|
|c0|rewrite|999/1000|999/1000|0.058140/0.020117/0.087212/6.398037|10.795876/10.268672/16.056259/24.007744|
|c0|rephrase|1940/2000|947/1000|1.388216/0.603655/3.972267/13.433740|8.992932/8.784340/13.507415/21.037389|
|c1|rewrite|1000/1000|1000/1000|0.039680/0.014133/0.060342/6.349159|11.218118/10.740218/16.260996/23.287218|
|c1|rephrase|1937/2000|941/1000|1.334686/0.486903/3.972267/13.100144|9.222917/9.087888/13.712799/20.709798|
|c3|rewrite|998/1000|998/1000|0.059838/0.020173/0.084153/6.350801|10.862917/10.352556/16.015209/23.499222|
|c3|rephrase|1930/2000|941/1000|1.382232/0.597201/3.972267/13.832182|9.085106/8.901217/13.746833/23.287881|

## 4. immediate-post W NLL·realization

|arm|prompt|success|strict|target-new mean/median/p90/max|target-true mean/median/p90/max|
|---|---|---:|---:|---:|---:|
|alphaedit|rewrite|1000/1000|1000/1000|0.001493/0.000804/0.003582/0.026080|15.100299/14.800235/20.819164/31.112593|
|alphaedit|rephrase|1948/2000|955/1000|1.159276/0.290779/3.386415/13.746431|10.758104/10.754892/15.675660/23.082351|
|memit|rewrite|996/1000|996/1000|0.076587/0.003204/0.022911/12.664020|12.895130/12.687969/18.405182/24.849831|
|memit|rephrase|1841/2000|879/1000|1.678150/0.603241/4.716975/18.988174|9.286820/9.209508/14.490139/21.984386|
|c0|rewrite|999/1000|999/1000|0.052697/0.018326/0.082074/6.204568|10.798762/10.335991/16.108162/24.129406|
|c0|rephrase|1661/2000|719/1000|2.651556/1.954561/6.214141/13.705603|7.129028/6.777652/11.718503/19.612071|
|c1|rewrite|1000/1000|1000/1000|0.035539/0.012934/0.052743/6.171869|11.232102/10.763325/16.391319/23.266331|
|c1|rephrase|1686/2000|739/1000|2.540690/1.845693/6.067925/13.710615|7.296926/7.000891/11.920898/19.580434|
|c3|rewrite|998/1000|998/1000|0.053884/0.018877/0.075232/6.184371|10.874818/10.489680/15.930799/23.573730|
|c3|rephrase|1682/2000|742/1000|2.526564/1.821163/5.940355/13.703525|7.309436/7.039367/11.763669/21.440433|

정확한 prompt-level z/W NLL과 gap은 `per-prompt-z-w-nll.*` 30,000행에, request-level 평균과 z-success→W-failure는 v1 `per-request-z-w-nll-gap.*` 5,000행에 있다.

## 5. C-arm K1→K8 (write 전)

K별 rewrite/rephrase evaluator NLL·success·accuracy·strict는 저장되지 않았다(`NOT_RECORDED`). 기록된 target objective, gradient norm, displacement, finite 상태만 산출했다.

|arm|K|objective mean/median/p90/max|gradient median/p90|displacement p90|finite|
|---|---:|---:|---:|---:|---:|
|c0|1|9.989300/9.963957/10.594587/10.795511|1.428208/1.771299|8.310757|True|
|c0|2|5.503402/5.299836/6.593369/7.726823|0.914714/1.048036|9.507865|True|
|c0|3|2.388933/2.174544/2.892046/4.738481|0.764882/0.889327|13.233691|True|
|c0|4|0.805217/0.733111/0.995761/1.884441|0.488024/0.650655|8.865977|True|
|c0|5|0.308651/0.218931/0.466689/0.770740|0.286797/0.361040|4.909978|True|
|c0|6|0.165022/0.117135/0.336139/0.349811|0.165454/0.260812|3.994230|True|
|c0|7|0.113334/0.085549/0.238397/0.272485|0.116945/0.246709|2.282254|True|
|c0|8|0.084933/0.053793/0.208904/0.216422|0.091385/0.154927|1.467021|True|
|c1|1|10.002295/9.963315/10.635802/10.819618|1.418144/1.773263|8.349317|True|
|c1|2|5.490120/5.307464/6.587813/7.726823|0.912234/1.056441|9.583584|True|
|c1|3|2.377394/2.169798/2.884268/4.738481|0.725287/0.915423|14.442777|True|
|c1|4|0.804102/0.724733/0.984792/1.884441|0.520467/0.681806|9.124430|True|
|c1|5|0.284864/0.202300/0.419736/0.770740|0.306958/0.388061|6.226765|True|
|c1|6|0.134733/0.091081/0.266453/0.336139|0.147237/0.212913|2.348297|True|
|c1|7|0.081726/0.060107/0.150742/0.234260|0.078558/0.217207|1.524145|True|
|c1|8|0.056244/0.034470/0.090308/0.213476|0.047189/0.154927|0.923538|True|
|c3|1|9.984260/9.971714/10.560407/10.696823|1.218471/1.531761|8.037109|True|
|c3|2|5.738911/5.617711/6.588177/7.726823|0.854724/0.958989|9.832605|True|
|c3|3|2.522831/2.344176/2.884153/4.738481|0.693109/0.801512|13.710849|True|
|c3|4|0.870733/0.788775/0.986295/1.884441|0.559706/0.631639|8.151444|True|
|c3|5|0.348205/0.335568/0.415515/0.770740|0.362067/0.434401|5.289805|True|
|c3|6|0.169573/0.168532/0.248703/0.336139|0.189932/0.260812|3.578228|True|
|c3|7|0.106204/0.098764/0.161616/0.206120|0.121939/0.223313|2.445585|True|
|c3|8|0.076480/0.059196/0.134519/0.185002|0.096511/0.197437|1.808283|True|

## 6. P/C route 및 layer update

|arm|solver success|mean writer pi L4…L8|selected P median|selected C median|minimax t median|fallback/hard-budget/weighted-sum|
|---|---:|---|---:|---:|---:|---:|
|c0|10/10|0.1754,0.2062,0.2136,0.1973,0.2075|0.008422|2.948927|0.995282|0/0/0|
|c1|10/10|0.1871,0.2152,0.2151,0.2004,0.1822|0.008965|3.087761|0.995436|0/0/0|
|c3|N/A|N/A|N/A|N/A|N/A|N/A|

C0/C1의 batch별 P/C cost, normalized certificate, slack, stationarity, control/joint/selected pi는 `pc-route-per-batch.*`에 있다. C3는 P/C router가 없는 direct Official writer라 N/A다.

|arm|layer|update norm median|energy median|energy share median|q norm median|rounding-error norm p90|
|---|---:|---:|---:|---:|---:|---:|
|c0|4|1.078374|1.163434|0.056526|3.405580|0.000001958|
|c0|5|1.317531|1.735973|0.081471|3.019861|0.000001967|
|c0|6|1.615429|2.609682|0.132958|2.941434|0.000001967|
|c0|7|2.123937|4.512218|0.219446|3.059832|0.000001996|
|c0|8|3.272556|10.728205|0.511084|3.025108|0.000001993|
|c1|4|1.221850|1.495336|0.064159|3.405580|0.000001959|
|c1|5|1.444507|2.088500|0.089425|3.020910|0.000001967|
|c1|6|1.764116|3.113071|0.137312|2.943210|0.000001967|
|c1|7|2.335873|5.458784|0.238878|3.058477|0.000001996|
|c1|8|3.272747|10.729975|0.472036|3.013787|0.000001993|
|c3|4|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|c3|5|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|c3|6|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|c3|7|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|c3|8|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|

C0/C1은 100/100 layer telemetry가 기록됐다. C3는 aggregate Official apply receipt만 있고 per-layer norm/energy가 없으므로 50개 layer slot을 명시적으로 `NOT_RECORDED_NATIVE_OFFICIAL_APPLY`로 두었다. FP32 actual delta와 prepared update의 작은 addition-rounding 차이는 observation-only이며 abort/router 영향0이다.

## 7. B10 final forgetting·locality

|arm|rewrite success Δ(final-immediate) mean|rephrase success Δ mean|rephrase accuracy Δ mean|LOC immediate→final|
|---|---:|---:|---:|---:|
|alphaedit|-0.006000|-0.007000|-0.008500|8183/10000 (81.83%) → 7810/10000 (78.10%)|
|memit|-0.042000|-0.017000|-0.025000|8113/10000 (81.13%) → 7239/10000 (72.39%)|
|c0|-0.001000|+0.001000|+0.023500|8590/10000 (85.90%) → 8371/10000 (83.71%)|
|c1|-0.001000|+0.002000|+0.018000|8579/10000 (85.79%) → 8379/10000 (83.79%)|
|c3|+0.000000|-0.007000|+0.001500|8646/10000 (86.46%) → 8508/10000 (85.08%)|

Batch별 immediate→final-W10 변화와 NLL 증가량은 `final-w10-forgetting.*`에 있다. 이는 동일 sequential denominator의 사실값이며 cache 폭 변화의 인과 효과로 단정하지 않는다.

## 8. 시간·compute

|arm|job total s|Δ vs AlphaEdit s|ratio|target median s|writer median s|case total median s|
|---|---:|---:|---:|---:|---:|---:|
|alphaedit|14890.526|+0.000|1.000|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|memit|13845.522|-1045.005|0.930|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|c0|12172.495|-2718.031|0.817|840.968836|126.471767|1145.878690|
|c1|11577.901|-3312.625|0.778|788.882933|119.351208|1075.644136|
|c3|12149.149|-2741.377|0.816|813.781800|165.231662|1112.669982|

AlphaEdit baseline에 per-batch edit-core/case timer가 없어 paired B1–B10 overhead는 `NOT_COMPARABLE`; job-total ratio는 model load/evaluator를 포함한 보조값이다. C-arm target 생성 median은 writer median보다 크지만 wall share만으로 인과를 주장하지 않는다. 기록된 F/B/token/QP/evaluator count는 `compute-job-aggregate.*`; solve/apply/capture는 C writer receipt에 있고 없는 값은 역추정하지 않았다.

## 9. 사실 분류

- immediate rewrite는 모든 arm이 996–1000/1000이지만, C-arm rephrase strict는 719–742/1000으로 AlphaEdit 955/1000보다 낮다. 반면 immediate LOC는 C0/C1/C3 85.79–86.46%로 AlphaEdit 81.83%, MEMIT 81.13%보다 높다.
- C0↔C1은 routing/allocation, C1↔C3은 Joint-P+C remaining-residual 대 direct Official writer, C0↔C3은 control writer 대 Official writer 차이다. target/z는 B1에서 동일하고 최초 writer 이후 궤적이 분기한다.
- 자동 promotion, 새 threshold, causal claim은 없다(`scientific_promotion=false`).

## 10. 재현성

- 입력 v1 rooted receipt `57d6b603b80945c5f6657c3403ba10ff294c0f2a9a08b616dfc37490c020e425`와 artifact inventory root `f5414ac8caf75430bc9b2665c4cfe81924d1194fc51a44398a2b023ff130a10f`를 재사용했다. 원 raw root는 immutable이다.
- ODE run source HEAD/tree는 봉인됐다. EasyEdit source SHA는 post-run read-only rehash이므로 runtime byte identity라고 과장하지 않는다. 모든 v2 표의 row count/SHA와 보고서 SHA는 manifest/rooted receipt에 결속한다.
