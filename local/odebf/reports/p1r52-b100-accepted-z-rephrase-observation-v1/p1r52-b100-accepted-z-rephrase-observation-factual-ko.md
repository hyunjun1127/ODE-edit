# P1R52 Llama 10×B100 accepted-z rewrite/rephrase 관측 보고서

> 네 방법의 method-native accepted z를 각 방법의 실제 target layer/subject-last 위치에 직접 주입한 observation-only 결과입니다. 기존 물리 W 결과는 봉인된 10×B100 보고서에서 결합했습니다.

## 핵심 경계

- 네 arm 모두 10개 B100, arm당 1,000 requests를 사용하며 missing/imputation/proxy는 0입니다.
- z 관측은 writer/router/controller/history/cache/selection/materialization 영향 0, 추가 backward/generation 0입니다.
- method-specific batch-entry W_(b-1) aggregate는 사용자 지시에 따라 측정·분석·보고하지 않습니다.
- 기존 `full-six z target_new NLL`은 방법 내부 target-objective panel이며, 이 보고서의 rewrite/rephrase NLL과 다른 지표입니다.
- 방법마다 accepted-z 생성 정의는 native implementation을 보존했습니다. 따라서 raw NLL은 나란히 제시하지만 완전히 동일한 latent intervention이라는 인과 주장은 하지 않습니다.

## B1–B10 및 전체: accepted-z와 물리 W target-new NLL

|Arm|범위|z rewrite|z rephrase|W post rewrite|W post rephrase|W10 rewrite|W10 rephrase|Wpost−z rewrite|Wpost−z rephrase|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|B1|0.000961|1.046106|0.245842|2.935170|2.336620|2.915932|+0.244882|+1.889064|
|Official EasyEdit MEMIT Sequential|B2|0.000936|1.076886|0.040514|2.497735|1.645686|2.532723|+0.039578|+1.420849|
|Official EasyEdit MEMIT Sequential|B3|0.000992|0.955792|0.096054|2.311932|1.359235|2.314725|+0.095063|+1.356141|
|Official EasyEdit MEMIT Sequential|B4|0.001038|0.891641|0.024946|1.662936|1.475121|2.167001|+0.023908|+0.771294|
|Official EasyEdit MEMIT Sequential|B5|0.001491|0.851034|0.006150|1.671538|0.848830|1.820792|+0.004659|+0.820503|
|Official EasyEdit MEMIT Sequential|B6|0.000993|0.720384|0.042507|1.781221|0.510214|1.602152|+0.041514|+1.060837|
|Official EasyEdit MEMIT Sequential|B7|0.001789|0.703446|0.004931|1.317199|0.247362|1.228061|+0.003142|+0.613753|
|Official EasyEdit MEMIT Sequential|B8|0.002966|0.667794|0.026736|1.184921|0.064296|0.927945|+0.023771|+0.517127|
|Official EasyEdit MEMIT Sequential|B9|0.005543|0.414584|0.070306|1.009609|0.092972|0.894111|+0.064762|+0.595025|
|Official EasyEdit MEMIT Sequential|B10|0.006105|0.405064|0.033218|1.129417|0.033218|1.129417|+0.027112|+0.724354|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B1|0.001234|1.039913|0.001309|1.816861|0.142481|2.069051|+0.000075|+0.776948|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B2|0.000877|0.998360|0.000934|1.576246|0.003082|1.618159|+0.000057|+0.577887|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B3|0.000949|0.800185|0.001045|1.522066|0.001930|1.577621|+0.000096|+0.721880|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B4|0.000934|0.781591|0.000986|1.087177|0.111800|1.098559|+0.000053|+0.305587|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B5|0.001191|0.950979|0.001294|1.219885|0.002136|1.201763|+0.000103|+0.268906|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B6|0.001031|0.801813|0.001085|1.313215|0.001332|1.328125|+0.000055|+0.511402|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B7|0.001310|0.732600|0.001400|0.964703|0.001418|0.960443|+0.000090|+0.232102|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B8|0.001945|0.816610|0.002081|1.092612|0.002181|1.072897|+0.000136|+0.276002|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B9|0.001856|0.578979|0.002051|0.836380|0.002194|0.830764|+0.000195|+0.257401|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|B10|0.003046|0.599214|0.003134|0.871778|0.003134|0.871778|+0.000087|+0.272564|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B1|0.023180|1.350741|0.073150|2.332472|0.097192|2.198852|+0.049970|+0.981731|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B2|0.027976|1.340242|0.036626|2.208965|0.040882|2.050267|+0.008650|+0.868723|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B3|0.015015|1.080996|0.036557|2.319338|0.036082|2.201169|+0.021542|+1.238342|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B4|0.021597|1.276346|0.026962|2.140571|0.100210|2.118011|+0.005364|+0.864224|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B5|0.012889|1.351685|0.025994|2.601523|0.024607|2.509101|+0.013105|+1.249838|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B6|0.022770|1.210138|0.026228|2.576337|0.039251|2.500658|+0.003458|+1.366199|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B7|0.051295|1.301918|0.065810|2.621618|0.053097|2.555930|+0.014515|+1.319700|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B8|0.115350|1.364368|0.116995|2.656359|0.119663|2.587544|+0.001644|+1.291991|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B9|0.012210|1.050987|0.018274|2.387324|0.017949|2.374438|+0.006064|+1.336336|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|B10|0.019018|1.389080|0.035641|3.211433|0.035641|3.211433|+0.016623|+1.822353|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B1|0.023180|1.350741|0.073150|2.332472|0.093359|2.197095|+0.049970|+0.981731|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B2|0.032200|1.339968|0.043493|2.220012|0.047637|2.068981|+0.011292|+0.880044|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B3|0.018342|1.097508|0.026246|2.242610|0.028106|2.110561|+0.007904|+1.145102|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B4|0.020572|1.288149|0.028372|2.204623|0.095595|2.205840|+0.007800|+0.916474|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B5|0.012185|1.318658|0.022286|2.549342|0.022106|2.488653|+0.010102|+1.230684|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B6|0.024559|1.226300|0.027896|2.507661|0.027737|2.437516|+0.003337|+1.281362|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B7|0.075691|1.232192|0.084228|2.457655|0.074909|2.410231|+0.008537|+1.225463|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B8|0.113189|1.369439|0.120483|2.720777|0.124681|2.645374|+0.007294|+1.351338|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B9|0.015127|1.105161|0.019592|2.392848|0.019961|2.400028|+0.004465|+1.287687|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|B10|0.024767|1.406252|0.046181|3.274713|0.046181|3.274713|+0.021413|+1.868461|
|Official EasyEdit MEMIT Sequential|전체 1000|0.002281|0.773273|0.059120|1.750168|0.861355|1.753286|+0.056839|+0.976895|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|전체 1000|0.001437|0.810024|0.001532|1.230092|0.027169|1.262916|+0.000095|+0.420068|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|전체 1000|0.032130|1.271650|0.046224|2.505594|0.056457|2.430740|+0.014093|+1.233944|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|전체 1000|0.035981|1.273437|0.049193|2.490271|0.058027|2.423899|+0.013211|+1.216835|

## 전체 1,000 requests 성공·정확도 절대값

|Arm|Panel|EFF/rewrite success|Rewrite Acc|GEN/rephrase success|GEN strict|Rephrase Acc|Acc strict|
|---|---|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|accepted z|1000/1000 (100.00%)|1000/1000 (100.00%)|1980/2000 (99.00%)|982/1000 (98.20%)|1628/2000 (81.40%)|692/1000 (69.20%)|
|Official EasyEdit MEMIT Sequential|W immediate-post|996/1000 (99.60%)|991/1000 (99.10%)|1825/2000 (91.25%)|862/1000 (86.20%)|1332/2000 (66.60%)|512/1000 (51.20%)|
|Official EasyEdit MEMIT Sequential|W final-W10|942/1000 (94.20%)|819/1000 (81.90%)|1777/2000 (88.85%)|838/1000 (83.80%)|1287/2000 (64.35%)|512/1000 (51.20%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|accepted z|1000/1000 (100.00%)|1000/1000 (100.00%)|1981/2000 (99.05%)|984/1000 (98.40%)|1628/2000 (81.40%)|694/1000 (69.40%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|W immediate-post|1000/1000 (100.00%)|1000/1000 (100.00%)|1922/2000 (96.10%)|937/1000 (93.70%)|1495/2000 (74.75%)|606/1000 (60.60%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|W final-W10|998/1000 (99.80%)|996/1000 (99.60%)|1909/2000 (95.45%)|929/1000 (92.90%)|1479/2000 (73.95%)|598/1000 (59.80%)|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|accepted z|999/1000 (99.90%)|997/1000 (99.70%)|1951/2000 (97.55%)|958/1000 (95.80%)|1432/2000 (71.60%)|567/1000 (56.70%)|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|W immediate-post|998/1000 (99.80%)|995/1000 (99.50%)|1687/2000 (84.35%)|748/1000 (74.80%)|1019/2000 (50.95%)|321/1000 (32.10%)|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|W final-W10|999/1000 (99.90%)|992/1000 (99.20%)|1696/2000 (84.80%)|751/1000 (75.10%)|1023/2000 (51.15%)|324/1000 (32.40%)|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|accepted z|999/1000 (99.90%)|997/1000 (99.70%)|1954/2000 (97.70%)|961/1000 (96.10%)|1435/2000 (71.75%)|566/1000 (56.60%)|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|W immediate-post|998/1000 (99.80%)|995/1000 (99.50%)|1680/2000 (84.00%)|737/1000 (73.70%)|1018/2000 (50.90%)|315/1000 (31.50%)|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|W final-W10|998/1000 (99.80%)|992/1000 (99.20%)|1696/2000 (84.80%)|748/1000 (74.80%)|1032/2000 (51.60%)|324/1000 (32.40%)|

## accepted-z NLL 분포와 z→W gap

|Arm|z rewrite mean/median/p90/max|z rephrase mean/median/p90/max|Wpost−z rewrite mean|Wpost−z rephrase mean|W10−z rewrite mean|W10−z rephrase mean|
|---|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|0.002281/0.000774/0.005191/0.221680|0.773273/0.026306/2.703125/16.375000|+0.056839|+0.976895|+0.859074|+0.980013|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|0.001437/0.000681/0.002945/0.105469|0.810024/0.022034/3.109375/15.375000|+0.000095|+0.420068|+0.025731|+0.452891|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|0.032130/0.012451/0.044678/7.218750|1.271650/0.211914/4.190625/16.875000|+0.014093|+1.233944|+0.024327|+1.159090|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|0.035981/0.013489/0.048584/7.031250|1.273437/0.218750/4.193750/16.875000|+0.013211|+1.216835|+0.022046|+1.150463|

## R52 controller / route / history

- P1R52 Repair-R1 Soft Sequential / Structural-H ON: PRIMARY `7870`, RESCUE `75`, CURRENT `55`, active-gradient `8000`, clamp-hit `99`, fallback `0`, negative physical progress `0`, history widths `[0, 100, 200, 300, 400, 500, 600, 700, 800, 900]`, H status `{'H_EMPTY_EXACT_ATOMIC_EQUIVALENCE': 8, 'H_ACTIVE_CERTIFIED': 72}`.
- P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF: PRIMARY `7888`, RESCUE `64`, CURRENT `48`, active-gradient `8000`, clamp-hit `114`, fallback `0`, negative physical progress `0`, history widths `[0, 100, 200, 300, 400, 500, 600, 700, 800, 900]`, H status `{'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 80}`.

## Hard / realization-loss cohort

|Arm|z rewrite fail|z rephrase strict fail|z rewrite success→Wpost fail|z rephrase strict success→Wpost fail|
|---|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|0|18|4|121|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|0|16|0|48|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|1|42|1|211|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|1|39|1|226|

## Observation compute

|Arm|F|B|generation|tokens|wall s|hook calls|action influence|
|---|---:|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|1000|0|0|301441|120.86|1000|0|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|1000|0|0|301441|120.15|1000|0|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|1000|0|0|301441|119.49|1000|0|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|1000|0|0|301441|120.01|1000|0|

## Machine-readable artifacts

- aggregate: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1/local/odebf/reports/p1r52-b100-accepted-z-rephrase-observation-v1/p1r52-accepted-z-aggregates.json` (4 rows)
- per-batch: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1/local/odebf/reports/p1r52-b100-accepted-z-rephrase-observation-v1/p1r52-accepted-z-per-batch.json` (40 rows)
- per-request: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1/local/odebf/reports/p1r52-b100-accepted-z-rephrase-observation-v1/p1r52-accepted-z-per-request.json` (4,000 rows)
- hard cohorts: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1/local/odebf/reports/p1r52-b100-accepted-z-rephrase-observation-v1/p1r52-accepted-z-hard-cohorts.json`
- routing/compute: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1/local/odebf/reports/p1r52-b100-accepted-z-rephrase-observation-v1/p1r52-accepted-z-routing-compute.json`

scientific_promotion=false
