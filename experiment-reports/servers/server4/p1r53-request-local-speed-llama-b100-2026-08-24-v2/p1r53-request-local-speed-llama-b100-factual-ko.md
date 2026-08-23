# P1R53 request-local target-speed Llama B100 — factual report

> **범위:** sealed B1 100 requests의 possibility experiment다. LP-S와 LFD-E만 새로 실행했으며, Global P1R52 Z0-COARSE 및 Native AlphaEdit/MEMIT는 동일 B1/W0/evaluator의 sealed external reference다(†). 단일 B100 결과로 promotion하거나 최종 정책을 선택하지 않는다.

## 방법·불변식

- LP-S: `a_i = m_i · s_B · ||g_i(t)|| / (||g_i(0)|| + eps_n)`. `s_B`는 K0에서 봉인한 batch median shared scale이다. 다른 request의 NLL/gradient/unused energy를 decision에 사용하지 않았다.
- LFD-E: K0에서 `kappa_i = a_LP,i(0) · sigma_i(0) / ell_i(0)`를 request별 1회 계산·고정하고, K1–K8에서 `a_i = m_i · kappa_i · ell_i / sigma_i`를 사용했다. `sigma_i = -g_i·d_i > 0`, `d_i = c_i/||c_i||`이다.
- 두 arm 모두 K=8, h=1/8, microstep=1, T=1이며 매 K의 final selected z만 Official AlphaEdit writer에 1회 전달했다. 총 writer 8회/layer apply 40회다.
- target 내부 W/teacher/origin/cache/factor는 고정했고, 추가 model F/B와 request별 backward loop는 0이다. selector/clamp/preservation/direction/evaluator/stream은 Global control과 동일하다.

## 1. 최종 K8 핵심 표

NLL 표기는 `mean/median/p90`이고 낮을수록 좋다. Eff는 W Rewrite success, Gen은 W Rephrase prompt/strict success, Loc는 W locality다.

|method|W Eff|W Gen prompt|W Gen strict|W Loc|z Rewrite NLL|z Rephrase NLL|W Rewrite NLL|W Rephrase NLL|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|LP-S|100/100 (1.000)|181/200 (0.905)|84/100 (0.840)|881/1000 (0.881)|0.102255/0.035672/0.261029|1.517653/0.327990/4.628400|0.101934/0.035611/0.253230|2.326221/0.980605/5.881143|
|LFD-E|95/100 (0.950)|153/200 (0.765)|66/100 (0.660)|888/1000 (0.888)|1.158980/0.418230/3.492959|2.648680/1.630453/6.590988|1.155583/0.419876/3.489254|3.677507/2.813759/8.635519|
|Global P1R52 Z0-COARSE†|100/100 (1.000)|185/200 (0.925)|87/100 (0.870)|889/1000 (0.889)|0.075409/0.018222/0.049245|1.424842/0.237647/4.197616|0.079716/0.017962/0.049609|2.138524/0.784661/5.717817|
|Native AlphaEdit†|100/100 (1.000)|184/200 (0.920)|87/100 (0.870)|872/1000 (0.872)|0.001362/0.000659/0.002425|1.103564/0.064396/3.287930|0.001915/0.000728/0.002553|1.940613/0.309863/5.383312|
|Native MEMIT†|98/100 (0.980)|164/200 (0.820)|76/100 (0.760)|886/1000 (0.886)|0.000885/0.000596/0.001915|1.092572/0.065640/3.515177|0.342457/0.004158/0.076267|2.974135/1.196294/8.534275|

## 2. Rewrite 상세

|method|endpoint|new NLL mean|median|p90|max|true NLL mean|margin mean (new−true)|success|accuracy|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|LP-S|accepted_z|0.102255|0.035672|0.261029|0.896901|10.897022|-10.794767|100/100|99/100|
|LP-S|post_W|0.101934|0.035611|0.253230|0.884429|10.884134|-10.782200|100/100|99/100|
|LFD-E|accepted_z|1.158980|0.418230|3.492959|8.489591|8.210246|-7.051266|95/100|73/100|
|LFD-E|post_W|1.155583|0.419876|3.489254|8.464552|8.212749|-7.057166|95/100|73/100|
|Global P1R52 Z0-COARSE†|accepted_z|0.075409|0.018222|0.049245|4.836492|11.690766|-11.615358|100/100|99/100|
|Global P1R52 Z0-COARSE†|post_W|0.079716|0.017962|0.049609|4.830842|11.667182|-11.587466|100/100|99/100|
|Native AlphaEdit†|accepted_z|0.001362|0.000659|0.002425|0.028182|14.538538|-14.537176|100/100|100/100|
|Native AlphaEdit†|post_W|0.001915|0.000728|0.002553|0.075494|14.423645|-14.421730|100/100|100/100|
|Native MEMIT†|accepted_z|0.000885|0.000596|0.001915|0.005452|14.762211|-14.761325|100/100|100/100|
|Native MEMIT†|post_W|0.342457|0.004158|0.076267|12.670905|11.508459|-11.166001|98/100|96/100|

## 3. Rephrase 상세

|method|endpoint|new NLL mean|median|p90|max|true NLL mean|margin mean (new−true)|success prompt/strict|accuracy prompt/strict|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|LP-S|accepted_z|1.517653|0.327990|4.628400|11.463405|9.471591|-7.953937|195/200 · 96/100|130/200 · 49/100|
|LP-S|post_W|2.326221|0.980605|5.881143|16.703304|8.109066|-5.782845|181/200 · 84/100|107/200 · 37/100|
|LFD-E|accepted_z|2.648680|1.630453|6.590988|12.353718|7.788278|-5.139598|180/200 · 87/100|92/200 · 28/100|
|LFD-E|post_W|3.677507|2.813759|8.635519|17.005484|6.859108|-3.181601|153/200 · 66/100|72/200 · 20/100|
|Global P1R52 Z0-COARSE†|accepted_z|1.424842|0.237647|4.197616|12.096726|9.867747|-8.442905|195/200 · 96/100|138/200 · 52/100|
|Global P1R52 Z0-COARSE†|post_W|2.138524|0.784661|5.717817|14.977410|8.297076|-6.158551|185/200 · 87/100|116/200 · 39/100|
|Native AlphaEdit†|accepted_z|1.103564|0.064396|3.287930|13.755802|11.192899|-10.089335|197/200 · 98/100|151/200 · 63/100|
|Native AlphaEdit†|post_W|1.940613|0.309863|5.383312|16.056799|9.413474|-7.472861|184/200 · 87/100|122/200 · 44/100|
|Native MEMIT†|accepted_z|1.092572|0.065640|3.515177|13.546903|11.240743|-10.148170|197/200 · 98/100|150/200 · 62/100|
|Native MEMIT†|post_W|2.974135|1.196294|8.534275|17.056204|7.998486|-5.024351|164/200 · 76/100|100/200 · 36/100|

## 4. 이 실험에 특수한 request-local speed 표

|arm|request×K|speed mean/median/p90/max|σ mean/median/p10/min|clamp|PRIMARY/RESCUE/CURRENT|
|---|---:|---:|---:|---:|---:|
|LP-S|800|4.92081/3.40677/9.65268/74.6494|2.9872/2.36619/0.304822/0.00129642|34/800 (0.043)|783/17/0|
|LFD-E|800|2.78939/1.79063/6.07364/14.6108|4.30024/3.81362/1.12127/0.00258875|0/800 (0.000)|794/6/0|

### K별 speed energy 집중도

|arm|K|speed mean|median|p90|max|top-1 energy|top-10|bottom-90|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|LP-S|1|6.07364|6.07364|6.07364|6.07364|0.010000|0.100000|0.900000|
|LP-S|2|5.90869|5.47473|9.423|13.831|0.045790|0.310864|0.689136|
|LP-S|3|6.98117|5.41062|11.4155|52.855|0.286079|0.661943|0.338057|
|LP-S|4|5.9839|3.78843|14.1822|45.2076|0.253944|0.675654|0.324346|
|LP-S|5|4.97859|2.28422|11.7862|44.0332|0.274268|0.788732|0.211268|
|LP-S|6|3.95567|1.44723|7.6402|61.4343|0.465682|0.937802|0.062198|
|LP-S|7|2.82096|0.881894|3.44298|46.0838|0.346313|0.973571|0.026429|
|LP-S|8|2.66383|0.598835|2.3653|74.6494|0.509058|0.994675|0.005325|
|LFD-E|1|6.07364|6.07364|6.07364|6.07364|0.010000|0.100000|0.900000|
|LFD-E|2|5.2628|4.7911|8.88479|14.6108|0.061150|0.342003|0.657997|
|LFD-E|3|2.93926|2.50139|5.43422|9.93722|0.084537|0.437503|0.562497|
|LFD-E|4|1.99541|1.79025|3.30629|5.69959|0.064428|0.385576|0.614424|
|LFD-E|5|1.67664|1.4593|2.54115|9.42664|0.213390|0.507065|0.492935|
|LFD-E|6|1.58391|1.30026|2.11719|10.8942|0.240326|0.696995|0.303005|
|LFD-E|7|1.54334|1.08912|1.85329|14.2743|0.322784|0.817297|0.182703|
|LFD-E|8|1.24007|0.90329|1.78776|12.1446|0.414530|0.757209|0.242791|

## 5. Paired delta와 writer 전달

Delta는 앞 method minus 뒤 method이며 음수가 NLL 개선이다. 아래 mean/median/p90은 먼저 pointwise delta를 만든 뒤 집계했다.

|prompt|comparison|z delta mean/median/p90|W delta mean/median/p90|
|---|---|---:|---:|
|rewrite|LFD-E − LP-S|1.056726/0.313460/3.374923|1.053649/0.323512/3.378897|
|rewrite|LP-S − Global|0.026846/0.014325/0.203680|0.022218/0.010518/0.196272|
|rewrite|LFD-E − Global|1.083572/0.388330/2.880940|1.075867/0.387186/2.883079|
|rewrite|LP-S − Native AlphaEdit|0.100892/0.033562/0.259891|0.100019/0.035087/0.252010|
|rewrite|LFD-E − Native AlphaEdit|1.157618/0.417939/3.492328|1.153668/0.419556/3.488831|
|rewrite|LP-S − Native MEMIT|0.101369/0.034049/0.259821|-0.240523/0.019941/0.219914|
|rewrite|LFD-E − Native MEMIT|1.158095/0.417934/3.492403|0.813126/0.376738/2.530274|
|rephrase|LFD-E − LP-S|1.131027/0.386514/3.732443|1.351286/0.548792/4.788666|
|rephrase|LP-S − Global|0.092811/0.040731/0.561127|0.187697/0.049661/1.152065|
|rephrase|LFD-E − Global|1.223838/0.533239/3.626003|1.538983/0.832199/4.370256|
|rephrase|LP-S − Native AlphaEdit|0.414090/0.138816/1.849608|0.385609/0.216393/2.039050|
|rephrase|LFD-E − Native AlphaEdit|1.545116/0.887466/4.151236|1.736895/1.075460/4.542185|
|rephrase|LP-S − Native MEMIT|0.425081/0.138634/1.824057|-0.647913/0.041873/1.433871|
|rephrase|LFD-E − Native MEMIT|1.556107/0.907088/4.151976|0.703372/0.380029/3.188304|

|arm|prompt|pointwise W−z mean/median/p90|max|z success→W failure|
|---|---|---:|---:|---:|
|LP-S|rewrite|-0.000320/-0.000144/0.001813/0.303116|0/100|
|LP-S|rephrase|0.808568/0.209886/2.127665/13.800835|14/200|
|LFD-E|rewrite|-0.003397/-0.000607/0.005108/0.021619|0/100|
|LFD-E|rephrase|1.028827/0.434893/3.206130/8.536153|27/200|

## 6. 계산량·완전성

|arm|Slurm|elapsed|field eval|writer/layer apply|추가 F/B|peak allocated/reserved|W0 restore|
|---|---|---:|---:|---:|---:|---:|---:|
|LP-S|23318_0 COMPLETED 0:0|1150s|8|8/40|0/0|38027805184/42618322944|True|
|LFD-E|23318_1 COMPLETED 0:0|1101s|8|8/40|0/0|38027805184/42618322944|True|

- 최초 `23314_[0-1]`은 모델 로드 전 local session-boundary 파일 부재로 exit `4:0`; PURE_TECHNICAL이며 scientific denominator 영향 0이다.
- 유효 실행 `23318_[0-1]`: 두 arm 모두 terminal, FULL-FP32, field eval8, writer8/layer40, cache reuse8/append1, W0 byte/pointer restore exact.
- K1–K7 heldout full evaluation은 0이고 K8 accepted-z/pre-W/post-W/locality만 관측했다. controller influence는 0이다.
- LP-S/LFD-E 추가 semantic/ KL F/B 및 per-request backward loop는 0이다.

## 7. 가설 판정과 경계

- `LP-S > Global`의 Rewrite accepted-z mean+median 방향: **일관된 지원 아님**. 이는 단일 B100 descriptive signal이며 hard promotion 판정이 아니다.
- `LFD-E > LP-S`의 Rewrite accepted-z mean+median 방향: **일관된 지원 아님**. Rephrase와 materialized W, tail 및 clamp/selector 결과를 함께 봐야 한다.
- † reference는 동일 sealed B1/W0/evaluator의 기존 실행을 재사용했지만 새 두 arm과 같은 job의 동시 causal panel은 아니다. 설정 선택 영향과 재실행은 0이다.
- request-local은 shared batch scale `s_B`를 유지하므로 완전한 batch-invariant method라고 주장하지 않는다.
- 단일 B100×1 결과로 sequential 성능, Qwen 전이, 최종 promotion을 주장하지 않는다. `scientific_promotion=false`다.

## 8. 세부 산출물

- `paired-request-deltas.csv`: arm/reference request-paired z/W delta
- `target-speed-trajectory.csv`: K×request speed, σ, κ, clamp, selection, finite-demand telemetry
- `terminal-z-w-nll.csv`: z/W rewrite/rephrase mean·median·p90·tail 및 success/accuracy/locality
- `writer-gap.csv`: pointwise W−z와 z-success/W-failure
- `compute.csv`: Slurm/runtime/field/writer/메모리 ledger
- `aggregates.json`, `analysis-manifest.json`, `source-receipt.json`: rooted factual package
