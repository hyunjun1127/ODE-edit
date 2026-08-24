# P1R54 FZ-C3 Independent 10×B100 — Llama FULL-FP32 native baseline 포함 사실 보고서

> **해석 경계:** 동일 sealed B1–B10×B100, 1,000 requests를 사용한다. Independent는 각 slice가 W0/cold cache에서 독립 시작하고, Sequential은 B1→B10 W 및 Alpha-cache 연속성을 유지한다. 두 실행 의미를 섞은 직접 paired 인과 비교는 하지 않는다. `scientific_promotion=false`.

## 1. 최종 한눈표

|method|final W Eff|final W Gen|strict Gen|final W Loc|z rewrite NLL mean/median/p90/max|z rephrase NLL mean/median/p90/max|post-W rewrite NLL mean/median/p90/max|post-W rephrase NLL mean/median/p90/max|
|---|---|---|---|---|---|---|---|---|
|Official AlphaEdit|99.90% (999/1000)|93.00% (1860/2000)|88.00% (880/1000)|86.26% (8626/10000)|0.002307 / 0.000690 / 0.002685 / 0.998165|1.100522 / 0.051957 / 4.308492 / 15.602556|0.003315 / 0.000747 / 0.002938 / 1.877316|1.848039 / 0.293467 / 5.901396 / 17.445372|
|Native MEMIT|98.30% (983/1000)|82.55% (1651/2000)|74.50% (745/1000)|87.89% (8789/10000)|0.001098 / 0.000645 / 0.002308 / 0.025492|1.104760 / 0.053958 / 4.294096 / 15.612510|0.196725 / 0.004256 / 0.109747 / 12.664020|2.875626 / 1.155112 / 8.353469 / 20.605066|
|P1R54 FZ-INDEPENDENT|99.80% (998/1000)|92.10% (1842/2000)|86.70% (867/1000)|87.47% (8747/10000)|0.036367 / 0.001228 / 0.020374 / 3.373269|1.290592 / 0.097049 / 4.662909 / 16.324144|0.037832 / 0.001213 / 0.020455 / 3.746590|1.965954 / 0.475650 / 6.100472 / 17.943241|

## 2. Rewrite 세부

|method|endpoint|success|accuracy|strict success|strict accuracy|target-new mean/median/p90/max|target-true mean/median/p90/max|
|---|---|---|---|---|---|---|---|
|Official AlphaEdit|accepted_z|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|0.002307 / 0.000690 / 0.002685 / 0.998165|14.628757 / 14.350395 / 19.818745 / 27.315966|
|Official AlphaEdit|post_W|99.90% (999/1000)|99.90% (999/1000)|99.90% (999/1000)|99.90% (999/1000)|0.003315 / 0.000747 / 0.002938 / 1.877316|14.510020 / 14.228350 / 19.696653 / 27.063623|
|Native MEMIT|accepted_z|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|0.001098 / 0.000645 / 0.002308 / 0.025492|14.820167 / 14.542142 / 19.804352 / 27.454245|
|Native MEMIT|post_W|98.30% (983/1000)|96.30% (963/1000)|98.30% (983/1000)|96.30% (963/1000)|0.196725 / 0.004256 / 0.109747 / 12.664020|11.534598 / 11.359380 / 17.133034 / 25.438904|
|P1R54 FZ-INDEPENDENT|accepted_z|99.90% (999/1000)|99.00% (990/1000)|99.90% (999/1000)|99.00% (990/1000)|0.036367 / 0.001228 / 0.020374 / 3.373269|14.056989 / 13.880352 / 19.066046 / 26.886267|
|P1R54 FZ-INDEPENDENT|post_W|99.80% (998/1000)|99.00% (990/1000)|99.80% (998/1000)|99.00% (990/1000)|0.037832 / 0.001213 / 0.020455 / 3.746590|14.042333 / 13.905948 / 19.069656 / 26.871216|

## 3. Rephrase 세부

|method|endpoint|success|accuracy|strict success|strict accuracy|target-new mean/median/p90/max|target-true mean/median/p90/max|
|---|---|---|---|---|---|---|---|
|Official AlphaEdit|accepted_z|98.70% (1974/2000)|76.25% (1525/2000)|97.60% (976/1000)|62.50% (625/1000)|1.100522 / 0.051957 / 4.308492 / 15.602556|11.050089 / 10.976290 / 16.343199 / 26.353649|
|Official AlphaEdit|post_W|93.00% (1860/2000)|64.60% (1292/2000)|88.00% (880/1000)|48.80% (488/1000)|1.848039 / 0.293467 / 5.901396 / 17.445372|9.396850 / 9.368317 / 14.641417 / 24.311455|
|Native MEMIT|accepted_z|98.60% (1972/2000)|76.05% (1521/2000)|97.40% (974/1000)|62.40% (624/1000)|1.104760 / 0.053958 / 4.294096 / 15.612510|11.066664 / 11.003042 / 16.426399 / 26.519575|
|Native MEMIT|post_W|82.55% (1651/2000)|51.05% (1021/2000)|74.50% (745/1000)|34.70% (347/1000)|2.875626 / 1.155112 / 8.353469 / 20.605066|7.794277 / 7.668746 / 13.102352 / 23.182947|
|P1R54 FZ-INDEPENDENT|accepted_z|97.75% (1955/2000)|72.55% (1451/2000)|96.50% (965/1000)|58.60% (586/1000)|1.290592 / 0.097049 / 4.662909 / 16.324144|10.833720 / 10.720064 / 16.003357 / 24.784622|
|P1R54 FZ-INDEPENDENT|post_W|92.10% (1842/2000)|61.75% (1235/2000)|86.70% (867/1000)|45.50% (455/1000)|1.965954 / 0.475650 / 6.100472 / 17.943241|9.027939 / 8.917210 / 14.186677 / 25.486071|

## 4. FZ 고유 실행 진단

- target field: 80회; writer: 80회; layer apply: 400회.
- clamp: 0/8000 (0.000000).
- forbidden decision access=0, additional F/B=0.
- writer actual per-layer update energy/share는 Official apply receipt에 저장되지 않아 `NOT_RECORDED`; cache layer Frobenius load만 별도 JSON/CSV에 유지한다.

|K|clamp|top1 energy share mean|top10 mean|bottom90 mean|allocation energy mean|
|---:|---:|---:|---:|---:|---:|
|1|0/1000|0.014321|0.129964|0.870036|3672.411877|
|2|0/1000|0.014321|0.129964|0.870036|3672.411877|
|3|0/1000|0.014321|0.129964|0.870036|3672.411877|
|4|0/1000|0.014321|0.129964|0.870036|3672.411877|
|5|0/1000|0.014321|0.129964|0.870036|3672.411877|
|6|0/1000|0.014321|0.129964|0.870036|3672.411877|
|7|0/1000|0.014321|0.129964|0.870036|3672.411877|
|8|0/1000|0.014321|0.129964|0.870036|3672.411877|

## 5. Independent 고유 표

각 B1–B10은 동일 W0/cold Alpha-cache width0에서 시작했고, cell-local append는 다른 cell로 전달되지 않았다. W0 restore=10/10, cross-batch state consumption=0.

|B|W rewrite NLL mean|W rephrase NLL mean|Eff|Gen|Loc|
|---:|---:|---:|---:|---:|---:|
|1|0.050604|1.983023|100.00% (100/100)|91.50% (183/200)|89.00% (890/1000)|
|2|0.058861|1.872667|100.00% (100/100)|94.00% (188/200)|91.70% (917/1000)|
|3|0.068267|2.028625|99.00% (99/100)|89.50% (179/200)|88.50% (885/1000)|
|4|0.036780|1.859684|100.00% (100/100)|92.00% (184/200)|84.20% (842/1000)|
|5|0.008586|2.114561|100.00% (100/100)|90.50% (181/200)|86.90% (869/1000)|
|6|0.049087|2.074488|100.00% (100/100)|94.00% (188/200)|84.80% (848/1000)|
|7|0.013438|1.897982|100.00% (100/100)|92.50% (185/200)|86.80% (868/1000)|
|8|0.017980|2.143030|100.00% (100/100)|91.50% (183/200)|84.10% (841/1000)|
|9|0.018097|1.748844|100.00% (100/100)|93.00% (186/200)|88.90% (889/1000)|
|10|0.056623|1.936637|99.00% (99/100)|92.50% (185/200)|89.80% (898/1000)|

### Exact native baseline 대비 B별 paired NLL delta (FZ−native; 음수일수록 FZ NLL이 낮음)

|baseline|B|z rewrite Δ|z rephrase Δ|W rewrite Δ|W rephrase Δ|
|---|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|1|+0.047118|+0.175902|+0.049207|+0.062195|
|OFFICIAL-ALPHAEDIT|2|+0.055406|+0.141416|+0.057191|-0.075576|
|OFFICIAL-ALPHAEDIT|3|+0.062393|+0.197668|+0.066486|-0.011735|
|OFFICIAL-ALPHAEDIT|4|+0.034608|+0.384747|+0.035729|+0.386541|
|OFFICIAL-ALPHAEDIT|5|+0.006870|+0.150683|+0.007388|+0.291654|
|OFFICIAL-ALPHAEDIT|6|+0.045691|+0.086943|+0.047715|-0.055668|
|OFFICIAL-ALPHAEDIT|7|+0.012511|+0.249685|+0.012090|+0.029936|
|OFFICIAL-ALPHAEDIT|8|+0.016597|+0.119959|+0.016407|+0.133953|
|OFFICIAL-ALPHAEDIT|9|+0.017536|+0.243831|+0.016858|+0.251766|
|OFFICIAL-ALPHAEDIT|10|+0.041879|+0.149866|+0.036100|+0.166084|
|OFFICIAL-MEMIT|1|+0.047403|+0.182813|-0.296251|-0.992591|
|OFFICIAL-MEMIT|2|+0.055659|+0.139063|-0.167133|-1.084705|
|OFFICIAL-MEMIT|3|+0.062818|+0.190797|-0.350491|-1.074416|
|OFFICIAL-MEMIT|4|+0.034676|+0.397456|-0.123401|-0.571480|
|OFFICIAL-MEMIT|5|+0.006990|+0.135152|-0.086764|-0.491090|
|OFFICIAL-MEMIT|6|+0.045951|+0.098724|-0.125715|-1.241962|
|OFFICIAL-MEMIT|7|+0.012672|+0.249142|-0.045988|-1.081068|
|OFFICIAL-MEMIT|8|+0.016777|+0.114311|-0.077994|-0.674343|
|OFFICIAL-MEMIT|9|+0.017585|+0.214870|-0.233333|-0.848920|
|OFFICIAL-MEMIT|10|+0.052159|+0.135987|-0.081858|-1.036139|

## 6. z→W gap 및 writer/cache load

|prompt/target|W−z NLL mean/median/p90/max|
|---|---:|
|rewrite_post_W_minus_accepted_z_nll|0.001465 / 0.000001 / 0.000304 / 0.419176|
|rephrase_post_W_minus_accepted_z_nll|0.675362 / 0.066883 / 2.387984 / 12.406072|

|layer|Alpha-cache load mean|max|mean share|actual writer update energy|
|---:|---:|---:|---:|---|
|L4|98.179415|104.955459|0.101679|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L5|152.566256|162.296235|0.157997|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L6|222.040694|238.977430|0.229911|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L7|258.106105|275.877241|0.267266|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L8|234.788337|248.651786|0.243146|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|

## 7. Native baseline 및 compute/overhead

- baseline source `27a5ea828af26e30887b9546b31f7f9584ef1e7c`, group `job22541`.
- baseline report/manifest/receipt SHA: `3ea9739fa9200fd2e9fbf36a7a6b9e86c0212279fc52f9c6b2c0aa6f5f42e217` / `b38d8b23d7d2876e72dcab35c8a64083ba6baa5951769998cf3c1726d9a77d09` / `e82fb1dd0f8c1596823979b63f66168682a15bf231621771dae95340e68c8356`; receipt identity `ea5261b1f4b08732fa7ae08895dbe655d53a630d3c4f5b7bb86a5a74b644ff60`.
- Official AlphaEdit와 MEMIT의 z는 각각 native compute_z/latent-z이며 FZ accepted-z와 같은 알고리즘으로 해석하지 않는다.
- setup/model-load/evaluator 포함 wall time과 edit-core time을 혼합하지 않는다. 없는 count/timer는 역추정하지 않고 `NOT_RECORDED`로 둔다.

|method|timer scope|total/sum s|case/batch mean s|target mean s|writer mean s|field eval|writer calls|layer apply|FZ/native ratio|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|P1R54-FZ|SUM_OF_10_INDEPENDENT_CELLS_POST_MODEL_PREFLIGHT|10055.563|1005.556|NOT_SEPARATELY_RECORDED|45.588|80|80|400|1.000|
|OFFICIAL-ALPHAEDIT|SUM_OF_10_INDEPENDENT_CASES_POST_MODEL_PREFLIGHT|9235.322|923.532|631.247|164.032|NATIVE_COMPUTE_Z_NOT_FZ_FIELD|10|50|1.089|
|OFFICIAL-MEMIT|SUM_OF_10_INDEPENDENT_CASES_POST_MODEL_PREFLIGHT|8671.598|867.160|631.449|107.688|NATIVE_COMPUTE_Z_NOT_FZ_FIELD|10|50|1.160|

## 8. 판정

- 결과는 Llama 단일 stream의 factual possibility evidence다. 새 hard threshold, imputation, baseline rerun, automatic promotion은 0이다.
- Independent와 Sequential 차이는 cross-batch W/cache continuity가 함께 달라지는 descriptive comparison이며 causal claim은 하지 않는다.
- P1R52 Phase2/Phase3 C3는 필요 시 same-sample non-native accepted-z 보조 reference일 뿐 native baseline이 아니다.

## 9. 재현성

- stream/order/ordered-record root: `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a` / `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3` / `af215235177d1ac07e82fadc62de7235244a82342a146c5867f673625486d0e1`.
- FZ source HEAD: `97e0cbb35568674cae83e3ab4e99b963f5116eda`; canonical Slurm job: `23787_[0-9] TECH-R2`.
- excluded technical attempts: `[{"class": "PURE_TECHNICAL_PRE_MODEL", "job": "23768", "reason": "local session config missing", "scientific_denominator_influence_count": 0}, {"class": "PURE_TECHNICAL_PRE_MODEL", "job": "23777", "reason": "logs parent missing", "scientific_denominator_influence_count": 0}]`.
- FZ analysis identity: `953f7bb08b47d63e81a82c272819a59c0910248321398e5247a0d76605e4e595`.
- raw results는 immutable이며 report 생성 중 mutation count=0. 모든 small member와 external input SHA는 analysis manifest/rooted receipt에 결속한다.
