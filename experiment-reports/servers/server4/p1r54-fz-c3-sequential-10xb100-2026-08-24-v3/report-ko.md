# P1R54 FZ-C3 Sequential 10×B100 — Llama FULL-FP32 native baseline 포함 사실 보고서

> **해석 경계:** 동일 sealed B1–B10×B100, 1,000 requests를 사용한다. Independent는 각 slice가 W0/cold cache에서 독립 시작하고, Sequential은 B1→B10 W 및 Alpha-cache 연속성을 유지한다. 두 실행 의미를 섞은 직접 paired 인과 비교는 하지 않는다. `scientific_promotion=false`.

## 1. 최종 한눈표

|method|final W Eff|final W Gen|strict Gen|final W Loc|z rewrite NLL mean/median/p90/max|z rephrase NLL mean/median/p90/max|post-W rewrite NLL mean/median/p90/max|post-W rephrase NLL mean/median/p90/max|
|---|---|---|---|---|---|---|---|---|
|Official AlphaEdit-cache|99.40% (994/1000)|96.70% (1934/2000)|94.30% (943/1000)|78.10% (7810/10000)|0.001392 / 0.000743 / 0.003357 / 0.025194|0.797691 / 0.069259 / 2.721582 / 12.862535|0.001493 / 0.000804 / 0.003582 / 0.026080|1.159276 / 0.290779 / 3.386415 / 13.746431|
|Native MEMIT|95.40% (954/1000)|90.35% (1807/2000)|86.10% (861/1000)|72.39% (7239/10000)|0.002097 / 0.000903 / 0.004589 / 0.043856|0.771799 / 0.081747 / 2.646872 / 12.733783|0.076587 / 0.003204 / 0.022911 / 12.664020|1.678150 / 0.603241 / 4.716975 / 18.988174|
|P1R54 FZ-SEQUENTIAL|100.00% (1000/1000)|84.95% (1699/2000)|75.60% (756/1000)|84.53% (8453/10000)|0.008817 / 0.000617 / 0.004828 / 2.775532|1.265033 / 0.065512 / 4.660754 / 21.401110|0.008849 / 0.000618 / 0.004897 / 2.764586|2.343741 / 0.628484 / 7.347599 / 18.269323|

Sequential 표의 Eff/Gen/Loc는 **최종 W10 전체 1,000-request 재평가**다. z와 post-W NLL은 각 batch의 K8 accepted-z 및 immediate post-W pooled 분포다. Pinned native package에는 final-W10 NLL 분포가 없어 이를 immediate post-W 값으로 대체하지 않고 분리 표기한다.

## 2. Rewrite 세부

|method|endpoint|success|accuracy|strict success|strict accuracy|target-new mean/median/p90/max|target-true mean/median/p90/max|
|---|---|---|---|---|---|---|---|
|Official AlphaEdit-cache|accepted_z|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|0.001392 / 0.000743 / 0.003357 / 0.025194|15.196513 / 14.896466 / 20.951120 / 31.449064|
|Official AlphaEdit-cache|post_W|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|0.001493 / 0.000804 / 0.003582 / 0.026080|15.100299 / 14.800235 / 20.819164 / 31.112593|
|Native MEMIT|accepted_z|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|100.00% (1000/1000)|0.002097 / 0.000903 / 0.004589 / 0.043856|14.776316 / 14.372349 / 20.241045 / 27.454245|
|Native MEMIT|post_W|99.60% (996/1000)|98.90% (989/1000)|99.60% (996/1000)|98.90% (989/1000)|0.076587 / 0.003204 / 0.022911 / 12.664020|12.895130 / 12.687969 / 18.405182 / 24.849831|
|P1R54 FZ-SEQUENTIAL|accepted_z|100.00% (1000/1000)|99.80% (998/1000)|100.00% (1000/1000)|99.80% (998/1000)|0.008817 / 0.000617 / 0.004828 / 2.775532|14.519902 / 14.051021 / 19.683716 / 28.370306|
|P1R54 FZ-SEQUENTIAL|post_W|100.00% (1000/1000)|99.80% (998/1000)|100.00% (1000/1000)|99.80% (998/1000)|0.008849 / 0.000618 / 0.004897 / 2.764586|14.514567 / 14.001203 / 19.708973 / 28.348249|

## 3. Rephrase 세부

|method|endpoint|success|accuracy|strict success|strict accuracy|target-new mean/median/p90/max|target-true mean/median/p90/max|
|---|---|---|---|---|---|---|---|
|Official AlphaEdit-cache|accepted_z|99.35% (1987/2000)|81.55% (1631/2000)|98.80% (988/1000)|69.40% (694/1000)|0.797691 / 0.069259 / 2.721582 / 12.862535|12.183187 / 11.778222 / 17.397113 / 24.276031|
|Official AlphaEdit-cache|post_W|97.40% (1948/2000)|75.55% (1511/2000)|95.50% (955/1000)|61.30% (613/1000)|1.159276 / 0.290779 / 3.386415 / 13.746431|10.758104 / 10.754892 / 15.675660 / 23.082351|
|Native MEMIT|accepted_z|99.00% (1980/2000)|81.30% (1626/2000)|98.10% (981/1000)|68.90% (689/1000)|0.771799 / 0.081747 / 2.646872 / 12.733783|11.586472 / 11.335657 / 16.863792 / 24.677603|
|Native MEMIT|post_W|92.05% (1841/2000)|67.20% (1344/2000)|87.90% (879/1000)|52.80% (528/1000)|1.678150 / 0.603241 / 4.716975 / 18.988174|9.286820 / 9.209508 / 14.490139 / 21.984386|
|P1R54 FZ-SEQUENTIAL|accepted_z|96.65% (1933/2000)|73.55% (1471/2000)|94.10% (941/1000)|59.10% (591/1000)|1.265033 / 0.065512 / 4.660754 / 21.401110|10.604270 / 10.390417 / 16.281460 / 27.655798|
|P1R54 FZ-SEQUENTIAL|post_W|85.15% (1703/2000)|57.40% (1148/2000)|76.10% (761/1000)|40.10% (401/1000)|2.343741 / 0.628484 / 7.347599 / 18.269323|8.485302 / 8.242339 / 14.347278 / 25.353165|

## 4. FZ 고유 실행 진단

- target field: 80회; writer: 80회; layer apply: 400회.
- clamp: 0/8000 (0.000000).
- forbidden decision access=0, additional F/B=0.
- writer actual per-layer update energy/share는 Official apply receipt에 저장되지 않아 `NOT_RECORDED`; cache layer Frobenius load만 별도 JSON/CSV에 유지한다.

|K|clamp|top1 energy share mean|top10 mean|bottom90 mean|allocation energy mean|
|---:|---:|---:|---:|---:|---:|
|1|0/1000|0.014564|0.132375|0.867625|4083.378066|
|2|0/1000|0.014564|0.132375|0.867625|4083.378066|
|3|0/1000|0.014564|0.132375|0.867625|4083.378066|
|4|0/1000|0.014564|0.132375|0.867625|4083.378066|
|5|0/1000|0.014564|0.132375|0.867625|4083.378066|
|6|0/1000|0.014564|0.132375|0.867625|4083.378066|
|7|0/1000|0.014564|0.132375|0.867625|4083.378066|
|8|0/1000|0.014564|0.132375|0.867625|4083.378066|

## 5. Sequential 고유 표 — immediate→final W10 forgetting

|B|final−immediate rewrite target-new NLL mean|final−immediate rephrase target-new NLL mean|rewrite success→failure|rephrase success→failure|
|---:|---:|---:|---:|---:|
|1|-0.000532|-0.092413|0|5|
|2|0.003224|-0.031101|0|3|
|3|0.002489|-0.089692|0|6|
|4|0.092616|0.079633|0|2|
|5|0.000529|-0.080847|0|3|
|6|0.000092|-0.030154|0|1|
|7|-0.000246|-0.067577|0|2|
|8|-0.000369|-0.033260|0|0|
|9|-0.000023|-0.031693|0|0|
|10|0.000000|0.000000|0|0|

Cache entry widths는 0→900, exit widths는 100→1000이며 성공 batch당 append1, 총 append10/consume80이다. B_r commit hash와 B_(r+1) entry hash는 9/9 exact 일치했고 종료 후 W0 bytes를 복원했다.

### 최종 W10 NLL 분포

|method|prompt|target-new mean/median/p90/max|target-true mean/median/p90/max|
|---|---|---:|---:|
|Official AlphaEdit-cache|rewrite/rephrase|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|
|Native MEMIT|rewrite/rephrase|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|
|P1R54 FZ-SEQUENTIAL|rewrite|0.018627 / 0.000580 / 0.005625 / 8.714952|14.517836 / 14.114930 / 20.189289 / 30.168802|
|P1R54 FZ-SEQUENTIAL|rephrase|2.306031 / 0.594127 / 7.233589 / 17.947861|8.516335 / 8.230486 / 14.528118 / 25.379133|

### 동일 FZ Independent 대비 descriptive delta (Sequential−Independent)

|B|z rewrite Δ|z rephrase Δ|immediate W rewrite Δ|immediate W rephrase Δ|
|---:|---:|---:|---:|---:|
|1|+0.000000|+0.000000|+0.000000|+0.000000|
|2|-0.052470|+0.017803|-0.054345|+0.281697|
|3|-0.058639|+0.024226|-0.063197|+0.421753|
|4|-0.032662|+0.077217|-0.033898|+0.381179|
|5|-0.006163|-0.085356|-0.006497|+0.318598|
|6|-0.045821|-0.096579|-0.048006|+0.630104|
|7|-0.011748|+0.019130|-0.011404|+0.489772|
|8|-0.013059|-0.175259|-0.012682|+0.192773|
|9|-0.003560|-0.106977|-0.005302|+0.260161|
|10|-0.051382|+0.070205|-0.054500|+0.801835|

|final endpoint rate|Sequential final W10 − Independent endpoint|
|---|---:|
|Eff|+0.200 percentage points|
|Gen|-7.150 percentage points|
|Gen_strict|-11.100 percentage points|
|Loc|-2.940 percentage points|

## 6. z→W gap 및 writer/cache load

|prompt/target|W−z NLL mean/median/p90/max|
|---|---:|
|rewrite_post_W_minus_accepted_z_target_new_nll|0.000032 / 0.000000 / 0.000054 / 0.263305|
|rewrite_post_W_minus_accepted_z_target_true_nll|-0.005335 / -0.001300 / 0.070772 / 0.622172|
|rephrase_post_W_minus_accepted_z_target_new_nll|1.078709 / 0.113517 / 4.138383 / 15.904956|
|rephrase_post_W_minus_accepted_z_target_true_nll|-2.118968 / -1.443852 / 1.068381 / 12.324736|

|layer|Alpha-cache load mean|max|mean share|actual writer update energy|
|---:|---:|---:|---:|---|
|L4|356.699184|601.384129|0.082145|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L5|600.675924|1041.576885|0.135341|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L6|1015.230116|1808.912749|0.222564|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L7|1351.196514|2457.749813|0.289920|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|
|L8|1268.609348|2345.250704|0.270029|NOT_RECORDED_NATIVE_OFFICIAL_APPLY|

## 7. Native baseline 및 compute/overhead

- baseline source `251e616cf95972e230ce40718e1e390ef7dc9eb4`, group `job22759_PHASE1_GROUP`.
- baseline report/manifest/receipt SHA: `b93391e5c344bd5635fdd1c2f6532d41e683a88d85dabb3be21e949f05f3891b` / `58ee504c70e2a3a5708a937a44f1d6cdb736240e38745e021d4eb1244ba97a70` / `6d269759f9cc8b458e9589c1ca64fcd5362681ec5c6cf455fe6376cf4e0a1f66`; receipt identity `fb9b1e9709ec8c86319e8f88a36a9d99726fd4a22177f5cf8c2cb2607347812e`.
- Official AlphaEdit와 MEMIT의 z는 각각 native compute_z/latent-z이며 FZ accepted-z와 같은 알고리즘으로 해석하지 않는다.
- setup/model-load/evaluator 포함 wall time과 edit-core time을 혼합하지 않는다. 없는 count/timer는 역추정하지 않고 `NOT_RECORDED`로 둔다.

|method|timer scope|total/sum s|case/batch mean s|target mean s|writer mean s|field eval|writer calls|layer apply|FZ/native ratio|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|P1R54-FZ|ONE_SEQUENTIAL_JOB_POST_MODEL_PREFLIGHT|11918.267|1151.218|NOT_SEPARATELY_RECORDED|45.627|80|80|400|1.000|
|OFFICIAL-ALPHAEDIT-CACHE|JOB_TOTAL_INCLUDES_SETUP_MODEL_LOAD_EVALUATOR|14890.526|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED_IN_PINNED_V5_PACKAGE|NOT_RECORDED_IN_PINNED_V5_PACKAGE|NOT_RECORDED_IN_PINNED_V5_PACKAGE|NOT_COMPARABLE_TIMER_SCOPE|
|OFFICIAL-MEMIT|JOB_TOTAL_INCLUDES_SETUP_MODEL_LOAD_EVALUATOR|13845.522|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED_IN_PINNED_V5_PACKAGE|NOT_RECORDED_IN_PINNED_V5_PACKAGE|NOT_RECORDED_IN_PINNED_V5_PACKAGE|NOT_COMPARABLE_TIMER_SCOPE|

## 8. 판정

- 결과는 Llama 단일 stream의 factual possibility evidence다. 새 hard threshold, imputation, baseline rerun, automatic promotion은 0이다.
- Independent와 Sequential 차이는 cross-batch W/cache continuity가 함께 달라지는 descriptive comparison이며 causal claim은 하지 않는다.
- P1R52 Phase2/Phase3 C3는 필요 시 same-sample non-native accepted-z 보조 reference일 뿐 native baseline이 아니다.

## 9. 재현성

- stream/order/ordered-record root: `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a` / `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3` / `af215235177d1ac07e82fadc62de7235244a82342a146c5867f673625486d0e1`.
- FZ source HEAD: `6b7364f4caf3b3fbead83a3f5bcb8c39a79b38f5`; canonical Slurm job: `23682`.
- excluded technical attempts: `[]`.
- FZ analysis identity: `de1aec92eb3e736aa508df23f6c0b3b41c0de365ed0df8871daba13aa7acee33`.
- raw results는 immutable이며 report 생성 중 mutation count=0. 모든 small member와 external input SHA는 analysis manifest/rooted receipt에 결속한다.
