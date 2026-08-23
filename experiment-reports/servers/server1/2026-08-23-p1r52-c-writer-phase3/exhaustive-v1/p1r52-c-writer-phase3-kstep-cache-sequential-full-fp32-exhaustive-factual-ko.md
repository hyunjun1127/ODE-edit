# P1R52 C-writer Phase 3 FULL-FP32 — 최종 상세 사실 보고서

> **최우선 실행 경계:** Phase 3는 B1→B10 sequential W/Alpha-cache continuity를 유지하면서 각 B100의 K1→K8마다 writer를 적용한다. 비교 대상 Phase 1은 같은 stream/cache sequential 경계에서 accepted-z를 K8까지 완성한 뒤 B100당 writer를 한 번만 적용한다.

> 세 arm 모두 model/algorithm/update/cache가 FULL FP32다. BF16/FP16/autocast/quantization/numeric storage cast=0, retry=0, imputation=0, scientific promotion=false다.

## 1. 최종 endpoint edit 성능

|방법/endpoint|EFF|rewrite acc|GEN|strict GEN|rephrase acc|strict rephrase acc|LOC|
|---|---:|---:|---:|---:|---:|---:|---:|
|Pre-edit (공통 W0)|8.80% (88/1000)|0.10% (1/1000)|10.40% (208/2000)|6.00% (60/1000)|0.40% (8/2000)|0.00% (0/1000)|88.39% (8839/10000)|
|C0-KSTEP-CACHE final W10|100.00% (1000/1000)|99.60% (996/1000)|84.25% (1685/2000)|74.10% (741/1000)|51.60% (1032/2000)|33.20% (332/1000)|83.57% (8357/10000)|
|Phase1 C0 final W10 (one write/B100)|99.80% (998/1000)|99.40% (994/1000)|83.15% (1663/2000)|72.20% (722/1000)|50.60% (1012/2000)|32.00% (320/1000)|83.71% (8371/10000)|
|C1-KSTEP-CACHE final W10|100.00% (1000/1000)|99.40% (994/1000)|84.75% (1695/2000)|74.80% (748/1000)|52.40% (1048/2000)|34.70% (347/1000)|83.34% (8334/10000)|
|Phase1 C1 final W10 (one write/B100)|99.90% (999/1000)|99.60% (996/1000)|84.50% (1690/2000)|74.20% (742/1000)|52.85% (1057/2000)|34.10% (341/1000)|83.79% (8379/10000)|
|C3-KSTEP-CACHE final W10|99.90% (999/1000)|99.50% (995/1000)|85.30% (1706/2000)|75.40% (754/1000)|52.05% (1041/2000)|33.90% (339/1000)|84.33% (8433/10000)|
|Phase1 C3 final W10 (one write/B100)|99.80% (998/1000)|99.40% (994/1000)|83.40% (1668/2000)|73.10% (731/1000)|51.30% (1026/2000)|33.50% (335/1000)|85.08% (8508/10000)|

## 2. 이번 Phase에 추가된 방법 정의

|방법|target|writer|route|cache/실행 경계|
|---|---|---|---|---|
|C0-KSTEP-CACHE|P1R52 accepted-z refreshed at every K|PIR-U control remaining-residual L4→L8 prefix sequential writer|PIR-U control allocation; P/C router call 0|ALPHA_CACHE_CONTINUITY_ON_BATCH_ENTRY_SNAPSHOT; B1→B10 sequential W/cache continuity; final exact W0 restore|
|C1-KSTEP-CACHE|P1R52 accepted-z refreshed at every K|Joint P+C minimax + remaining-residual L4→L8 prefix sequential writer|current-K P/C proxy and Joint-PC route recomputed; router call 1/K|ALPHA_CACHE_CONTINUITY_ON_BATCH_ENTRY_SNAPSHOT; B1→B10 sequential W/cache continuity; final exact W0 restore|
|C3-KSTEP-CACHE|P1R52 accepted-z refreshed at every K|direct Official EasyEdit AlphaEdit writer with P1R52 accepted-z|no P/C route; native AlphaEdit compute_z call 0|ALPHA_CACHE_CONTINUITY_ON_BATCH_ENTRY_SNAPSHOT; B1→B10 sequential W/cache continuity; final exact W0 restore|

## 3. Rewrite NLL

|방법/endpoint|rewrite target-new mean/median/p90/max|rewrite target-true mean/median/p90/max|
|---|---:|---:|
|Pre-edit (공통 W0)|11.040824/10.937304/16.200504/22.937422|4.591916/4.241698/9.646519/16.172873|
|C0-KSTEP-CACHE final W10|0.056816/0.017079/0.068917/9.133502|11.105907/10.733105/16.359140/28.336485|
|Phase1 C0 final W10 (one write/B100)|0.064041/0.015753/0.083150/6.199005|11.020352/10.479712/16.471176/24.588669|
|C1-KSTEP-CACHE final W10|0.061225/0.014526/0.054685/8.847200|11.261795/10.859377/16.430239/24.843119|
|Phase1 C1 final W10 (one write/B100)|0.049002/0.011052/0.050957/6.843416|11.469277/10.969878/16.915487/24.459251|
|C3-KSTEP-CACHE final W10|0.062325/0.016337/0.063799/11.388570|11.101294/10.713209/16.084743/22.967220|
|Phase1 C3 final W10 (one write/B100)|0.059180/0.018388/0.072232/6.017864|10.877238/10.516838/16.165775/23.483862|

## 4. Rephrase NLL

|방법/endpoint|rephrase target-new mean/median/p90/max|rephrase target-true mean/median/p90/max|
|---|---:|---:|
|Pre-edit (공통 W0)|10.081088/10.005895/14.581030/20.918356|4.700030/4.326975/9.530659/20.848831|
|C0-KSTEP-CACHE final W10|2.486663/1.230068/6.999107/17.494427|7.297526/6.907595/12.486785/22.757616|
|Phase1 C0 final W10 (one write/B100)|2.534352/1.299065/7.066687/17.685158|7.286662/6.855746/12.664206/22.995731|
|C1-KSTEP-CACHE final W10|2.413860/1.192088/6.742559/17.684713|7.419104/6.958610/12.679899/22.777285|
|Phase1 C1 final W10 (one write/B100)|2.424955/1.144244/6.852288/18.007715|7.463704/7.106835/12.992193/22.623468|
|C3-KSTEP-CACHE final W10|2.393893/1.224889/6.631929/17.783285|7.279232/6.887279/12.318640/23.155109|
|Phase1 C3 final W10 (one write/B100)|2.511754/1.275448/6.865308/17.589714|7.262072/6.902588/12.606893/23.327879|

## 5. Paired 기준 비교

|arm|EFF Δpp|GEN Δpp|strict GEN Δpp|LOC Δpp|rewrite new NLL Δ|rephrase new NLL Δ|
|---|---:|---:|---:|---:|---:|---:|
|c0|+0.200|+1.100|+1.900|-0.140|-0.007225|-0.047689|
|c1|+0.100|+0.250|+0.600|-0.450|+0.012223|-0.011095|
|c3|+0.100|+1.900|+2.300|-0.750|+0.003145|-0.117861|

|arm|paired case total Δ mean sec|paired case total ratio mean|target/writer timing comparison|update energy Δ mean|
|---|---:|---:|---|---:|
|c0|+2195.102336|2.918602|NOT_RECORDED_PER_BATCH|-5.744756|
|c1|+2214.499801|3.040724|NOT_RECORDED_PER_BATCH|-5.592533|
|c3|+2205.705143|2.938410|NOT_RECORDED_PER_BATCH|NOT_RECORDED|

모든 비교는 동일한 10/10 batch denominator와 sequential stream/order를 사용하며, Phase 3와 Phase 1의 최종 W10 endpoint를 paired 비교한다.

## 6. K1→K8 trajectory와 W−z gap

`per-k-endpoint-and-compute.*`, `per-unit-endpoint-summary.*`, `per-prompt-kstep-nll.csv`에 240개 K receipt와 모든 accepted-z/pre-W/post-W prompt NLL을 보존했다. 각 K에서 post-W가 다음 K의 entry-W와 SHA로 연결됨을 fail-close 검증했다.

`cross-arm-target-identity.*` 기준 80개 unit×K 중 세 arm accepted-z/selected-target가 모두 같은 지점은 1개다. 최초 writer-feedback divergence는 unit 1 K2이다. 이 표는 차이를 결함으로 간주하지 않고, 공통 entry에서 시작한 뒤 writer별 W trajectory가 target refresh를 바꾸는 시점을 기록한다.

|arm|K8 rewrite W−z mean|K8 rephrase W−z mean|K8 z→W rewrite loss|K8 z→W rephrase loss/strict|K8 update energy mean|K8 residual mean|
|---|---:|---:|---:|---:|---:|---:|
|c0|-0.000148|1.204710|0|277/226|0.030616636289913436|0.030110215636171334|
|c1|0.000259|1.180453|0|259/213|0.11990635083538878|0.036926642544264296|
|c3|0.000378|1.058148|0|223/190|NOT_RECORDED|NOT_RECORDED|

C3의 Official native apply receipt에는 C0/C1과 동형인 layer별 update norm/energy가 기록되지 않아 해당 400개 layer placeholder는 `NOT_RECORDED_NATIVE_OFFICIAL_APPLY`로 명시했다. 시간이나 다른 aggregate에서 이를 역추정하지 않았다.

## 7. P/C route와 writer layer telemetry

C0는 PIR-U control route라 P/C router call=0, C1은 current-K Joint-PC solver call=1/K, C3는 P/C router call=0이다. `per-k-route.*`는 C1의 selected π/P/C/minimax/KKT telemetry와 C0 control allocation, C3 direct-writer status를 분리한다. `per-layer-writer-update-energy.*`는 C0/C1의 L4→L8 residual/q/update/rounding observation을 기록한다.

FP32 actual delta는 `fl32(W+U)-W`이므로 prepared U와 byte-identical일 필요가 없다. rounding mismatch는 observation-only이고 route decision influence=0이다. Official assignment endpoint 자체는 reference expression과 byte-exact다.

## 8. 계산량·시간

|arm|unit total sum/mean/median/p90/max sec|recorded phase-component wall sum|K-step writer segment sum|job sec|eval accepted/pre/post|dense/apply/commit|router|peak alloc/reserved GiB|MaxRSS GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|c0|33398.50/3339.85/3362.36/3458.65/3459.39|NOT_RECORDED|NOT_RECORDED|34137.22|80/80/80|400/400/80|0|34.16/38.11|14.95|
|c1|32999.94/3299.99/3353.01/3389.88/3404.05|NOT_RECORDED|NOT_RECORDED|33738.81|80/80/80|400/400/80|80|34.16/38.11|14.82|
|c3|33479.18/3347.92/3353.43/3433.87/3464.07|NOT_RECORDED|NOT_RECORDED|34286.29|80/80/80|400/400/80|0|35.42/40.99|41.47|

Phase 3 raw batch receipts에는 Phase 2와 동일한 per-batch target phase wall decomposition이 직렬화되지 않았다. 따라서 Phase 3의 per-batch target/writer 세부 wall은 `NOT_RECORDED_PER_BATCH`이고, recorded unit/job total과 nested logical counts만 보고한다. wall time만으로 FLOPs나 원인을 역추정하지 않는다.

## 9. Alpha-cache sequential 감사

B1→B10 entry width는 arm마다 0,100,…,900이고 각 B에서 동일 batch-entry snapshot을 K1→K8 8회 재사용한다. current batch/prefix key inclusion=0, B 성공 뒤 append=1, 전체 append=10, rollback=0이다. Structural-H history와 Alpha cache는 분리되어 있다.

## 10. 사실 판정

이 보고서는 efficacy/generalization/locality/NLL/update/route/cache/compute의 관측값과 paired 차이를 제공한다. 새 threshold나 자동 promotion을 추가하지 않았으며 `scientific_promotion=false`다. 성능 차이는 해당 실행 경계의 사실 비교이지 단일 구성요소에 대한 인과 주장으로 해석하지 않는다.

## 11. 재현 산출물

- `method-definitions.*`: 추가 method와 실행 경계
- `terminal-endpoint-aggregate.*`: pre-edit/current/reference endpoint 종합
- `per-unit-endpoint-summary.*`: unit×K×endpoint summary
- `per-prompt-kstep-nll.csv`: 모든 prompt NLL/bit
- `nll-distribution-aggregates.*`: K/endpoint별 mean/median/p90/max/min
- `per-k-route.*`, `per-layer-writer-update-energy.*`: route/layer telemetry
- `per-unit-compute.*`, `compute-aggregate.*`: 계산량·시간·메모리
- `paired-unit-comparison.*`, `paired-comparison-aggregate.*`: 10/10 paired 비교
- `cross-arm-target-identity.*`: accepted-z/selected-target 동일성과 최초 divergence
- `artifact-inventory.*`, `analysis-manifest.json`, `rooted-analysis-receipt.json`: 재해시 결속
- `cache-sequential-audit.*`: B1→B10 cache entry/consume/append/rollback
