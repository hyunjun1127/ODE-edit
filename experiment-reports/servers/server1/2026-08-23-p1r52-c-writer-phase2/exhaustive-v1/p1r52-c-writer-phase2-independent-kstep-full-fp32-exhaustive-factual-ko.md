# P1R52 C-writer Phase 2 FULL-FP32 — 최종 상세 사실 보고서

> **최우선 실행 경계:** Phase 2는 각 independent B100 안에서 K1→K8마다 `W_(k-1) → accepted-z_k → writer_k → W_k`를 수행한다. 비교 대상 final-v6는 동일 W0/slice/order에서 accepted-z를 K8까지 완성한 뒤 writer를 한 번 적용한다. 따라서 same-z equivalence는 주장하지 않는다.

> 세 arm 모두 model/algorithm/update/cache가 FULL FP32다. BF16/FP16/autocast/quantization/numeric storage cast=0, retry=0, imputation=0, scientific promotion=false다.

## 1. 최종 endpoint edit 성능

|방법/endpoint|EFF|rewrite acc|GEN|strict GEN|rephrase acc|strict rephrase acc|LOC|
|---|---:|---:|---:|---:|---:|---:|---:|
|Pre-edit (공통 W0)|8.80% (88/1000)|0.10% (1/1000)|10.40% (208/2000)|6.00% (60/1000)|0.40% (8/2000)|0.00% (0/1000)|88.39% (8839/10000)|
|C0-KSTEP terminal K8 W|99.70% (997/1000)|99.00% (990/1000)|93.40% (1868/2000)|88.60% (886/1000)|60.35% (1207/2000)|43.40% (434/1000)|87.09% (8709/10000)|
|C0 final-v6 one-shot|99.90% (999/1000)|99.40% (994/1000)|92.80% (1856/2000)|88.20% (882/1000)|59.65% (1193/2000)|42.90% (429/1000)|87.25% (8725/10000)|
|C1-KSTEP terminal K8 W|99.70% (997/1000)|99.00% (990/1000)|93.65% (1873/2000)|89.00% (890/1000)|60.55% (1211/2000)|44.00% (440/1000)|87.09% (8709/10000)|
|C1 final-v6 one-shot|99.90% (999/1000)|99.40% (994/1000)|92.85% (1857/2000)|88.20% (882/1000)|59.55% (1191/2000)|42.70% (427/1000)|87.24% (8724/10000)|
|C3-KSTEP terminal K8 W|99.70% (997/1000)|99.10% (991/1000)|93.60% (1872/2000)|88.90% (889/1000)|60.65% (1213/2000)|44.20% (442/1000)|87.09% (8709/10000)|
|C3 final-v6 one-shot|99.90% (999/1000)|99.40% (994/1000)|92.75% (1855/2000)|88.10% (881/1000)|59.60% (1192/2000)|42.80% (428/1000)|87.26% (8726/10000)|

## 2. 이번 Phase에 추가된 방법 정의

|방법|target|writer|route|cache/실행 경계|
|---|---|---|---|---|
|C0-KSTEP|P1R52 accepted-z refreshed at every K|PIR-U control remaining-residual L4→L8 prefix sequential writer|PIR-U control allocation; P/C router call 0|ALPHA_CACHE_OFF_CONTROL; 10 independent B100; W0/cache0 restored between cases|
|C1-KSTEP|P1R52 accepted-z refreshed at every K|Joint P+C minimax + remaining-residual L4→L8 prefix sequential writer|current-K P/C proxy and Joint-PC route recomputed; router call 1/K|ALPHA_CACHE_OFF_CONTROL; 10 independent B100; W0/cache0 restored between cases|
|C3-KSTEP|P1R52 accepted-z refreshed at every K|direct Official EasyEdit AlphaEdit writer with P1R52 accepted-z|no P/C route; native AlphaEdit compute_z call 0|ALPHA_CACHE_OFF_CONTROL; 10 independent B100; W0/cache0 restored between cases|

## 3. Rewrite NLL

|방법/endpoint|rewrite target-new mean/median/p90/max|rewrite target-true mean/median/p90/max|
|---|---:|---:|
|Pre-edit (공통 W0)|11.040824/10.937304/16.200504/22.937422|4.591916/4.241698/9.646519/16.172873|
|C0-KSTEP terminal K8 W|0.062328/0.019277/0.058177/6.183085|11.448839/11.029643/16.274206/23.514282|
|C0 final-v6 one-shot|0.045779/0.014567/0.049932/6.721011|11.637259/11.189200/16.496340/23.568481|
|C1-KSTEP terminal K8 W|0.062777/0.019661/0.060640/6.160424|11.402699/11.004381/16.231215/23.576380|
|C1 final-v6 one-shot|0.045741/0.014504/0.050353/6.719562|11.641794/11.190708/16.505911/23.558739|
|C3-KSTEP terminal K8 W|0.058346/0.019126/0.057182/6.160111|11.453875/11.067334/16.239212/23.496971|
|C3 final-v6 one-shot|0.045636/0.014551/0.050300/6.686975|11.631583/11.190506/16.486992/23.524136|

## 4. Rephrase NLL

|방법/endpoint|rephrase target-new mean/median/p90/max|rephrase target-true mean/median/p90/max|
|---|---:|---:|
|Pre-edit (공통 W0)|10.081088/10.005895/14.581030/20.918356|4.700030/4.326975/9.530659/20.848831|
|C0-KSTEP terminal K8 W|2.011513/0.742238/5.996867/18.065041|8.265192/8.085827/13.036169/22.342285|
|C0 final-v6 one-shot|2.025021/0.764787/5.942323/18.160749|8.267789/8.061366/13.226711/23.160614|
|C1-KSTEP terminal K8 W|1.997650/0.728984/5.993178/18.061089|8.276484/8.105295/13.044255/21.360535|
|C1 final-v6 one-shot|2.028046/0.765858/5.954512/18.160404|8.263442/8.067918/13.241917/23.082550|
|C3-KSTEP terminal K8 W|1.994065/0.727342/5.934535/18.059998|8.290492/8.108884/13.035079/22.673908|
|C3 final-v6 one-shot|2.029781/0.771863/5.945938/18.137564|8.258608/8.079405/13.241051/23.322620|

## 5. Paired 기준 비교

|arm|EFF Δpp|GEN Δpp|strict GEN Δpp|LOC Δpp|rewrite new NLL Δ|rephrase new NLL Δ|
|---|---:|---:|---:|---:|---:|---:|
|c0|-0.200|+0.600|+0.400|-0.160|+0.016550|-0.013508|
|c1|-0.200|+0.800|+0.800|-0.150|+0.017036|-0.030396|
|c3|-0.200|+0.850|+0.800|-0.170|+0.012710|-0.035716|

|arm|paired case total Δ mean sec|paired case total ratio mean|target/writer timing comparison|update energy Δ mean|
|---|---:|---:|---|---:|
|c0|+2188.027156|2.960366|NOT_COMPARABLE_DIFFERENT_TIMER_BOUNDARY|-11.839453|
|c1|+2182.114068|2.933773|NOT_COMPARABLE_DIFFERENT_TIMER_BOUNDARY|-9.989170|
|c3|+2199.065691|2.914691|NOT_COMPARABLE_DIFFERENT_TIMER_BOUNDARY|NOT_RECORDED|

모든 비교는 동일한 10/10 unit denominator를 사용한다. Phase 2는 K-step feedback으로 accepted-z trajectory가 달라질 수 있으므로 same-entry/slice/order와 K1 pre-write identity만 결속한다.

## 6. K1→K8 trajectory와 W−z gap

`per-k-endpoint-and-compute.*`, `per-unit-endpoint-summary.*`, `per-prompt-kstep-nll.csv`에 240개 K receipt와 모든 accepted-z/pre-W/post-W prompt NLL을 보존했다. 각 K에서 post-W가 다음 K의 entry-W와 SHA로 연결됨을 fail-close 검증했다.

`cross-arm-target-identity.*` 기준 80개 unit×K 중 세 arm accepted-z/selected-target가 모두 같은 지점은 5개다. 최초 writer-feedback divergence는 unit 1 K2이다. 이 표는 차이를 결함으로 간주하지 않고, 공통 entry에서 시작한 뒤 writer별 W trajectory가 target refresh를 바꾸는 시점을 기록한다.

|arm|K8 rewrite W−z mean|K8 rephrase W−z mean|K8 z→W rewrite loss|K8 z→W rephrase loss/strict|K8 update energy mean|K8 residual mean|
|---|---:|---:|---:|---:|---:|---:|
|c0|-0.001108|0.607255|0|94/83|0.1939439439429938|0.07761037710668202|
|c1|0.000786|0.591816|0|90/81|0.3088787914573135|0.07546441829735522|
|c3|0.001109|0.597641|0|91/81|NOT_RECORDED|NOT_RECORDED|

C3의 Official native apply receipt에는 C0/C1과 동형인 layer별 update norm/energy가 기록되지 않아 해당 400개 layer placeholder는 `NOT_RECORDED_NATIVE_OFFICIAL_APPLY`로 명시했다. 시간이나 다른 aggregate에서 이를 역추정하지 않았다.

## 7. P/C route와 writer layer telemetry

C0는 PIR-U control route라 P/C router call=0, C1은 current-K Joint-PC solver call=1/K, C3는 P/C router call=0이다. `per-k-route.*`는 C1의 selected π/P/C/minimax/KKT telemetry와 C0 control allocation, C3 direct-writer status를 분리한다. `per-layer-writer-update-energy.*`는 C0/C1의 L4→L8 residual/q/update/rounding observation을 기록한다.

FP32 actual delta는 `fl32(W+U)-W`이므로 prepared U와 byte-identical일 필요가 없다. rounding mismatch는 observation-only이고 route decision influence=0이다. Official assignment endpoint 자체는 reference expression과 byte-exact다.

## 8. 계산량·시간

|arm|unit total sum/mean/median/p90/max sec|recorded phase-component wall sum|K-step writer segment sum|job sec|eval accepted/pre/post|dense/apply/commit|router|peak alloc/reserved GiB|MaxRSS GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|c0|33071.91/3307.19/3302.86/3428.37/3678.86|33964.25|27677.48|33277.22|80/80/80|400/400/80|0|34.16/38.11|12.41|
|c1|33167.28/3316.73/3264.95/3527.48/3665.24|34035.37|27796.34|33373.86|80/80/80|400/400/80|80|34.16/38.11|12.44|
|c3|33496.91/3349.69/3344.67/3400.09/3450.07|34342.90|28046.47|33702.64|80/80/80|400/400/80|0|35.42/40.99|21.25|

Phase 2는 per-case target phase-component wall과 `c_kstep_writer` segment를 기록한다. component timer 합은 case total과 반드시 분할합 관계가 아니므로 final-v6의 `target_plus_writer`와 ratio를 만들지 않고 NOT_COMPARABLE로 둔다. C3만 Official writer receipt 자체의 edit-core wall도 별도로 기록하며 C0/C1에 없는 동형 writer-only timer는 추정하지 않는다. wall time만으로 FLOPs나 원인을 역추정하지 않는다.

## 9. 사실 판정

이 보고서는 efficacy/generalization/locality/NLL/update/route/cache/compute의 관측값과 paired 차이를 제공한다. 새 threshold나 자동 promotion을 추가하지 않았으며 `scientific_promotion=false`다. 성능 차이는 해당 실행 경계의 사실 비교이지 단일 구성요소에 대한 인과 주장으로 해석하지 않는다.

## 10. 재현 산출물

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
