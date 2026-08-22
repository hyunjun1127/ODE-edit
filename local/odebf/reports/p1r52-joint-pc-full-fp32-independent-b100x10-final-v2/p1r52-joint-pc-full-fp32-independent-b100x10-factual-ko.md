# P1R52 Joint P+C FULL-FP32 독립 10×B100 사실 보고서

## 결론

- 상태: `TERMINAL_ANALYSIS_VALID`. 6개 방법 × 10개 독립 B100 = 60 endpoints, 6000 requests가 유효하다.
- 모든 floating model parameter와 writer 알고리즘 텐서는 FP32였다. BF16/FP16/autocast/quantization, numeric storage cast, retry, imputation, sequential carry는 모두 0이다.
- 각 case는 동일 slice/order와 동일 W0에서 시작했고 60/60 endpoint가 pointer/bytes exact W0 restore를 통과했다. C0–C3는 매 slice 같은 accepted-z 및 entry-W를 사용했다.
- 판정: `NO_SCIENTIFIC_PROMOTION_IN_THIS_INDEPENDENT_GATE`. 이 독립 gate는 방법을 자동 승격하거나 sequential endpoint와 혼합하지 않는다.

## 방법별 10-case 집계

|방법|W Rewrite|W Gen|W strict Gen|W Loc|rewrite W−z NLL|rephrase W−z NLL|평균 energy|L8 energy share|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|999/1000 (0.9990)|1860/2000 (0.9300)|880/1000 (0.8800)|8626/10000 (0.8626)|0.0010|0.7475|98.3771|0.4542|
|OFFICIAL-MEMIT|983/1000 (0.9830)|1651/2000 (0.8255)|745/1000 (0.7450)|8789/10000 (0.8789)|0.1956|1.7709|76.1607|0.4959|
|C0|999/1000 (0.9990)|1856/2000 (0.9280)|882/1000 (0.8820)|8725/10000 (0.8725)|-0.0037|0.6440|44.0590|0.4869|
|C1|999/1000 (0.9990)|1857/2000 (0.9285)|882/1000 (0.8820)|8724/10000 (0.8724)|-0.0037|0.6471|44.8301|0.4672|
|C2|996/1000 (0.9960)|1813/2000 (0.9065)|852/1000 (0.8520)|8718/10000 (0.8718)|0.1612|1.0603|12.9352|0.1435|
|C3|999/1000 (0.9990)|1855/2000 (0.9275)|881/1000 (0.8810)|8726/10000 (0.8726)|-0.0038|0.6488|43.3562|0.4978|

### Accepted-z와 accuracy

|방법|z Rewrite|z Gen|z strict Gen|W rewrite accuracy|W rephrase accuracy|W strict accuracy|
|---|---:|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|1000/1000|1974/2000|976/1000|999/1000|1292/2000|488/1000|
|OFFICIAL-MEMIT|1000/1000|1972/2000|974/1000|963/1000|1021/2000|347/1000|
|C0|999/1000|1966/2000|971/1000|994/1000|1193/2000|429/1000|
|C1|999/1000|1966/2000|971/1000|994/1000|1191/2000|427/1000|
|C2|999/1000|1966/2000|971/1000|966/1000|1009/2000|334/1000|
|C3|999/1000|1966/2000|971/1000|994/1000|1192/2000|428/1000|

### z→W loss와 writer realization

|방법|rewrite prompt loss|Gen prompt loss|strict Gen request loss|terminal residual|평균 L8 norm share|
|---|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|1|117|98|NOT_RECORDED|0.3177|
|OFFICIAL-MEMIT|17|322|230|NOT_RECORDED|0.3349|
|C0|0|115|92|1.2037|0.3327|
|C1|0|114|92|1.1929|0.3258|
|C2|3|157|122|26.9521|0.1700|
|C3|0|116|93|NOT_RECORDED|0.3377|

Success와 accuracy는 서로 바꾸어 쓰지 않았고 모든 numerator/denominator를 분리했다. W−z gap과 z-success→W-failure는 동일 endpoint의 prompt bit vectors에서 직접 계산했다.

### C-arm allocation 및 layer trajectory

|방법|mean π (L4→L8)|mean β (L4→L8)|predicted P(norm)|predicted C(norm)|minimax t|
|---|---|---|---:|---:|---:|
|C0|0.188, 0.216, 0.218, 0.187, 0.191|0.188, 0.266, 0.366, 0.495, 1.000|0.9927|0.9927|0.9927|
|C1|0.184, 0.215, 0.221, 0.202, 0.177|0.184, 0.264, 0.368, 0.533, 1.000|0.9927|0.9927|0.9927|
|C2|0.184, 0.215, 0.221, 0.202, 0.177|0.184, 0.264, 0.368, 0.533, 1.000|0.9927|0.9927|0.9927|
|C3|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|

P/C 값은 entry quadratic proxy가 예측한 값이다. 물리 endpoint의 별도 realized P/C scalar가 receipt에 없으므로 추정하지 않고, 실제 realization은 residual·update norm/energy·W 지표로 분리했다.

|방법/Layer|π|β|current residual|q norm|update norm share|energy share|
|---|---:|---:|---:|---:|---:|---:|
|C0/L4|0.1880|0.1880|28.5710|3.2850|0.1237|0.0674|
|C0/L5|0.2160|0.2661|26.9315|2.8712|0.1450|0.0925|
|C0/L6|0.2183|0.3664|25.5850|2.7291|0.1792|0.1413|
|C0/L7|0.1870|0.4953|23.1575|2.7889|0.2194|0.2118|
|C0/L8|0.1906|1.0000|18.6942|2.7237|0.3327|0.4869|
|C1/L4|0.1842|0.1842|28.5710|3.2850|0.1201|0.0635|
|C1/L5|0.2152|0.2638|26.9188|2.8705|0.1424|0.0893|
|C1/L6|0.2208|0.3676|25.5522|2.7281|0.1779|0.1393|
|C1/L7|0.2024|0.5330|23.1371|2.7886|0.2338|0.2407|
|C1/L8|0.1774|1.0000|18.6420|2.7144|0.3258|0.4672|
|C2/L4|0.1842|NOT_RECORDED|28.5710|3.2850|0.2107|0.2206|
|C2/L5|0.2152|NOT_RECORDED|26.9188|2.8705|0.2155|0.2305|
|C2/L6|0.2208|NOT_RECORDED|26.9641|2.7246|0.2087|0.2163|
|C2/L7|0.2024|NOT_RECORDED|27.6006|2.7938|0.1951|0.1890|
|C2/L8|0.1774|NOT_RECORDED|28.0840|2.7936|0.1700|0.1435|
|C3/L4|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|0.1332|0.0775|
|C3/L5|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|0.1383|0.0835|
|C3/L6|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|0.1653|0.1193|
|C3/L7|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|0.2255|0.2219|
|C3/L8|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|0.3377|0.4978|

### 동일-slice baseline 대비 성능 delta

|방법|기준|Δ Gen rate|Δ strict Gen rate|Δ Loc rate|Δ rephrase NLL|Δ energy|
|---|---|---:|---:|---:|---:|---:|
|C0|OFFICIAL-ALPHAEDIT|-0.0020|0.0020|0.0099|0.1770|-54.3181|
|C0|OFFICIAL-MEMIT|0.1025|0.1370|-0.0064|-0.8506|-32.1018|
|C1|OFFICIAL-ALPHAEDIT|-0.0015|0.0020|0.0098|0.1800|-53.5469|
|C1|OFFICIAL-MEMIT|0.1030|0.1370|-0.0065|-0.8476|-31.3306|
|C2|OFFICIAL-ALPHAEDIT|-0.0235|-0.0280|0.0092|0.5933|-85.4418|
|C2|OFFICIAL-MEMIT|0.0810|0.1070|-0.0071|-0.4343|-63.2255|
|C3|OFFICIAL-ALPHAEDIT|-0.0025|0.0010|0.0100|0.1817|-55.0209|
|C3|OFFICIAL-MEMIT|0.1020|0.1360|-0.0063|-0.8458|-32.8046|

## AlphaEdit 대비 시간 overhead

주 비교는 공통 evaluator와 model load를 제외한 `writer_edit_core_seconds`이다. `target_plus_writer_seconds`와 W0 entry→endpoint+restore의 `case_total_seconds`를 보조 경계로 제시한다. 각 값은 동일 slice 10/10 paired denominator이다.

|방법|edit-core 평균(s)|paired ratio 평균|overhead % 평균|target+writer ratio|case E2E ratio|
|---|---:|---:|---:|---:|---:|
|OFFICIAL-MEMIT|107.69|0.657|-34.3|0.929|0.939|
|C0|169.06|1.032|3.2|1.225|1.212|
|C1|166.03|1.017|1.7|1.243|1.229|
|C2|167.11|1.017|1.7|1.229|1.219|
|C3|161.52|0.985|-1.5|1.260|1.246|

관측 timer는 각 CUDA 구간 시작 전 synchronize를 interval 밖에 두고, 종료 synchronize의 wait를 interval 안에 포함했다. timing의 decision influence와 추가 F/B/evaluator는 0이다. 동시 task는 서로 다른 A6000에서 실행되었으므로 wall time에는 장치·contention 차이가 포함될 수 있고, 원인 단독 증거로 해석하지 않는다.

## FP32·독립성·transaction 무결성

- requested/loaded/storage dtype: `torch.float32`; parameter inventory는 cell마다 `torch.float32: 291`로 동일하다.
- accepted-z, terminal/residual, P/covariance/history/K/A/RHS/q, coefficient 및 update는 FP32이다.
- C3는 direct Official AlphaEdit entrypoint를 사용했고 native `compute_z` 호출 0, static-P FP32 load, case-entry cache width 0→exit 100을 만족했다.
- C0/C1/C2의 P/C route 수식 및 C1 remaining-residual, C2 fixed-quota 실행은 source receipt 그대로이며 이 분석은 solver를 다시 호출하지 않았다.
- case 간 physical W/controller/cache/history 전달은 0이다. 과거 sequential 결과는 slice identity 참조일 뿐 endpoint comparison denominator에 포함하지 않았다.

## SH2 sealed single-B100와 case01

- stream/order identity: True; stream-root identity: True.
- P1R52 accepted-z byte identity: False (`5772bc53771f446591d9c5a05cd598388fc26bcb41e5695f30438ac5bea45e57` → `9e9e518043a1dd8899a813ccebed7853f9e77df621f8898be285f86f242dbf50`).
- Official AlphaEdit/MEMIT 수치가 sealed report와 동일한지는 `pilot-case01-consistency.json`의 zero-delta로 확인한다. P1R52 accepted-z가 byte-different이면 current production 내부 4-arm same-z gate와 분리하여 factual cross-run difference로만 기록하며 endpoint를 대체하지 않는다.

## Compute와 자원

- `job-facts.json`은 cell별 model-load/preflight, job total, model F/B, processed tokens, peak allocated/reserved GPU memory, MaxRSS, GPU UUID를 보존한다.
- model load/job total은 one-time 사실값으로 overhead ratio에 섞지 않았다. technical-repair wasted compute는 valid job 22541에서 0이다.

|cell|job total(s)|model load(s)|model F|tokens|GPU peak alloc GiB|MaxRSS GiB|
|---|---:|---:|---:|---:|---:|---:|
|BASELINE|18348.3|11.52|54890|8114206|36.86|18.54|
|C0|11303.8|11.99|2192|618999|33.46|10.80|
|C1|11457.5|11.41|2192|618999|33.46|10.77|
|C2|11369.1|11.20|2192|618999|33.46|10.89|
|C3|11616.1|11.29|2742|1655919|34.32|19.95|

## Machine-readable denominators

- `per-case.json`: 60 rows.
- `per-layer.json`: 300 rows.
- `layer-aggregates.json`: 30 method/layer rows.
- `method-aggregates.json`: 6 methods, 각각 10 cases/1000 requests.
- `paired-performance.json`: C0–C3 × AlphaEdit/MEMIT × 10 slices.
- `paired-overhead.json`: 5 non-reference methods × 3 timing boundaries × 10 slices.
- `overhead-aggregates.json`: mean/median/p90/max seconds, paired delta, ratio, percent.
