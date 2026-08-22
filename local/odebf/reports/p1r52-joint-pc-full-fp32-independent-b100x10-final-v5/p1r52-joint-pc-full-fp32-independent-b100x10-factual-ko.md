# P1R52 Joint P+C FULL-FP32 독립 10×B100 사실 보고서

> **핵심 실행 경계:** 이번 C0–C3는 이전 J0의 step-wise write 경로와 달리, P1R52 target controller가 K8까지 z 최적화를 모두 완료한 뒤 최종 accepted-z를 고정하고 writer를 정확히 한 번만 실행한 one-shot 편집이다. 따라서 아래 K1→K8 표는 write 중간 endpoint가 아니라 최종 write 전에 수행된 z/target 최적화 telemetry이며, 실제 W endpoint는 그 뒤의 단일 writer 적용 결과다.

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

### Accepted-z 전용 요약

C0–C3는 각 slice 안에서 동일한 P1R52 K8 accepted-z를 사용했다(`PASS_10_OF_10`). AlphaEdit/MEMIT의 Z는 각 native method가 만든 별도 reference이므로 C-arm accepted-z와 byte-equivalence를 주장하지 않는다.

|방법|Z source|z Rewrite|z Gen|z strict Gen|z rewrite acc|z rephrase acc|z strict acc|z Loc|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|NATIVE_ALPHAEDIT_Z|1000/1000|1974/2000|976/1000|1000/1000|1525/2000|625/1000|8626/10000|
|OFFICIAL-MEMIT|NATIVE_MEMIT_Z|1000/1000|1972/2000|974/1000|1000/1000|1521/2000|624/1000|8789/10000|
|C0|P1R52_K8_ACCEPTED_Z_SHARED_C0_C3|999/1000|1966/2000|971/1000|993/1000|1411/2000|569/1000|8839/10000|
|C1|P1R52_K8_ACCEPTED_Z_SHARED_C0_C3|999/1000|1966/2000|971/1000|993/1000|1411/2000|569/1000|8839/10000|
|C2|P1R52_K8_ACCEPTED_Z_SHARED_C0_C3|999/1000|1966/2000|971/1000|993/1000|1411/2000|569/1000|8839/10000|
|C3|P1R52_K8_ACCEPTED_Z_SHARED_C0_C3|999/1000|1966/2000|971/1000|993/1000|1411/2000|569/1000|8839/10000|

### Z/W 새 target NLL: 전체 prompt mean·median·p90

Rewrite denominator는 10×100=1000 prompts, rephrase denominator는 10×100×2=2000 prompts이다. p90은 nearest-rank `ceil(0.9N)`이다. case-level mean의 10-case mean/median/p90도 별도 machine table에 보존했다.

|방법|Z rewrite mean/median/p90|Z rephrase mean/median/p90|W rewrite mean/median/p90|W rephrase mean/median/p90|
|---|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|0.0023 / 0.0007 / 0.0027|1.1005 / 0.0520 / 4.3085|0.0033 / 0.0007 / 0.0029|1.8480 / 0.2935 / 5.9014|
|OFFICIAL-MEMIT|0.0011 / 0.0006 / 0.0023|1.1048 / 0.0540 / 4.2941|0.1967 / 0.0043 / 0.1097|2.8756 / 1.1551 / 8.3535|
|C0|0.0494 / 0.0151 / 0.0561|1.3810 / 0.2505 / 4.7313|0.0458 / 0.0146 / 0.0499|2.0250 / 0.7648 / 5.9423|
|C1|0.0494 / 0.0151 / 0.0561|1.3810 / 0.2505 / 4.7313|0.0457 / 0.0145 / 0.0504|2.0280 / 0.7659 / 5.9545|
|C2|0.0494 / 0.0151 / 0.0561|1.3810 / 0.2505 / 4.7313|0.2107 / 0.0808 / 0.5079|2.4413 / 1.3029 / 6.4880|
|C3|0.0494 / 0.0151 / 0.0561|1.3810 / 0.2505 / 4.7313|0.0456 / 0.0146 / 0.0503|2.0298 / 0.7719 / 5.9459|

### Z/W 원 target NLL: 전체 prompt mean·median·p90

Rewrite denominator는 10×100=1000 prompts, rephrase denominator는 10×100×2=2000 prompts이다. p90은 nearest-rank `ceil(0.9N)`이다. case-level mean의 10-case mean/median/p90도 별도 machine table에 보존했다.

|방법|Z rewrite mean/median/p90|Z rephrase mean/median/p90|W rewrite mean/median/p90|W rephrase mean/median/p90|
|---|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|14.6288 / 14.3504 / 19.8187|11.0501 / 10.9763 / 16.3432|14.5100 / 14.2283 / 19.6967|9.3969 / 9.3683 / 14.6414|
|OFFICIAL-MEMIT|14.8202 / 14.5421 / 19.8044|11.0667 / 11.0030 / 16.4264|11.5346 / 11.3594 / 17.1330|7.7943 / 7.6687 / 13.1024|
|C0|11.7520 / 11.2935 / 16.5283|9.8306 / 9.6320 / 14.5532|11.6373 / 11.1892 / 16.4963|8.2678 / 8.0614 / 13.2267|
|C1|11.7520 / 11.2935 / 16.5283|9.8306 / 9.6320 / 14.5532|11.6418 / 11.1907 / 16.5059|8.2634 / 8.0679 / 13.2419|
|C2|11.7520 / 11.2935 / 16.5283|9.8306 / 9.6320 / 14.5532|9.9260 / 9.7130 / 14.7934|7.8787 / 7.7168 / 12.7360|
|C3|11.7520 / 11.2935 / 16.5283|9.8306 / 9.6320 / 14.5532|11.6316 / 11.1905 / 16.4870|8.2586 / 8.0794 / 13.2411|

### W accuracy

|방법|W rewrite accuracy|W rephrase accuracy|W strict accuracy|
|---|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|999/1000|1292/2000|488/1000|
|OFFICIAL-MEMIT|963/1000|1021/2000|347/1000|
|C0|994/1000|1193/2000|429/1000|
|C1|994/1000|1191/2000|427/1000|
|C2|966/1000|1009/2000|334/1000|
|C3|994/1000|1192/2000|428/1000|

### C 계열 K-step target progress

Runtime receipt에는 C0–C3 각각 K1→K8의 7개 delayed target transition이 기록돼 있었다. 아래는 각 transition의 10-case 평균이며, per-case 원값과 mean/median/p90은 `c-target-step.json` 및 `c-target-step-aggregates.json`에 보존했다. heldout evaluator와 candidate objective 재평가는 모두 0이다. `target_depth_inner_telemetry_enabled=false`이고 봉인된 bookkeeping row count는 case당 8로 별도 기록했다.

|방법|transition|source NLL|next-field NLL|predicted progress|actual progress|linearization error|realization ratio|
|---|---|---:|---:|---:|---:|---:|---:|
|C0|K1→K2|10.3736|8.2518|2.4877|2.1217|-0.3659|0.8532|
|C0|K2→K3|8.2518|4.7133|3.6924|3.5386|-0.1538|0.9587|
|C0|K3→K4|4.7133|2.5596|2.9696|2.1537|-0.8159|0.7240|
|C0|K4→K5|2.5596|1.1805|1.9270|1.3790|-0.5480|0.7161|
|C0|K5→K6|1.1805|0.5739|0.9189|0.6067|-0.3123|0.6599|
|C0|K6→K7|0.5739|0.2816|0.4420|0.2923|-0.1497|0.6602|
|C0|K7→K8|0.2816|0.1553|0.2055|0.1263|-0.0792|0.6101|
|C1|K1→K2|10.3736|8.2518|2.4877|2.1217|-0.3659|0.8532|
|C1|K2→K3|8.2518|4.7133|3.6924|3.5386|-0.1538|0.9587|
|C1|K3→K4|4.7133|2.5596|2.9696|2.1537|-0.8159|0.7240|
|C1|K4→K5|2.5596|1.1805|1.9270|1.3790|-0.5480|0.7161|
|C1|K5→K6|1.1805|0.5739|0.9189|0.6067|-0.3123|0.6599|
|C1|K6→K7|0.5739|0.2816|0.4420|0.2923|-0.1497|0.6602|
|C1|K7→K8|0.2816|0.1553|0.2055|0.1263|-0.0792|0.6101|
|C2|K1→K2|10.3736|8.2518|2.4877|2.1217|-0.3659|0.8532|
|C2|K2→K3|8.2518|4.7133|3.6924|3.5386|-0.1538|0.9587|
|C2|K3→K4|4.7133|2.5596|2.9696|2.1537|-0.8159|0.7240|
|C2|K4→K5|2.5596|1.1805|1.9270|1.3790|-0.5480|0.7161|
|C2|K5→K6|1.1805|0.5739|0.9189|0.6067|-0.3123|0.6599|
|C2|K6→K7|0.5739|0.2816|0.4420|0.2923|-0.1497|0.6602|
|C2|K7→K8|0.2816|0.1553|0.2055|0.1263|-0.0792|0.6101|
|C3|K1→K2|10.3736|8.2518|2.4877|2.1217|-0.3659|0.8532|
|C3|K2→K3|8.2518|4.7133|3.6924|3.5386|-0.1538|0.9587|
|C3|K3→K4|4.7133|2.5596|2.9696|2.1537|-0.8159|0.7240|
|C3|K4→K5|2.5596|1.1805|1.9270|1.3790|-0.5480|0.7161|
|C3|K5→K6|1.1805|0.5739|0.9189|0.6067|-0.3123|0.6599|
|C3|K6→K7|0.5739|0.2816|0.4420|0.2923|-0.1497|0.6602|
|C3|K7→K8|0.2816|0.1553|0.2055|0.1263|-0.0792|0.6101|

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
- `nll-prompt-distribution-aggregates.json`: 전체 prompt 기준 Z/W NLL mean/median/p90 48 rows.
- `nll-distribution-aggregates.json`: case-level mean 기준 10-case mean/median/p90 48 rows.
- `c-target-step.json`: C0–C3 × 10 cases × K1→K8의 280개 target-transition rows.
- `c-target-step-aggregates.json`: 28 method/transition mean·median·p90 rows.
- `paired-performance.json`: C0–C3 × AlphaEdit/MEMIT × 10 slices.
- `paired-overhead.json`: 5 non-reference methods × 3 timing boundaries × 10 slices.
- `overhead-aggregates.json`: mean/median/p90/max seconds, paired delta, ratio, percent.
