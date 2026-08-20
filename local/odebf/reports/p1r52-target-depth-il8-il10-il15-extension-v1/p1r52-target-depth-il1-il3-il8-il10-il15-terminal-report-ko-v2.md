# P1R52 Target Depth IL1/IL3/IL8/IL10/IL15 Atomic 사실 보고서

- 생성 시각(KST): `2026-08-20T17:51:19+09:00`
- 실행 source HEAD/tree: `8d94a8c96db99c5ffdadbcd13f6c0347e9f2ca8d` / `d974ee426a033285d5efaa7dcfa7f1e3a8effaf6`
- contract SHA256: `06e65d4a2df4a4610ef09818c2afd96b8dbe86cc3041afdbc5b099ca2358ec63`
- numerical lock SHA256: `82b9c996287b8d42d27193ac14ad765d158d32367f1c00a027aadc98987bcbdb`
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- immutable IL1/IL3 report/table references: independent SHA rehash PASS.
- 범위: Llama3-8B-Instruct, Soft J0, independent B10×10, K8; 신규 IL8/IL10/IL15. IL1/IL3는 immutable reference 재사용.
- IL5/Qwen/Neutral/Native/sequential/Historical 실행: `0`.

## 한 문단 결론

30/30 신규 endpoint가 유효했다. 관측 범위에서 z full-six NLL 최저는 IL15-FULL (0.00169062), W full-six NLL 최저는 IL15-FULL (0.00273839)였다. W Gen 최대값 184/200은 IL15-FULL, Loc 최대값 876/1000은 IL8-FULL, IL10-FULL, IL15-FULL에서 기록됐다. depth별 z/W·전달·보존·에너지·compute 절대값은 아래 표에 병렬 제시하며, 단일 NLL만으로 scientific promotion을 선택하지 않았다 (`scientific_promotion=false`).

## Depth 추세의 기계적 분류

- z full-six NLL은 `0.0414769 → 0.00685335 → 0.00255173 → 0.00228425 → 0.00169062`로 모든 인접 depth에서 감소했다: `MONOTONIC_Z_NLL_IMPROVEMENT`.
- W full-six NLL은 IL1→IL3→IL8에서 감소했고, IL8→IL10에서 `+0.000139474` 증가한 뒤 IL10→IL15에서 `-0.000458015` 감소했다: `W_TRANSFER_NON_MONOTONIC_WITH_IL10_REGRESSION`.
- W Gen은 `181 → 181 → 183 → 183 → 184 / 200`; Loc은 `871 → 874 → 876 → 876 → 876 / 1000`이다: IL8→IL10 endpoint count는 `PLATEAU`, IL15 W Gen은 `+1`이다.
- IL3→IL8→IL10→IL15 edit-core 평균은 `122.683 → 296.995 → 349.147 → 461.104 s`; P/capacity/BF16 path energy도 각각 `0.0104796/5.68461/3.81858 → 0.0145780/8.14928/6.54304 → 0.0148202/8.27252/6.74354 → 0.0149701/8.36357/6.88872`로 증가했다.
- 관측값 기준 IL15는 최저 z/W full-six NLL과 최고 W Gen을 동시에 기록했고, 신규 depth 중 가장 큰 compute/P/capacity/energy를 기록했다. 단일 최적 depth 또는 promotion 선택은 권한 범위 밖이며 `scientific_promotion=false`다.

## 절대 분모와 z/W 지표

성공은 pinned evaluator의 margin-positive prompt bit이며 strict Gen은 request의 모든 rephrase prompt 성공이다. accuracy는 별도 기록이 없어 `NOT_RECORDED`다.

| depth | valid/attempt | z Eff | z Gen/strict | W Eff | W Gen/strict | Loc | z/W/gap full6 NLL | z rewrite NLL mean/med/p90/max | z rephrase NLL mean/med/p90/max | W rewrite NLL mean/med/p90/max | W rephrase NLL mean/med/p90/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IL1 | 10/10 | 100/100 | 183/200; NOT_RECORDED | 100/100 | 181/200; NOT_RECORDED | 871/1000 | 0.0414769/0.0792804/0.0378035 | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL |
| IL3-FULL | 10/10 | 100/100 | 181/200; 84 | 100/100 | 181/200; 84 | 874/1000 | 0.00685335/0.00860811/0.00175476 | 0.00431773/0.0027771/0.00784912/0.0712891 | 2.04494/1.18451/4.31445/14.3438 | 0.0056534/0.00320435/0.00940552/0.118652 | 2.09174/1.19611/4.37012/14.2812 |
| IL8-FULL | 10/10 | 100/100 | 184/200; 87 | 100/100 | 183/200; 86 | 876/1000 | 0.00255173/0.00305693/0.000505204 | 0.00144992/0.00107193/0.00270538/0.0136719 | 1.84329/1.06826/4.18086/14.4062 | 0.00172076/0.00110626/0.00307465/0.024292 | 1.89016/1.07092/4.25449/14.375 |
| IL10-FULL | 10/10 | 100/100 | 184/200; 87 | 100/100 | 183/200; 86 | 876/1000 | 0.00228425/0.00319641/0.000912159 | 0.00120214/0.000955582/0.00242004/0.00695801 | 1.84372/1.06345/4.20215/14.4375 | 0.00134161/0.00104523/0.00254822/0.00848389 | 1.87587/1.08374/4.21641/14.4375 |
| IL15-FULL | 10/10 | 100/100 | 184/200; 87 | 100/100 | 184/200; 87 | 876/1000 | 0.00169062/0.00273839/0.00104777 | 0.000921946/0.000814438/0.0017662/0.00415802 | 1.82126/1.04414/4.14741/14.4062 | 0.00116826/0.000858307/0.00188446/0.015564 | 1.86325/1.01758/4.16616/14.375 |

## Rewrite/rephrase margin

| depth | z rewrite margin mean/p10 | z rephrase margin mean/p10 | W rewrite margin mean/p10 | W rephrase margin mean/p10 |
|---|---:|---:|---:|---:|
| IL1 | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL | NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL |
| IL3-FULL | 13.4313/8.98555 | 6.74152/1.8834 | 13.2853/8.72192 | 6.64474/1.54756 |
| IL8-FULL | 14.4573/9.64045 | 7.25418/2.12109 | 14.3073/9.7539 | 7.13457/1.76875 |
| IL10-FULL | 14.6325/10.0162 | 7.28385/1.93223 | 14.5605/10.0546 | 7.24503/1.90234 |
| IL15-FULL | 14.851/10.2668 | 7.40822/2.08398 | 14.7095/9.91756 | 7.40127/2.10547 |

## Writer 전달·보존·계산

| depth | predicted/actual/realization | neg | P endpoint mean | capacity endpoint mean | BF16 path energy mean | simplex/velocity/layer-energy top1 | target path outer/inner mean | final-inner ΔNLL/move | F/B/tokens/mat | target/inner-obs/outer-obs/edit/eval mean(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IL1 | NOT_RECORDED_IL1_AGGREGATE/NOT_RECORDED_IL1_AGGREGATE/NOT_RECORDED_IL1_AGGREGATE | 5 | 0.00347738 | 5.52713 | 3.64779 | NOT_RECORDED_IL1_AGGREGATE/NOT_RECORDED_IL1_AGGREGATE/NOT_RECORDED_IL1_AGGREGATE | NOT_RECORDED/NOT_RECORDED | NOT_APPLICABLE_IL1_SINGLE_INNER/NOT_RECORDED_IL1_PRE_TELEMETRY | NOT_RECORDED_IL1_LLAMA_SOFT_CELL/NOT_RECORDED_IL1_LLAMA_SOFT_CELL/NOT_RECORDED_IL1_LLAMA_SOFT_CELL/80 | NOT_RECORDED_IL1_LLAMA_SOFT_CELL/NOT_APPLICABLE_IL1_PRE_TELEMETRY/NOT_APPLICABLE_IL1_PRE_TELEMETRY/NOT_RECORDED_IL1_LLAMA_SOFT_CELL/NOT_RECORDED_IL1_LLAMA_SOFT_CELL |
| IL3-FULL | 113.079/103.799/0.917935 | 0 | 0.0104796 | 5.68461 | 3.81858 | 0.461288/NOT_RECORDED_IL3_AGGREGATE/NOT_RECORDED_IL3_AGGREGATE | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED_IL3_PRE_TELEMETRY/NOT_RECORDED_IL3_PRE_TELEMETRY | 4925/2850/791437/80 | NOT_RECORDED_IL3_AGGREGATE/NOT_APPLICABLE_IL3_PRE_TELEMETRY/NOT_APPLICABLE_IL3_PRE_TELEMETRY/122.683/34.1302 |
| IL8-FULL | 105.988/103.854/0.979872 | 3 | 0.014578 | 8.14928 | 6.54304 | 0.488492/0.400732/0.438034 | 10.6877/20.6735 | -0.00500599/0.100613 | 20785/6850/4444332/80 | 205.68/77.1233/19.1601/296.995/2.39196 |
| IL10-FULL | 105.657/103.853/0.982922 | 4 | 0.0148202 | 8.27252 | 6.74354 | 0.484407/0.397706/0.434169 | 10.385/20.6323 | -0.00076213/0.0334977 | 25590/8450/5442232/80 | 257.312/98.2714/19.5171/349.147/2.44014 |
| IL15-FULL | 105.698/103.858/0.982585 | 3 | 0.0149701 | 8.36357 | 6.88872 | 0.506317/0.43012/0.470596 | 10.3237/20.9993 | -0.000217883/0.0173138 | 36655/12160/7737947/80 | 372.715/143.413/19.4161/461.104/2.42305 |

## 인접 depth 산술 차이

| comparison | Δz full6 | ΔW full6 | ΔW-z gap | Δz E/G | ΔW E/G/Loc | ΔP/cap/energy | Δedit-core(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| IL3-FULL-IL1 | -0.0346236 | -0.0706723 | -0.0360487 | 0/-2 | 0/0/3 | 0.0070022/0.157474/0.170791 | NOT_RECORDED |
| IL8-FULL-IL3-FULL | -0.00430162 | -0.00555117 | -0.00124955 | 0/3 | 0/2/2 | 0.00409845/2.46467/2.72446 | 174.313 |
| IL10-FULL-IL8-FULL | -0.000267481 | 0.000139474 | 0.000406955 | 0/0 | 0/0/0 | 0.000242161/0.123245/0.200499 | 52.1515 |
| IL15-FULL-IL10-FULL | -0.00059363 | -0.000458015 | 0.000135614 | 0/0 | 0/1/0 | 0.000149921/0.0910489/0.145185 | 111.957 |

- IL1 continuous rewrite/rephrase NLL, strict Gen, per-case/per-request compute는 local immutable input에 없어 `NOT_RECORDED`; proxy/imputation 0.
- 신규 typed scientific failures: `0`; technical failure/retry/imputation: `0/0/0`.
- table rows: per-case `30`, per-step `240`, per-inner `2611`, per-inner-request `26110`, terminal per-request `300`, global-ordinal summary `264`.

## Per-inner telemetry 완전성

| depth | executed/configured max rows (10 cases) | byte-identical early-stop outers | outer rows | request-bound fields | duplicate eval | added B/gen/action influence |
|---|---:|---:|---:|---|---:|---:|
| IL1 | NOT_RECORDED_IL1_PRE_TELEMETRY/80 | NOT_RECORDED_IL1_PRE_TELEMETRY | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| IL3-FULL | NOT_RECORDED_IL3_PRE_TELEMETRY/240 | NOT_RECORDED_IL3_PRE_TELEMETRY | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| IL8-FULL | 640/640 | 0 | 80 | target objective/rewrite/rephrase/movement | 0 | 0/0/0 |
| IL10-FULL | 800/800 | 0 | 80 | target objective/rewrite/rephrase/movement | 0 | 0/0/0 |
| IL15-FULL | 1171/1200 | 6 | 80 | target objective/rewrite/rephrase/movement | 0 | 0/0/0 |

- IL15 일부 outer는 계약상 허용된 전체 selected FP32 tensor byte-identical early stop으로 15회 이전 종료됐다. 각 short trajectory는 마지막 두 `target_next_sha256` 동일성으로 검증했다; 누락/추정/imputation으로 채우지 않았다.
- 신규 각 accepted inner observation은 target objective/rewrite/rephrase/movement를 request/depth/case/outer/inner/ordinal에 결합하며 added backward/generation/action influence/duplicate evaluation은 모두 0이다.
- per-inner selection PRIMARY/RESCUE/CURRENT와 selected endpoint NLL은 기록됨. 원 target-step의 상세 clamp/KL/decay/reference-energy receipt 본문은 accepted result에 저장되지 않아 `NOT_RECORDED`; hash만 기록됨.

## 계약 질문 5개

1. target-depth 자체의 z 변화: 위 depth별 z full-six/rewrite/rephrase/Eff/Gen 및 인접 산술 차이에 기록했다.
2. z 변화의 J0 writer 전달: W 지표, W-z gap, predicted/actual/realization에 기록했다.
3. sequential 누적 유지: 이 extension에서 sequential 실행 0이므로 `NOT_EVALUATED`.
4. KDC/barrier 대 depth: KDC와 writer는 고정되고 depth만 바뀌었으나 barrier 단독 기여 분리는 수행하지 않았다. `TARGET_DEPTH_ONLY_ABLATION`.
5. Qwen/Llama 차이: 신규 실행은 Llama Soft만이므로 Qwen 신규 비교는 `NOT_EVALUATED`.

scientific_promotion=false. 추가 model/GPU/Slurm job=0.
