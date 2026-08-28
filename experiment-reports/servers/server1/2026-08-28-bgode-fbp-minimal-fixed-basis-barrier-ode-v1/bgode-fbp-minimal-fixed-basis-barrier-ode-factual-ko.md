# BGODE-FBP minimal fixed-basis barrier ODE — 8-case 양모델 사실 분석

> **최종 판정:** 2 models × 8 independent W0 cases × 4 arms의 64 endpoint가 모두 기술적으로 유효하다. Static N4 barrier-off는 Native AlphaEdit와 모든 저장 endpoint/event 값이 정확히 일치해 fixed-basis·terminal-only 구현 fidelity를 닫았다. 반면 KL half-space barrier는 event/locality drift를 줄였지만 target edit strength를 크게 잃었다. Llama의 Barrier N2/N4 rewrite exact는 모두 0/8, Qwen은 N2 5/8·N4 1/8이다. 따라서 이 실험은 **load-bearing locality/strength trade-off negative result**이며 barrier method promotion은 하지 않는다.

- scheduler: job `27342`, task0 Llama / task1 Qwen, both `COMPLETED 0:0`
- execution source: `29f14cc7f515d2efe31dfc868e5775e721ba61fe` / `2788c37993a4814160b51f9efaaebcc53eec0d5c`
- cohort: `5f290439f86adb476c8c27e059cc308e30f30a641d9184eb5758f55eda237902`; ordinals `26, 61, 110, 112, 178, 231, 269, 273`
- boundary: every case starts from exact W0/cold method state; no inter-case W/cache/history carry
- fixed target/basis: native accepted-z compute 1/recompute 0 and Official ordered AlphaEdit basis capture 1 per case
- writer boundary: Euler nodes physical write 0; every arm materializes exactly once at terminal and restores W0 pointer+bytes
- FULL-FP32: parameter, accepted-z, basis/write path FP32; autocast/TF32/BF16/FP16/quantization/storage cast 0
- scientific promotion: `false`

## 1. Method 경계

|arm|정의|node factor rebuild|node physical write|terminal write|
|---|---|---:|---:|---:|
|Native AlphaEdit N1/T1|fixed W0 accepted-z를 Official AlphaEdit entrypoint에 전달한 기준 endpoint|0|0|1/case|
|Static Split Off N4|같은 fixed normalized basis에서 `theta_AE/4`를 4회 더한 barrier-off identity control|0|0|1/case|
|Barrier ODE N2|`v0=theta_AE`; current q-KL gradient와 `g·v<=0` half-space projection, Euler 2 steps|0|0|1/case|
|Barrier ODE N4|동일 projection을 Euler 4 steps로 재관측|0|0|1/case|

이 설계는 node별 AlphaEdit factor 재생성이나 node별 durable writer가 아니다. current combined affine state에서 full-model output/JVP를 재관측하지만 proposal subspace는 W0에서 고정되며, 마지막 `theta`만 1회 물리 적용한다.

## 2. Accepted-z 독립 표

accepted-z는 네 arm이 공유하며 W endpoint와 별도 provenance이다. 표는 case-level NLL 분포다.

|model|rewrite NLL mean/median/p90/max|rephrase NLL mean/median/p90/max|rewrite exact|rephrase exact|compute/recompute|
|---|---:|---:|---:|---:|---:|
|Llama3-8B-Instruct|0.000954 / 0.000788 / 0.001556 / 0.002139|2.3313 / 2.4056 / 3.6746 / 3.9206|8/8|1/8|8/0|
|Qwen2.5-7B-Instruct|0.005875 / 0.004620 / 0.010421 / 0.017381|2.9492 / 2.9189 / 4.5014 / 4.7217|8/8|0/8|8/0|

## 3. Post-W Rewrite 성능

낮은 target-new NLL과 높은 exact rate가 좋다. target-true NLL은 편집 이전 정답 억제의 관측값이다.

|model|arm|target-new NLL mean/median/p90/max|target-true NLL mean/median/p90/max|exact|W-z gap mean|
|---|---|---:|---:|---:|---:|
|Llama3-8B-Instruct|NATIVE_ALPHAEDIT_N1_T1|0.0061 / 0.0017 / 0.0134 / 0.0321|12.9529 / 12.9814 / 17.5868 / 20.6270|8/8 (100.0%)|0.0052|
|Llama3-8B-Instruct|STATIC_SPLIT_OFF_N4|0.0061 / 0.0017 / 0.0134 / 0.0321|12.9529 / 12.9814 / 17.5868 / 20.6270|8/8 (100.0%)|0.0052|
|Llama3-8B-Instruct|BARRIER_ODE_N2|4.5581 / 5.0065 / 6.3049 / 7.1464|4.0318 / 3.9260 / 6.9705 / 7.1397|0/8 (0.0%)|4.5572|
|Llama3-8B-Instruct|BARRIER_ODE_N4|7.5069 / 7.9476 / 9.8994 / 10.8139|4.5536 / 4.0271 / 6.6831 / 8.0786|0/8 (0.0%)|7.5059|
|Qwen2.5-7B-Instruct|NATIVE_ALPHAEDIT_N1_T1|0.0090 / 0.0058 / 0.0158 / 0.0316|13.9590 / 12.0079 / 22.1413 / 23.1781|8/8 (100.0%)|0.0031|
|Qwen2.5-7B-Instruct|STATIC_SPLIT_OFF_N4|0.0090 / 0.0058 / 0.0158 / 0.0316|13.9590 / 12.0079 / 22.1413 / 23.1781|8/8 (100.0%)|0.0031|
|Qwen2.5-7B-Instruct|BARRIER_ODE_N2|0.3914 / 0.1650 / 1.0222 / 1.1593|10.6358 / 10.0950 / 15.5344 / 18.4496|5/8 (62.5%)|0.3856|
|Qwen2.5-7B-Instruct|BARRIER_ODE_N4|3.5918 / 1.9027 / 7.4715 / 12.3947|6.3820 / 6.5786 / 9.4381 / 10.0849|1/8 (12.5%)|3.5860|

## 4. Post-W Rephrase 성능

rephrase exact는 두 rephrase prompt를 포함한 저장 `exact_satisfied`이며 별도 strict/accuracy 스키마는 없었다.

|model|arm|target-new NLL mean/median/p90/max|target-true NLL mean/median/p90/max|exact|W-z gap mean|
|---|---|---:|---:|---:|---:|
|Llama3-8B-Instruct|NATIVE_ALPHAEDIT_N1_T1|2.8387 / 3.2086 / 3.5371 / 3.6501|9.0800 / 9.4646 / 13.7052 / 16.2555|0/8 (0.0%)|0.5075|
|Llama3-8B-Instruct|STATIC_SPLIT_OFF_N4|2.8387 / 3.2086 / 3.5371 / 3.6501|9.0800 / 9.4646 / 13.7052 / 16.2555|0/8 (0.0%)|0.5075|
|Llama3-8B-Instruct|BARRIER_ODE_N2|6.9904 / 7.1433 / 9.6418 / 10.2516|5.4984 / 4.1856 / 9.0947 / 11.0582|0/8 (0.0%)|4.6591|
|Llama3-8B-Instruct|BARRIER_ODE_N4|8.1834 / 8.2544 / 10.7985 / 12.7581|5.2070 / 3.9824 / 8.7962 / 10.1153|0/8 (0.0%)|5.8521|
|Qwen2.5-7B-Instruct|NATIVE_ALPHAEDIT_N1_T1|2.9747 / 3.0656 / 4.6651 / 5.4308|10.7099 / 9.4437 / 17.5130 / 19.5327|0/8 (0.0%)|0.0255|
|Qwen2.5-7B-Instruct|STATIC_SPLIT_OFF_N4|2.9747 / 3.0656 / 4.6651 / 5.4308|10.7099 / 9.4437 / 17.5130 / 19.5327|0/8 (0.0%)|0.0255|
|Qwen2.5-7B-Instruct|BARRIER_ODE_N2|3.3964 / 3.0827 / 5.0236 / 5.6687|9.5938 / 8.8327 / 15.1365 / 15.3758|0/8 (0.0%)|0.4472|
|Qwen2.5-7B-Instruct|BARRIER_ODE_N4|6.7726 / 5.5377 / 11.2337 / 16.5108|6.8277 / 7.5965 / 8.6343 / 8.9649|0/8 (0.0%)|3.8233|

## 5. Event/locality와 strength trade-off

`event_q_kl_from_w0`는 target-excluded first-departure q0 drift, `locality_forward_kl`은 저장 evaluator locality KL이다. 둘 다 낮을수록 보존적이다. 표의 target/source probability는 같은 rewrite event partition에서의 값이다.

|model|arm|event q-KL mean/median/p90/max|locality KL mean/median/p90/max|target p mean|source p mean|coefficient norm mean|
|---|---|---:|---:|---:|---:|---:|
|Llama3-8B-Instruct|NATIVE_ALPHAEDIT_N1_T1|4.0249 / 4.3386 / 5.8702 / 6.5770|1.3565 / 0.8253 / 3.0851 / 5.0721|0.998069|0.000001|0.9915|
|Llama3-8B-Instruct|STATIC_SPLIT_OFF_N4|4.0249 / 4.3386 / 5.8702 / 6.5770|1.3565 / 0.8253 / 3.0851 / 5.0721|0.998069|0.000001|0.9915|
|Llama3-8B-Instruct|BARRIER_ODE_N2|0.4659 / 0.4472 / 0.9195 / 1.0661|0.1469 / 0.0922 / 0.3306 / 0.4270|0.045532|0.016377|0.6762|
|Llama3-8B-Instruct|BARRIER_ODE_N4|0.0583 / 0.0425 / 0.1225 / 0.1609|0.0313 / 0.0138 / 0.0704 / 0.1053|0.001209|0.004770|0.5346|
|Qwen2.5-7B-Instruct|NATIVE_ALPHAEDIT_N1_T1|3.5898 / 2.8256 / 7.0065 / 7.4728|1.1380 / 0.6266 / 2.4631 / 4.8276|0.980719|0.000003|4.7643|
|Qwen2.5-7B-Instruct|STATIC_SPLIT_OFF_N4|3.5898 / 2.8256 / 7.0065 / 7.4728|1.1380 / 0.6266 / 2.4631 / 4.8276|0.980719|0.000003|4.7643|
|Qwen2.5-7B-Instruct|BARRIER_ODE_N2|2.6104 / 2.0346 / 5.8377 / 6.8956|0.9964 / 0.5798 / 1.9635 / 3.9931|0.698870|0.000036|3.3704|
|Qwen2.5-7B-Instruct|BARRIER_ODE_N4|0.7649 / 0.2270 / 1.8940 / 3.9027|0.3422 / 0.1798 / 0.8784 / 0.9594|0.162838|0.001884|3.3023|

### Paired factual reading

- Static-vs-Native: 16/16 paired cases에서 네 endpoint NLL과 event q-KL 최대 절대차가 `0.0`이고, logits hash도 `64/64` 일치했다.
- Llama: Native rewrite 8/8에서 N2/N4가 0/8로 감소했다. event q-KL mean은 4.0249→0.4659→0.0583, locality KL mean은 1.3565→0.1469→0.0313으로 낮아졌지만 rewrite NLL mean은 0.0061→4.5581→7.5069로 악화했다.
- Qwen: Native 8/8, N2 5/8, N4 1/8이다. event q-KL mean 3.5898→2.6104→0.7649, locality KL 1.1380→0.9964→0.3422와 함께 rewrite NLL 0.0090→0.3914→3.5918로 증가했다.
- N4는 N2보다 더 강한 q/locality 보존을 보였으나 두 모델 모두 target strength를 더 잃었다. 이번 projection은 `g·v<=0`만 지키며 target progress equality나 minimum efficacy constraint를 갖지 않으므로 이 결과는 구현식과 일관된다.
- target-new rephrase exact는 모든 arm/모델에서 0/8이다. 따라서 rewrite만으로 method 성공을 주장할 수 없다.

## 6. Node trajectory

node-level raw 96 rows는 `node-trajectory.csv`, model×arm×node aggregate는 `node-summary.csv`에 있다. 모든 case에서 node0 q0 barrier gradient가 numerical zero였고 첫 step은 nominal velocity였다. 이후 projection 활성화가 nominal AlphaEdit 방향의 q-KL 증가 성분을 제거하면서 coefficient path가 Native에서 이탈했다. N을 2→4로 늘리면 더 자주 재관측·투영되어 보존은 강화됐지만 target endpoint는 더 약해졌다.

이 결과는 `N4가 더 정확한 ODE 해이므로 과학적으로 우수하다`는 증거가 아니다. 이 실험은 convergence grid가 아니며 N2/N4 endpoint strength가 match되지 않는다.

## 7. Layer update

proposal factors는 서로 다른 layer block에서 Frobenius-normalized되어 `||Delta W||_F^2 = sum_j theta_j^2`가 성립한다. `layer-update.csv`는 320 model/case/arm/layer rows에 coefficient, block energy, energy share를 저장한다. Barrier N4의 더 작은 coefficient norm은 target strength 약화와 함께 관측되며, 별도 capacity/causal claim은 하지 않는다.

## 8. Compute·시간·메모리

|model|job wall|model load|case panel mean/median/p90/max|observer wall mean|JVP|observer forwards|peak alloc/reserved GiB|host RSS GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Llama3-8B-Instruct|2177.15s|4.83s|270.21 / 266.77 / 283.82 / 302.30s|13.98s|480|496|37.26/40.54|17.55|
|Qwen2.5-7B-Instruct|2590.47s|4.30s|321.49 / 320.34 / 339.33 / 353.02s|13.53s|480|496|38.46/41.77|29.09|

중요한 timing limitation: per-case functional-observer ledger는 q0+N2+N4를 합친 값이고 barrier arm별 integration wall을 분리하지 않았다. non-native `wall_seconds`는 trajectory 계산 뒤 terminal materialization+endpoint evaluation 경계다. 따라서 Native 대비 arm별 end-to-end overhead ratio는 `NOT_COMPARABLE_SCHEMA_GAP`; 시간을 역추정하거나 분배하지 않았다.

## 9. Completeness와 기술 gate

- scheduler terminals 2/2; model terminals 2/2; cases 16/16; endpoints 64/64; barrier nodes 96/96
- fixed-z private artifacts 16/16; failures 0; retries 0; imputation 0
- W0 pointer+bytes restore 16/16; accepted-z compute/recompute 16/0; basis capture/rebuild 16/0
- node physical writes 0; authoritative terminal writes 64/64; layer applies 320/320
- Official adapter fidelity audit는 첫 case에서만 실행하도록 봉인되어 model별 1/1 PASS, 두 audit 모두 layer delta/hash exact (`relative_frobenius=0`); 나머지 14 cases는 `NOT_RUN_BY_DESIGN`이다.
- raw result/log/private fixed-z bytes mutation 0; 분석은 read-only; GPU/model/Slurm replay 0

## 10. 결론과 다음 설계 판단

1. fixed normalized W0 AlphaEdit subspace와 terminal-only writer 구현은 Native identity control로 정확히 검증됐다.
2. 현재 q-KL non-increase half-space는 load-bearing하다. 실제로 q/locality drift를 줄였지만, 그 작동 방식은 efficacy 방향을 제거하는 것이었고 strength 보존 장치가 없어 편집 성능이 무너졌다.
3. 따라서 현재 Barrier ODE N2/N4를 method로 승격하지 않는다. 다음 구현이 필요하다면 outcome-tuned 재시도가 아니라 사전 정의된 target-progress equality/constraint 또는 strength-matched comparison을 먼저 수학적으로 닫아야 한다.
4. 표준 EFF/GEN/LOC 퍼센트는 이 run schema에 저장되지 않았다. 저장 `exact_satisfied`와 locality forward-KL만 보고했으며 이를 EFF/GEN/LOC로 이름 바꾸지 않았다.
5. eight-case cohort는 tokenizer-only natural unequal-nonprefix cohort이며 전체 dataset 대표성이나 통계적 유의성을 주장하지 않는다. cross-model 결과도 각각 factual endpoint로만 해석한다.

## 11. Machine-readable 산출물

- `accepted-z-summary.csv`: model별 accepted-z NLL/exact/locality
- `arm-summary.csv`: model×arm 8-row aggregate
- `case-arm.csv`: 64 paired endpoint rows
- `prompt-nll.csv`: accepted-z/post-W prompt-level NLL
- `paired-deltas.csv`: Native/N2/N4 paired case deltas
- `node-trajectory.csv`, `node-summary.csv`: 96 raw node rows와 aggregate
- `layer-update.csv`: 320 normalized block energy rows
- `compute-summary.csv`, `scheduler-terminal.json`
- `artifact-inventory.json`, `analysis-summary.json`, `analysis-manifest.json`, `rooted-receipt.json`
