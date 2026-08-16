# P2R7 P2TARGET-P1DW Shared Writer TECH-R2 파일럿 분석

- instruction: `ODEEDIT-S05-P2R7-P2TARGET-P1DW-SHARED-WRITER-V1`
- scheduler: array job `20148`, tasks `0` Llama / `1` Qwen, terminal `COMPLETED`, exit `0:0` (2/2)
- source HEAD/tree: `a7e533a6f65dec7add3f4f6b513f10d51ae81640` / `601af3498b6d1e957c1017b9096cfa11d05c29ee`
- exact parent: P2R2 TECH-R4 `c97e8619b42da7954ce0e824c215a8a82d70589a`
- contract SHA256: `d58e5c067d5687559e2bd809d9c5da17c4d546efa6e8feb679f3038f7ebaee5b`
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- evaluator/aggregator: `25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145` / `64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0`
- 분모: 8 attempts / 8 complete endpoints / 0 typed incomplete / 80 requests / 64 writer steps / 192 target microsteps
- 파일럿 경계: case01 한 건의 소표본이며 과학 승격은 `false`이다.

## 1. 기술 게이트

| 항목 | 관측값 | 상태 |
|---|---:|---|
| endpoint / failure | 8 / 0 | PASS |
| target microstep / writer transition / materialization | 192 / 64 / 64 | PASS |
| endpoint별 K8 / target24 / materialization8 | 8/8 | PASS |
| W0 pointer / byte restore | 8/8 / 8/8 | PASS |
| action-freeze 및 terminal-only evaluator | 8/8, heldout controller access 0 | PASS |
| routing 변수 | 전 step 5 | PASS |
| request×layer response matrix | 전 step 0 | PASS |
| residual `1/h` / physical `h` / second `h` | 전 step 1 / 1 / 0 | PASS |
| omega 합 최대 절대 오차 | `1.1102230246251565e-16` | PASS |
| `alpha_apply-alpha_req` 최대 절대 오차 | `1.7763568394002505e-15` | PASS |
| current-state refresh / batched VJP | 64 / 64 | PASS |
| retry / backtracking / candidate materialization | 0 / 0 / 0 | PASS |
| P2R6 QP import / shadow, hard-P, functional veto, history-H | 모두 0 | PASS |
| NO_SEMANTIC_DEFICIT / no-positive-direction | 0 / 0 step | 관측 없음 |
| Soft→Neutral fallback | 0/32 Soft step | 관측 없음 |

P2R2 matched case01 Neutral의 W0 첫 outer microstep 0–2와 비교한 의미 필드(`field_sha256`, target before/next SHA, request별 target NLL, semantic rate/tangent residual, clamp 전후 displacement norm)는 Llama와 Qwen에서 각각 3/3 byte-identical했다. P2R7 네 팔 내부에서도 같은 의미 필드의 첫 3개 microstep은 모델별 4/4 동일했다. 전체 target module SHA는 `98591cc1472c23d14178002dd535596916c93751d2d10b95f1306dc7705dac21`이다.

기술 게이트 상태는 `PILOT_TECHNICAL_PASS`이다. 계약의 파일럿 과학 지표는 lenient이며 기술적으로 유효하면 production을 결과 선택 없이 진행하도록 고정되어 있다. 따라서 기계적 다음 상태는 `PRODUCTION_RELEASE_ELIGIBLE_BY_CONTRACT`이고 과학 승격은 아니다.

## 2. 파일럿 endpoint 원값

### Llama3-8B-Instruct

| weighting | arm | W E/G/L | W Eff/Gen NLL | z E/G | z Eff/Gen NLL | W full6 NLL | terminal W−z full6 gap | actual/pred | P | common cap / BF16 energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1AGG | Neutral | 10/10, 19/20, 92/100 | 0.000275302 / 1.564606 | 10/10, 20/20 | 0.000188899 / 1.514061 | 0.000372093 | 0.000451630 | 0.915026 | 0.0217564 | 0.00581513 / 7.90255 |
| P1AGG | Soft | 10/10, 19/20, 92/100 | 0.000311923 / 1.617168 | 10/10, 20/20 | 0.000194061 / 1.550948 | 0.000355631 | 0.000566987 | 0.913837 | 0.0194440 | 0.00549193 / 7.35438 |
| P1DW | Neutral | 10/10, 19/20, 91/100 | 0.000292635 / 1.716608 | 10/10, 20/20 | 0.000204909 / 1.708746 | 0.000381919 | 0.000703252 | 0.884873 | 0.0228918 | 0.00801675 / 10.8497 |
| P1DW | Soft | 10/10, 19/20, 92/100 | 0.000240076 / 1.655709 | 10/10, 20/20 | 0.000196624 / 1.616983 | 0.000392131 | 0.000204425 | 0.894163 | 0.0213545 | 0.00767875 / 10.0431 |

### Qwen2.5-7B-Instruct

| weighting | arm | W E/G/L | W Eff/Gen NLL | z E/G | z Eff/Gen NLL | W full6 NLL | terminal W−z full6 gap | actual/pred | P | common cap / BF16 energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1AGG | Neutral | 10/10, 18/20, 92/100 | 0.00264664 / 2.860387 | 10/10, 18/20 | 0.00133219 / 2.791002 | 0.000754922 | 0.00116643 | 0.829065 | 0.723862 | 0.102018 / 21.4570 |
| P1AGG | Soft | 10/10, 19/20, 92/100 | 0.00292892 / 2.941042 | 10/10, 19/20 | 0.00113254 / 2.627010 | 0.000615723 | 0.00134048 | 0.846753 | 0.470459 | 0.0723055 / 13.2513 |
| P1DW | Neutral | 10/10, 18/20, 93/100 | 0.00280266 / 2.863287 | 10/10, 19/20 | 0.00114899 / 2.705231 | 0.000634468 | 0.00144075 | 0.836893 | 0.662270 | 0.103358 / 21.6309 |
| P1DW | Soft | 10/10, 19/20, 92/100 | 0.00250969 / 2.848138 | 10/10, 19/20 | 0.00129757 / 2.635972 | 0.000651329 | 0.000961690 | 0.859578 | 0.499861 | 0.0785811 / 14.4244 |

## 3. Deficit weighting 및 routing 관측

| model | weighting/arm | mean rho | mean deficit sum | mean effective requests | mean entropy / top1 | negative request-transition | fallback |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama | AGG-N | 1.59770 | 15.97697 | 10.000 | 1.36022 / 0.37149 | 0 | 0 |
| Llama | AGG-S | 1.59976 | 15.99761 | 10.000 | 1.28418 / 0.42966 | 2 | 0 |
| Llama | DW-N | 2.13673 | 15.37515 | 3.186 | 1.42132 / 0.33816 | 0 | 0 |
| Llama | DW-S | 2.12193 | 15.34719 | 3.445 | 1.36496 / 0.39109 | 0 | 0 |
| Qwen | AGG-N | 1.91704 | 19.17039 | 10.000 | 1.41212 / 0.33648 | 0 | 0 |
| Qwen | AGG-S | 1.87698 | 18.76981 | 10.000 | 1.19546 / 0.46047 | 0 | 0 |
| Qwen | DW-N | 2.27040 | 18.74747 | 5.449 | 1.35832 / 0.35467 | 0 | 0 |
| Qwen | DW-S | 2.29391 | 18.38550 | 5.609 | 1.23603 / 0.45121 | 0 | 0 |

각 step의 request별 `ellW`, `ellZ`, `d`, `omega`, actual progress, W−z gap 및 observation-only realization ratio `chi`는 `per-step.json`의 배열로 남겼다. deficit median/p90/max, 최대 omega, effective request count, 5개 slope/active mask/q/pi/v/hv도 64/64 row에 있다.

## 4. P1AGG → P1DW 짝 산술

아래 값은 `P1DW - P1AGG`이다.

| model | arm | ΔW E/G/L | ΔW Eff/Gen NLL | Δz Eff/Gen NLL | Δfull6 | Δterminal gap | Δpred / Δactual | Δrealization | ΔP | Δcommon cap / energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | Neutral | 0 / 0 / -1 | +0.00001733 / +0.152003 | +0.00001601 / +0.194685 | +0.00000983 | +0.00025162 | +4.31228 / +3.43042 | -0.030153 | +0.001135 | +0.002202 / +2.94717 |
| Llama | Soft | 0 / 0 / 0 | -0.00007185 / +0.038541 | +0.00000256 / +0.066035 | +0.00003650 | -0.00036256 | +4.17732 / +3.48342 | -0.019673 | +0.001910 | +0.002187 / +2.68869 |
| Qwen | Neutral | 0 / 0 / +1 | +0.00015602 / +0.002900 | -0.00018320 / -0.085771 | -0.00012045 | +0.00027432 | +2.82692 / +2.48588 | +0.007828 | -0.061592 | +0.001340 / +0.173878 |
| Qwen | Soft | 0 / 0 / 0 | -0.00041924 / -0.092904 | +0.00016503 / +0.008962 | +0.00003561 | -0.00037879 | +3.33541 / +3.05961 | +0.012825 | +0.029402 | +0.006276 / +1.17308 |

이 표는 case01의 직접 산술이며, B10x10 효과 추정값이 아니다.

## 5. DW Neutral → DW Soft 짝 산술

아래 값은 `P1DW-Soft - P1DW-Neutral`이다.

| model | ΔW E/G/L | ΔW Eff/Gen NLL | Δfull6 | Δterminal gap | Δpred / Δactual | Δrealization | ΔP | Δcommon cap / energy | Δentropy / top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | 0 / 0 / +1 | -0.00005256 / -0.060899 | +0.00001021 | -0.00049883 | -0.118447 / +0.052893 | +0.009290 | -0.001537 | -0.000338 / -0.806636 | -0.056355 / +0.052930 |
| Qwen | 0 / +1 / -1 | -0.00029297 / -0.015149 | +0.00001686 | -0.00047906 | +0.188025 / +0.573657 | +0.022685 | -0.162409 | -0.024777 / -7.20655 | -0.122298 / +0.096536 |

모든 Soft step의 strength residual은 위 기술 허용 범위 안이고, reduced-strength candidate와 fallback은 0이다.

## 6. P2 residual-transport 및 Official matched reference

P2R2 case01 raw-free report SHA는 `747365627c3c750d016335ec9fb7ae02c1bbdb480d0563a37b7d1c9c6f094606`이다. stream/order/evaluator/aggregator/request-order가 matched이며, 아래 P2R2 Soft는 `SOFTP`이다. `P1DW - P2R2` 직접 산술은 다음과 같다.

| model | arm | ΔW E/G/L | ΔW Eff/Gen NLL | Δfull6 | Δactual/pred |
|---|---|---:|---:|---:|---:|
| Llama | Neutral | 0 / 0 / -1 | +0.00005397 / +0.120012 | -0.00022132 | +0.280877 |
| Llama | Soft | 0 / 0 / 0 | +0.00000141 / +0.059112 | -0.00021111 | +0.290167 |
| Qwen | Neutral | 0 / +1 / 0 | +0.00171521 / -0.037301 | -0.00001594 | +0.411701 |
| Qwen | Soft | 0 / +2 / -1 | +0.00149889 / -0.076447 | +0.00001314 | +0.429966 |

P2R2 legacy capacity와 P2R7 common actual-BF16 capacity는 좌표가 다르므로 직접 산술하지 않았다.

Official AlphaEdit case01 matched reference는 Llama `E/G/L=10/19/92`, Eff NLL `0.0016578674`; Qwen `10/20/92`, Eff NLL `0.0073593140`이다. `P1DW - Official`은 다음과 같다.

| model | arm | ΔE/G/L | ΔEff NLL |
|---|---|---:|---:|
| Llama | Neutral | 0 / 0 / -1 | -0.00136523 |
| Llama | Soft | 0 / 0 / 0 | -0.00141779 |
| Qwen | Neutral | 0 / -2 / +1 | -0.00455666 |
| Qwen | Soft | 0 / -1 / 0 | -0.00484962 |

Official Gen/Loc continuous NLL·margin, Structural-P, common capacity, common BF16 energy 및 edit-core time은 `NOT_RECORDED`이다.

## 7. Compute ledger

- 전체 target forward/backward: `1320/960`; KL forward/backward: `1000/960`.
- physical capture forward: `360`; weighted response forward/backward: `320/320`; response batched VJP: `64`.
- post-write objective forward: `360`; writer materialization: `64`; candidate forward/materialization: `0/0`.
- weighted objective processed tokens: `63,680`.
- endpoint total wall: Llama AGG-N/S `151.63/155.01s`, DW-N/S `139.24/138.14s`; Qwen AGG-N/S `271.00/258.24s`, DW-N/S `162.30/207.06s`.
- array task wall: Llama `597.815s`, Qwen `915.125s`.

## 8. 분류

### FACT

- job20148은 2/2 task와 8/8 endpoint가 완결됐고 실패 endpoint는 0이다.
- 64 step 모두 5개 공유 writer 변수, full strength, residual 1/h 1회, physical h 1회, BF16 materialization 1회를 기록했다.
- DW effective request count 평균은 Llama N/S `3.186/3.445`, Qwen N/S `5.449/5.609`; AGG는 모두 10이다.
- Soft fallback은 0이며 Llama AGG-Soft에만 negative actual request-transition 2건이 기록됐다.

### INFERENCE / 계약 게이트

- 소표본 점수 차이는 파일럿에서 strict efficacy gate가 아니다.
- 기술적으로 유효한 파일럿이므로 production의 기계적 release 조건은 충족됐다.
- 이 판단은 P1DW 우월성, 일반화 성능, 승격 또는 후속 과학 결론을 뜻하지 않는다.

### TECHNICAL_FAIL

- TECH-R2 endpoint에는 typed technical failure가 없다.

### SCIENTIFIC_FAIL

- typed `SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION`과 `NO_SEMANTIC_DEFICIT`는 0건이다.
- 소표본 수치만으로 별도 scientific failure를 선언하지 않는다.

### NOT_RECORDED

- Official Gen/Loc continuous NLL·margin 및 common P/capacity/energy/edit-core.
- 별도 stepwise heldout E/G/L(계약상 0회).

## 9. 분석 산출물

- `per-endpoint.json`: 8 rows.
- `per-step.json`: 64 rows; request 배열 포함.
- `analysis-manifest.json`: 입력/출력 정체성과 파일 SHA.
- `analysis-receipt.json`: rooted receipt 및 gate 상태.

`scientific_promotion=false`.
