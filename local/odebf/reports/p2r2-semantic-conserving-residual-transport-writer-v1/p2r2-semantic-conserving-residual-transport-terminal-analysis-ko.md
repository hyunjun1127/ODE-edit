# P2R2 Semantic-Conserving Residual-Transport Writer 최종 분석

- 생성 시각(Asia/Seoul): `2026-08-15T17:39:25+09:00`
- source HEAD/tree: `c97e8619b42da7954ce0e824c215a8a82d70589a` / `711b33fc736641d1f3067c00bd067ed5e60606fe`
- exact P2R1 parent: `8f817e13167289dac190fe74bfa42e2b3e01372d`; P1R43 ancestor: `11508b6da11d606521b703037034e1814b70d8a8`
- contract SHA256: `f519e81fcc40870423774be0c507851332f604fde910dbcadad8faa73e0cf3f6`; stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- evaluator/aggregator: `25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145` / `64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0`
- 전체 분모: 40 attempts / 40 endpoints / 0 typed incomplete / 400 requests / 320 writer steps.
- scientific_promotion=false.

## 한 페이지 요약

| model | arm | z E/G | W E/G/L | z Eff NLL | W Eff NLL | z Gen NLL | W Gen NLL | W−z Eff/Gen NLL | actual/pred | P | capacity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 100/100, 189/200 | 100/100, 189/200, 841/1000 | 0.085484 | 0.103257 | 1.62397 | 1.64976 | 0.0177732 / 0.0257909 | 0.428253 | 0.0717322 | 44.1428 |
| llama3-8b-inst | SOFTP | 100/100, 189/200 | 100/100, 189/200, 842/1000 | 0.0917352 | 0.0920058 | 1.63709 | 1.65173 | 0.000270646 / 0.0146332 | 0.429855 | 0.0702831 | 43.2263 |
| qwen2.5-7b-inst | NEUTRAL | 100/100, 185/200 | 100/100, 185/200, 839/1000 | 0.00115011 | 0.00114072 | 2.2081 | 2.20401 | -9.38892e-06 / -0.00409376 | 0.357999 | 2.1869 | 188.79 |
| qwen2.5-7b-inst | SOFTP | 100/100, 185/200 | 100/100, 185/200, 839/1000 | 0.00113976 | 0.00113437 | 2.20948 | 2.20707 | -5.38349e-06 / -0.00241211 | 0.358377 | 2.18757 | 188.846 |

### Gate A–D

- Gate A: Eff z-inject는 P2R1 target-only와 네 cell 모두 100/100으로 동일했다. Gen z-inject는 Llama/Qwen 각 arm에서 P2R1 대비 −8/200이었다. ‘크게 잃지 않음’의 사전 수치 임계가 없어 binary state는 `NOT_BINARIZED_PREDECLARED_NUMERIC_THRESHOLD_ABSENT`이다.
- Gate B: 네 cell 모두 terminal z와 W의 Eff/Gen correct count가 동일했다. W−z NLL 평균과 내부 actual/predicted ratio는 표에 분리했다.
- Gate C: Llama Soft는 Neutral 대비 terminal P −0.00144911, capacity −0.916555이며 E/G count delta 0/0이다. Qwen Soft는 P +0.000672101, capacity +0.0560499이며 E/G count delta 0/0이다.
- Gate D: Official E/G/L count를 모두 동시에 충족한 arm은 0개다. 동시에 모든 cell의 terminal W E/G count가 current-W z E/G count와 같아서 사전 writer-aware-z branch 조건도 충족하지 않았다.
- 기계적 계약: endpoint integrity는 40/40이나 §11의 `outer_h_application_count=1`과 §17의 requestwise self-reachable `c_i` 필드가 없다. numerical lock의 `writer_h_application_count=0`, 320 receipt의 `physical_h_application_count=0`, joint materialization 320/320을 별도 기록했고 상태는 `MECHANICAL_CONTRACT_RECEIPT_FAIL`이다.

### 필수 14개 질문

1. P1R43 writer: execution AST gate와 320 receipt에서 legacy writer access/decision influence 합계 0이다.
2. Current-W target refresh: 각 case/arm에 8개 outer physical-state identity와 outer별 3 microstep 동일 W identity가 기록됐고 writer 뒤 1회 refresh가 320/320이다.
3. Residual 보존: 모든 step의 response shape은 10×50, 각 layer의 residual SHA는 step 내 동일, divisor는 1, active simplex residual 최대는 machine table에 기록됐다. request별 layer-sum 제약을 사용했다.
4. Neutral hard-request 보호: Neutral `NEUTRAL_CERTIFIED` 160/160, fairness lambda가 각 step에 기록됐고 unreachable request-step은 Llama 2, Qwen 3이다. unreachable은 별도 typed registry에 남았다.
5. Soft strength: strength attenuation 0, same-state no-weaker violation 최대는 1e-8 numerical tolerance 이내다. Soft는 Llama 10/80, Qwen 4/80 step에서 certified, 나머지는 Neutral fallback이다.
6. Structural-P 영향: exact paired same-state allocation 이동은 Llama 4/62, Qwen 2/69 step이다. terminal P delta는 Llama 음수, Qwen 양수다.
7. Barrier DOF: active request당 5-layer simplex의 형식 DOF는 4이며 step당 4×active-request 수다. actual same-state route movement와 status 분모는 위와 machine table에 기록했다.
8. z→W 소실: Eff/Gen correct 소실은 네 cell 모두 0이며 NLL gap은 요약표와 per-request table에 있다.
9. Subgroup: Llama clamp-hit request는 200/200 arm-request, Qwen은 0/200이다. Qwen ‘hard subgroup’의 별도 source threshold/registry는 `NOT_RECORDED`; request p90/max와 unreachable rows를 제공한다.
10. Official 대비: E/G/L count delta와 Eff NLL delta는 아래 표에 있다. Official P/capacity/update norm 및 edit-core는 `NOT_RECORDED`이다.
11. 병목 분리: P2R1 대비 current-W z Gen delta는 네 cell 모두 −8/200; current-W z→W E/G count delta는 0이다. Router attenuation은 0이고 Soft fallback은 146/160이다.
12. 다음 단계 gate: Official E/G/L count 동시충족 arm 0, writer-aware-z 조건 false로 `NEITHER_PREDECLARED_BRANCH_CONDITION_MET`이다.
13. Native edit-core ratio: Official edit-core seconds가 immutable matched reference에 없어 `NOT_RECORDED`이다.
14. Forbidden influence: legacy writer, candidate forward/materialization, debt, remaining division, hard-P budget, functional-P veto, retry/backtracking, heldout controller access는 모두 0이다. §11 h receipt gap은 별도 mechanical fail이다.

## P2R1 target-only 및 Official matched arithmetic

| model | arm | Δz Eff/G vs P2R1 | Δz Eff/Gen NLL vs P2R1 | ΔW E/G/L vs Official | ΔW Eff NLL vs Official |
|---|---|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 0 / -8 | -0.0496642 / 0.438055 | 0 / 4 / -18 | 0.102088 |
| llama3-8b-inst | SOFTP | 0 / -8 | -0.0434131 / 0.451175 | 0 / 4 / -17 | 0.0908368 |
| qwen2.5-7b-inst | NEUTRAL | 0 / -8 | 3.15309e-05 / 0.543384 | 0 / -7 / 11 | -0.0315053 |
| qwen2.5-7b-inst | SOFTP | 0 / -8 | 2.11763e-05 / 0.54476 | 0 / -7 / 11 | -0.0315116 |

- Exact P2R1 matched identity는 model/case request order, stream/order, evaluator/aggregator로 40/40 comparison rows에서 확인했다.
- Official identity SHA는 `e12b19ff941f39a594d2ff0f8718849d102644fd2ec79f53ccb6d7efc9fd8d3c`; 20/20 model×case가 matched이다.
- Official Gen/Loc continuous NLL·margin, Structural-P, capacity, update norm, edit-core time은 `NOT_RECORDED`이다.

## Neutral–Soft paired facts

| model | ΔE/G/L | ΔEff/Gen NLL | ΔP | Δcapacity | same-state moved/compared | Soft certified/fallback |
|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 0 / 0 / 1 | -0.0112514 / 0.00196201 | -0.00144911 | -0.916555 | 4/62 | 10/70 |
| qwen2.5-7b-inst | 0 / 0 / 0 | -6.34909e-06 / 0.00305773 | 0.000672101 | 0.0560499 | 2/69 | 4/76 |

## Compute 및 integrity

| model | arm | target/writer/total seconds(case mean) | target F/B | response VJP | writer mat | negative request-steps |
|---|---|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 26.5629 / 80.9842 / 123.453 | 1250/1200 | 400 | 80 | 40 |
| llama3-8b-inst | SOFTP | 29.0183 / 84.2911 / 129.372 | 1250/1200 | 400 | 80 | 40 |
| qwen2.5-7b-inst | NEUTRAL | 23.8607 / 102.716 / 143.762 | 1250/1200 | 400 | 80 | 9 |
| qwen2.5-7b-inst | SOFTP | 23.4914 / 108.295 / 149.192 | 1250/1200 | 400 | 80 | 10 |

- Scheduler submission receipt: array job `19944`, array `0-1%2`, source `c97e8619b42da7954ce0e824c215a8a82d70589a`; result terminal은 model별 20/20 arm-case complete다. 별도 scheduler terminal-state receipt는 `NOT_RECORDED`.
- Action freeze/W0 pointer/W0 bytes/evaluator firewall/K8/materialization8/target24는 각각 40/40이다.
- Retry/backtracking/candidate materialization/heldout decision influence는 0이다.

## FACT / INFERENCE / NOT_RECORDED / TECHNICAL_FAIL / SCIENTIFIC_FAIL

### FACT

- 위 표의 raw-free 값, exact identities, 40/40 endpoint 및 machine tables.
- P2R1 current-W z Gen count delta는 모든 arm에서 −8/200이며 current-W z→W E/G count delta는 0.
- Soft terminal P/capacity delta는 Llama에서 음수, Qwen에서 양수.

### INFERENCE

- Gate A의 ‘크게 잃지 않음’은 사전 수치 임계가 없어 binary 판정하지 않았다.
- Gate D는 contract branch 조건의 직접 산술만 적용했고 추가 권고를 만들지 않았다.

### NOT_RECORDED

- Official Gen/Loc continuous metrics, Official P/capacity/update norm/edit-core; Qwen hard-subgroup 별도 registry; requestwise self-reachable ceiling c_i; exact same-state internal Soft-vs-Neutral allocation for paired trajectories가 이미 갈라진 step; outer_h_application_count와 second_h_application_count.

### TECHNICAL_FAIL

- `MECHANICAL_CONTRACT_RECEIPT_FAIL`: contract §11 `outer_h_application_count=1` 및 §17 requestwise `c_i` 필드 부재. numerical lock `writer_h_application_count=0`, writer receipt `physical_h_application_count=0` (320/320), joint materialization 320/320.

### SCIENTIFIC_FAIL

- typed scientific failure case는 0/40. Gate A/C/D의 수치 상태는 별도 gate table에 기록했으며 scientific_promotion=false다.

## 산출물

- `p2r2-per-case.jsonl`: 40 rows
- `p2r2-per-step.jsonl`: 320 rows
- `p2r2-per-request.jsonl`: 400 rows
- `p2r2-per-step-request.jsonl`: 3,200 rows
- `p2r2-response-routing.jsonl`: 320 rows
- `p2r2-comparisons.jsonl`: 180 rows
- `p2r2-aggregates.json`, `p2r2-comparison-aggregates.json`, `p2r2-gates.json`
- `analysis-manifest.json`, `analysis-receipt.json`, `independent-verification.json`, `final-package-manifest.json`, `final-package-receipt.json`
