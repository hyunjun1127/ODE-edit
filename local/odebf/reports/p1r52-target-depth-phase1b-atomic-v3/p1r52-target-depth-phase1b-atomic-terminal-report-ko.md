# P1R52 Target Depth Phase 1B Atomic 사실 보고서

- 생성 시각(KST): `2026-08-20T15:00:53+09:00`
- source HEAD/tree: `98036fe23a71bd760384d65e3fb5da9d87f717bb` / `cc307ef381d03e621bebb3dab819284362750f6d`
- contract SHA256: `06e65d4a2df4a4610ef09818c2afd96b8dbe86cc3041afdbc5b099ca2358ec63`
- Slurm: `21357`; tasks `4/4 COMPLETED0`
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- attempts/endpoints/typed scientific failures/technical failures: `40/35/5/0`
- history/retry/imputation: `0/0/0`; endpoint 및 typed prefix W0 restore: `40/40`
- 범위: P1R52 J0 IL3-FULL Atomic; Llama/Qwen × Neutral/Soft; IL1은 immutable Repair-R1 결과 재사용

## IL3-FULL terminal 집계

| model | arm | endpoints | z/W/gap full6 NLL | z E/G | W E/G/Loc | predicted/actual/ratio | P/cap/energy | neg | F/B(valid)/mat(all-prefix) | edit-core(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | neutral | 10/10 | 0.007081/0.039751/0.032670 | 100/182 | 99/180/870 | 113.503/103.487/0.912 | 0.012186/0.278030/0.441970 | 3 | 4875/2850/80 | 118.75 |
| llama3-8b-inst | soft | 10/10 | 0.006853/0.008608/0.001755 | 100/181 | 100/181/874 | 113.079/103.799/0.918 | 0.010480/0.241372/0.400024 | 0 | 4925/2850/80 | 122.68 |
| qwen2.5-7b-inst | neutral | 7/10 | 0.004977/0.010115/0.005138 | 70/120 | 70/119/585 | 119.447/73.871/0.618 | 0.620154/9.267335/3.602691 | 1 | 3410/1995/75 | 132.08 |
| qwen2.5-7b-inst | soft | 8/10 | 0.021431/0.021632/0.000202 | 80/132 | 80/132/678 | 115.356/85.228/0.739 | 0.413206/3.674902/1.988810 | 1 | 4035/2280/77 | 133.18 |

## IL3-FULL − immutable IL1

| model | arm | Δz NLL | ΔW NLL | Δgap | Δz E/G | ΔW E/G/Loc | ΔP/cap/energy | endpoint gate | Phase2 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| llama3-8b-inst | neutral | -0.039690 | -0.058600 | -0.018910 | 0/-2 | -1/0/-2 | 0.008550/0.180006/0.182303 | zNLL=True, WNLL=True, zcount=False, Wcount=False, finite=True | STOP |
| llama3-8b-inst | soft | -0.034624 | -0.070672 | -0.036049 | 0/-2 | 0/0/3 | 0.007002/0.157474/0.170791 | zNLL=True, WNLL=True, zcount=False, Wcount=True, finite=True | STOP |
| qwen2.5-7b-inst | neutral | -0.450951 | -0.571457 | -0.120505 | UNMATCHED_DENOMINATOR | UNMATCHED_DENOMINATOR | 0.480673/7.786285/2.076891 | zNLL=False, WNLL=False, zcount=False, Wcount=False, finite=True | STOP |
| qwen2.5-7b-inst | soft | -0.453231 | -0.531875 | -0.078644 | UNMATCHED_DENOMINATOR | UNMATCHED_DENOMINATOR | 0.331148/3.197990/1.215196 | zNLL=False, WNLL=False, zcount=False, Wcount=False, finite=True | STOP |

## Official AlphaEdit 경계

- exact same stream/order/evaluator identity는 P1R31 raw-free package에서 확인됐다.
- Official endpoint: Llama `100/185/859`, efficacy NLL `0.00117`; Qwen `100/192/828`, efficacy NLL `0.03265`.
- Official latent-z trajectory, W8 full-six, Gen continuous NLL/margin, P/capacity/energy: `NOT_RECORDED`.
- Native/Official baseline은 이번 작업에서 재실행하지 않았다.

## accepted-z와 writer-W rewrite/rephrase

성공률은 pinned evaluator의 `margin>0` prompt bit이다. strict Gen은 한 request의 rephrase prompt가 모두 성공한 request 수다. 별도 accuracy 필드는 `NOT_RECORDED`다.

| model | arm | valid/attempt | z rewrite success/rate | z rewrite NLL mean/med/p90 | z rephrase success/rate; strict | z rephrase NLL mean/med/p90 | W rewrite success/rate | W rewrite NLL mean/med/p90 | W rephrase success/rate; strict | W rephrase NLL mean/med/p90 | W-z rewrite/rephrase gap mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | neutral | 10/10 | 100/100 (1.0000) | 0.004278/0.002693/0.007681 | 182/200 (0.9100); 85/100 | 2.003152/1.200867/4.283594 | 99/100 (0.9900) | 0.069284/0.003601/0.015240 | 180/200 (0.9000); 84/100 | 2.125322/1.275253/4.613281 | 0.065005/0.122171 |
| llama3-8b-inst | soft | 10/10 | 100/100 (1.0000) | 0.004318/0.002777/0.007849 | 181/200 (0.9050); 84/100 | 2.044944/1.184509/4.314453 | 100/100 (1.0000) | 0.005653/0.003204/0.009406 | 181/200 (0.9050); 84/100 | 2.091738/1.196106/4.370117 | 0.001336/0.046794 |
| qwen2.5-7b-inst | neutral | 7/10 | 70/70 (1.0000) | 0.008062/0.006866/0.018359 | 120/140 (0.8571); 53/70 | 3.088932/2.376771/6.904688 | 70/70 (1.0000) | 0.020687/0.008545/0.023376 | 119/140 (0.8500); 52/70 | 3.127703/2.694259/6.970313 | 0.012625/0.038772 |
| qwen2.5-7b-inst | soft | 8/10 | 80/80 (1.0000) | 0.011880/0.005768/0.014673 | 132/160 (0.8250); 56/80 | 3.339139/2.730469/7.270313 | 80/80 (1.0000) | 0.013103/0.006973/0.015881 | 132/160 (0.8250); 56/80 | 3.366532/2.754395/7.398438 | 0.001223/0.027393 |

- request-level W−z rewrite/rephrase NLL gap과 case-level gap은 `per-request`/`per-case` JSON·CSV에 기록했다.
- IL1의 연속 rewrite/rephrase NLL table은 server2 입력에 없어 IL3−IL1 해당 delta는 `NOT_RECORDED_LOCALLY`; full-six z/W NLL과 E/G counts만 exact same-stream aggregate로 비교했다.

## IL1 target gain의 writer 전달

| model | arm | IL1−IL3 z full6 gain | IL1−IL3 W full6 gain | arithmetic transfer ratio | IL3 writer predicted/actual/realization | endpoint denominator |
|---|---|---:|---:|---:|---:|---:|
| llama3-8b-inst | neutral | 0.039690 | 0.058600 | 1.476457 | 113.503/103.487/0.912 | 10/10 |
| llama3-8b-inst | soft | 0.034624 | 0.070672 | 2.041162 | 113.079/103.799/0.918 | 10/10 |
| qwen2.5-7b-inst | neutral | 0.450951 | 0.571457 | NOT_RECORDED | 119.447/73.871/0.618 | 7/10 |
| qwen2.5-7b-inst | soft | 0.453231 | 0.531875 | NOT_RECORDED | 115.356/85.228/0.739 | 8/10 |

- transfer ratio는 `(IL1 W full6 NLL − IL3 W full6 NLL)/(IL1 z full6 NLL − IL3 z full6 NLL)`의 산술값이며 full 10-case denominator일 때만 기록한다.
- Qwen은 typed scientific failure로 denominator가 달라 ratio를 `NOT_RECORDED`로 둔다.

## Typed scientific boundaries

- Qwen `P1R34NonSemanticTargetMove`: `5` cases; technical failure/retry/imputation `0/0/0`.
- 각 failure는 accepted prefix와 W0 byte/pointer restore를 보존했고 독립 후속 case가 계속 실행됐다.
- Llama typed failure `0/20`; Qwen endpoints `15/20`.

## 기계적 불변식·계산

- accepted steps/request endpoints: `312/350`.
- 모든 accepted outer에서 inner count `3`, inner writer/materialization/heldout `0/0/0`, outer writer materialization expected `1`.
- completed endpoint당 K8/materialization8; history/retry/backtracking `0`; target controller calibration `1/case`.
- 표의 F/B/token은 valid endpoint compute ledger만 합산했다. failed prefix F/B/token은 root receipt에 없어 `NOT_RECORDED`; accepted materialization은 prefix 포함 `312`다.
- IL3 uses inner h=1/8; h/3, IL8, IL25, inner rho summation, remaining division, debt counts `0`.
- exact per-case/per-step/per-request facts and source/result identities are separate tables.

## Phase 2 gate

- eligible combinations: `[]`.
- 계약의 7개 paired/lenient 조건을 개별 기록했고 새 threshold는 `0`이다.
- Llama Soft는 조건 1/2/3/4/7이 PASS지만, IL1 cost tail table과 Phase1B clamp/CURRENT/reference-energy 전체 분포가 server2에 없어 조건 5/6이 `NOT_RECORDED`; 따라서 Phase2를 열지 않았다.
- Llama Neutral은 W Eff가 100→99로 조건 3이 FAIL이고 조건 5/6은 `NOT_RECORDED`다.
- Qwen has unmatched endpoint denominators and is not eligible.
- IL1 case-level table bytes were not present on server2; IL3−IL1 arithmetic is exact same-stream cell aggregate, and per-case IL1 deltas are `NOT_RECORDED` locally.
- scientific_promotion: `false`.

### 7-condition detail

| model | arm | 1 target NLL | 2 heldout signal | 3 W Eff/Gen | 4 gap/realization | 5 preservation/cost tail | 6 clamp/CURRENT/ref-energy | 7 endpoint validity |
|---|---|---|---|---|---|---|---|---|
| llama3-8b-inst | neutral | True | True | False | True | NOT_RECORDED_IL1_COST_TAIL_TABLE_NOT_LOCAL | NOT_RECORDED_PHASE1B_FULL_CLAMP_CURRENT_REFERENCE_ENERGY_DISTRIBUTION | True |
| llama3-8b-inst | soft | True | True | True | True | NOT_RECORDED_IL1_COST_TAIL_TABLE_NOT_LOCAL | NOT_RECORDED_PHASE1B_FULL_CLAMP_CURRENT_REFERENCE_ENERGY_DISTRIBUTION | True |
| qwen2.5-7b-inst | neutral | False | True | False | False | NOT_RECORDED_IL1_COST_TAIL_TABLE_NOT_LOCAL | NOT_RECORDED_PHASE1B_FULL_CLAMP_CURRENT_REFERENCE_ENERGY_DISTRIBUTION | False |
| qwen2.5-7b-inst | soft | False | True | False | False | NOT_RECORDED_IL1_COST_TAIL_TABLE_NOT_LOCAL | NOT_RECORDED_PHASE1B_FULL_CLAMP_CURRENT_REFERENCE_ENERGY_DISTRIBUTION | False |
