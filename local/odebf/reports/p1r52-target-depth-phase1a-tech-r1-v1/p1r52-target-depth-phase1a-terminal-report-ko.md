# P1R52 Target Depth Phase 1A 사실 보고서

- 생성 시각(KST): `2026-08-20T13:57:17+09:00`
- instruction: `ODEEDIT-S05-P1R52-TARGET-DEPTH-IL1-IL3FULL-V1`
- source HEAD/tree: `d05fe39066c9450fa9f8521e0cae1174cc4fd383` / `c7c2cebf781d1a2c38a1d5078a959604aed6ec30`
- contract SHA256: `06e65d4a2df4a4610ef09818c2afd96b8dbe86cc3041afdbc5b099ca2358ec63`
- Slurm production job: `21328` (array 0–3)
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- 범위: frozen-W target-only, Llama/Qwen × IL1/IL3-FULL, 셀당 10개의 독립 B10
- 완결성: cases `40/40`, requests `400/400`, inner updates `640`, request×inner rows `6400`
- writer update/materialization: `0/0`; W0 restore: `40/40`

## 집계

| model | depth | full-six NLL mean | request NLL median/p90/max | NLL≥.05 | z-Eff | z-Gen | rewrite NLL | rephrase NLL | inner path | net displacement | target wall(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | IL1 | 0.03767551 | 0.028168/0.069136/0.252464 | 24/100 | 100/100 | 197/200 | 0.029526 | 1.679125 | 17.9089 | 8.3392 | 31.74 |
| llama3-8b-inst | IL3-FULL | 0.00616285 | 0.004622/0.010543/0.029432 | 0/100 | 100/100 | 195/200 | 0.005410 | 1.552012 | 19.1354 | 8.4907 | 55.71 |
| qwen2.5-7b-inst | IL1 | 0.45850230 | 0.031609/0.222531/11.476061 | 34/100 | 97/100 | 170/200 | 0.538902 | 3.270028 | 191.3934 | 86.9067 | 31.76 |
| qwen2.5-7b-inst | IL3-FULL | 0.31514330 | 0.004522/0.009605/11.476061 | 6/100 | 97/100 | 172/200 | 0.337545 | 3.047193 | 210.9148 | 89.3816 | 55.22 |

## IL3-FULL − IL1 짝지은 차이

### llama3-8b-inst

- terminal request NLL delta mean/median/p90/max: `-0.03151266` / `-0.02335112` / `-0.00619670` / `-0.00004699`
- rewrite NLL delta mean: `-0.02411610`
- rephrase NLL delta mean: `-0.12711342`
- train NLL lower / heldout positive / hard-tail positive: `True` / `True` / `True`
- reference-energy nonfinite / all-CURRENT / all-clamp: `0` / `False` / `False`
- model gate: `PASS`

### qwen2.5-7b-inst

- terminal request NLL delta mean/median/p90/max: `-0.14335900` / `-0.02350818` / `-0.00500794` / `0.00000000`
- rewrite NLL delta mean: `-0.20135691`
- rephrase NLL delta mean: `-0.22283531`
- train NLL lower / heldout positive / hard-tail positive: `True` / `True` / `True`
- reference-energy nonfinite / all-CURRENT / all-clamp: `0` / `False` / `False`
- model gate: `PASS`

## 기계적 게이트

- Phase 1A gate: `PASS`
- Phase 1B contract continuation: `True`
- new threshold count: `0`
- scientific_promotion: `false`

## 불변식·계산

- inner writer/materialization/heldout access: `0/0/0` across `640` updates
- entry norm calibration: `1/case`, total `40`
- physical W mutation: `0`; W0 restoration: `40/40`
- IL3-FULL uses `T=3`, inner `h=1/8`; h/3, IL8, IL25 counts: `0/0/0`
- exact per-case/per-inner/per-request metrics and compute counters are in the attached tables.

## 기록 경계

- Phase 1A has no W-only endpoint, writer realization, capacity, or update-energy values: `NOT_RECORDED` by target-only design.
- Official Native and atomic J0 metrics are not mixed into this target-only report.
