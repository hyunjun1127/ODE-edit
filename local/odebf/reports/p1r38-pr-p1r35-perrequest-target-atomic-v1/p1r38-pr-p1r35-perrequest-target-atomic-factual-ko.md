# P1R38 PR-P1R35 per-request target Atomic 결과 (사실 기록)

- instruction: `ODEEDIT-S05-P1R38-PR-P1R35-PERREQUEST-TARGET-ATOMIC-V1`
- source: `6f48ac2800b257ceb16368fff5137212dfa6037f` (scientific parent `a625e3d1cded3ced0e9128ef7a44205953041447`)
- scheduler: job `19566`; 4/4 tasks `COMPLETED(0)`
- attempts/endpoints/failures: 40/40/0; accepted K steps: 320/320
- stream root/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- W0 pointer/byte restore: 40/40; controller reset: 40/40; cross-case state: 0
- scientific_promotion: `false` (administrative boundary)

## 실행 경계

- B1 attempt1 job `19556`: 4개 task scheduler `COMPLETED(0)`; case-level technical failure 4/4, accepted transition 0, W0 restore 4/4.
- attempt1 exception SHA prefix: `08e3dc`; source-backed message: `P1R23 objective microbatch size differs` (B1 request_count=1, inherited production microbatch=2).
- technical child: microbatch를 `min(configured, request_count)`로 제한; scientific/numerical/controller 변경 0.
- B1 replacement job `19562`: 4/4 terminal PASS, K8/W0 restore, selection added F/B 0/0.
- production job `19566`: 4/4 `COMPLETED(0)`; 40/40 case endpoints.

## 셀별 집계

| model | arm | E | G | Loc | z8 full6 NLL mean | W8 full6 NLL mean | W-z gap | held | reactivated | rejected | NSTM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 99/100 | 186/200 | 827/1000 | 0.418843 | 0.628985 | 0.210143 | 0 | 0 | 4 | 0 |
| llama3-8b-inst | Soft | 99/100 | 184/200 | 827/1000 | 0.416417 | 0.552950 | 0.136533 | 0 | 0 | 5 | 0 |
| qwen2.5-7b-inst | Neutral | 98/100 | 173/200 | 842/1000 | 0.321769 | 0.408479 | 0.086710 | 31 | 0 | 24 | 0 |
| qwen2.5-7b-inst | Soft | 98/100 | 175/200 | 842/1000 | 0.297659 | 0.388452 | 0.090793 | 34 | 1 | 28 | 0 |

## target hard-tail 및 계산

| model | arm | terminal mean median | p90 | worst | count<.05 | F | B | tokens | materializations | edit-core s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0.101013 | 0.975970 | 8.713247 | 31 | 2200 | 1250 | 371778 | 80 | 992.914 |
| llama3-8b-inst | Soft | 0.102380 | 0.916720 | 8.692571 | 31 | 2200 | 1250 | 371778 | 80 | 1083.881 |
| qwen2.5-7b-inst | Neutral | 0.024113 | 0.558396 | 8.658664 | 73 | 2200 | 1250 | 340727 | 80 | 1361.779 |
| qwen2.5-7b-inst | Soft | 0.025291 | 0.478322 | 8.775531 | 68 | 2200 | 1250 | 340727 | 80 | 1448.945 |

## P1R36 RS 동일 batch 산술 차이 (P1R38 - P1R36)

| model | arm | paired endpoints | ΔE | ΔG | ΔLoc | mean ΔEff target-new NLL |
|---|---|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 10/10 | -1 | 5 | -48 | 0.458890 |
| llama3-8b-inst | Soft | 8/10 | -1 | 2 | -45 | 0.420903 |
| qwen2.5-7b-inst | Neutral | 8/10 | 2 | 13 | -2 | -0.028248 |
| qwen2.5-7b-inst | Soft | 9/10 | -1 | 16 | -1 | 0.111314 |

## 지정 sentinel: Llama / Soft / case02 / k7

- selected target NLL mean/median/p90/worst: 0.212118 / 0.049791 / 0.482696 / 0.916137
- active/held/reactivated/rejected: 10/0/0/0
- finite demand/predicted: 0.542382/0.542382; actual at k7: NOT_RECORDED (delayed receipt covers transitions1..7)
- terminal E/G/Loc: 10/19/85

## Official AlphaEdit direct-z matched reference

- exact same-stream identity: 20/20 model×batch references matched in immutable P1R31 identity receipt.

| model | P1R38 arm | P1R38 z8 full6 NLL mean | Official direct-z Eff NLL mean | arithmetic difference | Official E/G/Loc |
|---|---|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0.418843 | 0.001169 | 0.417674 | 100/185/859 |
| llama3-8b-inst | Soft | 0.416417 | 0.001169 | 0.415248 | 100/185/859 |
| qwen2.5-7b-inst | Neutral | 0.321769 | 0.032646 | 0.289123 | 100/192/828 |
| qwen2.5-7b-inst | Soft | 0.297659 | 0.032646 | 0.265013 | 100/192/828 |

- per-batch Official Eff/Gen/Loc, Eff target-new NLL/margin and P1R38 arithmetic gaps are in `p1r38-per-case.json`.
- standalone latent z trajectory and direct-z diagnostic runtime: `NOT_RECORDED`; no duplicate direct-z run was executed.

## 불변식

- inner heldout evaluation: 0/320; selection added F/B: 0/0; per-request Python model calls: 0.
- persistent/carried mask decision influence: 0/0; remaining-horizon divisions: 0; physical h/second-h: 320/0.
- history/H/retry/backtracking: 0; terminal evaluator ran after action freeze.
- stepwise heldout Eff/Gen/Loc: `NOT_RECORDED`.

## 산출물

- `p1r38-per-request.json/csv`: 3,200 request-step rows
- `p1r38-per-step.json/csv`: 320 accepted-step rows
- `p1r38-per-case.json/csv`: 40 case rows
- values are raw-free telemetry, denominators, identities and arithmetic deltas only.
