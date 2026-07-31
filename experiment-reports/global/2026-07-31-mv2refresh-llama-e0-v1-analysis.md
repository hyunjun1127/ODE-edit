# MV-2 Llama refresh 독립 분석 — 기술 차단

- run: `mv2refresh_llama_e0_v1`
- model: `llama3-8b-inst`
- 사전등록: `plans/global/2026-07-31-session01-mv2-implementation-spec.md`
- 분석기: `project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py`
- 결론: `BLOCK_TECHNICAL_INVALID` — 잠금 분석기가 입력 schema의 exact key/order 계약에서 fail-closed로 종료했으므로 scientific effect 판정은 생성되지 않았다.

## 잠금 분석 실행과 기술 유효성

오프라인 실행은 다음 고정 CLI로 수행했다.

```text
uv run --offline --no-sync python project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py \
  --input-jsonl local/results/raw/session01_motivation/mv2refresh_llama_e0_v1/analysis_cases.jsonl \
  --model-alias llama3-8b-inst \
  --output-json experiment-reports/global/2026-07-31-mv2refresh-llama-e0-v1.analysis.summary.json \
  --output-report experiment-reports/global/2026-07-31-mv2refresh-llama-e0-v1-analysis.md
```

종료 코드는 `2`였고, analyzer result object는 생성되지 않았다. 첫 record에서의 오류는 다음과 같다.

```text
records[0].technical: exact key/order 위반
expected: exact_panel, lineage_exact, matched_second_c, rollback_exact, firewall_pass, receipt_before_outcome
actual:   exact_panel, firewall_pass, lineage_exact, matched_second_c, receipt_before_outcome, rollback_exact
```

동일한 actual 순서는 12개 case 모두에서 확인됐다. 이는 Boolean 값의 pass 주장과 별개인 입력 계약 위반이다. raw의 사후 key 재정렬·정규화, threshold 변경, retune, 부분-arm rescue 또는 outcome 기반 재분석은 수행하지 않았다. 따라서 이 보고서의 `BLOCK_TECHNICAL_INVALID`은 사전등록 technical-failure 우선순위에 따른 차단 disposition이며, analyzer가 정상 반환한 scientific verdict는 아니다.

## 고정 estimand·불확실성 상태

`A=refreshed_direction_refreshed_coefficient`, `B=fixed_direction_refreshed_coefficient`, `C=fixed_direction_fixed_coefficient`이다. 분석기가 효과 계산 전 중단했으므로 아래 값은 모두 미산출이다.

| 항목 | 사전등록 정의 | 상태 |
| --- | --- | --- |
| `Delta_direction` | `A-B` | 기술 차단 후 미산출 |
| `Delta_coefficient` | `B-C` | 기술 차단 후 미산출 |
| `Delta_total` | `A-C` | 기술 차단 후 미산출 |
| `e_m` | `max_i(max(1e-4, abs(no_op), abs(h0_sham-no_op)))` | 기술 차단 후 미산출 |
| paired bootstrap mean 95% CI | seed `20260801`, 4,000 paired resamples | 기술 차단 후 미산출 |
| refresh opportunity oracle | `max(A,B,C)-C` | 기술 차단 후 미산출; outcome-selected upper bound이므로 method 성능이 아님 |
| generic continuation gain | `max(A,B,C)-partial_joint` | 기술 차단 후 미산출; refresh kill을 구제하지 않음 |

사전등록상 `Delta_direction`만 primary이고, `Delta_coefficient`는 conditional pivot estimand이며, `Delta_total`은 secondary current-method 기대효과다. CI는 descriptive uncertainty이지 gate를 뒤집는 수단이 아니다.

## 실행·NFE·비용 메타데이터

다음은 생산자 `summary.json`의 선언값 또는 사전등록 비용표다. 입력 schema가 분석기에 거부됐으므로 효과 결과 또는 technical pass의 독립 검증 결과로 사용하지 않는다.

| 항목 | 값 |
| --- | ---: |
| attempted / pass / failure cases | `12` / `12` / `0` |
| controlled NFE / case | `43` |
| proposal builds / case | `4` |
| probe panels / case | `3` |
| run wall seconds | `8713.666184685193` |
| GPU peak allocated bytes | `40718852608` |
| GPU peak reserved bytes | `43526389760` |
| host max RSS KiB | `10761128` |
| visible GPU count | `1` |

사전등록 W1 controller의 incremental cost/case(공통 W0 routing과 diagnostic outcome forward 제외)는 다음과 같다.

| policy | proposal build | allowed central-probe NFE |
| --- | ---: | ---: |
| A: refreshed direction + refreshed coefficient | `1` | `12` |
| B: fixed direction + refreshed coefficient | `0` | `12` |
| C: fixed direction + fixed coefficient | `0` | `0` |

Slurm job step `15610.0`의 기술 메타데이터는 `COMPLETED`, exit `0:0`, node `devbox`, allocated TRES `cpu=8,gres/gpu=1,mem=65000M,node=1`, elapsed `02:25:16`, total CPU `02:32:53`, MaxRSS `10196536K`, AveRSS `10196536K`이다. 이는 run summary의 host RSS 및 GPU peak allocated/reserved와 별도의 계측 원천이다.

## Provenance 및 네 범주

- provenance ID: `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b`
- ODE-Edit commit: `cdc70db327ff264442cf26641c898bdefed3eaf3`; manifest의 tracked worktree clean: `true`
- analysis stream SHA-256: `c5c698ee7828000e4ecc0c5c15b86f0dca9ba014dbb5b5f678c59f5ab7d23e38`
- manifest SHA-256: `ad771d07c8980883a534e1b58719c7ee360ed231fdce4a9ac8d407b5d31b0c03`
- selection manifest ID: `47bb14e437ce193397dd4650a1bb2e98d5f48ba10821dc24c3e2723712741b66`; frozen rank slice `[100:112]`, `12` cases

| 범주 | 기록 |
| --- | --- |
| Proposal에서 온 내용 | fixed direct-z guide 아래 partial update 뒤 state-dependent non-stationarity를 검정하는 최소 mechanism diagnostic이다. |
| Repo/protocol에서 확인한 사실 | MV-2는 model별 12 case, 고정 six-arm order, fixed bootstrap 및 technical fail-closed analyzer 계약이다. |
| GH 추정 | equal-C에서 refreshed direction 대 fixed W0 direction 비교가 ODE/relinearization의 최소 신호이고, coefficient-only라면 fixed-direction dynamic coefficient 방향이다. |
| 사용자 확인 필요 | 별도 확인 없음. Llama/Qwen 동시 MV-2 실행 지시와 사전등록 gate를 그대로 따른다. |

## Result firewall 및 no-peer broadcast 예외

manifest의 information firewall은 canonical request와 allowed rewrite contexts만 허용하고 evaluation prompts, generations, weights, logits, activations를 제외한다. 이 analyst는 허용된 Llama raw 3종, 사전등록/프로토콜, 잠금 analyzer와 직접 test, Slurm `15610.0` 기술 메타데이터만 사용했다. Qwen directory, pair audit, 다른 model report/summary 및 peer 결과는 읽거나 사용하지 않았다.

no-peer broadcast 예외를 기록한다. 사전등록된 per-model 독립 분석/result firewall 때문에 이 analyst는 peer raw·report broadcast를 요청하거나 수행하지 않았다. 이는 server-head의 ordinary `local/` artifact broadcast 의무를 면제하거나 그 수행 상태를 검증한 기록은 아니다.

## 해석 경계

`A-C`는 teacher-forced rewrite utility의 absolute progress일 뿐 accuracy, retention, full-method 성능 또는 paper claim이 아니다. 이 run은 locked analyzer가 schema-rejected했으므로 `A-C` 자체도 산출되지 않았다. 본 보고서는 MV-2의 성공, ODE-Edit 우위, locality/retention 개선, 또는 MV-3 진입을 주장하지 않는다.
