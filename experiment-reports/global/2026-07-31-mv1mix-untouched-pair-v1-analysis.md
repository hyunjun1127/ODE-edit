# MV-1 untouched pair 통합 분석

- 작성일: 2026-07-31 KST
- 실행: Slurm job `15586`, `odeedit_mv1mix_untouched_pair_v1`
- 실행 commit: `a7c929f11da540c7435f9b0f3d367c5009c1ff2d`
- 판정 규칙: `plans/global/2026-07-30-session01-mv1-implementation-spec.md` §10.9
- 연구 맥락: `project/proposals/00.proposal`
- 상세 red audit: `audits/global/2026-07-31-mv1mix-untouched-pair-v1.postrun.md`
- claim boundary: 고정된 MV-1 untouched 20-case pair의 aggregate replication
  판정이다. ODE-Edit 방법 효과, MV-2 결과, 일반화, novelty 또는 paper claim이
  아니다.

## 결론

Locked §10.9 rule을 threshold 변경 없이 적용하면 두 모델 모두
`clear input`이다. Llama와 Qwen 모두 primary mean이 replay envelope
`1e-12`보다 크고, 20% trimmed mean과 median이 같은 양의 방향이며 positive
sign이 `20/20`이다. 따라서 MV-1 primary direction은 fixed-model
untouched pair에서 재현되었고, **사전에 고정된 최소 MV-2 pair 한 건만
실행할 수 있다.**

이 결론은 다음을 뜻하지 않는다.

- MV-2의 stale-vs-refreshed 가설이 이미 성립했다는 뜻이 아니다.
- 현재 controller가 ODE-Edit 전체 방법보다 우수하다는 뜻이 아니다.
- CounterFact 20-case, 두 fixed model 밖의 성능이나 일반화를 주장하지 않는다.
- proposal을 최종 paper plan 또는 검증된 claim으로 승격하지 않는다.

## 입력 방화벽

이번 pair red 분석은 아래 compact summary 두 개만 실험 수치 입력으로
사용했다.

- `experiment-reports/global/2026-07-31-mv1mix-llama-untouched-v1.analysis.summary.json`
- `experiment-reports/global/2026-07-31-mv1mix-qwen-untouched-v1.analysis.summary.json`

규칙 입력은 §10.9와 `PROTOCOL.md`, 실행 종료 확인은 `sacct -j 15586`의
technical metadata로 제한했다. `local/results/`, full analysis JSON,
case-level diagnostic, C1/D1 report 및 raw outcome은 읽지 않았다.

## 네 범주

| 범주 | 내용 |
| --- | --- |
| proposal에서 온 내용 | 이번 red 분석에서는 proposal 원문을 실험 근거로 읽지 않았다. Proposal은 연구 출발점일 뿐 최종 claim이 아니며, 여기서는 §10.9의 사전 고정 규칙만 판정에 사용했다. |
| repo/protocol에서 확인한 사실 | §10.9는 untouched exact 20, seed `17`, `q=1/256`, bootstrap seed `20260731`/`4000`, 두 모델 별도 분석 후 pair red 판정을 고정한다. 두 compact summary는 아래 aggregate와 artifact gate를 보고하며 pair/MV-2 결정을 자체 계산하지 않았다. Slurm pair와 두 child는 모두 `COMPLETED`, `ExitCode=0:0`이다. |
| GH 추정 | Forecast는 두 모델 모두 양의 방향을 맞혔지만 크기 보정은 특히 Qwen에서 과대였다. 따라서 MV-1은 routing direction의 replication으로만 해석하고, 최소 MV-2에서 refresh mechanism을 새로 검증해야 한다. |
| 사용자 확인 필요 | Locked 최소 MV-2 pair 한 건을 여는 데 추가 확인은 필요하지 않다. 그보다 큰 실행, threshold/policy 변경, MV-3, paper claim 또는 일반화 주장은 별도 결정이 필요하다. |

## Technical gate

### 실행 및 provenance

| 항목 | Llama | Qwen | pair 판정 |
| --- | --- | --- | --- |
| schema | `ode-edit-mv1-untouched-single-model-compact-summary/v1` | `ode-edit-mv1-score-mix-untouched-compact-summary/v1` | key layout은 다르지만 공통 gate field의 의미는 호환 |
| compact file SHA-256 | `cb1a920e66e153e3a302d0f0e1682c21f03ba0bae6ba0489ebda80162653632d` | `69f6372abde5b22a1991cef4a029c26b4af395f1c139c1720127334cf11bd3fb` | 현재 허용 입력 고정 |
| run/model | `mv1mix_llama_untouched_v1` / `llama3-8b-inst` | `mv1mix_qwen_untouched_v1` / `qwen2.5-7b-inst` | model과 run은 서로 distinct |
| execution commit | `a7c929f…` | `a7c929f…` | exact match |
| verification head | `51d8597…` | `51d8597…` | exact match |
| analyzer unchanged | `true` | `true` | PASS |
| job | `15586`, paired job | `15586`, paired job | exact match |
| single analyzer pair decision | `false` | `false` | 독립성 PASS |
| single analyzer MV-2 decision | `false` | `false` | 독립성 PASS |

두 summary에 내장된 analysis/run-summary/manifest hash receipt는 모두
40/64자리 lowercase hexadecimal 형식이다. 방화벽상 source 파일 자체를
열어 재해시하지 않았으므로, 이 값은 compact summary가 제공한 provenance
receipt로만 취급한다.

`sacct`는 parent `15586`이 GPU 2개, CPU 16개, memory `130000M`으로
`2026-07-31T04:10:27`에 시작해 `06:30:42`에 종료되었고, 두 child가 같은
시각에 시작해 모두 `COMPLETED`, `0:0`임을 확인했다.

### lock 및 artifact parity

| 검사 | Llama | Qwen | 결과 |
| --- | ---: | ---: | --- |
| selection mode | `untouched` | `untouched` | PASS |
| planned/attempted/passed/failed | `20/20/20/0` | `20/20/20/0` | PASS |
| events/features/actions/outcomes | `20/20/20/120` | `20/20/20/120` | PASS |
| arms per case | `6` | `6` | PASS |
| action receipts | `20` | `20` | PASS |
| direct-z artifacts | `20` | `20` | PASS |
| rollback/equal-C/firewall | 모두 `true` | 모두 `true` | PASS |
| Git raw output | `false` | `false` | PASS |
| seed | `17` | `17` | PASS |
| q | `0.00390625` | `0.00390625` | PASS |
| estimand | `progress(score_mix)-progress(frozen_static_mix)` | 동일 | PASS |
| bootstrap | `20260731`, `4000` | `20260731`, `4000` | PASS |

Llama compact summary는
`selection_manifest_id=18637248…`, `selected_case_ids_sha256=e6658d6b…`를
노출하지만 Qwen compact summary는 대응하는 exact case-ID hash를 노출하지
않는다. 따라서 cross-model case-ID hash equality는 이 제한된 입력만으로
재계산할 수 없다. 다만 두 summary가 같은 paired job, canonical
`untouched` mode, exact `20/20`, 동일 seed/q/estimand/bootstrap을 보고하고
각 artifact validator가 valid를 보고하므로 이번 locked gate에는
비차단 문서화 caveat로 처리한다. 다음 compact schema에서는
`selected_case_ids_sha256`와 `selection_manifest_id`를 공통 필수 field로
통일해야 한다.

## Model별 aggregate

모든 값은 `adaptive score_mix - frozen static mix`의 paired progress다.

| model | replay envelope | mean | 20% trimmed mean | median | positive sign | paired bootstrap mean 95% CI | oracle mean | single-model input |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Llama3-8B-Instruct | `1e-12` | `0.0102627993` | `0.0073294242` | `0.0076470375` | `20/20` | `[0.0059305298, 0.0152926564]` | `0.0102792978` | clear |
| Qwen2.5-7B-Instruct | `1e-12` | `0.0483206153` | `0.0318814516` | `0.0217521191` | `20/20` | `[0.0270763853, 0.0734122896]` | `0.0484790206` | clear |

두 모델 모두 mean뿐 아니라 trimmed mean, median, sign의 세 보조 조건을
모두 같은 양의 방향으로 충족한다. Oracle-primary mean 차이는 Llama
`0.0000164986`, Qwen `0.0001584053`으로, primary가 null이고 oracle만
남는 `controller/static-policy pivot` 패턴도 아니다.

## Forecast accuracy와 기대효과 보정

| model | D0/D1 prior expected gap | untouched realized mean | realized-prior | realized/prior | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama | `0.0109636724` | `0.0102627993` | `-0.0007008732` | `0.9361` | prior가 약 `6.39%` 과대였지만 크기 수준은 근접 |
| Qwen | `0.1120153104` | `0.0483206153` | `-0.0636946951` | `0.4314` | prior가 약 `56.86%` 과대; magnitude forecast를 그대로 쓰면 안 됨 |

Untouched casewise frozen prediction과 비교하면 Llama predicted mean
`0.0084985140`은 realized보다 `0.0017642852` 낮아 약 `20.76%`
과소예측했고, Qwen predicted mean `0.0845762937`은 realized보다
`0.0362556784` 높아 약 `42.87%` 과대예측했다. 그럼에도 positive-direction
concordance는 두 모델 모두 `1.0`, Pearson은 Llama `0.9271`, Qwen
`0.8164`다.

따라서 현재 기대효과 예측의 안전한 사용 범위는 다음과 같다.

- 양의 routing direction이 untouched에서 유지될 가능성을 예측하는 데는
  유용한 신호가 있었다.
- Llama의 aggregate magnitude prior는 비교적 근접했지만 casewise intercept
  보정까지 완전하지 않다.
- Qwen은 방향은 맞았으나 크기를 크게 과대평가했으므로, 후속 stage의 효과
  크기나 compute 확대 근거로 `0.1120`을 재사용하면 안 된다.
- 이 forecast 표는 사후 threshold를 바꾸지 않았고 §10.9 판정에는 영향을
  주지 않았다.

## Locked kill/advance 적용

| 규칙 | 관측 | 결과 |
| --- | --- | --- |
| 양 model `mean>e_m` 및 trim/median/sign `>=11/20` 중 하나 이상 같은 방향 | 두 model 모두 mean/trim/median 양수, sign `20/20` | MV-1 reproduced |
| 한 model만 clear, 다른 model non-negative | 해당 없음 | architecture-conditional 아님 |
| 양 model mean/trim/oracle null, sign `<=10/20` | 해당 없음 | routing kill 아님 |
| oracle만 양수 | 해당 없음 | controller/static-policy pivot 아님 |

## 다음 행동과 claim boundary

허용되는 다음 행동은 사전에 locked된 최소 MV-2 stale-vs-refreshed pair 한
건의 실행뿐이다. MV-2는 새 technical/preflight gate를 통과해야 하며, 그
결과를 이 MV-1 보고서가 미리 보증하지 않는다. MV-2 결과 전에는 ODE refresh,
method gain, paper-ready evidence 또는 broader baseline superiority를
주장하지 않는다.
