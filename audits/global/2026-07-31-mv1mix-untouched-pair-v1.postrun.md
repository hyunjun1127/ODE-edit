# MV-1 untouched pair post-run red audit

- 작성일: 2026-07-31 KST
- 대상 job: `15586` / `odeedit_mv1mix_untouched_pair_v1`
- locked rule: `plans/global/2026-07-30-session01-mv1-implementation-spec.md` §10.9
- red status: `pass` (compact schema 통일 권고는 비차단)

## 감사 범위

허용 입력은 아래 두 compact summary, §10.9, `PROTOCOL.md`, `sacct -j
15586` technical metadata뿐이다.

- `experiment-reports/global/2026-07-31-mv1mix-llama-untouched-v1.analysis.summary.json`
- `experiment-reports/global/2026-07-31-mv1mix-qwen-untouched-v1.analysis.summary.json`

`local/results/`, full analysis JSON, case-level diagnostic, C1/D1 report,
raw outcome은 열지 않았다. Pair/MV-2 결과를 미리 주장하지 않는다.

## 네 범주

| 범주 | 감사 기록 |
| --- | --- |
| proposal에서 온 내용 | Proposal 원문은 허용 입력이 아니어서 읽지 않았다. Proposal은 최종 paper plan이나 검증 claim으로 취급하지 않았다. |
| repo/protocol에서 확인한 사실 | §10.9 locked rule, 두 compact aggregate, protocol의 post-run·artifact·claim boundary, Slurm technical 종료 상태를 확인했다. |
| GH 추정 | 두 모델의 방향 재현은 최소 MV-2 diagnostic을 열 근거이나, Qwen forecast magnitude의 큰 과대는 후속 효과 크기 claim을 제한한다. |
| 사용자 확인 필요 | Locked MV-2 pair 한 건에는 없음. 범위 확대·threshold 변경·추가 claim은 별도 확인이 필요하다. |

## Technical audit

| check | 결과 | 근거 |
| --- | --- | --- |
| summary status/schema | PASS + 비차단 caveat | 둘 다 `complete`; schema 문자열/key layout은 다르나 공통 gate field는 의미상 호환 |
| compact input integrity | PASS | 현재 파일 SHA-256: Llama `cb1a920e…`, Qwen `69f6372a…`; 내장 source receipt는 모두 유효한 hex 형식 |
| execution identity | PASS | execution commit `a7c929f…`, verification head `51d8597…`, job `15586` exact match |
| analyzer drift | PASS | 두 summary 모두 execution부터 verification head까지 unchanged를 보고 |
| Slurm terminal state | PASS | parent와 두 child 모두 `COMPLETED`, `ExitCode=0:0`; 두 child 동시 시작 |
| model/run distinct | PASS | `llama3-8b-inst`/`mv1mix_llama_untouched_v1`와 `qwen2.5-7b-inst`/`mv1mix_qwen_untouched_v1` |
| selection | PASS + 비차단 caveat | 둘 다 canonical `untouched`, exact 20; Qwen compact에는 exact case-ID hash가 없어 hash equality 재계산 불가 |
| artifact counts | PASS | 모델별 `20` event/feature/action, `120` outcomes, `20` receipts/direct-z, failed `0`, six arms |
| artifact validity | PASS | rollback/equal-C/outcome-firewall true, Git raw output false |
| estimand parity | PASS | 두 모델 모두 `progress(score_mix)-progress(frozen_static_mix)` |
| seed/q/bootstrap parity | PASS | `17`, `1/256`, `20260731`/`4000` exact |
| decision separation | PASS | 두 single-model summary 모두 pair decision과 MV-2 decision `false` |

내장 source hash는 방화벽 때문에 source 파일을 열어 재계산하지 않았고
provenance receipt로만 확인했다. Qwen compact에
`selected_case_ids_sha256`와 `selection_manifest_id`가 없는 문제는 같은
paired job, canonical mode, exact count와 동일 lock으로 보완되는 비차단
문서화 결함이다. 다음 pair부터 두 field와 schema를 공통으로 고정한다.

## Scientific gate

| model | mean | trim20 | median | sign | oracle mean | locked input |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Llama | `0.0102627993` | `0.0073294242` | `0.0076470375` | `20/20` | `0.0102792978` | clear |
| Qwen | `0.0483206153` | `0.0318814516` | `0.0217521191` | `20/20` | `0.0484790206` | clear |

두 model 모두 `mean>1e-12`이고 trim, median, sign `>=11/20`이 같은 양의
방향이다. 양 model clear이므로 architecture-conditional이 아니다. Primary가
양수이고 oracle과 거의 같아 kill 또는 oracle-only pivot도 아니다.

## Forecast calibration audit

- Llama: prior `0.0109636724`, realized `0.0102627993`,
  realized-prior `-0.0007008732`; prior가 약 `6.39%` 과대다.
- Qwen: prior `0.1120153104`, realized `0.0483206153`,
  realized-prior `-0.0636946951`; prior가 약 `56.86%` 과대다.
- Positive-direction concordance는 양 model `1.0`이지만 magnitude accuracy,
  특히 Qwen의 크기 예측은 후속 claim에 재사용할 수 없다.
- 이 비교로 replay envelope, sign threshold, policy 또는 estimand를
  변경하지 않았다.

## Kill/advance 및 claim boundary

§10.9를 기계 적용하면 MV-1 primary direction은 untouched pair에서
reproduced다. 허용 범위는 locked 최소 MV-2 pair 한 건의 준비·실행뿐이다.
이 audit은 MV-2 성공, ODE refresh 효과, ODE-Edit 방법 우위, 일반화,
novelty 또는 paper-ready claim을 승인하지 않는다.

- 최종 판정: `MV2 PREPARE` — MV-1 untouched pair reproduced; locked MV-2 pair 한 건 허용
