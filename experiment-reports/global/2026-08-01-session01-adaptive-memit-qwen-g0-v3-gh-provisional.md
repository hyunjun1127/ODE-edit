# Session 01 Motivation — adaptive MEMIT Qwen g0 v3 GH provisional analysis

날짜: 2026-08-01

방법명: **ODE-Edit**

상태: **technical-valid / precommitted lenient model gate pass**

분석 주체: GH. 요청된 Terra Ultra analysis agent는 runtime에서
`gpt-5.6-terra`, `reasoning=ultra` metadata를 검증할 수 없어 어떤 파일도 읽지
않고 종료했다. 따라서 이 문서는 독립 agent report가 아닌 compact scalar 기반
임시 GH 판정이다.

## Proposal에서 온 내용

- intermediate state에서 direction을 다시 계산하는 것이 ODE 동기의 핵심이다.
- state-dependent policy가 static alternative보다 유리하지 않으면 단순화해야 한다.
- proposal은 검증된 claim이나 확정된 paper plan이 아니다.

## Repo/protocol에서 확인한 사실

- model: `qwen2.5-7b-inst`; track: `memit`; cases: 정확히 8.
- features/actions/analysis/events 8, outcomes 56, receipts 8, direct-z artifacts 8.
- 모든 stream SHA-256가 summary와 일치하고 failure는 0이다.
- rollback, evaluation firewall, lineage, matched per-hop C-distance, precomputed
  covariance gate가 모두 통과했다.
- 사전고정 analyzer verdict는 `LENIENT_MODEL_PASS`다.
- small CounterFact teacher-forced Motivation signal이며 accuracy/locality/retention/
  downstream/lifelong/method superiority claim이 아니다.

## Scalar 결과

| Contrast | Mean | 8-case sign |
|---|---:|---:|
| always direction refresh `A4 - B4` | +1.583319 | 7 positive / 1 negative |
| coefficient refresh `B4 - C4` | +0.657322 | 8 positive / 0 negative |
| gated `G4 - A4` | +0.001426 | 1 positive / 7 floor |
| gated `G4 - B4` | +1.584745 | 7 positive / 1 negative |
| gated `G4 - C4` | +2.242067 | 8 positive / 0 negative |
| gated minus casewise max(`A4`,`B4`) | -0.012914 | 1 positive / 1 negative / 6 floor |

- gate decisions: 24; refresh 23; refresh rate 0.9583.
- mean predicted refreshed advantage: +0.374526.
- W0-to-refreshed C-cosine: mean 0.783496, minimum 0.160837.
- controlled NFE total 1312; proposal builds 64; measured wall time total
  25,768.16 seconds.

## GH 추정

Qwen에서는 central-probe selector가 거의 항상 refreshed direction을 택해 `G4`가
사실상 `A4`와 같았다. direction refresh는 coefficient-only refresh보다 평균적으로
크게 높았고, fixed direction보다도 높았다. 이는 Qwen에서 stale-direction
relinearization 가능성을 지지하는 작은 atomic signal이다.

그러나 Llama와 selector behavior가 크게 다르므로 Qwen pass만으로 adaptive gate를
유지할 수 없다. model별 margin, score sign, rescue branch는 금지한다. 양 모델 공통
후보는 selector가 아니라 already-defined always-refresh `A4`다.

## 다음 gate

- MEMIT pair에서는 adaptive selector를 kill하고 always-refresh를 post-result
  common-policy candidate로 명시한다.
- AlphaEdit projected pair에서 같은 `A4-B4` 방향이 양 모델에 나타나는지 본다.
- projected 결과 전에는 preservation/null-space synergy claim을 열지 않는다.

## 사용자 확인 필요

- 없음. 현재는 Motivation 내부의 exploratory pivot이며 다음-stage 승인이 아니다.
