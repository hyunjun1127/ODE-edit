# Session 01 Motivation — adaptive MEMIT Llama g0 v3 GH provisional analysis

날짜: 2026-08-01

방법명: **ODE-Edit**

상태: **technical-valid / precommitted lenient model gate fail**

분석 주체: GH. 요청된 독립 Terra Ultra analysis agent는 runtime에서
`gpt-5.6-terra`, `reasoning=ultra` metadata를 검증할 수 없어 어떤 파일도 읽지
않고 종료했다. 따라서 이 문서는 독립 agent report가 아니며, compact scalar에
대한 임시 GH 판정이다.

## Proposal에서 온 내용

- 작은 step 자체가 아니라 intermediate state에서의 direction relinearization이
  ODE 동기의 핵심이다.
- state-dependent direction이 고정 방향보다 유리하지 않으면 ODE controller를
  단순화하거나 static policy로 pivot해야 한다.
- proposal은 검증된 claim이나 확정된 paper plan이 아니다.

## Repo/protocol에서 확인한 사실

- model: `llama3-8b-inst`; track: `memit`; cases: 정확히 8.
- exact streams: features 8, actions 8, outcomes 56, analysis cases 8, events 8.
- exclusive receipts 8, direct-z artifacts 8, failure 0.
- stream SHA-256가 summary와 일치하며 rollback, evaluation firewall, lineage,
  matched per-hop C-distance, precomputed covariance 조건이 모두 통과했다.
- 사전고정 analyzer verdict는 `LENIENT_MODEL_COLLAPSE`다.
- claim boundary는 small CounterFact teacher-forced Motivation signal이며,
  accuracy/locality/retention/downstream/lifelong/method superiority를 뜻하지 않는다.

## Scalar 결과

모든 값은 terminal utility 차이이며 양수일수록 왼쪽 arm이 낫다.

| Contrast | Mean | 8-case sign |
|---|---:|---:|
| always direction refresh `A4 - B4` | +0.828098 | 8 positive / 0 negative |
| coefficient refresh `B4 - C4` | +0.385384 | 8 positive / 0 negative |
| gated `G4 - A4` | -0.772068 | 0 positive / 8 negative |
| gated `G4 - B4` | +0.056029 | 2 positive / 0 negative / 6 floor |
| gated `G4 - C4` | +0.441413 | 8 positive / 0 negative |
| gated minus casewise max(`A4`,`B4`) | -0.772068 | 0 positive / 8 negative |

- gate decisions: 24; refresh 2; refresh rate 0.0833.
- mean predicted refreshed advantage: -0.165817.
- W0-to-refreshed C-cosine: mean 0.736045, minimum 0.153754.
- controlled NFE total 1312; proposal builds 64; measured wall time total
  20,771.36 seconds.

## GH 추정

방향 재계산이 나쁜 것이 아니다. 항상 방향을 재계산한 `A4`는 `B4`와 `C4`보다
8/8 cases에서 높았다. 성능을 악화시킨 직접 원인은 current central-probe gate가
24번 중 22번 fixed direction을 택해, 좋은 refreshed endpoint로 가는 경로를 거의
차단한 것이다.

margin을 0.02에서 0으로 낮추는 정도는 충분하지 않다. recorded advantage 중
대부분이 음수여서 선택 수가 실질적으로 늘지 않는다. outcome을 본 뒤 score
부호를 뒤집는 rescue도 정당화하지 않는다. 현재 gate proxy는 Llama에서 kill
후보이고, 이미 사전 정의된 `always-refresh A4`가 다음 공통-model 후보이다.

같은 cases에서 policy를 `G4`에서 `A4`로 바꾸면 관측상 평균 utility를
`+0.772068` 회복한다. 이는 **in-sample arm substitution forecast**일 뿐이며,
새 data나 sequential preservation에서의 기대 개선으로 주장하지 않는다.

## 다음 gate

- Qwen terminal 결과에서 `A4-B4`와 `A4-C4`가 같은 방향인지 확인한다.
- 양 모델에서 공통이면 adaptive selector를 kill하고 same-code always-refresh
  ODE 후보로 단순화한다.
- Qwen에서 반대면 model별 policy는 금지하므로 direction-refresh common method
  claim을 열지 않는다.
- AlphaEdit projected pair는 별도 고정 track으로 실행해 projector 안에서도
  `A4`, `B4`, `C4`, `G4` 관계를 확인한다.

## 사용자 확인 필요

- 없음. 현재는 기존 고정 gate에 따른 Motivation 판정이며 외부 claim이나
  다음-stage 확장을 승인한 것이 아니다.
