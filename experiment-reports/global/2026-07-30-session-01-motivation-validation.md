# Session 01 — Motivation Validation: Global Evidence Index

- 최초 작성: 2026-07-30
- 최종 갱신: 2026-07-31
- 작성 주체: global-head
- 상태: **closed — `NO MV3`, no retune, no further Slurm**
- proposal:
  `project/proposals/sections/01-motivation-validation.md`
- final report:
  `experiment-reports/global/2026-07-31-session-01-motivation-final.md`

## 최종 질문과 답

same-snapshot allocation signal은 두 고정 모델에서 재현됐지만,
partial update 뒤 proposal **direction**을 refresh하는 ODE/relinearization
advantage는 재현되지 않았다. Qwen은 양의 신호였고 Llama는 primary와 total
모두 일관되게 음수였다. 사전등록 pair rule의 최종 판정은 `NO MV3`다.

## Curated evidence

### MV-0 — implementation fidelity

- Llama/Qwen pair job `15517`, 실제 동시 실행.
- 모델별 planned/attempted/pass `3/3/3`, rollback exact, primary error `0`.
- red:
  `audits/global/2026-07-30-mv0-pair-c3-v3.postrun.md`

### MV-1 — held-out allocation signal

- pair job `15586`, 두 child 동시 시작·완료.
- Llama adaptive-minus-static mean `+0.0102627993`, sign `20/20`,
  CI `[0.0059305298, 0.0152926564]`.
- Qwen mean `+0.0483206153`, sign `20/20`,
  CI `[0.0270763853, 0.0734122896]`.
- forecast direction은 맞았지만 magnitude prior는 Llama 약 `6.39%`, Qwen 약
  `56.86%` 과대였다.
- report:
  `experiment-reports/global/2026-07-31-mv1mix-untouched-pair-v1-analysis.md`
- red:
  `audits/global/2026-07-31-mv1mix-untouched-pair-v1.postrun.md`

### MV-2 — refreshed direction

- pair job `15610`, Llama/Qwen을 같은 parent에서 동시에 제출.
- parent와 두 child 모두 `COMPLETED`, exit `0:0`.
- raw analysis stream SHA-256:
  - Llama `c5c698ee7828000e4ecc0c5c15b86f0dca9ba014dbb5b5f678c59f5ab7d23e38`
  - Qwen `5fee05d7d6c3ec5870bdc3689f5422f38e588adfc23d45ab77401f9a11c5d618`
- analyzer의 최초 v1 key-order 거부는 JSON object serialization order에
  대한 기술 오류였다. Raw나 estimand를 바꾸지 않고 object key-set
  validation으로만 수리했으며 focused `19/19`, Motivation suite `172/172`
  test가 통과했다.
- 정상 v2 분석:
  - Llama:
    `experiment-reports/global/2026-07-31-mv2refresh-llama-e0-v2-analysis.md`
  - Qwen:
    `experiment-reports/global/2026-07-31-mv2refresh-qwen-e0-v2-analysis.md`
- pair red:
  `audits/global/2026-07-31-mv2refresh-pair-v2.postrun.md`

## 최종 aggregate

| contrast | Llama3-8B-Instruct | Qwen2.5-7B-Instruct |
| --- | ---: | ---: |
| MV-1 allocation mean | `+0.0102628` | `+0.0483206` |
| MV-2 direction refresh mean | `-0.0044081` | `+0.0068874` |
| MV-2 coefficient refresh mean | `+0.0001037` | `+0.0016649` |
| MV-2 total refresh mean | `-0.0043044` | `+0.0085523` |
| single-model MV-2 verdict | fixed-direction coefficient pivot | direction refresh clear |

Pair first-match trace에서 technical, redesign, reproduced,
architecture-conditional, common coefficient pivot, static kill이 모두
불일치했고 혼합 rule 7에서 종료했다.

## Compute와 artifact boundary

- MV-2 aggregate controlled NFE `1032`, proposal builds `96`, probe panels
  `72`, model-run wall seconds 합 `20241.0092`.
- Llama child elapsed `02:25:16`, MaxRSS `10196536K`; Qwen child elapsed
  `03:13:40`, MaxRSS `16474488K`.
- EasyEdit checkout은 수정하지 않았다.
- raw logs, raw streams, action receipts, direct-z와 failed retry artifacts는
  ignored `local/`에만 보존했다.
- Git에는 analyzer/test, compact JSON, Korean report, audit와 재현 지침만
  남겼다. Credential·private SSH material은 기록하지 않았다.

## 네 범주와 종료 결정

| 범주 | 기록 |
| --- | --- |
| Proposal에서 온 내용 | same-snapshot allocation 뒤 state-dependent direction refresh가 더 나을 수 있다는 반증 가능한 가설이었다. |
| Repo/protocol에서 확인한 사실 | MV-0 fidelity와 MV-1 allocation signal은 두 모델에서 통과했지만 MV-2 direction/total의 부호는 모델 사이에서 반대였고 pair red는 `NO MV3`다. |
| GH 추정 | allocation은 연구할 가치가 남지만 refreshed-direction ODE 일반성은 현재 설계에서 kill해야 한다. |
| 사용자 확인 필요 | Session 01 종결에는 없음. 별도 static/coefficient/architecture-specific 방향을 시작할지는 사용자 결정이 필요하다. |

추가 MV-3, MV-4, retune, threshold 변경, rescue fold 또는 외부 method claim은
이 evidence chain에서 허용하지 않는다.
