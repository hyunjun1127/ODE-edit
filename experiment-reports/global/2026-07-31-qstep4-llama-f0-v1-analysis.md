# Motivation quarter-step Llama 분석

- model: `llama3-8b-inst`
- run: `qstep4_llama_f0_v1`
- Slurm: parent `15701`, child `15701.0`, `COMPLETED 0:0`
- technical validity: `PASS` (`12/12`, 실패 `0`)
- locked analyzer verdict: `DIRECTION_REFRESH_CLEAR`
- GH 판정: **최소 direction gate는 통과했지만 비단조·고불확실 신호다. 계수 재계산이 더 안정적이다.**

## 무엇을 비교했는가

기존 MV-2는 첫 state treatment가 native update의 C-distance 약 `D/32`여서
실제 방향 변화보다 수치 잡음에 가까울 가능성이 있었다. 이번에는 같은 native
update 전체 거리 `D`를 정확히 네 구간으로 나눴다. 각 실제 hop은 `D/4`이고,
C-energy로는 native의 `1/16`이다. 네 hop의 경로 길이 합은 `D`이므로 전체
update를 절반만 쓴 실험이 아니다.

- A: 매 hop에서 방향과 계수를 모두 다시 계산
- B: 최초 방향은 고정하고 계수만 다시 계산
- C: 최초 방향과 계수를 모두 고정
- native controls: ordered MEMIT one-shot과 같은 update를 정확히 4등분해 적용

모든 분기는 같은 W0의 direct-z를 한 번만 계산해 고정했고, step 4의
`A4-B4`, `B4-C4`, `A4-C4`만 primary로 판정했다. 가장 좋아 보이는 중간
step을 사후 선택하지 않았다.

## 결과

| contrast | mean | median | positive | bootstrap mean 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 방향 재계산 `A4-B4` | `+0.0370` | `+0.0857` | `8/12` | `[-0.5154, +0.5684]` |
| 계수 재계산 `B4-C4` | `+0.1249` | `+0.0317` | `8/12` | `[+0.0070, +0.2554]` |
| 총 refresh `A4-C4` | `+0.1619` | `+0.1773` | `8/12` | `[-0.4569, +0.7424]` |
| full refresh 대 native MEMIT `A4-native` | `-0.2486` | `-0.2815` | `2/12` | `[-0.4942, +0.0399]` |

`progress`는 teacher-forced absolute rewrite utility proxy다. Accuracy,
retention 또는 benchmark score가 아니다.

## step별로 보면 왜 판정을 조심해야 하는가

| step | A: 방향+계수 refresh | B: 계수만 refresh | C: 모두 고정 | A-B |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `2.6019` | `2.6019` | `2.6019` | `0` |
| 2 | `6.3386` | `6.6027` | `6.5623` | `-0.2641` |
| 3 | `10.7525` | `11.2700` | `11.1797` | `-0.5175` |
| 4 | `13.2637` | `13.2267` | `13.1018` | `+0.0370` |

방향 재계산은 step 2와 3에서 오히려 손해이고 마지막 step에서만 작은 양수로
뒤집힌다. 따라서 locked minimum rule은 통과했어도 “Llama에서 방향을 매번
재계산하면 좋아진다”라고 일반화할 수 없다. 반면 계수 재계산은 step 4
bootstrap CI까지 양수라 현재 증거에서 더 안정적인 구성 요소다.

## 수치 noise인가, 실제 방향 변화인가

수치 noise 설명은 지지되지 않는다.

- native one-shot과 exact split4의 mean 차이: `1.15e-6`
- practical envelope: `1e-4`
- 12/12 모두 envelope 안
- refreshed direction의 W0 대비 C-cosine 평균:
  step 2 `0.9239`, step 3 `0.7681`, step 4 `0.6198`
- step 4 최소 cosine: `0.3798`

즉 같은 native update를 네 번 나눠 적용한 연산 오차는 사실상 0이지만,
현재 state에서 다시 계산한 방향은 진행하면서 크게 회전했다. 성능의 양·음
차이는 단순 반복 적용 noise가 아니라 이 회전 방향을 선택한 결과다.

## native MEMIT와 기대효과 경계

- native MEMIT mean progress: `13.5124`
- 모두 고정한 C4의 native 대비 gap: `-0.4106`
- full refresh가 그 proxy gap에서 회복한 비율: 약 `39.4%`
- 그중 direction 기여 proxy: 약 `9.0%p`
- coefficient 기여 proxy: 약 `30.4%p`
- 그러나 A4는 native보다 평균 `0.2486` 낮다.

이 회복률은 score-mix C와 ordered native MEMIT의 semantics가 다르므로
설명용 proxy다. 현재 ODE-style path가 native baseline을 이겼다는 뜻이 아니다.

Outcome을 보고 direction refresh가 양수인 case만 고르는 비실현 oracle은
평균 `+0.3714`이고, unconditional refresh는 `+0.0370`이다. 이는 state gate를
연구할 여지가 있다는 상한일 뿐, outcome-free gate의 예측 성능이나 method
gain으로 사용할 수 없다. 보수적으로 재사용할 수 있는 현재 기대효과는 계수
재계산의 `+0.1249` proxy이며, 다음 방향 gate는 Llama에서 음의 early-step
refresh를 피하는 것을 먼저 증명해야 한다.

## 이전 D/32 MV-2와의 관계

이전 direction mean은 `-0.00441`, 이번 step-4 direction mean은 `+0.0370`이다.
따라서 D/32가 너무 작았다는 가설은 일부 지지된다. 다만 treatment 경로가
16배 길어졌고 이번 Llama 효과는 중간 step에서 음수이며 CI가 넓다. 효과를
선형 배율로 비교하거나 under-manipulation 가설이 완전히 확정됐다고 해석하지
않는다.

## 실행 무결성과 비용

- exact artifact count: feature/action/receipt/event/analysis/direct-z 각 `12`,
  outcome `156`
- rollback, lineage, receipt-before-outcome, native control: 모두 pass
- controlled NFE: `1176`; proposal builds: `60`; probe panels: `84`
- runner wall: `23448.42s`; GPU peak allocated/reserved:
  `43071349248` / `46812626944` bytes
- host max RSS: `10770080 KiB`
- raw와 full analysis는 ignored `local/`에 유지하며 Git에는 이 compact
  report와 summary만 남긴다.

## 네 범주 기록

| 범주 | 기록 |
| --- | --- |
| proposal에서 온 내용 | ODE-Edit Motivation으로서 경로 중 방향 재계산이 stale direction보다 유리할 수 있다는 가설이다. Proposal 자체는 확정 method나 paper claim이 아니다. |
| repo/protocol에서 확인한 사실 | 사전등록된 fresh `[112:124]`, `K=4`, hop `D/4`, fixed step-4 contrast, job `15701.0`, exact artifact·rollback·hash와 위 집계값을 확인했다. |
| GH 추정 | Llama에서는 unconditional direction refresh보다 dynamic coefficient와 outcome-free state gate의 조합이 더 유망하다. |
| 사용자 확인 필요 | 다음 gated-direction diagnostic 또는 AlphaEdit factorial을 새로 실행하려면 별도 범위 확정이 필요하다. 현재 결과 자체의 판정에는 추가 확인이 없다. |

## Claim boundary

이 결과는 Llama에서 방향이 실제로 회전하고 step 4 minimum gate가 양수임을
보인다. Unconditional direction refresh의 안정성, ODE-Edit method 우위,
accuracy/locality/retention 개선, AlphaEdit와의 상호작용은 증명하지 않았다.
