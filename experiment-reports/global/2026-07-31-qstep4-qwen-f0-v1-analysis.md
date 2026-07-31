# Motivation quarter-step Qwen 분석

- model: `qwen2.5-7b-inst`
- run: `qstep4_qwen_f0_v1`
- Slurm: parent `15701`, child `15701.1`, `COMPLETED 0:0`
- technical validity: `PASS` (`12/12`, 실패 `0`)
- locked analyzer verdict: `DIRECTION_REFRESH_CLEAR`
- GH 판정: **Qwen에서는 방향과 계수 재계산 효과가 크고 step에 따라 일관되게 증가한다.**

## 무엇을 비교했는가

Native ordered MEMIT update의 C-distance 전체 `D`를 네 hop으로 나눴다. 각
hop은 거리 `D/4`, C-energy `E_native/16`이며 네 hop의 총 경로 길이는 `D`다.

- A: 방향과 계수를 매 hop 다시 계산
- B: W0 방향은 고정하고 계수만 매 hop 다시 계산
- C: W0 방향과 계수를 모두 고정
- controls: native one-shot, 동일 native update exact split4

Direct-z는 W0에서 한 번만 계산해 모든 lineage에 고정했다. Step 4의 세
contrast만 primary이며 best-hop 사후 선택은 금지했다.

## 결과

| contrast | mean | median | positive | bootstrap mean 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 방향 재계산 `A4-B4` | `+1.6590` | `+1.7470` | `11/12` | `[+0.9748, +2.4057]` |
| 계수 재계산 `B4-C4` | `+0.8742` | `+0.7670` | `11/12` | `[+0.5310, +1.2517]` |
| 총 refresh `A4-C4` | `+2.5332` | `+2.5140` | `12/12` | `[+1.6183, +3.5323]` |
| full refresh 대 native MEMIT `A4-native` | `-0.1574` | `-0.0722` | `6/12` | `[-0.5372, +0.2758]` |

세 refresh contrast의 mean CI가 모두 0보다 크다. 단위는 teacher-forced
absolute rewrite utility proxy이며 benchmark accuracy가 아니다.

## step별 진행

| step | A: 방향+계수 refresh | B: 계수만 refresh | C: 모두 고정 | A-B |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `8.2577` | `8.2577` | `8.2577` | `0` |
| 2 | `12.3733` | `11.5833` | `11.2587` | `+0.7900` |
| 3 | `13.7848` | `12.6103` | `11.9877` | `+1.1744` |
| 4 | `14.6540` | `12.9950` | `12.1208` | `+1.6590` |

방향 효과와 계수 효과가 step 2부터 4까지 함께 커진다. Qwen에서는 방향을
매번 다시 계산하는 것이 stale direction보다 나쁘다는 설명과 맞지 않는다.

## 수치 noise인가, 실제 방향 변화인가

- native one-shot과 exact split4 mean 차이: `1.13e-6`
- practical envelope: `1e-4`; 12/12 모두 envelope 안
- refreshed direction W0 대비 C-cosine 평균:
  step 2 `0.8538`, step 3 `0.7150`, step 4 `0.6270`
- step 4 최소 cosine: `0.1579`

네 번 적용한 산술 오차는 무시할 수준이지만, 방향은 state가 변하면서 크게
회전했다. 이번 성능 이득은 이 실제 회전을 따라간 효과와 일치한다.

## native MEMIT와 기대효과 경계

- native MEMIT mean progress: `14.8114`
- 모두 고정한 C4의 native 대비 gap: `-2.6906`
- full refresh가 그 proxy gap에서 회복한 비율: 약 `94.1%`
- direction 기여 proxy: 약 `61.7%p`
- coefficient 기여 proxy: 약 `32.5%p`
- A4는 native보다 평균 `0.1574` 낮고 CI는 0을 포함한다.

즉 refresh는 stale score-mix 경로의 손실 대부분을 회복했지만 native MEMIT를
평균으로 넘지는 못했다. 이 결과를 ODE-Edit baseline 우위로 쓰면 안 된다.

Outcome-selected direction gate oracle은 `+1.6661`, unconditional direction
refresh는 `+1.6590`으로 거의 같다. Qwen에서는 gating으로 얻을 추가 상한이
작고 unconditional refresh가 이미 안정적이다. 현재 재현 가능한 Motivation
기대효과는 stale C 대비 direction `+1.6590`, coefficient `+0.8742`, 합계
`+2.5332` proxy다. 이는 method/benchmark gain 예측이 아니다.

## 이전 D/32 MV-2와의 관계

이전 direction mean `+0.00689`에서 이번 `+1.6590`으로 신호가 크게
드러났다. Treatment 총 경로와 checkpoint가 달라 선형 배율 비교는 금지하지만,
D/32가 실제 방향 회전을 보기에는 지나치게 작았다는 under-manipulation
가설은 Qwen에서 강하게 지지된다.

## 실행 무결성과 비용

- exact artifact count: feature/action/receipt/event/analysis/direct-z 각 `12`,
  outcome `156`
- rollback, lineage, receipt-before-outcome, native control: 모두 pass
- controlled NFE: `1176`; proposal builds: `60`; probe panels: `84`
- runner wall: `28717.61s`; GPU peak allocated/reserved:
  `47497015808` / `49853497344` bytes
- host max RSS: `17368268 KiB`
- raw와 full analysis는 ignored `local/`에 유지하며 Git에는 compact
  report와 summary만 남긴다.

## 네 범주 기록

| 범주 | 기록 |
| --- | --- |
| proposal에서 온 내용 | ODE-Edit Motivation으로서 경로 중 방향 재계산이 stale direction보다 유리할 수 있다는 가설이다. Proposal은 확정 method나 검증된 claim이 아니다. |
| repo/protocol에서 확인한 사실 | Fresh `[112:124]`, `K=4`, hop `D/4`, fixed step-4 contrast, job `15701.1`, exact artifact·rollback·hash와 위 집계값을 확인했다. |
| GH 추정 | Qwen에서는 unconditional direction refresh와 dynamic coefficient가 유망하지만, native comparator와의 잔여 gap을 닫는 별도 method 설계가 필요하다. |
| 사용자 확인 필요 | 다음 gated-direction/cross-model diagnostic 또는 AlphaEdit factorial 실행은 별도 범위 확정이 필요하다. 현재 결과 판정에는 추가 확인이 없다. |

## Claim boundary

이 결과는 Qwen의 stale-vs-refreshed mechanism contrast를 지지한다. ODE-Edit의
accuracy/locality/retention, EasyEdit baseline 대비 우위, cross-model
일반화, AlphaEdit와의 상호작용은 측정하지 않았다.
