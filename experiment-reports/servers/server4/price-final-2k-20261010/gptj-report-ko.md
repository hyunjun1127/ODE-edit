# GPT-J Ours: 최종 PRICE method 2K 결과 반영 (CF·zsRE)

사용자 결정(2026-10-11): 본표의 Llama·GPT-J PRICE는 HC 가격 정규화를 하지 않은 지금 결과를 쓴다(정규화 전후 차이가 잡음 수준). Qwen만 정규화 HC 재실행을 쓴다.

## 실행

| 항목 | CF 2K | zsRE 2K |
|---|---|---|
| job | rent **101707** (COMPLETE, 7.6 h) | server4 **63027** (COMPLETE, 3.8 h) |
| source | rent snapshot `1f906c7e`(server4 `885a0e26`과 같은 내용) | `962687f3` |
| arm | `gptj-P-beta075`, resolved `d8f5ff72…` | 같음 |
| 데이터·평가 | eval-2K CF, strict NLL R/P/N | official zsRE first-2K, official zsRE evaluator |

- method: β = c = β_max = 0.75, unit-lr ρ0.05, γ1, cap 끝점 cast, HC-PRICE(가격 정규화 안 함), 요청별 early exit, sink EOT 규칙과 anchor guard 20.
- 가장 싼 층 HC 가격 중앙값은 CF B10 1.012, zsRE B16 1.014였다. 정규화하지 않아 생긴 강도 차이는 1–2% 이내다.

## W20 결과

| | Score | Eff | Gen | Loc |
|---|---:|---:|---:|---:|
| CF (분자/분모) | 88.22 | 99.80 (1,996/2,000) | 96.23 (3,849/4,000) | 73.57 (14,713/20,000) |
| zsRE | – | 99.81 | 96.96 | 29.59 |

- CF W0(340/2,000 · 772/4,000 · 16,495/20,000)와 zsRE W0(27.83 / 27.15 / 27.59)가 본표 W0 행과 같다.
- CF Score는 반올림 전 성공률의 조화평균(88.2203)이다. rent 보고의 N 73.56은 반올림 방식 차이이며, 정확한 값 73.565는 decimal half-up으로 73.57이다.
- 같은 W20 baseline: AlphaEdit-BLUE 89.23, AlphaEdit+SPHERE 88.39, AlphaEdit 88.22(99.70 / 96.33 / 73.56). PRICE는 AlphaEdit와 같은 수준이고 AlphaEdit-BLUE보다 1.01 낮다. zsRE는 Eff·Gen이 가장 높고, Loc은 MEMIT(30.81) 다음이다.
- CF Flu/Con은 W20 가중치로 devbox에서 baseline과 같은 생성 평가기로 평가한다(job 63207).

## zsRE 누적 결과 (all-seen)

| 시점 | Eff | Gen | Loc | W0 예측 일치율 |
|---|---:|---:|---:|---:|
| W5 | 99.77 | 97.44 | 28.71 | 88.27 |
| W10 | 99.78 | 97.43 | 29.33 | 83.85 |
| W15 | 99.84 | 97.45 | 29.69 | 81.04 |
| W20 | 99.81 | 96.96 | 29.59 | 78.64 |

정확한 값과 checkpoint SHA: [gptj-results.json](../../../../audits/servers/server4/price-final-2k-20261010/gptj-results.json).
