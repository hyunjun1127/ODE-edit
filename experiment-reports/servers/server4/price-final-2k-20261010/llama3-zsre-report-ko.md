# Llama Ours zsRE: 최종 PRICE method 2K 결과 반영

사용자 지시(2026-10-10): "server4에 llama zsre run 올리자. CF beta100 run과 동일한 세팅으로 올려라. 그리고 main table에 최신화해". 이 결과가 README 본표 Llama `PRICE (Ours)` 행의 zsRE 칸을 채운다.

## 실행

| 항목 | 값 |
|---|---|
| server / job | server4, Slurm **62889** `pf2k-llama3-zsre-P-beta100-ee` (2026-10-10 19:58–23:17 KST, COMPLETED) |
| source | commit `685322f54e43286b23126168bf626f4e3d16322a` (zsRE 경로 추가, method 코드는 CF 2K의 `885a0e26`과 같음) |
| arm | `llama3-P-beta100`, resolved sha256 `1208f9ce…` (CF 2K와 같음) |
| method | β = c = β_max = 1.0, unit-lr ρ0.05, γ1, cap 끝점 cast, HC-PRICE, 요청별 early exit |
| 데이터 | baseline과 같은 official zsRE first-2K stream(`f42ee4bc…`), B1–B20(100건씩) |
| 평가 | official zsRE evaluator. Eff = Efficacy, Gen = Generalization, Loc = Specificity_loc_ans(teacher-forced token 정확도, 요청 평균) |

- W0는 README zsRE W0 행과 같다(38.10 / 37.61 / 38.59). baseline과 같은 평가기·표본이라는 확인이다.
- arm과 runner는 server4 task branch에만 있고 main에는 아직 병합하지 않았다.

## 누적 결과 (all-seen)

| 시점 | Eff | Gen | Loc | W0 예측 일치율 |
|---|---:|---:|---:|---:|
| W0 | 38.10 | 37.61 | 38.59 | 100 |
| W5 | 99.42 | 95.88 | 45.17 | 68.77 |
| W10 | 99.61 | 95.33 | 44.86 | 66.69 |
| W15 | 99.60 | 95.04 | 45.17 | 64.55 |
| **W20** | **99.62** | **94.83** | **45.31** | 62.68 |

- 본표 최고 baseline AlphaEdit-BLUE W20(95.87 / 92.28 / 32.83)보다 Eff +3.75, Gen +2.55, Loc +12.48이다.
- **Loc이 W0보다 높다(45.31 대 38.59).** 같은 시점 W0 예측 일치율은 62.68%다. 무관한 NQ 질문의 예측이 바뀌는데, 바뀐 쪽이 정답과 더 자주 맞는다. 원인은 아직 확인하지 않았다. baseline은 모두 W0보다 낮다.
- 표시는 decimal half-up 둘째 자리다. 정확한 값은 [results.json](../../../../audits/servers/server4/price-final-2k-20261010/llama3-zsre-results.json)에 있다.

## checkpoint (500 edit마다, resumable)

| 시점 | editable weights sha256 | writer history H sha256 |
|---|---|---|
| W5 | `cb9563e6…` | `e6dc3c0c…` |
| W10 | `9c55488b…` | `8f4fc884…` |
| W15 | `8c206abc…` | `2cfb5539…` |
| W20 | `a9af1605…` | `a620167a…` |
